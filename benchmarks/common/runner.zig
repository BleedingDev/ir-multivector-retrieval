//! benchmarks/common/runner.zig — compose qrels + queries + index into a
//! `tac.retrieval.bench.runSweep` invocation. Per-dataset entry points
//! (msmarco_v1.zig, lotte_pooled.zig) call `runDatasetSweep`.

const std = @import("std");
const Allocator = std.mem.Allocator;

const tac = @import("tac");
const qrels_mod = @import("qrels.zig");
const queries_mod = @import("queries.zig");

pub const RunError = error{
    NoOverlap,
    ReadFailed,
} || queries_mod.QueriesError || Allocator.Error;

pub const RunArgs = struct {
    dataset: []const u8, // "msmarco-v1", "lotte-pooled"
    index_path: []const u8,
    queries_path: []const u8,
    qids_path: []const u8,
    qrels_path: []const u8,
    out_csv_path: []const u8,
    git_sha: []const u8,
    metric: tac.retrieval.bench.MetricKind,
    /// rel >= min_rel ⇒ relevant.
    min_rel: i32 = 1,
    /// Pin the benchmark thread to this core (paper §9). Best-effort per OS.
    /// Set to null on macOS — `pinToCore` is a no-op there (Mach
    /// `thread_policy_set` isn't surfaced in std), so leaving it 0 implied
    /// a guarantee we don't have. Linux still pins via `sched_setaffinity`.
    pin_core: ?u32 = null,
};

/// Read `BENCH_WARMUP` (count) and `BENCH_COOLDOWN_MS` (ms) from the
/// process environment to build a `BenchProtocol`. Falls back to defaults
/// when unset or unparseable, so production runs need no env at all but a
/// CI hostile-machine run can dial up cooldown without recompiling.
pub fn protocolFromEnv(environ: std.process.Environ) tac.retrieval.bench.BenchProtocol {
    var p = tac.retrieval.bench.BenchProtocol{};
    if (environ.getPosix("BENCH_WARMUP")) |v| {
        if (std.fmt.parseInt(u32, v, 10)) |n| {
            p.warmup_iters = n;
        } else |_| {}
    }
    if (environ.getPosix("BENCH_COOLDOWN_MS")) |v| {
        if (std.fmt.parseInt(u64, v, 10)) |ms| {
            p.cooldown_ns = ms * 1_000_000;
        } else |_| {}
    }
    return p;
}

const CsvCtx = struct {
    writer: *std.Io.Writer,
    dataset: []const u8,
    git_sha: []const u8,
};

fn writeRowCallback(row: tac.retrieval.bench.SweepRow, ctx: *anyopaque) anyerror!void {
    const c: *CsvCtx = @ptrCast(@alignCast(ctx));
    try tac.retrieval.bench.formatCsvRow(c.writer, c.dataset, c.git_sha, row);
    try c.writer.writeByte('\n');
    try c.writer.flush();
}

/// Read an entire file into a freshly-allocated buffer of exact size.
/// `align_8` requests an 8-aligned allocation (needed for token_dump and
/// storage parsing).
fn slurp(io: std.Io, dir: std.Io.Dir, path: []const u8, gpa: Allocator, comptime align_8: bool) ![]u8 {
    var f = try dir.openFile(io, path, .{});
    defer f.close(io);
    const stat = try f.stat(io);
    const size: usize = std.math.cast(usize, stat.size) orelse return error.ReadFailed;
    const buf: []u8 = if (align_8)
        try gpa.alignedAlloc(u8, .@"8", size)
    else
        try gpa.alloc(u8, size);
    errdefer gpa.free(buf);

    var read_buf: [16 * 1024]u8 = undefined;
    var fr = f.reader(io, &read_buf);
    const reader = &fr.interface;
    reader.readSliceAll(buf) catch return error.ReadFailed;
    return buf;
}

/// End-to-end harness: open inputs, fold them into a `bench.QueryPack`, run
/// the paper §6 grid sweep, write a CSV. Prints the bench protocol header
/// to stderr and to the CSV before the sweep so the captured output is
/// self-describing.
pub fn runDatasetSweep(args: RunArgs, protocol: tac.retrieval.bench.BenchProtocol, gpa: Allocator) !void {
    if (args.pin_core) |c| tac.eval.latency.pinToCore(c);

    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const cwd = std.Io.Dir.cwd();

    // ---- Index ----
    const idx_bytes_raw = try slurp(io, cwd, args.index_path, gpa, true);
    defer gpa.free(idx_bytes_raw);
    const idx_bytes: []align(8) u8 = @alignCast(idx_bytes_raw);
    var index = try tac.index.storage.parse(idx_bytes, gpa);
    defer index.deinit(gpa);

    // ---- Queries + qids ----
    const q_bytes_raw = try slurp(io, cwd, args.queries_path, gpa, true);
    errdefer gpa.free(q_bytes_raw);
    const q_bytes: []align(8) u8 = @alignCast(q_bytes_raw);
    const qids_bytes = try slurp(io, cwd, args.qids_path, gpa, false);
    defer gpa.free(qids_bytes);
    var enc = try queries_mod.loadFromBytes(q_bytes, qids_bytes, gpa);
    defer enc.deinit(gpa);

    // ---- Qrels ----
    const qrels_bytes = try slurp(io, cwd, args.qrels_path, gpa, false);
    defer gpa.free(qrels_bytes);
    var qrels = try qrels_mod.parseBytes(qrels_bytes, args.min_rel, gpa);
    defer qrels.deinit(gpa);

    // ---- Build a bench.QueryPack: keep only queries that have qrels. ----
    const dim: u32 = enc.dump.dim;
    const n_q_total: usize = @intCast(enc.nQueries());

    var pack_n_q: std.ArrayList(u32) = .empty;
    defer pack_n_q.deinit(gpa);
    var pack_q_offsets: std.ArrayList(u64) = .empty;
    defer pack_q_offsets.deinit(gpa);
    var pack_tokens: std.ArrayList(f32) = .empty;
    defer pack_tokens.deinit(gpa);
    var pack_qrels: std.ArrayList(u32) = .empty;
    defer pack_qrels.deinit(gpa);
    var pack_qrels_lo: std.ArrayList(u64) = .empty;
    defer pack_qrels_lo.deinit(gpa);

    try pack_q_offsets.append(gpa, 0);
    try pack_qrels_lo.append(gpa, 0);

    var q: usize = 0;
    while (q < n_q_total) : (q += 1) {
        const qid = enc.qids[q];
        const rels = qrels.forQuery(qid);
        if (rels.len == 0) continue;

        const range = enc.dump.docTokenRange(@intCast(q));
        const tok_lo: usize = @intCast(range[0]);
        const tok_hi: usize = @intCast(range[1]);
        const n_q_q: u32 = @intCast(tok_hi - tok_lo);
        const flat_lo: usize = tok_lo * dim;
        const flat_hi: usize = tok_hi * dim;

        try pack_n_q.append(gpa, n_q_q);
        try pack_tokens.appendSlice(gpa, enc.dump.vectors[flat_lo..flat_hi]);
        try pack_q_offsets.append(gpa, @as(u64, pack_tokens.items.len));
        try pack_qrels.appendSlice(gpa, rels);
        try pack_qrels_lo.append(gpa, @as(u64, pack_qrels.items.len));
    }
    if (pack_n_q.items.len == 0) return error.NoOverlap;

    const pack = tac.retrieval.bench.QueryPack{
        .n_q = pack_n_q.items,
        .q_offsets = pack_q_offsets.items,
        .tokens = pack_tokens.items,
        .qrels = pack_qrels.items,
        .qrels_lo = pack_qrels_lo.items,
    };

    // ---- Open output CSV + buffered writer. ----
    var out_file = try cwd.createFile(io, args.out_csv_path, .{});
    defer out_file.close(io);
    var write_buf: [16 * 1024]u8 = undefined;
    var fw = out_file.writer(io, &write_buf);
    var w = &fw.interface;

    // Self-describing header — emitted to stderr (so the human running the
    // bench sees it live) and as `#`-prefixed comments in the CSV (so the
    // captured artifact remembers what produced it). stderr path goes
    // through a fixed buffer + `std.debug.print` to avoid threading an Io
    // instance through `std.Io.File.stderr().writer`.
    var hdr_buf: [4 * 1024]u8 = undefined;
    var hdr_w = std.Io.Writer.fixed(&hdr_buf);
    try tac.retrieval.bench.writeProtocolHeader(&hdr_w, args.dataset, args.git_sha, pack_n_q.items.len, args.metric, protocol);
    std.debug.print("{s}", .{hdr_w.buffered()});
    try tac.retrieval.bench.writeProtocolHeader(w, args.dataset, args.git_sha, pack_n_q.items.len, args.metric, protocol);

    try w.writeAll(tac.retrieval.bench.csv_header);
    try w.writeByte('\n');

    var ctx = CsvCtx{
        .writer = w,
        .dataset = args.dataset,
        .git_sha = args.git_sha,
    };

    try tac.retrieval.bench.runSweepWithProtocol(&index, pack, args.metric, protocol, writeRowCallback, &ctx, gpa);
    try w.flush();
}
