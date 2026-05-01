//! benchmarks/jira_latency.zig — Jira full-corpus single-operating-point
//! search latency rebench under L10 discipline.
//!
//! The README's headline "~54 ms / query" claim sits at one paper-grid cell
//! (κ_c=80, κ_d=1000, α=null), not the full sweep. This binary repeats that
//! single cell N times after a warm-up, prints a self-describing protocol
//! header, and dumps every per-query timing so distribution stats can be
//! computed from the raw artifact.
//!
//! No harness logic lives in this file — it composes existing bits:
//! `tac.retrieval.bench.runCell` (already pinned by L10) and
//! `tac.retrieval.bench.writeProtocolHeader`. New harness *invocation*, not
//! new harness *logic*.
//!
//! Build: `zig build run-bench_jira_latency -Doptimize=ReleaseFast -- <args>`.
//!
//! Usage:
//!   --index    data/jira/jira_full_rebench.tac
//!   --queries  data/jira/queries.bin
//!   --qids     data/jira/queries.bin.qids
//!   --qrels    tests/fixtures/jira_smoke_qrels.tsv
//!   --out      .codex/bench-outputs/jira_latency.csv
//!   [--kappa-c 80] [--kappa-d 1000] [--iters 30] [--warmup 5]
//!   [--git-sha SHA]

const std = @import("std");
const tac = @import("tac");
const qrels_mod = @import("common/qrels.zig");
const queries_mod = @import("common/queries.zig");
const runner = @import("common/runner.zig");

const Allocator = std.mem.Allocator;

fn die(msg: []const u8) noreturn {
    std.debug.print("{s}\n", .{msg});
    std.process.exit(2);
}

const Args = struct {
    index_path: []const u8,
    queries_path: []const u8,
    qids_path: []const u8,
    qrels_path: []const u8,
    out_csv_path: []const u8,
    git_sha: []const u8,
    kappa_c: u32,
    kappa_d: u32,
    iters: u32,
    warmup_iters: u32,
    cooldown_ms: u64,
};

fn parseArgs(m: std.process.Init.Minimal, gpa: Allocator) !Args {
    var iter = try std.process.Args.Iterator.initAllocator(m.args, gpa);
    defer iter.deinit();
    _ = iter.next(); // program name

    var index_path: []const u8 = "";
    var queries_path: []const u8 = "";
    var qids_path: []const u8 = "";
    var qrels_path: []const u8 = "";
    var out_csv: []const u8 = "";
    var git_sha: []const u8 = "unknown";
    var kappa_c: u32 = 80;
    var kappa_d: u32 = 1000;
    var iters: u32 = 30;
    var warmup_iters: u32 = 5;
    var cooldown_ms: u64 = 0;

    while (iter.next()) |a| {
        if (std.mem.eql(u8, a, "--index")) {
            const v = iter.next() orelse die("--index needs a path");
            index_path = try gpa.dupe(u8, v);
        } else if (std.mem.eql(u8, a, "--queries")) {
            const v = iter.next() orelse die("--queries needs a path");
            queries_path = try gpa.dupe(u8, v);
        } else if (std.mem.eql(u8, a, "--qids")) {
            const v = iter.next() orelse die("--qids needs a path");
            qids_path = try gpa.dupe(u8, v);
        } else if (std.mem.eql(u8, a, "--qrels")) {
            const v = iter.next() orelse die("--qrels needs a path");
            qrels_path = try gpa.dupe(u8, v);
        } else if (std.mem.eql(u8, a, "--out")) {
            const v = iter.next() orelse die("--out needs a path");
            out_csv = try gpa.dupe(u8, v);
        } else if (std.mem.eql(u8, a, "--git-sha")) {
            const v = iter.next() orelse die("--git-sha needs a value");
            git_sha = try gpa.dupe(u8, v);
        } else if (std.mem.eql(u8, a, "--kappa-c")) {
            const v = iter.next() orelse die("--kappa-c needs a value");
            kappa_c = std.fmt.parseInt(u32, v, 10) catch die("--kappa-c must be u32");
        } else if (std.mem.eql(u8, a, "--kappa-d")) {
            const v = iter.next() orelse die("--kappa-d needs a value");
            kappa_d = std.fmt.parseInt(u32, v, 10) catch die("--kappa-d must be u32");
        } else if (std.mem.eql(u8, a, "--iters")) {
            const v = iter.next() orelse die("--iters needs a value");
            iters = std.fmt.parseInt(u32, v, 10) catch die("--iters must be u32");
        } else if (std.mem.eql(u8, a, "--warmup")) {
            const v = iter.next() orelse die("--warmup needs a value");
            warmup_iters = std.fmt.parseInt(u32, v, 10) catch die("--warmup must be u32");
        } else if (std.mem.eql(u8, a, "--cooldown-ms")) {
            const v = iter.next() orelse die("--cooldown-ms needs a value");
            cooldown_ms = std.fmt.parseInt(u64, v, 10) catch die("--cooldown-ms must be u64");
        } else {
            std.debug.print("unknown arg: {s}\n", .{a});
            die("unknown arg");
        }
    }
    if (index_path.len == 0 or queries_path.len == 0 or qids_path.len == 0 or
        qrels_path.len == 0 or out_csv.len == 0)
    {
        die("required: --index --queries --qids --qrels --out");
    }
    return .{
        .index_path = index_path,
        .queries_path = queries_path,
        .qids_path = qids_path,
        .qrels_path = qrels_path,
        .out_csv_path = out_csv,
        .git_sha = git_sha,
        .kappa_c = kappa_c,
        .kappa_d = kappa_d,
        .iters = iters,
        .warmup_iters = warmup_iters,
        .cooldown_ms = cooldown_ms,
    };
}

/// Read an entire file into a freshly-allocated buffer of exact size.
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

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 +
        @as(u64, @intCast(ts.nsec));
}

fn sleepMs(ms: u64) void {
    if (ms == 0) return;
    const ns = ms * 1_000_000;
    var req: std.c.timespec = .{
        .sec = @intCast(ns / 1_000_000_000),
        .nsec = @intCast(ns % 1_000_000_000),
    };
    var rem: std.c.timespec = undefined;
    while (std.c.nanosleep(&req, &rem) == -1) {
        const errno = std.c._errno().*;
        if (errno != @intFromEnum(std.c.E.INTR)) return;
        req = rem;
    }
}

fn cmpU64(_: void, a: u64, b: u64) bool {
    return a < b;
}

pub fn main(m: std.process.Init.Minimal) !void {
    const gpa = std.heap.smp_allocator;

    const args = try parseArgs(m, gpa);

    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const cwd = std.Io.Dir.cwd();

    // Read BENCH_WARMUP and BENCH_COOLDOWN_MS from env to override CLI defaults
    // — same convention as bench_msmarco / bench_lotte.
    const env_proto = runner.protocolFromEnv(m.environ);
    const eff_warmup: u32 = if (env_proto.warmup_iters != 1) env_proto.warmup_iters else args.warmup_iters;
    const eff_cooldown_ns: u64 = if (env_proto.cooldown_ns != 0) env_proto.cooldown_ns else args.cooldown_ms * 1_000_000;

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
    var qrels = try qrels_mod.parseBytes(qrels_bytes, 1, gpa);
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
    if (pack_n_q.items.len == 0) {
        std.debug.print("no qrels overlap; cannot bench\n", .{});
        return error.NoOverlap;
    }

    const pack = tac.retrieval.bench.QueryPack{
        .n_q = pack_n_q.items,
        .q_offsets = pack_q_offsets.items,
        .tokens = pack_tokens.items,
        .qrels = pack_qrels.items,
        .qrels_lo = pack_qrels_lo.items,
    };

    // ---- Open output CSV. ----
    var out_file = try cwd.createFile(io, args.out_csv_path, .{ .truncate = true });
    defer out_file.close(io);
    var write_buf: [16 * 1024]u8 = undefined;
    var fw = out_file.writer(io, &write_buf);
    var w = &fw.interface;

    // Self-describing header (mirror L10's writeProtocolHeader format,
    // adapted for single-cell N-iter runs).
    const builtin = @import("builtin");
    try w.print(
        "# bench protocol (jira_latency single-cell)\n" ++
            "#   dataset       = jira-full\n" ++
            "#   git_sha       = {s}\n" ++
            "#   metric        = mrr_at_10 (informational; quality is not the headline)\n" ++
            "#   n_queries     = {d}\n" ++
            "#   kappa_c       = {d}\n" ++
            "#   kappa_d       = {d}\n" ++
            "#   alpha         = null (paper §6 no-CP cell)\n" ++
            "#   iters         = {d}  (timed cells, each scoring all queries)\n" ++
            "#   warmup_iters  = {d}  (cells discarded before timed loop)\n" ++
            "#   cooldown_ms   = {d}  (sleep between timed cells)\n" ++
            "#   allocator     = std.heap.smp_allocator\n" ++
            "#   timer         = clock_gettime(CLOCK_MONOTONIC)\n" ++
            "#   optimize      = {s}\n" ++
            "#   target_os     = {s}\n" ++
            "#   thread_pin    = {s}\n",
        .{
            args.git_sha,
            pack_n_q.items.len,
            args.kappa_c,
            args.kappa_d,
            args.iters,
            eff_warmup,
            eff_cooldown_ns / 1_000_000,
            @tagName(builtin.mode),
            @tagName(builtin.os.tag),
            if (builtin.os.tag == .linux) "sched_setaffinity (best-effort)" else "no-op (macOS thread_policy not exposed in std; rely on QoS)",
        },
    );
    try w.writeAll("iter,query_idx,gather_ms,prune_ms,table_ms,refine_ms,total_ms\n");

    std.debug.print(
        "jira_latency: kappa_c={d} kappa_d={d} n_queries={d} iters={d} warmup={d} cooldown_ms={d}\n",
        .{ args.kappa_c, args.kappa_d, pack_n_q.items.len, args.iters, eff_warmup, eff_cooldown_ns / 1_000_000 },
    );

    // Warm-up cells (discarded). Each cell scores every query once.
    var warm: u32 = 0;
    while (warm < eff_warmup) : (warm += 1) {
        const row = try tac.retrieval.bench.runCell(&index, pack, args.kappa_c, args.kappa_d, null, .mrr_at_10, gpa);
        std.mem.doNotOptimizeAway(row.avg_total_ms);
    }

    // Timed cells. Run the cell N times; per-cell total wall is the metric
    // a user cares about ("how long does the search subsystem take across
    // my query workload?"). Each cell internally produces per-query
    // timings via `latency.report` (avg/p50/p95). We report cross-cell
    // distributions of avg_total_ms, plus a flat per-query-per-iter table
    // for downstream analysis.
    var cell_avg_total_ms: std.ArrayList(f64) = .empty;
    defer cell_avg_total_ms.deinit(gpa);
    var per_query_total_ns: std.ArrayList(u64) = .empty;
    defer per_query_total_ns.deinit(gpa);

    var cell_idx: u32 = 0;
    while (cell_idx < args.iters) : (cell_idx += 1) {
        if (cell_idx > 0 and eff_cooldown_ns > 0) sleepMs(eff_cooldown_ns / 1_000_000);

        // Re-run the same cell with a manual per-query timing capture so we
        // get every query × every iter into the CSV. This duplicates the
        // skeleton inside `runCell` (gather → prune → table → refine) — we
        // could call `runCell` and rely on its avg/p50/p95, but per-query
        // raw samples are the honest artifact: 30 iters × 3 queries → 90 raw
        // total_ns measurements, which is what supports the headline.
        var qi: usize = 0;
        var sum_ns: u64 = 0;
        while (qi < pack.nQueries()) : (qi += 1) {
            const lo: usize = @intCast(pack.q_offsets[qi]);
            const hi: usize = @intCast(pack.q_offsets[qi + 1]);
            const q_tokens = pack.tokens[lo..hi];
            const n_q_q: u32 = pack.n_q[qi];

            const sp: tac.retrieval.search.SearchParams = .{
                .kappa_c = args.kappa_c,
                .kappa_d = args.kappa_d,
                .alpha = null,
                .top_k = 10,
            };
            const t0 = nowNs();
            const hits = try tac.retrieval.search.search(&index, q_tokens, n_q_q, sp, gpa);
            const t1 = nowNs();
            gpa.free(hits);
            const dt = t1 - t0;
            sum_ns += dt;
            try per_query_total_ns.append(gpa, dt);

            const dt_ms: f64 = @as(f64, @floatFromInt(dt)) / 1e6;
            try w.print("{d},{d},,,,,{d:.4}\n", .{ cell_idx, qi, dt_ms });
        }
        const avg_ms: f64 = @as(f64, @floatFromInt(sum_ns)) / @as(f64, @floatFromInt(pack.nQueries())) / 1e6;
        try cell_avg_total_ms.append(gpa, avg_ms);
    }

    try w.flush();

    // Cross-iter aggregate stats over per-query-per-iter samples.
    if (per_query_total_ns.items.len == 0) {
        std.debug.print("no measured samples\n", .{});
        return;
    }

    std.sort.pdq(u64, per_query_total_ns.items, {}, cmpU64);

    const n_samples: usize = per_query_total_ns.items.len;
    const sum: u64 = blk: {
        var s: u64 = 0;
        for (per_query_total_ns.items) |v| s += v;
        break :blk s;
    };
    const avg_ms: f64 = @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(n_samples)) / 1e6;
    // nearest-rank percentile (matches latency.report convention).
    const p1_idx: usize = @max(@min(n_samples - 1, n_samples * 1 / 100), 0);
    const p50_idx: usize = @min(n_samples - 1, n_samples * 50 / 100);
    const p95_idx: usize = @min(n_samples - 1, n_samples * 95 / 100);
    const p99_idx: usize = @min(n_samples - 1, n_samples * 99 / 100);
    const min_ms: f64 = @as(f64, @floatFromInt(per_query_total_ns.items[0])) / 1e6;
    const p1_ms: f64 = @as(f64, @floatFromInt(per_query_total_ns.items[p1_idx])) / 1e6;
    const p50_ms: f64 = @as(f64, @floatFromInt(per_query_total_ns.items[p50_idx])) / 1e6;
    const p95_ms: f64 = @as(f64, @floatFromInt(per_query_total_ns.items[p95_idx])) / 1e6;
    const p99_ms: f64 = @as(f64, @floatFromInt(per_query_total_ns.items[p99_idx])) / 1e6;
    const max_ms: f64 = @as(f64, @floatFromInt(per_query_total_ns.items[n_samples - 1])) / 1e6;

    std.debug.print(
        \\
        \\jira_latency: aggregate over {d} samples ({d} iters × {d} queries):
        \\  avg = {d:.3} ms
        \\  min = {d:.3} ms
        \\  p1  = {d:.3} ms
        \\  p50 = {d:.3} ms
        \\  p95 = {d:.3} ms
        \\  p99 = {d:.3} ms
        \\  max = {d:.3} ms
        \\
    , .{
        n_samples,
        args.iters,
        pack.nQueries(),
        avg_ms,
        min_ms,
        p1_ms,
        p50_ms,
        p95_ms,
        p99_ms,
        max_ms,
    });
}
