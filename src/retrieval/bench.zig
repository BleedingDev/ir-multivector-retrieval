//! src/retrieval/bench.zig — benchmark plumbing & smoke test (paper §7-9).
//!
//! Owner: retriever.
//!
//! The real per-dataset harnesses live under `benchmarks/` and operate on
//! disk-backed inputs (encoded queries.bin + qrels.tsv from `tools/encode.py`).
//! This in-tree module hosts the reusable sweep core and a synthetic-fixture
//! smoke test that exercises the full search → metrics → latency pipeline
//! on every `zig build test`, guarding against integration regressions.
//!
//! Sweep grid is fixed by the paper:
//!   κ_c ∈ {15, 20, 40, 80, 100, 120}      (paper §6, retrieval grid)
//!   κ_d ∈ {250, 500, 1000, 2000, 4000}
//!   α   ∈ {null, 0.35, 0.40, 0.45, 0.50}
//!
//! Single-thread per paper §9.

const std = @import("std");
const Allocator = std.mem.Allocator;

const constants = @import("../constants.zig");
const storage = @import("../index/storage.zig");
const search_mod = @import("search.zig");
const gather = @import("gather.zig");
const prune = @import("prune.zig");
const refine = @import("refine.zig");
const metrics = @import("../eval/metrics.zig");
const latency = @import("../eval/latency.zig");

/// Monotonic ns timestamp via posix `clock_gettime(CLOCK_MONOTONIC)`. Zig
/// 0.16 moved std.time.Timer behind the std.Io abstraction; this avoids
/// threading an Io instance through the benchmark plumbing.
fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 +
        @as(u64, @intCast(ts.nsec));
}

const Stopwatch = struct {
    last: u64,
    fn start() Stopwatch {
        return .{ .last = nowNs() };
    }
    fn lap(self: *Stopwatch) u64 {
        const now = nowNs();
        const d = now - self.last;
        self.last = now;
        return d;
    }
    fn read(self: *const Stopwatch) u64 {
        return nowNs() - self.last;
    }
};

/// Operating point row: one (κ_c, κ_d, α) cell × (n_queries, dataset).
pub const SweepRow = struct {
    kappa_c: u32,
    kappa_d: u32,
    alpha_x100: i16, // -1 sentinel for null, else round(alpha*100)
    n_queries: u32,
    /// Dataset metric: MRR@10 (MS MARCO) or Success@5 (LoTTE) — caller picks
    /// the metric function; the row carries the result undifferentiated.
    quality: f32,
    avg_total_ms: f64,
    p50_total_ms: f64,
    p95_total_ms: f64,
    avg_gather_ms: f64,
    avg_prune_ms: f64,
    avg_table_ms: f64,
    avg_refine_ms: f64,
};

pub const QueryPack = struct {
    /// `n_q[i]` = query length (number of tokens) for query i.
    n_q: []const u32,
    /// `q_offsets[q+1] - q_offsets[q]` = `n_q[q] * dim`. Float offsets into
    /// `tokens` for query q.
    q_offsets: []const u64,
    /// All query tokens, row-major. Length sum_i n_q[i] * dim.
    tokens: []const f32,
    /// Per-query relevance lists (qrels). `qrels_lo[q+1] - qrels_lo[q]` = #
    /// relevant doc IDs for query q; `qrels[qrels_lo[q]..qrels_lo[q+1]]` are
    /// those doc IDs.
    qrels: []const u32,
    qrels_lo: []const u64,

    pub fn nQueries(self: QueryPack) usize {
        return self.n_q.len;
    }
};

pub const MetricKind = enum { mrr_at_10, success_at_5 };

/// Run one (κ_c, κ_d, α) cell across all queries. Returns the aggregated
/// row. `per_token_max_buf` and `centroid_buf` are caller-provided scratch
/// shared across all queries (avoid reallocating per query).
pub fn runCell(
    index: *const storage.Index,
    pack: QueryPack,
    kappa_c: u32,
    kappa_d: u32,
    alpha: ?f32,
    metric: MetricKind,
    gpa: Allocator,
) !SweepRow {
    const n_queries = pack.nQueries();
    var quality_sum: f64 = 0.0;
    var timings: std.ArrayList(latency.StageTimings) = .empty;
    defer timings.deinit(gpa);
    try timings.ensureTotalCapacity(gpa, n_queries);

    const top_k: u32 = switch (metric) {
        .mrr_at_10 => 10,
        .success_at_5 => 5,
    };

    var q: usize = 0;
    while (q < n_queries) : (q += 1) {
        const lo: usize = @intCast(pack.q_offsets[q]);
        const hi: usize = @intCast(pack.q_offsets[q + 1]);
        const q_tokens = pack.tokens[lo..hi];
        const n_q_q: u32 = pack.n_q[q];

        // Time the full search call as a single bucket. Per-stage breakdown
        // is approximated below by re-running each stage with explicit
        // timers — this matches what the paper reports (avg per-stage ms).
        var t = latency.StageTimings{};
        const timer_total = Stopwatch.start();

        // Per-stage timing: gather/prune/table/refine with explicit borders.
        var sub_timer = Stopwatch.start();

        const candidates = try gather.gather(index, q_tokens, n_q_q, .{ .kappa_c = kappa_c }, gpa);
        defer gpa.free(candidates);
        t.gather_ns = sub_timer.lap();

        const survivors = prune.prune(candidates, .{ .kappa_d = kappa_d, .alpha = alpha });
        t.prune_ns = sub_timer.lap();

        if (survivors.len > 0) {
            const table_len: usize =
                @as(usize, constants.PQ_M) * @as(usize, constants.PQ_CENTROIDS) * @as(usize, n_q_q);
            const table = try gpa.alloc(f32, table_len);
            defer gpa.free(table);
            try index.pq.buildDistanceTable(q_tokens, n_q_q, table);
            t.table_build_ns = sub_timer.lap();

            const per_token_max = try gpa.alloc(f32, n_q_q);
            defer gpa.free(per_token_max);

            // Score every survivor and collect ranking up to top_k.
            const refined = try gpa.alloc(search_mod.ScoredDoc, survivors.len);
            defer gpa.free(refined);
            for (survivors, 0..) |s, k| {
                refined[k] = .{
                    .doc_id = s.doc_id,
                    .score = try refine.refine(index, table, q_tokens, n_q_q, s, per_token_max),
                };
            }
            std.sort.pdq(search_mod.ScoredDoc, refined, {}, scoredDescending);
            t.refine_ns = sub_timer.lap();

            // Build a u32 ranking up to top_k for the metric.
            const want: usize = @min(refined.len, @as(usize, top_k));
            const ranking = try gpa.alloc(u32, want);
            defer gpa.free(ranking);
            for (refined[0..want], 0..) |r, idx| ranking[idx] = r.doc_id;

            const qr_lo: usize = @intCast(pack.qrels_lo[q]);
            const qr_hi: usize = @intCast(pack.qrels_lo[q + 1]);
            const qrels_q = pack.qrels[qr_lo..qr_hi];
            const m: f32 = switch (metric) {
                .mrr_at_10 => metrics.mrrAt(ranking, qrels_q, 10),
                .success_at_5 => metrics.successAt(ranking, qrels_q, 5),
            };
            quality_sum += m;
        }

        t.total_ns = timer_total.read();
        try timings.append(gpa, t);
    }

    const rep = latency.report(timings.items);
    const alpha_x100: i16 = if (alpha) |a| @intFromFloat(@round(a * 100.0)) else -1;
    return .{
        .kappa_c = kappa_c,
        .kappa_d = kappa_d,
        .alpha_x100 = alpha_x100,
        .n_queries = @intCast(n_queries),
        .quality = @floatCast(quality_sum / @as(f64, @floatFromInt(n_queries))),
        .avg_total_ms = rep.avg_total_ms,
        .p50_total_ms = rep.p50_total_ms,
        .p95_total_ms = rep.p95_total_ms,
        .avg_gather_ms = rep.avg_gather_ms,
        .avg_prune_ms = rep.avg_prune_ms,
        .avg_table_ms = rep.avg_table_ms,
        .avg_refine_ms = rep.avg_refine_ms,
    };
}

fn scoredDescending(_: void, a: search_mod.ScoredDoc, b: search_mod.ScoredDoc) bool {
    return a.score > b.score;
}

/// Paper §6 retrieval grid.
pub const kappa_c_grid: []const u32 = &.{ 15, 20, 40, 80, 100, 120 };
pub const kappa_d_grid: []const u32 = &.{ 250, 500, 1000, 2000, 4000 };
pub const alpha_grid: []const ?f32 = &.{ null, 0.35, 0.40, 0.45, 0.50 };

/// Bench harness discipline. Captured in the run header so each set of
/// numbers carries the protocol that produced them.
///
/// - `warmup_iters`: how many warm-up cells to run before the timed sweep.
///   Each warm-up cell uses `(kappa_c=80, kappa_d=1000, alpha=null)` — the
///   middle of the paper grid — and its results are discarded. Defaults to
///   1; set 0 to disable. Warm-up populates allocator pools, exercises code
///   paths, and lets the CPU hit a stable frequency before measurement.
/// - `cooldown_ns`: nanoseconds to sleep between timed cells via
///   `nanosleep(2)`. Lets CPU temperature recover so adjacent cells don't
///   contaminate one another. Defaults to 0; configure via `BENCH_COOLDOWN_MS`.
/// - `allocator_label`, `timer_label`: free-form strings emitted in the
///   header so output is self-describing.
pub const BenchProtocol = struct {
    warmup_iters: u32 = 1,
    cooldown_ns: u64 = 0,
    allocator_label: []const u8 = "smp_allocator",
    timer_label: []const u8 = "clock_gettime(CLOCK_MONOTONIC)",
};

/// Sleep for `ns` nanoseconds via `nanosleep(2)`. Returns even on EINTR.
/// Used between cells for thermal cooldown; not in the timed path.
fn sleepNs(ns: u64) void {
    if (ns == 0) return;
    var req: std.c.timespec = .{
        .sec = @intCast(ns / 1_000_000_000),
        .nsec = @intCast(ns % 1_000_000_000),
    };
    var rem: std.c.timespec = undefined;
    while (std.c.nanosleep(&req, &rem) == -1) {
        // EINTR: continue with remainder. Anything else: bail (best-effort).
        const errno = std.c._errno().*;
        if (errno != @intFromEnum(std.c.E.INTR)) return;
        req = rem;
    }
}

/// Print a self-describing protocol header to `writer`. Captures everything
/// needed to reproduce a number: dataset, git_sha, allocator, timer, warm-up
/// + cooldown configuration. Per finding L10: headline numbers must be
/// reproducible from a single command + commit + allocator + warm-up
/// protocol that's printed at the top of the run.
pub fn writeProtocolHeader(
    writer: *std.Io.Writer,
    dataset: []const u8,
    git_sha: []const u8,
    n_queries: usize,
    metric: MetricKind,
    protocol: BenchProtocol,
) std.Io.Writer.Error!void {
    const builtin = @import("builtin");
    const cooldown_ms = protocol.cooldown_ns / 1_000_000;
    try writer.print(
        "# bench protocol\n" ++
            "#   dataset       = {s}\n" ++
            "#   git_sha       = {s}\n" ++
            "#   metric        = {s}\n" ++
            "#   n_queries     = {d}\n" ++
            "#   sweep_cells   = {d} (κ_c × κ_d × α grid, paper §6)\n" ++
            "#   allocator     = {s}\n" ++
            "#   timer         = {s}\n" ++
            "#   warmup_iters  = {d}  (cells discarded before timed sweep)\n" ++
            "#   cooldown_ms   = {d}  (sleep between timed cells)\n" ++
            "#   optimize      = {s}\n" ++
            "#   target_os     = {s}\n" ++
            "#   thread_pin    = {s}\n",
        .{
            dataset,
            git_sha,
            @tagName(metric),
            n_queries,
            kappa_c_grid.len * kappa_d_grid.len * alpha_grid.len,
            protocol.allocator_label,
            protocol.timer_label,
            protocol.warmup_iters,
            cooldown_ms,
            @tagName(builtin.mode),
            @tagName(builtin.os.tag),
            if (builtin.os.tag == .linux) "sched_setaffinity (best-effort)" else "no-op (macOS thread_policy not exposed in std; rely on QoS)",
        },
    );
}

/// Run the full Cartesian sweep, calling `out_callback` for every cell. The
/// callback is the integration boundary with the CSV writer in the
/// per-dataset harness — we keep this module IO-free so it's testable.
///
/// Backwards-compatible wrapper around `runSweepWithProtocol` that uses the
/// default protocol (1 warm-up cell, no cooldown). New callers (the
/// per-dataset bench harnesses) should call `runSweepWithProtocol` directly
/// and pass an explicit protocol; this overload exists so `tac bench`
/// (src/main.zig) keeps compiling without a touch from the bench-discipline
/// lane (cross-lane scope coordination, see audit-fixes-master-plan.md).
pub fn runSweep(
    index: *const storage.Index,
    pack: QueryPack,
    metric: MetricKind,
    out_callback: *const fn (row: SweepRow, ctx: *anyopaque) anyerror!void,
    ctx: *anyopaque,
    gpa: Allocator,
) !void {
    return runSweepWithProtocol(index, pack, metric, .{}, out_callback, ctx, gpa);
}

/// Like `runSweep` but takes an explicit `BenchProtocol`.
///
/// Discipline: `protocol.warmup_iters` warm-up cells are run (and discarded)
/// before the timed sweep; `protocol.cooldown_ns` of `nanosleep` is inserted
/// between timed cells. Warm-up uses a fixed cell from the middle of the
/// paper grid so it's representative without being free.
pub fn runSweepWithProtocol(
    index: *const storage.Index,
    pack: QueryPack,
    metric: MetricKind,
    protocol: BenchProtocol,
    out_callback: *const fn (row: SweepRow, ctx: *anyopaque) anyerror!void,
    ctx: *anyopaque,
    gpa: Allocator,
) !void {
    // Warm-up: representative cell, results discarded. Picked from the
    // middle of the paper grid so caches/branch predictors see a workload
    // shape close to what the timed sweep will hit.
    var w: u32 = 0;
    while (w < protocol.warmup_iters) : (w += 1) {
        const warm_row = try runCell(index, pack, 80, 1000, null, metric, gpa);
        // Sink so the optimizer can't elide the warm-up.
        std.mem.doNotOptimizeAway(warm_row.avg_total_ms);
    }

    var first: bool = true;
    for (kappa_c_grid) |kc| {
        for (kappa_d_grid) |kd| {
            for (alpha_grid) |a| {
                if (!first) sleepNs(protocol.cooldown_ns);
                first = false;
                const row = try runCell(index, pack, kc, kd, a, metric, gpa);
                try out_callback(row, ctx);
            }
        }
    }
}

/// Format one CSV row (no trailing newline; caller adds it). Caller-provided
/// `dataset` and `git_sha` strings are prepended for traceability.
pub fn formatCsvRow(
    writer: *std.Io.Writer,
    dataset: []const u8,
    git_sha: []const u8,
    row: SweepRow,
) std.Io.Writer.Error!void {
    try writer.print(
        "{s},{d},{d},{},{d},{d:.6},{d:.4},{d:.4},{d:.4},{d:.4},{d:.4},{d:.4},{d:.4},{s}",
        .{
            dataset,
            row.kappa_c,
            row.kappa_d,
            row.alpha_x100, // -1 = "no CP" sentinel
            row.n_queries,
            row.quality,
            row.avg_total_ms,
            row.p50_total_ms,
            row.p95_total_ms,
            row.avg_gather_ms,
            row.avg_prune_ms,
            row.avg_table_ms,
            row.avg_refine_ms,
            git_sha,
        },
    );
}

pub const csv_header: []const u8 =
    "dataset,kappa_c,kappa_d,alpha_x100,n_queries,quality,avg_total_ms,p50_total_ms,p95_total_ms,avg_gather_ms,avg_prune_ms,avg_table_ms,avg_refine_ms,git_sha";

// ---------------------------------------------------------------------------
// Smoke test — the synthetic-fixture surrogate for the real Table 1 sweep.
// Runs a single representative cell so a regression in any retrieval phase
// or in the metric/latency wiring shows up in `zig build test`.
// ---------------------------------------------------------------------------

const testing = std.testing;
const synthetic_fixture = @import("../io/synthetic_fixture.zig");
const token_dump = @import("../io/token_dump.zig");

test "bench: end-to-end smoke — single-cell sweep on synthetic fixture" {
    const gpa = testing.allocator;

    var fx = try synthetic_fixture.build(gpa, .{
        .seed = 9001,
        .n_docs = 100,
        .dim = 32,
        .vocab_size = 16,
        .avg_doc_len = 8,
    });
    defer fx.deinit(gpa);

    const fx_bytes = try token_dump.writeAlloc(gpa, fx.toBuild());
    defer gpa.free(fx_bytes);
    const td = try token_dump.parseBytes(fx_bytes);

    var image = try storage.build(&td, .{
        .kappa_total = 32,
        .seed = 9002,
        .mu = 4,
        .tau = 8,
        .epsilon = 1,
        .theta = 1,
        .hnsw = .{ .ef_construction = 32, .m = 4 },
    }, gpa);
    defer image.deinit(gpa);

    var idx = try storage.parse(image.bytes, gpa);
    defer idx.deinit(gpa);

    // Build a tiny query pack: 5 queries, each with 2 tokens drawn from
    // arbitrary centroids. Qrels assign each query one "gold" doc whose
    // first token's centroid happens to coincide with q_0.
    const n_queries: u32 = 5;
    const dim: u32 = idx.header.dim;
    const n_q_per: u32 = 2;
    const tokens_per_query: u32 = n_q_per * dim;

    var n_q_arr = try gpa.alloc(u32, n_queries);
    defer gpa.free(n_q_arr);
    var q_offs = try gpa.alloc(u64, n_queries + 1);
    defer gpa.free(q_offs);
    var tokens = try gpa.alloc(f32, n_queries * tokens_per_query);
    defer gpa.free(tokens);

    q_offs[0] = 0;
    var q: u32 = 0;
    while (q < n_queries) : (q += 1) {
        n_q_arr[q] = n_q_per;
        q_offs[q + 1] = q_offs[q] + tokens_per_query;
        // Pick centroids (q % kappa) and ((q+1) % kappa) as the two query tokens.
        const c0_idx: u32 = q % idx.header.kappa;
        const c1_idx: u32 = (q + 1) % idx.header.kappa;
        const lo: usize = q * tokens_per_query;
        @memcpy(
            tokens[lo .. lo + dim],
            idx.centroids[c0_idx * dim ..][0..dim],
        );
        @memcpy(
            tokens[lo + dim .. lo + 2 * dim],
            idx.centroids[c1_idx * dim ..][0..dim],
        );
    }

    // Qrels: pretend doc q is relevant for query q (deterministic).
    var qrels = try gpa.alloc(u32, n_queries);
    defer gpa.free(qrels);
    var qrels_lo = try gpa.alloc(u64, n_queries + 1);
    defer gpa.free(qrels_lo);
    qrels_lo[0] = 0;
    q = 0;
    while (q < n_queries) : (q += 1) {
        qrels[q] = q;
        qrels_lo[q + 1] = q + 1;
    }

    const pack = QueryPack{
        .n_q = n_q_arr,
        .q_offsets = q_offs,
        .tokens = tokens,
        .qrels = qrels,
        .qrels_lo = qrels_lo,
    };

    const row = try runCell(&idx, pack, 16, 50, null, .mrr_at_10, gpa);
    try testing.expectEqual(@as(u32, 16), row.kappa_c);
    try testing.expectEqual(@as(u32, 50), row.kappa_d);
    try testing.expectEqual(@as(i16, -1), row.alpha_x100);
    try testing.expectEqual(@as(u32, n_queries), row.n_queries);
    try testing.expect(row.quality >= 0.0 and row.quality <= 1.0);
    try testing.expect(std.math.isFinite(row.avg_total_ms));
    try testing.expect(row.avg_total_ms > 0.0);
    // Per-stage breakdown is non-negative.
    try testing.expect(row.avg_gather_ms >= 0.0);
    try testing.expect(row.avg_prune_ms >= 0.0);
    try testing.expect(row.avg_table_ms >= 0.0);
    try testing.expect(row.avg_refine_ms >= 0.0);
}

test "bench: CSV header + row format are stable" {
    var buf: [512]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    const row = SweepRow{
        .kappa_c = 80,
        .kappa_d = 1000,
        .alpha_x100 = 40,
        .n_queries = 6980,
        .quality = 0.39000,
        .avg_total_ms = 10.0,
        .p50_total_ms = 9.5,
        .p95_total_ms = 12.0,
        .avg_gather_ms = 2.0,
        .avg_prune_ms = 0.5,
        .avg_table_ms = 1.0,
        .avg_refine_ms = 6.5,
    };
    try formatCsvRow(&w, "msmarco-v1", "abc1234", row);
    const out = w.buffered();
    try testing.expect(std.mem.indexOf(u8, out, "msmarco-v1") != null);
    try testing.expect(std.mem.indexOf(u8, out, "abc1234") != null);
    try testing.expect(std.mem.indexOf(u8, out, ",80,") != null);
    try testing.expect(std.mem.indexOf(u8, out, ",1000,") != null);
    try testing.expect(std.mem.indexOf(u8, out, ",40,") != null); // alpha_x100
    try testing.expect(std.mem.indexOf(u8, out, "0.390000") != null);
}

test "bench: alpha=null serialises as -1 sentinel" {
    var buf: [512]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    const row = SweepRow{
        .kappa_c = 15,
        .kappa_d = 250,
        .alpha_x100 = -1,
        .n_queries = 100,
        .quality = 0.0,
        .avg_total_ms = 1.0,
        .p50_total_ms = 1.0,
        .p95_total_ms = 1.0,
        .avg_gather_ms = 0.0,
        .avg_prune_ms = 0.0,
        .avg_table_ms = 0.0,
        .avg_refine_ms = 0.0,
    };
    try formatCsvRow(&w, "test", "0", row);
    try testing.expect(std.mem.indexOf(u8, w.buffered(), ",-1,") != null);
}
