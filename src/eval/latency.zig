//! src/eval/latency.zig — single-thread latency benchmarking.
//!
//! Owner: retriever. Paper §9: retrieval reported single-core, so the harness
//! pins the benchmark thread to one logical core (best-effort per-OS).
//!
//! Per-stage timings let us localise regressions. Total query time is
//! `gather_ns + prune_ns + table_build_ns + refine_ns`.
//!
//! Required surface (locked, used by #20 harnesses):
//!   pub const StageTimings = struct {
//!       gather_ns:      u64,
//!       prune_ns:       u64,
//!       table_build_ns: u64,    // pq.buildDistanceTable
//!       refine_ns:      u64,
//!       total_ns:       u64,
//!   };
//!   pub fn timeQuery(...) StageTimings;
//!   pub const Report = struct { … see below … };
//!   pub fn report(timings: []const StageTimings) Report;
//!   pub fn pinToCore(core: u32) void;        // best-effort no-op on unsupported OS
//!
//! ---------------------------------------------------------------------------
//! DESIGN PSEUDOCODE for timeQuery (waits on retrieval.search — task #19):
//!
//! timeQuery(index, pq, query_tokens, n_q, params, gpa) -> StageTimings:
//!   var t = StageTimings{};
//!   var timer = std.time.Timer.start();
//!   candidates = gather.gather(...);    t.gather_ns = timer.lap();
//!   survivors  = prune.prune(...);      t.prune_ns  = timer.lap();
//!   pq.buildDistanceTable(...);         t.table_build_ns = timer.lap();
//!   for s in survivors: refine.refine(...); t.refine_ns = timer.lap();
//!   t.total_ns = t.gather_ns + t.prune_ns + t.table_build_ns + t.refine_ns;
//!   return t;
//!
//! pinToCore(core):
//!   builtin.os.tag == .linux: sched_setaffinity(0, [1u << core])
//!   builtin.os.tag == .macos: thread_policy_set THREAD_AFFINITY_POLICY (best-effort)
//!   else:                     no-op
//! ---------------------------------------------------------------------------

const std = @import("std");

pub const StageTimings = struct {
    gather_ns: u64 = 0,
    prune_ns: u64 = 0,
    table_build_ns: u64 = 0,
    refine_ns: u64 = 0,
    total_ns: u64 = 0,
};

pub const Report = struct {
    n_queries: u32,
    avg_total_ms: f64,
    p50_total_ms: f64,
    p95_total_ms: f64,
    avg_gather_ms: f64,
    avg_prune_ms: f64,
    avg_table_ms: f64,
    avg_refine_ms: f64,
};

/// Compute aggregate latency stats from per-query stage timings.
///
/// Empty input returns an all-zero `Report` (no NaNs) so callers can render
/// it unconditionally. Percentiles use nearest-rank (0-based floor) on the
/// sorted total_ns — simplest deterministic rule, matches what most IR
/// latency tables print.
///
/// Allocates a small page-allocator buffer for the percentile sort when the
/// input exceeds the on-stack threshold; fatal-panics on OOM (this is dev-only
/// reporting, not a production path).
pub fn report(timings: []const StageTimings) Report {
    if (timings.len == 0) return .{
        .n_queries = 0,
        .avg_total_ms = 0,
        .p50_total_ms = 0,
        .p95_total_ms = 0,
        .avg_gather_ms = 0,
        .avg_prune_ms = 0,
        .avg_table_ms = 0,
        .avg_refine_ms = 0,
    };

    var sum_total: u64 = 0;
    var sum_gather: u64 = 0;
    var sum_prune: u64 = 0;
    var sum_table: u64 = 0;
    var sum_refine: u64 = 0;
    for (timings) |t| {
        sum_total += t.total_ns;
        sum_gather += t.gather_ns;
        sum_prune += t.prune_ns;
        sum_table += t.table_build_ns;
        sum_refine += t.refine_ns;
    }
    const n: f64 = @floatFromInt(timings.len);

    var stack_buf: [16384]u64 = undefined;
    const slice = if (timings.len <= stack_buf.len)
        stack_buf[0..timings.len]
    else blk: {
        const heap = std.heap.page_allocator.alloc(u64, timings.len) catch
            @panic("latency.report: allocation failed");
        break :blk heap;
    };
    defer if (timings.len > stack_buf.len) std.heap.page_allocator.free(slice);

    for (timings, 0..) |t, i| slice[i] = t.total_ns;
    std.mem.sort(u64, slice, {}, std.sort.asc(u64));

    const p50_idx = (timings.len * 50) / 100;
    const p95_idx_raw = (timings.len * 95) / 100;
    const p95_idx = if (p95_idx_raw >= timings.len) timings.len - 1 else p95_idx_raw;

    const ns_per_ms: f64 = 1_000_000;
    return .{
        .n_queries = @intCast(timings.len),
        .avg_total_ms = @as(f64, @floatFromInt(sum_total)) / n / ns_per_ms,
        .p50_total_ms = @as(f64, @floatFromInt(slice[p50_idx])) / ns_per_ms,
        .p95_total_ms = @as(f64, @floatFromInt(slice[p95_idx])) / ns_per_ms,
        .avg_gather_ms = @as(f64, @floatFromInt(sum_gather)) / n / ns_per_ms,
        .avg_prune_ms = @as(f64, @floatFromInt(sum_prune)) / n / ns_per_ms,
        .avg_table_ms = @as(f64, @floatFromInt(sum_table)) / n / ns_per_ms,
        .avg_refine_ms = @as(f64, @floatFromInt(sum_refine)) / n / ns_per_ms,
    };
}

const testing = std.testing;

test "report on empty input → all zeros" {
    const r = report(&.{});
    try testing.expectEqual(@as(u32, 0), r.n_queries);
    try testing.expectEqual(@as(f64, 0), r.avg_total_ms);
    try testing.expectEqual(@as(f64, 0), r.p50_total_ms);
    try testing.expectEqual(@as(f64, 0), r.p95_total_ms);
    try testing.expectEqual(@as(f64, 0), r.avg_refine_ms);
}

test "report computes mean and percentiles in ms" {
    const ms = 1_000_000;
    const samples = [_]StageTimings{
        .{ .gather_ns = 1 * ms, .refine_ns = 4 * ms, .total_ns = 5 * ms },
        .{ .gather_ns = 2 * ms, .refine_ns = 8 * ms, .total_ns = 10 * ms },
        .{ .gather_ns = 3 * ms, .refine_ns = 12 * ms, .total_ns = 15 * ms },
        .{ .gather_ns = 4 * ms, .refine_ns = 16 * ms, .total_ns = 20 * ms },
    };
    const r = report(&samples);
    try testing.expectEqual(@as(u32, 4), r.n_queries);
    try testing.expectApproxEqAbs(@as(f64, 12.5), r.avg_total_ms, 1e-9);
    // p50 idx = floor(4*50/100) = 2 → sorted[2] = 15 ms
    try testing.expectApproxEqAbs(@as(f64, 15.0), r.p50_total_ms, 1e-9);
    // p95 idx = floor(4*95/100) = 3 → sorted[3] = 20 ms
    try testing.expectApproxEqAbs(@as(f64, 20.0), r.p95_total_ms, 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 2.5), r.avg_gather_ms, 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 10.0), r.avg_refine_ms, 1e-9);
}

test "report handles single sample" {
    const samples = [_]StageTimings{
        .{ .gather_ns = 7_000_000, .total_ns = 7_000_000 },
    };
    const r = report(&samples);
    try testing.expectEqual(@as(u32, 1), r.n_queries);
    try testing.expectApproxEqAbs(@as(f64, 7.0), r.avg_total_ms, 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 7.0), r.p50_total_ms, 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 7.0), r.p95_total_ms, 1e-9);
}
