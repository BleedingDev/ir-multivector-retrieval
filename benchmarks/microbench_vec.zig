//! benchmarks/microbench_vec.zig — kernel-level timing for src/util/vec.zig.
//!
//! Establishes baseline / post-specialization GFLOPs for the dot/l2sq/normalize
//! kernels at the dims that actually appear in this codebase: 2 and 4 (PQ
//! subspaces), 64 (jina-colbert-v2-64), 128 (ColBERTv2.0). We also bench dim=96
//! as an unspecialized fallback control.
//!
//! Usage: `zig build -Doptimize=ReleaseFast run-microbench_vec`
//!
//! plan-11 (post-hackathon): we want to know how much the comptime
//! specialization buys us over the runtime-`lane_count` generic path before
//! claiming a perf win.

const std = @import("std");
const tac = @import("tac");
const vec = tac.util.vec;

const Allocator = std.mem.Allocator;

/// Monotonic ns timestamp; Zig 0.16 moved std.time.Timer/Instant behind std.Io,
/// so we go direct to libc (same pattern as src/main.zig:763).
fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 +
        @as(u64, @intCast(ts.nsec));
}

/// One bench result row.
const Row = struct {
    name: []const u8,
    dim: usize,
    iters: usize,
    ns_per_op: f64,
    gflops: f64,
};

fn fillRamp(buf: []f32, offset: f32) void {
    for (buf, 0..) |*x, i| {
        x.* = @as(f32, @floatFromInt(i)) * 0.001 + offset;
    }
}

fn benchDot(allocator: Allocator, comptime label: []const u8, dim: usize, iters: usize) !Row {
    // Allocate a 1024-row pool so we exercise different inputs each iteration;
    // a single pair lets LLVM hoist `vec.dot(a,b)` out of the loop now that the
    // dispatcher is `inline`. Cycling through cache-resident rows defeats LICM
    // without measuring memory bandwidth.
    const pool_rows: usize = 1024;
    const buf = try allocator.alloc(f32, dim * pool_rows);
    defer allocator.free(buf);
    fillRamp(buf, 0.1);

    var warm: usize = 0;
    var sink: f32 = 0.0;
    while (warm < @max(iters / 100, 1)) : (warm += 1) {
        const i = warm & (pool_rows - 1);
        const j = (warm + 1) & (pool_rows - 1);
        sink += try vec.dot(buf[i * dim ..][0..dim], buf[j * dim ..][0..dim]);
    }
    std.mem.doNotOptimizeAway(sink);

    const t0 = nowNs();
    var k: usize = 0;
    var acc: f32 = 0.0;
    while (k < iters) : (k += 1) {
        const i = k & (pool_rows - 1);
        const j = (k + 1) & (pool_rows - 1);
        acc += try vec.dot(buf[i * dim ..][0..dim], buf[j * dim ..][0..dim]);
    }
    const ns: u64 = nowNs() - t0;
    std.mem.doNotOptimizeAway(acc);

    const ns_per_op = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(iters));
    // dot = dim multiplies + (dim-1) adds ≈ 2*dim FLOPs.
    const flops_per_op = 2.0 * @as(f64, @floatFromInt(dim));
    const gflops = flops_per_op / ns_per_op;
    return .{ .name = label, .dim = dim, .iters = iters, .ns_per_op = ns_per_op, .gflops = gflops };
}

fn benchL2sq(allocator: Allocator, comptime label: []const u8, dim: usize, iters: usize) !Row {
    const pool_rows: usize = 1024;
    const buf = try allocator.alloc(f32, dim * pool_rows);
    defer allocator.free(buf);
    fillRamp(buf, 0.1);

    var warm: usize = 0;
    var sink: f32 = 0.0;
    while (warm < @max(iters / 100, 1)) : (warm += 1) {
        const i = warm & (pool_rows - 1);
        const j = (warm + 1) & (pool_rows - 1);
        sink += try vec.l2sq(buf[i * dim ..][0..dim], buf[j * dim ..][0..dim]);
    }
    std.mem.doNotOptimizeAway(sink);

    const t0 = nowNs();
    var k: usize = 0;
    var acc: f32 = 0.0;
    while (k < iters) : (k += 1) {
        const i = k & (pool_rows - 1);
        const j = (k + 1) & (pool_rows - 1);
        acc += try vec.l2sq(buf[i * dim ..][0..dim], buf[j * dim ..][0..dim]);
    }
    const ns: u64 = nowNs() - t0;
    std.mem.doNotOptimizeAway(acc);

    const ns_per_op = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(iters));
    // l2sq = dim subtracts + dim multiplies + (dim-1) adds ≈ 3*dim FLOPs.
    const flops_per_op = 3.0 * @as(f64, @floatFromInt(dim));
    const gflops = flops_per_op / ns_per_op;
    return .{ .name = label, .dim = dim, .iters = iters, .ns_per_op = ns_per_op, .gflops = gflops };
}

fn benchNormalize(allocator: Allocator, comptime label: []const u8, dim: usize, iters: usize) !Row {
    // We mutate in place; reset every iteration would dominate. Instead, fill
    // once, time the normalize, then refill. Refill cost is amortised across
    // iters but not zero — so this is a slight pessimism. Acceptable for a
    // relative comparison.
    const v = try allocator.alloc(f32, dim);
    defer allocator.free(v);

    const t0 = nowNs();
    var k: usize = 0;
    while (k < iters) : (k += 1) {
        // Refill is part of the timed loop so successive normalize calls
        // see a non-unit vector. Subtract a small ramp-fill cost in analysis
        // (it's identical across baseline + specialized so cancels for ratio).
        fillRamp(v, 0.5);
        try vec.normalizeInPlace(v);
    }
    const ns: u64 = nowNs() - t0;
    std.mem.doNotOptimizeAway(v[0]);

    const ns_per_op = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(iters));
    // normalize: 2*dim multiplies + ~dim adds + 1 sqrt + 1 divide + dim multiplies
    // ≈ 4*dim FLOPs (sqrt/div ignored).
    const flops_per_op = 4.0 * @as(f64, @floatFromInt(dim));
    const gflops = flops_per_op / ns_per_op;
    return .{ .name = label, .dim = dim, .iters = iters, .ns_per_op = ns_per_op, .gflops = gflops };
}

fn printRow(row: Row) void {
    std.debug.print(
        "  {s:<14} dim={d:<4}  iters={d:<10}  {d:>9.2} ns/op   {d:>7.3} GFLOP/s\n",
        .{ row.name, row.dim, row.iters, row.ns_per_op, row.gflops },
    );
}

/// Mimic the inner argmin loop that kmeans.fit (assignment phase) does on
/// every point: for each centroid, compute l2sq(point, centroid); track
/// argmin. n_centroids=256, n_points=1000 — small enough to fit in cache,
/// large enough to amortize loop overhead. Reports median ns per point.
fn benchKmeansAssign(allocator: Allocator, comptime label: []const u8, dim: usize, k: usize, n_points: usize) !Row {
    const points = try allocator.alloc(f32, dim * n_points);
    defer allocator.free(points);
    const cents = try allocator.alloc(f32, dim * k);
    defer allocator.free(cents);

    // Deterministic synthetic fill — we just need data, not realism.
    for (points, 0..) |*p, i| p.* = @sin(@as(f32, @floatFromInt(i)) * 0.013) * 0.5;
    for (cents, 0..) |*c, i| c.* = @cos(@as(f32, @floatFromInt(i)) * 0.017) * 0.5;

    // Warm-up.
    var warm_sink: usize = 0;
    {
        var p: usize = 0;
        while (p < n_points) : (p += 1) {
            const point = points[p * dim ..][0..dim];
            var best_d: f32 = std.math.inf(f32);
            var best_c: usize = 0;
            var c: usize = 0;
            while (c < k) : (c += 1) {
                const d = try vec.l2sq(point, cents[c * dim ..][0..dim]);
                if (d < best_d) {
                    best_d = d;
                    best_c = c;
                }
            }
            warm_sink +%= best_c;
        }
    }
    std.mem.doNotOptimizeAway(warm_sink);

    const t0 = nowNs();
    var sink: usize = 0;
    var p: usize = 0;
    while (p < n_points) : (p += 1) {
        const point = points[p * dim ..][0..dim];
        var best_d: f32 = std.math.inf(f32);
        var best_c: usize = 0;
        var c: usize = 0;
        while (c < k) : (c += 1) {
            const d = try vec.l2sq(point, cents[c * dim ..][0..dim]);
            if (d < best_d) {
                best_d = d;
                best_c = c;
            }
        }
        sink +%= best_c;
    }
    const ns: u64 = nowNs() - t0;
    std.mem.doNotOptimizeAway(sink);

    const ns_per_op = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(n_points));
    // Per point: k * (3*dim) FLOPs.
    const flops_per_op = @as(f64, @floatFromInt(k)) * 3.0 * @as(f64, @floatFromInt(dim));
    const gflops = flops_per_op / ns_per_op;
    return .{ .name = label, .dim = dim, .iters = n_points, .ns_per_op = ns_per_op, .gflops = gflops };
}

/// Mirrors pq.encode's exact call shape: l2sq(sub, codebook[m, c]) with
/// `sub_dim` read from a struct field (not a stack-local). Tests whether
/// plan-11's `pub inline fn` dispatcher is enough or if we need exact-dim
/// helpers exposed for callers that hold dim in a runtime field.
const MockPQ = struct {
    sub_dim: u32,
    codebook: []const f32,
};

fn benchPqEncodePattern(allocator: Allocator, comptime label: []const u8, sub_dim: u32, n_subspaces: usize, n_centroids: usize, n_tokens: usize) !Row {
    const dim = @as(usize, sub_dim) * n_subspaces;
    const cb_len = @as(usize, sub_dim) * n_centroids * n_subspaces;
    const codebook = try allocator.alloc(f32, cb_len);
    defer allocator.free(codebook);
    for (codebook, 0..) |*x, i| x.* = @sin(@as(f32, @floatFromInt(i)) * 0.011);
    const residuals = try allocator.alloc(f32, dim * n_tokens);
    defer allocator.free(residuals);
    for (residuals, 0..) |*x, i| x.* = @cos(@as(f32, @floatFromInt(i)) * 0.013);

    const pq = MockPQ{ .sub_dim = sub_dim, .codebook = codebook };
    // Indirection through a heap pointer mirrors `*const PQ` in pq.zig:59.
    const pq_ptr: *const MockPQ = &pq;

    var warm_sink: usize = 0;
    {
        var t: usize = 0;
        while (t < n_tokens / 100 + 1) : (t += 1) {
            const r = residuals[t * dim ..][0..dim];
            var m: usize = 0;
            while (m < n_subspaces) : (m += 1) {
                const sub = r[m * pq_ptr.sub_dim ..][0..pq_ptr.sub_dim];
                var best_c: usize = 0;
                var best_d: f32 = std.math.inf(f32);
                var c: usize = 0;
                while (c < n_centroids) : (c += 1) {
                    const cb_base = ((m * n_centroids) + c) * pq_ptr.sub_dim;
                    const cb = pq_ptr.codebook[cb_base..][0..pq_ptr.sub_dim];
                    const d = try vec.l2sq(sub, cb);
                    if (d < best_d) {
                        best_d = d;
                        best_c = c;
                    }
                }
                warm_sink +%= best_c;
            }
        }
    }
    std.mem.doNotOptimizeAway(warm_sink);

    const t0 = nowNs();
    var sink: usize = 0;
    var t: usize = 0;
    while (t < n_tokens) : (t += 1) {
        const r = residuals[t * dim ..][0..dim];
        var m: usize = 0;
        while (m < n_subspaces) : (m += 1) {
            const sub = r[m * pq_ptr.sub_dim ..][0..pq_ptr.sub_dim];
            var best_c: usize = 0;
            var best_d: f32 = std.math.inf(f32);
            var c: usize = 0;
            while (c < n_centroids) : (c += 1) {
                const cb_base = ((m * n_centroids) + c) * pq_ptr.sub_dim;
                const cb = pq_ptr.codebook[cb_base..][0..pq_ptr.sub_dim];
                const d = try vec.l2sq(sub, cb);
                if (d < best_d) {
                    best_d = d;
                    best_c = c;
                }
            }
            sink +%= best_c;
        }
    }
    const ns: u64 = nowNs() - t0;
    std.mem.doNotOptimizeAway(sink);

    const ns_per_op = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(n_tokens));
    const flops_per_op = @as(f64, @floatFromInt(n_subspaces)) *
        @as(f64, @floatFromInt(n_centroids)) *
        3.0 * @as(f64, @floatFromInt(sub_dim));
    const gflops = flops_per_op / ns_per_op;
    return .{ .name = label, .dim = sub_dim, .iters = n_tokens, .ns_per_op = ns_per_op, .gflops = gflops };
}

pub fn main() !void {
    var gpa_state: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa_state.deinit();
    const a = gpa_state.allocator();

    std.debug.print("microbench_vec — kernel timings (lane_count = {d})\n\n", .{vec.lane_count});

    // dim=2 / dim=4 are PQ subspaces; called *very* often but each call is tiny.
    // Use higher iter count.
    const small_iters: usize = 4_000_000;
    // dim=64 / dim=128 are token-level; each call does more work.
    const big_iters: usize = 1_000_000;

    std.debug.print("dot:\n", .{});
    printRow(try benchDot(a, "dot", 2, small_iters));
    printRow(try benchDot(a, "dot", 4, small_iters));
    printRow(try benchDot(a, "dot", 64, big_iters));
    printRow(try benchDot(a, "dot", 96, big_iters));
    printRow(try benchDot(a, "dot", 128, big_iters));

    std.debug.print("\nl2sq:\n", .{});
    printRow(try benchL2sq(a, "l2sq", 2, small_iters));
    printRow(try benchL2sq(a, "l2sq", 4, small_iters));
    printRow(try benchL2sq(a, "l2sq", 64, big_iters));
    printRow(try benchL2sq(a, "l2sq", 96, big_iters));
    printRow(try benchL2sq(a, "l2sq", 128, big_iters));

    std.debug.print("\nnormalizeInPlace (includes refill cost):\n", .{});
    printRow(try benchNormalize(a, "normalize", 64, big_iters));
    printRow(try benchNormalize(a, "normalize", 128, big_iters));

    // kmeans assignment hot loop: this is the dominant caller of l2sq in
    // both tac.clusterFlat and pq.train. Per-point cost = k centroids ×
    // l2sq(dim). If specialization helps anywhere, it shows here.
    std.debug.print("\nkmeans assignment (per-point l2sq vs k centroids):\n", .{});
    printRow(try benchKmeansAssign(a, "assign k=256", 2, 256, 5_000)); // PQ subspace
    printRow(try benchKmeansAssign(a, "assign k=256", 4, 256, 5_000)); // PQ subspace
    printRow(try benchKmeansAssign(a, "assign k=256", 64, 256, 1_000)); // jina-colbert
    printRow(try benchKmeansAssign(a, "assign k=256", 128, 256, 1_000)); // ColBERTv2

    // pq.encode pattern: M=32 subspaces × 256 centroids × per-token l2sq with
    // `sub_dim` from a struct field. Plan-11 follow-up: tests if the runtime
    // sub_dim through a `*const PQ` pointer breaks the inline-dispatcher's
    // ability to constant-fold the switch.
    std.debug.print("\npq.encode pattern (M=32, 256 centroids, runtime sub_dim from struct field):\n", .{});
    printRow(try benchPqEncodePattern(a, "pq.encode", 2, 32, 256, 2_000)); // jina-colbert-v2-64
    printRow(try benchPqEncodePattern(a, "pq.encode", 4, 32, 256, 2_000)); // ColBERTv2.0

    // pq.encode pattern with the dispatch HOISTED: branch ONCE on sub_dim
    // outside the (m, c) loops and call comptime-known l2sq. Upper-bound on
    // what finding #3 could buy us if pq.zig adopted exact-dim helpers.
    std.debug.print("\npq.encode pattern (dispatch hoisted to outer level):\n", .{});
    printRow(try benchPqEncodePatternHoisted(a, "pq.encode-h", 2, 32, 256, 2_000));
    printRow(try benchPqEncodePatternHoisted(a, "pq.encode-h", 4, 32, 256, 2_000));
}

/// Counterfactual: same workload as benchPqEncodePattern, but the sub_dim
/// switch is hoisted to the outer level. Mirrors what pq.zig would look like
/// if finding #3 were implemented (one runtime branch per token, then exact-
/// dim l2sq inside the inner loop).
fn benchPqEncodePatternHoisted(allocator: Allocator, comptime label: []const u8, sub_dim: u32, n_subspaces: usize, n_centroids: usize, n_tokens: usize) !Row {
    const dim = @as(usize, sub_dim) * n_subspaces;
    const cb_len = @as(usize, sub_dim) * n_centroids * n_subspaces;
    const codebook = try allocator.alloc(f32, cb_len);
    defer allocator.free(codebook);
    for (codebook, 0..) |*x, i| x.* = @sin(@as(f32, @floatFromInt(i)) * 0.011);
    const residuals = try allocator.alloc(f32, dim * n_tokens);
    defer allocator.free(residuals);
    for (residuals, 0..) |*x, i| x.* = @cos(@as(f32, @floatFromInt(i)) * 0.013);

    const pq = MockPQ{ .sub_dim = sub_dim, .codebook = codebook };
    const pq_ptr: *const MockPQ = &pq;

    // Warm-up + timed phase share a closure that branches once on sub_dim.
    const Inner = struct {
        fn run(comptime sd: u32, _pq: *const MockPQ, _r: []const f32, _ns: usize, _nc: usize) !usize {
            var sink: usize = 0;
            var t: usize = 0;
            const _dim = @as(usize, sd) * _ns;
            while (t < _r.len / _dim) : (t += 1) {
                const r = _r[t * _dim ..][0.._dim];
                var m: usize = 0;
                while (m < _ns) : (m += 1) {
                    const sub = r[m * sd ..][0..sd];
                    var best_c: usize = 0;
                    var best_d: f32 = std.math.inf(f32);
                    var c: usize = 0;
                    while (c < _nc) : (c += 1) {
                        const cb_base = ((m * _nc) + c) * sd;
                        const cb = _pq.codebook[cb_base..][0..sd];
                        // Public `vec.l2sq` here — but the slice length is
                        // built from comptime `sd`, so the inline dispatcher's
                        // switch folds at compile time. Same end-state as
                        // calling a hypothetical exact-dim helper.
                        const d = try vec.l2sq(sub, cb);
                        if (d < best_d) {
                            best_d = d;
                            best_c = c;
                        }
                    }
                    sink +%= best_c;
                }
            }
            return sink;
        }
    };

    // Warm-up
    var warm: usize = 0;
    if (pq_ptr.sub_dim == 2) {
        warm = try Inner.run(2, pq_ptr, residuals[0..dim], n_subspaces, n_centroids);
    } else if (pq_ptr.sub_dim == 4) {
        warm = try Inner.run(4, pq_ptr, residuals[0..dim], n_subspaces, n_centroids);
    }
    std.mem.doNotOptimizeAway(warm);

    const t0 = nowNs();
    var sink: usize = 0;
    if (pq_ptr.sub_dim == 2) {
        sink = try Inner.run(2, pq_ptr, residuals, n_subspaces, n_centroids);
    } else if (pq_ptr.sub_dim == 4) {
        sink = try Inner.run(4, pq_ptr, residuals, n_subspaces, n_centroids);
    } else {
        unreachable;
    }
    const ns: u64 = nowNs() - t0;
    std.mem.doNotOptimizeAway(sink);

    const ns_per_op = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(n_tokens));
    const flops_per_op = @as(f64, @floatFromInt(n_subspaces)) *
        @as(f64, @floatFromInt(n_centroids)) *
        3.0 * @as(f64, @floatFromInt(sub_dim));
    const gflops = flops_per_op / ns_per_op;
    return .{ .name = label, .dim = sub_dim, .iters = n_tokens, .ns_per_op = ns_per_op, .gflops = gflops };
}
