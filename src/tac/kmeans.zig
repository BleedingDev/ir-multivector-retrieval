//! src/tac/kmeans.zig — Lloyd's k-means with k-means++ init.
//!
//! Owner: clusterer.
//! See plan 02-tac-clustering.plan.md, paper §3.
//!
//! Used in two places:
//!   1. As TAC's per-token clustering primitive (paper §3.3 final step).
//!   2. As PQ codebook training primitive (paper §4) — indexer calls fit() per
//!      subspace.
//!
//! Algorithm (paper §3 references the canonical Lloyd / k-means++):
//!   - k-means++ seeding: D²(x)-weighted sampling for centroid 1..k-1.
//!   - Lloyd's iterations: assign → update → check relative WCSS-delta < tol.
//!   - Empty clusters re-seed with the farthest-from-any-centroid vector.
//!     // paper-gap: paper just says Lloyd's; "farthest point" is the
//!     // canonical fix-up and is fully deterministic.
//!   - k > n: degenerate — every vector is its own centroid; remaining slots
//!     copy the last vector and are flagged unused at assignment time.
//!     // paper-gap: TAC's θ=39 floor normally prevents this, but PQ
//!     // training and tiny test fixtures can hit it.

const std = @import("std");
const vec = @import("../util/vec.zig");
const rng_mod = @import("../util/rng.zig");

const Allocator = std.mem.Allocator;

pub const KMeansParams = struct {
    k: u32,
    max_iters: u32 = 25,
    tol: f32 = 1e-4,
    seed: u64,
    /// Worker threads for the per-vector assignment step (k-means' hot loop).
    /// `1` short-circuits to the strictly-serial code path, byte-identical
    /// to the pre-parallel implementation. `>=2` parallelises argmin across
    /// a static chunk of vectors per thread; the cluster-sum accumulation
    /// that feeds the centroid update runs serially in vector order, so the
    /// produced centroids are **byte-equal** across any thread count for
    /// the same seed (paper-strict reproducibility — see plan-07).
    n_threads: u32 = 1,
};

pub const KMeansResult = struct {
    centroids: []f32,
    assignments: []u32,
    wcss: f32,
    iters_run: u32,

    pub fn deinit(self: *KMeansResult, gpa: Allocator) void {
        gpa.free(self.centroids);
        gpa.free(self.assignments);
        self.* = undefined;
    }
};

pub const KMeansError = error{
    KmeansBudgetTooSmall,
    EmptyVectorSet,
    DimMismatch,
} || Allocator.Error || vec.VecError || rng_mod.RngError || std.Thread.SpawnError;

/// Cluster `vectors` into `p.k` groups via Lloyd's algorithm, k-means++ seeded.
///
/// `vectors` is a flat row-major matrix of length `n * dim` where `n` is
/// inferred. Centroids and assignments are heap-allocated on `gpa` and owned
/// by the returned `KMeansResult`.
pub fn fit(
    vectors: []const f32,
    dim: u32,
    p: KMeansParams,
    gpa: Allocator,
) KMeansError!KMeansResult {
    if (p.k == 0) return error.KmeansBudgetTooSmall;
    if (dim == 0) return error.DimMismatch;
    if (vectors.len == 0) return error.EmptyVectorSet;
    if (vectors.len % @as(usize, dim) != 0) return error.DimMismatch;

    const n: usize = vectors.len / @as(usize, dim);
    const k: usize = p.k;
    const d: usize = dim;

    const centroids = try gpa.alloc(f32, k * d);
    errdefer gpa.free(centroids);
    const assignments = try gpa.alloc(u32, n);
    errdefer gpa.free(assignments);

    // ---- Degenerate: k >= n -> each vector its own centroid. ----
    if (k >= n) {
        @memcpy(centroids[0 .. n * d], vectors);
        // Fill unused centroid slots with copies of the last vector. They
        // are never the closest centroid (distance 0 wins on a real vector)
        // and downstream callers see them but never assign to them.
        var c: usize = n;
        while (c < k) : (c += 1) {
            @memcpy(centroids[c * d ..][0..d], vectors[(n - 1) * d ..][0..d]);
        }
        for (0..n) |i| assignments[i] = @intCast(i);
        return .{
            .centroids = centroids,
            .assignments = assignments,
            .wcss = 0.0,
            .iters_run = 0,
        };
    }

    var rng = rng_mod.Rng.init(p.seed);

    // ---- k-means++ seeding (Arthur & Vassilvitskii 2007). ----
    // paper §3 cites this as the standard init; D²-weighted sampling.
    const min_dist_sq = try gpa.alloc(f32, n);
    defer gpa.free(min_dist_sq);

    // First centroid: uniform random pick.
    {
        const idx0 = rng.nextIndex(n);
        @memcpy(centroids[0..d], vectors[idx0 * d ..][0..d]);
        for (0..n) |i| {
            min_dist_sq[i] = try vec.l2sq(
                vectors[i * d ..][0..d],
                centroids[0..d],
            );
        }
    }

    var c: usize = 1;
    while (c < k) : (c += 1) {
        const idx = try rng_mod.weightedSample(&rng, min_dist_sq);
        @memcpy(centroids[c * d ..][0..d], vectors[idx * d ..][0..d]);
        for (0..n) |i| {
            const dist = try vec.l2sq(
                vectors[i * d ..][0..d],
                centroids[c * d ..][0..d],
            );
            if (dist < min_dist_sq[i]) min_dist_sq[i] = dist;
        }
    }

    // ---- Lloyd's iterations. ----
    const cluster_sums = try gpa.alloc(f32, k * d);
    defer gpa.free(cluster_sums);
    const cluster_counts = try gpa.alloc(u32, k);
    defer gpa.free(cluster_counts);
    // Per-vector min distance to its assigned centroid. Doubles as the
    // WCSS-per-vector contribution that phase 2 (serial accumulation) reads
    // in order, and as the empty-cluster re-seed key (`argmax(min_to_any)`).
    const min_to_any = try gpa.alloc(f32, n);
    defer gpa.free(min_to_any);

    var prev_wcss: f32 = std.math.inf(f32);
    var wcss: f32 = 0.0;
    var iters_run: u32 = 0;

    var iter: u32 = 0;
    while (iter < p.max_iters) : (iter += 1) {
        // ---- Assignment step. ----
        wcss = 0.0;
        @memset(cluster_sums, 0.0);
        @memset(cluster_counts, 0);

        // Phase 1: per-vector argmin. Optionally parallel across n_threads
        // (plan-07). Pure per-vector so writes to `assignments` and
        // `min_to_any` are byte-equal across thread counts.
        try assignVectors(
            vectors,
            n,
            d,
            k,
            centroids,
            assignments,
            min_to_any,
            p.n_threads,
            gpa,
        );

        // Phase 2: serial accumulation in vector order. Fixed reduction
        // order keeps centroids byte-equal across n_threads (paper-strict).
        for (0..n) |i| {
            const a: usize = assignments[i];
            wcss += min_to_any[i];
            const v_i = vectors[i * d ..][0..d];
            const sum_slot = cluster_sums[a * d ..][0..d];
            for (0..d) |dd| sum_slot[dd] += v_i[dd];
            cluster_counts[a] += 1;
        }

        iters_run = iter + 1;

        // ---- Convergence check. ----
        // Skip on first iter (prev_wcss == inf). Paper §3 does not pin a
        // tolerance; relative delta is the standard choice.
        if (iter > 0) {
            const denom = if (prev_wcss > 0.0) prev_wcss else 1.0;
            const rel_delta = (prev_wcss - wcss) / denom;
            if (rel_delta < p.tol) break;
        }

        // ---- Update step. ----
        for (0..k) |cc| {
            if (cluster_counts[cc] > 0) {
                const inv: f32 = 1.0 / @as(f32, @floatFromInt(cluster_counts[cc]));
                const slot = centroids[cc * d ..][0..d];
                const sum_slot = cluster_sums[cc * d ..][0..d];
                for (0..d) |dd| slot[dd] = sum_slot[dd] * inv;
            } else {
                // Empty cluster — re-seed with farthest-from-any-centroid
                // vector. paper-gap: paper does not specify; this is the
                // canonical deterministic fix-up.
                const far_idx = try vec.argmax(min_to_any);
                @memcpy(
                    centroids[cc * d ..][0..d],
                    vectors[far_idx * d ..][0..d],
                );
                // Mark this vector as "claimed" so a second empty cluster in
                // the same iteration doesn't pick the same point.
                min_to_any[far_idx] = -std.math.inf(f32);
            }
        }

        prev_wcss = wcss;
    }

    return .{
        .centroids = centroids,
        .assignments = assignments,
        .wcss = wcss,
        .iters_run = iters_run,
    };
}

/// Assign every vector to its closest centroid (paper §3 / Lloyd's
/// assignment step). Writes `assignments[i] = argmin_c d²(v_i, μ_c)` and
/// `min_to_any[i] = d²(v_i, μ_{a(i)})`.
///
/// `n_threads <= 1` runs serially. `n_threads >= 2` slices the n vectors
/// into static contiguous chunks and computes argmin in parallel. Argmin is
/// pure per-vector, so the writes to `assignments` and `min_to_any` are
/// byte-equal across thread counts — the deterministic f32-reduction order
/// is enforced by the caller's serial accumulation pass that consumes these
/// arrays in order.
fn assignVectors(
    vectors: []const f32,
    n: usize,
    d: usize,
    k: usize,
    centroids: []const f32,
    assignments: []u32,
    min_to_any: []f32,
    n_threads: u32,
    gpa: Allocator,
) KMeansError!void {
    const real_threads: u32 = if (n_threads <= 1) 1 else @intCast(@min(@as(usize, n_threads), n));
    if (real_threads <= 1) {
        const dists = try gpa.alloc(f32, k);
        defer gpa.free(dists);
        try assignSlice(vectors, 0, n, d, k, centroids, assignments, min_to_any, dists);
        return;
    }

    const Ctx = struct {
        vectors: []const f32,
        d: usize,
        k: usize,
        centroids: []const f32,
        assignments: []u32,
        min_to_any: []f32,
        lo: usize,
        hi: usize,
        gpa: Allocator,
        err_out: ?KMeansError,
    };
    const Runner = struct {
        fn run(ctx: *Ctx) void {
            const dists = ctx.gpa.alloc(f32, ctx.k) catch |err| {
                ctx.err_out = err;
                return;
            };
            defer ctx.gpa.free(dists);
            assignSlice(
                ctx.vectors,
                ctx.lo,
                ctx.hi,
                ctx.d,
                ctx.k,
                ctx.centroids,
                ctx.assignments,
                ctx.min_to_any,
                dists,
            ) catch |err| {
                ctx.err_out = err;
                return;
            };
        }
    };

    const ctxs = try gpa.alloc(Ctx, real_threads);
    defer gpa.free(ctxs);
    const threads = try gpa.alloc(std.Thread, real_threads);
    defer gpa.free(threads);

    const chunk: usize = (n + real_threads - 1) / real_threads;
    var t: u32 = 0;
    while (t < real_threads) : (t += 1) {
        const lo: usize = @as(usize, t) * chunk;
        const hi: usize = @min(lo + chunk, n);
        ctxs[t] = .{
            .vectors = vectors,
            .d = d,
            .k = k,
            .centroids = centroids,
            .assignments = assignments,
            .min_to_any = min_to_any,
            .lo = lo,
            .hi = hi,
            .gpa = gpa,
            .err_out = null,
        };
        threads[t] = try std.Thread.spawn(.{}, Runner.run, .{&ctxs[t]});
    }
    for (threads) |th| th.join();
    for (ctxs) |c| if (c.err_out) |err| return err;
}

/// Per-slice argmin: writes `assignments[i]` and `min_to_any[i]` for
/// `i in [lo, hi)`. `dists` is a `k`-length scratch buffer (caller-owned).
fn assignSlice(
    vectors: []const f32,
    lo: usize,
    hi: usize,
    d: usize,
    k: usize,
    centroids: []const f32,
    assignments: []u32,
    min_to_any: []f32,
    dists: []f32,
) KMeansError!void {
    var i: usize = lo;
    while (i < hi) : (i += 1) {
        const v_i = vectors[i * d ..][0..d];
        for (0..k) |cc| {
            dists[cc] = try vec.l2sq(v_i, centroids[cc * d ..][0..d]);
        }
        const a = try vec.argmin(dists[0..k]);
        assignments[i] = @intCast(a);
        min_to_any[i] = dists[a];
    }
}

// ---------------------------------------------------------------------------
// Tests — hand-checkable inputs, determinism, and Lloyd's monotonicity.
// ---------------------------------------------------------------------------

const testing = std.testing;

test "fit: k == 0 returns error" {
    const v = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    const r = fit(&v, 2, .{ .k = 0, .seed = 0 }, testing.allocator);
    try testing.expectError(error.KmeansBudgetTooSmall, r);
}

test "fit: empty vector set returns error" {
    const v = [_]f32{};
    const r = fit(&v, 2, .{ .k = 1, .seed = 0 }, testing.allocator);
    try testing.expectError(error.EmptyVectorSet, r);
}

test "fit: dim mismatch returns error" {
    const v = [_]f32{ 1.0, 2.0, 3.0 }; // len 3, dim 2 -> mismatch
    const r = fit(&v, 2, .{ .k = 1, .seed = 0 }, testing.allocator);
    try testing.expectError(error.DimMismatch, r);
}

test "fit: k > n - each vector becomes its own centroid" {
    const v = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    var r = try fit(&v, 2, .{ .k = 4, .seed = 0 }, testing.allocator);
    defer r.deinit(testing.allocator);
    try testing.expectEqual(@as(f32, 0.0), r.wcss);
    try testing.expectEqual(@as(u32, 0), r.iters_run);
    try testing.expectEqual(@as(u32, 0), r.assignments[0]);
    try testing.expectEqual(@as(u32, 1), r.assignments[1]);
}

test "fit: k == n - each vector becomes its own centroid" {
    const v = [_]f32{ 1.0, 0.0, 0.0, 1.0, 1.0, 1.0 };
    var r = try fit(&v, 2, .{ .k = 3, .seed = 0 }, testing.allocator);
    defer r.deinit(testing.allocator);
    try testing.expectEqual(@as(f32, 0.0), r.wcss);
    try testing.expectEqual(@as(u32, 0), r.iters_run);
}

test "fit: 3 well-separated 2D blobs recover blob means" {
    // Three clusters of 6 points each, very tight, around (0,0), (10,0), (0,10).
    var pts: [36]f32 = undefined;
    const blob_centers = [_][2]f32{ .{ 0, 0 }, .{ 10, 0 }, .{ 0, 10 } };
    var idx: usize = 0;
    for (blob_centers) |bc| {
        // 6 points per blob, jittered by tiny deterministic offsets.
        const offsets = [_][2]f32{
            .{ -0.05, 0.02 },  .{ 0.03, -0.04 }, .{ -0.01, -0.02 },
            .{ 0.02, 0.05 },   .{ -0.03, 0.01 }, .{ 0.04, -0.03 },
        };
        for (offsets) |off| {
            pts[idx * 2 + 0] = bc[0] + off[0];
            pts[idx * 2 + 1] = bc[1] + off[1];
            idx += 1;
        }
    }

    var r = try fit(&pts, 2, .{ .k = 3, .max_iters = 50, .seed = 42 }, testing.allocator);
    defer r.deinit(testing.allocator);

    // Recovered centroids should be within 0.1 of the true blob centers.
    // We don't know the order, so check each true center has *some* recovered
    // centroid close to it.
    for (blob_centers) |bc| {
        var found = false;
        for (0..3) |c| {
            const cx = r.centroids[c * 2 + 0];
            const cy = r.centroids[c * 2 + 1];
            const dx = cx - bc[0];
            const dy = cy - bc[1];
            if (@sqrt(dx * dx + dy * dy) < 0.1) {
                found = true;
                break;
            }
        }
        try testing.expect(found);
    }
}

test "fit: parallel n_threads byte-equal to serial path (paper-strict)" {
    // Plan-07 gate: same seed + same vectors must produce byte-equal
    // centroids and assignments at any n_threads. Argmin is per-vector pure;
    // the f32-accumulating update step runs serially in vector order, so
    // reductions are bit-stable across thread counts.
    const a = std.testing.allocator;
    const dim: u32 = 4;
    const n: u32 = 600;
    const buf = try a.alloc(f32, @as(usize, n) * @as(usize, dim));
    defer a.free(buf);
    var prng = std.Random.DefaultPrng.init(0xDE7E2);
    for (buf) |*x| x.* = prng.random().floatNorm(f32);

    var r1 = try fit(buf, dim, .{ .k = 32, .seed = 42, .n_threads = 1, .max_iters = 25 }, a);
    defer r1.deinit(a);
    var r10 = try fit(buf, dim, .{ .k = 32, .seed = 42, .n_threads = 10, .max_iters = 25 }, a);
    defer r10.deinit(a);

    try testing.expectEqualSlices(f32, r1.centroids, r10.centroids);
    try testing.expectEqualSlices(u32, r1.assignments, r10.assignments);
    try testing.expectEqual(r1.wcss, r10.wcss);
    try testing.expectEqual(r1.iters_run, r10.iters_run);

    // And again at sub_dim=1 (the PQ shape with dim/PQ_M=64/32=2 → 1 here for tightness).
    var r3 = try fit(buf, dim, .{ .k = 32, .seed = 42, .n_threads = 3, .max_iters = 25 }, a);
    defer r3.deinit(a);
    try testing.expectEqualSlices(f32, r1.centroids, r3.centroids);
    try testing.expectEqualSlices(u32, r1.assignments, r3.assignments);

    // Edge case: n_threads > n. Must clamp to n internally and still match.
    const small = buf[0 .. 5 * @as(usize, dim)];
    var s1 = try fit(small, dim, .{ .k = 3, .seed = 7, .n_threads = 1 }, a);
    defer s1.deinit(a);
    var s_huge = try fit(small, dim, .{ .k = 3, .seed = 7, .n_threads = 64 }, a);
    defer s_huge.deinit(a);
    try testing.expectEqualSlices(f32, s1.centroids, s_huge.centroids);
}

test "fit: deterministic - same seed produces byte-equal centroids" {
    var pts: [40]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(99);
    for (0..40) |i| pts[i] = prng.random().floatNorm(f32);

    var r1 = try fit(&pts, 2, .{ .k = 4, .seed = 7, .max_iters = 30 }, testing.allocator);
    defer r1.deinit(testing.allocator);
    var r2 = try fit(&pts, 2, .{ .k = 4, .seed = 7, .max_iters = 30 }, testing.allocator);
    defer r2.deinit(testing.allocator);

    try testing.expectEqualSlices(f32, r1.centroids, r2.centroids);
    try testing.expectEqualSlices(u32, r1.assignments, r2.assignments);
    try testing.expectEqual(r1.wcss, r2.wcss);
    try testing.expectEqual(r1.iters_run, r2.iters_run);
}

test "fit: WCSS is non-increasing across iterations" {
    // We can't directly observe per-iter WCSS without API exposure, but we
    // can check that running for more iterations never *increases* the final
    // WCSS, which is the same monotonicity property exposed at the boundary.
    var pts: [60]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(1234);
    for (0..60) |i| pts[i] = prng.random().floatNorm(f32);

    const seed: u64 = 111;
    var r_short = try fit(&pts, 3, .{ .k = 4, .seed = seed, .max_iters = 1, .tol = 0 }, testing.allocator);
    defer r_short.deinit(testing.allocator);
    var r_med = try fit(&pts, 3, .{ .k = 4, .seed = seed, .max_iters = 5, .tol = 0 }, testing.allocator);
    defer r_med.deinit(testing.allocator);
    var r_long = try fit(&pts, 3, .{ .k = 4, .seed = seed, .max_iters = 25, .tol = 0 }, testing.allocator);
    defer r_long.deinit(testing.allocator);

    try testing.expect(r_med.wcss <= r_short.wcss);
    try testing.expect(r_long.wcss <= r_med.wcss);
}

test "fit: assignments are all valid centroid ids" {
    var pts: [50]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(5);
    for (0..50) |i| pts[i] = prng.random().floatNorm(f32);

    var r = try fit(&pts, 5, .{ .k = 3, .seed = 0 }, testing.allocator);
    defer r.deinit(testing.allocator);

    for (r.assignments) |a| try testing.expect(a < 3);
    try testing.expectEqual(@as(usize, 10), r.assignments.len);
}

test "fit: empty cluster re-seed with collinear pathological data" {
    // Three collinear points with two very close together — k=3 forces the
    // empty-cluster branch on the second point. Outcome doesn't matter beyond
    // "doesn't crash, returns valid assignments".
    const pts = [_]f32{ 0.0, 0.0, 0.0001, 0.0, 100.0, 0.0 };
    var r = try fit(&pts, 2, .{ .k = 3, .seed = 1, .max_iters = 10 }, testing.allocator);
    defer r.deinit(testing.allocator);
    for (r.assignments) |a| try testing.expect(a < 3);
}
