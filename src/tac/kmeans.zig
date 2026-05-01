//! src/tac/kmeans.zig — Lloyd's k-means with k-means++ init.
//!
//! Owner: clusterer.
//! See plan 02-tac-clustering.plan.md, paper §3.
//!
//! Used in two places:
//!   1. As TAC's per-token clustering primitive (paper §3.3 phase 4 final step).
//!   2. As PQ codebook training primitive (paper §4) — indexer calls fit() per subspace.
//!
//! ============================================================================
//! DESIGN — pseudocode-as-comments while primitives (#1, #2) are not yet ready.
//! ============================================================================
//!
//! Public surface (frozen — indexer + tac.cluster will both call this):
//!
//!   pub const KMeansParams = struct {
//!       k: u32,                      // requested centroid count
//!       max_iters: u32 = 25,         // hard cap on Lloyd iterations
//!       tol: f32 = 1e-4,             // relative WCSS-delta convergence threshold
//!       seed: u64,                   // determinism — XORed with token id at TAC level
//!   };
//!
//!   pub const KMeansResult = struct {
//!       centroids: []f32,            // k * dim, caller-owned, allocator-attached
//!       assignments: []u32,          // n,     caller-owned (vector i → centroid id)
//!       wcss: f32,                   // final within-cluster sum of squares
//!       iters_run: u32,              // actual iterations performed (≤ max_iters)
//!   };
//!
//!   pub const KMeansError = error{
//!       KmeansBudgetTooSmall,        // k == 0
//!       EmptyVectorSet,              // n == 0
//!       DimMismatch,                 // vectors.len % dim != 0
//!       OutOfMemory,
//!   };
//!
//!   pub fn fit(
//!       vectors: []const f32,        // length n*dim, row-major
//!       dim: u32,
//!       p: KMeansParams,
//!       gpa: std.mem.Allocator,
//!   ) KMeansError!KMeansResult;
//!
//! ============================================================================
//! ALGORITHM (paper §3 references the standard Lloyd / k-means++):
//!
//!   1. Validate inputs.
//!        - If k == 0:                return error.KmeansBudgetTooSmall.
//!        - If n == 0:                return error.EmptyVectorSet.
//!        - If vectors.len != n*dim:  return error.DimMismatch.
//!
//!   2. Edge case: k >= n.
//!        Each vector becomes its own centroid (paper-gap; smallest correct
//!        choice — TAC phase 3's θ=39 floor normally prevents this, but PQ
//!        training with very small subspace blocks may hit it). Fill the
//!        remaining (k - n) centroids by repeating the last vector — never
//!        used at assignment time because we cap k_effective = n.
//!        WCSS = 0; iters_run = 0.
//!
//!   3. k-means++ seeding (Arthur & Vassilvitskii 2007).
//!        // paper §3 cites this as standard; we follow the canonical algorithm.
//!        a. Pick centroid_0 uniformly at random from {0..n-1} via
//!           rng.uniformInt(0, n).
//!        b. Maintain a buffer `min_dist_sq[i]` = squared L2 distance from
//!           vector i to the nearest centroid chosen so far.
//!           Initialise after centroid_0: min_dist_sq[i] = vec.l2sq(v_i, c_0).
//!        c. For c in 1..k-1:
//!             - Sample next centroid index via util.rng.weightedSample(
//!                   &rng, min_dist_sq);  // probability ∝ D²(x)
//!             - Copy v_idx into centroids[c*dim..].
//!             - Update min_dist_sq[i] = min(min_dist_sq[i],
//!                                           vec.l2sq(v_i, c_new))  for all i.
//!
//!        The weightedSample primitive (#2) does prefix-sum + binary search,
//!        so each pick is O(n + log n). Total seed cost: O(k·n·d) for the
//!        l2sq updates, dominated by the Lloyd loop below.
//!
//!   4. Lloyd's iterations.
//!        Allocate scratch buffers (single allocation each, freed via errdefer):
//!          - assignments_old: []u32, len n
//!          - cluster_sums:    []f32, len k*dim       — running mean accumulator
//!          - cluster_counts:  []u32, len k
//!          - centroid_dists:  []f32, len k           — per-vec scratch for argmin
//!
//!        prev_wcss = inf
//!        for iter in 0..max_iters:
//!            // Assignment step (paper §3, standard Lloyd):
//!            wcss = 0
//!            zero(cluster_sums); zero(cluster_counts)
//!            for i in 0..n:
//!                for c in 0..k:
//!                    centroid_dists[c] = vec.l2sq(
//!                        vectors[i*dim..(i+1)*dim],
//!                        centroids[c*dim..(c+1)*dim])
//!                a = vec.argmin(centroid_dists[0..k])
//!                assignments[i] = @intCast(a)
//!                wcss += centroid_dists[a]
//!                // Accumulate for mean update:
//!                add_inplace(cluster_sums[a*dim..(a+1)*dim],
//!                            vectors[i*dim..(i+1)*dim])
//!                cluster_counts[a] += 1
//!
//!            // Convergence check (relative WCSS delta).
//!            // First iter: prev_wcss == inf, never converges → continue.
//!            if iter > 0 and (prev_wcss - wcss) / prev_wcss < tol:
//!                iters_run = iter + 1; break
//!
//!            // Update step:
//!            for c in 0..k:
//!                if cluster_counts[c] > 0:
//!                    scale = 1 / @as(f32, @floatFromInt(cluster_counts[c]))
//!                    scale_inplace(cluster_sums[c*dim..(c+1)*dim], scale)
//!                    @memcpy(centroids[c*dim..(c+1)*dim],
//!                            cluster_sums[c*dim..(c+1)*dim])
//!                else:
//!                    // Empty cluster — re-seed with farthest vector from any centroid.
//!                    // paper-gap: paper just uses Lloyd's; we pick the canonical
//!                    // "farthest point" rule for determinism.
//!                    far_idx = argmax over i of (min over c' of l2sq(v_i, c_c'))
//!                    @memcpy(centroids[c*dim..(c+1)*dim],
//!                            vectors[far_idx*dim..(far_idx+1)*dim])
//!
//!            prev_wcss = wcss
//!            iters_run = iter + 1
//!
//!   5. Return KMeansResult with allocator-owned centroids/assignments.
//!      (cluster_sums/counts/dists are scratch — freed before return.)
//!
//! ============================================================================
//! INVARIANTS (asserted in tests):
//!   - WCSS is non-increasing across iterations until convergence.
//!     (Lloyd's algorithm is monotone — assignment step minimises WCSS given
//!      centroids; update step minimises WCSS given assignments.)
//!   - assignments[i] < k for all i.
//!   - Same seed → byte-equal centroids and assignments (determinism).
//!   - For 3 well-separated 2D Gaussian blobs (means (0,0), (10,0), (0,10),
//!     σ=0.3), recovered centroids match true means within 1e-2.
//!
//! ============================================================================
//! TESTS PLANNED:
//!   - test "fit recovers 3 well-separated 2D Gaussians within 1e-2"
//!   - test "fit is deterministic under fixed seed (byte-equal centroids)"
//!   - test "WCSS strictly non-increasing across iterations"
//!   - test "k > n returns trivial assignment of one centroid per vec"
//!   - test "k == 0 returns error.KmeansBudgetTooSmall"
//!   - test "n == 0 returns error.EmptyVectorSet"
//!   - test "empty cluster re-seeds with farthest point (1D pathological case)"
//!   - test "kmeans++ seeding is deterministic (centroid 0 from fixed seed)"
//!
//! ============================================================================
//! DEPENDENCIES (blocked on):
//!   - util/vec.zig:  dot, l2sq, argmin                                  (#1)
//!   - util/rng.zig:  Rng.init(seed), Rng.uniformInt, weightedSample     (#2)
//!   - util/alloc.zig: arena helpers (optional, may go without)          (#3)

const std = @import("std");

// Implementation lands once primitives #1, #2 are merged. See header for full
// algorithm and test plan. Keeping this file compileable so `zig build test`
// stays green for the rest of the team.

test "placeholder — implementation blocked on primitives #1, #2" {
    try std.testing.expect(true);
}
