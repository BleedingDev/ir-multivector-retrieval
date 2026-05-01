//! src/tac/kmeans.zig — Lloyd's k-means with k-means++ init.
//!
//! Owner: clusterer.
//! See plan 02-tac-clustering.plan.md.
//!
//! Required surface:
//!   pub const KMeansParams = struct {
//!       k: u32,
//!       max_iters: u32 = 25,
//!       tol: f32 = 1e-4,
//!       seed: u64,
//!   };
//!   pub const KMeansResult = struct {
//!       centroids: []f32,       // k * dim, caller-owned
//!       assignments: []u32,     // n, caller-owned
//!       wcss: f32,
//!       iters_run: u32,
//!   };
//!   pub fn fit(vectors: []const f32, dim: u32, p: KMeansParams, gpa: Allocator) !KMeansResult;

const std = @import("std");

test "placeholder" {
    try std.testing.expect(true);
}
