//! src/tac/tac.zig — Token-Aware Clustering (paper §3) four-phase pipeline.
//!
//! Owner: clusterer.
//! See plan 02-tac-clustering.plan.md.
//!
//! Required surface:
//!   pub const ClusteringParams = struct {
//!       kappa_total: u32,
//!       mu: u32 = constants.TAC_MU,
//!       tau: u32 = constants.TAC_TAU,
//!       epsilon: u32 = constants.TAC_EPSILON,
//!       theta: u32 = constants.TAC_THETA,
//!       seed: u64,
//!   };
//!   pub const ClusteringResult = struct {
//!       centroids: []const f32,        // kappa_total * dim
//!       assignments: []const u32,      // n_tokens, value in 0..kappa_total
//!       kappa_per_token: []const u32,  // n_distinct_tokens
//!   };
//!   pub fn cluster(td: TokenDump, p: ClusteringParams, gpa: Allocator) !ClusteringResult;

const std = @import("std");

test "placeholder" {
    try std.testing.expect(true);
}
