//! src/index/storage.zig — on-disk index format (paper §4 layout).
//!
//! Owner: indexer.
//! See plan 03-index-pq-hnsw-storage.plan.md.
//!
//! Per paper §4 doc layout:
//!   [c_1..c_{n_d}: u32  |  PQ_{1,1}..PQ_{n_d,M}: u8]
//!
//! On-disk: header (magic, version, dim, kappa, M, n_docs, …),
//!          centroids, HNSW graph, PQ codebooks, inverted lists, doc layouts.

const std = @import("std");

test "placeholder" {
    try std.testing.expect(true);
}
