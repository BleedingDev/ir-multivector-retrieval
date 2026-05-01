//! src/index/inverted_list.zig — per-centroid inverted lists of doc IDs.
//!
//! Owner: indexer.
//! See plan 03-index-pq-hnsw-storage.plan.md.
//!
//! Per paper §4: L_j = { d : exists token in d assigned to centroid j }.
//! Document-level grain (NOT token-level). De-dup doc IDs at build time.

const std = @import("std");

test "placeholder" {
    try std.testing.expect(true);
}
