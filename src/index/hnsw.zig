//! src/index/hnsw.zig — HNSW over centroids (paper §4).
//!
//! Owner: indexer.
//! See plan 03-index-pq-hnsw-storage.plan.md.
//! Defaults (paper-strict): M_hnsw=32, efc=1500. Runtime ef_s = 1.5·κ_c.
//!
//! Required surface:
//!   pub const Hnsw = struct { ... };
//!   pub fn build(centroids: []const f32, dim: u32, seed: u64, gpa: Allocator) !Hnsw;
//!   pub fn search(self: *const Hnsw, query: []const f32, k: u32, ef: u32, out: []u32) u32;

const std = @import("std");

test "placeholder" {
    try std.testing.expect(true);
}
