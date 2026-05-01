//! src/index/pq.zig — Product Quantization (paper §4).
//!
//! Owner: indexer.
//! See plan 03-index-pq-hnsw-storage.plan.md.
//! Defaults (paper-strict): M=32 subspaces, b=8 bits → 32 bytes/vector.
//!
//! Required surface:
//!   pub const PQ = struct {
//!       dim: u32,
//!       sub_dim: u32,                       // dim / M
//!       codebooks: []f32,                   // M * 256 * sub_dim
//!   };
//!   pub fn train(residuals: []const f32, dim: u32, seed: u64, gpa: Allocator) !PQ;
//!   pub fn encode(self: PQ, residual: []const f32, out: *[constants.PQ_M]u8) void;
//!   pub fn decode(self: PQ, codes: *const [constants.PQ_M]u8, out: []f32) void;
//!   pub fn buildDistanceTable(self: PQ, query_tokens: []const f32, n_q: u32, out: []f32) void;

const std = @import("std");

test "placeholder" {
    try std.testing.expect(true);
}
