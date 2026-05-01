//! src/eval/metrics.zig — IR metrics (MRR@10, Success@k).
//!
//! Owner: retriever.
//! See plan 04-retrieval-and-eval.plan.md.
//!
//! Required surface:
//!   pub fn mrrAt(ranking: []const u32, qrels: []const u32, k: u32) f32;
//!   pub fn successAt(ranking: []const u32, qrels: []const u32, k: u32) f32;

const std = @import("std");

test "placeholder" {
    try std.testing.expect(true);
}
