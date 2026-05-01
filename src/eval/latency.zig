//! src/eval/latency.zig — single-thread latency benchmarking.
//!
//! Owner: retriever.
//! See plan 04-retrieval-and-eval.plan.md.
//!
//! Required surface:
//!   pub const StageTimings = struct { gather_ns: u64, prune_ns: u64, refine_ns: u64 };
//!   pub fn timeQuery(...) StageTimings;
//!   pub fn report(timings: []const StageTimings) Report;

const std = @import("std");

test "placeholder" {
    try std.testing.expect(true);
}
