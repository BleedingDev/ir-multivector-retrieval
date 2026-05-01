//! src/util/vec.zig — SIMD-vectorised float operations.
//!
//! Owner: primitives-engineer.
//! See plan 01-primitives-and-io.plan.md.
//!
//! Required surface (subject to extension):
//!   pub fn dot(a: []const f32, b: []const f32) f32;
//!   pub fn l2sq(a: []const f32, b: []const f32) f32;
//!   pub fn normalizeInPlace(v: []f32) void;
//!   pub fn argmin(distances: []const f32) usize;

const std = @import("std");

test "placeholder" {
    try std.testing.expect(true);
}
