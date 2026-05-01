//! src/util/rng.zig — deterministic RNG + weighted sampler.
//!
//! Owner: primitives-engineer.
//! See plan 01-primitives-and-io.plan.md.
//!
//! Required surface:
//!   pub const Rng = struct { ... };
//!   pub fn init(seed: u64) Rng;
//!   pub fn weightedSample(rng: *Rng, weights: []const f32) usize;  // for k-means++

const std = @import("std");

test "placeholder" {
    try std.testing.expect(true);
}
