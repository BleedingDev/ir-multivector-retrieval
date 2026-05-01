//! src/util/rng.zig — deterministic RNG + weighted sampling.
//!
//! Owner: primitives-engineer.
//! Used by k-means++ initialisation in the TAC clusterer (paper §3, Phase 4)
//! and by any other randomised step (e.g. PQ centroid init in §4).
//!
//! Determinism is contractual: every public entry point takes a `seed: u64`
//! at construction time. Two `Rng` instances seeded identically produce
//! identical streams. We pin to `std.Random.DefaultPrng` (Xoshiro256++) so
//! the stream is stable across builds for a fixed Zig stdlib.

const std = @import("std");

pub const RngError = error{
    EmptyDistribution,
    NonFinite,
    AllZero,
};

/// Wrapper over std.Random.DefaultPrng. Pass by pointer so the internal
/// state mutates in place across draws.
pub const Rng = struct {
    inner: std.Random.DefaultPrng,

    pub fn init(seed: u64) Rng {
        return .{ .inner = std.Random.DefaultPrng.init(seed) };
    }

    pub fn random(self: *Rng) std.Random {
        return self.inner.random();
    }

    /// Uniform integer in `[0, n)`. Panics if `n == 0` — callers should
    /// validate they have a non-empty range.
    pub fn nextIndex(self: *Rng, n: usize) usize {
        std.debug.assert(n > 0);
        return self.random().intRangeLessThan(usize, 0, n);
    }

    /// Uniform `f32` in `[0, 1)`.
    pub fn nextFloat(self: *Rng) f32 {
        return self.random().float(f32);
    }
};

/// Sample an index in `[0, weights.len)` proportional to `weights[i]`.
/// Linear-scan inverse-CDF — fine for the inner loop of k-means++ where
/// `weights.len = N` (number of points) and we draw `κ` times. No allocation.
///
/// Returns:
///   - `error.EmptyDistribution` if `weights.len == 0`
///   - `error.NonFinite`         if any weight is NaN or negative
///   - `error.AllZero`           if every weight is exactly zero
///
/// paper §3.4, k-means++ init: distance² to nearest existing centroid is
/// the weight for the next centroid, ensuring spread.
pub fn weightedSample(rng: *Rng, weights: []const f32) RngError!usize {
    if (weights.len == 0) return error.EmptyDistribution;

    var total: f64 = 0.0;
    for (weights) |w| {
        if (!std.math.isFinite(w) or w < 0.0) return error.NonFinite;
        total += @floatCast(w);
    }
    if (total <= 0.0) return error.AllZero;

    // Draw uniformly in [0, total). Stay in f64 to avoid f32 quantisation
    // pinning the result to the first nonzero weight when total is large.
    const u: f64 = rng.random().float(f64) * total;
    var cum: f64 = 0.0;
    var i: usize = 0;
    while (i < weights.len) : (i += 1) {
        cum += @as(f64, weights[i]);
        if (u < cum) return i;
    }
    // Defensive fallback for the rare floating-point case where rounding
    // makes `u >= sum_of_all_weights`. Return the last positive-weight
    // index so we never report an index with weight 0.
    var last_positive: usize = weights.len - 1;
    while (last_positive > 0 and weights[last_positive] == 0.0) : (last_positive -= 1) {}
    return last_positive;
}

/// Fisher–Yates shuffle in place. Used wherever clustering needs an
/// unbiased random ordering (e.g. tie-breaking in centroid assignment).
pub fn shuffle(rng: *Rng, comptime T: type, items: []T) void {
    if (items.len < 2) return;
    var i: usize = items.len;
    while (i > 1) {
        i -= 1;
        const j = rng.random().intRangeAtMost(usize, 0, i);
        const tmp = items[i];
        items[i] = items[j];
        items[j] = tmp;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "Rng: same seed → same stream" {
    var a = Rng.init(42);
    var b = Rng.init(42);
    for (0..32) |_| {
        try testing.expectEqual(a.nextIndex(1000), b.nextIndex(1000));
    }
}

test "Rng: different seeds → different streams (with very high probability)" {
    var a = Rng.init(1);
    var b = Rng.init(2);
    var any_diff = false;
    for (0..16) |_| {
        if (a.nextIndex(1_000_000) != b.nextIndex(1_000_000)) any_diff = true;
    }
    try testing.expect(any_diff);
}

test "Rng.nextFloat: in [0,1)" {
    var r = Rng.init(7);
    for (0..256) |_| {
        const x = r.nextFloat();
        try testing.expect(x >= 0.0);
        try testing.expect(x < 1.0);
    }
}

test "weightedSample: deterministic with fixed seed" {
    const w = [_]f32{ 0.1, 0.2, 0.3, 0.4 };
    var r1 = Rng.init(123);
    var r2 = Rng.init(123);
    for (0..32) |_| {
        try testing.expectEqual(
            try weightedSample(&r1, &w),
            try weightedSample(&r2, &w),
        );
    }
}

test "weightedSample: only nonzero weights are ever picked" {
    const w = [_]f32{ 0.0, 1.0, 0.0, 0.0, 2.0 };
    var r = Rng.init(99);
    for (0..256) |_| {
        const idx = try weightedSample(&r, &w);
        try testing.expect(idx == 1 or idx == 4);
    }
}

test "weightedSample: empirical frequencies match weights" {
    // Weights [0.1, 0.3, 0.6] over 10000 draws — expect ~1000/3000/6000.
    const w = [_]f32{ 0.1, 0.3, 0.6 };
    var r = Rng.init(2026);
    var counts = [_]usize{ 0, 0, 0 };
    const n: usize = 10_000;
    for (0..n) |_| {
        const idx = try weightedSample(&r, &w);
        counts[idx] += 1;
    }
    // Generous tolerance — RNG-driven empirical test, not a goldenization.
    try testing.expect(counts[0] > 800 and counts[0] < 1200);
    try testing.expect(counts[1] > 2700 and counts[1] < 3300);
    try testing.expect(counts[2] > 5600 and counts[2] < 6400);
}

test "weightedSample: empty distribution" {
    var r = Rng.init(0);
    const w = [_]f32{};
    try testing.expectError(error.EmptyDistribution, weightedSample(&r, &w));
}

test "weightedSample: all-zero distribution" {
    var r = Rng.init(0);
    const w = [_]f32{ 0.0, 0.0, 0.0 };
    try testing.expectError(error.AllZero, weightedSample(&r, &w));
}

test "weightedSample: negative weight rejected" {
    var r = Rng.init(0);
    const w = [_]f32{ 0.5, -0.1, 0.3 };
    try testing.expectError(error.NonFinite, weightedSample(&r, &w));
}

test "weightedSample: NaN rejected" {
    var r = Rng.init(0);
    const w = [_]f32{ 0.5, std.math.nan(f32), 0.3 };
    try testing.expectError(error.NonFinite, weightedSample(&r, &w));
}

test "shuffle: permutes (likely) and is deterministic" {
    var a: [8]u32 = .{ 0, 1, 2, 3, 4, 5, 6, 7 };
    var b: [8]u32 = .{ 0, 1, 2, 3, 4, 5, 6, 7 };

    var r1 = Rng.init(31);
    var r2 = Rng.init(31);
    shuffle(&r1, u32, &a);
    shuffle(&r2, u32, &b);

    // Same seed → same permutation.
    try testing.expectEqualSlices(u32, &a, &b);

    // It's still a permutation: every element 0..8 appears exactly once.
    var seen = [_]bool{false} ** 8;
    for (a) |x| {
        try testing.expect(x < 8);
        try testing.expect(!seen[x]);
        seen[x] = true;
    }
}
