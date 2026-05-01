//! src/io/synthetic_fixture.zig — deterministic synthetic token dump.
//!
//! Owner: primitives-engineer.
//! Used by every downstream team to integration-test against a small,
//! reproducible corpus without touching real ColBERTv2 weights. The dump
//! is *not* a binary: this file is a builder that produces an in-memory
//! `Fixture`, which can either be passed straight to consumers or written
//! out to disk via `io/token_dump.zig`'s `write`.
//!
//! The lane plan called this `tests/fixtures/synthetic_tokens.zig`, but
//! the lib_mod's import root is `src/`, so the file lives here and is
//! re-exported through `src/root.zig` as `io.synthetic_fixture`.
//!
//! The synthetic distribution mimics the paper's setup at small scale:
//!   - a vocabulary with a deliberately skewed token-frequency tail, so
//!     TAC tail handling (paper §3 Phase 1) actually has work to do;
//!   - per-token "themes" (each token id maps to a unit-norm direction
//!     plus small Gaussian noise), so within-token clustering is
//!     non-trivial but checkable;
//!   - L2-normalised vectors, matching the ColBERT convention.

const std = @import("std");
const rng = @import("../util/rng.zig");
const vec = @import("../util/vec.zig");
const token_dump = @import("token_dump.zig");

pub const Options = struct {
    seed: u64,
    n_docs: u64,
    dim: u32,
    /// Vocabulary size. The frequency distribution is roughly Zipfian
    /// (id 0 is most frequent) so a few tokens dominate.
    vocab_size: u32 = 32,
    /// Average tokens per document. Each doc draws Poisson(mean), with
    /// length floor 1 and ceiling 2*mean. Determinism is preserved.
    avg_doc_len: u32 = 16,
};

/// In-memory synthetic dump. All slices are allocator-owned by the caller.
pub const Fixture = struct {
    dim: u32,
    n_docs: u64,
    n_tokens: u64,
    doc_offsets: []u64,
    token_ids: []u32,
    vectors: []f32,

    pub fn deinit(self: *Fixture, allocator: std.mem.Allocator) void {
        allocator.free(self.doc_offsets);
        allocator.free(self.token_ids);
        allocator.free(self.vectors);
        self.* = undefined;
    }

    pub fn toBuild(self: Fixture) token_dump.DumpBuild {
        return .{
            .dim = self.dim,
            .doc_offsets = self.doc_offsets,
            .token_ids = self.token_ids,
            .vectors = self.vectors,
        };
    }
};

/// Build a fixture with the given options. Determinism: identical
/// `(seed, n_docs, dim, vocab_size, avg_doc_len)` produces identical bytes.
pub fn build(allocator: std.mem.Allocator, opts: Options) !Fixture {
    if (opts.dim == 0) return error.InvalidOptions;
    if (opts.n_docs == 0) return error.InvalidOptions;
    if (opts.vocab_size == 0) return error.InvalidOptions;
    if (opts.avg_doc_len == 0) return error.InvalidOptions;

    var r = rng.Rng.init(opts.seed);

    // Step 1 — per-doc lengths, building doc_offsets.
    var doc_offsets = try allocator.alloc(u64, opts.n_docs + 1);
    errdefer allocator.free(doc_offsets);
    doc_offsets[0] = 0;
    var total_tokens: u64 = 0;
    var d: u64 = 0;
    while (d < opts.n_docs) : (d += 1) {
        const min_len: u32 = 1;
        const max_len: u32 = opts.avg_doc_len * 2;
        const range = max_len - min_len + 1;
        const len: u64 = min_len + r.nextIndex(range);
        total_tokens += len;
        doc_offsets[d + 1] = total_tokens;
    }

    // Step 2 — Zipfian-ish token-id assignments. Use a precomputed CDF
    // with weight ∝ 1 / (rank + 1).
    const cdf = try allocator.alloc(f64, opts.vocab_size);
    defer allocator.free(cdf);
    {
        var sum: f64 = 0.0;
        var k: u32 = 0;
        while (k < opts.vocab_size) : (k += 1) {
            sum += 1.0 / @as(f64, @floatFromInt(k + 1));
            cdf[k] = sum;
        }
        // Normalise so cdf[last] == 1.
        const total = cdf[opts.vocab_size - 1];
        for (cdf) |*x| x.* /= total;
    }

    var token_ids = try allocator.alloc(u32, total_tokens);
    errdefer allocator.free(token_ids);
    for (0..total_tokens) |i| {
        const u: f64 = r.random().float(f64);
        var k: u32 = 0;
        while (k + 1 < opts.vocab_size and u > cdf[k]) : (k += 1) {}
        token_ids[i] = k;
    }

    // Step 3 — per-token "theme" directions on the unit sphere. Each
    // token id has its own deterministic direction, derived from a
    // child seed so the theme is stable across builds.
    const themes = try allocator.alloc(f32, @as(usize, opts.vocab_size) * opts.dim);
    defer allocator.free(themes);
    var theme_rng = rng.Rng.init(opts.seed ^ 0x9E37_79B1_7F4A_7C15);
    for (0..themes.len) |i| {
        themes[i] = (theme_rng.nextFloat() * 2.0) - 1.0; // uniform [-1, 1)
    }
    var k: u32 = 0;
    while (k < opts.vocab_size) : (k += 1) {
        try vec.normalizeInPlace(themes[k * opts.dim ..][0..opts.dim]);
    }

    // Step 4 — token vectors = theme + small Gaussian-ish noise, then
    // L2-renormalised. We approximate Gaussian with a 6-uniform sum
    // (CLT) — more than good enough for fixture data.
    var vectors = try allocator.alloc(f32, total_tokens * opts.dim);
    errdefer allocator.free(vectors);
    const noise_scale: f32 = 0.1;
    var i: usize = 0;
    while (i < total_tokens) : (i += 1) {
        const tid = token_ids[i];
        const theme = themes[tid * opts.dim ..][0..opts.dim];
        const out = vectors[i * opts.dim ..][0..opts.dim];
        for (out, 0..) |*o, j| {
            // sum-of-12 minus 6 ≈ standard normal
            var noise: f32 = 0.0;
            var s: u32 = 0;
            while (s < 12) : (s += 1) noise += r.nextFloat();
            noise -= 6.0;
            o.* = theme[j] + noise * noise_scale;
        }
        try vec.normalizeInPlace(out);
    }

    return .{
        .dim = opts.dim,
        .n_docs = opts.n_docs,
        .n_tokens = total_tokens,
        .doc_offsets = doc_offsets,
        .token_ids = token_ids,
        .vectors = vectors,
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "synthetic fixture: deterministic under fixed seed" {
    const allocator = testing.allocator;

    var a = try build(allocator, .{ .seed = 42, .n_docs = 4, .dim = 6, .vocab_size = 8 });
    defer a.deinit(allocator);
    var b = try build(allocator, .{ .seed = 42, .n_docs = 4, .dim = 6, .vocab_size = 8 });
    defer b.deinit(allocator);

    try testing.expectEqualSlices(u64, a.doc_offsets, b.doc_offsets);
    try testing.expectEqualSlices(u32, a.token_ids, b.token_ids);
    try testing.expectEqualSlices(f32, a.vectors, b.vectors);
}

test "synthetic fixture: vectors are unit-norm" {
    const allocator = testing.allocator;
    var fx = try build(allocator, .{ .seed = 1, .n_docs = 6, .dim = 16, .vocab_size = 12 });
    defer fx.deinit(allocator);

    var i: u64 = 0;
    while (i < fx.n_tokens) : (i += 1) {
        const v = fx.vectors[@intCast(i * fx.dim) ..][0..fx.dim];
        var sq: f32 = 0.0;
        for (v) |x| sq += x * x;
        try testing.expectApproxEqAbs(@as(f32, 1.0), sq, 1e-4);
    }
}

test "synthetic fixture: csr offsets are well-formed" {
    const allocator = testing.allocator;
    var fx = try build(allocator, .{ .seed = 5, .n_docs = 10, .dim = 4, .vocab_size = 6 });
    defer fx.deinit(allocator);

    try testing.expectEqual(@as(u64, 0), fx.doc_offsets[0]);
    try testing.expectEqual(fx.n_tokens, fx.doc_offsets[fx.doc_offsets.len - 1]);
    var prev: u64 = 0;
    for (fx.doc_offsets) |o| {
        try testing.expect(o >= prev);
        prev = o;
    }
}

test "synthetic fixture: token ids are skewed (id 0 dominates)" {
    const allocator = testing.allocator;
    var fx = try build(allocator, .{
        .seed = 11,
        .n_docs = 100,
        .dim = 4,
        .vocab_size = 16,
        .avg_doc_len = 32,
    });
    defer fx.deinit(allocator);

    var counts = [_]u64{0} ** 16;
    for (fx.token_ids) |tid| counts[tid] += 1;

    // Zipfian by construction: id 0 must outnumber id 15.
    try testing.expect(counts[0] > counts[15]);
    // And substantially — at vocab_size=16 the head/tail ratio is
    // 1/(1) vs 1/(16) = 16:1 in expectation. Generous tolerance.
    try testing.expect(counts[0] > counts[15] * 4);
}
