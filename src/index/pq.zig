//! src/index/pq.zig — Product Quantization (paper §4 + §5.3).
//!
//! Owner: indexer. Plan: 03-index-pq-hnsw-storage.plan.md.
//!
//! Paper-strict defaults (constants.zig): M=32 subspaces, b=8 bits → 32 bytes
//! per encoded vector. With dim=128 (ColBERTv2), sub_dim = 4.
//!
//! "Homogeneous compression" (paper §4): residuals are L2-normalised before
//! PQ training and encoding. Their original norms are persisted separately
//! by the storage layer (see `storage.zig`'s `norms_off` section) and
//! re-applied at decode time. This module operates only on already-
//! normalised residuals — callers do the normalisation upstream.
//!
//! Layout (row-major):
//!   codebook[m][c][k] = codebooks[((m * 256) + c) * sub_dim + k]
//!
//! Distance table layout (paper §5.3) — three-level for cache locality:
//!   table[m][c][i] = ⟨ q_i_subspace_m , codebook[m][c] ⟩
//!   linear        = ((m * 256) + c) * n_q + i
//! Refine reads `table[base..][0..n_q]` contiguously per (m, c) hit.
//!
//! paper-gap §5.3: micro-block alignment for SIMD. We do NOT pad n_q here
//! — alignment is a storage-layer concern. If a SIMD pass needs aligned
//! micro-blocks later, allocate `ceil(n_q/lane)*lane` and zero the tail.

const std = @import("std");
const Allocator = std.mem.Allocator;
const constants = @import("../constants.zig");
const vec = @import("../util/vec.zig");
const kmeans = @import("../tac/kmeans.zig");

pub const PqError = error{
    DimNotDivisibleByM,
    DimMismatch,
    EmptyCorpus,
    OutBufferWrongSize,
    NQMismatch,
} || Allocator.Error || vec.VecError || kmeans.KMeansError || std.Thread.SpawnError;

/// Trained PQ codebooks. Owner of `codebooks`; call `deinit`.
pub const PQ = struct {
    dim: u32,
    sub_dim: u32,
    /// Length = `constants.PQ_M * constants.PQ_CENTROIDS * sub_dim`.
    codebooks: []f32,

    pub fn deinit(self: *PQ, gpa: Allocator) void {
        gpa.free(self.codebooks);
        self.* = undefined;
    }

    pub inline fn codebookEntry(self: *const PQ, m: u32, c: u32) []const f32 {
        const base = ((@as(usize, m) * @as(usize, constants.PQ_CENTROIDS)) +
            @as(usize, c)) * @as(usize, self.sub_dim);
        return self.codebooks[base .. base + @as(usize, self.sub_dim)];
    }

    /// Encode a single residual into M = 32 codes (one byte per subspace).
    pub fn encode(self: *const PQ, residual: []const f32, out: *[constants.PQ_M]u8) PqError!void {
        if (residual.len != @as(usize, self.dim)) return error.DimMismatch;
        var m: u32 = 0;
        while (m < constants.PQ_M) : (m += 1) {
            const sub = residual[m * self.sub_dim ..][0..self.sub_dim];
            var best_c: u32 = 0;
            var best_d: f32 = std.math.inf(f32);
            var c: u32 = 0;
            while (c < constants.PQ_CENTROIDS) : (c += 1) {
                const d = vec.l2sq(sub, self.codebookEntry(m, c)) catch unreachable;
                if (d < best_d) {
                    best_d = d;
                    best_c = c;
                }
            }
            out[m] = @intCast(best_c);
        }
    }

    /// Decode M codes back into a `dim`-length residual approximation.
    pub fn decode(self: *const PQ, codes: *const [constants.PQ_M]u8, out: []f32) PqError!void {
        if (out.len != @as(usize, self.dim)) return error.DimMismatch;
        var m: u32 = 0;
        while (m < constants.PQ_M) : (m += 1) {
            const code: u8 = codes[m];
            const src = self.codebookEntry(m, code);
            const dst = out[m * self.sub_dim ..][0..self.sub_dim];
            @memcpy(dst, src);
        }
    }

    /// Build a per-query distance table. `out` length must be
    /// `PQ_M * PQ_CENTROIDS * n_q` floats. Layout: [M][256][n_q].
    pub fn buildDistanceTable(
        self: *const PQ,
        query_tokens: []const f32,
        n_q: u32,
        out: []f32,
    ) PqError!void {
        if (n_q == 0) return error.NQMismatch;
        if (query_tokens.len != @as(usize, n_q) * @as(usize, self.dim))
            return error.NQMismatch;
        const expected_out: usize =
            @as(usize, constants.PQ_M) *
            @as(usize, constants.PQ_CENTROIDS) *
            @as(usize, n_q);
        if (out.len != expected_out) return error.OutBufferWrongSize;

        var m: u32 = 0;
        while (m < constants.PQ_M) : (m += 1) {
            var c: u32 = 0;
            while (c < constants.PQ_CENTROIDS) : (c += 1) {
                const code_vec = self.codebookEntry(m, c);
                const base = ((@as(usize, m) * @as(usize, constants.PQ_CENTROIDS)) +
                    @as(usize, c)) * @as(usize, n_q);
                var i: u32 = 0;
                while (i < n_q) : (i += 1) {
                    const q_sub = query_tokens[
                        @as(usize, i) * @as(usize, self.dim) + m * self.sub_dim ..
                    ][0..self.sub_dim];
                    out[base + @as(usize, i)] = vec.dot(q_sub, code_vec) catch unreachable;
                }
            }
        }
    }
};

/// Linear index into a distance table laid out [M][256][n_q].
pub inline fn tableIndex(n_q: u32, m: u32, code: u8, i: u32) usize {
    return ((@as(usize, m) * @as(usize, constants.PQ_CENTROIDS)) + @as(usize, code)) *
        @as(usize, n_q) + @as(usize, i);
}

/// Read a table entry. Inlined for the hot refine loop.
pub inline fn lookup(table: []const f32, n_q: u32, m: u32, code: u8, i: u32) f32 {
    return table[tableIndex(n_q, m, code, i)];
}

/// Train M codebooks from a flat row-major matrix of normalised residuals.
///
/// Each subspace runs independent k-means with `k = 256`. Per-subspace seeds
/// are derived from the master `seed` so a single change to `seed` shifts
/// every subspace identically (deterministic across runs).
///
/// paper-gap: PQ training corpus sampling rule. We train on the full
/// `residuals` slice the caller passes — the storage builder is responsible
/// for sampling if the corpus is too large. Documented at the call site in
/// `storage.zig`'s build pipeline.
/// Train PQ codebooks with M=32 independent subspace k-means. Setting
/// `n_threads > 1` parallelises the M subspaces across worker threads
/// (paper §4 + §5.3 layout assumes per-subspace independence). Subspace
/// k-means is the dominant cost on large corpora; this is the
/// highest-leverage parallelism in the build pipeline.
pub fn train(
    residuals: []const f32,
    dim: u32,
    seed: u64,
    n_threads: u32,
    gpa: Allocator,
) PqError!PQ {
    if (dim == 0) return error.DimMismatch;
    if (dim % constants.PQ_M != 0) return error.DimNotDivisibleByM;
    if (residuals.len == 0) return error.EmptyCorpus;
    if (residuals.len % @as(usize, dim) != 0) return error.DimMismatch;

    const sub_dim: u32 = dim / constants.PQ_M;
    const n_residuals: usize = residuals.len / @as(usize, dim);

    const total_codebook_floats: usize =
        @as(usize, constants.PQ_M) *
        @as(usize, constants.PQ_CENTROIDS) *
        @as(usize, sub_dim);
    const codebooks = try gpa.alloc(f32, total_codebook_floats);
    errdefer gpa.free(codebooks);

    if (n_threads <= 1) {
        // ---- Serial across subspaces; within-subspace parallelism
        // (plan-07) — we pass `n_threads` straight through so a `--threads 1`
        // build still benefits from kmeans-side parallelism if it is bumped
        // upstream. Today this is the same 1; preserved for symmetry.
        const sub_buf = try gpa.alloc(f32, n_residuals * @as(usize, sub_dim));
        defer gpa.free(sub_buf);

        var m: u32 = 0;
        while (m < constants.PQ_M) : (m += 1) {
            try trainOneSubspace(residuals, dim, sub_dim, n_residuals, m, seed, n_threads, sub_buf, codebooks, gpa);
        }
    } else {
        // ---- Parallel: chunk M=32 across workers. ----
        const ChunkCtx = struct {
            residuals: []const f32,
            dim: u32,
            sub_dim: u32,
            n_residuals: usize,
            seed: u64,
            kmeans_n_threads: u32,
            codebooks: []f32,
            gpa: Allocator,
            err_out: ?PqError,
            lo: u32,
            hi: u32,
        };
        const ChunkRunner = struct {
            fn run(ctx: *ChunkCtx) void {
                const sub_buf = ctx.gpa.alloc(f32, ctx.n_residuals * @as(usize, ctx.sub_dim)) catch |err| {
                    ctx.err_out = err;
                    return;
                };
                defer ctx.gpa.free(sub_buf);
                var m = ctx.lo;
                while (m < ctx.hi) : (m += 1) {
                    trainOneSubspace(ctx.residuals, ctx.dim, ctx.sub_dim, ctx.n_residuals, m, ctx.seed, ctx.kmeans_n_threads, sub_buf, ctx.codebooks, ctx.gpa) catch |err| {
                        ctx.err_out = err;
                        return;
                    };
                }
            }
        };

        const real_threads = @min(n_threads, constants.PQ_M);
        const ctxs = try gpa.alloc(ChunkCtx, real_threads);
        defer gpa.free(ctxs);
        const threads = try gpa.alloc(std.Thread, real_threads);
        defer gpa.free(threads);

        // Within-subspace parallelism (plan-07): when subspace-level
        // parallelism already saturates the cores (n_threads <= real_threads
        // = min(n_threads, PQ_M=32)) further inner workers only oversubscribe
        // and lose to scheduling overhead — measured 113s → 121s at inner=10
        // and 113s → 135s at inner=2 on M=32 / outer=10 / 10-core Apple Si.
        // Reserve within-subspace parallelism for the case where the user's
        // total thread budget exceeds M (n_threads > 32) — there's leftover
        // budget to soak up inside each subspace's argmin loop.
        const kmeans_inner: u32 = blk: {
            if (n_threads <= real_threads) break :blk 1;
            break :blk @max(1, n_threads / real_threads);
        };

        const chunk = (constants.PQ_M + real_threads - 1) / real_threads;
        for (0..real_threads) |t| {
            const lo: u32 = @intCast(t * chunk);
            const hi: u32 = @min(@as(u32, @intCast((t + 1) * chunk)), constants.PQ_M);
            ctxs[t] = .{
                .residuals = residuals,
                .dim = dim,
                .sub_dim = sub_dim,
                .n_residuals = n_residuals,
                .seed = seed,
                .kmeans_n_threads = kmeans_inner,
                .codebooks = codebooks,
                .gpa = gpa,
                .err_out = null,
                .lo = lo,
                .hi = hi,
            };
            threads[t] = try std.Thread.spawn(.{}, ChunkRunner.run, .{&ctxs[t]});
        }
        for (threads) |th| th.join();
        for (ctxs) |c| if (c.err_out) |err| return err;
    }

    return .{
        .dim = dim,
        .sub_dim = sub_dim,
        .codebooks = codebooks,
    };
}

/// Train one PQ subspace's codebook. Materialises the m-th sub_dim columns
/// of every residual into `sub_buf`, runs k=256 k-means, copies centroids
/// into `codebooks[m*256*sub_dim..]`. Pure for distinct `m`.
///
/// `kmeans_n_threads` controls within-subspace parallelism (plan-07). At 1
/// the kmeans loop is byte-identical to the pre-parallel path; at >=2 the
/// argmin-per-vector step is sliced across worker threads but produces
/// byte-equal centroids across thread counts (paper-strict reproducibility).
fn trainOneSubspace(
    residuals: []const f32,
    dim: u32,
    sub_dim: u32,
    n_residuals: usize,
    m: u32,
    seed: u64,
    kmeans_n_threads: u32,
    sub_buf: []f32,
    codebooks: []f32,
    gpa: Allocator,
) PqError!void {
    var i: usize = 0;
    while (i < n_residuals) : (i += 1) {
        const src = residuals[i * @as(usize, dim) + m * sub_dim ..][0..sub_dim];
        const dst = sub_buf[i * @as(usize, sub_dim) ..][0..sub_dim];
        @memcpy(dst, src);
    }
    var res = try kmeans.fit(sub_buf, sub_dim, .{
        .k = constants.PQ_CENTROIDS,
        .max_iters = 25,
        .tol = 1e-4,
        .seed = seed +% @as(u64, m),
        .n_threads = kmeans_n_threads,
    }, gpa);
    defer res.deinit(gpa);

    const codebook_base: usize =
        @as(usize, m) *
        @as(usize, constants.PQ_CENTROIDS) *
        @as(usize, sub_dim);
    @memcpy(
        codebooks[codebook_base .. codebook_base + res.centroids.len],
        res.centroids,
    );
}

// ---------------------------------------------------------------------------
// TESTS
// ---------------------------------------------------------------------------

const testing = std.testing;

test "tableIndex matches the [M][256][n_q] layout" {
    try testing.expectEqual(@as(usize, 0), tableIndex(8, 0, 0, 0));
    try testing.expectEqual(@as(usize, 7), tableIndex(8, 0, 0, 7));
    try testing.expectEqual(@as(usize, 8), tableIndex(8, 0, 1, 0));
    try testing.expectEqual(@as(usize, 256 * 8), tableIndex(8, 1, 0, 0));
}

test "PQ defaults match paper §4" {
    try testing.expectEqual(@as(u32, 32), constants.PQ_M);
    try testing.expectEqual(@as(u32, 8), constants.PQ_BITS);
    try testing.expectEqual(@as(u32, 256), constants.PQ_CENTROIDS);
}

test "train: dim must be divisible by PQ_M" {
    const a = std.testing.allocator;
    // dim=33 is not divisible by 32 → error.
    const v = [_]f32{0.0} ** 33;
    const r = train(&v, 33, 1, 1, a);
    try testing.expectError(error.DimNotDivisibleByM, r);
}

test "train: empty corpus returns error" {
    const a = std.testing.allocator;
    const v = [_]f32{};
    const r = train(&v, 32, 1, 1, a);
    try testing.expectError(error.EmptyCorpus, r);
}

test "train + decode + encode round-trip on synthetic blobs" {
    const a = std.testing.allocator;
    // dim=32 → sub_dim=1 (the trivial case where each subspace has a
    // single coordinate). 256 distinct values in each subspace exactly
    // → reconstruction is *exact* (codebook contains every value).
    const dim: u32 = 32;
    const n: u32 = 300;
    const buf = try a.alloc(f32, @as(usize, n) * @as(usize, dim));
    defer a.free(buf);
    var prng = std.Random.DefaultPrng.init(0x52017);
    for (buf) |*x| x.* = prng.random().floatNorm(f32);
    // Normalise to mimic post-residual unit-norm vectors.
    try vec.normalizeRowsInPlace(buf, dim);

    var pq = try train(buf, dim, 17, 1, a);
    defer pq.deinit(a);

    try testing.expectEqual(@as(u32, 32), pq.dim);
    try testing.expectEqual(@as(u32, 1), pq.sub_dim);
    try testing.expectEqual(
        @as(usize, 32) * 256 * 1,
        pq.codebooks.len,
    );

    // Round-trip: encode → decode is idempotent on the codeword. Re-encoding
    // the decoded vector must produce the same code (decode lands you back
    // on the exact codebook entry, so its closest centroid is itself).
    const residual = try a.alloc(f32, dim);
    defer a.free(residual);
    @memcpy(residual, buf[0..dim]);

    var codes_a: [constants.PQ_M]u8 = undefined;
    try pq.encode(residual, &codes_a);

    const decoded = try a.alloc(f32, dim);
    defer a.free(decoded);
    try pq.decode(&codes_a, decoded);

    var codes_b: [constants.PQ_M]u8 = undefined;
    try pq.encode(decoded, &codes_b);
    try testing.expectEqualSlices(u8, &codes_a, &codes_b);
}

test "train: reconstruction MSE bounded on synthetic Gaussians" {
    const a = std.testing.allocator;
    const dim: u32 = 32;
    const n: u32 = 1024;
    const buf = try a.alloc(f32, @as(usize, n) * @as(usize, dim));
    defer a.free(buf);
    var prng = std.Random.DefaultPrng.init(0xC0DE);
    for (buf) |*x| x.* = prng.random().floatNorm(f32) * 0.1;

    var pq = try train(buf, dim, 7, 1, a);
    defer pq.deinit(a);

    var codes: [constants.PQ_M]u8 = undefined;
    const decoded = try a.alloc(f32, dim);
    defer a.free(decoded);

    var total_mse: f64 = 0.0;
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        const v = buf[@as(usize, i) * dim .. (@as(usize, i) + 1) * dim];
        try pq.encode(v, &codes);
        try pq.decode(&codes, decoded);
        const sq = try vec.l2sq(v, decoded);
        total_mse += @as(f64, sq) / @as(f64, dim);
    }
    const avg_mse = total_mse / @as(f64, n);
    // sub_dim=1, 256 codes covering N(0, 0.01) — quantisation error is
    // tiny. Expect mean per-dim MSE well below 1e-3.
    try testing.expect(avg_mse < 1e-3);
}

test "train: deterministic — same seed → byte-equal codebooks" {
    const a = std.testing.allocator;
    const dim: u32 = 32;
    const n: u32 = 300;
    const buf = try a.alloc(f32, @as(usize, n) * @as(usize, dim));
    defer a.free(buf);
    var prng = std.Random.DefaultPrng.init(99);
    for (buf) |*x| x.* = prng.random().floatNorm(f32);

    var p1 = try train(buf, dim, 1234, 1, a);
    defer p1.deinit(a);
    var p2 = try train(buf, dim, 1234, 1, a);
    defer p2.deinit(a);
    try testing.expectEqualSlices(f32, p1.codebooks, p2.codebooks);
}

test "buildDistanceTable: matches direct dot product" {
    const a = std.testing.allocator;
    const dim: u32 = 32;
    const n_q: u32 = 4;
    const n_train: u32 = 256;

    const buf = try a.alloc(f32, @as(usize, n_train) * @as(usize, dim));
    defer a.free(buf);
    var prng = std.Random.DefaultPrng.init(11);
    for (buf) |*x| x.* = prng.random().floatNorm(f32);
    try vec.normalizeRowsInPlace(buf, dim);

    var pq = try train(buf, dim, 1, 1, a);
    defer pq.deinit(a);

    const queries = try a.alloc(f32, @as(usize, n_q) * @as(usize, dim));
    defer a.free(queries);
    for (queries) |*x| x.* = prng.random().floatNorm(f32);
    try vec.normalizeRowsInPlace(queries, dim);

    const out = try a.alloc(
        f32,
        @as(usize, constants.PQ_M) * @as(usize, constants.PQ_CENTROIDS) * @as(usize, n_q),
    );
    defer a.free(out);

    try pq.buildDistanceTable(queries, n_q, out);

    // Spot-check: pick (m, c, i) and recompute via dot product.
    const m: u32 = 5;
    const c: u8 = 17;
    const i: u32 = 2;
    const expected = try vec.dot(
        queries[@as(usize, i) * dim + m * pq.sub_dim ..][0..pq.sub_dim],
        pq.codebookEntry(m, c),
    );
    const got = lookup(out, n_q, m, c, i);
    try testing.expectApproxEqAbs(expected, got, 1e-6);

    // Another spot.
    const m_b: u32 = 31;
    const c_b: u8 = 0;
    const i_b: u32 = 3;
    const expected_b = try vec.dot(
        queries[@as(usize, i_b) * dim + m_b * pq.sub_dim ..][0..pq.sub_dim],
        pq.codebookEntry(m_b, c_b),
    );
    try testing.expectApproxEqAbs(expected_b, lookup(out, n_q, m_b, c_b, i_b), 1e-6);
}

test "buildDistanceTable: bad output length is an error" {
    const a = std.testing.allocator;
    const dim: u32 = 32;
    const buf = try a.alloc(f32, 200 * @as(usize, dim));
    defer a.free(buf);
    for (buf, 0..) |*x, idx| x.* = @as(f32, @floatFromInt(idx)) * 0.001;
    var pq = try train(buf, dim, 0, 1, a);
    defer pq.deinit(a);
    const n_q: u32 = 2;
    const queries = try a.alloc(f32, @as(usize, n_q) * @as(usize, dim));
    defer a.free(queries);
    @memset(queries, 0.0);
    var bogus_out: [4]f32 = undefined;
    try testing.expectError(
        error.OutBufferWrongSize,
        pq.buildDistanceTable(queries, n_q, bogus_out[0..]),
    );
}
