//! src/index/pq.zig — Product Quantization (paper §4 + §5.3).
//!
//! Owner: indexer.
//! See plan 03-index-pq-hnsw-storage.plan.md.
//! Defaults (paper-strict): M=32 subspaces, b=8 bits → 32 bytes/vector.
//!
//! Per paper §4 ("homogeneous compression"): residuals are L2-normalised
//! before PQ training and encoding; the discarded norms are kept in a
//! separate side array (lives in storage.zig, not here) and re-applied at
//! decode time. This module operates only on already-normalised residuals.
//!
//! Required surface (locked by task #11/#12):
//!   pub const PQ = struct {
//!       dim: u32,
//!       sub_dim: u32,                       // dim / M  (must divide cleanly)
//!       codebooks: []f32,                   // M * 256 * sub_dim, row-major:
//!                                           //   codebook[m][c][k] = codebooks[((m*256)+c)*sub_dim + k]
//!   };
//!   pub fn train(residuals: []const f32, dim: u32, seed: u64, gpa: Allocator) !PQ;
//!   pub fn encode(self: PQ, residual: []const f32, out: *[constants.PQ_M]u8) void;
//!   pub fn decode(self: PQ, codes: *const [constants.PQ_M]u8, out: []f32) void;
//!   pub fn buildDistanceTable(self: PQ, query_tokens: []const f32, n_q: u32, out: []f32) void;
//!   pub fn lookup(table: []const f32, n_q: u32, m: u32, code: u8, i: u32) f32;
//!
//! ---------------------------------------------------------------------------
//! DESIGN PSEUDOCODE (waiting on #6 kmeans.fit before implementing).
//! ---------------------------------------------------------------------------
//!
//! train(residuals, dim, seed, gpa):
//!   assert dim % PQ_M == 0;                    // required: sub_dim integer
//!   sub_dim = dim / PQ_M;                      // = 4 when dim=128, M=32
//!   n = residuals.len / dim;
//!   codebooks = gpa.alloc(f32, PQ_M * 256 * sub_dim);
//!   tmp_sub = gpa.alloc(f32, n * sub_dim);     // reused per subspace
//!   for m in 0..PQ_M:
//!       // Slice subspace m out of every residual (interleave → contiguous).
//!       for i in 0..n:
//!           memcpy(tmp_sub[i*sub_dim ..], residuals[i*dim + m*sub_dim ..], sub_dim);
//!       // K-means with k=256 (= 2^PQ_BITS), reuse src/tac/kmeans.zig (#6).
//!       res = kmeans.fit(tmp_sub, sub_dim, .{ .k=256, .seed=seed +% m, .max_iters=25 }, gpa);
//!       memcpy(codebooks[m*256*sub_dim ..], res.centroids, 256*sub_dim);
//!       free(res.centroids); free(res.assignments);
//!   free(tmp_sub);
//!   return PQ{ dim, sub_dim, codebooks };
//!
//! encode(self, residual, out):
//!   assert residual.len == self.dim;
//!   for m in 0..PQ_M:
//!       sub = residual[m*sub_dim .. (m+1)*sub_dim];
//!       best_c = 0; best_d = +inf;
//!       for c in 0..256:
//!           code = codebooks[m*256*sub_dim + c*sub_dim ..][0..sub_dim];
//!           d = l2sq(sub, code);                // util.vec.l2sq once #1 ships
//!           if d < best_d: best_d = d; best_c = c;
//!       out[m] = best_c;
//!
//! decode(self, codes, out):
//!   assert out.len == self.dim;
//!   for m in 0..PQ_M:
//!       c = codes[m];
//!       memcpy(out[m*sub_dim ..], codebooks[m*256*sub_dim + c*sub_dim ..], sub_dim);
//!
//! ---------------------------------------------------------------------------
//! Distance table (paper §5.3). Three-level layout, "up to 3.8×" speedup.
//!
//!   table[m][c][i]  ==  ⟨ q_i_subspace_m , codebook[m][c] ⟩
//!
//! Linear index:
//!   table_idx(m,c,i) = ((m * 256) + c) * n_q + i
//!   total bytes    = PQ_M * 256 * n_q * sizeof(f32)
//!
//! For n_q=32, M=32: 32 · 256 · 32 · 4 = 1 MiB — comfortably L2-resident.
//!
//! buildDistanceTable(self, query_tokens, n_q, out):
//!   assert query_tokens.len == n_q * self.dim;
//!   assert out.len == PQ_M * 256 * n_q;
//!   for m in 0..PQ_M:
//!       for c in 0..256:
//!           code_vec = codebooks[((m*256)+c)*sub_dim ..][0..sub_dim];
//!           base = ((m*256) + c) * n_q;
//!           for i in 0..n_q:
//!               q_sub = query_tokens[i*dim + m*sub_dim ..][0..sub_dim];
//!               out[base + i] = dot(q_sub, code_vec);
//!   // refine inner loop reads `out[base + 0 .. n_q]` contiguously per (m,c) hit.
//!
//! paper-gap §5.3: micro-block alignment for SIMD. We do NOT pad n_q; loop
//! over the exact n_q. If a SIMD pass later wants padded reads, that's a
//! storage-layer concern (allocate ceil(n_q / lane) * lane and zero-fill).
//! ---------------------------------------------------------------------------

const std = @import("std");
const Allocator = std.mem.Allocator;
const constants = @import("../constants.zig");

/// Trained PQ codebooks. Owner of the `codebooks` slice — call `deinit`.
pub const PQ = struct {
    dim: u32,
    sub_dim: u32,
    codebooks: []f32, // PQ_M * 256 * sub_dim

    pub fn deinit(self: *PQ, gpa: Allocator) void {
        gpa.free(self.codebooks);
        self.* = undefined;
    }
};

/// Linear index into a distance table laid out [M][256][n_q].
pub inline fn tableIndex(n_q: u32, m: u32, code: u8, i: u32) usize {
    return ((@as(usize, m) * 256) + @as(usize, code)) * @as(usize, n_q) + @as(usize, i);
}

test "tableIndex matches the [M][256][n_q] layout" {
    // (m=0, c=0, i=0) → 0
    try std.testing.expectEqual(@as(usize, 0), tableIndex(8, 0, 0, 0));
    // n_q=8: (m=0, c=0, i=7) → 7
    try std.testing.expectEqual(@as(usize, 7), tableIndex(8, 0, 0, 7));
    // (m=0, c=1, i=0) → 8
    try std.testing.expectEqual(@as(usize, 8), tableIndex(8, 0, 1, 0));
    // (m=1, c=0, i=0) → 256·8 = 2048
    try std.testing.expectEqual(@as(usize, 2048), tableIndex(8, 1, 0, 0));
}

test "PQ defaults match paper §4" {
    try std.testing.expectEqual(@as(u32, 32), constants.PQ_M);
    try std.testing.expectEqual(@as(u32, 8), constants.PQ_BITS);
    try std.testing.expectEqual(@as(u32, 256), constants.PQ_CENTROIDS);
}
