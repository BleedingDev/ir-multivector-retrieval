//! src/retrieval/refine.zig — Refine phase (paper §5.3).
//!
//! Owner: retriever.
//! See plan 04-retrieval-and-eval.plan.md.
//!
//! For each surviving candidate document d:
//!   Pass 1: stream centroid IDs → per-(q_i, k) centroid contribution
//!           ⟨q_i, c_{c_k}⟩.
//!   Pass 2: stream PQ codes → residual contribution via the cache-optimised
//!           distance table laid out [M macro][256 code][n_q micro], scaled
//!           by the stored residual norm.
//! Output: exact MaxSim score for the (q, d) pair.
//!
//! Original token vector t_k = c_k + norm_k · r_k (paper §4 "homogeneous
//! compression": residuals normalised, norms saved). So the inner product
//! decomposes:
//!     ⟨q_i, t_k⟩ = ⟨q_i, c_k⟩ + norm_k · ⟨q_i, r_k⟩
//! and r_k is approximated by PQ decode → ⟨q_i, r_k⟩ ≈ Σ_m table[m][code][i].
//! Refine therefore computes:
//!     score_i(k, d) = ⟨q_i, c_{c_k}⟩
//!                   + norm_k · Σ_{m=0..M-1} table[m][PQ_{k,m}][i]
//! Per-token MaxSim:
//!     s_i(d) = max_{k=1..n_d} score_i(k, d)
//! Final:
//!     S(q, d) = Σ_{i=1..n_q} s_i(d)

const std = @import("std");
const Allocator = std.mem.Allocator;

const constants = @import("../constants.zig");
const vec = @import("../util/vec.zig");
const storage = @import("../index/storage.zig");
const pq_mod = @import("../index/pq.zig");
const gather = @import("gather.zig");

pub const RefineError = error{
    QueryShapeMismatch,
    TableSizeMismatch,
    DocOutOfRange,
    DocLayoutCorrupt,
} || vec.VecError;

/// Decoded view into a doc's payload. Slices alias `Index.doc_payload`.
const DocLayout = struct {
    n_d: u32,
    centroid_ids: []align(1) const u32,
    pq_codes: []const u8,
};

fn decodeDocLayout(layout_bytes: []const u8) RefineError!DocLayout {
    if (layout_bytes.len < 4) return error.DocLayoutCorrupt;
    const n_d = std.mem.readInt(u32, layout_bytes[0..4], .little);
    const centroid_bytes_len: usize = @as(usize, n_d) * @sizeOf(u32);
    const codes_len: usize = @as(usize, n_d) * @as(usize, constants.PQ_M);
    const expected_min: usize = 4 + centroid_bytes_len + codes_len;
    if (layout_bytes.len < expected_min) return error.DocLayoutCorrupt;
    const centroid_ids: []align(1) const u32 = std.mem.bytesAsSlice(
        u32,
        layout_bytes[4 .. 4 + centroid_bytes_len],
    );
    const pq_codes = layout_bytes[4 + centroid_bytes_len ..][0..codes_len];
    return .{ .n_d = n_d, .centroid_ids = centroid_ids, .pq_codes = pq_codes };
}

/// Compute exact MaxSim S(q, d) for a single candidate using the
/// pre-built per-query distance table (paper §5.3).
///
/// Args:
///   index            - parsed Index. Refine reads `centroids`, `pq`, the
///                      doc payload via `index.docLayout(d)`, and slices
///                      `residual_norms` via `index.docTokenRange(d)`.
///   distance_table   - PQ_M · 256 · n_q floats, built once per query via
///                      `pq.buildDistanceTable`.
///   query_tokens     - n_q · dim, row-major, L2-normalised. Used for the
///                      Pass-1 centroid term.
///   per_token_max_buf - caller-provided scratch of length ≥ n_q. Reused
///                      across all surviving candidates so refine itself
///                      doesn't allocate.
///
/// The simpler fused-(k,i) loop is implemented first; the cache-optimised
/// outer-m / inner-(k,i) restructuring (paper §5.3, "up to 3.8×") is a
/// later optimisation gated on latency-harness evidence.
pub fn refine(
    index: *const storage.Index,
    distance_table: []const f32,
    query_tokens: []const f32,
    n_q: u32,
    candidate: gather.Candidate,
    per_token_max_buf: []f32,
) RefineError!f32 {
    const dim: usize = @intCast(index.header.dim);
    if (n_q == 0 or query_tokens.len != @as(usize, n_q) * dim) {
        return error.QueryShapeMismatch;
    }
    const expected_table: usize =
        @as(usize, constants.PQ_M) * @as(usize, constants.PQ_CENTROIDS) * @as(usize, n_q);
    if (distance_table.len != expected_table) return error.TableSizeMismatch;
    if (per_token_max_buf.len < n_q) return error.QueryShapeMismatch;
    const n_docs: u32 = @intCast(index.header.n_docs);
    if (candidate.doc_id >= n_docs) return error.DocOutOfRange;

    const layout = try decodeDocLayout(index.docLayout(candidate.doc_id));
    const n_d = layout.n_d;
    const tok_range = index.docTokenRange(candidate.doc_id);
    const norms_lo: usize = @intCast(tok_range[0]);
    const norms_hi: usize = @intCast(tok_range[1]);
    if (norms_hi - norms_lo != n_d) return error.DocLayoutCorrupt;
    const norms = index.residual_norms[norms_lo..norms_hi];

    // Initialise per-token max to -inf so the first valid score wins.
    var i: u32 = 0;
    while (i < n_q) : (i += 1) per_token_max_buf[i] = -std.math.inf(f32);

    // Fused (k, i) loop — paper §5.3 Pass 1 + Pass 2 interleaved per token.
    var k: u32 = 0;
    while (k < n_d) : (k += 1) {
        const c_id: u32 = layout.centroid_ids[k];
        const c_vec = index.centroids[@as(usize, c_id) * dim ..][0..dim];
        const norm_k = norms[k];
        const code_base: usize = @as(usize, k) * @as(usize, constants.PQ_M);

        i = 0;
        while (i < n_q) : (i += 1) {
            const q_i = query_tokens[@as(usize, i) * dim ..][0..dim];
            const cent_term = try vec.dot(q_i, c_vec);

            // Residual term: ⟨q_i, r_k⟩ via PQ-decoded codes.
            var res_term: f32 = 0.0;
            var m: u32 = 0;
            while (m < constants.PQ_M) : (m += 1) {
                const code: u8 = layout.pq_codes[code_base + @as(usize, m)];
                res_term += pq_mod.lookup(distance_table, n_q, m, code, i);
            }

            const score = cent_term + norm_k * res_term;
            if (score > per_token_max_buf[i]) per_token_max_buf[i] = score;
        }
    }

    // Aggregate. n_d == 0 → no per-token max was written; return 0 so the
    // score sits below any non-empty doc.
    if (n_d == 0) return 0.0;
    var sum: f32 = 0.0;
    i = 0;
    while (i < n_q) : (i += 1) sum += per_token_max_buf[i];
    return sum;
}

// ---------------------------------------------------------------------------
// Tests — end-to-end against a synthetic Index built via storage.build.
// ---------------------------------------------------------------------------

const testing = std.testing;
const synthetic_fixture = @import("../io/synthetic_fixture.zig");
const token_dump = @import("../io/token_dump.zig");

test "constants visible to refine module" {
    try testing.expectEqual(@as(u32, 32), constants.PQ_M);
    try testing.expectEqual(@as(usize, 8), @sizeOf(gather.Candidate));
}

fn buildTestIndex(
    gpa: Allocator,
    seed: u64,
    n_docs: u64,
    kappa: u32,
) !struct { index: storage.Index, image: []align(8) u8, fx_bytes: []align(8) u8 } {
    var fx = try synthetic_fixture.build(gpa, .{
        .seed = seed,
        .n_docs = n_docs,
        .dim = 32,
        .vocab_size = 16,
        .avg_doc_len = 8,
    });
    defer fx.deinit(gpa);

    const fx_bytes = try token_dump.writeAlloc(gpa, fx.toBuild());
    errdefer gpa.free(fx_bytes);
    const td = try token_dump.parseBytes(fx_bytes);

    var image = try storage.build(&td, .{
        .kappa_total = kappa,
        .seed = seed +% 1,
        .mu = 4,
        .tau = 8,
        .epsilon = 1,
        .theta = 1,
        .hnsw = .{ .ef_construction = 32, .m = 4 },
    }, gpa);
    errdefer image.deinit(gpa);

    const idx = try storage.parse(image.bytes, gpa);
    return .{ .index = idx, .image = image.bytes, .fx_bytes = fx_bytes };
}

test "refine: exact MaxSim agrees with brute-force baseline (within 1e-3)" {
    const gpa = testing.allocator;
    var built = try buildTestIndex(gpa, 2002, 50, 16);
    defer {
        built.index.deinit(gpa);
        gpa.free(built.image);
        gpa.free(built.fx_bytes);
    }
    const idx = &built.index;
    const dim: usize = @intCast(idx.header.dim);

    // Synthesise n_q = 2 query tokens from centroids 0 and 2 — guaranteed
    // to produce non-trivial cosine sims with a wide spread of doc tokens.
    const n_q: u32 = 2;
    var query = try gpa.alloc(f32, @as(usize, n_q) * dim);
    defer gpa.free(query);
    @memcpy(query[0..dim], idx.centroids[0..dim]);
    @memcpy(query[dim..], idx.centroids[2 * dim .. 3 * dim]);

    const table_len: usize =
        @as(usize, constants.PQ_M) * @as(usize, constants.PQ_CENTROIDS) * n_q;
    const table = try gpa.alloc(f32, table_len);
    defer gpa.free(table);
    try idx.pq.buildDistanceTable(query, n_q, table);

    const per_token_max = try gpa.alloc(f32, n_q);
    defer gpa.free(per_token_max);

    // Brute force: reconstruct each doc token as `c_k + norm_k · decode(codes)`
    // and compute MaxSim directly. This validates both the formula and the
    // table layout used by refine.
    var d: u32 = 0;
    var checked: u32 = 0;
    while (d < idx.header.n_docs) : (d += 1) {
        const cand = gather.Candidate{ .doc_id = d, .score = 0.0 };
        const got = try refine(idx, table, query, n_q, cand, per_token_max);

        const layout = try decodeDocLayout(idx.docLayout(d));
        const tok_range = idx.docTokenRange(d);
        const tok_lo: usize = @intCast(tok_range[0]);

        var bf_sum: f32 = 0.0;
        var i: u32 = 0;
        while (i < n_q) : (i += 1) {
            const q_i = query[@as(usize, i) * dim ..][0..dim];
            var best: f32 = -std.math.inf(f32);
            var k: u32 = 0;
            while (k < layout.n_d) : (k += 1) {
                const c_id: u32 = layout.centroid_ids[k];
                const c_vec = idx.centroids[@as(usize, c_id) * dim ..][0..dim];
                const norm_k = idx.residual_norms[tok_lo + k];

                var residual = try gpa.alloc(f32, dim);
                defer gpa.free(residual);
                var m: u32 = 0;
                while (m < constants.PQ_M) : (m += 1) {
                    const code: u8 = layout.pq_codes[k * constants.PQ_M + m];
                    const sub = idx.pq.codebookEntry(m, code);
                    const dst = residual[m * idx.pq.sub_dim ..][0..idx.pq.sub_dim];
                    @memcpy(dst, sub);
                }

                var sim: f32 = 0.0;
                var dd: usize = 0;
                while (dd < dim) : (dd += 1) {
                    sim += q_i[dd] * (c_vec[dd] + norm_k * residual[dd]);
                }
                if (sim > best) best = sim;
            }
            if (layout.n_d > 0) bf_sum += best;
        }

        try testing.expectApproxEqAbs(bf_sum, got, 1e-3);
        checked += 1;
    }
    try testing.expect(checked > 0);
}

test "refine: deterministic — same inputs produce byte-identical score" {
    const gpa = testing.allocator;
    var built = try buildTestIndex(gpa, 3003, 50, 16);
    defer {
        built.index.deinit(gpa);
        gpa.free(built.image);
        gpa.free(built.fx_bytes);
    }
    const idx = &built.index;
    const dim: usize = @intCast(idx.header.dim);
    const n_q: u32 = 2;
    var query = try gpa.alloc(f32, @as(usize, n_q) * dim);
    defer gpa.free(query);
    @memcpy(query[0..dim], idx.centroids[0..dim]);
    @memcpy(query[dim..], idx.centroids[dim .. 2 * dim]);

    const table_len: usize =
        @as(usize, constants.PQ_M) * @as(usize, constants.PQ_CENTROIDS) * n_q;
    const table = try gpa.alloc(f32, table_len);
    defer gpa.free(table);
    try idx.pq.buildDistanceTable(query, n_q, table);

    const buf_a = try gpa.alloc(f32, n_q);
    defer gpa.free(buf_a);
    const buf_b = try gpa.alloc(f32, n_q);
    defer gpa.free(buf_b);

    const cand = gather.Candidate{ .doc_id = 7, .score = 0.0 };
    const a = try refine(idx, table, query, n_q, cand, buf_a);
    const b = try refine(idx, table, query, n_q, cand, buf_b);
    try testing.expectEqual(a, b);
}

test "refine: doc_id out of range rejected" {
    const gpa = testing.allocator;
    var built = try buildTestIndex(gpa, 4004, 50, 16);
    defer {
        built.index.deinit(gpa);
        gpa.free(built.image);
        gpa.free(built.fx_bytes);
    }
    const idx = &built.index;
    const dim: usize = @intCast(idx.header.dim);
    const n_q: u32 = 1;
    const query = try gpa.alloc(f32, dim);
    defer gpa.free(query);
    @memcpy(query, idx.centroids[0..dim]);

    const table_len: usize =
        @as(usize, constants.PQ_M) * @as(usize, constants.PQ_CENTROIDS) * n_q;
    const table = try gpa.alloc(f32, table_len);
    defer gpa.free(table);
    try idx.pq.buildDistanceTable(query, n_q, table);

    const buf = try gpa.alloc(f32, n_q);
    defer gpa.free(buf);

    const cand = gather.Candidate{ .doc_id = 9999, .score = 0.0 };
    try testing.expectError(
        error.DocOutOfRange,
        refine(idx, table, query, n_q, cand, buf),
    );
}

test "refine: query shape and table size mismatches rejected" {
    const gpa = testing.allocator;
    var built = try buildTestIndex(gpa, 5005, 50, 16);
    defer {
        built.index.deinit(gpa);
        gpa.free(built.image);
        gpa.free(built.fx_bytes);
    }
    const idx = &built.index;
    const dim: usize = @intCast(idx.header.dim);
    const n_q: u32 = 2;
    const query = try gpa.alloc(f32, @as(usize, n_q) * dim);
    defer gpa.free(query);
    @memcpy(query[0..dim], idx.centroids[0..dim]);
    @memcpy(query[dim..], idx.centroids[dim .. 2 * dim]);

    const buf = try gpa.alloc(f32, n_q);
    defer gpa.free(buf);

    const bad_table = try gpa.alloc(f32, 16);
    defer gpa.free(bad_table);
    try testing.expectError(
        error.TableSizeMismatch,
        refine(idx, bad_table, query, n_q, .{ .doc_id = 0, .score = 0 }, buf),
    );

    const table_len: usize =
        @as(usize, constants.PQ_M) * @as(usize, constants.PQ_CENTROIDS) * n_q;
    const table = try gpa.alloc(f32, table_len);
    defer gpa.free(table);
    try idx.pq.buildDistanceTable(query, n_q, table);
    const short_query = try gpa.alloc(f32, dim - 1);
    defer gpa.free(short_query);
    try testing.expectError(
        error.QueryShapeMismatch,
        refine(idx, table, short_query, n_q, .{ .doc_id = 0, .score = 0 }, buf),
    );
}
