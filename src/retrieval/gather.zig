//! src/retrieval/gather.zig — Gather phase (paper §5.1).
//!
//! Owner: retriever.
//! See plan 04-retrieval-and-eval.plan.md.
//!
//! For each query token q_i:
//!   - HNSW top-κ_c centroids with ef_s = 1.5·κ_c.
//!   - Walk the inverted list L_j of each hit centroid, accumulate
//!     `s̃_i(d) = max_{ j : d ∈ L_j } ⟨q_i, c_j⟩`.
//! Aggregate across query tokens:
//!   `S̃(q, d) = Σ_{i=1..n_q} s̃_i(d)`.
//!
//! No PQ touches in this phase — that's the whole point of the
//! gather/refine split. Returned candidates are unsorted; `prune.prune`
//! sorts and truncates.

const std = @import("std");
const Allocator = std.mem.Allocator;

const constants = @import("../constants.zig");
const vec = @import("../util/vec.zig");
const storage = @import("../index/storage.zig");
const hnsw_mod = @import("../index/hnsw.zig");

pub const Candidate = struct {
    doc_id: u32,
    score: f32,
};

pub const GatherParams = struct {
    kappa_c: u32,
};

pub const GatherError = error{
    QueryShapeMismatch,
    KappaCZero,
} || Allocator.Error || vec.VecError || hnsw_mod.HnswError;

/// Run the paper §5.1 gather over `query_tokens` (n_q · dim, row-major,
/// L2-normalised). Returns the union of touched docs with their aggregate
/// `S̃(q, d)` scores. Caller frees the slice.
///
/// paper-gap §5.1: the paper writes ef_s = 1.5·κ_c without saying whether
/// the HNSW search runs per-token or per-query. We run per-token — matches
/// the natural reading "for each query token … HNSW top-κ_c". Cross-check
/// vs reference Rust impl tracked in plan 06.
pub fn gather(
    index: *const storage.Index,
    query_tokens: []const f32,
    n_q: u32,
    params: GatherParams,
    gpa: Allocator,
) GatherError![]Candidate {
    if (params.kappa_c == 0) return error.KappaCZero;
    const dim: usize = @intCast(index.header.dim);
    if (n_q == 0 or query_tokens.len != @as(usize, n_q) * dim) {
        return error.QueryShapeMismatch;
    }

    const n_docs: usize = @intCast(index.header.n_docs);
    if (n_docs == 0) return gpa.alloc(Candidate, 0);

    const ef_s = constants.hnswEfSearch(params.kappa_c);

    // Per-token max similarity (s̃_i(d)) and the running aggregate (S̃(q, d)).
    // Dense f32 buffers are faster than a hash map for typical candidate
    // counts (~10–50K on MS MARCO @ κ_c=80). Two parallel u8 "seen" bitmaps
    // record whether a doc was touched in the current query-token pass and
    // whether it ever made it into `touched_global`. We reset per_tok_seen
    // entries from `touched_for_tok` so we never do an O(n_docs) clear.
    const per_token_max = try gpa.alloc(f32, n_docs);
    defer gpa.free(per_token_max);
    const per_tok_seen = try gpa.alloc(bool, n_docs);
    defer gpa.free(per_tok_seen);
    @memset(per_tok_seen, false);

    const accum = try gpa.alloc(f32, n_docs);
    defer gpa.free(accum);
    @memset(accum, 0.0);
    const global_seen = try gpa.alloc(bool, n_docs);
    defer gpa.free(global_seen);
    @memset(global_seen, false);

    var touched_for_tok: std.ArrayList(u32) = .empty;
    defer touched_for_tok.deinit(gpa);
    var touched_global: std.ArrayList(u32) = .empty;
    defer touched_global.deinit(gpa);

    const centroid_buf = try gpa.alloc(u32, params.kappa_c);
    defer gpa.free(centroid_buf);

    var i: u32 = 0;
    while (i < n_q) : (i += 1) {
        const q_i = query_tokens[@as(usize, i) * dim ..][0..dim];

        // Reset only the per-tok-seen slots we wrote on the previous token.
        for (touched_for_tok.items) |d| per_tok_seen[d] = false;
        touched_for_tok.clearRetainingCapacity();

        const n_hit = try hnsw_mod.search(&index.hnsw, q_i, params.kappa_c, ef_s, centroid_buf, gpa);

        var j_idx: u32 = 0;
        while (j_idx < n_hit) : (j_idx += 1) {
            const j: u32 = centroid_buf[j_idx];
            const c_j = index.centroids[@as(usize, j) * dim ..][0..dim];
            const sim = try vec.dot(q_i, c_j);
            for (index.ilists.list(j)) |d| {
                if (!per_tok_seen[d]) {
                    per_tok_seen[d] = true;
                    per_token_max[d] = sim;
                    try touched_for_tok.append(gpa, d);
                } else if (sim > per_token_max[d]) {
                    per_token_max[d] = sim;
                }
            }
        }

        for (touched_for_tok.items) |d| {
            if (!global_seen[d]) {
                global_seen[d] = true;
                try touched_global.append(gpa, d);
            }
            accum[d] += per_token_max[d];
        }
    }

    const out = try gpa.alloc(Candidate, touched_global.items.len);
    for (touched_global.items, 0..) |d, k| {
        out[k] = .{ .doc_id = d, .score = accum[d] };
    }
    return out;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "Candidate is 8 bytes (u32 + f32)" {
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(Candidate));
}

test "GatherParams round-trips kappa_c" {
    const p = GatherParams{ .kappa_c = 80 };
    try std.testing.expectEqual(@as(u32, 80), p.kappa_c);
}

const synthetic_fixture = @import("../io/synthetic_fixture.zig");
const token_dump = @import("../io/token_dump.zig");

/// Build a small Index from a synthetic fixture for end-to-end gather tests.
/// The caller must `built.index.deinit(gpa)`, `gpa.free(built.image)`, and
/// `gpa.free(built.fx_bytes)`. Fixture parameters mirror the indexer's own
/// integration test in src/index/storage.zig — those are the smallest
/// numbers that satisfy TAC budget allocation with permissive thresholds.
fn buildTestIndex(
    gpa: Allocator,
    seed: u64,
    n_docs: u64,
    kappa: u32,
) !struct { index: storage.Index, image: []align(8) u8, fx_bytes: []align(8) u8 } {
    var fx = try synthetic_fixture.build(gpa, .{
        .seed = seed,
        .n_docs = n_docs,
        .dim = 32, // matches PQ_M so sub_dim = 1.
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
        // Test-only TAC overrides: paper-strict (μ=128, τ=256, θ=39) is
        // impossible at fixture scale.
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

test "gather: returns deduped candidate set with positive scores" {
    const gpa = std.testing.allocator;
    var built = try buildTestIndex(gpa, 7, 50, 16);
    defer {
        built.index.deinit(gpa);
        gpa.free(built.image);
        gpa.free(built.fx_bytes);
    }

    // Use the first doc's first 3 tokens as a query — guaranteed to hit.
    const dim = built.index.header.dim;
    const q_doc_lo: usize = @intCast(built.index.doc_index[0]);
    _ = q_doc_lo;
    // Build a synthetic query directly from a centroid — ensures HNSW finds
    // exactly that centroid's posting list. We pick centroid 0.
    const n_q: u32 = 2;
    const c0 = built.index.centroids[0..dim];
    const c1 = built.index.centroids[dim .. 2 * dim];
    var query = try gpa.alloc(f32, @as(usize, n_q) * dim);
    defer gpa.free(query);
    @memcpy(query[0..dim], c0);
    @memcpy(query[dim..], c1);

    const cands = try gather(&built.index, query, n_q, .{ .kappa_c = 4 }, gpa);
    defer gpa.free(cands);

    try std.testing.expect(cands.len > 0);
    // All scores positive: queries are centroids themselves, so ⟨q, c_q⟩ = 1.
    for (cands) |c| {
        try std.testing.expect(c.score > 0.0);
        try std.testing.expect(c.doc_id < built.index.header.n_docs);
    }
    // Doc IDs unique.
    var seen: std.AutoHashMap(u32, void) = .init(gpa);
    defer seen.deinit();
    for (cands) |c| {
        const r = try seen.getOrPut(c.doc_id);
        try std.testing.expect(!r.found_existing);
    }
}

test "gather: deterministic — same query produces same candidate set + scores" {
    const gpa = std.testing.allocator;
    var built = try buildTestIndex(gpa, 13, 50, 16);
    defer {
        built.index.deinit(gpa);
        gpa.free(built.image);
        gpa.free(built.fx_bytes);
    }
    const dim = built.index.header.dim;
    const n_q: u32 = 2;
    var query = try gpa.alloc(f32, @as(usize, n_q) * dim);
    defer gpa.free(query);
    @memcpy(query[0..dim], built.index.centroids[0..dim]);
    @memcpy(query[dim..], built.index.centroids[dim .. 2 * dim]);

    const a = try gather(&built.index, query, n_q, .{ .kappa_c = 3 }, gpa);
    defer gpa.free(a);
    const b = try gather(&built.index, query, n_q, .{ .kappa_c = 3 }, gpa);
    defer gpa.free(b);

    try std.testing.expectEqual(a.len, b.len);
    for (a, b) |x, y| {
        try std.testing.expectEqual(x.doc_id, y.doc_id);
        try std.testing.expectEqual(x.score, y.score);
    }
}

test "gather: matches brute-force S̃ for kappa_c = kappa (all centroids)" {
    const gpa = std.testing.allocator;
    const kappa: u32 = 16;
    var built = try buildTestIndex(gpa, 21, 50, kappa);
    defer {
        built.index.deinit(gpa);
        gpa.free(built.image);
        gpa.free(built.fx_bytes);
    }
    const idx = &built.index;
    const dim: usize = @intCast(idx.header.dim);
    const n_docs: u32 = @intCast(idx.header.n_docs);

    // Synthesise 2 query tokens — pick centroids 0 and 2.
    const n_q: u32 = 2;
    var query = try gpa.alloc(f32, @as(usize, n_q) * dim);
    defer gpa.free(query);
    @memcpy(query[0..dim], idx.centroids[0..dim]);
    @memcpy(query[dim..], idx.centroids[2 * dim .. 3 * dim]);

    const cands = try gather(idx, query, n_q, .{ .kappa_c = kappa }, gpa);
    defer gpa.free(cands);

    // Brute force: for each (i, d), compute s̃_i(d) = max over centroids j
    // such that d ∈ L_j of ⟨q_i, c_j⟩. Sum across i.
    const expected = try gpa.alloc(f32, n_docs);
    defer gpa.free(expected);
    @memset(expected, 0.0);

    var i: u32 = 0;
    while (i < n_q) : (i += 1) {
        const q_i = query[@as(usize, i) * dim ..][0..dim];
        // For each doc, scan centroids it appears in.
        var d: u32 = 0;
        while (d < n_docs) : (d += 1) {
            var best: f32 = 0.0;
            var found = false;
            var j: u32 = 0;
            while (j < kappa) : (j += 1) {
                const lst = idx.ilists.list(j);
                var contains = false;
                for (lst) |x| if (x == d) {
                    contains = true;
                    break;
                };
                if (!contains) continue;
                const c_j = idx.centroids[@as(usize, j) * dim ..][0..dim];
                const sim = try vec.dot(q_i, c_j);
                if (!found or sim > best) {
                    best = sim;
                    found = true;
                }
            }
            if (found) expected[d] += best;
        }
    }

    // Every candidate score must equal the brute-force aggregate within FP
    // tolerance. Docs not in `cands` must have expected[d] == 0.
    for (cands) |c| {
        try std.testing.expectApproxEqAbs(expected[c.doc_id], c.score, 1e-4);
    }
    var seen = try gpa.alloc(bool, n_docs);
    defer gpa.free(seen);
    @memset(seen, false);
    for (cands) |c| seen[c.doc_id] = true;
    var d: u32 = 0;
    while (d < n_docs) : (d += 1) {
        if (!seen[d]) try std.testing.expectEqual(@as(f32, 0.0), expected[d]);
    }
}

test "gather: kappa_c = 0 rejected" {
    const gpa = std.testing.allocator;
    var built = try buildTestIndex(gpa, 3, 50, 16);
    defer {
        built.index.deinit(gpa);
        gpa.free(built.image);
        gpa.free(built.fx_bytes);
    }
    const dim = built.index.header.dim;
    const query = try gpa.alloc(f32, dim);
    defer gpa.free(query);
    @memcpy(query, built.index.centroids[0..dim]);
    try std.testing.expectError(
        error.KappaCZero,
        gather(&built.index, query, 1, .{ .kappa_c = 0 }, gpa),
    );
}

test "gather: query shape mismatch rejected" {
    const gpa = std.testing.allocator;
    var built = try buildTestIndex(gpa, 4, 50, 16);
    defer {
        built.index.deinit(gpa);
        gpa.free(built.image);
        gpa.free(built.fx_bytes);
    }
    const dim = built.index.header.dim;
    const query = try gpa.alloc(f32, dim - 1); // wrong size
    defer gpa.free(query);
    try std.testing.expectError(
        error.QueryShapeMismatch,
        gather(&built.index, query, 1, .{ .kappa_c = 2 }, gpa),
    );
}
