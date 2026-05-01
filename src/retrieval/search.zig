//! src/retrieval/search.zig — Retrieval.search driver (paper §5).
//!
//! Owner: retriever. Stitches gather → prune → refine into the public
//! retrieval entry point.
//!
//! NOTE: this file is not yet wired into `src/root.zig`'s `retrieval` block.
//! Adding the line is lead-owned (root.zig is a cross-cutting file). Until
//! that lands, search() is callable from in-tree tests but doesn't appear
//! on the public `tac.retrieval.*` namespace.

const std = @import("std");
const Allocator = std.mem.Allocator;

const constants = @import("../constants.zig");
const storage = @import("../index/storage.zig");
const pq_mod = @import("../index/pq.zig");
const gather = @import("gather.zig");
const prune = @import("prune.zig");
const refine = @import("refine.zig");

pub const SearchParams = struct {
    kappa_c: u32,
    kappa_d: u32,
    alpha: ?f32 = null,
    top_k: u32,
};

pub const ScoredDoc = struct {
    doc_id: u32,
    score: f32,
};

pub const SearchError = error{
    TopKExceedsKappaD,
} || gather.GatherError || pq_mod.PqError || refine.RefineError;

fn cmpDescScored(_: void, a: ScoredDoc, b: ScoredDoc) bool {
    return a.score > b.score;
}

/// Run the full paper §5 retrieval pipeline.
///
/// Returns an allocator-owned slice of `≤ top_k` ScoredDocs sorted descending
/// by exact MaxSim. Caller frees with `gpa.free(result)`.
///
/// Determinism: gather emits the same candidate set for a given query/index;
/// prune sorts deterministically; refine is pure; final sort uses strict `>`
/// so equal-score ties keep input order. Sticky doc-id tie-breaks would
/// need a secondary key — defer until benchmarks reveal a need.
pub fn search(
    index: *const storage.Index,
    query_tokens: []const f32,
    n_q: u32,
    params: SearchParams,
    gpa: Allocator,
) SearchError![]ScoredDoc {
    if (params.top_k > params.kappa_d) return error.TopKExceedsKappaD;

    // Gather (paper §5.1): walk inverted lists, accumulate S̃(q, d).
    const candidates = try gather.gather(
        index,
        query_tokens,
        n_q,
        .{ .kappa_c = params.kappa_c },
        gpa,
    );
    defer gpa.free(candidates);

    // Prune (paper §5.2): top-κ_d truncation + optional adaptive CP.
    const survivors = prune.prune(candidates, .{
        .kappa_d = params.kappa_d,
        .alpha = params.alpha,
    });
    if (survivors.len == 0) return gpa.alloc(ScoredDoc, 0);

    // Build the per-query distance table once; shared across all refine calls.
    const table_len: usize =
        @as(usize, constants.PQ_M) * @as(usize, constants.PQ_CENTROIDS) * @as(usize, n_q);
    const table = try gpa.alloc(f32, table_len);
    defer gpa.free(table);
    try index.pq.buildDistanceTable(query_tokens, n_q, table);

    // Refine (paper §5.3): exact MaxSim per surviving candidate.
    var refined = try gpa.alloc(ScoredDoc, survivors.len);
    errdefer gpa.free(refined);
    const per_token_max = try gpa.alloc(f32, n_q);
    defer gpa.free(per_token_max);

    for (survivors, 0..) |s, k| {
        refined[k] = .{
            .doc_id = s.doc_id,
            .score = try refine.refine(index, table, query_tokens, n_q, s, per_token_max),
        };
    }

    std.sort.pdq(ScoredDoc, refined, {}, cmpDescScored);

    // Trim to top_k.
    const want: usize = @min(refined.len, @as(usize, params.top_k));
    if (want < refined.len) {
        const trimmed = try gpa.alloc(ScoredDoc, want);
        @memcpy(trimmed, refined[0..want]);
        gpa.free(refined);
        return trimmed;
    }
    return refined;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;
const synthetic_fixture = @import("../io/synthetic_fixture.zig");
const token_dump = @import("../io/token_dump.zig");

test "SearchParams + ScoredDoc field shapes" {
    try testing.expectEqual(@as(usize, 8), @sizeOf(ScoredDoc));
    const p = SearchParams{ .kappa_c = 80, .kappa_d = 1000, .top_k = 10 };
    try testing.expectEqual(@as(u32, 80), p.kappa_c);
    try testing.expect(p.alpha == null);
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

test "search: end-to-end returns ≤ top_k results sorted by descending score" {
    const gpa = testing.allocator;
    var built = try buildTestIndex(gpa, 8001, 50, 16);
    defer {
        built.index.deinit(gpa);
        gpa.free(built.image);
        gpa.free(built.fx_bytes);
    }
    const idx = &built.index;
    const dim: usize = @intCast(idx.header.dim);

    // 3-token query from centroids 0, 2, 4.
    const n_q: u32 = 3;
    var query = try gpa.alloc(f32, @as(usize, n_q) * dim);
    defer gpa.free(query);
    @memcpy(query[0..dim], idx.centroids[0..dim]);
    @memcpy(query[dim .. 2 * dim], idx.centroids[2 * dim .. 3 * dim]);
    @memcpy(query[2 * dim .. 3 * dim], idx.centroids[4 * dim .. 5 * dim]);

    const result = try search(idx, query, n_q, .{
        .kappa_c = 8,
        .kappa_d = 20,
        .alpha = null,
        .top_k = 5,
    }, gpa);
    defer gpa.free(result);

    try testing.expect(result.len <= 5);
    try testing.expect(result.len > 0);
    var i: usize = 1;
    while (i < result.len) : (i += 1) {
        try testing.expect(result[i - 1].score >= result[i].score);
    }
}

test "search: top_k > kappa_d rejected" {
    const gpa = testing.allocator;
    var built = try buildTestIndex(gpa, 8002, 50, 16);
    defer {
        built.index.deinit(gpa);
        gpa.free(built.image);
        gpa.free(built.fx_bytes);
    }
    const idx = &built.index;
    const dim: usize = @intCast(idx.header.dim);
    const query = try gpa.alloc(f32, dim);
    defer gpa.free(query);
    @memcpy(query, idx.centroids[0..dim]);
    try testing.expectError(error.TopKExceedsKappaD, search(idx, query, 1, .{
        .kappa_c = 8,
        .kappa_d = 5,
        .top_k = 100,
    }, gpa));
}

test "search: deterministic — same query → same result" {
    const gpa = testing.allocator;
    var built = try buildTestIndex(gpa, 8003, 50, 16);
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

    const params = SearchParams{ .kappa_c = 8, .kappa_d = 20, .alpha = 0.4, .top_k = 5 };
    const a = try search(idx, query, n_q, params, gpa);
    defer gpa.free(a);
    const b = try search(idx, query, n_q, params, gpa);
    defer gpa.free(b);
    try testing.expectEqual(a.len, b.len);
    for (a, b) |x, y| {
        try testing.expectEqual(x.doc_id, y.doc_id);
        try testing.expectEqual(x.score, y.score);
    }
}

test "search: kappa_d > total candidates returns full sorted set up to top_k" {
    const gpa = testing.allocator;
    var built = try buildTestIndex(gpa, 8004, 50, 16);
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

    const result = try search(idx, query, n_q, .{
        .kappa_c = 16,
        .kappa_d = 10_000,
        .top_k = 5,
    }, gpa);
    defer gpa.free(result);

    try testing.expect(result.len <= 5);
}
