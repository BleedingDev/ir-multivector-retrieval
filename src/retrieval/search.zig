//! src/retrieval/search.zig — Retrieval.search driver (paper §5).
//!
//! Owner: retriever.
//!
//! Stitches gather → prune → refine into the public retrieval entry point.
//! `root.zig` import wiring is lead-owned (`pub const search = @import("retrieval/search.zig")`
//! goes into the `retrieval` namespace block) — message the lead once #17/#18/#19
//! are real to add the line.
//!
//! Required surface (locked by task #17–#19 dependencies):
//!   pub const SearchParams = struct {
//!       kappa_c: u32,         // centroids per query token (HNSW)
//!       kappa_d: u32,         // candidates kept after Stage 1 truncation
//!       alpha: ?f32 = null,   // adaptive Candidates Pruning fraction
//!       top_k: u32,           // final result count
//!   };
//!   pub const ScoredDoc = struct { doc_id: u32, score: f32 };
//!   pub fn search(
//!       index: *const tac.index.storage.Index,
//!       pq:    *const tac.index.pq.PQ,
//!       query_tokens: []const f32,
//!       n_q: u32,
//!       params: SearchParams,
//!       gpa: Allocator,
//!   ) ![]ScoredDoc;            // caller frees; sorted descending by exact MaxSim
//!
//! ---------------------------------------------------------------------------
//! DESIGN PSEUDOCODE (assembles after #17, #18, #19 land).
//! ---------------------------------------------------------------------------
//!
//! search(index, pq, query_tokens, n_q, params, gpa):
//!   assert query_tokens.len == n_q * index.dim;
//!   assert params.top_k <= params.kappa_d;          // can't return more than survive prune
//!
//!   // --- Gather (paper §5.1) -------------------------------------------------
//!   candidates = gather.gather(index, query_tokens, n_q,
//!                              .{ .kappa_c = params.kappa_c }, gpa);
//!   defer gpa.free(candidates);
//!
//!   // --- Prune (paper §5.2) --------------------------------------------------
//!   survivors = prune.prune(candidates,
//!                           .{ .kappa_d = params.kappa_d, .alpha = params.alpha });
//!     // survivors is a sub-slice of candidates, sorted descending by S̃.
//!
//!   if (survivors.len == 0) return gpa.dupe(ScoredDoc, &.{});
//!
//!   // --- Build the cache-optimised distance table once per query.
//!   table = gpa.alloc(f32, constants.PQ_M * 256 * n_q);
//!   defer gpa.free(table);
//!   pq.buildDistanceTable(query_tokens, n_q, table);
//!
//!   // --- Refine (paper §5.3) -------------------------------------------------
//!   // Each candidate gets an exact MaxSim score.
//!   refined = gpa.alloc(ScoredDoc, survivors.len);
//!   for k in 0..survivors.len:
//!       refined[k] = .{
//!           .doc_id = survivors[k].doc_id,
//!           .score  = refine.refine(index, pq, table, query_tokens, n_q,
//!                                   survivors[k], gpa),
//!       };
//!
//!   // Final ranking: descending by exact score.
//!   std.sort.pdq(ScoredDoc, refined, {},
//!                fn(_, a, b) bool { return a.score > b.score; });
//!
//!   if refined.len > params.top_k:
//!       // Trim allocation to top_k. Keep the same slice header by realloc.
//!       result = gpa.realloc(refined, params.top_k);
//!       return result;
//!   return refined;
//!
//! ---------------------------------------------------------------------------
//! Determinism: gather emits the same candidate set for a given query/index;
//!   prune sorts; refine is pure on (centroids, PQ codes, table); final sort
//!   is deterministic (pdq with strict `>`). Sticky tie-breaks would require
//!   a secondary key — defer until benchmarks reveal a need.
//!
//! Tests planned alongside #19 / inside #20 harness:
//!   - End-to-end on a 100-doc fixture: known relevance pattern → top_k matches
//!     hand-computed exact MaxSim ordering.
//!   - With kappa_d > total_docs → degenerates gracefully (no truncation).
//!   - With top_k > kappa_d → returns kappa_d entries (assertion or clamp).

const std = @import("std");

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

test "ScoredDoc and SearchParams field shapes" {
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(ScoredDoc));
    const p = SearchParams{ .kappa_c = 80, .kappa_d = 1000, .top_k = 10 };
    try std.testing.expectEqual(@as(u32, 80), p.kappa_c);
    try std.testing.expect(p.alpha == null);
}
