//! src/retrieval/gather.zig — Gather phase (paper §5.1).
//!
//! Owner: retriever.
//! See plan 04-retrieval-and-eval.plan.md.
//!
//! For each query token q_i:
//!   - HNSW top-κ_c centroids with ef_s = 1.5·κ_c.
//!   - Walk inverted lists L_j of those centroids, accumulate
//!     `s̃_i(d) = max_{ j : d ∈ L_j } ⟨q_i, c_j⟩`.
//! Aggregate across query tokens:
//!   `S̃(q, d) = Σ_{i=1..n_q} s̃_i(d)`.
//!
//! Required surface (locked by task #17):
//!   pub const Candidate = struct { doc_id: u32, score: f32 };
//!   pub const GatherParams = struct { kappa_c: u32 };
//!   pub fn gather(
//!       index: *const tac.index.storage.Index,
//!       query_tokens: []const f32,    // n_q · dim, row-major, L2-normalised
//!       n_q: u32,
//!       params: GatherParams,
//!       gpa: Allocator,
//!   ) ![]Candidate;                   // unsorted, deduped per (doc_id), score = S̃
//!
//! ---------------------------------------------------------------------------
//! DESIGN PSEUDOCODE — implemented under #17 (waits on #15 Index.build).
//! ---------------------------------------------------------------------------
//!
//! gather(index, query_tokens, n_q, params, gpa):
//!   assert query_tokens.len == n_q * index.dim;
//!   ef_s = constants.hnswEfSearch(params.kappa_c);
//!     // paper-gap: §5.1 writes ef_s = 1.5·κ_c without specifying per-token vs
//!     // per-query. We run HNSW per-token (one search per q_i) — matches the
//!     // natural reading "for each query token". Per-query (one HNSW pass with
//!     // a fattened ef) is the alternative; cross-check vs reference Rust impl
//!     // tracked in plan 06.
//!
//!   // Dense scratch over docs: faster than a hash map for typical hit counts
//!   // (~10–50K candidates touched on MS MARCO at κ_c=80).
//!   per_token_max := gpa.alloc(f32, index.n_docs)
//!   touched_for_tok : ArrayList(u32)
//!   accum := gpa.alloc(f32, index.n_docs)             // S̃(q,d)
//!   touched_global : ArrayList(u32)
//!   centroid_buf := gpa.alloc(u32, params.kappa_c)
//!
//!   for i in 0..n_q:
//!     q_i = query_tokens[i*dim .. (i+1)*dim]
//!     n_hit = index.hnsw.search(q_i, params.kappa_c, ef_s, centroid_buf)
//!
//!     // Reset only the slots we wrote last token.
//!     for d in touched_for_tok: per_token_max[d] = 0
//!     touched_for_tok.clear()
//!
//!     for j_idx in 0..n_hit:
//!       j = centroid_buf[j_idx]
//!       sim = util.vec.dot(q_i, index.centroid_at(j))
//!       for d in index.invertedList(j):
//!         if per_token_max[d] == 0: touched_for_tok.append(d)
//!         if sim > per_token_max[d]: per_token_max[d] = sim
//!         // paper-gap: zero-sentinel works because ⟨q,c⟩ for L2-normalised
//!         // unit vectors is in [-1, 1]; a true 0 sim is identical to
//!         // "untouched" only at f32 zero, fine for ranking.
//!         // Switch to a u32 generation counter if exactness ever bites.
//!
//!     for d in touched_for_tok:
//!       if accum[d] == 0: touched_global.append(d)
//!       accum[d] += per_token_max[d]
//!
//!   out := gpa.alloc(Candidate, touched_global.len)
//!   for k in 0..touched_global.len:
//!     d = touched_global[k]
//!     out[k] = .{ .doc_id = d, .score = accum[d] }
//!   return out;                                        // unsorted; prune sorts.
//!
//! Tests planned for #17:
//!   - 4 centroids, 8 docs, n_q=2 — hand-compute s̃_i(d) and S̃(q,d), match
//!     within 1e-4 of brute-force baseline.
//!   - Determinism: same query → byte-identical candidate set & scores.
//!   - Empty inverted list for a hit centroid → contributes nothing, no crash.

const std = @import("std");

pub const Candidate = struct {
    doc_id: u32,
    score: f32,
};

pub const GatherParams = struct {
    kappa_c: u32,
};

test "Candidate is 8 bytes (u32 + f32)" {
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(Candidate));
}

test "GatherParams round-trips kappa_c" {
    const p = GatherParams{ .kappa_c = 80 };
    try std.testing.expectEqual(@as(u32, 80), p.kappa_c);
}
