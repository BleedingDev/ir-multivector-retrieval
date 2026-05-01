//! src/retrieval/refine.zig — Refine phase (paper §5.3).
//!
//! Owner: retriever.
//! See plan 04-retrieval-and-eval.plan.md.
//!
//! For each surviving candidate document d:
//!   Pass 1: stream centroid IDs → per-(q_i, k) centroid contribution
//!           ⟨q_i, c_{c_k}⟩ scaled by stored residual norm.
//!   Pass 2: stream PQ codes → residual contribution via the cache-optimised
//!           distance table laid out [M macro][256 code][n_q micro].
//! Output: exact MaxSim score for the (q, d) pair.
//!
//! For token k of doc d the contribution from query token i is:
//!     score_i(k, d) = ⟨q_i, c_{c_k}⟩ · norm_k                  // centroid term × stored residual norm
//!                   + Σ_{m=0..M-1} table[m][PQ_{k,m}][i]       // residual term
//! Per-token MaxSim:
//!     s_i(d) = max_{k=1..n_d} score_i(k, d)
//! Final:
//!     S(q, d) = Σ_{i=1..n_q} s_i(d)
//!
//! Required surface (locked by task #19):
//!   pub fn refine(
//!       index: *const tac.index.storage.Index,
//!       pq:    *const tac.index.pq.PQ,
//!       distance_table: []const f32,           // PQ_M · 256 · n_q (built once per query)
//!       query_tokens: []const f32,             // n_q · dim
//!       n_q: u32,
//!       candidate: gather.Candidate,
//!       gpa: Allocator,                        // n_q f32 scratch
//!   ) f32;
//!
//! ---------------------------------------------------------------------------
//! DESIGN PSEUDOCODE — implemented under #19 (waits on #12 PQ table + #18).
//! ---------------------------------------------------------------------------
//!
//! refine(index, pq, table, query_tokens, n_q, candidate, gpa):
//!   d = candidate.doc_id;
//!   layout       = index.docLayout(d);          // paper §4 doc layout
//!   centroid_ids = layout.centroid_ids;         // []u32, length n_d
//!   pq_codes     = layout.pq_codes;             // []u8,  length n_d * M
//!   norms        = layout.residual_norms;       // []f32, length n_d
//!   n_d = centroid_ids.len;
//!
//!   per_token_max := gpa.alloc(f32, n_q);
//!   for i in 0..n_q: per_token_max[i] = -inf;
//!
//!   for k in 0..n_d:
//!     c_k    = centroid_ids[k];
//!     c_vec  = index.centroid_at(c_k);
//!     norm_k = norms[k];
//!
//!     for i in 0..n_q:
//!       q_i = query_tokens[i*dim .. (i+1)*dim];
//!       cent_term = util.vec.dot(q_i, c_vec) * norm_k;
//!
//!       // Residual term: stream PQ codes for this k, sum table[m][code][i] over m.
//!       res_term = 0;
//!       for m in 0..PQ_M:
//!         code = pq_codes[k*PQ_M + m];
//!         res_term += table[ pq.tableIndex(n_q, m, code, i) ];
//!
//!       score = cent_term + res_term;
//!       if score > per_token_max[i]: per_token_max[i] = score;
//!
//!   S = 0;
//!   for i in 0..n_q: S += per_token_max[i];
//!   gpa.free(per_token_max);
//!   return S;
//!
//! Cache-optimised variant for the inner loop (paper §5.3, "up to 3.8×"):
//!   Outer-m, inner-(k, i) restructuring lets each (m, code) lookup yield a
//!   contiguous n_q-wide row → unit-stride SIMD load:
//!
//!     centroid_terms : [n_d][n_q]f32   // computed once per doc
//!     for m in 0..PQ_M:
//!       for k in 0..n_d:
//!         code = pq_codes[k*PQ_M + m]
//!         row  = table[ tableIndex(n_q, m, code, 0) .. + n_q ]  // contiguous
//!         per_k_residual[k] += row                              // SIMD vector-add
//!     for k in 0..n_d:
//!       for i in 0..n_q:
//!         per_token_max[i] = max(per_token_max[i],
//!                                centroid_terms[k][i] + per_k_residual[k][i])
//!
//!   Implement the simpler fused variant first (#19); switch to outer-m if the
//!   latency harness shows the residual loop dominating.
//!
//! Tests planned for #19 (tiny synthetic on a built index):
//!   - 1 doc, n_d=4, n_q=2, hand-built PQ codebooks. Norms = 0 → assert score
//!     equals brute-force `Σ_i max_k ⟨q_i, c_{c_k}⟩` within 1e-4.
//!   - Non-zero norms but PQ codes all 0 → assert linear combination matches
//!     brute-force.
//!   - Determinism: same inputs → byte-identical f32 score.

const std = @import("std");
const constants = @import("../constants.zig");
const gather = @import("gather.zig");

test "constants visible to refine pseudocode" {
    try std.testing.expectEqual(@as(u32, 32), constants.PQ_M);
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(gather.Candidate));
}
