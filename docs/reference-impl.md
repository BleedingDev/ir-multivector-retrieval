# Reference Implementation Comparison

## Status: reference impl not yet public

As of 2026-05-01, the authors' Rust implementation referenced in the paper is **not yet published**. We work strictly from the paper text in `docs/paper.md`.

When the reference impl is released:

1. Clone into `third_party/tachiom/` (gitignored).
2. Run our Zig pipeline and the reference impl on the same MS MARCO-v1 / LoTTE encodings.
3. Validate per the table below.

## Numeric verification plan (when ref impl lands)

For each operating point in Table 1, freeze `(κ_c, κ_d, α)` and compare:

1. **Final MRR@10 / Success@5** — within ±0.1.
2. **Top-100 candidate set overlap** — ≥ 95%.
3. **Final top-10 ranking Kendall τ** — > 0.95.
4. **Per-stage latency breakdown** — order of magnitude is acceptable; absolute parity not expected from a Zig vs Rust+AVX2 comparison until we add hand-tuned SIMD.

## Open paper-gaps to settle by ref-impl inspection

Tracked here so we can resolve them once source is available:

- **Phase 4 reconciliation order.** When rounding leaves a deficit/surplus vs `κ`, what order are centroids handed out / taken back? Largest `w_j` first, round-robin, fractional remainder, …?
- **CP threshold update.** Exact formula for the running maximum and where `α` is applied during candidate accumulation in the gather phase.
- **PQ training corpus.** Trained on residuals from a sample (sampling rule?) or all residuals?
- **HNSW search ef.** `ef_s = 1.5 · κ_c` — floored or ceiled? Per-token or per-query?
- **Distance-table micro-block alignment.** Is `n_q` padded to a fixed alignment for SIMD?
- **Inverted list compression.** Plain `[]u32` or compressed (delta + variable-byte / Roaring)?
- **Single-core retrieval.** Paper says retrieval is single-core for benchmarking — the reference impl may still parallelize internally.

Until the reference is public, these are documented as `// paper-gap:` comments at the relevant call sites.
