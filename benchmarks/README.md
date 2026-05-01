# Benchmarks (paper §7, §8 reproduction)

Owned by the **retriever** teammate. The reusable sweep core lives in
`src/retrieval/bench.zig` and is exercised on every `zig build test` via a
synthetic-fixture smoke harness. The per-dataset entry points in this
directory are thin executables that load real on-disk data and call into
`tac.retrieval.bench.runSweep`; they're not part of the default `zig build
test` pipeline because they depend on multi-GB encoded corpora.

## Target operating points (paper Table 1)

| Dataset       | Quality          | Tachiom (paper) | Our target |
|---------------|------------------|-----------------|------------|
| MS MARCO-v1   | MRR@10 = 39.0    | 10 ms           | match within margin |
| MS MARCO-v1   | MRR@10 = 39.3    | 14 ms           | match within margin |
| LoTTE-pooled  | Success@5 = 67.5 | 11 ms           | match within margin |

Latency is single-core wall-clock (paper §9). We pin to one core via
`tac.eval.latency.pinToCore` (best-effort per OS).

## Layout

```
src/retrieval/bench.zig    — reusable sweep core (paper §6 grid). All test
                             coverage lives here; imports search/refine/etc.

benchmarks/
  README.md                — this file
  msmarco_v1.zig           — MS MARCO-v1 harness entry (executable)
  lotte_pooled.zig         — LoTTE-pooled harness entry (executable)
  results/                 — written CSVs, untracked (.gitignore)
    YYYY-MM-DD-msmarco.csv
    YYYY-MM-DD-lotte.csv
```

## Sweep grid (paper §6)

- `κ_c ∈ {15, 20, 40, 80, 100, 120}`
- `κ_d ∈ {250, 500, 1000, 2000, 4000}`
- `α ∈ {null, 0.35, 0.40, 0.45, 0.50}`  (null = top-κ_d only)

Total cells: 6 × 5 × 5 = 150 per dataset.

## CSV schema

```
dataset,kappa_c,kappa_d,alpha_x100,n_queries,quality,avg_total_ms,p50_total_ms,p95_total_ms,avg_gather_ms,avg_prune_ms,avg_table_ms,avg_refine_ms,git_sha
```

`alpha_x100` is `round(alpha · 100)` or `-1` for the no-CP cell.

## How to run a real-data sweep

1. Encode the corpus into `tokens.bin` via `tools/encode.py`. See
   `tools/LIVE_ENCODER_NOTES.md` for the ColBERTv2 setup.
2. Encode the queries into a separate `queries.bin` (same format) and load
   qrels.tsv (TREC format).
3. Build the index: `zig build run -Doptimize=ReleaseFast -- index <tokens.bin> <out.tac>`
   (CLI to be wired by lead).
4. Run the harness: `zig build run-bench-msmarco -Doptimize=ReleaseFast -- \
       --index <out.tac> --queries <queries.bin> --qrels <qrels.tsv> \
       --out benchmarks/results/msmarco-$(date +%Y%m%d).csv`

Step 4 is gated on `b.addExecutable` blocks in `build.zig` (lead-owned).

## CI smoke

`zig build test` runs `bench.test "bench: end-to-end smoke ..."` which builds a
100-doc synthetic fixture and runs a single sweep cell, asserting the CSV
schema, latency report finite-ness, and quality range. Real-data cells are
not part of CI — running 150 cells × 6,980 queries × MS MARCO scale needs
hours of single-core compute and is a separate operational task.
