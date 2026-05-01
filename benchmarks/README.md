# Benchmarks (paper §7, §8 reproduction)

Owned by the **retriever** teammate. These executables are wired by the lead into
`build.zig` as separate test/run steps once #20 is unblocked; until then the
files here are scaffolding.

## Target operating points (paper Table 1)

| Dataset       | Quality          | Tachiom (paper) | Our target |
|---------------|------------------|-----------------|------------|
| MS MARCO-v1   | MRR@10 = 39.0    | 10 ms           | match within margin |
| MS MARCO-v1   | MRR@10 = 39.3    | 14 ms           | match within margin |
| LoTTE-pooled  | Success@5 = 67.5 | 11 ms           | match within margin |

Latency is single-core wall-clock (paper §9). We pin to one core via
`tac.eval.latency.pinToCore`.

## Layout

```
benchmarks/
  README.md                 — this file
  msmarco_v1.zig            — MS MARCO-v1 harness (sweep κ_c × κ_d × α)
  lotte_pooled.zig          — LoTTE-pooled harness
  common/
    qrels.zig               — qrels file parsers (TREC/MS MARCO formats)
    queries.zig             — encoded query loader (.bin + sidecar)
    sweep.zig               — grid sweep driver: emits CSV per (κ_c, κ_d, α)
  results/
    YYYY-MM-DD-msmarco.csv  — one row per operating point
    YYYY-MM-DD-lotte.csv
    README.md               — how to read the CSVs
```

## Sweep grid (paper §6)

- `κ_c ∈ {15, 20, 40, 80, 100, 120}`
- `κ_d ∈ {250, 500, 1000, 2000, 4000}`
- `α ∈ {null, 0.35, 0.40, 0.45, 0.50}`  (null = top-κ_d only)

Total cells: 6 × 5 × 5 = 150 per dataset. Each cell runs all queries (6,980 for
MS MARCO-v1 dev.small, 2,931 for LoTTE search/dev) and reports
(MRR@10 or Success@5, avg latency, p50, p95).

## CSV schema

```
dataset,kappa_c,kappa_d,alpha,n_queries,quality,avg_total_ms,p50_total_ms,p95_total_ms,avg_gather_ms,avg_prune_ms,avg_table_ms,avg_refine_ms,git_sha,host
```

## Inputs (not in repo)

The harnesses expect:
- `<dataset>.tac.idx` — built via `tac index ...` (lead-owned CLI; once #15 lands).
- `<dataset>.queries.bin` + `.qrels.tsv` — produced by `tools/encode.py` (#5).

A synthetic 100-doc smoke harness will live alongside the real ones for CI.
