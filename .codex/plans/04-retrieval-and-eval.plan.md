---
name: Retrieval and Evaluation
overview: Gather/prune/refine pipeline (paper §5) + MRR@10 / Success@k evaluation harness + latency benchmarks reproducing Table 1 operating points.
todos:
  - id: gather-phase
    content: "src/retrieval/gather.zig — for each query token: HNSW top-κ_c centroids (ef_s = 1.5·κ_c), walk inverted lists, accumulate s̃_i(d) = max_j ⟨q_i, c_j⟩. Aggregate S̃(q,d) = Σ s̃_i(d)."
    status: pending
  - id: prune-phase
    content: "src/retrieval/prune.zig — top-κ_d after S̃ ranking + adaptive Candidates Pruning with α ∈ {0.35,0.4,0.45,0.5}. Document exact CP update as paper-gap."
    status: pending
  - id: refine-phase
    content: "src/retrieval/refine.zig — exact MaxSim using centroids (Pass 1) + PQ-decompressed residuals (Pass 2) with cache-optimised distance tables."
    status: pending
  - id: search-driver
    content: "Retrieval.search(index, query_tokens, params) -> [top_k]ScoredDoc. Stitches gather → prune → refine."
    status: pending
  - id: metrics
    content: "src/eval/metrics.zig — MRR@10, Success@k. Tests on tiny held-out qrels."
    status: pending
  - id: latency-bench
    content: "src/eval/latency.zig — per-stage timing (gather/prune/refine), single-thread pinning. Report avg ms/query."
    status: pending
  - id: msmarco-harness
    content: "benchmarks/msmarco_v1.zig — load queries + qrels, sweep (κ_c, κ_d, α), report MRR@10 and avg latency. Target operating points: MRR@10=39.0 (10ms) and 39.3 (14ms)."
    status: pending
  - id: lotte-harness
    content: "benchmarks/lotte_pooled.zig — Success@5 target 67.5 at 11ms."
    status: pending
isProject: false
---

# Retrieval and Evaluation — paper §5, §7

Owned by the **retriever** teammate.

## Strict paper defaults

Retrieval grid: `κ_c ∈ {15,20,40,80,100,120}`, `κ_d ∈ {250,500,1000,2000,4000}`, `α ∈ {0.35,0.4,0.45,0.5}`. `ef_s = 1.5 · κ_c`.

## Paper-gaps to flag

- Exact CP update rule.
- Whether `ef_s` is per-token or per-query.

## Output

A `tac search` and `tac eval` CLI path (lead wires into `src/main.zig`) and a benchmark report under `benchmarks/results/`.
