# ir-multivector-retrieval

A from-paper reimplementation of **Tachiom** — *Efficient Multivector Retrieval with Token-Aware Clustering and Hierarchical Indexing* — in **Zig** for maximum single-core throughput.

- **Paper:** https://arxiv.org/html/2604.28142v1
- **Tweet that sparked this:** https://x.com/lateinteraction/status/2050055160634708328
- **Authors' Rust impl:** **not yet published** as of 2026-05-01. We work strictly from the paper.

## Targets to reproduce (Table 1, single-core retrieval)

| Dataset       | Quality target  | Tachiom reported | Our impl |
| ------------- | --------------- | ---------------: | -------- |
| MS MARCO-v1   | MRR@10 = 39.0   | 10 ms / query    | TBD      |
| MS MARCO-v1   | MRR@10 = 39.3   | 14 ms / query    | TBD      |
| LoTTE-pooled  | Success@5 = 67.5 | 11 ms / query   | TBD      |

Clustering target: 600M token vectors → 262K centroids in ~8 minutes on a 64-thread Xeon.

## Measured performance (Apple Silicon, 10 cores)

Two corpora on a real Czech Jira ticket archive, `jinaai/jina-colbert-v2-64`, dim=64.

### 2,000-doc chunk (117,645 token vectors, kappa=10,240, θ=10 paper-relax)

| Build phase       | 1 thread | 10 threads | Speedup |
|-------------------|---------:|-----------:|--------:|
| `tac.clusterFlat` |    218 ms |     321 ms | —    |
| residuals + norms |      9 ms |       4 ms | 2.4× |
| **`pq.train`**    |  42,575 ms |  **9,011 ms** | **4.7×** |
| `pq.encode`       |    820 ms |     296 ms | 2.8× |
| **`hnsw.build`**  |  3,838 ms |   **867 ms** | **4.4×** |
| serialise         |     16 ms |      24 ms | —    |
| **Total**         |  **47,477 ms** |  **10,527 ms** | **4.5×** |

### Full 26,678-doc corpus (1,521,797 token vectors, kappa=32,768, paper-strict μ/τ/ε/θ)

19,221 distinct token IDs < 39,020 paper-strict max → no relaxation needed.

| Phase | Time | Notes |
|---|---:|---|
| Metal encode (pylate, fp32, batch=64) | **48 min** | GPU-bound; FP16 + larger batch is 4-6× ahead |
| `tac.clusterFlat` (parallel) | 5.6 s | |
| residuals + norms (parallel) | 29 ms | |
| **`pq.train` (parallel + work-stealing outer dispatch)** | **97.1 s** | atomic-counter outer dispatch ([plan-10](.codex/plans/post-hackathon-10-pq-workstealing.plan.md)) — idle workers pull the next subspace from a shared `std.atomic.Value(u32)` instead of being assigned a static chunk of 4. Saves 18.7 s (~16 %) over the static-chunk baseline at `--threads 10`. Still paper-strict deterministic: codebook layout is m-indexed, per-subspace seed is `base_seed +% m`, byte-equal across n_threads ∈ {1, 4, 10}. |
| `pq.encode` (parallel) | 2.2 s | |
| **`hnsw.build` (parallel, 10 threads)** | **2.8 s** | chunked deferred-commit, paper-strict deterministic CSR ([plan-06](.codex/plans/post-hackathon-06-hnsw-parallel.plan.md)) |
| serialise | 0.37 s | |
| **Build total (10 threads)** | **1 min 48 s** | work-stealing pq.train ([plan-10](.codex/plans/post-hackathon-10-pq-workstealing.plan.md)) closes the static-chunker idle gap; within-subspace inner parallelism still gated to `n_threads > M` |
| **Search latency** | **~54 ms / query** | paper-strict single-core, `kappa_c=80, kappa_d=1000` |
| **Index size** | **75.8 MB** | for 26,678 Jira tickets |

Quality smoke on Czech queries:
- `"rozesílání hromadné pošty z eshop domény"` → top hit `jira:ABC-1` (DKIM/SPF newsletter ticket) — matches WARP / `setup-note.md` ground truth.
- 0 docs dropped during encode — pylate's ColBERTv2 path handles all 26k Jira tickets including Czech multilingual content cleanly.

### Optimization story

Build was 47.5 s single-thread on the 2 k chunk; we got it to 12.4 s by parallelising the four embarrassingly-parallel stages (TAC per-token, PQ train per-subspace, PQ encode per-token, residuals per-token). HNSW build was the next lever — at 26 k it was 17.0 s serial; we replaced it with a chunked deferred-commit parallel build (paper-strict deterministic) which lands at **2.8 s on 10 threads (≈6× speedup)**, saving 14 s on full-corpus build wall time. PQ training stayed dominant because M=32 subspaces / 10 threads with a static `chunk = ceil(32/10) = 4` left fast workers idle while slow ones ground out the last subspace in their chunk. **Plan-10** swapped the static chunker for a shared `std.atomic.Value(u32)` work-stealing counter — every worker pulls the next subspace via `fetchAdd(1, .monotonic)` until the index reaches `PQ_M`. Lands at **97.1 s on full-Jira (was 115.8 s, ≈16 % off)**; the gain is bounded because most subspaces converge in similar Lloyd-iter counts, so the slack the static chunker left on the table was real but smaller than the 30-40 % ceiling. Paper-strict reproducibility preserved (codebook layout m-indexed, seed `base_seed +% m`, byte-equal across n_threads ∈ {1, 4, 10}). The remaining levers, in order of payoff:

1. **Within-subspace k-means parallelism** — *infrastructure shipped ([plan-07](.codex/plans/post-hackathon-07-within-subspace-kmeans.plan.md))* but inactive at the typical 10-thread invocation: `kmeans.fit` takes `n_threads` and parallelises the per-vector argmin across static chunks while keeping the f32 reduction order serial-in-vector-order, so n_threads=1 ↔ n_threads=10 produce **byte-equal** centroids (paper-strict). At `--threads 10` on M=32 / 10 cores the subspace-level parallelism already saturates; passing more inner threads only oversubscribes. The within-subspace path gates on `n_threads > M`, so it activates when a user runs e.g. `--threads 64`.
2. **FP16 Metal inference + bigger batches in the encoder** — 4-6× on the 48-minute encode; the single biggest absolute win since encode is 95 % of total ingestion cost.

## Why Zig

- Single-core retrieval target → no GC pauses, predictable allocation, manual SIMD via `@Vector(N, T)`.
- `comptime` for codegen-style specialization (PQ subspace count, HNSW edge count, distance kernels per dim).
- Direct mmap of binary index files — zero-copy load.
- Reference impl (Rust + AVX2) is unpublished; we implement strictly from paper text.

## Strict-paper rule

When the paper specifies a parameter, **use that value exactly**. Defaults:

- TAC: `μ=128, τ=256, ε=4, θ=39`
- PQ: `M=32 subspaces, b=8 bits`
- HNSW: `M_hnsw=32, efc=1500, ef_s=1.5·κ_c`
- Retrieval grid: `κ_c ∈ {15,20,40,80,100,120}`, `κ_d ∈ {250,500,1000,2000,4000}`, `α ∈ {0.35,0.4,0.45,0.5}`

Where the paper is silent (e.g. tie-breaking in Phase 4 reconciliation, exact CP update rule), each implementation choice is documented in the source comment block beside it as a `# paper-gap:` note.

## Workflow

- **Push directly to `main`** once `zig build test` is green. No PR review for this repo.
- Commit progressively per component (k-means → TAC → PQ → HNSW → gather → refine → eval).
- Numerical correctness validated on synthetic datasets in unit tests before each commit.

## Repo layout

```
ir-multivector-retrieval/
├── README.md
├── AGENTS.md                — context for AI coding agents
├── docs/
│   ├── paper.md             — full extracted paper notes (frozen ref)
│   ├── notation.md          — symbols & their Zig identifiers
│   └── reference-impl.md    — comparison plan (post-publication)
├── build.zig
├── build.zig.zon
├── src/
│   ├── main.zig             — CLI: index | search | eval | bench
│   ├── root.zig             — public module surface
│   ├── util/                — SIMD primitives, RNG, allocator helpers
│   ├── tac/                 — k-means + TAC four-phase pipeline (paper §3)
│   ├── index/               — HNSW + inverted lists + PQ residuals (paper §4)
│   ├── retrieval/           — gather / prune / refine (paper §5)
│   ├── eval/                — MRR@10, Success@k, latency harness
│   └── io/                  — binary token-dump format, mmap loader
├── tools/
│   └── encode.py            — Python-side ColBERTv2 → flat binary dump
├── tests/                   — integration tests on tiny synthetic data
├── benchmarks/              — Table 1 reproduction scripts
└── .codex/plans/            — plan-graph .plan.md files
```

## Build & run

```bash
zig build                          # release build
zig build test                     # unit + integration tests
zig build run -- index --help      # CLI help
```

## Encoder pipeline

ColBERTv2 inference stays in Python (pylate). It dumps a flat binary `tokens.bin + tokens.meta.json` that Zig mmaps. See `tools/encode.py` and `src/io/token_dump.zig`.

## Status

Bootstrap complete. Plan files in `.codex/plans/` (00 root, 01–09 lanes) decompose the work. Implementation starts at the lowest layer (`src/util`, `src/tac/kmeans.zig`).
