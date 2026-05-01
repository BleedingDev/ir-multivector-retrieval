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

Initial benchmark on a 2,000-doc Jira corpus (Czech), `jinaai/jina-colbert-v2-64`, dim=64, 117,645 token vectors, kappa=10240:

| Build phase       | 1 thread | 10 threads | Speedup |
|-------------------|---------:|-----------:|--------:|
| `tac.clusterFlat` |    218 ms |     164 ms | 1.3× |
| residuals + norms |      9 ms |       9 ms | —    |
| **`pq.train`**    |  42,575 ms |  **7,605 ms** | **5.6×** |
| `pq.encode`       |    820 ms |     827 ms | —    |
| `hnsw.build`      |  3,838 ms |   3,790 ms | —    |
| serialise         |     16 ms |      16 ms | —    |
| **Total**         |  **47,477 ms** |  **12,413 ms** | **3.8×** |

Search latency (paper-strict single-core): **~27 ms / query** at `kappa_c=80, kappa_d=1000` on the same index. Quality smoke: top hit on `"Jak nastavit DKIM a SPF pro newslettery z jiné domény?"` is `jira:ABC-1` (the matching Jira ticket), agreeing with the WARP baseline at `ir-expo/services/warp-service`.

PQ training was the dominant bottleneck (90% of single-thread build); the M=32 subspaces are independent so static-chunk parallelism scales nearly linearly to ~32 threads. HNSW build is the next target — currently single-threaded.

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
