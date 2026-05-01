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
| `tac.clusterFlat` (parallel) | 4.2 s | comptime-specialized distance kernels ([plan-11](.codex/plans/post-hackathon-11-vec-simd-specialization.plan.md)) — 1.28× over the generic @Vector path |
| residuals + norms (parallel) | 24 ms | |
| **`pq.train` (parallel + work-stealing outer dispatch + comptime-specialized kernels)** | **74.9 s** | atomic-counter outer dispatch ([plan-10](.codex/plans/post-hackathon-10-pq-workstealing.plan.md)) plus comptime-specialized `vec.l2sq` for hot dims ([plan-11](.codex/plans/post-hackathon-11-vec-simd-specialization.plan.md)) — every PQ encode/refit triple `(subspace × centroid × token)` hits `vec.l2sq` on a 2-element subspace; the inline dispatcher lets the call-site `a.len` propagate through the switch and DCE the rest. ~12 % off vs work-stealing-only baseline; byte-equal across n_threads ∈ {1, 4, 10} preserved. |
| `pq.encode` (parallel) | 0.9 s | |
| **`hnsw.build` (parallel, 10 threads)** | **2.7 s** | chunked deferred-commit, paper-strict deterministic CSR ([plan-06](.codex/plans/post-hackathon-06-hnsw-parallel.plan.md)) |
| serialise | 0.15 s | |
| **Build total (10 threads)** | **1 min 23 s** | work-stealing pq.train ([plan-10](.codex/plans/post-hackathon-10-pq-workstealing.plan.md)) + comptime-specialized distance kernels ([plan-11](.codex/plans/post-hackathon-11-vec-simd-specialization.plan.md)). 1.13× over the work-stealing-only baseline (1:34); index byte-equal vs the generic kernel. |
| **Search latency** | **~54 ms / query** | paper-strict single-core, `kappa_c=80, kappa_d=1000` |
| **Index size** | **75.8 MB** | for 26,678 Jira tickets |

Quality smoke on Czech queries:
- `"rozesílání hromadné pošty z eshop domény"` → top hit `jira:ABC-1` (DKIM/SPF newsletter ticket) — matches WARP / `setup-note.md` ground truth.
- 0 docs dropped during encode — pylate's ColBERTv2 path handles all 26k Jira tickets including Czech multilingual content cleanly.

### Optimization story

Build was 47.5 s single-thread on the 2 k chunk; we got it to 12.4 s by parallelising the four embarrassingly-parallel stages (TAC per-token, PQ train per-subspace, PQ encode per-token, residuals per-token). HNSW build was the next lever — at 26 k it was 17.0 s serial; we replaced it with a chunked deferred-commit parallel build (paper-strict deterministic) which lands at **2.8 s on 10 threads (≈6× speedup)**, saving 14 s on full-corpus build wall time. PQ training stayed dominant because M=32 subspaces / 10 threads with a static `chunk = ceil(32/10) = 4` left fast workers idle while slow ones ground out the last subspace in their chunk. **Plan-10** swapped the static chunker for a shared `std.atomic.Value(u32)` work-stealing counter — every worker pulls the next subspace via `fetchAdd(1, .monotonic)` until the index reaches `PQ_M`. Lands at **97.1 s on full-Jira (was 115.8 s, ≈16 % off)**; the gain is bounded because most subspaces converge in similar Lloyd-iter counts, so the slack the static chunker left on the table was real but smaller than the 30-40 % ceiling. **Plan-11** then comptime-specialized the distance kernels (`vec.dot`, `vec.l2sq`, `vec.normalizeInPlace`) for the four dims that actually appear in the codebase — 2/4 (PQ subspaces) and 64/128 (jina-colbert-v2-64, ColBERTv2.0). **Full-Jira build dropped from 1:34 → 1:23 (1.13×, ≈12 % off total wall time), with `pq.train` itself going 84.8 s → 74.9 s.** Index file is byte-equal vs the generic kernel (paper §9 single-thread retrieval semantics unchanged) and the build's determinism contract is: kmeans centroids + PQ codebooks + residual norms are byte-equal across n_threads ∈ {1, 4, 10}; the HNSW graph CSR is byte-equal across n_threads ∈ {2, 4, 10} (the chunked deferred-commit parallel path is deterministic across thread counts ≥ 2); n_threads=1 takes a separate serial HNSW path that produces a *different* deterministic graph, so the full `.tac` image at n_threads=1 is byte-stable to itself across runs but is not byte-equal to a parallel build. Pinned by `tests/integration/build_determinism_test.zig`. Supporting microbench (median of 5, kmeans-assign k=256, the dominant `vec.l2sq` caller in pq.train): dim=2 1.14×, dim=64 1.53×, dim=128 1.42× — the per-call wins are larger than the full-build delta because the kernels also amortize fixed pq.train scaffolding (codebook init, partition counts, Lloyd convergence detection) that didn't get faster. The remaining levers, in order of payoff:

1. **Within-subspace k-means parallelism** — *infrastructure shipped ([plan-07](.codex/plans/post-hackathon-07-within-subspace-kmeans.plan.md))* but inactive at the typical 10-thread invocation: `kmeans.fit` takes `n_threads` and parallelises the per-vector argmin across static chunks while keeping the f32 reduction order serial-in-vector-order, so n_threads=1 ↔ n_threads=10 produce **byte-equal** centroids (paper-strict). At `--threads 10` on M=32 / 10 cores the subspace-level parallelism already saturates; passing more inner threads only oversubscribes. The within-subspace path gates on `n_threads > M`, so it activates when a user runs e.g. `--threads 64`.
2. **FP16 Metal inference + bigger batches in the encoder** — 4-6× on the 48-minute encode; the single biggest absolute win since encode is 95 % of total ingestion cost.

#### Why `pub inline fn` matters for the dispatcher

The plan-11 dispatcher in `src/util/vec.zig` looks innocuous:

```zig
pub inline fn l2sq(a: []const f32, b: []const f32) VecError!f32 {
    if (a.len != b.len) return error.LengthMismatch;
    return switch (a.len) {
        2 => /* hand-inlined scalar */,
        4 | 64 | 128 => l2sqComptime(a.len, a, b),
        else => l2sqGeneric(a, b),
    };
}
```

The `inline` is load-bearing. Without it, `switch (a.len)` is a runtime branch — and dim=2 (the PQ-subspace path that pq.train hits **billions of times**) regressed from **112 ns → 207 ns per kmeans-assign step (1.85× slower)** even with the dim=2 arm picking the comptime kernel. Adding `pub inline fn` lets the call-site's slice length (which pq.train's inner loop knows is constant 2) propagate through the switch; LLVM dead-code-eliminates every other arm and emits two scalar mul-adds. With `inline` the same workload is **98 ns (1.14× faster than the original generic kernel).** Same source, same kernels — only difference is whether the dispatcher can be folded into the caller. This is the kind of win that doesn't show up on a single-call microbench (where `a` and `b` are loop-invariant pointers and LLVM can hoist anything) but dominates real hot loops where the slice length is a compile-time constant at the call site but not at the function definition.

Same lesson applies to the `*Generic` fallbacks: marked `inline fn` so the unspecialized arm is also folded into the caller.

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
