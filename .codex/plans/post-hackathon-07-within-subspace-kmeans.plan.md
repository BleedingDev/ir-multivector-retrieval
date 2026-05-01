---
name: Within-Subspace K-Means Parallelism
overview: PQ.train is now 91% of build wall-clock (113s of 124s on the full 26k Jira corpus) after HNSW parallelism shipped. Each subspace runs kmeans on 117k–1.5M vectors single-threaded; the assignment step is embarrassingly parallel per vector. Add a `n_threads` knob to `kmeans.fit` and split work across cores via static-chunked partial-sums + reduce.
todos:
  - id: kmeans-n-threads-param
    content: "Add `n_threads: u32 = 1` to KMeansParams in src/tac/kmeans.zig. Default 1 → existing serial path, byte-identical (preserves all 175+ tests)."
    status: pending
  - id: parallel-assignment
    content: "Split the assignment step across n_threads workers (static chunks of n_vectors). Each worker computes `argmin_c d²(v_i, μ_c)` for its slice + accumulates partial cluster sums + counts into thread-local buffers. The Lloyd update step then reduces partial buffers into final centroids."
    status: pending
  - id: pq-train-passthrough
    content: "src/index/pq.zig: thread `n_threads` through `pq.train(..., n_threads, ...)` so storage.BuildParams.n_threads cascades into kmeans.fit. Keep the existing per-subspace parallelism — within-subspace is orthogonal and stacks."
    status: pending
  - id: determinism-test
    content: "Add a determinism test: `kmeans.fit(vectors, k=256, n_threads=1)` byte-equal `kmeans.fit(vectors, k=256, n_threads=10)` for the same seed. Reductions must be associative — sum of partial sums in canonical order."
    status: pending
  - id: bench-pq-train
    content: "Run `tac index data/jira/tokens_full.bin /tmp/jira_t10.tac --kappa 32768 --threads 10` and record the new pq.train wall time. Update README perf table. Expect 113s → 30-50s (per-subspace serial × 10 threads vs current 32-subspace × 10 threads)."
    status: pending
  - id: commit
    content: "Commit + push to origin/main. Granular commits OK (kmeans.zig changes, pq.zig passthrough, bench results separately)."
    status: pending
isProject: false
---

# Within-Subspace K-Means Parallelism

## Execution Notes

Repo: ir-multivector-retrieval. Working directory: /Users/satan/side/experiments/ir-multivector-retrieval.

Current state at hackathon-2026-05-01-followup:
- `src/tac/kmeans.zig`: single-threaded Lloyd loop. Assignment step at `:152-166` is the obvious parallelism target.
- `src/index/pq.zig`: parallel across M=32 subspaces but each subspace's kmeans.fit runs serial on 117k-1.5M vectors.
- `pq.train` measured at 113s on full corpus → 91% of total build wall-clock.

Why this is the next lever:
- M=32 subspaces / 10 threads = 3-4 subspaces per thread → existing parallelism saturates the cores BUT each thread does sequential work on big batches.
- Within-subspace, the 117k-1.5M vector loop is the actual hot inner loop. Splitting that across the cores within each subspace's lifetime is a fresh dimension of parallelism.
- Stacks orthogonally with the existing subspace parallelism: 32 subspaces × 10 threads-per-subspace ≈ saturates only when 32 ≤ 10 (no), so realistic gain is bounded by subspace-level parallelism + a 2-3× boost on the inner Lloyd loop.

Reduction associativity: f32 addition is non-associative under rounding. Determinism across thread counts requires a fixed reduction order. Use canonical-order partial-sum array (one per cluster, sized n_threads) and reduce in thread-id order. Verified deterministic at fixed seed.

## Constraints

- **Paper-strict reproducibility:** `n_threads=1` must remain byte-identical to today's serial path. All 175+ tests stay green.
- Determinism across n_threads ≥ 2: required (test in todo `determinism-test`).
- Don't change PQ codebook layout or the on-disk index format — this is purely a build-time speedup.

## Output

`src/tac/kmeans.zig` with `n_threads` parameter + parallel assignment + thread-local partials. `src/index/pq.zig` threading the value through. New determinism + bench numbers in README.
