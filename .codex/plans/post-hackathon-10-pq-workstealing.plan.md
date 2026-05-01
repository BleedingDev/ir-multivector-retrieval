---
name: Work-Stealing Outer pq.train Dispatch
overview: Plan 07 added within-subspace kmeans parallelism but didn't help at default --threads 10 because the outer subspace chunker uses static chunks. With M=32 subspaces / 10 threads = chunk 4, threads finishing their cheap subspaces idle while threads stuck on hard ones grind on. Replace static chunking with a shared atomic counter so any idle thread pulls the next available subspace. Real next perf lever; ~10 lines plus a test.
todos:
  - id: locate-static-chunker
    content: "Re-read src/index/pq.zig parallel block around line 237. Confirm the current shape: each worker thread is given a contiguous [lo, hi) range of subspace indices and processes them sequentially. Identify the smallest atomic refactor."
    status: pending
  - id: atomic-counter-impl
    content: "Replace the per-thread (lo, hi) range with a shared `next_subspace: std.atomic.Value(u32)` initialized to 0. Each worker loops: `m = atomic.fetchAdd(1)`; if `m >= constants.PQ_M` then break; else process subspace m. The same per-worker scratch (sub_buf, kmeans_inner) stays."
    status: pending
  - id: determinism-stays-paper-strict
    content: "Determinism check: subspace order in `codebooks` is fixed by m (not by completion order), so worker scheduling can't observe it. The kmeans.fit per subspace is already deterministic via seed = base_seed +% m. Net: byte-equal codebooks across n_threads ∈ {1, 4, 10}. Add a test that asserts this for the parallel pq.train path explicitly."
    status: pending
  - id: bench
    content: "Run `tac index data/jira/tokens_full.bin /tmp/jira_workstealing.tac --kappa 32768 --threads 10` and record the new pq.train wall time. Expect 113s baseline → 65-80s (30-40% off). If gain is below 15%, the bottleneck is elsewhere (kmeans iter count variance per subspace probably small) and we ship as-is plus a paper-gap note."
    status: pending
  - id: readme-update
    content: "Update README perf table: pq.train new number, add to Optimization story section. If headline-worthy, update the 26k corpus total row too."
    status: pending
  - id: commit
    content: "Commit + push to origin/main. Single commit OK since it's a small change. Use git author Petr Glaser <petr@glaser.cz>."
    status: pending
isProject: false
---

# Work-Stealing Outer pq.train Dispatch

## Execution Notes

Repo: ir-multivector-retrieval. Working directory: /Users/satan/side/experiments/ir-multivector-retrieval.

This is plan-07's diagnosed-but-deferred follow-up. The kmeans-vector engineer's wrap-up message named it explicitly:

> The actual remaining lever (callout in README): replace the static outer chunking in src/index/pq.zig with a shared atomic counter for work-stealing — that would let all 10 outer threads pull the next available subspace and saturate cores. That's a ~10-line change inside the existing parallel block.

Why static chunking under-saturates:
- M=32 subspaces / n_threads=10 → ceil(32/10) = 4 subspaces per worker.
- Each kmeans.fit converges in 5-25 Lloyd iterations; convergence count varies per subspace (depends on residual distribution).
- A worker that lucks into 4 fast subspaces finishes ~40% earlier than one stuck with 4 slow ones. Total wall = max-lane wall, not avg.
- Atomic counter lets a worker that finished early grab the next pending subspace from a slower lane.

## Constraints

- Paper-strict reproducibility preserved: codebook layout indexed by m (subspace), not by worker completion order.
- All 188 existing tests must remain green.
- n_threads=1 path unchanged (atomic counter trivially serializes if only one thread).
- Don't change kmeans.fit's signature or PQ codebook layout on disk.

## Output

src/index/pq.zig with the static chunker replaced by atomic-counter work-stealing. New determinism test asserting byte-equal pq.train output across n_threads ∈ {1, 4, 10}. README perf row updated.
