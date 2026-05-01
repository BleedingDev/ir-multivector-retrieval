---
name: HNSW Parallel Insertion (paper-strict deterministic)
overview: HNSW build is currently the only single-threaded stage of the Tachiom Zig pipeline (~17s on 26k Jira corpus, ~30% of build wall time after PQ.train parallelism). Per-insert work is mostly read-only (greedySearchLayer, searchLayer), with a small mutating tail (q_neighbours/r_neighbours appends, shrinkNeighbours). Parallelize via layer-batched deferred-commit so paper-strict determinism (identical seed → identical CSR) is preserved.
todos:
  - id: profile-current-build
    content: "Add per-iteration timing to hnsw.build() (only enabled with verbose=true on BuildParams) — split greedy descent vs searchLayer vs heuristic vs connect. Quantifies which phase dominates and confirms the parallelism opportunity."
    status: pending
  - id: design-deferred-commit
    content: "Write a design comment at top of hnsw.zig describing the layer-batched deferred-commit approach: collect (q, chosen[], r_mutations[]) per chunk in parallel using read-only adj snapshots, then commit serially in q-order. Document why per-node spinlocks are NOT used (would break determinism)."
    status: pending
  - id: implement-batched-build
    content: "Refactor hnsw.build()'s outer while-loop to process q in chunks of (e.g.) 256 nodes. For each chunk: spawn n_threads workers each handling a slice; each worker computes greedy descent + searchLayer + selectHeuristic on read-only adj; results collected into a per-q TempInsertion struct. Then a single-threaded commit pass applies them in q-order via the existing append + shrinkNeighbours code."
    status: pending
  - id: determinism-test
    content: "Add a test: build same corpus twice (different n_threads, same seed) and assert byte-equal serialised CSR. This is the paper-strict reproducibility invariant from hnsw.zig:184-185."
    status: pending
  - id: bench-sweep
    content: "Bench full 26k Jira corpus build at n_threads ∈ {1, 4, 10}; expect HNSW phase to drop from 17s → ~5-7s on 10 threads (35-50% off). Update README performance table."
    status: pending
  - id: commit
    content: "Commit + push to main on github.com/BleedingDev/ir-multivector-retrieval. Include benchmark numbers in commit message."
    status: pending
isProject: false
---

# HNSW Parallel Insertion (paper-strict deterministic)

## Execution Notes

Repo: ir-multivector-retrieval (sibling to ir-expo). All work in src/index/hnsw.zig + src/index/storage.zig (only the BuildParams schema change in storage.zig).

Current state at tag hackathon-2026-05-01 (HEAD 48ce549):
- src/index/hnsw.zig:226-330 — sequential while-loop, one insert per iteration.
- Per-iteration phases:
  - **Read-only on adj:** greedySearchLayer (line 234-241), searchLayer (line 254-262, uses generation-counter on visited_gen — not yet thread-safe; needs per-worker copy)
  - **Mutating on adj:** q_neighbours.appendAssumeCapacity (line 308-310), r_neighbours.append + shrinkNeighbours (line 313-322)
- Doc string at line 184-185 promises: "Determinism: identical (centroids, dim, seed, params) → identical CSR."

Two parallelism approaches considered:

(a) Per-node spinlocks on neighbour lists. Simplest, lowest overhead, but **breaks determinism** — `r_neighbours.append(q)` order becomes non-deterministic across runs. Rejected.

(b) **Layer-batched deferred-commit** (chosen): collect insertion deltas per chunk in parallel using read-only adj snapshots, then commit serially in q-order. Preserves the determinism invariant. Estimated overhead: ~5-10% from the staging buffers vs the speedup gain.

Worker scratch: each thread needs its own `visited_gen` buffer (currently shared across iterations at line 218-220). Bumping that to per-worker is a minor allocation change.

Estimated speedup on 26k Jira corpus (10 cores):
- Current HNSW phase: ~17s of ~135s total build (12.6%)
- Expected after parallel: ~5-7s of ~123-125s total (4-6%)
- Total build time: 2:15 → ~2:00 (~10% wall-time savings)

## Constraints

- **Paper-strict reproducibility** — the test in todo `determinism-test` is the gate. If different n_threads produce different CSR for same seed, ship is blocked.
- All 175 existing Zig tests must remain green.
- No external Zig dependencies — `std.Thread.spawn` static-chunk pattern matches what's already used in src/tac/tac.zig and src/index/pq.zig.
- Default behavior unchanged: BuildParams.n_threads=1 → identical to today's serial path.

## Output

src/index/hnsw.zig with parallel-aware build() that respects the determinism invariant. Updated benchmark numbers in README.md. Committed and pushed to origin/main on the publication-track repo.
