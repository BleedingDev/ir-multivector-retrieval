---
name: TAC Clustering
overview: Token-Aware Clustering — paper §3. K-means++ baseline, then the four-phase pipeline (tail handling, damped scoring, bounding, budget reconciliation) with paper defaults.
todos:
  - id: kmeans-lloyd
    content: "src/tac/kmeans.zig — Lloyd's algorithm with k-means++ init, parameterised by max_iters and tolerance. Tests on Gaussian blobs with hand-known centroids."
    status: pending
  - id: kmeans-balanced
    content: "Mini-kmeans variant that handles tiny n (n < k handled gracefully with single-vector centroids; warns if budget exceeds vector count)."
    status: pending
  - id: tac-tail-phase1
    content: "src/tac/tac.zig phase 1 — partition tokens into micro/small/active by frequency thresholds μ=128, τ=256. Allocate κ_j=1 for micro and κ_j=2 for small."
    status: pending
  - id: tac-damped-phase2
    content: "Phase 2 — for active tokens: compute s_j = (1/n_j)Σ‖t_{j,i}-t̄_j‖², w_j = √n_j · s_j, κ_j = ⌊(w_j/Σw)·B⌋. Test on a vocabulary with known frequency/spread split."
    status: pending
  - id: tac-bounding-phase3
    content: "Phase 3 — enforce κ_j ≥ ε=4 floor and n_j/κ_j ≥ θ=39 cap. Document tie-breaking choice as paper-gap."
    status: pending
  - id: tac-reconcile-phase4
    content: "Phase 4 — redistribute deficit/surplus from rounding+bounds back to active tokens until Σκ_j = κ_total. Document order rule as paper-gap."
    status: pending
  - id: tac-driver
    content: "TAC.cluster(token_dump, kappa_total, params, allocator) -> ClusteringResult{centroids, assignments, kappa_per_token}. End-to-end test on synthetic vocab with known answer."
    status: pending
  - id: tac-quality-vs-kmeans
    content: "Quality regression test: TAC vs vanilla k-means at fixed kappa on synthetic data — TAC should match or beat WCSS for the same budget after reconciliation."
    status: pending
isProject: false
---

# TAC Clustering — paper §3

Owned by the **clusterer** teammate.

## Equations (verbatim from paper)

```
spread:        s_j = (1/n_j) · Σ_{i=1..n_j} ‖t_{j,i} − t̄_j‖²
damped weight: w_j = √(n_j) · s_j
allocation:    κ_j = ⌊(w_j / Σ_i w_i) · B⌋
```

`B` = κ_total − Σ_{tail tokens} κ_j (tail = micro+small).

## Strict paper defaults

`μ = 128, τ = 256, ε = 4, θ = 39`. Configurable but defaults must be those values.

## Paper-gaps to flag

- **Phase 4 redistribution order.** Document the chosen rule (e.g. largest `w_j` first for surplus, smallest `w_j` first for deficit) and ensure determinism.
- **Phase 3 cap interaction with Phase 2 floor.** Resolve overlap deterministically.

## Output

`src/tac/tac.zig` exposing `ClusteringParams`, `ClusteringResult`, `cluster(...)`. Result memory-owned by caller's allocator.
