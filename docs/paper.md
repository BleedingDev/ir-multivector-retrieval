# Paper: *Efficient Multivector Retrieval with Token-Aware Clustering and Hierarchical Indexing* (Tachiom)

- **arXiv HTML:** https://arxiv.org/html/2604.28142v1
- **Reference impl:** https://github.com/TusKANNy/tachiom (Rust 1.92.0-nightly, kANNolo AVX2 kernels)
- **System name:** Tachiom · **Algorithm name:** TAC (Token-Aware Clustering)
- **Extraction captured:** 2026-05-01 (frozen reference for this repo; re-fetch and diff if updating)

## 1. Headline claims

- 247× faster clustering than FAISS κ-means (AVX2); 84× vs MKL FAISS; 230× vs FastKMeans-rs.
- Clusters 600M vectors → 262K centroids in **8 minutes** on a 64-thread Xeon. Scales to 4M centroids in 102 min. Competitors time out at 131–524K centroids in 24 h.
- 9.8× retrieval speedup over WARP at MS MARCO-v1 MRR@10 = 39.0 (10 ms/query).
- 11 ms/query at LoTTE Success@5 = 67.5.

## 2. Problem setup

ColBERT-style multivector retrieval encodes each document as a *set* of contextualized token embeddings; relevance = sum over query tokens of their max similarity to any document token (MaxSim / late interaction). Two structural pain points:

- **Token frequency imbalance.** Top 100 tokens hold 41–45% of all vectors on MS MARCO-v1 / LoTTE. Global κ-means floods centroids onto common tokens that are low-discriminative; rare domain tokens get starved.
- **κ-means cost dominates index build** at this scale.

## 3. TAC — Token-Aware Clustering

Reformulate clustering as per-token subproblems with a *damped* allocation that balances frequency and intra-token semantic spread.

### 3.1 Setup

For token `j` (vocabulary index), let

- `n_j` — number of occurrences in the corpus
- `t_{j,i}` — `i`-th occurrence embedding (`d`-dim)
- `t̄_j = (1/n_j) · Σ_i t_{j,i}` — token mean

Spread (component-wise variance, paper's symbol `s_j`):

```
s_j  =  (1/n_j) · Σ_{i=1..n_j}  ‖t_{j,i} − t̄_j‖²
```

### 3.2 Damped weight & allocation

```
w_j   = √(n_j) · s_j                              (Eq. damped weight)
κ_j   = ⌊ (w_j / Σ_i w_i) · B ⌋                   (Eq. allocation)
```

`B` is the "active" centroid budget after tail handling. The √n damping is the key: linear in n would just reproduce the imbalance; this gives diminishing returns to frequency.

### 3.3 Four-phase pipeline

**Phase 1 — Tail handling.** Tokens with very low frequency get a fixed allocation:
- Micro: `n_j < μ` → `κ_j = 1`
- Small: `μ ≤ n_j < τ` → `κ_j = 2`
- Active: `n_j ≥ τ` → enter Phase 2
- Defaults: `μ = 128, τ = 256`.

**Phase 2 — Damped scoring.** Compute `w_j` and proportional `κ_j` for active tokens against budget `B = κ_total − Σ_tail κ_j`.

**Phase 3 — Bounding.**
- Floor: `κ_j ≥ ε` (default `ε = 4`).
- Cap: enforce `n_j / κ_j ≥ θ` so each centroid covers at least θ vectors (default `θ = 39`).

**Phase 4 — Budget reconciliation.** Surplus/deficit from rounding + bounds redistributed to land exactly at the global budget `κ`. Then run independent `κ_j`-means per token.

### 3.4 Speedup intuition

Standard κ-means: `O(I · N · κ · d)`.
TAC: `O(I · Σ_j n_j · κ_j · d)`.
Lower bound on speedup:

```
(Σ_j w_j) / max_j w_j
```

Square-root damping both balances allocation and shrinks the dominant-token term that drives the max.

## 4. Index structure

Three layers:

1. **HNSW proximity graph over centroids.** Built once per index. `M = 32` neighbors per node, `efc = 1500` at construction.
2. **Inverted lists at centroids.** `L_j = { d : ∃ token t in d s.t. assign(t)=j }`. Document-level grain (not token-level) — that's what makes the gather phase fast.
3. **PQ-compressed residuals.** Per token vector `t`, store `(c_id, residual − norm)`. Residuals normalized; norms stored separately. Per-document layout interleaves centroid IDs first, then PQ codes:
   ```
   doc d:  [c_1, …, c_{n_d}  |  PQ_{1,1}, …, PQ_{n_d, M}]
   ```
   Defaults: `M = 32` subspaces, `b = 8` bits/code → 32 bytes per token vector.

## 5. Query pipeline

### 5.1 Gather (centroid-only)

For each query token `q_i`:
- HNSW search → top-`κ_c` centroids; `ef_s = 1.5 · κ_c`.
- Walk inverted lists of those centroids, accumulating per-document max:
  ```
  s̃_i(d) = max_{ j : d ∈ L_j }  ⟨q_i, c_j⟩
  ```
- Sum across query tokens:
  ```
  S̃(q, d) = Σ_{i=1..n_q}  s̃_i(d)
  ```

This avoids touching any per-token PQ code in the gather phase.

### 5.2 Pruning

Rank candidates by `S̃`; truncate to top-`κ_d`. Optional adaptive Candidates Pruning (CP) using a fraction `α` of the running maximum (`α ∈ {0.35, 0.4, 0.45, 0.5}`).

### 5.3 Refine

For surviving candidates, compute exact MaxSim using centroids + PQ-decompressed residuals.

**Cache-optimized distance table.** Three-level layout to give every per-token lookup contiguous bytes:
- Macro-blocks indexed by PQ subspace (M = 32 macro-blocks).
- Inside each: blocks indexed by PQ centroid ID (256 blocks for `b = 8`).
- Inside each: micro-blocks of `n_q` consecutive distances (one per query token) — paper claims **up to 3.8×** over standard PQ layout.

Document scoring streams the centroid IDs first (Pass 1), then the PQ codes (Pass 2).

## 6. Hyperparameters

### TAC defaults

| Symbol | Default | Meaning |
|--------|--------:|---------|
| μ      | 128     | micro-token threshold (`κ_j = 1` below this) |
| τ      | 256     | small-token threshold (`κ_j = 2` between μ and τ) |
| ε      | 4       | floor on `κ_j` for active tokens |
| θ      | 39      | min vectors per centroid (cap on `κ_j`) |

### PQ defaults

| Symbol | Default | Meaning |
|--------|--------:|---------|
| M      | 32      | PQ subspaces |
| b      | 8       | bits per code |

### HNSW

| Symbol | Default | Meaning |
|--------|--------:|---------|
| M_hnsw | 32      | edges per node |
| efc    | 1500    | construction-time efSearch |
| ef_s   | 1.5·κ_c | runtime efSearch |

### Retrieval grid

- `κ_c ∈ {15, 20, 40, 80, 100, 120}`
- `κ_d ∈ {250, 500, 1000, 2000, 4000}`
- `α ∈ {0.35, 0.4, 0.45, 0.5}`

## 7. Datasets & evaluation

| Dataset       | #passages | #token vectors | Queries | Metric |
|---------------|----------:|---------------:|--------:|--------|
| MS MARCO-v1   | 8.8M      | 598M           | 6,980 (dev.small) | MRR@10 |
| LoTTE-pooled  | 2.4M      | 266M           | 2,931 (search/dev) | Success@5 |

Encoder: ColBERTv2 (Santhanam et al., 2022b), max 180 tokens/doc on LoTTE.

## 8. Reported results (Table 1, average query time)

| Dataset       | Quality          | Warp     | IGP   | Emvb  | **Tachiom** |
|---------------|------------------|---------:|------:|------:|-------------:|
| MS MARCO-v1   | MRR@10 = 39.0    | 98 ms    | 72 ms | 55 ms | **10 ms** |
| MS MARCO-v1   | MRR@10 = 39.3    | 98 ms    | —     | 156 ms | **14 ms** |
| LoTTE-pooled  | Success@5 = 67.5 | 49 ms    | 48 ms | 54 ms | **11 ms** |

Speedups vs SOTA: 2.5×–9.8× across the operating points. Note Emvb uses JMPQ/OPQ (supervised, more expensive); Tachiom matches it with vanilla PQ.

## 9. Hardware

Intel Xeon Silver 4314 @ 2.40 GHz, 64 threads. Clustering = all 64 threads. Retrieval = single core.

## 10. Open questions / things the paper does *not* spell out

- **No formal pseudocode** for TAC or the gather/refine inner loops — they're described narratively. Reimplementations should derive pseudocode and cross-check against the reference Rust impl.
- **HNSW construction time** not reported separately.
- **Total memory footprint** not quantified explicitly.
- **Budget reconciliation rule** in Phase 4 is described qualitatively; exact tie-breaking is left to the implementation.
- **Adaptive CP** with `α` — the running maximum used for thresholding is described in the operating-point table but the precise update rule isn't formalized in the extracted text.

These gaps are tracked in `.codex/plans/06-reference-impl-comparison.plan.md`.

## 11. Notation cheat sheet

See `docs/notation.md`. Use the same letters in code (`kappa_c`, `mu`, etc.).
