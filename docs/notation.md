# Notation

Single source of truth for symbols used across the paper, code, and docs. Match the paper letter-for-letter so equations are easy to cross-reference.

## Token-level

| Symbol  | Code identifier | Meaning |
|---------|-----------------|---------|
| `t_{j,i}` | `token_vec`     | the *i*-th embedding of vocabulary token *j* |
| `n_j`     | `n_j`           | corpus frequency of token *j* |
| `t̄_j`    | `t_bar_j`       | mean embedding of token *j* |
| `s_j`     | `s_j`           | spread (component-wise variance) of token *j* |
| `w_j`     | `w_j`           | damped weight `√(n_j) · s_j` |
| `N_T`     | `n_tokens`      | distinct token count in vocabulary |
| `d`       | `dim`           | embedding dimensionality |

## Clustering

| Symbol   | Code identifier | Default | Meaning |
|----------|-----------------|--------:|---------|
| `κ`      | `kappa`         |       — | total centroid budget |
| `B`      | `budget_active` |       — | centroid budget for active tokens (after tail handling) |
| `κ_j`    | `kappa_j`       |       — | centroids assigned to token *j* |
| `c_k`    | `centroid`      |       — | the *k*-th centroid |
| `μ`      | `mu`            |     128 | micro-token threshold |
| `τ`      | `tau`           |     256 | small-token threshold |
| `ε`      | `epsilon`       |       4 | floor on `κ_j` for active tokens |
| `θ`      | `theta`         |      39 | min vectors per centroid |

## Index

| Symbol   | Code identifier | Default | Meaning |
|----------|-----------------|--------:|---------|
| `L_j`    | `inverted_list` |       — | doc IDs whose tokens land in centroid *j* |
| `M`      | `pq_M`          |      32 | PQ subspaces |
| `b`      | `pq_bits`       |       8 | bits per PQ code |
| `M_hnsw` | `hnsw_m`        |      32 | HNSW edges per node |
| `efc`    | `hnsw_efc`      |    1500 | HNSW construction-time efSearch |

## Query / retrieval

| Symbol   | Code identifier | Range | Meaning |
|----------|-----------------|-------|---------|
| `q_i`    | `query_vec`     | —     | *i*-th query token embedding |
| `n_q`    | `n_q`           | —     | query length in tokens |
| `κ_c`    | `kappa_c`       | {15,20,40,80,100,120} | centroids retrieved per query token |
| `κ_d`    | `kappa_d`       | {250,500,1000,2000,4000} | document candidates after pruning |
| `α`      | `alpha`         | {0.35,0.4,0.45,0.5} | adaptive pruning threshold |
| `ef_s`   | `hnsw_ef_s`     | `1.5·κ_c` | HNSW runtime efSearch |
| `s̃_i(d)` | `s_tilde_i_d`  | —     | per-token approximate similarity |
| `S̃(q,d)`| `S_tilde`       | —     | gather-phase aggregate score |

## Conventions

- All inner products `⟨q, c⟩` assume L2-normalized vectors (cosine similarity), matching ColBERT.
- Residuals are normalized before PQ training; their norms are stored separately and re-applied at decode.
- Floats are `float32` end-to-end unless explicitly noted.
