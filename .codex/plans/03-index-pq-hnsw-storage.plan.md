---
name: Index — PQ, HNSW, Inverted Lists, Storage
overview: Hierarchical index — paper §4. Product Quantization (M=32, b=8) on residuals, HNSW (M=32, efc=1500) over centroids, per-centroid inverted lists, on-disk storage with magic+versioning and mmap-friendly layout.
todos:
  - id: pq-train
    content: "src/index/pq.zig — train M=32 codebooks of 256 entries each via per-subspace k-means on a residual sample. f32 end-to-end."
    status: pending
  - id: pq-encode
    content: "PQ.encode(residual) -> [M]u8 codes. Test round-trip reconstruction error on synthetic Gaussians."
    status: pending
  - id: pq-distance-tables
    content: "Cache-optimised distance table layout: [subspace M][centroid 256][query token n_q]. Up to 3.8× per paper §5.3. Document micro-block alignment as paper-gap."
    status: pending
  - id: hnsw-build
    content: "src/index/hnsw.zig — HNSW build with M=32 neighbours, efc=1500. Heuristic neighbour selection. Test recall@10 on a 1k-vector toy dataset."
    status: pending
  - id: hnsw-search
    content: "HNSW.search(query, ef) returning top-k by inner product (centroids are L2-normalised). ef_s = 1.5·κ_c at runtime."
    status: pending
  - id: inverted-lists
    content: "src/index/inverted_list.zig — per-centroid posting list of doc IDs (a doc appears in L_j iff at least one of its tokens was assigned to centroid j). Plain []u32 first; revisit compression in a follow-up todo."
    status: pending
  - id: storage-format
    content: "src/index/storage.zig — on-disk format: header (magic TAC_IDX1, version, params), centroids, HNSW graph, PQ codebooks, inverted lists, per-doc layout [c_1..c_{n_d} | PQ codes]. Save/load round-trip test."
    status: pending
  - id: index-builder
    content: "End-to-end Index.build(token_dump, params) -> Index, that runs TAC, residual extraction, PQ training, HNSW build, inverted-list construction, then writes to disk."
    status: pending
isProject: false
---

# Index — paper §4

Owned by the **indexer** teammate.

## Strict paper defaults

`M = 32, b = 8, M_hnsw = 32, efc = 1500`.

## Per-document on-disk layout (paper §4)

```
doc d:  [centroid_id_1 .. centroid_id_{n_d} : u32  |  PQ_{1,1} .. PQ_{n_d, M} : u8]
```

Centroid IDs first (Pass 1 streams these), then PQ codes (Pass 2). Stream-friendly for the refine phase.

## Paper-gaps to flag

- PQ training corpus sampling rule.
- Inverted list compression (plain or delta+VB / Roaring) — pick simplest first.
- Distance-table micro-block alignment.

## Output

`src/index/storage.zig` exposing `Index.open(path)` and `Index.build(...)`, mmap-backed for query-time loads.
