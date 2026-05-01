---
name: Primitives and IO
overview: Foundation layer — SIMD-vectorised float ops, deterministic RNG, allocator helpers, binary token-dump format with mmap loader, and the Python ColBERTv2 encoder script.
todos:
  - id: vec-simd
    content: "src/util/vec.zig — @Vector dot product, L2 squared distance, argmin over batch, batched normalise. Tests on tiny inputs with hand-checked values."
    status: pending
  - id: rng-deterministic
    content: "src/util/rng.zig — wrapper over std.Random.DefaultPrng pinned to a u64 seed, with kmeans++ weighted sampler. Test reproducibility under fixed seed."
    status: pending
  - id: alloc-helpers
    content: "src/util/alloc.zig — arena helpers, slice-of-slices builder, alignment helpers for SIMD loads."
    status: pending
  - id: token-dump-format
    content: "src/io/token_dump.zig + docs/token-dump-format.md — flat binary {magic, version, dim, n_tokens, n_docs, dtype, ...} with sidecar metadata. mmap loader returns a TokenDump view."
    status: pending
  - id: encode-py
    content: "tools/encode.py — pylate ColBERTv2 over a docs.jsonl, dumps tokens.bin matching the format spec. Reuse the jina/colbert path proven in sibling ir-expo if helpful for smoke."
    status: pending
  - id: synthetic-fixture
    content: "Generate a small deterministic synthetic token dump for downstream teams to test against. Land under tests/fixtures/synthetic_tokens.zig (a builder, not a binary)."
    status: pending
isProject: false
---

# Primitives and IO

Owned by the **primitives-engineer** teammate.

## Constraints

- All public functions take `std.mem.Allocator` explicitly.
- `f32` end to end.
- SIMD via `@Vector(N, f32)` with N derived from `std.simd.suggestVectorLength(f32)`. Fallback path for sizes that don't divide the vector length.
- All randomness reproducible from a `seed: u64`.
- `TokenDump` magic/version/dtype validated on load; bad files return `error.InvalidTokenDump`.

## Token-dump file format (proposed; finalise in docs/token-dump-format.md)

```
[magic: 8 bytes "TAC_TKN1"]
[version: u32 le]
[dim: u32 le]
[n_docs: u64 le]
[n_tokens: u64 le]
[dtype: u8 (0=f32)]
[reserved: 7 bytes]
[doc_offsets: u64 * (n_docs+1)]   // CSR-style; doc d tokens are [offsets[d]..offsets[d+1])
[token_ids: u32 * n_tokens]       // vocabulary IDs (for TAC token-aware grouping)
[vectors: f32 * n_tokens * dim]
```

Sidecar `tokens.meta.json` records: encoder name, encoder version, doc-id mapping, build timestamp.

## Output

A buildable `src/util/` and `src/io/` that the clusterer / indexer / retriever can depend on.
