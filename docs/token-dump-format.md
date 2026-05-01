# Token-dump file format (`tokens.bin`)

Status: **frozen** for index version 2.

Version history:
- v1 (deprecated, rejected by the loader): `token_ids` were *positional*
  per-doc indices (0..n_tok). This silently degenerated paper §3 TAC
  bucketing and is no longer accepted.
- v2 (current): `token_ids` are real BERT vocabulary IDs from the
  tokenizer, with the ColBERT [D] prefix preserved and the model's
  skiplist (punctuation by default) dropped — matching what
  `pylate.models.ColBERT.encode(..., is_query=False, normalize=True)`
  keeps.

This file is the bridge between the Python ColBERTv2 encoder
(`tools/encode.py`, owned by primitives-engineer) and the Zig pipeline
(`src/io/token_dump.zig`). It carries every token embedding for the corpus
in a single flat, mmap-friendly blob. A sidecar JSON file holds
human-readable metadata that we never read at index-build time but want for
provenance.

All multi-byte integers are **little-endian**. Floats are **IEEE-754
binary32** (`f32`), little-endian.

## Layout

```
offset    size                   field
0         8                      magic           = "TAC_TKN1" (ASCII)
8         4                      version         = u32, currently 2
12        4                      dim             = u32, embedding dimensionality
16        8                      n_docs          = u64, number of documents
24        8                      n_tokens        = u64, total token vectors
32        1                      dtype           = u8, 0 = f32 (only value supported in v1)
33        7                      reserved        = zero bytes (padding to 8-byte alignment)
40        8 * (n_docs + 1)       doc_offsets     = u64[], CSR offsets into token_ids/vectors
?         4 * n_tokens           token_ids       = u32[], vocabulary IDs (one per token vector)
?         4 * n_tokens * dim     vectors         = f32[], row-major n_tokens × dim
```

`?` denotes an offset computed from the preceding fields. Vectors start at
the first multiple of `4` after `token_ids` ends; since both `token_ids`
length (`4 * n_tokens` bytes) and the position of `token_ids`
(`40 + 8*(n_docs+1)` bytes) are multiples of 4, no padding is required.

## CSR semantics

`doc_offsets[d]` is the index into `token_ids` / `vectors` of the first
token belonging to document `d`. The tokens of document `d` are at indices
`doc_offsets[d]..doc_offsets[d+1]`. By definition `doc_offsets[0] = 0` and
`doc_offsets[n_docs] = n_tokens`.

To recover the embedding of the `k`-th token of document `d`:

```
let i = doc_offsets[d] + k         // 0 <= k < doc_offsets[d+1] - doc_offsets[d]
let vid = token_ids[i]             // vocabulary ID
let emb = vectors[i*dim .. (i+1)*dim]
```

## Validation rules (Zig loader, `error.InvalidTokenDump` on any failure)

- File length matches header sizes exactly. Truncated or oversized files
  are rejected.
- `magic == "TAC_TKN1"` byte for byte.
- `version == 2`. v1 files (positional ids) are explicitly rejected.
- `dim > 0` and `dim <= 4096` (sanity ceiling — current ColBERTv2 outputs
  128-d).
- `n_tokens > 0`. Empty corpora have nothing for downstream code to do.
- `dtype == 0` (only `f32` is supported in v1).
- Every reserved byte equals `0`.
- `doc_offsets[0] == 0`, `doc_offsets[n_docs] == n_tokens`.
- `doc_offsets` is monotonically non-decreasing.
- All `n_tokens` entries in `vectors` are finite (no NaN, no ±Inf).

The loader does **not** verify L2 normalisation — that is the encoder's
responsibility.

## Sidecar metadata

Alongside `tokens.bin` we write `tokens.meta.json`. The Zig side never
reads this; it exists purely so a human can audit which encoder produced
a dump.

```json
{
  "format_version": 2,
  "encoder": "colbert-ir/colbertv2.0",
  "encoder_revision": "<hf revision hash>",
  "encoder_dim": 128,
  "n_docs": 1234,
  "n_tokens": 567890,
  "doc_id_map": ["doc-0", "doc-1", "..."],
  "built_at": "2026-05-01T12:34:56Z",
  "tool": "tools/encode.py",
  "tool_version": "0.2"
}
```

`doc_id_map` is indexed by integer doc-id used in `tokens.bin`; if a row
maps back to a string identifier from `docs.jsonl`, this is where we
recover it.

## Versioning

Increment `version` on any breaking change to the binary layout. The Zig
loader pins `TOKEN_DUMP_VERSION` in `src/constants.zig`; bumping it
requires the loader to grow a forward-compatible branch.

## Why this design

- Flat / mmap-able: the clusterer needs O(1) random access to per-token
  rows, and HNSW build needs to hand `[]const f32` views to the SIMD
  primitives without copying.
- CSR doc offsets: lets the inverted-list builder (paper §4) reconstruct
  per-document token spans cheaply.
- `token_ids` stored separately from vectors: TAC (paper §3) buckets by
  vocabulary id, so we don't want it interleaved with the f32 stream.
- Magic + version + reserved bytes: every file format we'll ship gets
  versioned magic; cheap insurance.
