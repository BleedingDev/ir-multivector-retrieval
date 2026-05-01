#!/usr/bin/env python3
"""tools/_tokens_bin.py — shared writer for the tokens.bin v2 format.

Used by tools/encode.py (PyTorch path) and tools/encode_mlx.py (MLX path)
so both encoders produce byte-identical output. The format constants and
writer mirror src/io/token_dump.zig::parseBytes — any change here MUST
update the Zig parser in lockstep.

This module is intentionally tiny and dependency-free (only stdlib) so
importing it does not pay the torch/pylate/mlx import cost. Both encoders
import this; no circular dep risk.
"""
from __future__ import annotations

import datetime as _dt
import json
import struct
from dataclasses import dataclass
from pathlib import Path

# Keep these in lockstep with src/io/token_dump.zig + src/constants.zig.
TOKEN_DUMP_MAGIC = b"TAC_TKN1"
TOKEN_DUMP_VERSION = 2  # v2 = real BERT vocab IDs
DTYPE_F32 = 0
HEADER_SIZE = 40  # magic(8) + version(4) + dim(4) + n_docs(8) + n_tokens(8) + dtype(1) + reserved(7)
MAX_DIM = 4096


@dataclass
class EncodedDoc:
    """Encoder-agnostic per-document output.

    Both encode.py (pylate/torch) and encode_mlx.py (MLX) build this and
    hand it to write_tokens_bin. token_ids are real BERT vocabulary IDs;
    vectors are L2-normalised f32 row-major n_tokens × dim.

    `vectors` accepts either list[list[float]] (legacy encode.py path) or a
    numpy.ndarray of shape (n_tokens, dim) and dtype float32 (encode_mlx.py
    fast path). The writer picks a bulk-write path when given numpy.
    """

    doc_id: str
    token_ids: list[int]
    vectors: object  # list[list[float]] | numpy.ndarray (f32 [n,dim])


def write_tokens_bin(out_path: Path, encoded: list[EncodedDoc], dim: int) -> dict:
    """Write the flat binary in the on-disk format and return summary stats.

    Byte layout (must match src/io/token_dump.zig::parseBytes):
      header (40B): magic(8) || version(u32 LE) || dim(u32 LE) ||
                    n_docs(u64 LE) || n_tokens(u64 LE) || dtype(u8) || zeros(7)
      doc_offsets[n_docs+1] u64 LE  (CSR)
      token_ids[n_tokens]   u32 LE
      vectors[n_tokens*dim] f32 LE  (row-major, L2-normalised)

    Fast path: if every doc's `vectors` is a numpy.ndarray of dtype float32
    and shape (n_tok_d, dim), the writer concatenates and emits via
    `fp.write(arr.tobytes())` instead of struct.pack-per-float. This is
    ~50-100x faster on large outputs.
    """
    n_docs = len(encoded)
    n_tokens = sum(len(d.token_ids) for d in encoded)
    if n_tokens == 0:
        raise SystemExit("aborting: 0 tokens after encoding")

    doc_offsets = [0]
    for d in encoded:
        doc_offsets.append(doc_offsets[-1] + len(d.token_ids))
    assert doc_offsets[-1] == n_tokens

    # Check whether every doc has numpy vectors → bulk-write path.
    try:
        import numpy as np  # type: ignore
        np_ndarray = np.ndarray
    except ImportError:  # pragma: no cover - numpy is a hard dep in practice
        np = None
        np_ndarray = ()

    bulk_ok = np is not None and all(
        isinstance(d.vectors, np_ndarray)
        and d.vectors.dtype == np.float32
        and d.vectors.ndim == 2
        and d.vectors.shape == (len(d.token_ids), dim)
        for d in encoded
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(TOKEN_DUMP_MAGIC)
        f.write(struct.pack("<I", TOKEN_DUMP_VERSION))
        f.write(struct.pack("<I", dim))
        f.write(struct.pack("<Q", n_docs))
        f.write(struct.pack("<Q", n_tokens))
        f.write(struct.pack("<B", DTYPE_F32))
        f.write(b"\x00" * 7)
        assert f.tell() == HEADER_SIZE

        if bulk_ok:
            offsets_arr = np.asarray(doc_offsets, dtype="<u8")
            f.write(offsets_arr.tobytes())

            ids_arr = np.fromiter(
                (int(tid) for d in encoded for tid in d.token_ids),
                dtype="<u4",
                count=n_tokens,
            )
            f.write(ids_arr.tobytes())

            # Concatenate L2-normalized rows in CSR order; ensure
            # native-little-endian f32 contiguous bytes.
            vecs_arr = np.concatenate(
                [np.ascontiguousarray(d.vectors, dtype="<f4") for d in encoded],
                axis=0,
            )
            f.write(vecs_arr.tobytes())
        else:
            for o in doc_offsets:
                f.write(struct.pack("<Q", o))

            for d in encoded:
                for tid in d.token_ids:
                    f.write(struct.pack("<I", int(tid)))

            for d in encoded:
                for vec in d.vectors:
                    if len(vec) != dim:
                        raise SystemExit(
                            f"internal: doc {d.doc_id!r} vector dim {len(vec)} != {dim}"
                        )
                    for x in vec:
                        f.write(struct.pack("<f", float(x)))

    return {"n_docs": n_docs, "n_tokens": n_tokens, "dim": dim}


def write_qids_sidecar(qids_path: Path, qid_strings: list[str]) -> None:
    """Flat little-endian u32 array, one entry per encoded query in CSR row
    order. The Zig bench harness mmaps it as `[]const u32` directly. qids
    must be integer-parseable (MS MARCO + LoTTE qualify)."""
    qids_path.parent.mkdir(parents=True, exist_ok=True)
    with qids_path.open("wb") as f:
        for q in qid_strings:
            try:
                qi = int(q)
            except ValueError as e:
                raise SystemExit(
                    f"qid {q!r} is not int-parseable; .qids sidecar requires "
                    "numeric qids (MS MARCO / LoTTE qualify)."
                ) from e
            if qi < 0 or qi > 0xFFFF_FFFF:
                raise SystemExit(
                    f"qid {qi} outside u32 range; .qids sidecar requires "
                    "0 <= qid <= 2^32-1."
                )
            f.write(struct.pack("<I", qi))


def write_metadata(
    meta_path: Path,
    *,
    mode: str,
    model_name: str,
    dim: int,
    n_docs: int,
    n_tokens: int,
    doc_ids: list[str],
    dropped: list[tuple[str, str]],
    docs_jsonl: Path,
    tool: str,
    tool_version: str,
) -> None:
    """Sidecar JSON describing the dump. `tool` + `tool_version` are
    parameterised so encode.py and encode_mlx.py can identify their
    output (e.g. `tools/encode.py` 0.3 vs `tools/encode_mlx.py` 0.1)."""
    is_query = mode == "queries"
    id_field = "qid" if is_query else "doc_id"
    map_field = "qid_map" if is_query else "doc_id_map"
    if is_query:
        notes = [
            "token_ids are real BERT vocabulary IDs from "
            "tokenizer(text, is_query=True)['input_ids']: the ColBERT [Q] "
            "prefix is preserved and ALL attended tokens are kept (no "
            "skiplist drop on the query side, per paper §5). Format v2.",
        ]
    else:
        notes = [
            "token_ids are real BERT vocabulary IDs from "
            "tokenizer(text)['input_ids'] with the ColBERT [D] prefix "
            "preserved and skiplist (punctuation) tokens dropped, exactly "
            "matching pylate.models.ColBERT.encode's keep-mask. Format v2.",
        ]
    meta = {
        "format_version": TOKEN_DUMP_VERSION,
        "mode": mode,
        "encoder": model_name,
        "encoder_dim": dim,
        "n_docs": n_docs,
        "n_tokens": n_tokens,
        map_field: doc_ids,
        "dropped": [{id_field: d, "reason": r} for d, r in dropped],
        "built_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "tool": tool,
        "tool_version": tool_version,
        "source_jsonl": str(docs_jsonl),
        "notes": notes,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        f.write("\n")
