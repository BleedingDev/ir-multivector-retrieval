#!/usr/bin/env python3
"""tools/encode.py — ColBERTv2 → token-dump binary.

Owner: primitives-engineer.
See plan 01-primitives-and-io.plan.md and docs/token-dump-format.md.

Reads docs.jsonl (one JSON object per line with at least "doc_id" and "text"
fields), encodes each document with a ColBERT-style model (default
`colbert-ir/colbertv2.0` via pylate), L2-normalises the per-token
embeddings, and writes the flat-binary format defined in
docs/token-dump-format.md plus a sidecar `tokens.meta.json`.

Usage:
    python tools/encode.py \\
        --docs   path/to/docs.jsonl \\
        --out    data/msmarco_v1/tokens.bin \\
        --model  colbert-ir/colbertv2.0 \\
        --device cpu \\
        --batch  16

The model defaults to `colbert-ir/colbertv2.0` to match the paper. A
sibling project (ir-expo) found `jinaai/jina-colbert-v2-64` works
identically through the same pylate path; pass `--model` (with
`--trust-remote-code` for jina) to override.

Determinism: pylate/transformers pin model weights by revision; the
output bytes for a given (model, revision, doc list) are reproducible
modulo CPU/GPU non-determinism in the encoder. We do not introduce extra
randomness here.

Best-effort robust to encoder failures: a doc that fails encoding is
skipped with a warning printed to stderr, and the metadata file records
both the surviving and dropped doc IDs.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

# torch / pylate are imported lazily inside encode_docs() so the CLI's
# --help and read_docs_jsonl path don't pay the multi-second import cost.


TOKEN_DUMP_MAGIC = b"TAC_TKN1"  # must match src/constants.zig
TOKEN_DUMP_VERSION = 1
DTYPE_F32 = 0
HEADER_SIZE = 40  # magic(8) + version(4) + dim(4) + n_docs(8) + n_tokens(8) + dtype(1) + reserved(7)

# Format spec ceiling — must agree with parser in src/io/token_dump.zig.
MAX_DIM = 4096


@dataclass
class EncodedDoc:
    doc_id: str
    token_ids: list[int]
    vectors: list[list[float]]  # n_tokens × dim, L2-normalised


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Encode docs.jsonl into tokens.bin")
    p.add_argument("--docs", type=Path, required=True, help="path to docs.jsonl")
    p.add_argument("--out", type=Path, required=True, help="output tokens.bin path")
    p.add_argument(
        "--model",
        type=str,
        default="colbert-ir/colbertv2.0",
        help="ColBERT model name (HuggingFace ID or local path)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="torch device (cpu, cuda, cuda:0, mps)",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=16,
        help="encoder batch size",
    )
    p.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="optional cap on docs (smoke testing)",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="pass through to pylate.models.ColBERT (needed for jina models)",
    )
    p.add_argument(
        "--meta-out",
        type=Path,
        default=None,
        help="sidecar metadata path; defaults to <out>.meta.json",
    )
    return p.parse_args()


def read_docs_jsonl(path: Path, max_docs: int | None) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{lineno}: malformed JSON: {e}") from e
            if "doc_id" not in obj or "text" not in obj:
                raise SystemExit(
                    f"{path}:{lineno}: each line needs 'doc_id' and 'text' fields"
                )
            out.append(obj)
            if max_docs is not None and len(out) >= max_docs:
                break
    return out


def encode_docs(
    docs: list[dict],
    *,
    model_name: str,
    device: str,
    batch_size: int,
    trust_remote_code: bool,
) -> tuple[list[EncodedDoc], list[tuple[str, str]], int]:
    """Returns (encoded, dropped, dim).

    `dropped` is a list of (doc_id, reason) for docs that failed to encode.
    """
    # Lazy import so --help is fast.
    import torch  # type: ignore
    from pylate import models  # type: ignore

    model = models.ColBERT(
        model_name_or_path=model_name,
        trust_remote_code=trust_remote_code,
        device=device,
    )

    encoded: list[EncodedDoc] = []
    dropped: list[tuple[str, str]] = []
    dim: int | None = None

    for batch_start in range(0, len(docs), batch_size):
        batch = docs[batch_start : batch_start + batch_size]
        texts = [d["text"] for d in batch]
        try:
            outs = model.encode(
                texts,
                convert_to_numpy=False,
                convert_to_tensor=False,
                is_query=False,
                show_progress_bar=False,
            )
        except Exception as e:  # noqa: BLE001
            for d in batch:
                dropped.append((str(d["doc_id"]), f"batch encode failed: {e!r}"))
            continue

        for d, out in zip(batch, outs):
            try:
                t = (
                    out.detach().cpu().float()
                    if isinstance(out, torch.Tensor)
                    else torch.tensor(out).float()
                )
                # pylate already L2-normalises ColBERT outputs but we
                # renormalise defensively — the Zig parser doesn't.
                t = torch.nn.functional.normalize(t, dim=1)
                if t.dim() != 2:
                    dropped.append(
                        (str(d["doc_id"]), f"unexpected encoder shape {tuple(t.shape)}")
                    )
                    continue
                n_tok, d_dim = int(t.shape[0]), int(t.shape[1])
                if d_dim == 0 or d_dim > MAX_DIM:
                    dropped.append(
                        (str(d["doc_id"]), f"dim {d_dim} outside [1, {MAX_DIM}]")
                    )
                    continue
                if n_tok == 0:
                    dropped.append((str(d["doc_id"]), "0-token output"))
                    continue
                if dim is None:
                    dim = d_dim
                elif dim != d_dim:
                    dropped.append(
                        (
                            str(d["doc_id"]),
                            f"dim drift: expected {dim}, got {d_dim}",
                        )
                    )
                    continue
                # paper-gap: pylate doesn't surface real vocab ids in a
                # uniform way across revisions, so v1 of tokens.bin stores
                # positional ids 0..n_tok per doc. The TAC clusterer treats
                # token_id as a grouping key, so today the per-doc bucketing
                # is degenerate. We document this in tokens.meta.json so a
                # future encoder upgrade can populate real ids without an
                # on-disk format change.
                token_ids = list(range(n_tok))
                vectors = t.tolist()
                encoded.append(
                    EncodedDoc(
                        doc_id=str(d["doc_id"]),
                        token_ids=token_ids,
                        vectors=vectors,
                    )
                )
            except Exception as e:  # noqa: BLE001
                dropped.append((str(d["doc_id"]), f"post-process failed: {e!r}"))

    if dim is None:
        raise SystemExit("no docs encoded successfully — aborting before writing")
    return encoded, dropped, dim


def write_tokens_bin(out_path: Path, encoded: list[EncodedDoc], dim: int) -> dict:
    """Write the flat binary; return summary stats for metadata."""
    n_docs = len(encoded)
    n_tokens = sum(len(d.token_ids) for d in encoded)
    if n_tokens == 0:
        raise SystemExit("aborting: 0 tokens after encoding")

    # CSR-style doc_offsets.
    doc_offsets = [0]
    for d in encoded:
        doc_offsets.append(doc_offsets[-1] + len(d.token_ids))
    assert doc_offsets[-1] == n_tokens

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        # Header.
        f.write(TOKEN_DUMP_MAGIC)
        f.write(struct.pack("<I", TOKEN_DUMP_VERSION))
        f.write(struct.pack("<I", dim))
        f.write(struct.pack("<Q", n_docs))
        f.write(struct.pack("<Q", n_tokens))
        f.write(struct.pack("<B", DTYPE_F32))
        f.write(b"\x00" * 7)  # reserved
        assert f.tell() == HEADER_SIZE

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


def write_metadata(
    meta_path: Path,
    *,
    model_name: str,
    dim: int,
    n_docs: int,
    n_tokens: int,
    doc_ids: list[str],
    dropped: list[tuple[str, str]],
    docs_jsonl: Path,
) -> None:
    meta = {
        "format_version": TOKEN_DUMP_VERSION,
        "encoder": model_name,
        "encoder_dim": dim,
        "n_docs": n_docs,
        "n_tokens": n_tokens,
        "doc_id_map": doc_ids,
        "dropped": [{"doc_id": d, "reason": r} for d, r in dropped],
        "built_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "tool": "tools/encode.py",
        "tool_version": "0.1",
        "source_jsonl": str(docs_jsonl),
        "notes": [
            "token_ids are positional within each doc, not vocabulary ids "
            "(see encode.py paper-gap comment).",
        ],
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    args = parse_args()

    docs = read_docs_jsonl(args.docs, args.max_docs)
    if not docs:
        raise SystemExit(f"{args.docs}: no docs found")
    print(f"loaded {len(docs)} docs from {args.docs}", file=sys.stderr)

    encoded, dropped, dim = encode_docs(
        docs,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch,
        trust_remote_code=args.trust_remote_code,
    )
    print(
        f"encoded {len(encoded)} docs (dim={dim}, dropped={len(dropped)})",
        file=sys.stderr,
    )

    stats = write_tokens_bin(args.out, encoded, dim)
    print(
        f"wrote {args.out} "
        f"(n_docs={stats['n_docs']}, n_tokens={stats['n_tokens']}, dim={stats['dim']})",
        file=sys.stderr,
    )

    meta_path = args.meta_out or args.out.with_suffix(args.out.suffix + ".meta.json")
    write_metadata(
        meta_path,
        model_name=args.model,
        dim=dim,
        n_docs=stats["n_docs"],
        n_tokens=stats["n_tokens"],
        doc_ids=[d.doc_id for d in encoded],
        dropped=dropped,
        docs_jsonl=args.docs,
    )
    print(f"wrote metadata {meta_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
