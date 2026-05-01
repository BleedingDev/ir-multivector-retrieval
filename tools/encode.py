#!/usr/bin/env python3
"""tools/encode.py — ColBERTv2 → token-dump binary (format v2, real vocab IDs).

Owner: primitives-engineer.
See plan 01-primitives-and-io.plan.md and docs/token-dump-format.md.

Reads docs.jsonl (one JSON object per line with at least "doc_id" and "text"
fields), encodes each document with a ColBERT-style model (default
`colbert-ir/colbertv2.0` via pylate), L2-normalises the per-token
embeddings, and writes the flat-binary format defined in
docs/token-dump-format.md plus a sidecar `tokens.meta.json`.

Format v2 (this version) emits **real BERT vocabulary IDs** in
`token_ids[i]` so paper §3 TAC bucketing aggregates by surface vocabulary
("the" across docs lands in one cluster). v1 stored positional ids per
doc and degenerated TAC; the Zig parser rejects v1 files outright since
the constant flipped to 2.

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

We bypass `pylate.models.ColBERT.encode` because that wrapper does not
return per-token vocab IDs in lockstep with the token embeddings.
Instead we drive `model.tokenize` → `model.forward` → `model.skiplist_mask`
ourselves and apply the exact same keep-mask to both the input_ids and
the token_embeddings. This mirrors the embedding-keep logic from
`pylate/models/colbert.py::encode` (lines ~688–720) one-for-one so the
emitted vectors are byte-identical to what `model.encode` would have
produced.

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
TOKEN_DUMP_VERSION = 2  # v2 = real vocab IDs (lead bumps src/constants.zig in lockstep)
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

    Per-doc shape contract (mirrors `pylate.models.ColBERT.encode` for
    is_query=False, normalize=True):
        features = model.tokenize(texts, is_query=False)
            features["input_ids"]:      [B, T]   real BERT vocab IDs,
                                                  with [D] prefix inserted
            features["attention_mask"]: [B, T]
        out = model.forward(features)
            out["token_embeddings"]:    [B, T, dim]
            out["attention_mask"]:      [B, T]
        skip = model.skiplist_mask(features["input_ids"], model.skiplist)
        keep = (skip & out["attention_mask"]).bool()
        for b in range(B):
            kept_ids[b] = features["input_ids"][b, keep[b]]
            kept_emb[b] = out["token_embeddings"][b, keep[b], :]
            kept_emb[b] = F.normalize(kept_emb[b], p=2, dim=1)

    The keep-mask is applied identically to both input_ids and embeddings,
    so token_ids[i] is the BERT vocab id of the same row as vectors[i].
    """
    import torch  # type: ignore
    from pylate import models  # type: ignore

    model = models.ColBERT(
        model_name_or_path=model_name,
        trust_remote_code=trust_remote_code,
        device=device,
    )
    model.eval()

    encoded: list[EncodedDoc] = []
    dropped: list[tuple[str, str]] = []
    dim: int | None = None

    target_device = torch.device(device)

    for batch_start in range(0, len(docs), batch_size):
        batch = docs[batch_start : batch_start + batch_size]
        texts = [d["text"] for d in batch]
        try:
            features = model.tokenize(texts, is_query=False)
            features = {
                k: (v.to(target_device) if hasattr(v, "to") else v)
                for k, v in features.items()
            }
            with torch.no_grad():
                out = model.forward(input=features)
            input_ids = features["input_ids"]  # [B, T]
            tok_emb = out["token_embeddings"]  # [B, T, dim]
            attn = out["attention_mask"].bool()
            skip = model.skiplist_mask(input_ids=input_ids, skiplist=model.skiplist).bool()
            keep = skip & attn  # [B, T]
        except Exception as e:  # noqa: BLE001
            for d in batch:
                dropped.append((str(d["doc_id"]), f"batch encode failed: {e!r}"))
            continue

        for b, d in enumerate(batch):
            try:
                kept_mask = keep[b]
                kept_ids = input_ids[b][kept_mask].detach().cpu().tolist()
                kept_emb = tok_emb[b][kept_mask].detach().cpu().float()
                # Defensive renormalize — matches the encode() path
                # (normalize_embeddings=True is the pylate default).
                kept_emb = torch.nn.functional.normalize(kept_emb, p=2, dim=1)

                if kept_emb.dim() != 2:
                    dropped.append(
                        (str(d["doc_id"]), f"unexpected encoder shape {tuple(kept_emb.shape)}")
                    )
                    continue
                n_tok, d_dim = int(kept_emb.shape[0]), int(kept_emb.shape[1])
                if d_dim == 0 or d_dim > MAX_DIM:
                    dropped.append(
                        (str(d["doc_id"]), f"dim {d_dim} outside [1, {MAX_DIM}]")
                    )
                    continue
                if n_tok == 0:
                    dropped.append((str(d["doc_id"]), "0-token output after masking"))
                    continue
                if len(kept_ids) != n_tok:
                    dropped.append(
                        (
                            str(d["doc_id"]),
                            f"id/embedding length mismatch: {len(kept_ids)} != {n_tok}",
                        )
                    )
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
                encoded.append(
                    EncodedDoc(
                        doc_id=str(d["doc_id"]),
                        token_ids=[int(x) for x in kept_ids],
                        vectors=kept_emb.tolist(),
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
        "tool_version": "0.2",
        "source_jsonl": str(docs_jsonl),
        "notes": [
            "token_ids are real BERT vocabulary IDs from "
            "tokenizer(text)['input_ids'] with the ColBERT [D] prefix "
            "preserved and skiplist (punctuation) tokens dropped, exactly "
            "matching pylate.models.ColBERT.encode's keep-mask. Format v2.",
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
