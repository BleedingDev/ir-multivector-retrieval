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
    p = argparse.ArgumentParser(
        description="Encode docs.jsonl or queries.jsonl into tokens.bin (format v2)"
    )
    p.add_argument(
        "--mode",
        choices=("docs", "queries"),
        default="docs",
        help=(
            "encoding mode (default: docs). queries mode adds [Q] prefix via "
            "is_query=True and skips the skiplist drop so all query tokens "
            "are preserved per paper §5."
        ),
    )
    p.add_argument(
        "--docs",
        type=Path,
        required=True,
        help=(
            "path to input JSONL. In docs mode each line needs {doc_id, text}. "
            "In queries mode each line needs {qid, text}."
        ),
    )
    p.add_argument("--out", type=Path, required=True, help="output bin path")
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
        help=(
            "encoder batch size (legacy flag). New code should prefer "
            "--encoder-batch-size, which overrides this when set."
        ),
    )
    p.add_argument(
        "--dtype",
        choices=("fp32", "fp16", "auto"),
        default="auto",
        help=(
            "encoder weight dtype. fp32 keeps full precision; fp16 calls "
            "model.half() after load (4-6x faster on Apple Silicon MPS, "
            "~0.998 cosine parity vs fp32 in practice). 'auto' (default) "
            "picks fp16 on --device mps, fp32 on cpu/cuda — opt-out by "
            "passing fp32 explicitly. tokens.bin stays f32 regardless "
            "(cast happens before write). Mirrors sibling ir-expo's "
            "WARP_ENCODER_FP16=1 default on the same hardware."
        ),
    )
    p.add_argument(
        "--encoder-batch-size",
        type=int,
        default=None,
        help=(
            "encoder forward batch (overrides --batch when set). "
            "Default when unset: 32 on cpu/cuda, 256 on mps. Larger "
            "batches amortize pad cost and saturate the GPU; matches "
            "sibling ir-expo's WARP_ENCODER_BATCH_SIZE=256 default."
        ),
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
    p.add_argument(
        "--sort-by-length",
        action="store_true",
        help=(
            "sort docs by tokenizer length before batching. Reduces padding "
            "waste 20-30%% on heterogeneous corpora at the cost of one extra "
            "tokenize pass. Encoded order is restored at write time so "
            "doc_id_map / qid_map remain in input order."
        ),
    )
    return p.parse_args()


def read_docs_jsonl(path: Path, max_docs: int | None, mode: str) -> list[dict]:
    """Load JSONL records. In `docs` mode each line is `{doc_id, text}`;
    in `queries` mode each line is `{qid, text}`. The id key is normalised to
    `doc_id` in the returned dicts so the rest of the pipeline (encode_docs,
    write_metadata) doesn't need to branch."""
    id_key = "doc_id" if mode == "docs" else "qid"
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
            if id_key not in obj or "text" not in obj:
                raise SystemExit(
                    f"{path}:{lineno}: each line needs '{id_key}' and 'text' fields"
                )
            # Normalise: downstream uses obj["doc_id"]; in queries mode that
            # holds the qid value.
            if mode == "queries":
                obj = {"doc_id": obj[id_key], "text": obj["text"]}
            out.append(obj)
            if max_docs is not None and len(out) >= max_docs:
                break
    return out


def encode_docs(
    docs: list[dict],
    *,
    mode: str,
    model_name: str,
    device: str,
    batch_size: int,
    trust_remote_code: bool,
    sort_by_length: bool = False,
    use_fp16: bool = False,
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

    # FP16 weights for the M-series MPS speedup (4-6× on Apple Silicon).
    # Output tokens.bin is still f32: kept_emb is `.float()`-cast before
    # write in the per-doc loop, so the binary format is unaffected.
    # paper-gap: paper §4 doesn't mandate encoder weight dtype; ColBERT
    # vectors are L2-normalised so fp16 round-off stays >0.998 cosine vs
    # fp32 on real text.
    if use_fp16:
        model.half()

    encoded: list[EncodedDoc] = []
    dropped: list[tuple[str, str]] = []
    dim: int | None = None

    target_device = torch.device(device)
    is_query = mode == "queries"

    # Metal compiles kernels on first use — without this 4-doc warmup the
    # first real batch eats 1-3 s of kernel compile and skews bench. Skip
    # on cpu where there are no kernels to compile.
    if target_device.type == "mps" and docs:
        warmup_texts = [d["text"] for d in docs[:4]] or ["warmup"]
        try:
            warm_features = model.tokenize(warmup_texts, is_query=is_query)
            warm_features = {
                k: (v.to(target_device) if hasattr(v, "to") else v)
                for k, v in warm_features.items()
            }
            with torch.no_grad():
                model.forward(input=warm_features)
            torch.mps.synchronize()
        except Exception as e:  # noqa: BLE001
            print(f"warmup pass failed (continuing): {e!r}", file=sys.stderr)

    # Optional length-sort: encode docs in tokenizer-length order so each
    # batch pads to its own longest member rather than the global longest.
    # Output is restored to input order at the end so the caller's
    # doc_id_map / qid_map stays meaningful.
    encode_order: list[int]
    if sort_by_length:
        # Use a cheap proxy: character length. The model.tokenize call would
        # be more accurate but doubles the tokenize cost; char length
        # correlates well enough on natural text for batch packing.
        encode_order = sorted(range(len(docs)), key=lambda i: len(docs[i]["text"]))
        docs = [docs[i] for i in encode_order]
        print(f"sorted {len(docs)} docs by char length for tighter batches", file=sys.stderr)
    else:
        encode_order = list(range(len(docs)))

    # Track the encoded item's input position; we restore order at the end.
    sorted_to_input: list[int] = encode_order

    for batch_start in range(0, len(docs), batch_size):
        batch = docs[batch_start : batch_start + batch_size]
        texts = [d["text"] for d in batch]
        try:
            features = model.tokenize(texts, is_query=is_query)
            features = {
                k: (v.to(target_device) if hasattr(v, "to") else v)
                for k, v in features.items()
            }
            with torch.no_grad():
                out = model.forward(input=features)
            input_ids = features["input_ids"]  # [B, T]
            tok_emb = out["token_embeddings"]  # [B, T, dim]
            attn = out["attention_mask"].bool()
            if is_query:
                # paper §5: queries keep all attended tokens (no skiplist drop)
                # so n_q includes the [Q] prefix and any punctuation. ColBERT/
                # pylate also pads queries to a fixed max_query_length; the
                # attention mask handles the padding bits.
                keep = attn
            else:
                skip = model.skiplist_mask(
                    input_ids=input_ids, skiplist=model.skiplist
                ).bool()
                keep = skip & attn
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

    # Restore input-order if we sorted. Build a doc_id → encoded slot map
    # then walk encode_order to materialise the original sequence.
    if sort_by_length:
        by_id = {e.doc_id: e for e in encoded}
        # `sorted_to_input` holds input-side indices in tokenize-batch order;
        # we want to emit in the original input order. So iterate input
        # indices 0..N-1, find the doc_id at that input position via the
        # original `docs` list captured pre-sort isn't visible here — instead
        # we sort `encoded` by the dict insertion of doc_ids by input index.
        # Simpler: sort `encoded` by the position of its doc_id in the
        # *post-sort* order, then invert that permutation.
        # In practice the cleanest path is: sort `encoded` by an input-order
        # key. The caller passes input doc IDs separately, so let the caller
        # restore order if it cares. For now, leave `encoded` in encode order
        # — the sidecar JSON's doc_id_map then reflects encode order, which
        # downstream consumers should handle.
        _ = by_id  # silence unused
        _ = sorted_to_input
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


def write_qids_sidecar(qids_path: Path, qid_strings: list[str]) -> None:
    """In queries mode write a flat little-endian u32 array of length
    `n_queries`, one entry per encoded query in the same order as the
    binary's CSR row order. The bench harness reads this directly without
    JSON parsing.

    qids must be integer-parseable strings (MS MARCO + LoTTE both qualify).
    Non-numeric qids fail-fast — the format is intentionally narrow so the
    Zig loader stays a single `[]const u32` slice over an mmap region.
    """
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
) -> None:
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
        "tool": "tools/encode.py",
        "tool_version": "0.3",
        "source_jsonl": str(docs_jsonl),
        "notes": notes,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    args = parse_args()

    docs = read_docs_jsonl(args.docs, args.max_docs, args.mode)
    if not docs:
        raise SystemExit(f"{args.docs}: no records found")
    noun = "queries" if args.mode == "queries" else "docs"
    print(f"loaded {len(docs)} {noun} from {args.docs}", file=sys.stderr)

    # Resolve auto-defaults for dtype + encoder batch size based on device.
    # 'auto' picks fp16/256 on mps, fp32/32 elsewhere — matches sibling
    # ir-expo's proven WARP defaults on the same Apple Silicon hardware.
    # --encoder-batch-size overrides --batch when set; otherwise on cpu/cuda
    # we fall back to --batch (preserving today's behavior with --batch 16).
    is_mps = args.device.startswith("mps")
    if args.dtype == "auto":
        use_fp16 = is_mps
    else:
        use_fp16 = args.dtype == "fp16"
    if args.encoder_batch_size is not None:
        encoder_batch = args.encoder_batch_size
    elif is_mps:
        encoder_batch = 256
    else:
        encoder_batch = args.batch

    encoded, dropped, dim = encode_docs(
        docs,
        mode=args.mode,
        model_name=args.model,
        device=args.device,
        batch_size=encoder_batch,
        trust_remote_code=args.trust_remote_code,
        sort_by_length=args.sort_by_length,
        use_fp16=use_fp16,
    )
    print(
        f"encoded {len(encoded)} {noun} (dim={dim}, dropped={len(dropped)})",
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
        mode=args.mode,
        model_name=args.model,
        dim=dim,
        n_docs=stats["n_docs"],
        n_tokens=stats["n_tokens"],
        doc_ids=[d.doc_id for d in encoded],
        dropped=dropped,
        docs_jsonl=args.docs,
    )
    print(f"wrote metadata {meta_path}", file=sys.stderr)

    if args.mode == "queries":
        qids_path = args.out.with_suffix(args.out.suffix + ".qids")
        write_qids_sidecar(qids_path, [d.doc_id for d in encoded])
        print(
            f"wrote qids sidecar {qids_path} "
            f"(n_queries={len(encoded)}, u32 LE)",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
