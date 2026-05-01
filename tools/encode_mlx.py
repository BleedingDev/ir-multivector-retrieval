#!/usr/bin/env python3
"""tools/encode_mlx.py — MLX (Apple Silicon native) port of tools/encode.py.

Owner: post-hackathon plan-12 (mlx-encoder-engineer).

EXPERIMENTAL until the 4-gate parity contract in
`tests/live/mlx_parity.py` is green.

Reuses the HuggingFace tokenizer via pylate's `model.tokenize` (HF
tokenizers are deterministic CPU code, fast, and the only way to get
byte-equal token_ids). Implements the BERT base forward pass directly in
mlx.nn (no mlx_lm BERT class today). Outputs the same tokens.bin v2
format via the shared writer in tools/_tokens_bin.py.

Usage:
    tools/.venv/bin/python tools/encode_mlx.py \\
        --docs   tests/fixtures/live/docs.jsonl \\
        --out    /tmp/parity_mlx.bin \\
        --model  colbert-ir/colbertv2.0 \\
        --weights tools/.cache/colbert-ir__colbertv2.0/mlx_weights.safetensors \\
        --dtype  fp16

Run tools/export_colbert_to_mlx.py first to produce the safetensors file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Re-use the shared writer from encode.py's refactor (Phase B).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _tokens_bin import EncodedDoc, write_metadata as _write_metadata_shared, write_tokens_bin

import mlx.core as mx
import mlx.nn as nn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MLX-native ColBERT encoder (parity-tested companion to encode.py)"
    )
    p.add_argument("--docs", type=Path, required=True, help="path to docs.jsonl")
    p.add_argument("--out", type=Path, required=True, help="output tokens.bin path")
    p.add_argument(
        "--model",
        type=str,
        default="colbert-ir/colbertv2.0",
        help="ColBERT model name (HuggingFace ID or local path); used for tokenizer + skiplist",
    )
    p.add_argument(
        "--weights",
        type=Path,
        default=None,
        help=(
            "MLX safetensors file produced by tools/export_colbert_to_mlx.py. "
            "Defaults to tools/.cache/<safe_model>/mlx_weights.safetensors. "
            "config.json must sit next to it."
        ),
    )
    p.add_argument(
        "--dtype",
        choices=("fp32", "fp16"),
        default="fp16",
        help="MLX compute dtype. fp16 default to mirror encode.py's MPS default.",
    )
    p.add_argument(
        "--batch", type=int, default=32, help="docs per forward pass batch"
    )
    p.add_argument(
        "--max-docs", type=int, default=None, help="optional cap (smoke testing)"
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="pass through to pylate.models.ColBERT for tokenizer load (jina models)",
    )
    p.add_argument(
        "--meta-out",
        type=Path,
        default=None,
        help="sidecar metadata path; defaults to <out>.meta.json",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# BERT-base forward in MLX
# ---------------------------------------------------------------------------


class BertSelfAttention(nn.Module):
    """Multi-head self-attention block matching HF BertSelfAttention's
    weight layout exactly. q/k/v are separate Linear layers (HF stores
    them split, not as a fused QKV)."""

    def __init__(self, hidden: int, n_heads: int):
        super().__init__()
        if hidden % n_heads != 0:
            raise SystemExit(f"hidden {hidden} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, hidden)
        self.ln = nn.LayerNorm(hidden)

    def __call__(self, x: mx.array, attn_bias: mx.array) -> mx.array:
        B, T, H = x.shape
        n = self.n_heads
        d = self.head_dim
        scale = 1.0 / (d ** 0.5)

        q = self.q(x).reshape(B, T, n, d).transpose(0, 2, 1, 3)  # [B,n,T,d]
        k = self.k(x).reshape(B, T, n, d).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(B, T, n, d).transpose(0, 2, 1, 3)

        # Fused SDPA: softmax internally promotes to fp32, fewer kernel launches
        # and less memory traffic than the manual q@k.T -> softmax -> @v path.
        # attn_bias is additive [B,1,1,T] and broadcasts to [B,n,T,T].
        ctx = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask=attn_bias
        )  # [B, n, T, d]
        ctx = ctx.transpose(0, 2, 1, 3).reshape(B, T, H)

        # output projection + residual + LN (BERT post-LN)
        return self.ln(self.out(ctx) + x)


class BertFFN(nn.Module):
    """Position-wise FFN: up-project (gelu) → down-project → residual + LN."""

    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.up = nn.Linear(hidden, intermediate)
        self.down = nn.Linear(intermediate, hidden)
        self.ln = nn.LayerNorm(hidden)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.up(x)
        h = nn.gelu(h)
        h = self.down(h)
        return self.ln(h + x)


class BertLayer(nn.Module):
    def __init__(self, hidden: int, n_heads: int, intermediate: int):
        super().__init__()
        self.attn = BertSelfAttention(hidden, n_heads)
        self.ff = BertFFN(hidden, intermediate)

    def __call__(self, x: mx.array, attn_bias: mx.array) -> mx.array:
        x = self.attn(x, attn_bias)
        x = self.ff(x)
        return x


class BertEmbeddings(nn.Module):
    def __init__(self, vocab: int, max_pos: int, type_vocab: int, hidden: int):
        super().__init__()
        self.word = nn.Embedding(vocab, hidden)
        self.position = nn.Embedding(max_pos, hidden)
        self.token_type = nn.Embedding(type_vocab, hidden)
        self.ln = nn.LayerNorm(hidden)

    def __call__(self, input_ids: mx.array, token_type_ids: mx.array) -> mx.array:
        T = input_ids.shape[1]
        pos_ids = mx.arange(T, dtype=mx.int32)
        # broadcast pos_ids over batch
        x = self.word(input_ids) + self.position(pos_ids) + self.token_type(token_type_ids)
        return self.ln(x)


class ColBertMLX(nn.Module):
    """BERT base + final linear projection (no bias) + optional L2-normalize.
    Forward returns the per-token L2-normalised projected embeddings."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.embeddings = BertEmbeddings(
            vocab=cfg["vocab_size"],
            max_pos=cfg["max_position_embeddings"],
            type_vocab=cfg["type_vocab_size"],
            hidden=cfg["hidden_size"],
        )
        self.layers = [
            BertLayer(
                hidden=cfg["hidden_size"],
                n_heads=cfg["n_heads"],
                intermediate=cfg["intermediate_size"],
            )
            for _ in range(cfg["n_layers"])
        ]
        self.projection = nn.Linear(
            cfg["projection_in"], cfg["projection_out"], bias=cfg["projection_has_bias"]
        )

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
        token_type_ids: mx.array,
    ) -> mx.array:
        # attention bias: 0 where mask=1, -inf where mask=0 (so softmax→0).
        # shape: [B, 1, 1, T] to broadcast over (n_heads, query_seq).
        bias = (1.0 - attention_mask.astype(input_ids.dtype)).astype(mx.float32) * -1e4
        bias = bias.reshape(bias.shape[0], 1, 1, bias.shape[1])

        x = self.embeddings(input_ids, token_type_ids)
        # Cast bias to x's dtype so attention math stays in compute precision.
        bias = bias.astype(x.dtype)

        for layer in self.layers:
            x = layer(x, bias)

        # 768 -> 128 projection
        x = self.projection(x)
        # L2-normalize per token along last axis
        norm = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True) + 1e-12)
        return x / norm


def _load_weights_into_model(model: ColBertMLX, weights: dict) -> None:
    """Bind the flat safetensors dict produced by export_colbert_to_mlx.py
    into the MLX module tree."""
    # Build params tree matching mlx.nn's expected layout.
    params = {
        "embeddings": {
            "word": {"weight": weights["embeddings.word.weight"]},
            "position": {"weight": weights["embeddings.position.weight"]},
            "token_type": {"weight": weights["embeddings.token_type.weight"]},
            "ln": {
                "weight": weights["embeddings.ln.weight"],
                "bias": weights["embeddings.ln.bias"],
            },
        },
        "layers": [],
        "projection": {"weight": weights["projection.weight"]},
    }
    if "projection.bias" in weights:
        params["projection"]["bias"] = weights["projection.bias"]

    n_layers = model.cfg["n_layers"]
    for i in range(n_layers):
        params["layers"].append(
            {
                "attn": {
                    "q": {
                        "weight": weights[f"layers.{i}.attn.q.weight"],
                        "bias": weights[f"layers.{i}.attn.q.bias"],
                    },
                    "k": {
                        "weight": weights[f"layers.{i}.attn.k.weight"],
                        "bias": weights[f"layers.{i}.attn.k.bias"],
                    },
                    "v": {
                        "weight": weights[f"layers.{i}.attn.v.weight"],
                        "bias": weights[f"layers.{i}.attn.v.bias"],
                    },
                    "out": {
                        "weight": weights[f"layers.{i}.attn.out.weight"],
                        "bias": weights[f"layers.{i}.attn.out.bias"],
                    },
                    "ln": {
                        "weight": weights[f"layers.{i}.attn.ln.weight"],
                        "bias": weights[f"layers.{i}.attn.ln.bias"],
                    },
                },
                "ff": {
                    "up": {
                        "weight": weights[f"layers.{i}.ff.up.weight"],
                        "bias": weights[f"layers.{i}.ff.up.bias"],
                    },
                    "down": {
                        "weight": weights[f"layers.{i}.ff.down.weight"],
                        "bias": weights[f"layers.{i}.ff.down.bias"],
                    },
                    "ln": {
                        "weight": weights[f"layers.{i}.ff.ln.weight"],
                        "bias": weights[f"layers.{i}.ff.ln.bias"],
                    },
                },
            }
        )

    model.update(params)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


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


def encode_docs_mlx(
    docs: list[dict],
    *,
    model_name: str,
    weights_path: Path,
    dtype: str,
    batch_size: int,
    trust_remote_code: bool,
) -> tuple[list[EncodedDoc], list[tuple[str, str]], int]:
    """Encode docs through MLX. Returns (encoded, dropped, dim).

    Tokenizer + skiplist come from pylate (CPU, deterministic, fast).
    Forward pass + L2-norm run in MLX. Output rows match the keep-mask
    (skiplist & attention_mask), exactly as encode.py does.
    """
    # Tokenizer comes from pylate. We don't run model.forward through pylate
    # — only model.tokenize and model.skiplist_mask.
    from pylate import models  # type: ignore

    pl = models.ColBERT(
        model_name_or_path=model_name,
        trust_remote_code=trust_remote_code,
        device="cpu",
    )
    pl.eval()
    skiplist = pl.skiplist  # list[int]

    # Read MLX safetensors + config.
    if not weights_path.exists():
        raise SystemExit(
            f"weights file {weights_path} missing — run tools/export_colbert_to_mlx.py first"
        )
    config_path = weights_path.parent / "config.json"
    if not config_path.exists():
        raise SystemExit(f"config.json missing next to {weights_path}")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    weights = mx.load(str(weights_path))
    compute_dtype = mx.float16 if dtype == "fp16" else mx.float32
    # Cast weights to compute dtype so the forward pass is consistent.
    weights = {k: v.astype(compute_dtype) for k, v in weights.items()}

    model = ColBertMLX(cfg)
    _load_weights_into_model(model, weights)
    model.eval()

    skip_set = set(int(s) for s in skiplist)

    encoded: list[EncodedDoc] = []
    dropped: list[tuple[str, str]] = []
    dim = cfg["projection_out"]

    for batch_start in range(0, len(docs), batch_size):
        batch = docs[batch_start : batch_start + batch_size]
        texts = [d["text"] for d in batch]
        try:
            features = pl.tokenize(texts, is_query=False)
            input_ids_pt = features["input_ids"]
            attn_pt = features["attention_mask"]
            type_ids_pt = features.get("token_type_ids")
            if type_ids_pt is None:
                # If absent (some tokenizers), default to zeros.
                import torch  # type: ignore

                type_ids_pt = torch.zeros_like(input_ids_pt)

            input_ids = mx.array(input_ids_pt.cpu().numpy().astype("int32"))
            attn = mx.array(attn_pt.cpu().numpy().astype("int32"))
            type_ids = mx.array(type_ids_pt.cpu().numpy().astype("int32"))

            tok_emb = model(input_ids, attn, type_ids)  # [B, T, dim]
            # Force evaluation so we have concrete arrays for slicing.
            mx.eval(tok_emb)
        except Exception as e:  # noqa: BLE001
            for d in batch:
                dropped.append((str(d["doc_id"]), f"batch encode failed: {e!r}"))
            continue

        # Apply keep mask (skiplist & attention_mask) per row.
        ids_np = input_ids_pt.cpu().numpy()
        attn_np = attn_pt.cpu().numpy()
        emb_np = tok_emb.astype(mx.float32)
        # mlx.array → host: prefer mx.eval + .tolist or numpy() if available
        if hasattr(emb_np, "__array__"):
            import numpy as np  # noqa: F401

            emb_arr = mx.eval(emb_np)
            # mlx.core arrays support np.array(arr) conversion
            import numpy as np

            emb_arr = np.array(emb_np, copy=False)
        else:
            import numpy as np

            emb_arr = np.array(emb_np.tolist(), dtype=np.float32)

        for b, d in enumerate(batch):
            try:
                row_ids = ids_np[b]
                row_attn = attn_np[b]
                kept_idx = [
                    i
                    for i in range(len(row_ids))
                    if row_attn[i] == 1 and int(row_ids[i]) not in skip_set
                ]
                if not kept_idx:
                    dropped.append((str(d["doc_id"]), "0-token output after masking"))
                    continue
                kept_ids = [int(row_ids[i]) for i in kept_idx]
                kept_emb = emb_arr[b, kept_idx, :]
                # Cast to float32 list-of-list for the shared writer.
                vectors = kept_emb.astype("float32").tolist()
                encoded.append(
                    EncodedDoc(
                        doc_id=str(d["doc_id"]),
                        token_ids=kept_ids,
                        vectors=vectors,
                    )
                )
            except Exception as e:  # noqa: BLE001
                dropped.append((str(d["doc_id"]), f"post-process failed: {e!r}"))

    if not encoded:
        raise SystemExit("no docs encoded successfully — aborting before writing")
    return encoded, dropped, dim


def main() -> None:
    args = parse_args()

    if args.weights is None:
        from export_colbert_to_mlx import _safe_name  # type: ignore

        args.weights = (
            Path(__file__).resolve().parent / ".cache" / _safe_name(args.model) / "mlx_weights.safetensors"
        )

    docs = read_docs_jsonl(args.docs, args.max_docs)
    if not docs:
        raise SystemExit(f"{args.docs}: no records found")
    print(f"loaded {len(docs)} docs from {args.docs}", file=sys.stderr)

    encoded, dropped, dim = encode_docs_mlx(
        docs,
        model_name=args.model,
        weights_path=args.weights,
        dtype=args.dtype,
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
    _write_metadata_shared(
        meta_path,
        mode="docs",
        model_name=args.model,
        dim=dim,
        n_docs=stats["n_docs"],
        n_tokens=stats["n_tokens"],
        doc_ids=[d.doc_id for d in encoded],
        dropped=dropped,
        docs_jsonl=args.docs,
        tool="tools/encode_mlx.py",
        tool_version="0.1",
    )
    print(f"wrote metadata {meta_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
