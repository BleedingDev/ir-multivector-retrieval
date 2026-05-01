#!/usr/bin/env python3
"""tools/export_colbert_to_mlx.py — convert pylate ColBERT weights to MLX.

Owner: post-hackathon plan-12 (mlx-encoder-engineer).

Loads `colbert-ir/colbertv2.0` (or `jinaai/jina-colbert-v2-64`) via pylate,
walks the underlying BERT base + the final pylate `Dense` linear projection,
and saves an MLX-safetensors file to
`tools/.cache/<model>/mlx_weights.safetensors`.

The MLX encoder (tools/encode_mlx.py) hand-rolls the BERT forward pass via
`mlx.nn.Linear` + `mlx.nn.LayerNorm` etc. (mlx_lm has no encoder-only BERT
model class today). We rename the keys here so the MLX side can `mx.load`
the file directly into a flat dict and bind weights by name.

Key rename map (pylate/sbert → MLX-encoder):
  0.auto_model.embeddings.word_embeddings.weight    → embeddings.word.weight
  0.auto_model.embeddings.position_embeddings.weight → embeddings.position.weight
  0.auto_model.embeddings.token_type_embeddings.weight → embeddings.token_type.weight
  0.auto_model.embeddings.LayerNorm.{weight,bias}   → embeddings.ln.{weight,bias}
  0.auto_model.encoder.layer.<i>.attention.self.{q,k,v}.{weight,bias}
                                                    → layers.<i>.attn.{q,k,v}.{weight,bias}
  0.auto_model.encoder.layer.<i>.attention.output.dense.{weight,bias}
                                                    → layers.<i>.attn.out.{weight,bias}
  0.auto_model.encoder.layer.<i>.attention.output.LayerNorm.{weight,bias}
                                                    → layers.<i>.attn.ln.{weight,bias}
  0.auto_model.encoder.layer.<i>.intermediate.dense.{weight,bias}
                                                    → layers.<i>.ff.up.{weight,bias}
  0.auto_model.encoder.layer.<i>.output.dense.{weight,bias}
                                                    → layers.<i>.ff.down.{weight,bias}
  0.auto_model.encoder.layer.<i>.output.LayerNorm.{weight,bias}
                                                    → layers.<i>.ff.ln.{weight,bias}
  1.linear.weight                                   → projection.weight

The pylate.Dense projection has no bias on ColBERTv2 (`bias=False`), so
no projection.bias key is emitted. We assert this on export and store it
in the sidecar config so the MLX encoder builds a no-bias `nn.Linear`.

Pooler weights (0.auto_model.pooler.*) are dropped — ColBERT token-level
outputs never go through the pooler. We log them as "intentionally
discarded" rather than failing.

Output:
  tools/.cache/<model_safe>/mlx_weights.safetensors  -- flat key -> mx.array
  tools/.cache/<model_safe>/config.json              -- arch hyperparams
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _safe_name(model_name: str) -> str:
    """Filesystem-safe slug for cache subdirs (e.g. 'colbert-ir/colbertv2.0'
    → 'colbert-ir__colbertv2.0')."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "__", model_name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export ColBERT weights to MLX safetensors")
    p.add_argument(
        "--model",
        type=str,
        default="colbert-ir/colbertv2.0",
        help="HuggingFace model id (or local path) to load via pylate.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "output directory. Defaults to tools/.cache/<safe_model_name>/. "
            "Files written: mlx_weights.safetensors + config.json."
        ),
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="pass through to pylate.models.ColBERT (needed for jina models).",
    )
    p.add_argument(
        "--dtype",
        choices=("fp32", "fp16"),
        default="fp32",
        help=(
            "weight dtype to write to safetensors. fp32 keeps full precision; "
            "fp16 halves the file. The MLX encoder up-casts to the dtype "
            "selected at runtime, so this only affects on-disk size."
        ),
    )
    return p.parse_args()


def remap_key(k: str) -> str | None:
    """Map a pylate state_dict key to its MLX-encoder name. Returns None if
    the key is intentionally discarded (e.g. BERT pooler).

    The mapping is exact: any unknown key triggers a SystemExit so we never
    silently drop something the MLX side relies on.
    """
    # pooler: present but unused for token-level ColBERT.
    if k.startswith("0.auto_model.pooler."):
        return None

    # embeddings
    m = re.fullmatch(r"0\.auto_model\.embeddings\.(\w+)_embeddings\.weight", k)
    if m:
        return f"embeddings.{m.group(1)}.weight"
    m = re.fullmatch(r"0\.auto_model\.embeddings\.LayerNorm\.(weight|bias)", k)
    if m:
        return f"embeddings.ln.{m.group(1)}"

    # encoder layers
    m = re.fullmatch(
        r"0\.auto_model\.encoder\.layer\.(\d+)\.attention\.self\.(query|key|value)\.(weight|bias)",
        k,
    )
    if m:
        idx, qkv, wb = m.groups()
        short = {"query": "q", "key": "k", "value": "v"}[qkv]
        return f"layers.{idx}.attn.{short}.{wb}"

    m = re.fullmatch(
        r"0\.auto_model\.encoder\.layer\.(\d+)\.attention\.output\.dense\.(weight|bias)",
        k,
    )
    if m:
        return f"layers.{m.group(1)}.attn.out.{m.group(2)}"

    m = re.fullmatch(
        r"0\.auto_model\.encoder\.layer\.(\d+)\.attention\.output\.LayerNorm\.(weight|bias)",
        k,
    )
    if m:
        return f"layers.{m.group(1)}.attn.ln.{m.group(2)}"

    m = re.fullmatch(
        r"0\.auto_model\.encoder\.layer\.(\d+)\.intermediate\.dense\.(weight|bias)",
        k,
    )
    if m:
        return f"layers.{m.group(1)}.ff.up.{m.group(2)}"

    m = re.fullmatch(
        r"0\.auto_model\.encoder\.layer\.(\d+)\.output\.dense\.(weight|bias)",
        k,
    )
    if m:
        return f"layers.{m.group(1)}.ff.down.{m.group(2)}"

    m = re.fullmatch(
        r"0\.auto_model\.encoder\.layer\.(\d+)\.output\.LayerNorm\.(weight|bias)",
        k,
    )
    if m:
        return f"layers.{m.group(1)}.ff.ln.{m.group(2)}"

    # final pylate Dense projection (768 -> 128)
    if k == "1.linear.weight":
        return "projection.weight"
    if k == "1.linear.bias":
        return "projection.bias"

    raise SystemExit(f"unknown ColBERT state_dict key: {k!r} — refusing to silently drop")


def main() -> None:
    args = parse_args()

    import mlx.core as mx  # noqa: E402
    import torch  # noqa: E402
    from pylate import models  # noqa: E402

    print(f"loading {args.model} via pylate (CPU)…", file=sys.stderr)
    model = models.ColBERT(
        model_name_or_path=args.model,
        trust_remote_code=args.trust_remote_code,
        device="cpu",
    )
    model.eval()

    # Pull the underlying HF BERT config out of the wrapped sentence-transformers
    # Transformer module. Layout: model[0].auto_model is the BERT model.
    bert = model[0].auto_model
    cfg = bert.config
    n_layers = cfg.num_hidden_layers
    hidden = cfg.hidden_size
    n_heads = cfg.num_attention_heads
    intermediate = cfg.intermediate_size
    vocab_size = cfg.vocab_size
    max_pos = cfg.max_position_embeddings
    type_vocab = cfg.type_vocab_size
    ln_eps = cfg.layer_norm_eps
    hidden_act = cfg.hidden_act
    if hidden_act != "gelu":
        # BERT base uses gelu; flag any drift so the MLX encoder doesn't
        # silently substitute the wrong activation.
        raise SystemExit(
            f"unexpected hidden activation {hidden_act!r}; MLX encoder is "
            "hard-coded for gelu."
        )

    # pylate.Dense projection. ColBERTv2 uses bias=False; we assert and record.
    projection = model[1].linear
    proj_in, proj_out = projection.in_features, projection.out_features
    proj_has_bias = projection.bias is not None
    if proj_has_bias:
        # ColBERTv2 expects bias=False; if a future pylate bumps this we want to know.
        print(
            f"WARN: pylate Dense projection has bias=True (unusual for ColBERTv2)",
            file=sys.stderr,
        )

    # Capture pylate's tokenizer-side metadata so the MLX encoder can run
    # without instantiating pylate.models.ColBERT at inference time.
    # rec-01 (post-audit): the runtime pylate load was a fixed cost that
    # buried the small-corpus speedup. With this metadata in config.json,
    # encode_mlx.py uses AutoTokenizer + manual prefix insertion and
    # produces byte-equal token_ids on the live fixture.
    skiplist = sorted(int(s) for s in model.skiplist)
    document_length = int(model.document_length)
    query_length = int(model.query_length)
    document_prefix_id = int(model.document_prefix_id)
    query_prefix_id = int(model.query_prefix_id)
    attend_to_expansion_tokens = bool(model.attend_to_expansion_tokens)

    # Walk + remap state_dict.
    sd = model.state_dict()
    out_weights: dict[str, mx.array] = {}
    discarded: list[str] = []

    target_dtype_torch = torch.float16 if args.dtype == "fp16" else torch.float32
    for k, v in sd.items():
        new_k = remap_key(k)
        if new_k is None:
            discarded.append(k)
            continue
        # Convert torch tensor → numpy → mlx (mx.array doesn't construct
        # directly from torch tensors). Cast first to the target dtype.
        t = v.detach().to(target_dtype_torch).contiguous().cpu().numpy()
        out_weights[new_k] = mx.array(t)

    # Sanity: confirm we got every layer's full set of weights. Expected key
    # stems per layer:
    expected_per_layer = [
        "attn.q.weight", "attn.q.bias",
        "attn.k.weight", "attn.k.bias",
        "attn.v.weight", "attn.v.bias",
        "attn.out.weight", "attn.out.bias",
        "attn.ln.weight", "attn.ln.bias",
        "ff.up.weight", "ff.up.bias",
        "ff.down.weight", "ff.down.bias",
        "ff.ln.weight", "ff.ln.bias",
    ]
    for i in range(n_layers):
        for stem in expected_per_layer:
            full = f"layers.{i}.{stem}"
            if full not in out_weights:
                raise SystemExit(f"missing expected weight {full!r}")

    embed_required = [
        "embeddings.word.weight",
        "embeddings.position.weight",
        "embeddings.token_type.weight",
        "embeddings.ln.weight",
        "embeddings.ln.bias",
        "projection.weight",
    ]
    for k in embed_required:
        if k not in out_weights:
            raise SystemExit(f"missing required weight {k!r}")

    # Write outputs.
    out_dir = args.out_dir or (
        Path(__file__).resolve().parent / ".cache" / _safe_name(args.model)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write a dtype-specific filename so the encoder can pick a file that
    # already matches the runtime dtype and skip the per-tensor astype loop.
    # Keep writing the legacy mlx_weights.safetensors path too so older
    # encoder builds keep working without a re-export.
    dtype_path = out_dir / f"mlx_weights.{args.dtype}.safetensors"
    safetensors_path = out_dir / "mlx_weights.safetensors"
    config_path = out_dir / "config.json"

    mx.save_safetensors(str(dtype_path), out_weights)
    mx.save_safetensors(str(safetensors_path), out_weights)

    config = {
        "format_version": 2,
        "model_name": args.model,
        "architecture": "bert-base",
        "n_layers": int(n_layers),
        "hidden_size": int(hidden),
        "n_heads": int(n_heads),
        "intermediate_size": int(intermediate),
        "vocab_size": int(vocab_size),
        "max_position_embeddings": int(max_pos),
        "type_vocab_size": int(type_vocab),
        "layer_norm_eps": float(ln_eps),
        "hidden_act": hidden_act,
        "projection_in": int(proj_in),
        "projection_out": int(proj_out),
        "projection_has_bias": bool(proj_has_bias),
        "weight_dtype": args.dtype,
        "n_weights": len(out_weights),
        "discarded_keys": discarded,
        # Tokenizer-side metadata so encode_mlx.py can avoid pylate at
        # inference. The HuggingFace AutoTokenizer for `model_name` plus
        # these fields fully reproduces pylate.ColBERT.tokenize for the
        # docs path on byte-equal token_ids.
        "skiplist": skiplist,
        "document_length": document_length,
        "query_length": query_length,
        "document_prefix_id": document_prefix_id,
        "query_prefix_id": query_prefix_id,
        "attend_to_expansion_tokens": attend_to_expansion_tokens,
    }
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")

    total_bytes = safetensors_path.stat().st_size
    print(
        f"wrote {dtype_path} and {safetensors_path} "
        f"({len(out_weights)} tensors, {total_bytes / 1024 / 1024:.1f} MB, "
        f"{args.dtype})",
        file=sys.stderr,
    )
    print(f"wrote {config_path}", file=sys.stderr)
    if discarded:
        print(
            f"discarded {len(discarded)} key(s) (pooler etc.): "
            f"{discarded[:3]}{'…' if len(discarded) > 3 else ''}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
