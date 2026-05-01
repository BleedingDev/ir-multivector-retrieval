#!/usr/bin/env python3
"""tools/export_colbert_to_mlx.py — convert pylate ColBERT weights to MLX.

Owner: post-hackathon plan-12 (mlx-encoder-engineer).

Loads `colbert-ir/colbertv2.0` (or `jinaai/jina-colbert-v2-64`) via pylate,
walks the underlying BERT state_dict + the final pylate linear projection,
and saves an MLX-compatible safetensors file to
`tools/.cache/<model>/mlx_weights.safetensors`. The MLX encoder consumes
that file directly; HF's tokenizer is reused on the Python side.
"""
from __future__ import annotations

import sys


def main() -> None:
    print("export_colbert_to_mlx.py: not yet implemented (Phase A scaffold)", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
