#!/usr/bin/env python3
"""tools/encode_mlx.py — MLX (Apple Silicon native) port of tools/encode.py.

Owner: post-hackathon plan-12 (mlx-encoder-engineer).

EXPERIMENTAL until the 4-gate parity contract in
`tests/live/mlx_parity.py` is green:
  1. token-id byte equality vs encode.py
  2. per-token cosine >= 0.998 (min >= 0.99) vs encode.py fp16
  3. two-run determinism (byte-equal output)
  4. downstream TAC clustering overlap (kappa_per_token + centroid means)

Output format is the same `tokens.bin` v2 (TAC_TKN1 magic) that
`src/io/token_dump.zig` parses; the writer comes from
`tools/_tokens_bin.py` so encode.py and encode_mlx.py share byte-for-byte
identical I/O.
"""
from __future__ import annotations

import sys


def main() -> None:
    print("encode_mlx.py: not yet implemented (Phase A scaffold)", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
