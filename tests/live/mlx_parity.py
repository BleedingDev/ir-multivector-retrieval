#!/usr/bin/env python3
"""tests/live/mlx_parity.py — 4-gate parity contract for tools/encode_mlx.py.

Run:
    tools/.venv/bin/python tests/live/mlx_parity.py --mlx-parity

Gates (each must pass; none skippable):
  1. Token-id byte equality between encode.py and encode_mlx.py
  2. Per-token cosine >= 0.998 mean (min >= 0.99) on the live fixture
  3. Two-run MLX determinism (byte-equal output across two runs)
  4. Downstream TAC clustering overlap (kappa_per_token + centroid means)
"""
from __future__ import annotations

import sys


def main() -> None:
    print("mlx_parity.py: not yet implemented (Phase A scaffold)", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
