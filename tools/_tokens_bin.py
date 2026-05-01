#!/usr/bin/env python3
"""tools/_tokens_bin.py — shared writer for the tokens.bin v2 format.

Used by tools/encode.py (PyTorch path) and tools/encode_mlx.py (MLX path)
so both encoders produce byte-identical output. The format constants and
writer mirror src/io/token_dump.zig::parseBytes — any change here MUST
update the Zig parser in lockstep.

This module is intentionally tiny and dependency-free (only `struct`,
`json`, `pathlib`, stdlib datetime). Importing it does not pay the
torch/pylate import cost.
"""
from __future__ import annotations

import sys


def main() -> None:
    print("_tokens_bin.py: not yet implemented (Phase A scaffold)", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
