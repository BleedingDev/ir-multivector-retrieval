#!/usr/bin/env python3
"""Run a single named bench cell from run_bench.py and append a JSON line
to wave2-rebench/<rec>.jsonl.

Usage:
    python bench_cell.py <rec_label> "<cell_label>"
    # e.g.
    python bench_cell.py rec1 "100doc/encode_mlx/fp16/batch=128"
    python bench_cell.py rec1 "1000doc/encode_mlx/fp16/batch=128/sort"

Discipline mirrors A2's run_bench.py (≥2 warm-ups discarded, 3 measured
runs, cooldown gap between back-to-back runs).
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / ".codex" / "bench-outputs"))
import run_bench  # type: ignore

OUT_DIR = REPO / ".codex" / "bench-outputs" / "wave2-rebench"


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(f"usage: {sys.argv[0]} <rec_label> <cell_label> [<cell_label> ...]")
    rec = sys.argv[1]
    targets = sys.argv[2:]
    out_path = OUT_DIR / f"{rec}.jsonl"

    cells = run_bench.cells_for(None)
    by_label = {c[0]: c[1] for c in cells}
    for t in targets:
        if t not in by_label:
            sys.exit(f"unknown cell: {t}\nknown:\n  " + "\n  ".join(by_label))

    # Override RESULTS_JSONL to point at our rec-specific file. Cooldown
    # between cells preserved.
    run_bench.RESULTS_JSONL = out_path
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for i, t in enumerate(targets):
        if i > 0:
            print(f"\n[cooldown {run_bench.COOLDOWN_BETWEEN_CELLS}s]", flush=True)
            time.sleep(run_bench.COOLDOWN_BETWEEN_CELLS)
        run_bench.bench_cell(t, by_label[t])

    print(f"\nappended {len(targets)} cell(s) to {out_path}", flush=True)


if __name__ == "__main__":
    main()
