#!/usr/bin/env python3
"""Post-audit MLX rebench harness (cell-by-cell, restartable).

Discipline:
- ≥2 warm-up runs discarded (HF cache + MLX kernel JIT warmup)
- ≥3 measured runs, report median + min + max
- Cooldown gap (sleep) between cells to keep the SoC in a similar thermal band
- Each cell timed by wall-clock around the encoder subprocess; we measure the
  same thing the doc claims (end-to-end CLI wall-clock)
- After each cell finishes, append a single JSON line to rebench_results.jsonl
  so a partial run is never lost
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools"
VENV = TOOLS / ".venv" / "bin" / "python"
MLX_WEIGHTS = TOOLS / ".cache" / "colbert-ir__colbertv2.0" / "mlx_weights.safetensors"
LIVE_FIXTURE = REPO / "tests" / "fixtures" / "live" / "docs.jsonl"
JIRA_1000 = REPO / ".codex" / "bench-outputs" / "jira_1000.jsonl"
OUT_DIR = REPO / ".codex" / "bench-outputs"
RESULTS_JSONL = OUT_DIR / "rebench_results.jsonl"

WARMUPS = 2
MEASURED = 3
COOLDOWN_BETWEEN_RUNS = 4      # seconds between back-to-back runs in the same cell
COOLDOWN_BETWEEN_CELLS = 25    # seconds between cells


def run_once(cmd: list[str]) -> float:
    t0 = time.perf_counter()
    res = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if res.returncode != 0:
        sys.stderr.write(f"FAILED: {' '.join(cmd)}\nstderr:\n{res.stderr}\n")
        sys.exit(2)
    return elapsed


def bench_cell(label: str, cmd: list[str]) -> dict:
    print(f"\n=== {label} ===", flush=True)
    print(f"warmups (discarded): {WARMUPS}", flush=True)
    for i in range(WARMUPS):
        t = run_once(cmd)
        print(f"  warmup[{i}]: {t:.2f}s", flush=True)
        time.sleep(COOLDOWN_BETWEEN_RUNS)
    samples = []
    print(f"measured: {MEASURED}", flush=True)
    for i in range(MEASURED):
        t = run_once(cmd)
        samples.append(t)
        print(f"  measured[{i}]: {t:.2f}s", flush=True)
        time.sleep(COOLDOWN_BETWEEN_RUNS)
    median = statistics.median(samples)
    smin = min(samples)
    smax = max(samples)
    print(f"  median={median:.2f}s min={smin:.2f}s max={smax:.2f}s", flush=True)
    rec = {
        "label": label,
        "cmd": cmd,
        "warmups_discarded": WARMUPS,
        "samples": samples,
        "median_s": round(median, 3),
        "min_s": round(smin, 3),
        "max_s": round(smax, 3),
    }
    with RESULTS_JSONL.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def cells_for(target: str | None) -> list[tuple[str, list[str]]]:
    all_cells: list[tuple[str, list[str]]] = [
        # 100-doc fixture, encode.py fp32 cpu (canonical baseline)
        (
            "100doc/encode.py/fp32/cpu/batch=32",
            [
                str(VENV),
                str(TOOLS / "encode.py"),
                "--docs", str(LIVE_FIXTURE),
                "--out", str(OUT_DIR / "100doc_pt_fp32.bin"),
                "--device", "cpu",
                "--dtype", "fp32",
                "--batch", "32",
            ],
        ),
        # 100-doc fixture, encode_mlx fp16 batch=128
        (
            "100doc/encode_mlx/fp16/batch=128",
            [
                str(VENV),
                str(TOOLS / "encode_mlx.py"),
                "--docs", str(LIVE_FIXTURE),
                "--out", str(OUT_DIR / "100doc_mlx_fp16_b128.bin"),
                "--dtype", "fp16",
                "--batch", "128",
                "--weights", str(MLX_WEIGHTS),
            ],
        ),
        # 100-doc fixture, encode_mlx fp32 batch=128
        (
            "100doc/encode_mlx/fp32/batch=128",
            [
                str(VENV),
                str(TOOLS / "encode_mlx.py"),
                "--docs", str(LIVE_FIXTURE),
                "--out", str(OUT_DIR / "100doc_mlx_fp32_b128.bin"),
                "--dtype", "fp32",
                "--batch", "128",
                "--weights", str(MLX_WEIGHTS),
            ],
        ),
        # 1000-doc Jira, encode.py fp32 cpu
        (
            "1000doc/encode.py/fp32/cpu/batch=32",
            [
                str(VENV),
                str(TOOLS / "encode.py"),
                "--docs", str(JIRA_1000),
                "--out", str(OUT_DIR / "1000doc_pt_fp32.bin"),
                "--device", "cpu",
                "--dtype", "fp32",
                "--batch", "32",
            ],
        ),
        # 1000-doc Jira, encode_mlx fp16 batch=32 (parity-harness config)
        (
            "1000doc/encode_mlx/fp16/batch=32",
            [
                str(VENV),
                str(TOOLS / "encode_mlx.py"),
                "--docs", str(JIRA_1000),
                "--out", str(OUT_DIR / "1000doc_mlx_fp16_b32.bin"),
                "--dtype", "fp16",
                "--batch", "32",
                "--weights", str(MLX_WEIGHTS),
            ],
        ),
        # 1000-doc Jira, encode_mlx fp16 batch=128
        (
            "1000doc/encode_mlx/fp16/batch=128",
            [
                str(VENV),
                str(TOOLS / "encode_mlx.py"),
                "--docs", str(JIRA_1000),
                "--out", str(OUT_DIR / "1000doc_mlx_fp16_b128.bin"),
                "--dtype", "fp16",
                "--batch", "128",
                "--weights", str(MLX_WEIGHTS),
            ],
        ),
        # 1000-doc Jira, encode_mlx fp16 batch=128 + sort
        (
            "1000doc/encode_mlx/fp16/batch=128/sort",
            [
                str(VENV),
                str(TOOLS / "encode_mlx.py"),
                "--docs", str(JIRA_1000),
                "--out", str(OUT_DIR / "1000doc_mlx_fp16_b128_sort.bin"),
                "--dtype", "fp16",
                "--batch", "128",
                "--sort-by-length",
                "--weights", str(MLX_WEIGHTS),
            ],
        ),
        # 1000-doc Jira, encode_mlx fp32 batch=128 + sort
        (
            "1000doc/encode_mlx/fp32/batch=128/sort",
            [
                str(VENV),
                str(TOOLS / "encode_mlx.py"),
                "--docs", str(JIRA_1000),
                "--out", str(OUT_DIR / "1000doc_mlx_fp32_b128_sort.bin"),
                "--dtype", "fp32",
                "--batch", "128",
                "--sort-by-length",
                "--weights", str(MLX_WEIGHTS),
            ],
        ),
    ]
    if target is None:
        return all_cells
    sel = [c for c in all_cells if c[0] == target]
    if not sel:
        sys.exit(f"unknown cell label: {target}\nknown:\n  " + "\n  ".join(c[0] for c in all_cells))
    return sel


def main() -> None:
    for p in (VENV, MLX_WEIGHTS, LIVE_FIXTURE, JIRA_1000):
        if not p.exists():
            sys.exit(f"missing prereq: {p}")

    target = sys.argv[1] if len(sys.argv) > 1 else None
    cells = cells_for(target)

    for i, (label, cmd) in enumerate(cells):
        if i > 0:
            print(f"\n[cooldown {COOLDOWN_BETWEEN_CELLS}s before next cell]", flush=True)
            time.sleep(COOLDOWN_BETWEEN_CELLS)
        bench_cell(label, cmd)

    print(f"\nappended {len(cells)} cell(s) to {RESULTS_JSONL}", flush=True)


if __name__ == "__main__":
    main()
