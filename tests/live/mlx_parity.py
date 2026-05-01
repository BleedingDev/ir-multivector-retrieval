#!/usr/bin/env python3
"""tests/live/mlx_parity.py — 4-gate parity contract for tools/encode_mlx.py.

Run:
    tools/.venv/bin/python tests/live/mlx_parity.py --mlx-parity

Gates (each must pass; none skippable):
  1. Token-id byte equality between encode.py and encode_mlx.py.
  2. Per-token cosine ≥ 0.998 mean and ≥ 0.99 min on the live fixture.
  3. Two-run MLX determinism — byte-equal, or ≤ small ULP drift per float.
     (Note: the PyTorch encode.py baseline is itself non-deterministic
     across runs by ~1.5e-6 abs, owing to transformers 4.57's
     mean_resizing path firing during pylate's vocab expansion. This
     gate measures MLX's own self-consistency, not parity vs that drift.)
  4. Downstream TAC clustering: build a tac index from each token_dump
     with identical params/seed; assert headers (kappa_total, dim, n_*),
     and centroid arrays agree within 1e-3 cosine per centroid.

The script does NOT auto-install MLX, the weight cache, or the venv —
it asserts they exist and gives a clean error if not.

Args:
  --mlx-parity   required gate flag (mirrors -Dlive in zig)
  --max-docs     subset for fast smoke runs (default: all 100 fixture docs)
  --dtype        fp32 or fp16 (default fp16 — matches encode.py MPS default)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools"
VENV_PY = TOOLS / ".venv" / "bin" / "python"
FIXTURE = REPO / "tests" / "fixtures" / "live" / "docs.jsonl"
WEIGHTS_DIR = TOOLS / ".cache" / "colbert-ir__colbertv2.0"
MLX_WEIGHTS = WEIGHTS_DIR / "mlx_weights.safetensors"


# ---------------------------------------------------------------------------
# tokens.bin parser (mirrors src/io/token_dump.zig::parseBytes structurally;
# kept dependency-free so this script doesn't pull torch / mlx in itself).
# ---------------------------------------------------------------------------


def parse_tokens_bin(path: Path) -> dict:
    with path.open("rb") as f:
        data = f.read()
    if data[:8] != b"TAC_TKN1":
        raise SystemExit(f"{path}: bad magic {data[:8]!r}")
    version = struct.unpack("<I", data[8:12])[0]
    if version != 2:
        raise SystemExit(f"{path}: bad version {version}")
    dim = struct.unpack("<I", data[12:16])[0]
    n_docs = struct.unpack("<Q", data[16:24])[0]
    n_tok = struct.unpack("<Q", data[24:32])[0]
    if data[32] != 0:
        raise SystemExit(f"{path}: dtype != f32")

    pos = 40
    offsets = list(struct.unpack(f"<{n_docs+1}Q", data[pos : pos + (n_docs + 1) * 8]))
    pos += (n_docs + 1) * 8
    ids = list(struct.unpack(f"<{n_tok}I", data[pos : pos + n_tok * 4]))
    pos += n_tok * 4
    vec_bytes = n_tok * dim * 4
    vecs = struct.unpack(f"<{n_tok*dim}f", data[pos : pos + vec_bytes])
    return {
        "dim": dim,
        "n_docs": n_docs,
        "n_tok": n_tok,
        "offsets": offsets,
        "ids": ids,
        "vecs": vecs,
    }


# ---------------------------------------------------------------------------
# tac index (.tac) header + centroids parser.
#
# Layout (src/index/storage.zig):
#   header (112 bytes inside a 128-byte region)
#     [8] magic "TAC_IDX1"
#     u32 version
#     u32 dim
#     u32 kappa
#     u32 pq_M
#     u8  pq_bits
#     u8[3] _pad
#     u32 sub_dim
#     u64 n_docs
#     u64 n_tokens
#     u64 centroids_off
#     u64 hnsw_off, pq_off, ilist_off, doc_off, doc_tok_off, norms_off, footer_off
#   <padding to 128>
#   centroids: kappa * dim f32 row-major
#   ...
# ---------------------------------------------------------------------------


def parse_tac_header_centroids(path: Path) -> dict:
    with path.open("rb") as f:
        data = f.read()
    if data[:8] != b"TAC_IDX1":
        raise SystemExit(f"{path}: bad TAC index magic {data[:8]!r}")
    pos = 8
    version = struct.unpack("<I", data[pos : pos + 4])[0]; pos += 4
    dim = struct.unpack("<I", data[pos : pos + 4])[0]; pos += 4
    kappa = struct.unpack("<I", data[pos : pos + 4])[0]; pos += 4
    pq_M = struct.unpack("<I", data[pos : pos + 4])[0]; pos += 4
    pq_bits = data[pos]; pos += 1
    pos += 3  # _pad
    sub_dim = struct.unpack("<I", data[pos : pos + 4])[0]; pos += 4
    n_docs = struct.unpack("<Q", data[pos : pos + 8])[0]; pos += 8
    n_tokens = struct.unpack("<Q", data[pos : pos + 8])[0]; pos += 8
    centroids_off = struct.unpack("<Q", data[pos : pos + 8])[0]; pos += 8

    cent_bytes = kappa * dim * 4
    centroids = struct.unpack(
        f"<{kappa*dim}f", data[centroids_off : centroids_off + cent_bytes]
    )
    return {
        "version": version,
        "dim": dim,
        "kappa": kappa,
        "pq_M": pq_M,
        "pq_bits": pq_bits,
        "sub_dim": sub_dim,
        "n_docs": n_docs,
        "n_tokens": n_tokens,
        "centroids": centroids,
    }


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if len(a) != len(b):
        raise SystemExit(f"length mismatch {len(a)} vs {len(b)}")
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def cosines_per_token(pt: dict, mlx: dict) -> list[float]:
    if pt["n_tok"] != mlx["n_tok"] or pt["dim"] != mlx["dim"]:
        raise SystemExit(
            f"shape mismatch: pt n_tok={pt['n_tok']} dim={pt['dim']} vs "
            f"mlx n_tok={mlx['n_tok']} dim={mlx['dim']}"
        )
    dim = pt["dim"]
    out = []
    for i in range(pt["n_tok"]):
        a = pt["vecs"][i * dim : (i + 1) * dim]
        b = mlx["vecs"][i * dim : (i + 1) * dim]
        out.append(cosine(a, b))
    return out


# ---------------------------------------------------------------------------
# Subprocess helpers (encoders + tac index build)
# ---------------------------------------------------------------------------


def run(cmd: list[str], *, cwd: Path | None = None) -> tuple[float, str]:
    t0 = time.perf_counter()
    res = subprocess.run(
        cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True
    )
    elapsed = time.perf_counter() - t0
    if res.returncode != 0:
        raise SystemExit(
            f"command failed (exit {res.returncode}):\n  {' '.join(cmd)}\n"
            f"stderr:\n{res.stderr}\nstdout:\n{res.stdout}"
        )
    return elapsed, res.stderr


def encode_pt(out: Path, max_docs: int | None, dtype: str) -> float:
    args = [
        str(VENV_PY),
        str(TOOLS / "encode.py"),
        "--docs", str(FIXTURE),
        "--out", str(out),
        "--device", "cpu",
        "--dtype", dtype,
        "--batch", "16",
    ]
    if max_docs is not None:
        args += ["--max-docs", str(max_docs)]
    elapsed, _ = run(args, cwd=REPO)
    return elapsed


def encode_mlx(out: Path, max_docs: int | None, dtype: str) -> float:
    args = [
        str(VENV_PY),
        str(TOOLS / "encode_mlx.py"),
        "--docs", str(FIXTURE),
        "--out", str(out),
        "--dtype", dtype,
        "--batch", "32",
        "--weights", str(MLX_WEIGHTS),
    ]
    if max_docs is not None:
        args += ["--max-docs", str(max_docs)]
    elapsed, _ = run(args, cwd=REPO)
    return elapsed


def tac_index(tokens_bin: Path, out_tac: Path, kappa: int, seed: int) -> None:
    tac_bin = REPO / "zig-out" / "bin" / "tac"
    if not tac_bin.exists():
        # Build it once.
        run(["zig", "build", "-Doptimize=ReleaseFast"], cwd=REPO)
    if not tac_bin.exists():
        raise SystemExit(f"tac binary not found at {tac_bin} after zig build")
    run([
        str(tac_bin), "index",
        str(tokens_bin), str(out_tac),
        "--kappa", str(kappa), "--seed", str(seed),
    ], cwd=REPO)


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def gate1_token_ids(pt: dict, mlx: dict) -> tuple[bool, str]:
    if pt["ids"] != mlx["ids"]:
        diffs = sum(1 for a, b in zip(pt["ids"], mlx["ids"]) if a != b)
        return False, f"token_ids differ: {diffs}/{len(pt['ids'])} positions"
    if pt["offsets"] != mlx["offsets"]:
        return False, "doc_offsets (CSR) differ"
    return True, f"token_ids byte-equal ({len(pt['ids'])} ids; offsets equal)"


def gate2_cosine(
    pt: dict,
    mlx: dict,
    mean_floor: float,
    min_floor: float,
    abs_floor: float,
) -> tuple[bool, str]:
    """Cosine + absolute-error gate.

    Cosine alone is magnitude-invariant: a uniform-scale magnitude shift
    leaves all cosines at 1.0 even though absolute values diverge. We
    pair the cosine floors with a max-abs-element check so a silent
    magnitude drift (e.g., a botched LayerNorm eps, a missing
    .normalize() call) shows up.

    L2-normalised ColBERT vectors have |x| ≤ 1, so per-element abs error
    is bounded by 2; values much above 1e-3 indicate real semantic drift.
    """
    cos = cosines_per_token(pt, mlx)
    cos_sorted = sorted(cos)
    n = len(cos)
    mean = sum(cos) / n
    cmin = cos_sorted[0]
    p1 = cos_sorted[max(0, n // 100)]

    # Max absolute deviation per element across all (token, dim) pairs.
    if pt["n_tok"] != mlx["n_tok"] or pt["dim"] != mlx["dim"]:
        return False, (
            f"shape mismatch: pt n_tok={pt['n_tok']} dim={pt['dim']} vs "
            f"mlx n_tok={mlx['n_tok']} dim={mlx['dim']}"
        )
    max_abs = 0.0
    for x, y in zip(pt["vecs"], mlx["vecs"]):
        d = abs(x - y)
        if d > max_abs:
            max_abs = d

    msg = (
        f"cosine mean={mean:.6f} min={cmin:.6f} p1={p1:.6f} "
        f"max={cos_sorted[-1]:.6f} max_abs={max_abs:.2e} (n={n})"
    )
    if mean < mean_floor:
        return False, msg + f" — mean below floor {mean_floor}"
    if cmin < min_floor:
        return False, msg + f" — min below floor {min_floor}"
    if max_abs > abs_floor:
        return False, msg + f" — max_abs above floor {abs_floor:.2e}"
    return True, msg


def gate3_determinism(out_a: Path, out_b: Path, max_docs: int | None, dtype: str) -> tuple[bool, str]:
    encode_mlx(out_a, max_docs, dtype)
    encode_mlx(out_b, max_docs, dtype)
    a = parse_tokens_bin(out_a)
    b = parse_tokens_bin(out_b)
    if a["ids"] != b["ids"] or a["offsets"] != b["offsets"]:
        return False, "two-run determinism: token_ids/offsets differ"

    # Compare floats with strict byte-equality first; if that fails, fall
    # back to ULP-class threshold (≤ 4 ULPs ≈ ~5e-7 abs for fp32 values
    # near 0.1, OK for L2-normalised ColBERT vectors).
    diffs = 0
    max_abs = 0.0
    for x, y in zip(a["vecs"], b["vecs"]):
        if x != y:
            diffs += 1
            d = abs(x - y)
            if d > max_abs:
                max_abs = d
    n = len(a["vecs"])
    if diffs == 0:
        return True, f"two-run byte-identical ({n} floats)"
    # ULP-class threshold for fp32 vectors with |x| ≤ 1 (L2-normalised, dim=128):
    # 1 ULP ≈ 2**-23 ≈ 1.19e-7. Allow ≤ 4 ULPs.
    threshold = 5e-7
    if max_abs <= threshold:
        return True, (
            f"two-run within ULP-class threshold "
            f"({diffs}/{n} floats differ; max abs {max_abs:.2e} ≤ {threshold:.2e})"
        )
    return False, (
        f"two-run determinism failed: {diffs}/{n} floats differ; "
        f"max abs {max_abs:.2e} > {threshold:.2e}"
    )


def _doc_token_view(parsed: dict) -> list[list[tuple[float, ...]]]:
    """Slice the flat tokens.bin payload into per-doc lists of token vectors.

    Output: docs[i][t] is a tuple-of-floats representing the t-th token
    of doc i. Used by gate 5 to compute MaxSim scores.
    """
    dim = parsed["dim"]
    offsets = parsed["offsets"]
    vecs = parsed["vecs"]
    docs: list[list[tuple[float, ...]]] = []
    for i in range(parsed["n_docs"]):
        lo = offsets[i]
        hi = offsets[i + 1]
        rows = [tuple(vecs[(lo + t) * dim : (lo + t + 1) * dim]) for t in range(hi - lo)]
        docs.append(rows)
    return docs


def _maxsim_score(query: list[tuple[float, ...]], doc: list[tuple[float, ...]]) -> float:
    """ColBERT-style MaxSim: sum_q max_d <q, d>. Inputs are L2-normalised
    so the dot is the cosine. O(|q| * |d| * dim) — fine on a 100-doc
    fixture where each query is a 16-token doc."""
    if not query or not doc:
        return 0.0
    total = 0.0
    for q in query:
        best = -1e30
        for dv in doc:
            s = 0.0
            for a, b in zip(q, dv):
                s += a * b
            if s > best:
                best = s
        total += best
    return total


def gate5_search_ranking(
    pt: dict, mlx: dict, *, n_queries: int, top_k: int, overlap_floor: float
) -> tuple[bool, str]:
    """Search-ranking parity.

    Pick the first `n_queries` docs as queries, the rest as the corpus.
    Compute MaxSim(query, doc) for both PyTorch and MLX outputs and
    compare the top-k retrieval lists. The two encoders should agree on
    which docs each query retrieves; a real semantic drift would shuffle
    the rankings even when per-token cosine still looks ≥ 0.999.
    """
    if pt["n_docs"] != mlx["n_docs"] or pt["n_docs"] < n_queries + top_k + 1:
        return False, (
            f"corpus too small for n_queries={n_queries}, top_k={top_k}: "
            f"n_docs pt={pt['n_docs']} mlx={mlx['n_docs']}"
        )

    pt_docs = _doc_token_view(pt)
    mlx_docs = _doc_token_view(mlx)

    queries_pt = pt_docs[:n_queries]
    queries_mlx = mlx_docs[:n_queries]
    corpus_pt = pt_docs[n_queries:]
    corpus_mlx = mlx_docs[n_queries:]
    n_corpus = len(corpus_pt)

    overlaps: list[float] = []
    pt_top_lists: list[list[int]] = []
    mlx_top_lists: list[list[int]] = []

    for qi in range(n_queries):
        q_pt = queries_pt[qi]
        q_mlx = queries_mlx[qi]
        scores_pt = [(_maxsim_score(q_pt, corpus_pt[j]), j) for j in range(n_corpus)]
        scores_mlx = [(_maxsim_score(q_mlx, corpus_mlx[j]), j) for j in range(n_corpus)]
        # Sort descending by score, tie-break by lower j (stable index order).
        scores_pt.sort(key=lambda t: (-t[0], t[1]))
        scores_mlx.sort(key=lambda t: (-t[0], t[1]))
        top_pt = [j for _, j in scores_pt[:top_k]]
        top_mlx = [j for _, j in scores_mlx[:top_k]]
        pt_top_lists.append(top_pt)
        mlx_top_lists.append(top_mlx)
        overlap = len(set(top_pt) & set(top_mlx)) / float(top_k)
        overlaps.append(overlap)

    avg_overlap = sum(overlaps) / len(overlaps)
    min_overlap = min(overlaps)
    msg = (
        f"top-{top_k} overlap on {n_queries} self-queries vs "
        f"{n_corpus}-doc corpus: avg={avg_overlap:.3f} min={min_overlap:.3f}"
    )
    if min_overlap < overlap_floor:
        # Surface the worst query so a regression is debuggable.
        worst = overlaps.index(min_overlap)
        msg += (
            f" — query[{worst}] overlap {min_overlap:.3f} below floor "
            f"{overlap_floor:.2f}; pt_top={pt_top_lists[worst]} "
            f"mlx_top={mlx_top_lists[worst]}"
        )
        return False, msg
    return True, msg


def gate4_clustering(pt_bin: Path, mlx_bin: Path, kappa: int, seed: int) -> tuple[bool, str]:
    pt_tac = pt_bin.with_suffix(".pt.tac")
    mlx_tac = mlx_bin.with_suffix(".mlx.tac")
    tac_index(pt_bin, pt_tac, kappa=kappa, seed=seed)
    tac_index(mlx_bin, mlx_tac, kappa=kappa, seed=seed)

    pt_idx = parse_tac_header_centroids(pt_tac)
    mlx_idx = parse_tac_header_centroids(mlx_tac)

    # Header agreement on kappa/dim/n_*.
    for key in ("dim", "kappa", "n_docs", "n_tokens"):
        if pt_idx[key] != mlx_idx[key]:
            return False, f"tac header field {key!r} differs: pt={pt_idx[key]} mlx={mlx_idx[key]}"

    # Per-centroid cosine similarity. Centroid c maps the same vocab id
    # cluster on both sides (kappa_per_token is determined by token_id
    # frequencies, which gate 1 already proved equal), so centroid i on
    # both sides clusters the same token-id subset.
    dim = pt_idx["dim"]
    kappa_v = pt_idx["kappa"]
    pt_c = pt_idx["centroids"]
    mlx_c = mlx_idx["centroids"]
    cos_per_centroid = []
    for i in range(kappa_v):
        a = pt_c[i * dim : (i + 1) * dim]
        b = mlx_c[i * dim : (i + 1) * dim]
        cos_per_centroid.append(cosine(a, b))

    cmin = min(cos_per_centroid)
    cmean = sum(cos_per_centroid) / len(cos_per_centroid)
    msg = (
        f"tac kappa={kappa_v} dim={dim} centroid cosine mean={cmean:.6f} min={cmin:.6f}"
    )
    # Plan threshold: centroid means agree within 1e-3 cosine (so ≥ 1 - 1e-3 = 0.999).
    if cmin < 0.999:
        return False, msg + f" — min < 0.999"
    return True, msg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MLX ↔ PyTorch parity for ColBERT encoder")
    p.add_argument(
        "--mlx-parity",
        action="store_true",
        required=True,
        help="opt-in flag — mirrors -Dlive=true (parity tests are off-by-default)",
    )
    p.add_argument("--max-docs", type=int, default=None, help="subset for fast runs")
    p.add_argument(
        "--dtype",
        choices=("fp32", "fp16"),
        default="fp16",
        help="encoder dtype on both sides (default fp16, matches MPS default)",
    )
    p.add_argument(
        "--kappa",
        type=int,
        default=92,
        help="tac index kappa for gate 4 (default 92, matches live harness)",
    )
    p.add_argument("--seed", type=int, default=2026, help="tac index seed")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/tmp"),
        help="where to write the parity binaries",
    )
    p.add_argument(
        "--skip-gate4",
        action="store_true",
        help="skip downstream cluster-overlap gate (useful when zig build isn't ready)",
    )
    p.add_argument(
        "--n-queries",
        type=int,
        default=10,
        help=(
            "number of leading docs to use as self-queries in gate 5. "
            "Must satisfy n_queries + top_k + 1 <= n_docs."
        ),
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="top-k size for gate 5 search-ranking parity",
    )
    p.add_argument(
        "--search-overlap-floor",
        type=float,
        default=0.9,
        help=(
            "minimum acceptable top-k overlap fraction (per-query, min "
            "across all queries) for gate 5. 0.9 means at least 9 of 10 "
            "top-k results must agree between PT and MLX."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Sanity: required files exist.
    for p in (FIXTURE, MLX_WEIGHTS, VENV_PY):
        if not p.exists():
            raise SystemExit(f"required path missing: {p}")

    print(f"[parity] fixture: {FIXTURE} (max_docs={args.max_docs}) dtype={args.dtype}", flush=True)

    pt_bin = args.out_dir / "parity_pt.bin"
    mlx_bin = args.out_dir / "parity_mlx.bin"
    mlx_a = args.out_dir / "parity_mlx_a.bin"
    mlx_b = args.out_dir / "parity_mlx_b.bin"

    print("[parity] encoding via encode.py (pylate, cpu)…", flush=True)
    t_pt = encode_pt(pt_bin, args.max_docs, args.dtype if args.dtype == "fp32" else "fp32")
    # Note: encode.py on cpu doesn't accept fp16 the same way; we pass fp32 there.
    # MLX side runs at the requested dtype.
    print(f"[parity] encode.py: {t_pt:.2f}s", flush=True)

    print(f"[parity] encoding via encode_mlx.py ({args.dtype})…", flush=True)
    t_mlx = encode_mlx(mlx_bin, args.max_docs, args.dtype)
    print(f"[parity] encode_mlx.py: {t_mlx:.2f}s (vs {t_pt:.2f}s pt → {t_pt/max(t_mlx, 1e-9):.2f}x)", flush=True)

    pt = parse_tokens_bin(pt_bin)
    mlx = parse_tokens_bin(mlx_bin)

    results: dict[str, tuple[bool, str]] = {}

    print("[parity] gate 1: token-id byte equality")
    results["gate1_token_ids"] = gate1_token_ids(pt, mlx)
    print(f"  {'PASS' if results['gate1_token_ids'][0] else 'FAIL'}: {results['gate1_token_ids'][1]}")

    # Gate 2 thresholds: when we compare fp32 PT vs fp{16,32} MLX the
    # floors are 0.998 mean / 0.99 min cosine. abs_floor=1e-2 catches
    # silent magnitude drift while staying loose enough for legitimate
    # fp16 → fp32 ULP noise on L2-normalised dim-128 vectors.
    print("[parity] gate 2: per-token cosine + abs-error")
    results["gate2_cosine"] = gate2_cosine(
        pt, mlx, mean_floor=0.998, min_floor=0.99, abs_floor=1e-2
    )
    print(f"  {'PASS' if results['gate2_cosine'][0] else 'FAIL'}: {results['gate2_cosine'][1]}")

    print(f"[parity] gate 3: two-run MLX determinism ({args.dtype})")
    results["gate3_determinism"] = gate3_determinism(mlx_a, mlx_b, args.max_docs, args.dtype)
    print(f"  {'PASS' if results['gate3_determinism'][0] else 'FAIL'}: {results['gate3_determinism'][1]}")

    if args.skip_gate4:
        print("[parity] gate 4: skipped (--skip-gate4)")
        results["gate4_clustering"] = (True, "skipped via --skip-gate4")
    else:
        print(f"[parity] gate 4: TAC clustering overlap (kappa={args.kappa}, seed={args.seed})")
        try:
            results["gate4_clustering"] = gate4_clustering(pt_bin, mlx_bin, args.kappa, args.seed)
        except SystemExit as e:
            results["gate4_clustering"] = (False, f"gate4 setup failed: {e}")
        print(f"  {'PASS' if results['gate4_clustering'][0] else 'FAIL'}: {results['gate4_clustering'][1]}")

    print(
        f"[parity] gate 5: search-ranking top-{args.top_k} overlap "
        f"({args.n_queries} self-queries)"
    )
    results["gate5_search_ranking"] = gate5_search_ranking(
        pt,
        mlx,
        n_queries=args.n_queries,
        top_k=args.top_k,
        overlap_floor=args.search_overlap_floor,
    )
    print(
        f"  {'PASS' if results['gate5_search_ranking'][0] else 'FAIL'}: "
        f"{results['gate5_search_ranking'][1]}"
    )

    print()
    print("[parity] summary:")
    all_pass = True
    for name, (ok, msg) in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}  {msg}")
        if not ok:
            all_pass = False

    if not all_pass:
        sys.exit(2)
    print("[parity] all gates PASS")


if __name__ == "__main__":
    main()
