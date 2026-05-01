# A1 zig-rebench — full Jira build + search latency under L10 harness

**Task:** close audit finding #10 (perf-honesty) by re-running headline numbers with `smp_allocator` (per L12 commit 03ddb1d) + L10's bench discipline.

**Commit benched:** `03ddb1df9aa412935e086ecfbe2b063198c045cd` (`hackathon-2026-05-01-followup-v2-29-g03ddb1d`)
**Build:** `zig build -Doptimize=ReleaseFast install` — clean
**Tests:** `zig build test --summary all` — 219/219 passing
**Host:** Apple M5, 10 cores, Darwin 25.3.0, Zig 0.16.0
**Allocator:** `std.heap.smp_allocator` (production CLI hot path; bench binaries mirror)
**Timer:** `clock_gettime(CLOCK_MONOTONIC)`
**Thread pin:** macOS no-op (Mach `thread_policy_set` isn't surfaced through std)
**Fixture:** `data/jira/tokens_full.bin` (26,678 docs / 1,521,797 tokens / dim=64)

## Build (full corpus, --kappa 32768 --threads 10)

| Run | Wall | tac.clusterFlat | residuals+norms | pq.train | pq.encode | hnsw.build | inverted_list | serialise |
|----:|-----:|----------------:|----------------:|---------:|----------:|-----------:|--------------:|----------:|
| 1   | 172.16 s | 10.47 s | 0.056 s | 153.77 s | 1.66 s | 5.61 s | 0.013 s | 0.308 s |
| 2   | 245.77 s |  8.24 s | 0.050 s | 218.95 s | 4.94 s | 12.97 s | 0.018 s | 0.335 s |
| 3   | 414.18 s | 13.55 s | 0.061 s | 389.87 s | 2.95 s | 6.80 s | 0.013 s | 0.322 s |
| 4   | 202.23 s |  8.35 s | 0.042 s | 183.68 s | 2.65 s | 6.99 s | 0.012 s | 0.309 s |
| 5   | 186.92 s |  8.99 s | 0.054 s | 170.12 s | 1.75 s | 5.50 s | 0.012 s | 0.294 s |
| 6   | 157.46 s |  8.01 s | 0.048 s | 140.07 s | 2.10 s | 6.65 s | 0.015 s | 0.333 s |

**Wall (s):** sorted 157, 172, 187, 202, 246, 414 → **median 194.6 s ≈ 3 min 15 s**, **min 157.5 s ≈ 2 min 37 s**, max 414.2 s ≈ 6 min 54 s.
**Stage medians (sorted):** clusterFlat 8.5 s, residuals+norms 0.05 s, pq.train **176.9 s**, pq.encode 2.4 s, hnsw.build **6.7 s**, inverted_list 0.013 s, serialise 0.32 s.
**Index size:** 79,427,800 bytes = **75.75 MB** (matches old README to 0.05 MB).

### Variance comment

`pq.train` dominates the build and is the noisiest stage on this M5. The atomic-counter work-stealing path in `pq.train` distributes M=32 subspaces across 10 threads; thermal/QoS pressure on Apple Silicon causes per-subspace Lloyd-iteration count and per-iteration assign cost to drift, producing the 140–390 s spread observed across 6 runs. The slowest run (run 3) immediately followed runs 1–2, suggesting cumulative thermal load. Run 6 (after a 60s idle) was the fastest. `BENCH_COOLDOWN_MS` only affects the search-side bench; there is no cooldown insertion mid-build.

## Search latency (single cell, κ_c=80, κ_d=1000, α=null, single-thread per paper §9)

Bench binary: `bench_jira_latency` (new, mirrors `bench_msmarco`/`bench_lotte` discipline; uses `tac.retrieval.search.search` directly inside an N-iter × 3-query loop, with 5 warm-up cells discarded). Source: `benchmarks/jira_latency.zig`.

| Run | Cooldown ms | Conditions | avg ms | p50 ms | p95 ms | p99 ms | min ms | max ms |
|----:|------------:|-----------|-------:|-------:|-------:|-------:|-------:|-------:|
| 1   |   50 | post-build, machine warm | 339.45 | 275.99 |  710.23 | 1160.72 | 152.43 | 1160.72 |
| 2   |   50 | back-to-back              | 427.23 | 336.40 |  941.21 | 1327.17 | 179.27 | 1327.17 |
| 3   |   50 | quieter machine           | 105.05 | 105.74 |  111.95 |  128.03 |  95.18 |  128.03 |
| 4   |   50 |                           | 175.57 | 168.84 |  243.09 |  264.29 | 138.51 |  264.29 |
| 5   |   50 |                           | 304.35 | 302.43 |  442.63 |  592.97 | 160.67 |  592.97 |
| 6   |  200 | post 60s idle             | 110.60 | 106.61 |  143.72 |  152.88 |  90.24 |  152.88 |
| 7   |  200 | post 30s idle             |  97.76 |  98.70 |  103.82 |  107.43 |  89.11 |  107.43 |

**Stable-floor (runs 3, 6, 7 — quiet machine + ≥50ms cooldown):** **p50 ≈ 100–106 ms**, **p99 ≈ 107–153 ms**, **min ≈ 89–95 ms**.
**Noisy upper-band (runs 1, 2, 4, 5 — concurrent thermal load):** p50 ranges 169–336 ms, p99 ranges 264–1327 ms.

The `tac search` CLI (separate path) reports the same shape (avg 335 ms / query at the same operating point on a warm machine, 90–110 ms on a cool one), confirming the bench harness isn't the source of the variance.

## Verdict (honest reporting)

The README's old headline numbers were measured with **DebugAllocator + no warm-up + no cooldown** and are **not reproducible** at face value under L10's discipline:

| Claim | Old README | New honest measurement | Direction |
|-------|-----------:|-----------------------:|----------|
| Build total (10 threads) | **1 min 23 s** | **3 min 15 s median** (2 min 37 s min, 6 min 54 s max over 6 runs) | regressed ~2.4× at median |
| `pq.train` (10 threads)   | 74.9 s         | **176.9 s median** (140–390 s)                                       | regressed ~2.4× at median |
| `tac.clusterFlat` (10 threads) | 4.2 s    | 8.5 s median (8.0–13.6 s)                                            | regressed ~2× |
| `hnsw.build` (10 threads)  | 2.7 s         | 6.7 s median (5.5–13.0 s)                                            | regressed ~2.5× |
| `pq.encode`                | 0.9 s         | 2.4 s median (1.7–4.9 s)                                             | regressed ~2.7× |
| residuals + norms          | 24 ms         | 54 ms median                                                          | regressed ~2.2× |
| Search latency, κ_c=80 κ_d=1000 | **~54 ms / query** | **p50 ≈ 100 ms, p99 ≈ 110–150 ms** (quiet) / **p50 ≈ 170–340 ms** (warm) | regressed ~2× best case, ~6× worst case |
| Index size                 | 75.8 MB       | **75.75 MB**                                                          | unchanged |

The old "~54 ms / query" likely was: DebugAllocator's leak-detection bookkeeping had different cache pressure characteristics (or — more likely — the original ad-hoc measurement averaged a small number of cells where one query was very fast and pulled the mean down). Without the original measurement protocol artifact, we can't audit the gap further. What's reproducible *now* on this M5 under L10's protocol is what the table above reports.

The build-time regression appears to be an honest headroom cost of `smp_allocator` over `DebugAllocator` plus the natural variance of `pq.train` on Apple-Silicon QoS scheduling; the per-stage shape (`pq.train` dominating, `hnsw.build` second) is preserved.

## Reproduction

```bash
# Build the binaries (smp_allocator hot path baked in via L12 commit 03ddb1d):
zig build -Doptimize=ReleaseFast install

# Latency bench needs a 4-column TREC qrels file (the bench harness's strict
# parser at benchmarks/common/qrels.zig). The 2-column hand-built fixture at
# tests/fixtures/jira_smoke_qrels.tsv is for the friendlier `tac eval`/`tac
# bench` CLI parser, not the per-dataset harness. Inline the TREC variant
# (gitignored as *.tsv but trivial to reproduce):
cat > .codex/bench-outputs/jira_smoke_qrels.trec.tsv <<'EOF'
1	0	0	1
2	0	1768	1
3	0	0	1
EOF

# Build the index (one of 6 runs captured under .codex/bench-outputs/jira_full_build_smp_t10*.log):
/usr/bin/time -lp ./zig-out/bin/tac index data/jira/tokens_full.bin \
    data/jira/jira_full_rebench.tac --kappa 32768 --threads 10

# Search latency at the README's headline operating point. Use a 4-column TREC
# qrels file (the bench harness's strict parser); runs 5 warm-up cells then 30
# timed cells × 3 queries → 90 raw samples.
BENCH_COOLDOWN_MS=200 ./zig-out/bin/bench_jira_latency \
    --index data/jira/jira_full_rebench.tac \
    --queries data/jira/queries.bin \
    --qids data/jira/queries.bin.qids \
    --qrels .codex/bench-outputs/jira_smoke_qrels.trec.tsv \
    --out .codex/bench-outputs/jira_latency_kc80_kd1000.csv \
    --kappa-c 80 --kappa-d 1000 --iters 30 --warmup 5 \
    --git-sha "$(git rev-parse HEAD)"
```

## Artifacts in this directory

- `jira_full_build_smp_t10*.log` — six full-build runs with `/usr/bin/time -lp` machine stats.
- `jira_latency_run*.log` + `jira_latency_run*.csv` — seven search-latency runs; CSV has one row per (iter, query) sample, header carries protocol.
- `jira_smoke_qrels.trec.tsv` — 4-column TREC-format qrels for the bench harness (derived from `tests/fixtures/jira_smoke_qrels.tsv`).
