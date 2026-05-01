---
name: tac eval + bench CLI subcommands
overview: src/eval/metrics.zig has tested MRR@10 / Success@k. src/retrieval/bench.zig has a working sweep harness. But neither is reachable from the `tac` binary today — the CLI stops at index/search. Wire `tac eval` and `tac bench` so the demo + Table 1 work is one-command-driven.
todos:
  - id: read-existing-surfaces
    content: "Re-read src/eval/metrics.zig (mrrAt, successAt) and src/retrieval/bench.zig (runCell, runSweep, formatCsvRow) to lock the public surface that main.zig will call into."
    status: pending
  - id: tac-eval-cmd
    content: "Add `cmdEval` to src/main.zig: `tac eval <index.tac> <queries.bin> <qrels.tsv> [--metric {mrr,success}] [--k N] [--kappa-c N] [--kappa-d N] [--alpha F]`. Reads qrels TSV (qid \\t doc_id format), runs gather→prune→refine per query, computes the metric, prints `<dataset> avg=<metric> n=<queries> elapsed=<ms>`."
    status: pending
  - id: tac-bench-cmd
    content: "Add `cmdBench` to src/main.zig: `tac bench <index.tac> <queries.bin> <qrels.tsv> [--out results.csv]`. Calls bench.runSweep over the paper grid (κ_c × κ_d × α) and writes a CSV row per cell. This is the Table 1 reproduction harness from the user's perspective."
    status: pending
  - id: qrels-loader
    content: "If benchmarks/common/qrels.zig already has a TSV reader, reuse it. If not, write a small one in src/eval/qrels.zig — line-delimited <qid>\\t<doc_id> with optional relevance integer column."
    status: pending
  - id: smoke-test
    content: "Encode 5 hand-built Czech queries against the existing data/jira/jira_full.tac, write a tiny qrels.tsv (3 lines: q1→ABC-1, q2→PT-9, q3→<random>), run `tac eval` and verify it prints a sensible MRR@10. Commit a fixture under tests/fixtures/jira_smoke_qrels.tsv if the qrels need persisting."
    status: pending
  - id: usage-help
    content: "Update `printUsage` in src/main.zig to list eval + bench. Confirm `tac --help` is readable."
    status: pending
  - id: commit
    content: "Commit + push to origin/main. Granular commits per subcommand OK."
    status: pending
isProject: false
---

# tac eval + bench CLI subcommands

## Execution Notes

Repo: ir-multivector-retrieval. Working directory: /Users/satan/side/experiments/ir-multivector-retrieval.

Current state:
- `src/main.zig` has `tac index` and `tac search` (lead-wired earlier). Stops there.
- `src/eval/metrics.zig` is fully implemented with hand-checked tests (MRR@10, Success@k).
- `src/retrieval/bench.zig` has `runCell`, `runSweep`, `formatCsvRow` — runs the paper §6 grid sweep.
- `benchmarks/common/queries.zig` + `benchmarks/common/runner.zig` exist for the per-dataset binaries (`bench_msmarco`, `bench_lotte`) — reuse what they offer.

Why this is the right lane now:
- The publication-track demo story currently requires writing a one-off Zig harness to run any eval — friction.
- All the underlying pieces are tested and shipped; only the CLI surface is missing.
- Lets the next person reproduce paper Table 1 with one command per dataset (after they have data).

## Constraints

- `src/main.zig` is lead-owned but adding two subcommands is a clean extension — no other lane touches it.
- Keep error messages as actionable as the existing `tac index/search` ones.
- Don't introduce new dependencies; the existing modules cover everything.

## Output

`src/main.zig` with `cmdEval` and `cmdBench`. Updated `--help`. Optional `src/eval/qrels.zig` if a qrels reader doesn't already exist. Smoke test against the existing 26k Jira index produces sensible numbers.
