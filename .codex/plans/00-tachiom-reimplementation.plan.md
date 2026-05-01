---
name: Tachiom Reimplementation in Zig
overview: Faithful reimplementation of arxiv 2604.28142v1 (Tachiom — token-aware clustering + hierarchical index for multivector retrieval) in Zig, targeting paper Table 1 numerics on MS MARCO-v1 and LoTTE-pooled.
todos:
  - id: bootstrap
    content: "Lead — repo scaffolding, build.zig, root.zig, main.zig CLI skeleton, docs, plan files."
    status: in_progress
  - id: primitives
    content: "Primitives engineer — src/util (vec/rng/alloc/SIMD) + src/io (binary token dump + mmap) + tools/encode.py (ColBERTv2 → tokens.bin)."
    status: pending
  - id: tac
    content: "Clusterer — src/tac (k-means++ + four-phase TAC pipeline), with paper-default μ/τ/ε/θ."
    status: pending
  - id: index
    content: "Indexer — src/index (PQ M=32 b=8, HNSW M=32 efc=1500, inverted lists, on-disk storage with magic + versioning)."
    status: pending
  - id: retrieval
    content: "Retriever — src/retrieval (gather, prune, refine with cache-optimised distance tables) + src/eval (MRR@10, Success@k, latency)."
    status: pending
  - id: integration
    content: "Lead — wire build, end-to-end CLI subcommands (index/search/eval/bench), README target table, push to main."
    status: pending
  - id: reproduce-table-1
    content: "Retriever — reproduce paper Table 1 operating points on MS MARCO-v1 (39.0/39.3) and LoTTE-pooled (67.5)."
    status: pending
isProject: true
---

# Tachiom Reimplementation in Zig

## Goal

Reproduce the paper's numerical claims (Table 1) under matching defaults and hardware-equivalent single-core retrieval. Latency parity vs the reference Rust+AVX2 impl is a stretch goal; **numerical fidelity** is required.

## Constraints

- Zig 0.16+, single binary, no external linker hacks.
- Strict paper defaults: `μ=128, τ=256, ε=4, θ=39, M=32, b=8, M_hnsw=32, efc=1500`.
- Push directly to `main` once `zig build test` passes.
- File ownership is exclusive per teammate (see AGENTS.md). No cross-directory edits.
- Numerical tests on synthetic data with hand-checkable values for every component.

## Output

A working `tac` binary with `index | search | eval | bench` subcommands, plus a benchmark report reproducing the operating points from Table 1.
