# Agent Context — ir-multivector-retrieval

For Claude Code teammates and any other AI coding agents working in this repo.

## What this project is

An independent **Zig** reimplementation of **Tachiom** (`arxiv.org/html/2604.28142v1`) — token-aware clustering + hierarchical index for multivector (ColBERT-style) retrieval.

The authors' Rust implementation **is not yet published**. Work strictly from the paper text. `docs/paper.md` is the frozen reference extraction.

## Workflow rules

- **Strict-paper.** Every parameter the paper specifies (μ=128, τ=256, ε=4, θ=39, M=32, b=8, M_hnsw=32, efc=1500, ef_s=1.5·κ_c, the κ_c/κ_d/α grid) is used **exactly**. When the paper is silent, document the choice with a `// paper-gap:` comment and pick the simplest correct option.
- **Push to `main` directly.** Once `zig build test` is green, commit and push. No PR review.
- **Heavily guarded.** Every public function validates inputs at boundaries, uses `error{}` sets explicitly, and never silently truncates. Index files are versioned and magic-checked. No `unreachable` without a justification comment.
- **Heavily tested.** Each algorithmic component has unit tests on tiny synthetic data with hand-checkable expected values. Integration tests assemble the pipeline end-to-end on a 100-doc fixture. Latency tests run the gather/refine on a small index and assert order-of-magnitude expectations.
- **Cite the paper in code.** Every nontrivial function gets a `// paper §X.Y, eq: ...` comment so the math is traceable.

## Team structure

This work is split across a team of 4 teammates plus the lead. **File ownership is exclusive** to avoid conflicts — only the owning teammate writes to their directory.

| Teammate | Owns | Paper sections |
|---|---|---|
| `primitives-engineer` | `src/util/`, `src/io/`, `tools/encode.py` | foundation (no specific section) |
| `clusterer` | `src/tac/` | §3 — TAC four-phase pipeline |
| `indexer` | `src/index/` | §4 — HNSW + inverted lists + PQ |
| `retriever` | `src/retrieval/`, `src/eval/`, `benchmarks/` | §5 — gather/prune/refine, §7 evaluation |
| **lead** | `build.zig`, `build.zig.zon`, `src/main.zig`, `src/root.zig`, `README.md`, `AGENTS.md`, `docs/`, `.codex/plans/` | integration + cross-cutting concerns |

If you need a change in another teammate's directory, send them a message — do not edit it yourself.

## Coordination

- Shared task list is the source of truth. Claim tasks by ID, lowest first.
- After each task: run `zig build test`, commit with paper-citing message, push to `main`. Then check the task list for next work.
- If blocked, message the teammate whose work you depend on, then claim the next available task. Don't idle.
- Numerical correctness > speed. Make it work and tested first; optimize after we have measurements.

## Strict-paper hyperparameter table

```zig
// src/constants.zig (lead-owned, do not redefine)
pub const TAC_MU: u32 = 128;       // micro threshold
pub const TAC_TAU: u32 = 256;      // small threshold
pub const TAC_EPSILON: u32 = 4;    // floor on κ_j
pub const TAC_THETA: u32 = 39;     // min vectors per centroid
pub const PQ_M: u32 = 32;          // PQ subspaces
pub const PQ_BITS: u32 = 8;        // bits per code (256 entries)
pub const HNSW_M: u32 = 32;        // HNSW edges/node
pub const HNSW_EFC: u32 = 1500;    // HNSW construction efSearch
```

`ef_s` is computed at query time from `κ_c`. The retrieval grid lives in the eval harness.

## Conventions

- **Zig style:** `snake_case_files.zig`, `PascalCaseTypes`, `camelCase` functions, `SCREAMING_SNAKE_CASE` constants — match Zig std.
- **No global allocators.** Pass `std.mem.Allocator` explicitly. Long-lived data uses arena allocators owned by the caller.
- **No panics on user data.** Validate, return errors, fail closed.
- **Float type:** `f32` end to end (matches paper / ColBERT).
- **Determinism:** all randomized operations take a `seed: u64` parameter.

## Datasets

ColBERTv2 inference stays in Python (`tools/encode.py`, owned by `primitives-engineer`). It dumps a flat binary `tokens.bin` + sidecar metadata that Zig mmaps via `src/io/token_dump.zig`. Format spec lives in `docs/token-dump-format.md` (created by the primitives-engineer).

## When in doubt

1. Re-read `docs/paper.md` for the relevant section.
2. Add a `// paper-gap:` comment with your interpretation.
3. Message the lead.
