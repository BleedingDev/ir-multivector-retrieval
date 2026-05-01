---
name: SIMD Specialization for Hot Distance Kernels
overview: src/util/vec.zig uses generic `@Vector(N, f32)` with N from `std.simd.suggestVectorLength(f32) orelse 4`. The hot kernels (`dot`, `l2sq`, `normalizeInPlace`) are called millions of times per build (kmeans assignment, PQ encode, gather phase). Add comptime-specialized paths for the dims that actually appear in the codebase (64 = jina-colbert-v2-64; 128 = ColBERTv2.0; 2/4 = PQ subspaces) and dispatch from the existing generic entry points so callers don't change.
todos:
  - id: profile-current-kernels
    content: "Run `zig build -Doptimize=ReleaseFast install` then time dot/l2sq on dim=64 and dim=128 in a microbenchmark. Establish baseline GFLOPs and verify the generic @Vector path is the actual bottleneck before specializing."
    status: pending
  - id: comptime-specialize
    content: "In src/util/vec.zig, add `pub fn dotComptime(comptime dim: u32, a, b) f32` and likewise for l2sq + normalizeInPlace. Specialize for {2, 4, 64, 128}. Use `@Vector(dim, f32)` directly when dim ≤ 64 (full SIMD register), or unrolled lanes for dim=128 (typically 2× M2 NEON 128-bit registers per op)."
    status: pending
  - id: dispatch-from-generic
    content: "The existing generic dot/l2sq stay as-is and become dispatchers: `if (dim == 128) return dotComptime(128, a, b); if (dim == 64) ...; else return generic_loop(...)`. Zero caller-side change."
    status: pending
  - id: numeric-parity-tests
    content: "For each specialized dim, add a parity test that computes the kernel via the specialized path AND a known-good scalar reference, asserts `expectApproxEqAbs(specialized, scalar, 1e-5)`. Also a permutation test: dot(a,b) == dot(b,a). Hand-checked input vectors (3 cases per dim) — small enough to verify by hand."
    status: pending
  - id: assignment-bench
    content: "Run a real kmeans.fit on 10k vectors × dim=128 × k=256 with the specialized vs generic paths (toggle via comptime const for the bench only — production uses dispatcher). Record speedup. Expect 1.3-2× on Apple Silicon if the generic path was sub-optimally lowered."
    status: pending
  - id: full-build-bench
    content: "Run `tac index data/jira/tokens_full.bin /tmp/jira_simd.tac --kappa 32768 --threads 10` and compare against the post-Wave-5 baseline (1:48). The contribution of vec.zig kernels to total build is bounded — gather/refine isn't measured here, only build phases. Honest reporting if speedup < 5%."
    status: pending
  - id: readme-perf
    content: "Update README perf table with new numbers. If speedup is below 5%, fold into a 'minor optimizations' note rather than the headline."
    status: pending
  - id: commit
    content: "Commit + push to origin/main. Single commit OK for the kernel + dispatch + tests; separate commit for README if it makes review easier."
    status: pending
isProject: false
---

# SIMD Specialization for Hot Distance Kernels

## Execution Notes

Repo: ir-multivector-retrieval. Working directory: /Users/satan/side/experiments/ir-multivector-retrieval.

Why this might or might not pay off:
- Apple M-series has 128-bit NEON SIMD (4×f32 per register) plus newer SVE/SME on M4. Zig's `std.simd.suggestVectorLength(f32)` typically returns 4 on M-series. Using `@Vector(4, f32)` is already nearly optimal for a single SIMD register.
- The generic loop in vec.zig:30-44 already uses tight `acc += va * vb` reductions — Zig's LLVM backend should auto-vectorize this well.
- The actual win comes from **comptime-known dim** letting the compiler unroll the loop and avoid the runtime tail-handling branch. For dim=128, that's 32 iterations of the lane loop; comptime unrolling drops the branch overhead and lets the hot path live in registers.

What we won't try (out of scope):
- Hand-rolled NEON intrinsics — Zig's @Vector lowering is good enough; portability would suffer.
- AVX2/AVX-512 paths — we're targeting Apple Silicon for the demo.
- FMA-specific tuning — `va * vb + acc` should fuse automatically; if it doesn't, bench reveals.

## Constraints

- All 189 existing tests stay green.
- Numeric parity within 1e-5 absolute on hand-checked inputs (FMA reordering can shift the LSB; 1e-5 is loose enough).
- Generic path must remain as fallback for any dim not in {2, 4, 64, 128} so the code keeps working on tiny test fixtures.
- Single-thread retrieval (paper §9) preserved — this is purely a per-call kernel optimization.

## Output

`src/util/vec.zig` with comptime-specialized kernels + dispatchers. Parity tests in the same file. README perf row updated honestly with measured (not projected) speedup.
