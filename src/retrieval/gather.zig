//! src/retrieval/gather.zig — Gather phase (paper §5.1).
//!
//! Owner: retriever.
//! See plan 04-retrieval-and-eval.plan.md.
//!
//! For each query token q_i:
//!   - HNSW top-κ_c centroids with ef_s = 1.5·κ_c
//!   - walk inverted lists, accumulate s̃_i(d) = max_j ⟨q_i, c_j⟩
//! Aggregate S̃(q,d) = Σ_i s̃_i(d).

const std = @import("std");

test "placeholder" {
    try std.testing.expect(true);
}
