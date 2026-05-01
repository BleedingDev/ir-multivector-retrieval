//! src/retrieval/refine.zig — Refine phase (paper §5.3).
//!
//! Owner: retriever.
//! See plan 04-retrieval-and-eval.plan.md.
//!
//! For each surviving candidate document d:
//!   Pass 1: stream centroid IDs → centroid contributions.
//!   Pass 2: stream PQ codes → residual contributions via cache-optimised
//!           distance table indexed [subspace M][PQ centroid 256][query token n_q].
//! Output: exact MaxSim score for the (q, d) pair.

const std = @import("std");

test "placeholder" {
    try std.testing.expect(true);
}
