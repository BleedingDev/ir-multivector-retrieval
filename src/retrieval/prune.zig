//! src/retrieval/prune.zig — Candidate pruning (paper §5.2).
//!
//! Owner: retriever.
//! See plan 04-retrieval-and-eval.plan.md.
//!
//! Two stages:
//!   1. Top-κ_d truncation by S̃(q, d).
//!   2. Optional adaptive Candidates Pruning (CP) using α ∈ {0.35,0.4,0.45,0.5}.
//!
//! Required surface (locked by task #18):
//!   pub const PruneParams = struct {
//!       kappa_d: u32,
//!       alpha: ?f32 = null,           // null → top-κ_d only, no CP
//!   };
//!   /// Truncates `candidates` in-place to ≤ kappa_d entries (descending by score),
//!   /// then applies adaptive CP if `alpha` is set. Returns the surviving sub-slice.
//!   pub fn prune(candidates: []gather.Candidate, params: PruneParams) []gather.Candidate;
//!
//! ---------------------------------------------------------------------------
//! DESIGN PSEUDOCODE — implemented under #18 (waits on #17 gather output).
//! ---------------------------------------------------------------------------
//!
//! Stage 1 — top-κ_d:
//!   std.sort.pdq descending by score. Quickselect to the κ_d-th element is
//!   asymptotically faster, but pdq is simple, deterministic, and the gather
//!   output is at most ~50K entries on MS MARCO @ κ_c=80 — pdq fits in budget.
//!   Revisit if latency harness shows prune > 5% of total query time.
//!
//! Stage 2 — adaptive Candidates Pruning (paper §5.2):
//!   The paper says "using a fraction α of the running maximum"; the exact
//!   update rule is not formalised in the extracted text.
//!
//!   // paper-gap §5.2: §5.2 doesn't pin the running-max update rule. We use
//!   // the simplest deterministic rule that respects "fraction of running
//!   // maximum":
//!   //   running_max := candidates[0].score    (largest after sort)
//!   //   threshold   := alpha * running_max
//!   //   keep candidate c iff c.score >= threshold
//!   //
//!   // Because candidates are sorted descending after Stage 1, this is a
//!   // single-pass cutoff: scan until the first c.score < threshold, drop
//!   // the tail. Deterministic at every (κ_d, α). Cross-check against the
//!   // reference Rust impl tracked in plan 06.
//!
//! prune(candidates, params):
//!   if candidates.len == 0: return candidates;
//!
//!   std.sort.pdq(gather.Candidate, candidates, {},
//!                fn(_, a, b) bool { return a.score > b.score; });
//!   keep = @min(candidates.len, params.kappa_d);
//!
//!   if (params.alpha) |a|:
//!     threshold = a * candidates[0].score;
//!     cut = keep;
//!     for i in 0..keep:
//!       if candidates[i].score < threshold: cut = i; break;
//!     keep = cut;
//!
//!   return candidates[0..keep];
//!
//! Tests planned for #18:
//!   - Top-κ_d truncation: 100 random scored candidates → exactly κ_d returned,
//!     all surviving ≥ all dropped, sorted descending.
//!   - α=0.5 with running_max=10 → drops everything < 5; hand-checked tiny set.
//!   - α=null (CP disabled) → identical to top-κ_d.
//!   - kappa_d > n → returns full sorted set; kappa_d == 0 → empty slice.
//!   - Stable across input shuffles: same surviving doc IDs.

const std = @import("std");
const gather = @import("gather.zig");

pub const PruneParams = struct {
    kappa_d: u32,
    /// `null` disables adaptive Candidates Pruning (top-κ_d only).
    alpha: ?f32 = null,
};

test "PruneParams default disables CP" {
    const p = PruneParams{ .kappa_d = 1000 };
    try std.testing.expect(p.alpha == null);
    try std.testing.expectEqual(@as(u32, 1000), p.kappa_d);
}

test "PruneParams accepts paper alpha grid" {
    inline for (.{ 0.35, 0.4, 0.45, 0.5 }) |a| {
        const p = PruneParams{ .kappa_d = 500, .alpha = a };
        try std.testing.expect(p.alpha.? == a);
    }
}

test "gather.Candidate visible from prune namespace" {
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(gather.Candidate));
}
