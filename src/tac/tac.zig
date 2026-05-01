//! src/tac/tac.zig — Token-Aware Clustering (paper §3) four-phase pipeline.
//!
//! Owner: clusterer.
//! See plan 02-tac-clustering.plan.md.
//!
//! TAC is the paper's headline contribution: instead of one global k-means
//! with κ centroids over all N≈600M token vectors, it runs N_T independent
//! per-token k-means with budgets κ_j allocated proportional to a damped
//! frequency-spread weight. The √n damping is what shifts allocation from
//! common low-discriminative tokens toward rare but semantically rich ones
//! (paper §2 motivation, §3.4 speedup analysis).
//!
//! ============================================================================
//! DESIGN — pseudocode-as-comments while primitives (#1, #4) are not yet ready.
//! ============================================================================
//!
//! Public surface:
//!
//!   pub const ClusteringParams = struct {
//!       kappa_total: u32,                       // global centroid budget κ
//!       mu: u32 = constants.TAC_MU,             // 128 — micro/small split
//!       tau: u32 = constants.TAC_TAU,           // 256 — small/active split
//!       epsilon: u32 = constants.TAC_EPSILON,   //   4 — κ_j floor for active
//!       theta: u32 = constants.TAC_THETA,       //  39 — n_j/κ_j ≥ θ cap
//!       max_iters: u32 = 25,                    // per-token Lloyd cap
//!       tol: f32 = 1e-4,                        // per-token Lloyd tolerance
//!       seed: u64,                              // base seed; per-token = seed XOR token_id
//!   };
//!
//!   pub const ClusteringResult = struct {
//!       centroids: []f32,                       // kappa_total * dim — caller-owned
//!       assignments: []u32,                     // n_tokens — global centroid id per vec
//!       kappa_per_token: []u32,                 // n_distinct_tokens — Σ == kappa_total
//!       wcss_total: f32,                        // sum of per-token WCSS (regression metric)
//!   };
//!
//!   pub const ClusteringError = error{
//!       BudgetTooSmall,        // kappa_total can't satisfy tail-handling minimums
//!       BudgetTooLarge,        // kappa_total > Σ_j (n_j / θ)  — paper bound impossible
//!       InvalidThresholds,     // mu >= tau, epsilon == 0, theta == 0
//!       EmptyTokenDump,
//!       OutOfMemory,
//!   } || kmeans.KMeansError;
//!
//!   pub fn cluster(
//!       td: io.token_dump.TokenDump,
//!       p: ClusteringParams,
//!       gpa: std.mem.Allocator,
//!   ) ClusteringError!ClusteringResult;
//!
//! ============================================================================
//! ALGORITHM — paper §3.3 four phases.
//!
//! Step 0: Group token vectors by token_id.
//!   The TokenDump stores vectors in document order with a parallel
//!   token_ids[] array. We need vectors_by_token[j] = list of indices into
//!   td.vectors corresponding to token j. Two-pass build (paper-gap; not
//!   specified, but this is the obvious linear-time approach):
//!     pass 1: count n_j[j]++ for each token_ids[i]
//!     pass 2: prefix-sum into offsets, scatter indices into a flat buffer
//!   Memory: O(n_tokens) for index buffer + O(N_T+1) for offsets. Avoids
//!   per-token slice-of-slices (alloc churn).
//!
//!   Distinct token count N_T = (max token_id seen) + 1. Tokens with n_j == 0
//!   are skipped — they get κ_j = 0 and contribute nothing.
//!
//! ----------------------------------------------------------------------------
//! Phase 1 — Tail handling (paper §3.3):
//!   - micro:  n_j < μ          → κ_j = 1
//!   - small:  μ ≤ n_j < τ      → κ_j = 2
//!   - active: n_j ≥ τ          → κ_j = TBD by phase 2
//!
//!   tail_used = Σ_{micro} 1 + Σ_{small} 2
//!   B = kappa_total - tail_used                  // active-token budget
//!
//!   Validation:
//!     - If tail_used > kappa_total                    → error.BudgetTooSmall.
//!     - If no active tokens:                            B == 0; phase 2 trivially
//!       allocates nothing. Caller still gets valid result if tail uses == κ.
//!     - If active token count > B (can't even floor each at 1):
//!       paper-gap: prioritise highest-w_j active tokens; remaining active
//!       tokens get κ_j = 1 (degraded mode). Document at the call site.
//!
//! ----------------------------------------------------------------------------
//! Phase 2 — Damped scoring (paper §3.2):
//!   For each active token j (with n_j ≥ τ):
//!     // paper §3.2 eq spread: s_j = (1/n_j) · Σ_i ‖t_{j,i} - t̄_j‖²
//!     1. Compute mean t̄_j of token j's vectors:
//!          t̄_j[d] = (1/n_j) · Σ_i t_{j,i}[d]
//!        Streaming sum — no need to materialise t̄_j into a separate buffer
//!        if we accumulate s_j in the same pass (Welford's algorithm):
//!          mean_old = mean
//!          mean = mean + (x - mean) / count
//!          M2 += (x - mean_old) · (x - mean)        // component-wise sums
//!          s_j = sum(M2) / n_j   // we want trace-of-covariance, not full cov
//!        For numerical stability, prefer Welford even though paper just says
//!        "(1/n_j) Σ ‖x − t̄‖²" — equivalent in exact arithmetic, more stable
//!        in f32 with large n_j (paper-gap; documented).
//!
//!     // paper §3.2 eq damped weight: w_j = √(n_j) · s_j
//!     2. w_j = @sqrt(@as(f32, @floatFromInt(n_j))) * s_j
//!
//!   Sum_w = Σ_{active j} w_j
//!
//!     // paper §3.2 eq allocation: κ_j = ⌊(w_j / Σ_i w_i) · B⌋
//!   3. For each active token: κ_j = @intFromFloat(@floor((w_j / Sum_w) * B))
//!      Save fractional remainder f_j = (w_j / Sum_w) * B - κ_j  for phase 4.
//!
//!   Edge: if Sum_w == 0 (e.g. all active tokens have zero spread —
//!   pathological synthetic data only): paper-gap. Distribute B equally
//!   among active tokens by floor(B / n_active), with leftover going to
//!   smallest token_id first (deterministic).
//!
//! ----------------------------------------------------------------------------
//! Phase 3 — Bounding (paper §3.3):
//!   For each active token j:
//!     // paper §3.3: floor κ_j ≥ ε=4
//!     if κ_j < ε:        κ_j = ε
//!     // paper §3.3: cap n_j/κ_j ≥ θ=39  →  κ_j ≤ ⌊n_j / θ⌋
//!     cap_j = @divFloor(n_j, theta)
//!     if κ_j > cap_j:    κ_j = cap_j
//!
//!   // paper-gap: floor and cap can conflict when n_j/θ < ε
//!   //            (i.e. n_j < ε·θ = 4·39 = 156). With τ=256 ≤ n_j for active
//!   //            tokens this never happens for paper defaults — but we
//!   //            assert it and return error.InvalidThresholds if a future
//!   //            caller sets τ < ε·θ. Document the assertion.
//!   assert(tau >= epsilon * theta);  // 256 ≥ 4·39 = 156: holds for paper defaults
//!
//! ----------------------------------------------------------------------------
//! Phase 4 — Budget reconciliation (paper §3.3, paper-gap on exact rule):
//!
//!   After phases 1+2+3, Σκ_j may differ from kappa_total because:
//!     - Phase 2's @floor() drops fractional mass.
//!     - Phase 3's floor (ε) can add centroids; cap (n_j/θ) can remove them.
//!
//!   Let delta = kappa_total - Σκ_j.
//!     delta > 0: need to add `delta` more centroids (surplus to distribute).
//!     delta < 0: need to remove `-delta` centroids (deficit to absorb).
//!     delta == 0: done.
//!
//!   // paper-gap: paper §3.3 says only "redistribute surplus/deficit". We pick
//!   //            a deterministic rule that respects bounds and converges in
//!   //            O(N_T·log N_T):
//!
//!   SURPLUS RULE (delta > 0):
//!     Among active tokens with κ_j < ⌊n_j/θ⌋ (i.e. cap-room available),
//!     repeatedly give one centroid to the token with the largest fractional
//!     remainder f_j (broken by smallest token_id). Decrement remainder
//!     buffer to avoid giving the same token two surplus centroids before
//!     others get one (round-robin within tied fractionals).
//!
//!     Implementation: sort active token indices by (-f_j, token_id), walk;
//!     when we exhaust the sorted list and still have surplus, do another
//!     pass (giving each token one more centroid as long as cap_j allows).
//!     Termination: each pass gives at least one centroid (else we'd be at
//!     cap on every token, which means kappa_total > Σ⌊n_j/θ⌋ which we
//!     reject up-front as error.BudgetTooLarge). Bound: O(B / n_active) passes.
//!
//!   DEFICIT RULE (delta < 0):
//!     Among active tokens with κ_j > ε (i.e. floor-room available),
//!     repeatedly take one centroid from the token with the smallest
//!     fractional remainder f_j (broken by smallest token_id, deterministic).
//!     Same multi-pass termination logic; bound: O(|delta| / n_active) passes.
//!
//!     Edge: if every active token is at ε floor, we can't reduce further
//!     without violating the floor. paper-gap: shrink one of the small-tier
//!     tokens (κ_j=2) down to 1 — degrade gracefully. If even that's not
//!     enough we degrade some micro tokens to κ_j=0 (paper-gap; never
//!     happens for sensible kappa_total ≥ N_T).
//!
//!   Why this rule:
//!     - Deterministic (no float ties broken by hash order).
//!     - Respects ε floor and n_j/θ cap as hard constraints.
//!     - Lands exactly on Σκ_j == kappa_total in finite time.
//!     - Matches Hamilton's apportionment method (largest-remainder),
//!       which is the canonical solution to integer-allocation rounding —
//!       so even if the paper's reference impl differs in tie-breaking,
//!       the global allocation will be near-identical on real data.
//!
//! ----------------------------------------------------------------------------
//! Step 5 — Per-token k-means (paper §3.3 phase 4 final):
//!   For each token j with κ_j ≥ 1:
//!     - Materialise token j's vectors into a contiguous f32 buffer of
//!       length n_j*dim by gathering from td.vectors using vectors_by_token
//!       offsets (cache-friendly for the inner Lloyd loop).
//!     - res = kmeans.fit(buf, dim, .{
//!           .k = κ_j,
//!           .max_iters = p.max_iters,
//!           .tol = p.tol,
//!           .seed = p.seed ^ @as(u64, j),     // per-token determinism
//!       }, gpa)
//!     - Copy res.centroids into the global centroids[ ] at offset
//!       global_centroid_offset[j] * dim. Free res.centroids.
//!     - Translate res.assignments local IDs (0..κ_j) → global IDs by adding
//!       global_centroid_offset[j]. Scatter into result.assignments using the
//!       original td.vectors index from vectors_by_token. Free res.assignments.
//!     - wcss_total += res.wcss
//!
//!   global_centroid_offset[j] = exclusive prefix sum of kappa_per_token.
//!   Tokens with n_j == 0 contribute nothing.
//!
//! ----------------------------------------------------------------------------
//! Step 6 — Validation (sanity asserts before return):
//!   - Σ kappa_per_token == kappa_total
//!   - all assignments[i] < kappa_total
//!   - all assignments populated (no leftover garbage from gather/scatter)
//!   - Optional in-debug: every centroid has at least θ vectors in its
//!     assignment (paper §3.3 cap holds).
//!
//! ============================================================================
//! TIME / SPACE COMPLEXITY
//!   Time:  O(n_tokens·dim)            for grouping + Welford
//!        + O(N_T·log N_T)             for phase-4 sort
//!        + Σ_j O(I_j · n_j · κ_j · d) for per-token Lloyd
//!   Space: O(kappa_total·dim) centroids + O(n_tokens) assignments
//!        + O(n_tokens) gather buffer (peak; reused across tokens)
//!        + O(N_T) per-token scalars
//!
//! ============================================================================
//! TESTS PLANNED (all hand-checkable):
//!   - test "phase 1 partitions tokens correctly with mu=3, tau=5"
//!       Synthetic: token freqs [1,2,3,4,5,6,7]
//!       → micro {0,1}, small {2,3}, active {4,5,6}; tail_used = 2+4 = 6;
//!         B = kappa_total - 6.
//!
//!   - test "phase 2 computes w_j matching the paper formula"
//!       Hand-build a vocab where two active tokens have:
//!         token A: n=8, vectors all zero except one component varies → s_A small.
//!         token B: n=8, vectors fully spread → s_B large.
//!       Expect w_B / w_A == s_B / s_A (since √n equal). Assert κ_B > κ_A
//!       under same Sum_w.
//!
//!   - test "phase 3 floor lifts κ_j up to epsilon"
//!       Active token with phase-2 κ_j = 1 → after floor κ_j = 4.
//!
//!   - test "phase 3 cap caps κ_j at n_j/theta"
//!       Active token n_j = 100, theta = 39 → cap_j = 2; phase-2 κ_j = 5
//!       → after cap κ_j = 2.
//!
//!   - test "phase 4 surplus reconciliation is deterministic"
//!       Construct vocab where Σκ_j = kappa_total - 3; assert the three
//!       extra centroids go to predicted (largest fractional remainder) tokens.
//!
//!   - test "phase 4 deficit reconciliation respects epsilon floor"
//!       Construct overshoot; assert reconciliation never drops κ_j below ε.
//!
//!   - test "cluster end-to-end on synthetic 16-token vocab" (integration)
//!       Use tests/fixtures/synthetic.zig (when #4 lands):
//!         50 docs, 16-token vocab, 8-dim, kappa_total=64.
//!       Assert: Σ kappa_per_token == 64, all assignments valid, wcss_total
//!       finite and > 0 (synthetic data is not collapsed to centroids).
//!
//!   - test "cluster is deterministic (byte-equal centroids under fixed seed)"
//!
//!   - test "kappa_total too small returns error.BudgetTooSmall"
//!       Set kappa_total = 1 with two micro tokens → tail_used = 2 > 1.
//!
//! Quality regression test (#10) lives in tests/quality_tac_vs_kmeans.zig
//! once the fixture infra is up.
//!
//! ============================================================================
//! DEPENDENCIES (blocked on):
//!   - util/vec.zig:        l2sq, dot, normalizeInPlace                  (#1)
//!   - util/rng.zig:        Rng + weightedSample (transitively via kmeans) (#2)
//!   - io/token_dump.zig:   TokenDump struct + open()                    (#4)
//!   - tests/fixtures/synthetic.zig: in-mem TokenDump builder            (#4)
//!   - kmeans.fit                                                         (#6)

const std = @import("std");
const constants = @import("../constants.zig");

// Implementation lands once primitives #1, #4 (and our own #6) are merged.
// See header for the full algorithm and test plan.
//
// Skeleton type declarations are intentionally NOT added yet — they would
// reference io.token_dump.TokenDump which is currently a stub. Adding them
// now would force fake fields and risk drift from primitives-engineer's
// final API. The header pseudocode is the authoritative design contract.

test "placeholder — implementation blocked on primitives #1, #4 and own #6" {
    _ = constants.TAC_MU;
    _ = constants.TAC_TAU;
    _ = constants.TAC_EPSILON;
    _ = constants.TAC_THETA;
    try std.testing.expect(true);
}
