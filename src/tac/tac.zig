//! src/tac/tac.zig — Token-Aware Clustering (paper §3) four-phase pipeline.
//!
//! Owner: clusterer.
//! See plan 02-tac-clustering.plan.md.
//!
//! TAC is the paper's headline contribution: instead of one global k-means
//! with κ centroids over all N≈600M token vectors, it runs N_T independent
//! per-token k-means with budgets κ_j allocated by a damped weight
//! (paper §3.2 eq: w_j = √(n_j) · s_j). The √n damping shifts allocation from
//! common low-discriminative tokens toward rare but semantically rich ones
//! (paper §2 motivation, §3.4 speedup analysis).
//!
//! Four phases (paper §3.3):
//!   Phase 1: tail handling (micro/small/active split by frequency).
//!   Phase 2: damped scoring κ_j = ⌊(w_j / Σw) · B⌋ for active tokens.
//!   Phase 3: floor (κ_j ≥ ε) and cap (n_j/κ_j ≥ θ).
//!   Phase 4: budget reconciliation to land at Σκ_j = κ_total exactly.
//!
//! This file currently implements Phases 1–4 as pure functions operating on
//! pre-grouped per-token arrays (n_j, s_j). The end-to-end public driver
//! `cluster(td, ...)` lands once `io/token_dump.zig` exposes its grouping
//! API (task #4). All phase logic is unit-testable today.

const std = @import("std");
const constants = @import("../constants.zig");
const vec = @import("../util/vec.zig");
const kmeans = @import("kmeans.zig");

const Allocator = std.mem.Allocator;

// ---------------------------------------------------------------------------
// Public types — frozen by plan 02; consumed by indexer #15 and tests.
// ---------------------------------------------------------------------------

pub const ClusteringParams = struct {
    kappa_total: u32,
    mu: u32 = constants.TAC_MU,
    tau: u32 = constants.TAC_TAU,
    epsilon: u32 = constants.TAC_EPSILON,
    theta: u32 = constants.TAC_THETA,
    max_iters: u32 = 25,
    tol: f32 = 1e-4,
    seed: u64,
};

pub const ClusteringError = error{
    BudgetTooSmall,
    BudgetTooLarge,
    InvalidThresholds,
    EmptyTokenDump,
} || Allocator.Error || kmeans.KMeansError;

/// Token classification per paper §3.3 Phase 1.
pub const TokenTier = enum(u8) {
    empty, // n_j == 0 — token never appears in corpus, κ_j = 0
    micro, // n_j < μ      → κ_j = 1
    small, // μ ≤ n_j < τ  → κ_j = 2
    active, // n_j ≥ τ      → enter Phase 2
};

// ---------------------------------------------------------------------------
// Phase 1: tail handling (paper §3.3).
// ---------------------------------------------------------------------------

/// Classify a single token by its frequency.
/// paper §3.3: micro<μ, small∈[μ,τ), active≥τ.
pub fn classifyToken(n_j: u32, mu: u32, tau: u32) TokenTier {
    if (n_j == 0) return .empty;
    if (n_j < mu) return .micro;
    if (n_j < tau) return .small;
    return .active;
}

/// Phase 1 — partition tokens by frequency tier and assign tail κ_j.
///
/// Inputs: `freqs[j]` = n_j for each token j in 0..N_T.
/// Output: `kappa[j]` written for tail tokens (micro=1, small=2, empty=0).
///         Active tokens leave kappa[j]=0; phase 2 fills them in.
///
/// Returns the active-token budget B = kappa_total - Σ_tail κ_j, or
/// `error.BudgetTooSmall` if tail demand alone exceeds kappa_total.
///
/// paper §3.3 Phase 1: `// micro: n<μ → κ_j=1; small: μ≤n<τ → κ_j=2`
pub fn allocateTail(
    freqs: []const u32,
    kappa: []u32,
    mu: u32,
    tau: u32,
    kappa_total: u32,
) ClusteringError!u32 {
    if (mu == 0 or tau == 0 or mu >= tau) return error.InvalidThresholds;
    if (freqs.len != kappa.len) return error.InvalidThresholds;

    var tail_used: u32 = 0;
    for (freqs, 0..) |n_j, j| {
        switch (classifyToken(n_j, mu, tau)) {
            .empty => kappa[j] = 0,
            .micro => {
                kappa[j] = 1;
                tail_used += 1;
            },
            .small => {
                kappa[j] = 2;
                tail_used += 2;
            },
            .active => kappa[j] = 0, // filled by phase 2
        }
    }

    if (tail_used > kappa_total) return error.BudgetTooSmall;
    return kappa_total - tail_used;
}

// ---------------------------------------------------------------------------
// Phase 2: damped scoring (paper §3.2).
// ---------------------------------------------------------------------------

/// Compute the damped weight for one active token.
/// paper §3.2 eq damped weight: `w_j = √(n_j) · s_j`
pub fn dampedWeight(n_j: u32, s_j: f32) f32 {
    return @sqrt(@as(f32, @floatFromInt(n_j))) * s_j;
}

/// Phase 2 — proportional κ_j allocation for active tokens.
///
/// Inputs:
///   - freqs[j]:    n_j for each token (used to find active tokens)
///   - spreads[j]:  s_j for each active token; ignored for tail tokens
///                  (caller can leave 0 for empty/micro/small).
///   - kappa[j]:    Phase-1 output (tail κ_j filled, active = 0).
///   - B:           active-token budget from Phase 1.
///   - mu, tau:     thresholds, used to identify active tokens.
///
/// Output:
///   - kappa[j] for active tokens written as ⌊(w_j/Σw) · B⌋.
///   - frac[j]:    fractional remainder (w_j/Σw)·B − κ_j, used by Phase 4.
///                 Tail/empty tokens get frac[j] = 0.
///
/// paper §3.2 eq allocation: `κ_j = ⌊(w_j / Σ_i w_i) · B⌋`.
///
/// paper-gap: if every active token has s_j == 0 (Σw = 0), distribute B
/// equally across active tokens (token_id order, leftover to lowest IDs).
/// Pathological synthetic case only; documented for determinism.
pub fn dampedPhase2(
    freqs: []const u32,
    spreads: []const f32,
    kappa: []u32,
    frac: []f32,
    budget_active: u32,
    mu: u32,
    tau: u32,
) ClusteringError!void {
    if (freqs.len != spreads.len or freqs.len != kappa.len or freqs.len != frac.len) {
        return error.InvalidThresholds;
    }

    @memset(frac, 0.0);

    // First pass: compute Σw across active tokens.
    var sum_w: f64 = 0.0;
    var n_active: u32 = 0;
    for (freqs, 0..) |n_j, j| {
        if (classifyToken(n_j, mu, tau) != .active) continue;
        const w = dampedWeight(n_j, spreads[j]);
        if (!std.math.isFinite(w) or w < 0.0) return error.InvalidThresholds;
        sum_w += @floatCast(w);
        n_active += 1;
    }

    if (n_active == 0) return; // nothing to allocate; Phase 1 handled all tokens.

    if (sum_w <= 0.0) {
        // paper-gap: zero total spread — uniform fallback in token-id order.
        const base: u32 = if (n_active == 0) 0 else budget_active / n_active;
        var leftover: u32 = budget_active - base * n_active;
        for (freqs, 0..) |n_j, j| {
            if (classifyToken(n_j, mu, tau) != .active) continue;
            kappa[j] = base;
            if (leftover > 0) {
                kappa[j] += 1;
                leftover -= 1;
            }
        }
        return;
    }

    // Second pass: floor allocation; record fractional remainder.
    const B_f: f64 = @floatFromInt(budget_active);
    for (freqs, 0..) |n_j, j| {
        if (classifyToken(n_j, mu, tau) != .active) continue;
        const w: f64 = @floatCast(dampedWeight(n_j, spreads[j]));
        const target: f64 = (w / sum_w) * B_f;
        const floor_target: u32 = @intFromFloat(@floor(target));
        kappa[j] = floor_target;
        frac[j] = @floatCast(target - @as(f64, @floatFromInt(floor_target)));
    }
}

// ---------------------------------------------------------------------------
// Phase 3: bounding (paper §3.3).
// ---------------------------------------------------------------------------

/// Phase 3 — enforce κ_j floor (≥ε) and cap (n_j/κ_j ≥ θ ⟺ κ_j ≤ ⌊n_j/θ⌋).
///
/// Mutates `kappa[j]` for active tokens only. Tail/empty tokens are left as-is
/// (their κ_j was set by Phase 1 and is exempt from these bounds).
///
/// paper §3.3: `floor κ_j ≥ ε` and `cap n_j/κ_j ≥ θ`.
///
/// paper-gap: if cap < floor (i.e. n_j < ε·θ) for some active token, the two
/// bounds conflict. With paper defaults ε=4, θ=39, τ=256 this never happens
/// (ε·θ = 156 < τ = 256, so n_j ≥ 256 ⇒ cap = ⌊n_j/39⌋ ≥ 6 > 4 = floor).
/// We assert this precondition; an out-of-spec caller setting `tau < epsilon*theta`
/// gets `error.InvalidThresholds`.
pub fn boundPhase3(
    freqs: []const u32,
    kappa: []u32,
    mu: u32,
    tau: u32,
    epsilon: u32,
    theta: u32,
) ClusteringError!void {
    if (freqs.len != kappa.len) return error.InvalidThresholds;
    if (epsilon == 0 or theta == 0) return error.InvalidThresholds;
    // Reject thresholds that make floor and cap mutually unsatisfiable on
    // the smallest active token (n_j == tau).
    if (tau < epsilon * theta) return error.InvalidThresholds;

    for (freqs, 0..) |n_j, j| {
        if (classifyToken(n_j, mu, tau) != .active) continue;
        const cap: u32 = n_j / theta; // floor division — paper uses ⌊·⌋
        if (kappa[j] < epsilon) kappa[j] = epsilon;
        if (kappa[j] > cap) kappa[j] = cap;
    }
}

// ---------------------------------------------------------------------------
// Phase 4: budget reconciliation (paper §3.3, paper-gap on rule).
// ---------------------------------------------------------------------------

/// Phase 4 — drive Σκ_j to exactly `kappa_total` using bounded redistribution.
///
/// paper-gap: paper §3.3 says only "redistribute surplus/deficit". We implement
/// the canonical Hamilton largest-remainder rule, deterministically tie-broken
/// by token_id so two runs always land identically:
///
///   SURPLUS (delta > 0):
///     Among active tokens with κ_j < cap_j (room to grow), give one centroid
///     to whichever has the largest fractional remainder frac[j]. Ties broken
///     by smallest token_id. Repeat until either delta == 0 or every active
///     token is at its cap.
///
///   DEFICIT (delta < 0):
///     Among active tokens with κ_j > epsilon (room to shrink), take one
///     centroid from whichever has the smallest frac[j]. Ties broken by
///     smallest token_id. Repeat until either delta == 0 or every active
///     token is at its floor.
///
///     If we exhaust active room and still have deficit, we degrade tail tokens:
///     small (κ_j=2) → 1, then micro (κ_j=1) → 0. paper-gap; never happens
///     for paper-default parameters in production but kept for robustness.
///
/// Termination: each loop iteration changes exactly one κ_j by ±1 toward the
/// goal. The total work is bounded by |delta| ≤ kappa_total, so worst case
/// O(kappa_total · N_T) — fine since N_T is small (vocab ≤ 30k for ColBERT).
///
/// `error.BudgetTooLarge` if the total cap-room is insufficient for surplus.
pub fn reconcilePhase4(
    freqs: []const u32,
    kappa: []u32,
    frac: []const f32,
    kappa_total: u32,
    mu: u32,
    tau: u32,
    epsilon: u32,
    theta: u32,
) ClusteringError!void {
    if (freqs.len != kappa.len or freqs.len != frac.len) return error.InvalidThresholds;

    var current: u64 = 0;
    for (kappa) |k| current += k;

    if (current == kappa_total) return;

    if (current < kappa_total) {
        // SURPLUS — grow.
        var remaining: u32 = @intCast(kappa_total - current);
        while (remaining > 0) {
            // Find active token with largest frac[j] subject to κ_j < cap_j.
            var best: ?usize = null;
            var best_frac: f32 = -std.math.inf(f32);
            for (freqs, 0..) |n_j, j| {
                if (classifyToken(n_j, mu, tau) != .active) continue;
                const cap: u32 = n_j / theta;
                if (kappa[j] >= cap) continue;
                if (frac[j] > best_frac or (frac[j] == best_frac and best == null)) {
                    best = j;
                    best_frac = frac[j];
                }
            }
            if (best) |idx| {
                kappa[idx] += 1;
                remaining -= 1;
            } else {
                return error.BudgetTooLarge;
            }
        }
        return;
    }

    // DEFICIT — shrink.
    var excess: u32 = @intCast(current - kappa_total);
    while (excess > 0) {
        // Step 1: shrink active tokens (κ_j > epsilon) by smallest frac.
        var best: ?usize = null;
        var best_frac: f32 = std.math.inf(f32);
        for (freqs, 0..) |n_j, j| {
            if (classifyToken(n_j, mu, tau) != .active) continue;
            if (kappa[j] <= epsilon) continue;
            if (frac[j] < best_frac) {
                best = j;
                best_frac = frac[j];
            }
        }
        if (best) |idx| {
            kappa[idx] -= 1;
            excess -= 1;
            continue;
        }

        // Step 2: degrade small tokens (κ_j == 2) to κ_j == 1.
        // paper-gap: paper does not address; we degrade in token-id order.
        var found: ?usize = null;
        for (freqs, 0..) |n_j, j| {
            if (classifyToken(n_j, mu, tau) == .small and kappa[j] == 2) {
                found = j;
                break;
            }
        }
        if (found) |idx| {
            kappa[idx] = 1;
            excess -= 1;
            continue;
        }

        // Step 3: degrade micro tokens (κ_j == 1) to κ_j == 0.
        for (freqs, 0..) |n_j, j| {
            if (classifyToken(n_j, mu, tau) == .micro and kappa[j] == 1) {
                found = j;
                break;
            }
        }
        if (found) |idx| {
            kappa[idx] = 0;
            excess -= 1;
            continue;
        }

        // No room left anywhere — kappa_total cannot be expressed.
        // This is the symmetric BudgetTooSmall case discovered late.
        return error.BudgetTooSmall;
    }
}

// ---------------------------------------------------------------------------
// Phase combinator: run 1+2+3+4 on per-token arrays.
// ---------------------------------------------------------------------------

/// Run all four phases on pre-grouped per-token (freq, spread) arrays.
///
/// Output `kappa[j]` satisfies all paper §3.3 invariants:
///   - tier semantics (micro→1, small→2, active≥ε with cap n_j/θ)
///   - Σ kappa[j] == kappa_total (mod degraded edge cases — see Phase 4)
///
/// `frac` is scratch (len = freqs.len), used by Phase 4. Caller-allocated to
/// avoid hidden allocation in this function.
pub fn allocateBudgets(
    freqs: []const u32,
    spreads: []const f32,
    kappa: []u32,
    frac: []f32,
    kappa_total: u32,
    p: ClusteringParams,
) ClusteringError!void {
    const B = try allocateTail(freqs, kappa, p.mu, p.tau, kappa_total);
    try dampedPhase2(freqs, spreads, kappa, frac, B, p.mu, p.tau);
    try boundPhase3(freqs, kappa, p.mu, p.tau, p.epsilon, p.theta);
    try reconcilePhase4(freqs, kappa, frac, kappa_total, p.mu, p.tau, p.epsilon, p.theta);
}

// ---------------------------------------------------------------------------
// Public driver — TAC.cluster end-to-end.
// ---------------------------------------------------------------------------

const token_dump = @import("../io/token_dump.zig");

pub const ClusteringResult = struct {
    /// `kappa_total · dim`, concatenated in token-id order.
    centroids: []f32,
    /// Length `n_tokens` — global centroid id for input vector `i`, in `[0, kappa_total)`.
    assignments: []u32,
    /// Length `n_distinct_tokens` — Σ == kappa_total.
    kappa_per_token: []u32,
    /// Σ_j WCSS_j across all tokens (regression metric for #10).
    wcss_total: f32,

    pub fn deinit(self: *ClusteringResult, gpa: Allocator) void {
        gpa.free(self.centroids);
        gpa.free(self.assignments);
        gpa.free(self.kappa_per_token);
        self.* = undefined;
    }
};

/// Cluster a TokenDump corpus using TAC (paper §3 four-phase pipeline).
/// Thin wrapper over `clusterFlat`; the flat form is the testable contract.
pub fn cluster(
    td: token_dump.TokenDump,
    p: ClusteringParams,
    gpa: Allocator,
) ClusteringError!ClusteringResult {
    return clusterFlat(td.token_ids, td.vectors, td.dim, p, gpa);
}

/// Cluster a corpus of token vectors using TAC (paper §3 four-phase pipeline).
///
/// Inputs:
///   - `token_ids[i]`: vocabulary id of the i-th vector. Vocabulary size N_T
///     is inferred as max(token_ids)+1.
///   - `vectors[i*dim..(i+1)*dim]`: the i-th token vector.
///   - `p.kappa_total`: global centroid budget κ.
///
/// Output (caller-owned via `ClusteringResult.deinit`):
///   - `centroids`:        kappa_total · dim, concatenated in token-id order
///   - `assignments[i]`:   global centroid id for input vector i (in 0..κ)
///   - `kappa_per_token[j]`: κ_j; Σ == kappa_total
///   - `wcss_total`:       Σ_j WCSS_j over all tokens (regression metric)
///
/// paper §3.3 phase order: Phase 1+2+3+4 produce κ_j, then per-token Lloyd
/// produces centroids. Per-token determinism: seed = p.seed XOR token_id.
pub fn clusterFlat(
    token_ids: []const u32,
    vectors: []const f32,
    dim: u32,
    p: ClusteringParams,
    gpa: Allocator,
) ClusteringError!ClusteringResult {
    if (token_ids.len == 0) return error.EmptyTokenDump;
    if (dim == 0) return error.InvalidThresholds;
    if (vectors.len != token_ids.len * @as(usize, dim)) return error.InvalidThresholds;
    if (p.kappa_total == 0) return error.BudgetTooSmall;

    // ---- Step 0: vocabulary size + per-token frequencies. ----
    var max_id: u32 = 0;
    for (token_ids) |t| {
        if (t > max_id) max_id = t;
    }
    const n_distinct: usize = @as(usize, max_id) + 1;

    const freqs = try gpa.alloc(u32, n_distinct);
    defer gpa.free(freqs);
    @memset(freqs, 0);
    for (token_ids) |t| freqs[t] += 1;

    // ---- Step 1: CSR-grouping of input indices by token id. ----
    const group_offsets = try gpa.alloc(u32, n_distinct + 1);
    defer gpa.free(group_offsets);
    group_offsets[0] = 0;
    for (0..n_distinct) |j| group_offsets[j + 1] = group_offsets[j] + freqs[j];

    const group_indices = try gpa.alloc(u32, token_ids.len);
    defer gpa.free(group_indices);
    {
        const cursor = try gpa.alloc(u32, n_distinct);
        defer gpa.free(cursor);
        @memcpy(cursor, group_offsets[0..n_distinct]);
        for (token_ids, 0..) |t, i| {
            group_indices[cursor[t]] = @intCast(i);
            cursor[t] += 1;
        }
    }

    // ---- Step 2: per-token spread s_j (paper §3.2) via Welford's algorithm.
    // Welford gives a numerically stable streaming variance; mathematically
    // equivalent to the paper's `s_j = (1/n_j)·Σ‖t_{j,i} - t̄_j‖²`. We
    // accumulate component-wise M2 then sum to get the trace-of-covariance.
    const spreads = try gpa.alloc(f32, n_distinct);
    defer gpa.free(spreads);
    @memset(spreads, 0.0);

    {
        const mean_buf = try gpa.alloc(f32, dim);
        defer gpa.free(mean_buf);
        const m2_buf = try gpa.alloc(f32, dim);
        defer gpa.free(m2_buf);

        for (0..n_distinct) |j| {
            const start = group_offsets[j];
            const end = group_offsets[j + 1];
            const n_j = end - start;
            if (n_j == 0) continue;
            @memset(mean_buf, 0.0);
            @memset(m2_buf, 0.0);
            var count: f32 = 0.0;
            var idx_i: u32 = start;
            while (idx_i < end) : (idx_i += 1) {
                const v = vectors[group_indices[idx_i] * @as(usize, dim) ..][0..dim];
                count += 1.0;
                const inv_count: f32 = 1.0 / count;
                for (0..dim) |dd| {
                    const x = v[dd];
                    const old_mean = mean_buf[dd];
                    const delta = x - old_mean;
                    mean_buf[dd] = old_mean + delta * inv_count;
                    m2_buf[dd] += delta * (x - mean_buf[dd]);
                }
            }
            var trace: f32 = 0.0;
            for (m2_buf) |m| trace += m;
            spreads[j] = trace / @as(f32, @floatFromInt(n_j));
        }
    }

    // ---- Step 3: phases 1+2+3+4 produce κ_j per token. ----
    const kappa_per_token = try gpa.alloc(u32, n_distinct);
    errdefer gpa.free(kappa_per_token);
    const frac = try gpa.alloc(f32, n_distinct);
    defer gpa.free(frac);

    try allocateBudgets(freqs, spreads, kappa_per_token, frac, p.kappa_total, p);

    // Post-condition: any token with freq > 0 must have κ_j ≥ 1, otherwise
    // we'd be unable to assign its vectors to any centroid. Phase 4's deficit
    // fallback can degrade micro tokens to κ_j=0; surface that as
    // `error.BudgetTooSmall` (paper-gap) rather than ship corrupt assignments.
    for (freqs, kappa_per_token) |n_j, k_j| {
        if (n_j > 0 and k_j == 0) return error.BudgetTooSmall;
    }

    // ---- Step 4: per-token Lloyd's, gathering each token's vectors. ----
    const global_offsets = try gpa.alloc(u32, n_distinct + 1);
    defer gpa.free(global_offsets);
    global_offsets[0] = 0;
    for (0..n_distinct) |j| global_offsets[j + 1] = global_offsets[j] + kappa_per_token[j];

    const total_centroids = global_offsets[n_distinct];
    if (total_centroids != p.kappa_total) {
        return error.BudgetTooSmall;
    }

    const centroids = try gpa.alloc(f32, @as(usize, total_centroids) * dim);
    errdefer gpa.free(centroids);
    const assignments = try gpa.alloc(u32, token_ids.len);
    errdefer gpa.free(assignments);

    var max_n_j: u32 = 0;
    for (freqs) |n_j| {
        if (n_j > max_n_j) max_n_j = n_j;
    }
    const gather = try gpa.alloc(f32, @as(usize, max_n_j) * dim);
    defer gpa.free(gather);

    var wcss_total: f32 = 0.0;

    for (0..n_distinct) |j| {
        const k_j = kappa_per_token[j];
        if (k_j == 0) continue;
        const start = group_offsets[j];
        const end = group_offsets[j + 1];
        const n_j = end - start;
        if (n_j == 0) continue;

        // Materialise this token's vectors contiguously.
        var w: usize = 0;
        var idx_i: u32 = start;
        while (idx_i < end) : (idx_i += 1) {
            const src_idx: usize = group_indices[idx_i];
            @memcpy(
                gather[w * @as(usize, dim) ..][0..dim],
                vectors[src_idx * @as(usize, dim) ..][0..dim],
            );
            w += 1;
        }

        var res = try kmeans.fit(
            gather[0 .. @as(usize, n_j) * dim],
            dim,
            .{
                .k = k_j,
                .max_iters = p.max_iters,
                .tol = p.tol,
                .seed = p.seed ^ @as(u64, j),
            },
            gpa,
        );
        defer res.deinit(gpa);

        const global_off = global_offsets[j];
        @memcpy(
            centroids[@as(usize, global_off) * dim ..][0 .. @as(usize, k_j) * dim],
            res.centroids[0 .. @as(usize, k_j) * dim],
        );

        w = 0;
        idx_i = start;
        while (idx_i < end) : (idx_i += 1) {
            const src_idx: usize = group_indices[idx_i];
            assignments[src_idx] = global_off + res.assignments[w];
            w += 1;
        }

        wcss_total += res.wcss;
    }

    // Final invariant: every assignment is a valid centroid id.
    for (assignments) |a| {
        if (a >= total_centroids) return error.InvalidThresholds;
    }

    return .{
        .centroids = centroids,
        .assignments = assignments,
        .kappa_per_token = kappa_per_token,
        .wcss_total = wcss_total,
    };
}

// ---------------------------------------------------------------------------
// Tests — phases 1–4 with hand-checkable inputs.
// ---------------------------------------------------------------------------

const testing = std.testing;

test "classifyToken: empty/micro/small/active" {
    try testing.expectEqual(TokenTier.empty, classifyToken(0, 128, 256));
    try testing.expectEqual(TokenTier.micro, classifyToken(1, 128, 256));
    try testing.expectEqual(TokenTier.micro, classifyToken(127, 128, 256));
    try testing.expectEqual(TokenTier.small, classifyToken(128, 128, 256));
    try testing.expectEqual(TokenTier.small, classifyToken(255, 128, 256));
    try testing.expectEqual(TokenTier.active, classifyToken(256, 128, 256));
    try testing.expectEqual(TokenTier.active, classifyToken(10_000, 128, 256));
}

test "allocateTail: small thresholds (mu=3, tau=5)" {
    // freqs:        [0, 1, 2, 3, 4, 5, 6, 7]
    // tier:         [E, M, M, S, S, A, A, A]
    // tail kappa:   [0, 1, 1, 2, 2, 0, 0, 0]
    const freqs = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7 };
    var kappa = [_]u32{0} ** 8;
    const B = try allocateTail(&freqs, &kappa, 3, 5, 50);
    // tail_used = 1+1+2+2 = 6; B = 50 - 6 = 44.
    try testing.expectEqual(@as(u32, 44), B);
    try testing.expectEqualSlices(u32, &.{ 0, 1, 1, 2, 2, 0, 0, 0 }, &kappa);
}

test "allocateTail: kappa_total too small for tail" {
    const freqs = [_]u32{ 1, 1, 4 }; // tail_used = 1+1+2 = 4
    var kappa = [_]u32{0} ** 3;
    try testing.expectError(error.BudgetTooSmall, allocateTail(&freqs, &kappa, 3, 5, 3));
}

test "allocateTail: invalid thresholds" {
    const freqs = [_]u32{1};
    var kappa = [_]u32{0};
    try testing.expectError(error.InvalidThresholds, allocateTail(&freqs, &kappa, 5, 5, 10));
    try testing.expectError(error.InvalidThresholds, allocateTail(&freqs, &kappa, 6, 5, 10));
    try testing.expectError(error.InvalidThresholds, allocateTail(&freqs, &kappa, 0, 5, 10));
}

test "dampedWeight: paper formula w_j = sqrt(n_j) * s_j" {
    try testing.expectApproxEqAbs(@as(f32, 0.0), dampedWeight(0, 5.0), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 4.0), dampedWeight(4, 2.0), 1e-6); // sqrt(4)*2 = 4
    try testing.expectApproxEqAbs(@as(f32, 30.0), dampedWeight(100, 3.0), 1e-4); // 10*3
    try testing.expectApproxEqAbs(@as(f32, 0.0), dampedWeight(9, 0.0), 1e-6);
}

test "dampedPhase2: two equal-spread tokens split B in proportion to sqrt(n)" {
    // mu=3, tau=5: active threshold = 5.
    // Token A: n=9,  s=2   → w_A = 3*2 = 6
    // Token B: n=16, s=2   → w_B = 4*2 = 8
    // Σw = 14; B = 70 → κ_A = floor(6/14*70) = floor(30) = 30
    //                    κ_B = floor(8/14*70) = floor(40) = 40
    const freqs = [_]u32{ 9, 16 };
    const spreads = [_]f32{ 2.0, 2.0 };
    var kappa = [_]u32{ 0, 0 };
    var frac = [_]f32{ 0, 0 };
    try dampedPhase2(&freqs, &spreads, &kappa, &frac, 70, 3, 5);
    try testing.expectEqual(@as(u32, 30), kappa[0]);
    try testing.expectEqual(@as(u32, 40), kappa[1]);
    try testing.expectApproxEqAbs(@as(f32, 0.0), frac[0], 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 0.0), frac[1], 1e-3);
}

test "dampedPhase2: spread dominates over equal-frequency tokens" {
    // Two tokens with same n, different s — proportional to s.
    // Token A: n=25, s=1 → w=5
    // Token B: n=25, s=4 → w=20
    // Σw = 25; B = 100 → κ_A = 20, κ_B = 80.
    const freqs = [_]u32{ 25, 25 };
    const spreads = [_]f32{ 1.0, 4.0 };
    var kappa = [_]u32{ 0, 0 };
    var frac = [_]f32{ 0, 0 };
    try dampedPhase2(&freqs, &spreads, &kappa, &frac, 100, 3, 5);
    try testing.expectEqual(@as(u32, 20), kappa[0]);
    try testing.expectEqual(@as(u32, 80), kappa[1]);
}

test "dampedPhase2: zero-spread fallback splits uniformly" {
    const freqs = [_]u32{ 10, 10, 10 }; // all active under mu=3, tau=5
    const spreads = [_]f32{ 0.0, 0.0, 0.0 };
    var kappa = [_]u32{ 0, 0, 0 };
    var frac = [_]f32{ 0, 0, 0 };
    try dampedPhase2(&freqs, &spreads, &kappa, &frac, 11, 3, 5);
    // 11 / 3 = 3 base, leftover 2 → first two tokens get +1.
    try testing.expectEqualSlices(u32, &.{ 4, 4, 3 }, &kappa);
}

test "dampedPhase2: fractional remainder recorded" {
    // n=100, s=1 ⇒ w=10. Σw = 10. B=10 → κ = floor(10) = 10, frac = 0.
    // Make it not divide evenly: B=7 → target=7.0, κ=7, frac=0.
    // Better: use 2 tokens.
    const freqs = [_]u32{ 9, 16 };
    const spreads = [_]f32{ 1.0, 1.0 }; // w=3, 4. Σw=7.
    var kappa = [_]u32{ 0, 0 };
    var frac = [_]f32{ 0, 0 };
    try dampedPhase2(&freqs, &spreads, &kappa, &frac, 10, 3, 5);
    // target_A = 3/7*10 = 4.2857… → κ=4, frac≈0.2857
    // target_B = 4/7*10 = 5.7142… → κ=5, frac≈0.7143
    try testing.expectEqual(@as(u32, 4), kappa[0]);
    try testing.expectEqual(@as(u32, 5), kappa[1]);
    try testing.expectApproxEqAbs(@as(f32, 0.2857), frac[0], 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 0.7143), frac[1], 1e-3);
}

test "boundPhase3: floor lifts kappa up to epsilon" {
    // Active token with n=200, kappa=1 → after floor (eps=4) kappa=4.
    // Cap = 200/39 = 5; so 4 ≤ 5, accepted.
    const freqs = [_]u32{200};
    var kappa = [_]u32{1};
    try boundPhase3(&freqs, &kappa, 128, 156, 4, 39);
    try testing.expectEqual(@as(u32, 4), kappa[0]);
}

test "boundPhase3: cap caps kappa at floor(n_j/theta)" {
    // n=156, theta=39 → cap = 4. kappa=10 → after cap kappa=4.
    // (paper-default thresholds: tau=256, but we use tau=156 here so this token IS active.)
    const freqs = [_]u32{156};
    var kappa = [_]u32{10};
    try boundPhase3(&freqs, &kappa, 128, 156, 4, 39);
    try testing.expectEqual(@as(u32, 4), kappa[0]);
}

test "boundPhase3: ignores tail tokens" {
    // mu=128, tau=256. Tokens with n<128 (micro) and 128≤n<256 (small) are skipped.
    const freqs = [_]u32{ 50, 200, 500 }; // micro, small, active
    var kappa = [_]u32{ 1, 2, 1 };
    try boundPhase3(&freqs, &kappa, 128, 256, 4, 39);
    try testing.expectEqual(@as(u32, 1), kappa[0]); // micro untouched
    try testing.expectEqual(@as(u32, 2), kappa[1]); // small untouched
    try testing.expectEqual(@as(u32, 4), kappa[2]); // active floored to eps
}

test "boundPhase3: rejects threshold combo where cap < floor" {
    // tau=100 < epsilon*theta = 4*39 = 156 → conflict possible at smallest active.
    const freqs = [_]u32{100};
    var kappa = [_]u32{0};
    try testing.expectError(error.InvalidThresholds, boundPhase3(&freqs, &kappa, 50, 100, 4, 39));
}

test "reconcilePhase4: surplus distributed to largest fractional remainder" {
    // Two active tokens (mu=3, tau=5). Phase-2 leaves κ=4,5 (Σ=9), kappa_total=12 → surplus 3.
    // frac = [0.3, 0.8] — most surplus goes to j=1 first (largest frac).
    const freqs = [_]u32{ 10_000, 10_000 }; // big enough caps don't bind
    var kappa = [_]u32{ 4, 5 };
    const frac = [_]f32{ 0.3, 0.8 };
    try reconcilePhase4(&freqs, &kappa, &frac, 12, 3, 5, 4, 39);
    // Largest frac wins each step, but we re-evaluate every iter — j=1 keeps
    // winning because frac doesn't change. So j=1 gets all 3 surplus.
    try testing.expectEqual(@as(u32, 4), kappa[0]);
    try testing.expectEqual(@as(u32, 8), kappa[1]);
    try testing.expectEqual(@as(u32, 12), kappa[0] + kappa[1]);
}

test "reconcilePhase4: deficit shrinks smallest-frac token first, respects epsilon floor" {
    // Active tokens (mu=3, tau=5). Phase-3 left κ=[10, 8] Σ=18; kappa_total=14 → deficit 4.
    // frac = [0.1, 0.6] → smallest frac (j=0) loses centroids first.
    // Floor ε=4; j=0 can shrink from 10 down to 4 (room for 6 takedowns) — enough.
    const freqs = [_]u32{ 10_000, 10_000 };
    var kappa = [_]u32{ 10, 8 };
    const frac = [_]f32{ 0.1, 0.6 };
    try reconcilePhase4(&freqs, &kappa, &frac, 14, 3, 5, 4, 39);
    try testing.expectEqual(@as(u32, 6), kappa[0]); // 10 → 6
    try testing.expectEqual(@as(u32, 8), kappa[1]); // unchanged
    try testing.expectEqual(@as(u32, 14), kappa[0] + kappa[1]);
}

test "reconcilePhase4: deficit respects epsilon floor and walks to next token" {
    // Active tokens. κ=[10, 5] Σ=15, kappa_total=8 → deficit 7. eps=4.
    // j=0 can shrink 10→4 (6 takedowns). j=1 can shrink 5→4 (1 takedown).
    // frac=[0.0, 0.9] — j=0 wins repeatedly until κ[0]=4, then j=1 gives 1.
    const freqs = [_]u32{ 10_000, 10_000 };
    var kappa = [_]u32{ 10, 5 };
    const frac = [_]f32{ 0.0, 0.9 };
    try reconcilePhase4(&freqs, &kappa, &frac, 8, 3, 5, 4, 39);
    try testing.expectEqual(@as(u32, 4), kappa[0]);
    try testing.expectEqual(@as(u32, 4), kappa[1]);
}

test "reconcilePhase4: surplus exhausts cap room → BudgetTooLarge" {
    // Active n=156, theta=39 → cap=4. κ starts at 4 (already at cap). kappa_total=10 → surplus 6.
    // No room anywhere → error.BudgetTooLarge.
    const freqs = [_]u32{156};
    var kappa = [_]u32{4};
    const frac = [_]f32{0.5};
    try testing.expectError(error.BudgetTooLarge, reconcilePhase4(&freqs, &kappa, &frac, 10, 128, 156, 4, 39));
}

test "reconcilePhase4: deficit degrades small token if active room exhausted" {
    // Two tokens: small (n=200, kappa=2) + active (n=10_000, kappa=4 at floor).
    // Σ=6, kappa_total=5 → deficit 1. Active is at floor; degrade small from 2→1.
    const freqs = [_]u32{ 200, 10_000 };
    var kappa = [_]u32{ 2, 4 };
    const frac = [_]f32{ 0.0, 0.0 };
    try reconcilePhase4(&freqs, &kappa, &frac, 5, 128, 256, 4, 39);
    try testing.expectEqual(@as(u32, 1), kappa[0]); // small degraded
    try testing.expectEqual(@as(u32, 4), kappa[1]); // active untouched
}

test "allocateBudgets: end-to-end on small synthetic vocab" {
    // 6 tokens, paper-mini thresholds: mu=3, tau=5, eps=4, theta=1.
    // theta=1 keeps caps non-binding (cap = n_j) so we exercise the floor +
    // damped-allocation path without phase 3 cap dominating. The valid
    // constraint tau ≥ ε·θ holds: 5 ≥ 4·1.
    //
    // freqs:    [0,    2,    4,     20,    20,    20]
    // tier:     [E,    M,    S,     A,     A,     A]
    // spreads:  [-,    -,    -,     1.0,   2.0,   3.0]
    // tail κ:   [0,    1,    2,     0,     0,     0]    tail_used=3
    // B = kappa_total - 3 = 18 - 3 = 15.
    //
    // Active w_j = sqrt(20) * s_j ≈ 4.472*s.
    // w = [4.472, 8.944, 13.416], Σw ≈ 26.83.
    //   j=3: 4.472/26.83*15 ≈ 2.5  → κ=2, frac≈0.5
    //   j=4: 8.944/26.83*15 ≈ 5.0  → κ=5, frac≈0.0
    //   j=5: 13.416/26.83*15 ≈ 7.5 → κ=7, frac≈0.5
    // Phase 3: ε=4, θ=1. cap=n_j=20 (never binds).
    //   j=3: 2 < 4 → lifted to 4. j=4 ≥ 4 ok. j=5 ≥ 4 ok.
    // After phase 3: κ=[0,1,2,4,5,7] Σ=19; kappa_total=18 → deficit 1.
    // Phase 4 deficit: smallest frac among shrinkable actives (κ>ε).
    //   j=3 κ=4, at floor — skip.
    //   j=4 κ=5, frac=0.0 → candidate.
    //   j=5 κ=7, frac=0.5 → candidate.
    //   Smallest frac (0.0) wins → j=4 shrinks 5→4.
    // Final: [0,1,2,4,4,7] Σ=18.
    const freqs = [_]u32{ 0, 2, 4, 20, 20, 20 };
    const spreads = [_]f32{ 0.0, 0.0, 0.0, 1.0, 2.0, 3.0 };
    var kappa = [_]u32{0} ** 6;
    var frac = [_]f32{0} ** 6;

    const p = ClusteringParams{
        .kappa_total = 18,
        .mu = 3,
        .tau = 5,
        .epsilon = 4,
        .theta = 1,
        .seed = 0,
    };
    try allocateBudgets(&freqs, &spreads, &kappa, &frac, 18, p);

    try testing.expectEqualSlices(u32, &.{ 0, 1, 2, 4, 4, 7 }, &kappa);
    var sum: u32 = 0;
    for (kappa) |k| sum += k;
    try testing.expectEqual(@as(u32, 18), sum);
}

test "allocateBudgets: paper defaults sanity" {
    // One micro, one small, two active. kappa_total chosen so reconciliation
    // has light work. Tests that constants.TAC_* defaults compose cleanly.
    const freqs = [_]u32{ 50, 200, 5000, 5000 };
    const spreads = [_]f32{ 0.0, 0.0, 1.0, 2.0 };
    var kappa = [_]u32{0} ** 4;
    var frac = [_]f32{0} ** 4;

    const p = ClusteringParams{ .kappa_total = 100, .seed = 0 };
    try allocateBudgets(&freqs, &spreads, &kappa, &frac, 100, p);

    var sum: u32 = 0;
    for (kappa) |k| sum += k;
    try testing.expectEqual(@as(u32, 100), sum);
    // Tail tokens at their fixed allocations.
    try testing.expectEqual(@as(u32, 1), kappa[0]); // micro
    try testing.expectEqual(@as(u32, 2), kappa[1]); // small
    // Active tokens respect floor=4 and cap=⌊5000/39⌋=128.
    try testing.expect(kappa[2] >= 4 and kappa[2] <= 128);
    try testing.expect(kappa[3] >= 4 and kappa[3] <= 128);
    // Higher-spread token has at least as many centroids.
    try testing.expect(kappa[3] >= kappa[2]);
}

test "allocateBudgets: deterministic — same inputs land identical kappa" {
    const freqs = [_]u32{ 0, 100, 250, 1000, 5000 };
    const spreads = [_]f32{ 0.0, 0.0, 0.0, 0.7, 2.3 };
    var kappa1 = [_]u32{0} ** 5;
    var kappa2 = [_]u32{0} ** 5;
    var frac1 = [_]f32{0} ** 5;
    var frac2 = [_]f32{0} ** 5;

    const p = ClusteringParams{ .kappa_total = 50, .seed = 999 };
    try allocateBudgets(&freqs, &spreads, &kappa1, &frac1, 50, p);
    try allocateBudgets(&freqs, &spreads, &kappa2, &frac2, 50, p);

    try testing.expectEqualSlices(u32, &kappa1, &kappa2);
}

// ---------------------------------------------------------------------------
// clusterFlat — end-to-end driver tests.
// ---------------------------------------------------------------------------

test "clusterFlat: tiny corpus invariants (Σκ_j == κ, all assignments valid)" {
    // 4-token vocab, 2-dim, 30 vectors. mu=3, tau=5, eps=1, theta=1.
    // eps=1 keeps the floor non-binding so we exercise damped allocation
    // proper. theta=1 keeps the cap non-binding (cap = n_j).
    // Token 0: 10 vectors around (1, 0). active.
    // Token 1: 10 vectors around (0, 1). active.
    // Token 2: 6  vectors around (-1, 0). active.
    // Token 3: 4  vectors around (0, -1). small (3 ≤ 4 < 5).
    const allocator = testing.allocator;

    var token_ids: [30]u32 = undefined;
    var vectors: [60]f32 = undefined; // 30 × 2

    const counts = [_]u32{ 10, 10, 6, 4 };
    const themes = [_][2]f32{ .{ 1, 0 }, .{ 0, 1 }, .{ -1, 0 }, .{ 0, -1 } };
    var i: usize = 0;
    var prng = std.Random.DefaultPrng.init(7);
    for (counts, 0..) |c, tid| {
        var k: u32 = 0;
        while (k < c) : (k += 1) {
            token_ids[i] = @intCast(tid);
            vectors[i * 2 + 0] = themes[tid][0] + (prng.random().floatNorm(f32) * 0.05);
            vectors[i * 2 + 1] = themes[tid][1] + (prng.random().floatNorm(f32) * 0.05);
            i += 1;
        }
    }

    const p = ClusteringParams{
        .kappa_total = 14,
        .mu = 3,
        .tau = 5,
        .epsilon = 1,
        .theta = 1,
        .seed = 42,
    };
    var res = try clusterFlat(&token_ids, &vectors, 2, p, allocator);
    defer res.deinit(allocator);

    var sum: u32 = 0;
    for (res.kappa_per_token) |k| sum += k;
    try testing.expectEqual(@as(u32, 14), sum);

    // Token 3 is small: gets exactly κ=2.
    try testing.expectEqual(@as(u32, 2), res.kappa_per_token[3]);

    for (res.assignments) |a| try testing.expect(a < 14);
    try testing.expectEqual(@as(usize, 30), res.assignments.len);
    try testing.expectEqual(@as(usize, 14 * 2), res.centroids.len);

    try testing.expect(std.math.isFinite(res.wcss_total));
    try testing.expect(res.wcss_total >= 0.0);
}

test "clusterFlat: deterministic — same inputs produce identical centroids" {
    const allocator = testing.allocator;

    // 5 active tokens of n=8 each; epsilon=1 keeps the floor light so a
    // smallish kappa_total is still achievable.
    var token_ids: [40]u32 = undefined;
    var vectors: [120]f32 = undefined; // 40 × 3
    var prng = std.Random.DefaultPrng.init(1);
    for (0..40) |idx| {
        token_ids[idx] = @intCast(idx % 5);
        vectors[idx * 3 + 0] = prng.random().floatNorm(f32);
        vectors[idx * 3 + 1] = prng.random().floatNorm(f32);
        vectors[idx * 3 + 2] = prng.random().floatNorm(f32);
    }

    const p = ClusteringParams{
        .kappa_total = 18,
        .mu = 3,
        .tau = 5,
        .epsilon = 1,
        .theta = 1,
        .seed = 12345,
    };
    var r1 = try clusterFlat(&token_ids, &vectors, 3, p, allocator);
    defer r1.deinit(allocator);
    var r2 = try clusterFlat(&token_ids, &vectors, 3, p, allocator);
    defer r2.deinit(allocator);

    try testing.expectEqualSlices(f32, r1.centroids, r2.centroids);
    try testing.expectEqualSlices(u32, r1.assignments, r2.assignments);
    try testing.expectEqualSlices(u32, r1.kappa_per_token, r2.kappa_per_token);
    try testing.expectEqual(r1.wcss_total, r2.wcss_total);
}

test "clusterFlat: token 0 absent → kappa_per_token[0] == 0" {
    const allocator = testing.allocator;

    var token_ids = [_]u32{ 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3 };
    var vectors = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2 };

    const p = ClusteringParams{
        .kappa_total = 6,
        .mu = 3,
        .tau = 5,
        .epsilon = 1,
        .theta = 1,
        .seed = 0,
    };
    var res = try clusterFlat(&token_ids, &vectors, 1, p, allocator);
    defer res.deinit(allocator);

    // Each present token (n=4) is small (3 ≤ 4 < 5), each gets κ=2.
    try testing.expectEqualSlices(u32, &.{ 0, 2, 2, 2 }, res.kappa_per_token);
}

test "clusterFlat: empty input rejected" {
    const allocator = testing.allocator;
    const ids = [_]u32{};
    const v = [_]f32{};
    const p = ClusteringParams{ .kappa_total = 1, .seed = 0 };
    try testing.expectError(error.EmptyTokenDump, clusterFlat(&ids, &v, 2, p, allocator));
}

test "clusterFlat: kappa_total == 0 rejected" {
    const allocator = testing.allocator;
    const ids = [_]u32{ 0, 0 };
    const v = [_]f32{ 1, 0, 0, 1 };
    const p = ClusteringParams{ .kappa_total = 0, .seed = 0 };
    try testing.expectError(error.BudgetTooSmall, clusterFlat(&ids, &v, 2, p, allocator));
}

test "clusterFlat: vector/token-id length mismatch rejected" {
    const allocator = testing.allocator;
    const ids = [_]u32{ 0, 0, 0 };
    const v = [_]f32{ 1, 0, 0, 1 };
    const p = ClusteringParams{ .kappa_total = 4, .seed = 0 };
    try testing.expectError(error.InvalidThresholds, clusterFlat(&ids, &v, 2, p, allocator));
}

test "clusterFlat: centroids land near each token's theme (κ_j == 1 each)" {
    // 3 tokens, each gets exactly 1 centroid, each clusters tightly around
    // a far-apart theme. Recovered centroid must land near theme.
    const allocator = testing.allocator;

    var token_ids: [18]u32 = undefined;
    var vectors: [36]f32 = undefined;
    const themes = [_][2]f32{ .{ 10, 0 }, .{ 0, 10 }, .{ -10, 0 } };
    var prng = std.Random.DefaultPrng.init(99);
    var i: usize = 0;
    for (themes, 0..) |t, tid| {
        var k: u32 = 0;
        while (k < 6) : (k += 1) {
            token_ids[i] = @intCast(tid);
            vectors[i * 2 + 0] = t[0] + (prng.random().floatNorm(f32) * 0.05);
            vectors[i * 2 + 1] = t[1] + (prng.random().floatNorm(f32) * 0.05);
            i += 1;
        }
    }

    const p = ClusteringParams{
        .kappa_total = 3,
        .mu = 2,
        .tau = 5,
        .epsilon = 1,
        .theta = 1,
        .seed = 0,
    };
    var res = try clusterFlat(&token_ids, &vectors, 2, p, allocator);
    defer res.deinit(allocator);

    try testing.expectEqualSlices(u32, &.{ 1, 1, 1 }, res.kappa_per_token);

    for (themes, 0..) |t, j| {
        const cx = res.centroids[j * 2 + 0];
        const cy = res.centroids[j * 2 + 1];
        try testing.expectApproxEqAbs(t[0], cx, 0.1);
        try testing.expectApproxEqAbs(t[1], cy, 0.1);
    }

    for (token_ids, res.assignments) |tid, a| {
        try testing.expectEqual(@as(u32, tid), a);
    }
}

test "cluster: TokenDump wrapper produces same result as clusterFlat" {
    // Build a TokenDump in-memory and confirm cluster(td, ...) matches
    // clusterFlat(td.token_ids, td.vectors, ...).
    const allocator = testing.allocator;

    var token_ids = [_]u32{ 0, 1, 0, 1, 2, 2, 2, 2 };
    var vectors = [_]f32{
        1.0, 0.0,
        0.0, 1.0,
        0.9, 0.1,
        0.1, 0.9,
        -1.0, 0.0,
        -0.9, 0.1,
        -1.0, -0.05,
        -0.95, 0.0,
    };
    const offsets = [_]u64{ 0, 4, 8 };
    const td = token_dump.TokenDump{
        .dim = 2,
        .n_docs = 2,
        .n_tokens = 8,
        .doc_offsets = &offsets,
        .token_ids = &token_ids,
        .vectors = &vectors,
    };

    const p = ClusteringParams{
        .kappa_total = 4,
        .mu = 2,
        .tau = 4,
        .epsilon = 1,
        .theta = 1,
        .seed = 7,
    };
    var via_td = try cluster(td, p, allocator);
    defer via_td.deinit(allocator);
    var via_flat = try clusterFlat(&token_ids, &vectors, 2, p, allocator);
    defer via_flat.deinit(allocator);

    try testing.expectEqualSlices(f32, via_flat.centroids, via_td.centroids);
    try testing.expectEqualSlices(u32, via_flat.assignments, via_td.assignments);
    try testing.expectEqualSlices(u32, via_flat.kappa_per_token, via_td.kappa_per_token);
}

test "cluster: integration with synthetic_fixture (paper-style Zipf vocab)" {
    // Use the primitives-engineer fixture: deterministic Zipfian token-freq
    // distribution with per-token themes. Verify TAC ships valid
    // ClusteringResult on a paper-scale-mini input.
    const allocator = testing.allocator;
    const synthetic = @import("../io/synthetic_fixture.zig");

    var fx = try synthetic.build(allocator, .{
        .seed = 2026,
        .n_docs = 50,
        .dim = 8,
        .vocab_size = 16,
        .avg_doc_len = 8,
    });
    defer fx.deinit(allocator);

    // Paper-scale-mini thresholds: most tokens will be small/active given
    // n_docs=50 × avg_len=8 = ~400 tokens spread over 16 vocab ids ≈ 25/token.
    const p = ClusteringParams{
        .kappa_total = 64,
        .mu = 5,
        .tau = 10,
        .epsilon = 1,
        .theta = 1,
        .seed = 0,
    };
    var res = try clusterFlat(fx.token_ids, fx.vectors, fx.dim, p, allocator);
    defer res.deinit(allocator);

    var sum: u32 = 0;
    for (res.kappa_per_token) |k| sum += k;
    try testing.expectEqual(@as(u32, 64), sum);
    for (res.assignments) |a| try testing.expect(a < 64);
    try testing.expect(std.math.isFinite(res.wcss_total));
}

// ---------------------------------------------------------------------------
// Quality regression: TAC vs vanilla k-means at fixed κ (paper §3.4).
// ---------------------------------------------------------------------------
//
// The paper's headline claim (§3.4): at the same total centroid budget κ,
// TAC matches or beats vanilla k-means on clustering quality, because the
// damped allocation directs centroids to high-spread (semantically rich)
// tokens rather than flooding common tokens.
//
// Test approach: build a Zipfian-frequency synthetic corpus with per-token
// themes — common tokens cluster tightly around a single direction, rare
// tokens have higher intra-token spread (multiple sub-themes). Compare:
//   (a) TAC: Σ_j WCSS_j  — sum of per-token Lloyd's residuals.
//   (b) Vanilla: WCSS of one global Lloyd's run with k = kappa_total over
//       all vectors, no token awareness.
// Both metrics measure squared distance to assigned centroid summed over
// all input vectors; lower is better.
//
// On a token-skewed corpus, TAC should be at most a small constant worse
// than vanilla in absolute WCSS (and often better) because it allocates
// centroids where they're most useful per the damped weight. This test
// asserts TAC ≤ 1.25·Vanilla; we don't pin a tighter bound because the
// margin depends on the noise scale and the centroid-per-token budget.

fn computeGlobalWcss(
    vectors: []const f32,
    dim: u32,
    centroids: []const f32,
    k: u32,
    gpa: Allocator,
) !f32 {
    const n: usize = vectors.len / @as(usize, dim);
    const dist_buf = try gpa.alloc(f32, k);
    defer gpa.free(dist_buf);

    var wcss: f32 = 0.0;
    for (0..n) |i| {
        for (0..k) |c| {
            dist_buf[c] = try vec.l2sq(
                vectors[i * @as(usize, dim) ..][0..dim],
                centroids[c * @as(usize, dim) ..][0..dim],
            );
        }
        const a = try vec.argmin(dist_buf);
        wcss += dist_buf[a];
    }
    return wcss;
}

test "quality regression: TAC matches or beats vanilla k-means at fixed κ" {
    const allocator = testing.allocator;
    const synthetic = @import("../io/synthetic_fixture.zig");

    // Paper-scale-mini Zipf corpus: token ids drawn from 1/(rank+1) so a
    // few tokens dominate frequency — exactly the regime TAC was designed
    // for (paper §2 motivation).
    var fx = try synthetic.build(allocator, .{
        .seed = 2026,
        .n_docs = 200,
        .dim = 16,
        .vocab_size = 32,
        .avg_doc_len = 12,
    });
    defer fx.deinit(allocator);

    const kappa_total: u32 = 96;

    // (a) TAC.
    const tac_params = ClusteringParams{
        .kappa_total = kappa_total,
        .mu = 5,
        .tau = 15,
        .epsilon = 1,
        .theta = 1,
        .seed = 7,
    };
    var tac_res = try clusterFlat(
        fx.token_ids,
        fx.vectors,
        fx.dim,
        tac_params,
        allocator,
    );
    defer tac_res.deinit(allocator);

    // (b) Vanilla k-means with same kappa_total over all vectors.
    var vanilla_res = try kmeans.fit(
        fx.vectors,
        fx.dim,
        .{
            .k = kappa_total,
            .max_iters = 25,
            .tol = 1e-4,
            .seed = 7,
        },
        allocator,
    );
    defer vanilla_res.deinit(allocator);

    // For TAC, wcss_total is already the sum of per-token Lloyd's WCSS,
    // which is the same as Σ_i ‖v_i - c_{a_i}‖² across all input vectors
    // since each vector contributes exactly once.
    const tac_wcss = tac_res.wcss_total;
    const vanilla_wcss = vanilla_res.wcss;

    // Log the comparison so a CI failure is debuggable.
    std.debug.print(
        "\n[quality §3.4] κ={}, n_vectors={}: TAC WCSS={d:.3}, Vanilla WCSS={d:.3}, ratio={d:.3}x\n",
        .{ kappa_total, fx.n_tokens, tac_wcss, vanilla_wcss, tac_wcss / vanilla_wcss },
    );

    // Paper §3.4 claim: TAC matches or beats vanilla. We allow a modest 25%
    // tolerance because the synthetic corpus and per-token theme noise make
    // exact ties unrealistic and we don't tune κ_c/κ_d here.
    try testing.expect(tac_wcss <= vanilla_wcss * 1.25);
}
