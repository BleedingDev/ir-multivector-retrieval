//! src/retrieval/prune.zig — Candidate pruning (paper §5.2).
//!
//! Owner: retriever.
//! See plan 04-retrieval-and-eval.plan.md.
//!
//! Two stages:
//!   1. Top-κ_d truncation by S̃(q, d).
//!   2. Optional adaptive Candidates Pruning (CP) using α ∈ {0.35,0.4,0.45,0.5}.

const std = @import("std");
const gather = @import("gather.zig");

pub const PruneParams = struct {
    kappa_d: u32,
    /// `null` disables adaptive Candidates Pruning (top-κ_d only).
    alpha: ?f32 = null,
};

/// Descending-by-score comparator with ascending `doc_id` as a deterministic
/// secondary key. The secondary key matters because `std.sort.pdq` is not a
/// stable sort: equal-score candidates can otherwise reorder run-to-run
/// depending on input layout. Pinning ties to `doc_id` makes pruned output
/// bytewise reproducible across input shuffles, which is what every test
/// (and the paper §9 byte-equality story) relies on.
fn cmpDesc(_: void, a: gather.Candidate, b: gather.Candidate) bool {
    if (a.score != b.score) return a.score > b.score;
    return a.doc_id < b.doc_id;
}

/// Mutate `candidates` in-place: sort descending by score and truncate to
/// `≤ kappa_d` entries. If `alpha` is set, additionally drop the tail of
/// candidates whose score falls below `alpha * candidates[0].score`.
///
/// Returns the surviving prefix slice (no allocations). Caller still owns
/// `candidates`'s backing memory and should free the original allocation.
///
/// paper-gap §5.2: the paper says "using a fraction α of the running
/// maximum" but doesn't formalise the running-max update rule. We use the
/// simplest deterministic interpretation that matches the wording:
///
///   running_max := candidates[0].score   (the post-sort top)
///   threshold   := alpha * running_max
///   keep iff score >= threshold
///
/// Because Stage 1 leaves `candidates` sorted descending, Stage 2 collapses
/// to a single-pass cutoff: walk until the first score < threshold, drop
/// the rest. Deterministic at every (κ_d, α). Cross-check vs the reference
/// Rust impl tracked in plan 06.
///
/// CP guard: when `top <= 0` (e.g. every candidate has a non-positive score
/// for an out-of-domain query, or a synthetic test corner) the
/// `alpha * top` threshold inverts — for negative `top`, multiplying by
/// `α ∈ (0, 1)` gives a *larger* number, so `score < threshold` would be
/// true for every entry and CP would discard everything. That's never what
/// the user wants: the contract is "prune the tail", not "drop the head".
/// In that regime we skip CP entirely and fall back to top-κ_d, leaving at
/// least the best candidates available to refine. Equivalent to treating
/// `α` as undefined when the running max is non-positive.
pub fn prune(candidates: []gather.Candidate, params: PruneParams) []gather.Candidate {
    if (candidates.len == 0 or params.kappa_d == 0) return candidates[0..0];

    // Stage 1 — descending sort by score, ascending doc_id on ties. pdq is
    // O(n log n) with tiny constants and is deterministic across runs;
    // quickselect to κ_d would shave a log factor but the gather output is
    // bounded by candidate counts in the tens of thousands on MS MARCO @
    // κ_c=80, well within budget. Revisit if the latency harness shows
    // prune dominating.
    std.sort.pdq(gather.Candidate, candidates, {}, cmpDesc);

    var keep: usize = @min(candidates.len, @as(usize, params.kappa_d));

    // Stage 2 — adaptive Candidates Pruning.
    if (params.alpha) |a| {
        const top = candidates[0].score;
        // Guard: see doc-comment above. CP is only well-defined when the
        // running max is positive; otherwise skip it and keep top-κ_d.
        if (top > 0) {
            const threshold = a * top;
            var cut: usize = keep;
            var i: usize = 0;
            while (i < keep) : (i += 1) {
                if (candidates[i].score < threshold) {
                    cut = i;
                    break;
                }
            }
            keep = cut;
        }
    }

    return candidates[0..keep];
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

fn makeCands(buf: []gather.Candidate, scores: []const f32) []gather.Candidate {
    for (scores, 0..) |s, i| {
        buf[i] = .{ .doc_id = @intCast(i), .score = s };
    }
    return buf[0..scores.len];
}

test "PruneParams default disables CP" {
    const p = PruneParams{ .kappa_d = 1000 };
    try testing.expect(p.alpha == null);
    try testing.expectEqual(@as(u32, 1000), p.kappa_d);
}

test "PruneParams accepts paper alpha grid" {
    inline for (.{ 0.35, 0.4, 0.45, 0.5 }) |a| {
        const p = PruneParams{ .kappa_d = 500, .alpha = a };
        try testing.expect(p.alpha.? == a);
    }
}

test "prune: descending sort + top-κ_d truncation" {
    var buf: [8]gather.Candidate = undefined;
    const cands = makeCands(&buf, &.{ 1.0, 5.0, 3.0, 2.0, 4.0, 0.5, 6.0, 7.0 });
    const out = prune(cands, .{ .kappa_d = 4 });
    try testing.expectEqual(@as(usize, 4), out.len);
    try testing.expectEqual(@as(f32, 7.0), out[0].score);
    try testing.expectEqual(@as(f32, 6.0), out[1].score);
    try testing.expectEqual(@as(f32, 5.0), out[2].score);
    try testing.expectEqual(@as(f32, 4.0), out[3].score);
}

test "prune: kappa_d > n returns full sorted set" {
    var buf: [3]gather.Candidate = undefined;
    const cands = makeCands(&buf, &.{ 2.0, 1.0, 3.0 });
    const out = prune(cands, .{ .kappa_d = 100 });
    try testing.expectEqual(@as(usize, 3), out.len);
    try testing.expectEqual(@as(f32, 3.0), out[0].score);
    try testing.expectEqual(@as(f32, 2.0), out[1].score);
    try testing.expectEqual(@as(f32, 1.0), out[2].score);
}

test "prune: kappa_d == 0 returns empty slice" {
    var buf: [3]gather.Candidate = undefined;
    const cands = makeCands(&buf, &.{ 2.0, 1.0, 3.0 });
    const out = prune(cands, .{ .kappa_d = 0 });
    try testing.expectEqual(@as(usize, 0), out.len);
}

test "prune: empty input → empty output" {
    var buf: [0]gather.Candidate = undefined;
    const out = prune(buf[0..0], .{ .kappa_d = 10 });
    try testing.expectEqual(@as(usize, 0), out.len);
}

test "prune: adaptive CP α=0.5 cuts at half the running max" {
    var buf: [6]gather.Candidate = undefined;
    // After sort: 10, 8, 6, 4, 3, 1. running_max=10, threshold=5 → keep 10,8,6.
    const cands = makeCands(&buf, &.{ 1.0, 8.0, 4.0, 10.0, 3.0, 6.0 });
    const out = prune(cands, .{ .kappa_d = 100, .alpha = 0.5 });
    try testing.expectEqual(@as(usize, 3), out.len);
    try testing.expectEqual(@as(f32, 10.0), out[0].score);
    try testing.expectEqual(@as(f32, 8.0), out[1].score);
    try testing.expectEqual(@as(f32, 6.0), out[2].score);
}

test "prune: adaptive CP α=0.35 keeps more than α=0.5" {
    const scores = [_]f32{ 10.0, 8.0, 5.0, 4.0, 3.0, 1.0 };
    var buf_a: [6]gather.Candidate = undefined;
    var buf_b: [6]gather.Candidate = undefined;
    const a = makeCands(&buf_a, &scores);
    const b = makeCands(&buf_b, &scores);

    const out_strict = prune(a, .{ .kappa_d = 100, .alpha = 0.5 });
    const out_loose = prune(b, .{ .kappa_d = 100, .alpha = 0.35 });
    try testing.expect(out_loose.len >= out_strict.len);
    // strict α=0.5 → threshold=5.0, keep {10,8,5} (>= 5.0).
    try testing.expectEqual(@as(usize, 3), out_strict.len);
    // loose α=0.35 → threshold=3.5, keep {10,8,5,4} (>= 3.5).
    try testing.expectEqual(@as(usize, 4), out_loose.len);
}

test "prune: alpha=null is identical to top-κ_d" {
    const scores = [_]f32{ 1.0, 8.0, 4.0, 10.0, 3.0, 6.0 };
    var buf_a: [6]gather.Candidate = undefined;
    var buf_b: [6]gather.Candidate = undefined;
    const a = makeCands(&buf_a, &scores);
    const b = makeCands(&buf_b, &scores);
    const out_top = prune(a, .{ .kappa_d = 4 });
    const out_null_alpha = prune(b, .{ .kappa_d = 4, .alpha = null });
    try testing.expectEqual(out_top.len, out_null_alpha.len);
    for (out_top, out_null_alpha) |x, y| {
        try testing.expectEqual(x.doc_id, y.doc_id);
        try testing.expectEqual(x.score, y.score);
    }
}

test "prune: CP can drop below kappa_d (CP is more restrictive)" {
    var buf: [5]gather.Candidate = undefined;
    // Scores: 10, 1, 1, 1, 1 → α=0.5 threshold=5 → keep just 1.
    const cands = makeCands(&buf, &.{ 10.0, 1.0, 1.0, 1.0, 1.0 });
    const out = prune(cands, .{ .kappa_d = 100, .alpha = 0.5 });
    try testing.expectEqual(@as(usize, 1), out.len);
    try testing.expectEqual(@as(f32, 10.0), out[0].score);
}

test "prune: CP threshold of 0.0 keeps everything ≥ 0" {
    var buf: [3]gather.Candidate = undefined;
    const cands = makeCands(&buf, &.{ 0.0, 0.5, 1.0 });
    const out = prune(cands, .{ .kappa_d = 10, .alpha = 0.0 });
    try testing.expectEqual(@as(usize, 3), out.len);
}

test "prune: stable across input shuffles — same surviving doc IDs (in score order)" {
    var seed: u64 = 99;
    var rng = std.Random.DefaultPrng.init(seed);
    seed = 0; // unused after init

    var buf_orig: [20]gather.Candidate = undefined;
    var buf_shuf: [20]gather.Candidate = undefined;
    var i: u32 = 0;
    while (i < 20) : (i += 1) {
        const s: f32 = @as(f32, @floatFromInt(i));
        buf_orig[i] = .{ .doc_id = i, .score = s };
        buf_shuf[i] = .{ .doc_id = i, .score = s };
    }
    rng.random().shuffle(gather.Candidate, &buf_shuf);

    const a = prune(buf_orig[0..], .{ .kappa_d = 5, .alpha = 0.4 });
    const b = prune(buf_shuf[0..], .{ .kappa_d = 5, .alpha = 0.4 });
    try testing.expectEqual(a.len, b.len);
    for (a, b) |x, y| {
        try testing.expectEqual(x.doc_id, y.doc_id);
        try testing.expectEqual(x.score, y.score);
    }
}

test "prune: tied scores break by ascending doc_id deterministically" {
    // Five candidates with the same score; after sort the order must be by
    // ascending doc_id regardless of input order, because pdq is unstable.
    var buf: [5]gather.Candidate = .{
        .{ .doc_id = 42, .score = 1.0 },
        .{ .doc_id = 7, .score = 1.0 },
        .{ .doc_id = 99, .score = 1.0 },
        .{ .doc_id = 3, .score = 1.0 },
        .{ .doc_id = 17, .score = 1.0 },
    };
    const out = prune(buf[0..], .{ .kappa_d = 5 });
    try testing.expectEqual(@as(usize, 5), out.len);
    try testing.expectEqual(@as(u32, 3), out[0].doc_id);
    try testing.expectEqual(@as(u32, 7), out[1].doc_id);
    try testing.expectEqual(@as(u32, 17), out[2].doc_id);
    try testing.expectEqual(@as(u32, 42), out[3].doc_id);
    try testing.expectEqual(@as(u32, 99), out[4].doc_id);
}

test "prune: tie-break is deterministic across input shuffles when scores tie" {
    // 16 candidates with only 4 distinct scores → many ties. The kappa_d=8
    // cut must select the same doc_ids regardless of input layout.
    var buf_orig: [16]gather.Candidate = undefined;
    var buf_shuf: [16]gather.Candidate = undefined;
    var i: u32 = 0;
    while (i < 16) : (i += 1) {
        const s: f32 = @as(f32, @floatFromInt(i % 4)); // 0,1,2,3,0,1,2,3,...
        buf_orig[i] = .{ .doc_id = i, .score = s };
        buf_shuf[i] = .{ .doc_id = i, .score = s };
    }
    var rng = std.Random.DefaultPrng.init(0xdeadbeef);
    rng.random().shuffle(gather.Candidate, &buf_shuf);

    const a = prune(buf_orig[0..], .{ .kappa_d = 8 });
    const b = prune(buf_shuf[0..], .{ .kappa_d = 8 });
    try testing.expectEqual(a.len, b.len);
    for (a, b) |x, y| {
        try testing.expectEqual(x.doc_id, y.doc_id);
        try testing.expectEqual(x.score, y.score);
    }
}

test "prune: CP guard — all-negative scores keeps top-κ_d instead of dropping everything" {
    // Without the guard, top=-1.0 and α=0.5 give threshold=-0.5, which is
    // larger than every score, so CP would drop the entire candidate list.
    // With the guard, CP is skipped and top-κ_d returns intact.
    var buf: [5]gather.Candidate = undefined;
    const cands = makeCands(&buf, &.{ -3.0, -1.0, -2.0, -5.0, -4.0 });
    const out = prune(cands, .{ .kappa_d = 100, .alpha = 0.5 });
    try testing.expectEqual(@as(usize, 5), out.len);
    // Sort still applied: descending by score (least-negative first).
    try testing.expectEqual(@as(f32, -1.0), out[0].score);
    try testing.expectEqual(@as(f32, -2.0), out[1].score);
    try testing.expectEqual(@as(f32, -3.0), out[2].score);
    try testing.expectEqual(@as(f32, -4.0), out[3].score);
    try testing.expectEqual(@as(f32, -5.0), out[4].score);
}

test "prune: CP guard — top exactly 0 also skips CP (no divide-by-zero corner)" {
    // top=0 → threshold=0, score < 0 trivially drops every negative entry.
    // The guard's `top > 0` check means we skip CP and return top-κ_d.
    var buf: [4]gather.Candidate = undefined;
    const cands = makeCands(&buf, &.{ 0.0, -1.0, -2.0, -3.0 });
    const out = prune(cands, .{ .kappa_d = 100, .alpha = 0.5 });
    try testing.expectEqual(@as(usize, 4), out.len);
    try testing.expectEqual(@as(f32, 0.0), out[0].score);
}

test "prune: CP guard — mixed-sign scores with positive top still applies CP" {
    // Sanity: the guard only triggers when top <= 0. With a positive top,
    // CP runs normally even if the tail is negative.
    var buf: [5]gather.Candidate = undefined;
    const cands = makeCands(&buf, &.{ 10.0, 6.0, 1.0, -2.0, -5.0 });
    const out = prune(cands, .{ .kappa_d = 100, .alpha = 0.5 });
    // top=10, threshold=5, keep scores >= 5: {10, 6}.
    try testing.expectEqual(@as(usize, 2), out.len);
    try testing.expectEqual(@as(f32, 10.0), out[0].score);
    try testing.expectEqual(@as(f32, 6.0), out[1].score);
}
