//! src/eval/metrics.zig — IR metrics (MRR@10, Success@k).
//!
//! Owner: retriever. Paper §7 (datasets and metrics): MS MARCO-v1 reports
//! MRR@10 and LoTTE-pooled reports Success@5. Both metrics are computed
//! per-query and averaged externally by the eval harness.
//!
//! `ranking` is an ordered list of doc IDs (most relevant first) returned by
//! the retrieval pipeline. `qrels` is the (small, unordered) set of doc IDs
//! considered relevant for the query — derived from the qrels file shipped
//! with each dataset. Both lists are pre-deduplicated by the caller.

const std = @import("std");

/// Mean Reciprocal Rank at cutoff k for a single query (paper §7).
///
/// Returns `1 / r` where `r` is the 1-based rank of the first relevant doc in
/// `ranking[0..min(k, ranking.len)]`, or 0 if no relevant doc is found in
/// that window. The arithmetic mean across queries is the dataset-level
/// MRR@k computed by the harness.
///
/// Boundary behaviour:
///   - `k == 0` → 0 (empty cutoff window).
///   - `qrels.len == 0` → 0 (no relevant docs to find).
///   - `k > ranking.len` → effectively scans the whole ranking.
pub fn mrrAt(ranking: []const u32, qrels: []const u32, k: u32) f32 {
    if (k == 0 or qrels.len == 0 or ranking.len == 0) return 0.0;
    const limit = @min(@as(usize, k), ranking.len);
    var i: usize = 0;
    while (i < limit) : (i += 1) {
        if (containsId(qrels, ranking[i])) {
            const rank: f32 = @floatFromInt(i + 1);
            return 1.0 / rank;
        }
    }
    return 0.0;
}

/// Success@k for a single query (paper §7): 1.0 if any qrel doc appears in
/// `ranking[0..min(k, ranking.len)]`, else 0.0. Dataset-level Success@k is
/// the mean across queries.
pub fn successAt(ranking: []const u32, qrels: []const u32, k: u32) f32 {
    if (k == 0 or qrels.len == 0 or ranking.len == 0) return 0.0;
    const limit = @min(@as(usize, k), ranking.len);
    var i: usize = 0;
    while (i < limit) : (i += 1) {
        if (containsId(qrels, ranking[i])) return 1.0;
    }
    return 0.0;
}

/// Linear scan — qrels sets are tiny (typically 1–10 entries on MS MARCO /
/// LoTTE), so a hash set would be slower than the direct compare.
fn containsId(qrels: []const u32, id: u32) bool {
    for (qrels) |q| {
        if (q == id) return true;
    }
    return false;
}

const testing = std.testing;

test "MRR@10: relevant doc at rank 2 → 0.5" {
    const ranking = [_]u32{ 5, 2, 9, 1, 4 };
    const qrels = [_]u32{2};
    try testing.expectApproxEqAbs(@as(f32, 0.5), mrrAt(&ranking, &qrels, 10), 1e-6);
}

test "MRR@10: relevant doc at rank 1 → 1.0" {
    const ranking = [_]u32{ 7, 2, 9 };
    const qrels = [_]u32{7};
    try testing.expectApproxEqAbs(@as(f32, 1.0), mrrAt(&ranking, &qrels, 10), 1e-6);
}

test "MRR@10: no relevant doc in ranking → 0" {
    const ranking = [_]u32{ 5, 2, 9, 1, 4 };
    const qrels = [_]u32{7};
    try testing.expectEqual(@as(f32, 0.0), mrrAt(&ranking, &qrels, 10));
}

test "MRR@10: relevant doc beyond cutoff → 0" {
    const ranking = [_]u32{ 5, 2, 9, 1, 4, 7 };
    const qrels = [_]u32{7};
    try testing.expectEqual(@as(f32, 0.0), mrrAt(&ranking, &qrels, 5));
}

test "MRR@10: only the *first* relevant counts (rank 3 not 5)" {
    const ranking = [_]u32{ 5, 2, 9, 1, 4 };
    const qrels = [_]u32{ 9, 4 };
    try testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), mrrAt(&ranking, &qrels, 10), 1e-6);
}

test "MRR@k: k > ranking.len scans the whole ranking" {
    const ranking = [_]u32{ 5, 2, 9 };
    const qrels = [_]u32{9};
    try testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), mrrAt(&ranking, &qrels, 100), 1e-6);
}

test "MRR@k: empty qrels → 0" {
    const ranking = [_]u32{ 5, 2, 9, 1, 4 };
    const qrels = [_]u32{};
    try testing.expectEqual(@as(f32, 0.0), mrrAt(&ranking, &qrels, 10));
}

test "MRR@k: k = 0 → 0" {
    const ranking = [_]u32{ 5, 2 };
    const qrels = [_]u32{2};
    try testing.expectEqual(@as(f32, 0.0), mrrAt(&ranking, &qrels, 0));
}

test "MRR@k: empty ranking → 0" {
    const ranking = [_]u32{};
    const qrels = [_]u32{2};
    try testing.expectEqual(@as(f32, 0.0), mrrAt(&ranking, &qrels, 10));
}

test "Success@5: relevant in top-5 → 1.0" {
    const ranking = [_]u32{ 5, 2, 9, 1, 4 };
    const qrels = [_]u32{1};
    try testing.expectEqual(@as(f32, 1.0), successAt(&ranking, &qrels, 5));
}

test "Success@5: no relevant in top-5 → 0.0" {
    const ranking = [_]u32{ 5, 2, 9, 1, 4 };
    const qrels = [_]u32{7};
    try testing.expectEqual(@as(f32, 0.0), successAt(&ranking, &qrels, 5));
}

test "Success@5: relevant beyond cutoff → 0.0" {
    const ranking = [_]u32{ 5, 2, 9, 1, 4, 7 };
    const qrels = [_]u32{7};
    try testing.expectEqual(@as(f32, 0.0), successAt(&ranking, &qrels, 5));
}

test "Success@k: empty qrels → 0" {
    const ranking = [_]u32{ 5, 2 };
    const qrels = [_]u32{};
    try testing.expectEqual(@as(f32, 0.0), successAt(&ranking, &qrels, 5));
}

test "Success@k: k > ranking.len handled" {
    const ranking = [_]u32{ 5, 2, 9 };
    const qrels = [_]u32{9};
    try testing.expectEqual(@as(f32, 1.0), successAt(&ranking, &qrels, 100));
}

test "Success@k: k = 0 → 0" {
    const ranking = [_]u32{ 5, 2 };
    const qrels = [_]u32{2};
    try testing.expectEqual(@as(f32, 0.0), successAt(&ranking, &qrels, 0));
}

test "Success@k: empty ranking → 0" {
    const ranking = [_]u32{};
    const qrels = [_]u32{2};
    try testing.expectEqual(@as(f32, 0.0), successAt(&ranking, &qrels, 5));
}

test "MRR@10 matches Success@10 on a match (sanity)" {
    const ranking = [_]u32{ 5, 2, 9 };
    const qrels = [_]u32{2};
    try testing.expect(successAt(&ranking, &qrels, 10) == 1.0);
    try testing.expectApproxEqAbs(@as(f32, 0.5), mrrAt(&ranking, &qrels, 10), 1e-6);
}
