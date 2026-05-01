//! Strict-paper hyperparameter constants.
//!
//! These mirror the defaults from arxiv 2604.28142v1 exactly. They are LEAD-OWNED
//! and must not be redefined elsewhere — every module imports from here.
//!
//! Configurable params (e.g. retrieval-time κ_c, κ_d, α) live in the eval/CLI
//! layer; only the strict paper defaults live in this file.

// TAC — paper §3
pub const TAC_MU: u32 = 128; // micro-token frequency threshold
pub const TAC_TAU: u32 = 256; // small-token frequency threshold
pub const TAC_EPSILON: u32 = 4; // floor on κ_j for active tokens
pub const TAC_THETA: u32 = 39; // min vectors per centroid (cap on κ_j)

// Product Quantization — paper §4
pub const PQ_M: u32 = 32; // PQ subspaces
pub const PQ_BITS: u32 = 8; // bits per code
pub const PQ_CENTROIDS: u32 = 1 << PQ_BITS; // 256

// HNSW over centroids — paper §4
pub const HNSW_M: u32 = 32; // edges per node
pub const HNSW_EFC: u32 = 1500; // construction-time efSearch

// Compute runtime ef from κ_c (paper §5.1: ef_s = 1.5 · κ_c).
//
// paper-gap §5.1: the paper writes `ef_s = 1.5 · κ_c` without a rounding
// rule. We pick **floor** (integer `(κ_c * 3) / 2`) and apply it everywhere
// in the codebase. Rationale:
//   * conservative — never widens ef_s past the paper's nominal value, so
//     recall can only go down vs. the implicit rule, never spuriously up;
//   * the grid κ_c ∈ {15, 20, 40, 80, 100, 120} only triggers fractional
//     rounding at κ_c=15 (22.5 → 22 floor vs 23 ceil); for every other
//     paper grid point the two rules agree, so the choice only affects one
//     small-κ_c configuration;
//   * matches the existing `(κ_c * 3) / 2` integer expression — no extra
//     ceiling math at every retrieval call.
pub fn hnswEfSearch(kappa_c: u32) u32 {
    return (kappa_c * 3) / 2;
}

// Storage / file format
pub const TOKEN_DUMP_MAGIC: [8]u8 = "TAC_TKN1".*;
pub const INDEX_MAGIC: [8]u8 = "TAC_IDX1".*;
pub const TOKEN_DUMP_VERSION: u32 = 2; // v1 → v2: real BERT vocab IDs (paper §3 fidelity, task #21)
pub const INDEX_VERSION: u32 = 2; // v1 → v2: doc_token_offsets section for O(1) refine norm lookup

test "ef_s = floor(1.5 · κ_c) per paper §5.1 (floor rule documented above)" {
    const std = @import("std");
    // κ_c=15 is the only paper-grid point where floor(22.5)=22 differs from
    // ceil(22.5)=23 — locking it in here pins the rounding rule.
    try std.testing.expectEqual(@as(u32, 22), hnswEfSearch(15));
    try std.testing.expectEqual(@as(u32, 30), hnswEfSearch(20));
    try std.testing.expectEqual(@as(u32, 60), hnswEfSearch(40));
    try std.testing.expectEqual(@as(u32, 120), hnswEfSearch(80));
    try std.testing.expectEqual(@as(u32, 150), hnswEfSearch(100));
    try std.testing.expectEqual(@as(u32, 180), hnswEfSearch(120));
    // Edge cases: κ_c=0 → ef_s=0; κ_c=1 → floor(1.5)=1; κ_c=3 → floor(4.5)=4.
    try std.testing.expectEqual(@as(u32, 0), hnswEfSearch(0));
    try std.testing.expectEqual(@as(u32, 1), hnswEfSearch(1));
    try std.testing.expectEqual(@as(u32, 4), hnswEfSearch(3));
}
