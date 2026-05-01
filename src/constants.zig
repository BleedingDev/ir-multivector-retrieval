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

// Compute runtime ef from κ_c (paper §5.1: ef_s = 1.5 · κ_c)
pub fn hnswEfSearch(kappa_c: u32) u32 {
    return (kappa_c * 3) / 2;
}

// Storage / file format
pub const TOKEN_DUMP_MAGIC: [8]u8 = "TAC_TKN1".*;
pub const INDEX_MAGIC: [8]u8 = "TAC_IDX1".*;
pub const TOKEN_DUMP_VERSION: u32 = 2; // v1 → v2: real BERT vocab IDs (paper §3 fidelity, task #21)
pub const INDEX_VERSION: u32 = 1;

test "ef_s = ceil(1.5 · κ_c) per paper §5.1" {
    const std = @import("std");
    try std.testing.expectEqual(@as(u32, 22), hnswEfSearch(15));
    try std.testing.expectEqual(@as(u32, 30), hnswEfSearch(20));
    try std.testing.expectEqual(@as(u32, 60), hnswEfSearch(40));
    try std.testing.expectEqual(@as(u32, 120), hnswEfSearch(80));
    try std.testing.expectEqual(@as(u32, 150), hnswEfSearch(100));
    try std.testing.expectEqual(@as(u32, 180), hnswEfSearch(120));
}
