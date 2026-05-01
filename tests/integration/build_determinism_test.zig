//! tests/integration/build_determinism_test.zig
//!
//! Owner: zig-determinism-doc-engineer (L8). Plan: audit-fixes-master-plan.md
//! finding zig-3.
//!
//! Locks down the *real* determinism contract of the full `.tac` build image,
//! not just the HNSW CSR. The README previously claimed full-build byte
//! equality across `n_threads ∈ {1, 4, 10}`. That claim was false: HNSW's
//! `n_threads=1` short-circuit takes the serial path and produces a *different*
//! deterministic graph from the chunked deferred-commit parallel path
//! (`n_threads >= 2`). storage.build threads `--threads` straight into HNSW
//! (src/index/storage.zig:399-402), so the full image cannot be byte-equal
//! between `1` and `{4, 10}`.
//!
//! What IS byte-equal across `n_threads ∈ {1, 4, 10}` (per the consult and
//! the codebase comments): the kmeans centroid output and the PQ codebooks.
//! The HNSW CSR — and therefore the full `.tac` image — is byte-equal only
//! across `n_threads ∈ {2, 4, 10}`.
//!
//! Coverage:
//!   1. Full image byte-equal across n_threads ∈ {2, 4, 10}.
//!   2. Full image byte-equal for two `n_threads=1` runs (serial is
//!      deterministically reproducible to itself).
//!   3. Full image at n_threads=1 differs from n_threads=4 (proves the README's
//!      old claim was actually false — and that the new one is necessary).
//!   4. Centroids and PQ codebooks are byte-equal across n_threads ∈ {1, 4, 10}.
//!      This is the slice of the contract that survives the README correction.
//!
//! Wired into `zig build test` via build.zig (test_step depends on this file).

const std = @import("std");
const tac_lib = @import("tac");
const testing = std.testing;

const synthetic_fixture = tac_lib.io.synthetic_fixture;
const token_dump = tac_lib.io.token_dump;
const storage = tac_lib.index.storage;

// Tiny-fixture TAC params — same shape as the round-trip test in
// src/index/storage.zig (μ=4/τ=8/ε=1/θ=1) so the synthetic distribution
// satisfies n_j/κ_j ≥ θ at every token. Paper-strict (μ=128, τ=256, ε=4,
// θ=39) is impossible at n_tokens ≈ 400.
const FixtureOpts = synthetic_fixture.Options{
    .seed = 20260501,
    .n_docs = 50,
    .dim = 32,
    .vocab_size = 16,
    .avg_doc_len = 8,
};

const KAPPA_TOTAL: u32 = 32;
const SEED: u64 = 7;

fn buildImage(a: std.mem.Allocator, n_threads: u32) !storage.BuiltImage {
    var fx = try synthetic_fixture.build(a, FixtureOpts);
    defer fx.deinit(a);

    const td = token_dump.TokenDump{
        .dim = fx.dim,
        .n_docs = fx.n_docs,
        .n_tokens = fx.n_tokens,
        .doc_offsets = fx.doc_offsets,
        .token_ids = fx.token_ids,
        .vectors = fx.vectors,
    };

    return try storage.build(&td, .{
        .kappa_total = KAPPA_TOTAL,
        .seed = SEED,
        .hnsw = .{ .ef_construction = 32, .m = 4, .n_threads = n_threads },
        .mu = 4,
        .tau = 8,
        .epsilon = 1,
        .theta = 1,
        .n_threads = n_threads,
    }, a);
}

test "full .tac image byte-equal across n_threads ∈ {2, 4, 10}" {
    const a = testing.allocator;

    var img2 = try buildImage(a, 2);
    defer img2.deinit(a);
    var img4 = try buildImage(a, 4);
    defer img4.deinit(a);
    var img10 = try buildImage(a, 10);
    defer img10.deinit(a);

    // Length first — a length divergence is a faster signal to debug than
    // a byte-by-byte mismatch deep in the image.
    try testing.expectEqual(img2.bytes.len, img4.bytes.len);
    try testing.expectEqual(img2.bytes.len, img10.bytes.len);

    try testing.expectEqualSlices(u8, img2.bytes, img4.bytes);
    try testing.expectEqualSlices(u8, img2.bytes, img10.bytes);
}

test "full .tac image byte-stable across two n_threads=1 runs (serial reproducibility)" {
    const a = testing.allocator;

    var img_a = try buildImage(a, 1);
    defer img_a.deinit(a);
    var img_b = try buildImage(a, 1);
    defer img_b.deinit(a);

    try testing.expectEqual(img_a.bytes.len, img_b.bytes.len);
    try testing.expectEqualSlices(u8, img_a.bytes, img_b.bytes);
}

test "full .tac image at n_threads=1 differs from n_threads=4 (HNSW serial vs parallel paths)" {
    // This test pins the *honest* claim: serial HNSW (n_threads=1) takes a
    // different code path from the chunked parallel build, so the resulting
    // graph CSR — and therefore the full image — is provably not byte-equal
    // to a {2,4,10} build.
    //
    // On this fixture (50 docs, dim=32, kappa=32) the divergence even shows
    // up as a different total image *length*, because HNSW's per-layer
    // neighbour CSR sizes depend on graph topology (storage.hnswSize). A
    // length difference is a strict superset of a byte difference, so we
    // cover both: assert lengths differ if they differ, fall through to
    // expectByteInequality if lengths happen to match for a different
    // fixture in the future.
    //
    // If a future change unifies the two paths into a serial-equivalent
    // parallel build, this test will start failing and the README claim
    // should be widened back to {1, 2, 4, 10}.
    const a = testing.allocator;

    var img1 = try buildImage(a, 1);
    defer img1.deinit(a);
    var img4 = try buildImage(a, 4);
    defer img4.deinit(a);

    if (img1.bytes.len == img4.bytes.len) {
        try testing.expect(!std.mem.eql(u8, img1.bytes, img4.bytes));
    } else {
        try testing.expect(img1.bytes.len != img4.bytes.len);
    }
}

test "centroids + PQ codebooks byte-equal across n_threads ∈ {1, 4, 10}" {
    // The slice of the determinism contract that DOES hold across all thread
    // counts including the serial path: kmeans output (centroids) and PQ
    // codebook training (work-stealing dispatch with serial-in-subspace
    // reduction). Only the HNSW graph diverges between serial and parallel.
    const a = testing.allocator;

    var img1 = try buildImage(a, 1);
    defer img1.deinit(a);
    var img4 = try buildImage(a, 4);
    defer img4.deinit(a);
    var img10 = try buildImage(a, 10);
    defer img10.deinit(a);

    var idx1 = try storage.parse(img1.bytes, a);
    defer idx1.deinit(a);
    var idx4 = try storage.parse(img4.bytes, a);
    defer idx4.deinit(a);
    var idx10 = try storage.parse(img10.bytes, a);
    defer idx10.deinit(a);

    // Centroids — produced by tac.clusterFlat → kmeans.fit. Reduction order
    // is serial-in-vector-order regardless of n_threads, so bytes match.
    try testing.expectEqualSlices(f32, idx1.centroids, idx4.centroids);
    try testing.expectEqualSlices(f32, idx1.centroids, idx10.centroids);

    // PQ codebooks — work-stealing outer dispatch (plan-10) plus
    // comptime-specialized kernels (plan-11) preserve byte-equality.
    try testing.expectEqualSlices(f32, idx1.pq.codebooks, idx4.pq.codebooks);
    try testing.expectEqualSlices(f32, idx1.pq.codebooks, idx10.pq.codebooks);

    // Residual norms — produced from (vectors, centroids, assignments) per
    // token, no cross-token reduction, so they're byte-equal too.
    try testing.expectEqualSlices(f32, idx1.residual_norms, idx4.residual_norms);
    try testing.expectEqualSlices(f32, idx1.residual_norms, idx10.residual_norms);
}
