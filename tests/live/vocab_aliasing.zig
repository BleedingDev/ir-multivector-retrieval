//! tests/live/vocab_aliasing.zig — paper §3 fidelity check using a real
//! ColBERTv2 encode of a 100-doc fixture.
//!
//! Owner: primitives-engineer (task #24).
//! Gated behind `-Dlive=true` in build.zig so default `zig build test`
//! stays Python-free.
//!
//! What this asserts:
//!   1. The fixture parses cleanly through the v2 token-dump format.
//!   2. n_j ≥ 2 for ≥50% of distinct token_ids (paper §3 cross-doc
//!      vocabulary aliasing — the whole point of the v2 vocab-id fix
//!      from task #21).
//!   3. tac.cluster runs to completion on the real corpus and returns
//!      Σ kappa_per_token == kappa_total.
//!
//! Re-run the upstream encode by:
//!   python tests/fixtures/live/build_docs.py
//!   python tools/encode.py --docs tests/fixtures/live/docs.jsonl \
//!       --out tests/fixtures/live/tokens.bin --device cpu --batch 16
//!
//! See tools/LIVE_ENCODER_NOTES.md for the actual encoder dim, vocab-id
//! distribution, and run cost — that doc is the contract retriever's #20
//! benchmark harness consumes.

const std = @import("std");
const tac = @import("tac");

const ParseBytes = tac.io.token_dump.parseBytes;

/// Path is resolved relative to the project root (where `zig build test`
/// runs). @embedFile would fail because Zig 0.16 forbids embedding files
/// outside the module's package path; runtime read is the simpler escape
/// hatch. The cost (~820KB read once) is irrelevant under `-Dlive=true`.
const FIXTURE_PATH = "tests/fixtures/live/tokens.bin";

fn loadFixture(allocator: std.mem.Allocator) ![]align(8) u8 {
    const io = std.testing.io;
    var f = try std.Io.Dir.cwd().openFile(io, FIXTURE_PATH, .{});
    defer f.close(io);
    const stat = try f.stat(io);
    const size: usize = @intCast(stat.size);
    const buf = try allocator.alignedAlloc(u8, .@"8", size);
    errdefer allocator.free(buf);
    var read_total: usize = 0;
    while (read_total < size) {
        const n = try f.readPositional(io, &.{buf[read_total..]}, read_total);
        if (n == 0) return error.ShortRead;
        read_total += n;
    }
    return buf;
}

test "live: token dump parses with v2 magic + version" {
    const a = std.testing.allocator;
    const buf = try loadFixture(a);
    defer a.free(buf);

    const td = try ParseBytes(buf);
    // Sanity: 100-doc fixture, ColBERTv2 dim 128.
    try std.testing.expectEqual(@as(u64, 100), td.n_docs);
    try std.testing.expectEqual(@as(u32, 128), td.dim);
    try std.testing.expect(td.n_tokens > 0);
    try std.testing.expect(td.n_tokens == td.token_ids.len);
    try std.testing.expect(td.vectors.len == td.n_tokens * td.dim);
}

test "live: ≥50% of distinct token_ids show n_j ≥ 2 (paper §3 vocab aliasing)" {
    const a = std.testing.allocator;
    const buf = try loadFixture(a);
    defer a.free(buf);

    const td = try ParseBytes(buf);

    // Count occurrences per vocab id with a hashmap; we don't know the
    // upper bound on BERT id values cheaply, so a hashmap is simpler than
    // a max-id scan + dense histogram, and the cost is irrelevant for a
    // 1.5K-token corpus.
    var freq = std.AutoHashMap(u32, u32).init(a);
    defer freq.deinit();
    for (td.token_ids) |id| {
        const gop = try freq.getOrPut(id);
        if (gop.found_existing) gop.value_ptr.* += 1 else gop.value_ptr.* = 1;
    }

    var distinct: u32 = 0;
    var ge2: u32 = 0;
    var it = freq.valueIterator();
    while (it.next()) |v| {
        distinct += 1;
        if (v.* >= 2) ge2 += 1;
    }
    try std.testing.expect(distinct > 0);

    const ratio: f32 = @as(f32, @floatFromInt(ge2)) / @as(f32, @floatFromInt(distinct));
    std.debug.print(
        "[live §3 fidelity] distinct vocab_ids={d}, n_j≥2: {d} ({d:.1}%)\n",
        .{ distinct, ge2, ratio * 100.0 },
    );
    try std.testing.expect(ratio >= 0.5);
}

test "live: tac.cluster runs end-to-end on the real corpus" {
    const a = std.testing.allocator;
    const buf = try loadFixture(a);
    defer a.free(buf);

    const td = try ParseBytes(buf);

    // Budget calibration for this small fixture (paper §3.3):
    //   - All 91 distinct vocab IDs have n_j < τ=256, so none enter the
    //     active tier — every token is in micro (n_j<μ=128, κ_j=1) or
    //     small (n_j≥μ, κ_j=2).
    //   - Phase 1 produces tail κ-sum = 90·1 + 1·2 = 92 (90 micro + 1
    //     small; see tools/LIVE_ENCODER_NOTES.md vocab table).
    //   - With no active tokens to absorb additional budget, κ_total must
    //     equal the tail sum exactly. Pick 92.
    //
    // Bigger corpora (e.g. retriever's #20 MS MARCO sample) will have
    // active tokens and can use larger κ_total — this fixture is
    // deliberately small to keep the live test cheap.
    const params: tac.tac.ClusteringParams = .{
        .kappa_total = 92,
        .seed = 0x20260501_deadbeef,
    };

    var result = try tac.tac.cluster(td, params, a);
    defer result.deinit(a);

    // Σ κ_j must equal kappa_total exactly (paper §3.3 Phase 4 invariant).
    var sum: u64 = 0;
    for (result.kappa_per_token) |k| sum += k;
    try std.testing.expectEqual(@as(u64, params.kappa_total), sum);

    // Every assignment must point inside the centroid array.
    for (result.assignments) |a_id| {
        try std.testing.expect(a_id < params.kappa_total);
    }

    std.debug.print(
        "[live §3 cluster] κ={d} centroids over {d} vectors, WCSS_total={d:.3}\n",
        .{ params.kappa_total, td.n_tokens, result.wcss_total },
    );
}
