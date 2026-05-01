//! benchmarks/common/queries.zig — encoded queries.bin + qids sidecar loader.
//!
//! Owner: retriever. Pairs with `tools/encode.py --mode queries` (task #29):
//! the encoder writes `queries.bin` in the standard token-dump format, treating
//! each query as a "doc" of tokens, plus a sidecar `<queries.bin>.qids` —
//! flat little-endian u32 array, one qid per query in order.
//!
//! At load time we mmap (or read) both files, validate cardinality matches
//! `n_docs` from the dump header, and expose them as a `EncodedQueries`
//! struct that the harness folds into `tac.retrieval.bench.QueryPack`.

const std = @import("std");
const Allocator = std.mem.Allocator;

const tac = @import("tac");
const token_dump = tac.io.token_dump;

pub const QueriesError = error{
    QidCountMismatch,
    SidecarReadFailed,
} || token_dump.TokenDumpError || Allocator.Error;

pub const EncodedQueries = struct {
    /// Owns the queries.bin byte buffer (so the parsed dump's slice views
    /// stay alive). Caller frees via `deinit`.
    bytes: []align(8) u8,
    qids: []u32,
    dump: token_dump.TokenDump,

    pub fn deinit(self: *EncodedQueries, gpa: Allocator) void {
        gpa.free(self.bytes);
        gpa.free(self.qids);
        self.* = undefined;
    }

    pub fn nQueries(self: *const EncodedQueries) u64 {
        return self.dump.n_docs;
    }
};

pub fn loadFromBytes(
    bytes: []align(8) u8,
    qids_bytes: []const u8,
    gpa: Allocator,
) QueriesError!EncodedQueries {
    const dump = try token_dump.parseBytes(bytes);

    const expected_qids_bytes: usize = @intCast(dump.n_docs * @sizeOf(u32));
    if (qids_bytes.len != expected_qids_bytes) return error.QidCountMismatch;

    const qids = try gpa.alloc(u32, @intCast(dump.n_docs));
    errdefer gpa.free(qids);
    var i: usize = 0;
    while (i < qids.len) : (i += 1) {
        qids[i] = std.mem.readInt(u32, qids_bytes[i * 4 ..][0..4], .little);
    }

    return .{ .bytes = bytes, .qids = qids, .dump = dump };
}

// File-based loader was removed in the Zig 0.16 std.Io migration; the
// per-dataset harness slurps both files via `std.Io` and hands the byte
// buffers to `loadFromBytes`. See benchmarks/common/runner.zig.

// ---------------------------------------------------------------------------
// Tests — round-trip a synthetic dump via the encoder's writer and verify
// that the loader recovers byte-identical queries + qids.
// ---------------------------------------------------------------------------

const testing = std.testing;

test "queries: round-trip 3-query synthetic dump + qids sidecar" {
    const gpa = testing.allocator;

    // Build a tiny 3-query × 4-token×dim=2 dump in-memory.
    const dim: u32 = 2;
    const doc_offsets = [_]u64{ 0, 2, 3, 4 };
    const token_ids = [_]u32{ 0, 1, 2, 3 };
    const vectors = [_]f32{
        1.0, 0.0, // q0 token 0
        0.0, 1.0, // q0 token 1
        1.0, 0.0, // q1 token 0
        0.0, 1.0, // q2 token 0
    };
    const build = token_dump.DumpBuild{
        .dim = dim,
        .doc_offsets = &doc_offsets,
        .token_ids = &token_ids,
        .vectors = &vectors,
    };
    const dump_bytes = try token_dump.writeAlloc(gpa, build);
    errdefer gpa.free(dump_bytes);

    // Build the qids sidecar: 3 qids, little-endian.
    const qids_in = [_]u32{ 100, 200, 300 };
    var qids_bytes: [12]u8 = undefined;
    for (qids_in, 0..) |qid, idx| {
        std.mem.writeInt(u32, qids_bytes[idx * 4 ..][0..4], qid, .little);
    }

    var eq = try loadFromBytes(dump_bytes, &qids_bytes, gpa);
    defer eq.deinit(gpa);

    try testing.expectEqual(@as(u64, 3), eq.nQueries());
    try testing.expectEqualSlices(u32, &qids_in, eq.qids);
    try testing.expectEqual(@as(u32, 2), eq.dump.dim);
}

test "queries: qids cardinality mismatch is rejected" {
    const gpa = testing.allocator;
    const dim: u32 = 2;
    const doc_offsets = [_]u64{ 0, 1 };
    const token_ids = [_]u32{0};
    const vectors = [_]f32{ 1.0, 0.0 };
    const dump_bytes = try token_dump.writeAlloc(gpa, .{
        .dim = dim,
        .doc_offsets = &doc_offsets,
        .token_ids = &token_ids,
        .vectors = &vectors,
    });
    defer gpa.free(dump_bytes);

    // 1 query but 2 qids in the sidecar.
    var qids_bytes: [8]u8 = undefined;
    std.mem.writeInt(u32, qids_bytes[0..4], 100, .little);
    std.mem.writeInt(u32, qids_bytes[4..8], 200, .little);

    // loadFromBytes takes ownership of bytes on success; on this error path
    // we hold a separate ref. Make a copy to round-trip the test cleanly.
    const copy = try gpa.alignedAlloc(u8, .@"8", dump_bytes.len);
    defer gpa.free(copy);
    @memcpy(copy, dump_bytes);
    try testing.expectError(error.QidCountMismatch, loadFromBytes(copy, &qids_bytes, gpa));
}
