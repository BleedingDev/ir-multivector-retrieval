//! src/io/token_dump.zig — flat-binary token-dump reader.
//!
//! Owner: primitives-engineer.
//! Format spec: docs/token-dump-format.md
//!
//! The dump is the bridge from `tools/encode.py` (Python ColBERTv2) into the
//! Zig pipeline: a single mmap-able blob carrying a CSR-style index of
//! per-document token vectors. We expose three slice views into the
//! caller-provided byte buffer so the clusterer / indexer / retriever can
//! read at f32 granularity without copies. Boundary validation is strict —
//! every offset, every count, and every float is checked before any view
//! escapes the parser.
//!
//! Two entry points:
//!   - `parseBytes`  — the canonical, allocator-free, IO-free parser. Caller
//!                     is responsible for the lifetime of the byte buffer
//!                     (mmap region, embedded test fixture, …).
//!   - `openMmap`    — convenience that wraps `parseBytes` with the new
//!                     std.Io memory-mapped read path; returns an
//!                     `OwnedTokenDump` that carries the file + mapping
//!                     handles.
//!
//! The parser pinpoints alignment requirements explicitly. Vector and
//! offset slices are interpreted from byte offsets that are statically
//! aligned to their element size by the format design (see spec).

const std = @import("std");
const constants = @import("../constants.zig");
const fixture = @import("synthetic_fixture.zig");

pub const TokenDumpError = error{InvalidTokenDump};

/// Read-only view of a parsed token dump. All slices alias the source
/// byte buffer the caller passed in. The parser requires the buffer to be
/// 8-byte aligned at offset 0 — both `writeAlloc`'s allocator slice and
/// any mmap region naturally satisfy this. The ascending header layout
/// then lines up u64 / u32 / f32 sub-slices on their native alignment.
pub const TokenDump = struct {
    dim: u32,
    n_docs: u64,
    n_tokens: u64,
    /// Length `n_docs + 1`. `doc_offsets[d]..doc_offsets[d+1]` is the
    /// half-open token range of document `d` (CSR semantics).
    doc_offsets: []const u64,
    /// Length `n_tokens`. Vocabulary id of each token vector.
    token_ids: []const u32,
    /// Length `n_tokens * dim`. Row-major token embeddings.
    vectors: []const f32,

    /// Slice the embedding of token row `i`. O(1).
    pub fn vectorAt(self: TokenDump, i: u64) []const f32 {
        std.debug.assert(i < self.n_tokens);
        const start: usize = @intCast(i * self.dim);
        return self.vectors[start..][0..self.dim];
    }

    /// Tokens belonging to document `d` as a half-open `[start, end)` range.
    pub fn docTokenRange(self: TokenDump, d: u64) [2]u64 {
        std.debug.assert(d < self.n_docs);
        return .{ self.doc_offsets[d], self.doc_offsets[d + 1] };
    }
};

const header_size: usize = 40; // magic(8) + version(4) + dim(4) + n_docs(8) + n_tokens(8) + dtype(1) + reserved(7)
const max_dim: u32 = 4096; // sanity ceiling, see docs/token-dump-format.md

fn readU32Le(bytes: []const u8, offset: usize) u32 {
    return std.mem.readInt(u32, bytes[offset..][0..4], .little);
}

fn readU64Le(bytes: []const u8, offset: usize) u64 {
    return std.mem.readInt(u64, bytes[offset..][0..8], .little);
}

/// Validate `bytes` against the on-disk format and return slice views.
/// Returns `error.InvalidTokenDump` on any structural problem (bad magic,
/// bad version, NaN/Inf vectors, non-monotonic offsets, length mismatch).
///
/// `bytes` must be 8-byte aligned at offset 0 — any allocator-backed
/// buffer or mmap region trivially is. The returned slices alias `bytes`;
/// the caller must keep `bytes` alive for as long as the returned
/// `TokenDump` is used.
///
/// paper-gap: the paper doesn't specify on-disk endianness. We pick
/// little-endian to match every modern x86/ARM target we ship to;
/// big-endian hosts are rejected here.
pub fn parseBytes(bytes: []align(8) const u8) TokenDumpError!TokenDump {
    if (@import("builtin").cpu.arch.endian() != .little) {
        return error.InvalidTokenDump;
    }
    if (bytes.len < header_size) return error.InvalidTokenDump;

    if (!std.mem.eql(u8, bytes[0..8], &constants.TOKEN_DUMP_MAGIC)) {
        return error.InvalidTokenDump;
    }
    const version = readU32Le(bytes, 8);
    if (version != constants.TOKEN_DUMP_VERSION) return error.InvalidTokenDump;

    const dim = readU32Le(bytes, 12);
    if (dim == 0 or dim > max_dim) return error.InvalidTokenDump;

    const n_docs = readU64Le(bytes, 16);
    const n_tokens = readU64Le(bytes, 24);
    if (n_tokens == 0) return error.InvalidTokenDump;

    if (bytes[32] != 0) return error.InvalidTokenDump; // dtype: f32 only

    for (bytes[33..40]) |b| {
        if (b != 0) return error.InvalidTokenDump;
    }

    // Compute the expected total file size, watching for overflow.
    const offsets_count: u64 = n_docs + 1;
    const offsets_bytes: u64 = std.math.mul(u64, offsets_count, 8) catch
        return error.InvalidTokenDump;
    const ids_bytes: u64 = std.math.mul(u64, n_tokens, 4) catch
        return error.InvalidTokenDump;
    const vec_count: u64 = std.math.mul(u64, n_tokens, dim) catch
        return error.InvalidTokenDump;
    const vec_bytes: u64 = std.math.mul(u64, vec_count, 4) catch
        return error.InvalidTokenDump;

    const e1 = std.math.add(u64, header_size, offsets_bytes) catch
        return error.InvalidTokenDump;
    const e2 = std.math.add(u64, e1, ids_bytes) catch
        return error.InvalidTokenDump;
    const expected_total = std.math.add(u64, e2, vec_bytes) catch
        return error.InvalidTokenDump;

    if (expected_total != bytes.len) return error.InvalidTokenDump;

    const offsets_off: usize = header_size;
    const ids_off: usize = @intCast(offsets_off + offsets_bytes);
    const vec_off: usize = @intCast(ids_off + ids_bytes);

    // The header is exactly 40 bytes; offsets land at byte 40 (8-byte
    // aligned given a 8-byte aligned buffer). Each subsequent section's
    // alignment follows automatically: offsets_bytes is a multiple of 8,
    // and `ids_off + ids_bytes` is a multiple of 4, so the f32 region is
    // at least 4-byte aligned. We assert this and `@alignCast` to native
    // element alignment so callers get cheap aligned loads.
    std.debug.assert(offsets_off % @alignOf(u64) == 0);
    std.debug.assert(ids_off % @alignOf(u32) == 0);
    std.debug.assert(vec_off % @alignOf(f32) == 0);

    const offsets_slice: []const u64 = @alignCast(std.mem.bytesAsSlice(
        u64,
        bytes[offsets_off..ids_off],
    ));
    const ids_slice: []const u32 = @alignCast(std.mem.bytesAsSlice(
        u32,
        bytes[ids_off..vec_off],
    ));
    const vec_slice: []const f32 = @alignCast(std.mem.bytesAsSlice(
        f32,
        bytes[vec_off..],
    ));

    if (offsets_slice.len != offsets_count) return error.InvalidTokenDump;
    if (ids_slice.len != n_tokens) return error.InvalidTokenDump;
    if (vec_slice.len != vec_count) return error.InvalidTokenDump;

    // Validate doc_offsets — sequence shape and bounds.
    if (offsets_slice[0] != 0) return error.InvalidTokenDump;
    if (offsets_slice[offsets_slice.len - 1] != n_tokens) {
        return error.InvalidTokenDump;
    }
    var prev: u64 = 0;
    for (offsets_slice) |o| {
        if (o < prev) return error.InvalidTokenDump;
        if (o > n_tokens) return error.InvalidTokenDump;
        prev = o;
    }

    // Reject NaN / ±Inf token values up front. A single corrupt encoder
    // run shouldn't be allowed to silently poison every distance.
    for (vec_slice) |x| {
        if (!std.math.isFinite(x)) return error.InvalidTokenDump;
    }

    return .{
        .dim = dim,
        .n_docs = n_docs,
        .n_tokens = n_tokens,
        .doc_offsets = offsets_slice,
        .token_ids = ids_slice,
        .vectors = vec_slice,
    };
}

// ---------------------------------------------------------------------------
// Optional convenience: open + memory-map a file.
//
// Uses the new std.Io API. Returns an owning handle that must be released
// with `OwnedTokenDump.close`. The view sub-field aliases the mapped bytes.
// ---------------------------------------------------------------------------

pub const OwnedTokenDump = struct {
    view: TokenDump,
    /// Underlying file/mapping handles, kept alive for `view`.
    _file: std.Io.File,
    _map: std.Io.File.MemoryMap,

    pub fn close(self: *OwnedTokenDump, io: std.Io) void {
        self._map.destroy(io);
        self._file.close(io);
        self.* = undefined;
    }
};

pub const OpenError = error{
    InvalidTokenDump,
    FileSizeExceedsUsize,
} || std.Io.File.OpenError || std.Io.File.MemoryMap.CreateError ||
    std.Io.File.StatError;

pub fn openMmap(io: std.Io, dir: std.Io.Dir, sub_path: []const u8) OpenError!OwnedTokenDump {
    var file = try dir.openFile(io, sub_path, .{});
    errdefer file.close(io);

    const stat = try file.stat(io);
    const size_usize: usize = std.math.cast(usize, stat.size) orelse
        return error.FileSizeExceedsUsize;
    if (size_usize == 0) return error.InvalidTokenDump;

    var map = try std.Io.File.MemoryMap.create(io, file, .{
        .len = size_usize,
        .protection = .{ .read = true, .write = false },
    });
    errdefer map.destroy(io);

    const td = try parseBytes(map.memory[0..size_usize]);

    return .{ .view = td, ._file = file, ._map = map };
}

// ---------------------------------------------------------------------------
// Writer for tests and fixtures (allocator-driven, returns []u8).
// ---------------------------------------------------------------------------

/// In-memory description of a dump. Used by tests, the synthetic fixture
/// builder, and the encoder script's smoke tests.
pub const DumpBuild = struct {
    dim: u32,
    /// Length `n_docs + 1`. CSR offsets into `token_ids` / `vectors`.
    doc_offsets: []const u64,
    /// Length `n_tokens`.
    token_ids: []const u32,
    /// Length `n_tokens * dim`. Row-major.
    vectors: []const f32,
};

pub const WriteError = error{InvalidBuild} || std.mem.Allocator.Error;

/// Serialize `build` into the on-disk binary format and return an
/// allocator-owned, 8-byte aligned byte buffer the caller frees with
/// `freeAlloc`. The 8-byte alignment is required by `parseBytes`.
pub fn writeAlloc(allocator: std.mem.Allocator, build: DumpBuild) WriteError![]align(8) u8 {
    if (build.dim == 0) return error.InvalidBuild;
    if (build.doc_offsets.len == 0) return error.InvalidBuild;

    const n_docs: u64 = build.doc_offsets.len - 1;
    const n_tokens_from_offsets = build.doc_offsets[build.doc_offsets.len - 1];
    if (n_tokens_from_offsets != build.token_ids.len) return error.InvalidBuild;
    if (build.vectors.len != build.token_ids.len * build.dim) {
        return error.InvalidBuild;
    }
    const n_tokens: u64 = @intCast(build.token_ids.len);

    const offsets_bytes = build.doc_offsets.len * 8;
    const ids_bytes = build.token_ids.len * 4;
    const vec_bytes = build.vectors.len * 4;
    const total = header_size + offsets_bytes + ids_bytes + vec_bytes;

    const buf = try allocator.alignedAlloc(u8, .@"8", total);
    errdefer allocator.free(buf);

    @memcpy(buf[0..8], &constants.TOKEN_DUMP_MAGIC);
    std.mem.writeInt(u32, buf[8..12], constants.TOKEN_DUMP_VERSION, .little);
    std.mem.writeInt(u32, buf[12..16], build.dim, .little);
    std.mem.writeInt(u64, buf[16..24], n_docs, .little);
    std.mem.writeInt(u64, buf[24..32], n_tokens, .little);
    buf[32] = 0; // dtype
    @memset(buf[33..40], 0); // reserved

    var pos: usize = header_size;
    for (build.doc_offsets) |o| {
        std.mem.writeInt(u64, buf[pos..][0..8], o, .little);
        pos += 8;
    }
    for (build.token_ids) |id| {
        std.mem.writeInt(u32, buf[pos..][0..4], id, .little);
        pos += 4;
    }
    for (build.vectors) |v| {
        std.mem.writeInt(u32, buf[pos..][0..4], @bitCast(v), .little);
        pos += 4;
    }
    std.debug.assert(pos == total);
    return buf;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "round-trip: tiny dump (3 docs × 6 tokens × dim 4)" {
    const dim: u32 = 4;
    const offsets = [_]u64{ 0, 2, 3, 6 }; // doc lengths 2, 1, 3
    const ids = [_]u32{ 100, 101, 200, 300, 301, 302 };
    var vectors: [6 * 4]f32 = undefined;
    for (0..vectors.len) |i| vectors[i] = @floatFromInt(i);

    const buf = try writeAlloc(testing.allocator, .{
        .dim = dim,
        .doc_offsets = &offsets,
        .token_ids = &ids,
        .vectors = &vectors,
    });
    defer testing.allocator.free(buf);

    const td = try parseBytes(buf);
    try testing.expectEqual(@as(u32, 4), td.dim);
    try testing.expectEqual(@as(u64, 3), td.n_docs);
    try testing.expectEqual(@as(u64, 6), td.n_tokens);
    try testing.expectEqualSlices(u64, &offsets, td.doc_offsets);
    try testing.expectEqualSlices(u32, &ids, td.token_ids);
    try testing.expectEqualSlices(f32, &vectors, td.vectors);

    // vectorAt is a stride view.
    const v0 = td.vectorAt(0);
    try testing.expectEqual(@as(usize, 4), v0.len);
    try testing.expectEqual(@as(f32, 0.0), v0[0]);
    try testing.expectEqual(@as(f32, 1.0), v0[1]);

    const v5 = td.vectorAt(5);
    try testing.expectEqual(@as(f32, 20.0), v5[0]);
    try testing.expectEqual(@as(f32, 23.0), v5[3]);

    // docTokenRange.
    const r1 = td.docTokenRange(1);
    try testing.expectEqual(@as(u64, 2), r1[0]);
    try testing.expectEqual(@as(u64, 3), r1[1]);
}

test "round-trip: synthetic fixture" {
    const allocator = testing.allocator;
    var fx = try fixture.build(allocator, .{ .seed = 7, .n_docs = 5, .dim = 8 });
    defer fx.deinit(allocator);

    const buf = try writeAlloc(allocator, fx.toBuild());
    defer allocator.free(buf);

    const td = try parseBytes(buf);
    try testing.expectEqual(@as(u32, 8), td.dim);
    try testing.expectEqual(@as(u64, 5), td.n_docs);
    try testing.expectEqual(fx.n_tokens, td.n_tokens);
    try testing.expectEqualSlices(u64, fx.doc_offsets, td.doc_offsets);
    try testing.expectEqualSlices(u32, fx.token_ids, td.token_ids);
    try testing.expectEqualSlices(f32, fx.vectors, td.vectors);
}

test "parseBytes: bad magic returns InvalidTokenDump" {
    var buf: [header_size]u8 align(8) = undefined;
    @memcpy(buf[0..8], "BADMAGIC");
    @memset(buf[8..], 0);
    try testing.expectError(error.InvalidTokenDump, parseBytes(&buf));
}

test "parseBytes: truncated buffer returns InvalidTokenDump" {
    const buf: [16]u8 align(8) = undefined;
    try testing.expectError(error.InvalidTokenDump, parseBytes(&buf));
}

test "parseBytes: dim > max_dim rejected" {
    var buf: [header_size]u8 align(8) = undefined;
    @memcpy(buf[0..8], &constants.TOKEN_DUMP_MAGIC);
    std.mem.writeInt(u32, buf[8..12], constants.TOKEN_DUMP_VERSION, .little);
    std.mem.writeInt(u32, buf[12..16], 99_999, .little); // dim > max
    std.mem.writeInt(u64, buf[16..24], 0, .little); // n_docs
    std.mem.writeInt(u64, buf[24..32], 1, .little); // n_tokens
    buf[32] = 0;
    @memset(buf[33..40], 0);
    try testing.expectError(error.InvalidTokenDump, parseBytes(&buf));
}

test "parseBytes: dtype != 0 rejected" {
    var buf: [header_size]u8 align(8) = undefined;
    @memcpy(buf[0..8], &constants.TOKEN_DUMP_MAGIC);
    std.mem.writeInt(u32, buf[8..12], constants.TOKEN_DUMP_VERSION, .little);
    std.mem.writeInt(u32, buf[12..16], 4, .little);
    std.mem.writeInt(u64, buf[16..24], 0, .little);
    std.mem.writeInt(u64, buf[24..32], 1, .little);
    buf[32] = 1; // unsupported dtype
    @memset(buf[33..40], 0);
    try testing.expectError(error.InvalidTokenDump, parseBytes(&buf));
}

test "parseBytes: NaN in vectors is rejected" {
    const offsets = [_]u64{ 0, 2 };
    const ids = [_]u32{ 1, 2 };
    var vectors = [_]f32{ 0.1, 0.2, std.math.nan(f32), 0.4 };
    const buf = try writeAlloc(testing.allocator, .{
        .dim = 2,
        .doc_offsets = &offsets,
        .token_ids = &ids,
        .vectors = &vectors,
    });
    defer testing.allocator.free(buf);
    try testing.expectError(error.InvalidTokenDump, parseBytes(buf));
}

test "parseBytes: non-monotonic doc_offsets rejected" {
    // We bypass writeAlloc's validation by hand-crafting the dump.
    const dim: u32 = 1;
    const offsets = [_]u64{ 0, 2, 1, 3 }; // non-monotonic
    const ids = [_]u32{ 1, 2, 3 };
    const vectors = [_]f32{ 0.1, 0.2, 0.3 };

    const total = header_size + offsets.len * 8 + ids.len * 4 + vectors.len * 4;
    const buf = try testing.allocator.alignedAlloc(u8, .@"8", total);
    defer testing.allocator.free(buf);

    @memcpy(buf[0..8], &constants.TOKEN_DUMP_MAGIC);
    std.mem.writeInt(u32, buf[8..12], constants.TOKEN_DUMP_VERSION, .little);
    std.mem.writeInt(u32, buf[12..16], dim, .little);
    std.mem.writeInt(u64, buf[16..24], offsets.len - 1, .little);
    std.mem.writeInt(u64, buf[24..32], ids.len, .little);
    buf[32] = 0;
    @memset(buf[33..40], 0);

    var pos: usize = header_size;
    for (offsets) |o| {
        std.mem.writeInt(u64, buf[pos..][0..8], o, .little);
        pos += 8;
    }
    for (ids) |id| {
        std.mem.writeInt(u32, buf[pos..][0..4], id, .little);
        pos += 4;
    }
    for (vectors) |v| {
        std.mem.writeInt(u32, buf[pos..][0..4], @bitCast(v), .little);
        pos += 4;
    }

    try testing.expectError(error.InvalidTokenDump, parseBytes(buf));
}

test "parseBytes: trailing bytes rejected" {
    const offsets = [_]u64{ 0, 1 };
    const ids = [_]u32{1};
    var vectors = [_]f32{ 0.1, 0.2 };
    const buf = try writeAlloc(testing.allocator, .{
        .dim = 2,
        .doc_offsets = &offsets,
        .token_ids = &ids,
        .vectors = &vectors,
    });
    defer testing.allocator.free(buf);

    const padded = try testing.allocator.alignedAlloc(u8, .@"8", buf.len + 4);
    defer testing.allocator.free(padded);
    @memcpy(padded[0..buf.len], buf);
    @memset(padded[buf.len..], 0);
    try testing.expectError(error.InvalidTokenDump, parseBytes(padded));
}

test "writeAlloc: bad build (vectors length wrong) rejected" {
    const offsets = [_]u64{ 0, 2 };
    const ids = [_]u32{ 1, 2 };
    var vectors = [_]f32{ 0.1, 0.2 }; // should be 4 elements (n_tokens=2, dim=2)
    try testing.expectError(error.InvalidBuild, writeAlloc(testing.allocator, .{
        .dim = 2,
        .doc_offsets = &offsets,
        .token_ids = &ids,
        .vectors = &vectors,
    }));
}
