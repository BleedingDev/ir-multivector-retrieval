//! benchmarks/common/qrels.zig — TREC / MS MARCO qrels loader.
//!
//! Owner: retriever.
//!
//! qrels file formats:
//!   TREC (LoTTE-pooled):    qid<TAB>iter<TAB>doc_id<TAB>rel
//!   MS MARCO dev.small:     qid<TAB>0<TAB>doc_id<TAB>1
//!   (the two are byte-equivalent in practice; we pick the parser by
//!    relevance-grade threshold — TREC uses graded rel, MS MARCO is binary.)
//!
//! Output: a CSR-shaped `QrelsTable` keyed by qid, sorted ascending so
//! `forQuery(qid)` is an O(log n) binary search.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const QrelsError = error{
    MalformedQrelsLine,
    DuplicatePair,
} || Allocator.Error;

pub const QrelsTable = struct {
    /// Sorted ascending unique qids. Length n_queries.
    qids: []u32,
    /// CSR offsets into `doc_ids`. Length n_queries + 1.
    qrel_offsets: []u64,
    /// Relevant doc IDs, sorted ascending within each qid bucket.
    doc_ids: []u32,

    pub fn deinit(self: *QrelsTable, gpa: Allocator) void {
        gpa.free(self.qids);
        gpa.free(self.qrel_offsets);
        gpa.free(self.doc_ids);
        self.* = undefined;
    }

    /// Doc IDs relevant to `qid`. Empty slice if qid not present.
    pub fn forQuery(self: *const QrelsTable, qid: u32) []const u32 {
        var lo: usize = 0;
        var hi: usize = self.qids.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            const v = self.qids[mid];
            if (v == qid) {
                const a: usize = @intCast(self.qrel_offsets[mid]);
                const b: usize = @intCast(self.qrel_offsets[mid + 1]);
                return self.doc_ids[a..b];
            } else if (v < qid) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return &.{};
    }
};

const Pair = struct { qid: u32, doc_id: u32 };

fn cmpPair(_: void, a: Pair, b: Pair) bool {
    if (a.qid != b.qid) return a.qid < b.qid;
    return a.doc_id < b.doc_id;
}

/// Parse qrels in-memory from `bytes` (raw TSV). Caller frees the table.
///
/// `min_rel` is the relevance-grade threshold: keep only `rel >= min_rel`.
/// MS MARCO uses 1 (binary); TREC graded relevance commonly uses 1 too,
/// but harnesses sometimes filter at >=2.
pub fn parseBytes(bytes: []const u8, min_rel: i32, gpa: Allocator) QrelsError!QrelsTable {
    var pairs: std.ArrayList(Pair) = .empty;
    defer pairs.deinit(gpa);

    var line_it = std.mem.tokenizeScalar(u8, bytes, '\n');
    while (line_it.next()) |raw_line| {
        var line = raw_line;
        // Strip trailing CR (Windows-encoded files).
        if (line.len > 0 and line[line.len - 1] == '\r') line = line[0 .. line.len - 1];
        if (line.len == 0) continue;
        if (line[0] == '#') continue;

        // Split on TAB (canonical) or fall back to whitespace.
        var fields: [4][]const u8 = undefined;
        var fcount: usize = 0;
        var col_it = std.mem.tokenizeAny(u8, line, "\t ");
        while (col_it.next()) |f| {
            if (fcount >= 4) {
                fcount += 1;
                break;
            }
            fields[fcount] = f;
            fcount += 1;
        }
        if (fcount != 4) return error.MalformedQrelsLine;

        const qid = std.fmt.parseInt(u32, fields[0], 10) catch return error.MalformedQrelsLine;
        // fields[1] = iter (ignored).
        const doc_id = std.fmt.parseInt(u32, fields[2], 10) catch return error.MalformedQrelsLine;
        const rel = std.fmt.parseInt(i32, fields[3], 10) catch return error.MalformedQrelsLine;

        if (rel < min_rel) continue;
        try pairs.append(gpa, .{ .qid = qid, .doc_id = doc_id });
    }

    // Sort + dedupe in-place.
    std.sort.pdq(Pair, pairs.items, {}, cmpPair);

    // Count distinct qids and check dedupe.
    var n_q: usize = 0;
    if (pairs.items.len > 0) n_q = 1;
    var i: usize = 1;
    while (i < pairs.items.len) : (i += 1) {
        if (pairs.items[i].qid == pairs.items[i - 1].qid and
            pairs.items[i].doc_id == pairs.items[i - 1].doc_id)
        {
            return error.DuplicatePair;
        }
        if (pairs.items[i].qid != pairs.items[i - 1].qid) n_q += 1;
    }

    const qids = try gpa.alloc(u32, n_q);
    errdefer gpa.free(qids);
    const offsets = try gpa.alloc(u64, n_q + 1);
    errdefer gpa.free(offsets);
    const doc_ids = try gpa.alloc(u32, pairs.items.len);
    errdefer gpa.free(doc_ids);

    offsets[0] = 0;
    var qi: usize = 0;
    var written_docs: u64 = 0;
    var k: usize = 0;
    while (k < pairs.items.len) {
        const cur_qid = pairs.items[k].qid;
        qids[qi] = cur_qid;
        var j = k;
        while (j < pairs.items.len and pairs.items[j].qid == cur_qid) : (j += 1) {
            doc_ids[@intCast(written_docs)] = pairs.items[j].doc_id;
            written_docs += 1;
        }
        offsets[qi + 1] = written_docs;
        qi += 1;
        k = j;
    }

    return .{ .qids = qids, .qrel_offsets = offsets, .doc_ids = doc_ids };
}

// File-based loader was removed in the Zig 0.16 std.Io migration; the
// per-dataset harness slurps the file via `std.Io` and hands the bytes to
// `parseBytes`. See benchmarks/common/runner.zig.

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "qrels: parse 2 queries × 3 docs sorted" {
    const tsv = "10\t0\t100\t1\n10\t0\t200\t1\n20\t0\t300\t1\n";
    var t = try parseBytes(tsv, 1, testing.allocator);
    defer t.deinit(testing.allocator);

    try testing.expectEqualSlices(u32, &.{ 10, 20 }, t.qids);
    try testing.expectEqualSlices(u64, &.{ 0, 2, 3 }, t.qrel_offsets);
    try testing.expectEqualSlices(u32, &.{ 100, 200, 300 }, t.doc_ids);
    try testing.expectEqualSlices(u32, &.{ 100, 200 }, t.forQuery(10));
    try testing.expectEqualSlices(u32, &.{300}, t.forQuery(20));
    try testing.expectEqual(@as(usize, 0), t.forQuery(99).len);
}

test "qrels: rel < min_rel is filtered" {
    const tsv = "10\t0\t100\t2\n10\t0\t200\t1\n10\t0\t300\t0\n";
    var t = try parseBytes(tsv, 2, testing.allocator);
    defer t.deinit(testing.allocator);
    try testing.expectEqualSlices(u32, &.{100}, t.forQuery(10));
}

test "qrels: tabs and spaces both work" {
    const tsv = "10 0 100 1\n10\t0\t200\t1\n";
    var t = try parseBytes(tsv, 1, testing.allocator);
    defer t.deinit(testing.allocator);
    try testing.expectEqualSlices(u32, &.{ 100, 200 }, t.forQuery(10));
}

test "qrels: comments + blank lines tolerated" {
    const tsv = "# comment header\n\n10\t0\t100\t1\n\n# another comment\n10\t0\t200\t1\n";
    var t = try parseBytes(tsv, 1, testing.allocator);
    defer t.deinit(testing.allocator);
    try testing.expectEqualSlices(u32, &.{ 100, 200 }, t.forQuery(10));
}

test "qrels: malformed line returns error" {
    const tsv = "10 0 100\n"; // 3 columns, not 4
    try testing.expectError(error.MalformedQrelsLine, parseBytes(tsv, 1, testing.allocator));
}

test "qrels: duplicate (qid, doc_id) pairs are rejected" {
    const tsv = "10\t0\t100\t1\n10\t0\t100\t1\n";
    try testing.expectError(error.DuplicatePair, parseBytes(tsv, 1, testing.allocator));
}

test "qrels: forQuery returns empty for unknown qid" {
    const tsv = "10\t0\t100\t1\n";
    var t = try parseBytes(tsv, 1, testing.allocator);
    defer t.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 0), t.forQuery(999).len);
}

test "qrels: empty input → empty table" {
    var t = try parseBytes("", 1, testing.allocator);
    defer t.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 0), t.qids.len);
    try testing.expectEqual(@as(usize, 1), t.qrel_offsets.len);
    try testing.expectEqual(@as(u64, 0), t.qrel_offsets[0]);
}

test "qrels: CRLF line endings tolerated" {
    const tsv = "10\t0\t100\t1\r\n10\t0\t200\t1\r\n";
    var t = try parseBytes(tsv, 1, testing.allocator);
    defer t.deinit(testing.allocator);
    try testing.expectEqualSlices(u32, &.{ 100, 200 }, t.forQuery(10));
}

test "qrels: out-of-order input is sorted" {
    const tsv = "20\t0\t300\t1\n10\t0\t200\t1\n10\t0\t100\t1\n";
    var t = try parseBytes(tsv, 1, testing.allocator);
    defer t.deinit(testing.allocator);
    try testing.expectEqualSlices(u32, &.{ 10, 20 }, t.qids);
    try testing.expectEqualSlices(u32, &.{ 100, 200 }, t.forQuery(10));
}
