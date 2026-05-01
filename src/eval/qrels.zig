//! src/eval/qrels.zig — minimal qrels TSV reader for the `tac eval` /
//! `tac bench` CLI.
//!
//! The per-dataset benchmark harness lives at `benchmarks/common/qrels.zig`
//! and parses 4-column TREC / MS MARCO qrels (`qid<TAB>iter<TAB>doc_id<TAB>rel`).
//! The CLI takes a friendlier 2- or 3-column format produced ad-hoc:
//!
//!   `qid<TAB>doc_id`
//!   `qid<TAB>doc_id<TAB>rel`
//!
//! With the 4-column TREC layout also accepted (the iter column is ignored),
//! so a single CLI can consume both hand-built fixtures and the real qrels
//! files. Output is the same CSR-shaped table the harness expects, exposed
//! via a `forQuery(qid) -> []const u32` lookup.
//!
//! Owner: retriever. The 4-column TREC reader at benchmarks/common/qrels.zig
//! stays as-is — it has stricter validation appropriate for the per-dataset
//! Table 1 sweep harness.

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

/// Parse qrels from `bytes` (raw TSV). Caller frees the table via
/// `QrelsTable.deinit(gpa)`.
///
/// Accepted column counts per non-blank, non-comment line:
///   2: qid<TAB>doc_id                        (binary, all lines kept)
///   3: qid<TAB>doc_id<TAB>rel                (rel<min_rel ⇒ skipped)
///   4: qid<TAB>iter<TAB>doc_id<TAB>rel       (TREC; iter ignored)
///
/// Tabs and runs of spaces both delimit columns. CRLF tolerated. `#` lines
/// and blank lines are skipped. `min_rel` is ignored for the 2-column form.
pub fn parseBytes(bytes: []const u8, min_rel: i32, gpa: Allocator) QrelsError!QrelsTable {
    var pairs: std.ArrayList(Pair) = .empty;
    defer pairs.deinit(gpa);

    var line_it = std.mem.tokenizeScalar(u8, bytes, '\n');
    while (line_it.next()) |raw_line| {
        var line = raw_line;
        if (line.len > 0 and line[line.len - 1] == '\r') line = line[0 .. line.len - 1];
        if (line.len == 0) continue;
        if (line[0] == '#') continue;

        var fields: [4][]const u8 = undefined;
        var fcount: usize = 0;
        var col_it = std.mem.tokenizeAny(u8, line, "\t ");
        while (col_it.next()) |f| {
            if (fcount >= 4) {
                fcount += 1; // signal "too many"
                break;
            }
            fields[fcount] = f;
            fcount += 1;
        }

        const qid_str: []const u8 = fields[0];
        var doc_str: []const u8 = undefined;
        var rel: i32 = 1;
        switch (fcount) {
            2 => {
                doc_str = fields[1];
            },
            3 => {
                doc_str = fields[1];
                rel = std.fmt.parseInt(i32, fields[2], 10) catch return error.MalformedQrelsLine;
            },
            4 => {
                // TREC: qid iter doc_id rel
                doc_str = fields[2];
                rel = std.fmt.parseInt(i32, fields[3], 10) catch return error.MalformedQrelsLine;
            },
            else => return error.MalformedQrelsLine,
        }

        if (rel < min_rel) continue;

        const qid = std.fmt.parseInt(u32, qid_str, 10) catch return error.MalformedQrelsLine;
        const doc_id = std.fmt.parseInt(u32, doc_str, 10) catch return error.MalformedQrelsLine;
        try pairs.append(gpa, .{ .qid = qid, .doc_id = doc_id });
    }

    std.sort.pdq(Pair, pairs.items, {}, cmpPair);

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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "qrels (cli): 2-column TSV — qid + doc_id only" {
    const tsv = "1\t0\n2\t1768\n3\t0\n";
    var t = try parseBytes(tsv, 1, testing.allocator);
    defer t.deinit(testing.allocator);
    try testing.expectEqualSlices(u32, &.{ 1, 2, 3 }, t.qids);
    try testing.expectEqualSlices(u32, &.{0}, t.forQuery(1));
    try testing.expectEqualSlices(u32, &.{1768}, t.forQuery(2));
    try testing.expectEqualSlices(u32, &.{0}, t.forQuery(3));
}

test "qrels (cli): 3-column TSV with relevance grade" {
    const tsv = "10\t100\t2\n10\t200\t1\n20\t300\t0\n";
    var t = try parseBytes(tsv, 1, testing.allocator);
    defer t.deinit(testing.allocator);
    try testing.expectEqualSlices(u32, &.{10}, t.qids);
    try testing.expectEqualSlices(u32, &.{ 100, 200 }, t.forQuery(10));
}

test "qrels (cli): 4-column TREC TSV — iter column ignored" {
    const tsv = "10\t0\t100\t1\n10\t0\t200\t1\n20\t0\t300\t1\n";
    var t = try parseBytes(tsv, 1, testing.allocator);
    defer t.deinit(testing.allocator);
    try testing.expectEqualSlices(u32, &.{ 10, 20 }, t.qids);
    try testing.expectEqualSlices(u32, &.{ 100, 200 }, t.forQuery(10));
    try testing.expectEqualSlices(u32, &.{300}, t.forQuery(20));
}

test "qrels (cli): comments + CRLF + spaces all tolerated" {
    const tsv = "# header\r\n1 0\r\n2\t1768\r\n";
    var t = try parseBytes(tsv, 1, testing.allocator);
    defer t.deinit(testing.allocator);
    try testing.expectEqualSlices(u32, &.{0}, t.forQuery(1));
    try testing.expectEqualSlices(u32, &.{1768}, t.forQuery(2));
}

test "qrels (cli): malformed (single column) is rejected" {
    const tsv = "10\n";
    try testing.expectError(error.MalformedQrelsLine, parseBytes(tsv, 1, testing.allocator));
}

test "qrels (cli): >4 columns is rejected" {
    const tsv = "10\t0\t100\t1\textra\n";
    try testing.expectError(error.MalformedQrelsLine, parseBytes(tsv, 1, testing.allocator));
}

test "qrels (cli): duplicate (qid, doc_id) pairs are rejected" {
    const tsv = "10\t100\n10\t100\n";
    try testing.expectError(error.DuplicatePair, parseBytes(tsv, 1, testing.allocator));
}

test "qrels (cli): forQuery returns empty for unknown qid" {
    const tsv = "10\t100\n";
    var t = try parseBytes(tsv, 1, testing.allocator);
    defer t.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 0), t.forQuery(999).len);
}

test "qrels (cli): empty input → empty table" {
    var t = try parseBytes("", 1, testing.allocator);
    defer t.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 0), t.qids.len);
    try testing.expectEqual(@as(usize, 1), t.qrel_offsets.len);
}

test "qrels (cli): out-of-order input is sorted" {
    const tsv = "20\t300\n10\t200\n10\t100\n";
    var t = try parseBytes(tsv, 1, testing.allocator);
    defer t.deinit(testing.allocator);
    try testing.expectEqualSlices(u32, &.{ 10, 20 }, t.qids);
    try testing.expectEqualSlices(u32, &.{ 100, 200 }, t.forQuery(10));
}
