//! src/index/inverted_list.zig — per-centroid inverted lists of doc IDs.
//!
//! Owner: indexer. Plan: 03-index-pq-hnsw-storage.plan.md.
//!
//! Per paper §4: `L_j = { d : ∃ token t in d s.t. assign(t) = j }`.
//! Document-level grain (NOT token-level). Multiple tokens of the same doc
//! mapping to the same centroid contribute one entry — that's what makes
//! the gather phase (paper §5.1) cheap: walk centroid posting lists,
//! aggregate per-doc max ⟨q_i, c_j⟩ without ever touching a PQ code.
//!
//! Layout: CSR. `offsets[j+1] - offsets[j]` is the size of L_j;
//! `payload[offsets[j] .. offsets[j+1]]` is the doc IDs (ascending).

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const InvertedListError = error{
    AssignmentOutOfRange,
    DocOffsetsBad,
    KappaZero,
} || Allocator.Error;

/// Compressed-sparse-row view of all centroid posting lists.
///
/// Lifetime: the buffers are HEAP-OWNED when this struct is returned by
/// `build`, and BORROWED when reconstructed from a mmap region. Use
/// `deinit` only when `owns_buffers == true`.
pub const InvertedLists = struct {
    kappa: u32,
    offsets: []u64, // len = kappa + 1
    payload: []u32, // len = offsets[kappa]
    owns_buffers: bool,

    pub fn deinit(self: *InvertedLists, gpa: Allocator) void {
        if (self.owns_buffers) {
            gpa.free(self.offsets);
            gpa.free(self.payload);
        }
        self.* = undefined;
    }

    /// Posting list for centroid `j`. Returned slice is alive as long as the
    /// underlying buffers are.
    pub fn list(self: *const InvertedLists, j: u32) []const u32 {
        std.debug.assert(j < self.kappa);
        const start: usize = @intCast(self.offsets[j]);
        const end: usize = @intCast(self.offsets[j + 1]);
        return self.payload[start..end];
    }
};

/// Build inverted lists from per-token assignments and document offsets.
///
/// `assignments[i]` is the centroid id (∈ [0, kappa)) of token `i`.
/// `doc_offsets[d] .. doc_offsets[d+1]` is the half-open token range of
/// document `d` (CSR semantics, matches `io.token_dump.TokenDump`).
///
/// Algorithm: two-pass CSR.
///   Pass 1 — count uniques per (doc, centroid) via a "stamp" set so we
///            never scan kappa to clear between docs (O(touched) clears).
///   Pass 2 — write doc IDs into payload at per-centroid cursors. Because
///            we iterate docs in ascending order, each posting list comes
///            out ascending without a sort.
pub fn build(
    assignments: []const u32,
    doc_offsets: []const u64,
    kappa: u32,
    gpa: Allocator,
) InvertedListError!InvertedLists {
    if (kappa == 0) return error.KappaZero;
    if (doc_offsets.len < 1) return error.DocOffsetsBad;

    const n_docs: u64 = @as(u64, doc_offsets.len) - 1;
    if (n_docs > std.math.maxInt(u32)) return error.DocOffsetsBad;
    const n_tokens: u64 = doc_offsets[doc_offsets.len - 1];
    if (assignments.len != @as(usize, @intCast(n_tokens))) return error.DocOffsetsBad;

    if (doc_offsets[0] != 0) return error.DocOffsetsBad;
    var d_check: usize = 1;
    while (d_check < doc_offsets.len) : (d_check += 1) {
        if (doc_offsets[d_check] < doc_offsets[d_check - 1]) {
            return error.DocOffsetsBad;
        }
    }
    for (assignments) |a| {
        if (a >= kappa) return error.AssignmentOutOfRange;
    }

    // Stamp set: stamps[j] holds the doc id for which we last saw centroid
    // j. Compare against the running doc id; sentinel u64::max = "never
    // seen" — n_docs ≤ u32::max so a u64 doc id never collides.
    const stamps = try gpa.alloc(u64, kappa);
    defer gpa.free(stamps);
    @memset(stamps, std.math.maxInt(u64));

    // ---- Pass 1: per-centroid de-duped counts. ----
    const offsets = try gpa.alloc(u64, @as(usize, kappa) + 1);
    errdefer gpa.free(offsets);
    @memset(offsets, 0);

    var d: u32 = 0;
    while (d < n_docs) : (d += 1) {
        const lo: usize = @intCast(doc_offsets[d]);
        const hi: usize = @intCast(doc_offsets[d + 1]);
        var t: usize = lo;
        while (t < hi) : (t += 1) {
            const j = assignments[t];
            if (stamps[j] != d) {
                stamps[j] = d;
                offsets[@as(usize, j) + 1] += 1;
            }
        }
    }

    // Exclusive scan: offsets[0]=0, offsets[j+1]=Σ counts up to j.
    var total: u64 = 0;
    var j_scan: u32 = 0;
    while (j_scan < kappa) : (j_scan += 1) {
        total += offsets[@as(usize, j_scan) + 1];
        offsets[@as(usize, j_scan) + 1] = total;
    }

    // ---- Pass 2: write doc IDs at per-centroid cursors. ----
    const payload = try gpa.alloc(u32, @as(usize, @intCast(total)));
    errdefer gpa.free(payload);

    const cursor = try gpa.alloc(u64, kappa);
    defer gpa.free(cursor);
    @memcpy(cursor, offsets[0..kappa]);

    @memset(stamps, std.math.maxInt(u64));
    d = 0;
    while (d < n_docs) : (d += 1) {
        const lo: usize = @intCast(doc_offsets[d]);
        const hi: usize = @intCast(doc_offsets[d + 1]);
        var t: usize = lo;
        while (t < hi) : (t += 1) {
            const j = assignments[t];
            if (stamps[j] != d) {
                stamps[j] = d;
                const slot: usize = @intCast(cursor[j]);
                payload[slot] = d;
                cursor[j] += 1;
            }
        }
    }

    return .{
        .kappa = kappa,
        .offsets = offsets,
        .payload = payload,
        .owns_buffers = true,
    };
}

// ---------------------------------------------------------------------------
// TESTS
// ---------------------------------------------------------------------------

const testing = std.testing;

test "build: hand-checked tiny case" {
    const a = std.testing.allocator;
    // 3 docs, 2 tokens each, kappa=4.
    //   doc 0 tokens → centroids {1, 3}
    //   doc 1 tokens → centroids {1, 1}   ← dedup to {1}
    //   doc 2 tokens → centroids {0, 3}
    const assignments = [_]u32{ 1, 3, 1, 1, 0, 3 };
    const doc_offsets = [_]u64{ 0, 2, 4, 6 };
    var lists = try build(&assignments, &doc_offsets, 4, a);
    defer lists.deinit(a);

    try testing.expectEqual(@as(u32, 4), lists.kappa);
    try testing.expectEqualSlices(u32, &.{2}, lists.list(0)); // centroid 0: doc 2
    try testing.expectEqualSlices(u32, &.{ 0, 1 }, lists.list(1)); // centroid 1: docs 0, 1
    try testing.expectEqual(@as(usize, 0), lists.list(2).len); // centroid 2: empty
    try testing.expectEqualSlices(u32, &.{ 0, 2 }, lists.list(3)); // centroid 3: docs 0, 2
}

test "build: per-row payload is ascending" {
    const a = std.testing.allocator;
    const assignments = [_]u32{ 0, 0, 0, 0, 0 };
    const doc_offsets = [_]u64{ 0, 1, 2, 3, 4, 5 };
    var lists = try build(&assignments, &doc_offsets, 1, a);
    defer lists.deinit(a);
    try testing.expectEqualSlices(u32, &.{ 0, 1, 2, 3, 4 }, lists.list(0));
}

test "build: doc with all tokens in one centroid appears once (de-dup)" {
    const a = std.testing.allocator;
    const assignments = [_]u32{ 2, 2, 2, 2 };
    const doc_offsets = [_]u64{ 0, 4 };
    var lists = try build(&assignments, &doc_offsets, 5, a);
    defer lists.deinit(a);
    try testing.expectEqualSlices(u32, &.{0}, lists.list(2));
    try testing.expectEqual(@as(u64, 1), lists.offsets[5]);
}

test "build: empty doc skipped (zero tokens contribute zero entries)" {
    const a = std.testing.allocator;
    // doc 0 has tokens [0]; doc 1 is empty; doc 2 has tokens [1].
    const assignments = [_]u32{ 0, 1 };
    const doc_offsets = [_]u64{ 0, 1, 1, 2 };
    var lists = try build(&assignments, &doc_offsets, 3, a);
    defer lists.deinit(a);
    try testing.expectEqualSlices(u32, &.{0}, lists.list(0));
    try testing.expectEqualSlices(u32, &.{2}, lists.list(1));
    try testing.expectEqual(@as(usize, 0), lists.list(2).len);
}

test "build: assignment out of range is an error" {
    const a = std.testing.allocator;
    const assignments = [_]u32{ 0, 5 };
    const doc_offsets = [_]u64{ 0, 2 };
    try testing.expectError(
        error.AssignmentOutOfRange,
        build(&assignments, &doc_offsets, 3, a),
    );
}

test "build: non-monotonic doc_offsets is an error" {
    const a = std.testing.allocator;
    const assignments = [_]u32{ 0, 0, 0 };
    const doc_offsets = [_]u64{ 0, 2, 1, 3 }; // 2 → 1 reverses
    try testing.expectError(
        error.DocOffsetsBad,
        build(&assignments, &doc_offsets, 1, a),
    );
}

test "build: offsets monotonic and last == payload.len invariant" {
    const a = std.testing.allocator;
    const assignments = [_]u32{ 0, 1, 2, 0, 1, 2, 0, 1, 2 };
    const doc_offsets = [_]u64{ 0, 3, 6, 9 };
    var lists = try build(&assignments, &doc_offsets, 3, a);
    defer lists.deinit(a);
    var i: usize = 1;
    while (i < lists.offsets.len) : (i += 1) {
        try testing.expect(lists.offsets[i] >= lists.offsets[i - 1]);
    }
    try testing.expectEqual(
        @as(u64, lists.payload.len),
        lists.offsets[lists.kappa],
    );
}
