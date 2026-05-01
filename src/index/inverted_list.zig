//! src/index/inverted_list.zig — per-centroid inverted lists of doc IDs.
//!
//! Owner: indexer.
//! See plan 03-index-pq-hnsw-storage.plan.md.
//!
//! Per paper §4: `L_j = { d : ∃ token t in d s.t. assign(t) = j }`.
//! Document-level grain (NOT token-level). De-dup doc IDs at build time —
//! that's what makes the gather phase cheap (paper §5.1: walk inverted lists,
//! accumulate per-doc max ⟨q_i, c_j⟩ without ever touching a PQ code).
//!
//! ---------------------------------------------------------------------------
//! DESIGN PSEUDOCODE — to be implemented in task #14, after #11 + #13.
//!
//! On-disk and in-memory layout: CSR (compressed sparse row).
//!   offsets:  []u64  of length kappa + 1         // monotonically non-decreasing
//!   payload:  []u32                              // sorted doc IDs per row
//!
//! Centroid j's posting list is `payload[offsets[j] .. offsets[j+1]]`.
//! Sorted ascending so we can binary-search and so on-disk delta-coding (a
//! follow-up todo) becomes trivial later.
//!
//! ---------------------------------------------------------------------------
//! Public API:
//!
//!   pub const InvertedLists = struct {
//!       kappa: u32,
//!       offsets: []u64,                  // len = kappa + 1
//!       payload: []u32,                  // len = offsets[kappa]
//!       owns_buffers: bool,              // true if heap-built; false if mmap-borrowed
//!   };
//!
//!   pub fn build(
//!       assignments: []const u32,        // n_tokens entries; assignments[i] ∈ [0, kappa)
//!       doc_offsets: []const u32,        // n_docs+1; doc d owns tokens [doc_offsets[d] .. doc_offsets[d+1])
//!       kappa: u32,
//!       gpa: Allocator,
//!   ) !InvertedLists;
//!
//!   pub fn list(self: *const InvertedLists, j: u32) []const u32;
//!   pub fn deinit(self: *InvertedLists, gpa: Allocator) void;
//!
//! ---------------------------------------------------------------------------
//! Build algorithm — two-pass CSR, O(n_tokens + kappa·log(avg_list)) time:
//!
//! 1. counts = [0; kappa];
//!    For each doc d:
//!        seen = bitset over kappa, cleared via a "stamp" buffer for O(touched);
//!        for tok in doc_offsets[d] .. doc_offsets[d+1]:
//!            j = assignments[tok];
//!            if !seen[j]:
//!                seen.set(j); stamps[++stamp_top] = j;
//!                counts[j] += 1;
//!        // clear seen via stamps so the next doc starts fresh
//!        while stamp_top > 0: seen.clear(stamps[stamp_top--]);
//!
//! 2. offsets[0] = 0; offsets[j+1] = offsets[j] + counts[j];
//!    cursor = offsets.dup();             // write head per centroid
//!    payload = gpa.alloc(u32, offsets[kappa]);
//!
//! 3. For each doc d (same de-dup pattern):
//!        for j unique-in-doc:
//!            payload[cursor[j]] = d; cursor[j] += 1;
//!
//! 4. (Optional) for j in 0..kappa: std.sort(payload[offsets[j] .. offsets[j+1]]).
//!    Already in ascending doc order if we iterate docs in order in step 3 —
//!    so the sort is unnecessary. Document this invariant in code.
//!
//! De-dup data structure choice: a "stamp" set (u32 generation counter array
//! of size kappa) avoids O(kappa) clear per doc. Memory: kappa·4B; for
//! kappa=4M that's 16 MiB — acceptable.
//!
//! ---------------------------------------------------------------------------
//! Tests planned for #14:
//!   - 5 docs, 4 tokens each, kappa=8, hand-checked posting lists.
//!   - A doc whose tokens all land in the same centroid appears in that list
//!     exactly once (de-dup correctness).
//!   - Empty doc (zero tokens) does NOT appear in any list.
//!   - offsets monotonic; offsets[kappa] == payload.len.
//!   - Per-row payload is ascending.

const std = @import("std");

test "CSR layout invariants are sane" {
    // Smoke test of intended invariant: offsets monotonic, last == payload.len.
    const offsets = [_]u64{ 0, 2, 2, 5 };
    const payload = [_]u32{ 0, 3, 1, 2, 4 };
    try std.testing.expectEqual(@as(u64, payload.len), offsets[offsets.len - 1]);
    var i: usize = 1;
    while (i < offsets.len) : (i += 1) {
        try std.testing.expect(offsets[i] >= offsets[i - 1]);
    }
}
