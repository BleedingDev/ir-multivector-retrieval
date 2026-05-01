//! src/index/storage.zig — on-disk index format (paper §4 layout).
//!
//! Owner: indexer.
//! See plan 03-index-pq-hnsw-storage.plan.md.
//!
//! Per paper §4 doc layout (Pass 1 streams centroid IDs, Pass 2 PQ codes):
//!
//!   doc d:  [c_1 .. c_{n_d} : u32  |  PQ_{1,1} .. PQ_{n_d, M} : u8]
//!
//! On-disk layout (single file, mmap-friendly, little-endian, all sections
//! aligned to 8 bytes via zero-padding):
//!
//!   ┌─────────────────────────────────────────────────────────────────┐
//!   │ HEADER (64 bytes, fixed)                                        │
//!   │   magic         : [8]u8  = constants.INDEX_MAGIC ("TAC_IDX1")   │
//!   │   version       : u32     = constants.INDEX_VERSION             │
//!   │   dim           : u32                                            │
//!   │   kappa         : u32     // # centroids                         │
//!   │   pq_M          : u32     // = 32                                │
//!   │   pq_bits       : u8      // = 8                                 │
//!   │   _pad          : [3]u8                                          │
//!   │   sub_dim       : u32     // = dim / pq_M, redundant for sanity  │
//!   │   n_docs        : u64                                            │
//!   │   n_tokens      : u64     // total tokens across all docs        │
//!   │   centroids_off : u64                                            │
//!   │   hnsw_off      : u64                                            │
//!   │   pq_off        : u64                                            │
//!   │   ilist_off     : u64                                            │
//!   │   doc_off       : u64                                            │
//!   │   norms_off     : u64                                            │
//!   │   footer_off    : u64                                            │
//!   └─────────────────────────────────────────────────────────────────┘
//!     (Total header = 8+4+4+4+4+1+3+4+8+8 + 8·8 = 112 bytes; round up to
//!      a 128-byte cache-line boundary with zero padding so subsequent
//!      sections start aligned. Final size pinned in code.)
//!
//!   centroids_off →  [f32 ; kappa * dim]                                │
//!   hnsw_off      →  HNSW serialised:                                   │
//!                       max_level u8, _pad[7]u8, entry_point u32, _pad[4]u8,
//!                       n u32, _pad[4]u8,
//!                       node_levels [n]u8, _pad to 8B,
//!                       per layer L = max_level..0:
//!                          offsets [n+1]u32, _pad to 8B,
//!                          payload [offsets[n]]u32, _pad to 8B
//!   pq_off        →  [f32 ; pq_M * 256 * sub_dim]                       │
//!   ilist_off     →  inverted lists CSR:
//!                       offsets [kappa+1]u64
//!                       payload [offsets[kappa]]u32, _pad to 8B
//!   doc_off       →  per-doc layouts:
//!                       doc_index [n_docs+1]u64    // byte offsets into doc payload
//!                       doc payload, packed: for each doc d
//!                          n_d         u32
//!                          centroid_ids[n_d]  u32       (Pass 1 region)
//!                          pq_codes   [n_d * pq_M] u8   (Pass 2 region)
//!                          _pad to 8B
//!   norms_off     →  [f32 ; n_tokens]   // residual norms, paper §4
//!                                       // "homogeneous compression" side data
//!   footer_off    →  crc32 u32 of bytes [0 .. footer_off), _pad[4]u8
//!
//! ---------------------------------------------------------------------------
//! Validation on open():
//!   - magic == INDEX_MAGIC                       → else error.InvalidIndex
//!   - version == INDEX_VERSION                   → else error.UnsupportedVersion
//!   - sub_dim * pq_M == dim                      → else error.InvalidIndex
//!   - all *_off within file bounds, monotonic    → else error.InvalidIndex
//!   - footer crc32 matches recomputed value      → else error.IndexCorrupt
//!
//! ---------------------------------------------------------------------------
//! Public API (locked by tasks #14, #15):
//!
//!   pub const Index = struct {
//!       header: Header,                    // parsed copy
//!       centroids: []const f32,            // borrowed slice into mmap region
//!       pq: pq.PQ,                         // codebooks slice borrowed
//!       hnsw: hnsw.Hnsw,                   // CSR slices borrowed
//!       ilists: inverted_list.InvertedLists,
//!       doc_index: []const u64,
//!       doc_payload: []const u8,
//!       residual_norms: []const f32,
//!       _mmap: ?MmapHandle,                // set when opened from disk
//!   };
//!
//!   pub fn build(
//!       td: *const io.token_dump.TokenDump,
//!       params: BuildParams,
//!       out_path: []const u8,
//!       gpa: Allocator,
//!   ) !void;
//!
//!   pub fn open(path: []const u8, gpa: Allocator) !Index;
//!   pub fn deinit(self: *Index, gpa: Allocator) void;
//!
//!   pub const BuildParams = struct {
//!       kappa_total: u32,
//!       seed: u64,
//!       // TAC defaults pulled from constants; can be overridden for tests.
//!       mu: u32 = constants.TAC_MU,
//!       tau: u32 = constants.TAC_TAU,
//!       epsilon: u32 = constants.TAC_EPSILON,
//!       theta: u32 = constants.TAC_THETA,
//!   };
//!
//! ---------------------------------------------------------------------------
//! Index.build pipeline (task #15) — paper §3 + §4 cited per stage:
//!
//!   1. (paper §3) result = tac.cluster(td, params)
//!         → centroids [kappa·dim], assignments [n_tokens]
//!   2. (paper §4 "homogeneous compression")
//!         residuals[i] = token_vec[i] - centroids[assignments[i]]
//!         norms[i]     = ‖residuals[i]‖
//!         residuals[i] /= norms[i]            // unit-normalise in place
//!   3. (paper §4) pq_model = pq.train(residuals, dim, seed, gpa)
//!   4. for each token i: pq_codes[i] = pq_model.encode(residuals[i])
//!      → flat [n_tokens · pq_M] u8
//!   5. (paper §4 HNSW over centroids) graph = hnsw.build(centroids, dim, seed, gpa)
//!   6. (paper §4 inverted lists, doc-level) ilists = inverted_list.build(
//!         assignments, td.doc_offsets, kappa, gpa)
//!   7. (paper §4 per-doc layout, Pass1 + Pass2) for each doc d:
//!         emit n_d, then assignments[doc_offsets[d] .. doc_offsets[d+1]],
//!         then pq_codes[doc_offsets[d]·pq_M .. doc_offsets[d+1]·pq_M]
//!   8. Write file with header, sections, footer crc32.
//!
//! ---------------------------------------------------------------------------
//! Tests planned for #14 (round-trip) and #15 (integration):
//!   - Build a 50-doc, 16-vocab, dim=8 fixture; kappa=64; M=2; bits=8 (overrides
//!     paper-strict M=32 for tiny test geometry — `// paper-gap:` annotated).
//!     Wait — PQ_M is in constants. For a fixture-sized test we can't change
//!     the global. Either pin dim to a multiple of 32 (e.g. dim=64 → sub_dim=2)
//!     or expose `PQ.train` to take `m` as an arg. We choose dim=64 to stay
//!     paper-strict on M=32; sub_dim=2 is fine for 1k-token fixtures.
//!   - After build → open → assert magic, version, every section's bytes
//!     equal the in-memory builder output (round-trip byte-equal).
//!   - Tampered magic → error.InvalidIndex.
//!   - Tampered footer crc → error.IndexCorrupt.

const std = @import("std");
const constants = @import("../constants.zig");

/// Header struct laid out to match the on-disk format. `extern` ensures field
/// order and no implicit padding beyond what we declare; the explicit `_pad`
/// fields make the layout match the comment above byte-for-byte.
pub const Header = extern struct {
    magic: [8]u8,
    version: u32,
    dim: u32,
    kappa: u32,
    pq_M: u32,
    pq_bits: u8,
    _pad0: [3]u8,
    sub_dim: u32,
    n_docs: u64,
    n_tokens: u64,
    centroids_off: u64,
    hnsw_off: u64,
    pq_off: u64,
    ilist_off: u64,
    doc_off: u64,
    norms_off: u64,
    footer_off: u64,
};

test "Header layout sanity" {
    // 8(magic) + 4(version) + 4(dim) + 4(kappa) + 4(pq_M) + 1(pq_bits)
    // + 3(_pad0) + 4(sub_dim) + 2·8(n_docs, n_tokens) + 7·8(section offsets)
    // + 1·8(footer_off) = 104. We will pad the on-disk header region to 128
    // bytes so the first centroid section starts on a 16-byte boundary.
    try std.testing.expectEqual(@as(usize, 104), @sizeOf(Header));
}

test "magic + version constants match paper repo conventions" {
    try std.testing.expectEqualSlices(u8, "TAC_IDX1", &constants.INDEX_MAGIC);
    try std.testing.expectEqual(@as(u32, 1), constants.INDEX_VERSION);
}
