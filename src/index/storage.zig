//! src/index/storage.zig — on-disk index format (paper §4 layout).
//!
//! Owner: indexer. Plan: 03-index-pq-hnsw-storage.plan.md.
//!
//! Per paper §4 doc layout (Pass 1 streams centroid IDs, Pass 2 PQ codes):
//!
//!   doc d:  [c_1 .. c_{n_d} : u32  |  PQ_{1,1} .. PQ_{n_d, M} : u8]
//!
//! On-disk format (single file, mmap-friendly, little-endian, every section
//! starts on an 8-byte boundary). Section order:
//!
//!   1. Header (104 bytes, padded to 128).
//!   2. Centroids       — `kappa * dim` f32, row-major.
//!   3. HNSW graph      — serialised CSR per layer (see writeHnsw).
//!   4. PQ codebooks    — `PQ_M * 256 * sub_dim` f32, row-major.
//!   5. Inverted lists  — CSR (`offsets [kappa+1]u64`, `payload`).
//!   6. Per-doc layouts — for each doc d: n_d u32, centroid_ids u32[n_d],
//!                        pq_codes u8[n_d * PQ_M], pad to 8B.
//!                        Preceded by `doc_index [n_docs+1]u64` byte-offsets
//!                        into this section.
//!   7. Doc token offsets — `[n_docs+1]u64`. Mirrors `TokenDump.doc_offsets`
//!                          shape: `[lo, hi)` global token range per doc, so
//!                          refine (paper §5.3) can slice `residual_norms`
//!                          for a candidate in O(1).
//!   8. Residual norms  — `n_tokens` f32 (paper §4 "homogeneous compression").
//!   9. Footer          — crc32 u32 of bytes [0 .. footer_off), pad[4]u8.
//!
//! Validation on parse:
//!   - magic == INDEX_MAGIC                   → else error.InvalidIndex
//!   - version == INDEX_VERSION               → else error.UnsupportedVersion
//!   - sub_dim * pq_M == dim                  → else error.InvalidIndex
//!   - all *_off in bounds and monotonic      → else error.InvalidIndex
//!   - footer crc32 matches recomputed value  → else error.IndexCorrupt

const std = @import("std");
const Allocator = std.mem.Allocator;
const constants = @import("../constants.zig");
const pq_mod = @import("pq.zig");
const hnsw_mod = @import("hnsw.zig");
const inverted_list = @import("inverted_list.zig");

pub const StorageError = error{
    InvalidIndex,
    UnsupportedVersion,
    IndexCorrupt,
    BufferTooSmall,
} || Allocator.Error;

/// Header struct laid out to match the on-disk format. `extern` ensures
/// field order and no implicit padding beyond what we declare; the explicit
/// `_pad0` field pins the on-the-wire layout byte-for-byte.
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
    doc_tok_off: u64,
    norms_off: u64,
    footer_off: u64,
};

/// On-disk header region size — Header struct is 104 bytes; we pad to 128
/// so the first content section is 16-byte aligned (cache-line friendly,
/// SIMD friendly). Pinned in code so writer + parser agree.
pub const header_region_size: usize = 128;

comptime {
    std.debug.assert(@sizeOf(Header) == 112);
    std.debug.assert(header_region_size >= @sizeOf(Header));
    std.debug.assert(header_region_size % 8 == 0);
}

/// Parsed view over an in-memory image. Slices alias `bytes` — the caller
/// must keep `bytes` alive for the lifetime of the `Index`. No allocations
/// beyond what's needed to materialise PQ / HNSW / InvertedLists structs.
pub const Index = struct {
    header: Header,
    centroids: []const f32,
    pq: pq_mod.PQ,
    hnsw: hnsw_mod.Hnsw,
    ilists: inverted_list.InvertedLists,
    /// `n_docs+1` byte offsets into `doc_payload` (per-doc start).
    doc_index: []const u64,
    doc_payload: []const u8,
    /// `n_docs+1` global token offsets — `[doc_token_offsets[d], doc_token_offsets[d+1])`
    /// is the half-open token range of doc d, mirroring `TokenDump.doc_offsets`.
    /// Refine slices `residual_norms` against this in O(1) per candidate doc.
    doc_token_offsets: []const u64,
    residual_norms: []const f32,

    pub fn deinit(self: *Index, gpa: Allocator) void {
        // PQ.codebooks are heap-owned (memcpy'd from the byte image so the
        // mmap can be unmapped independently). HNSW likewise heap-owns its
        // CSR copy; inverted_list flips owns_buffers on/off explicitly.
        self.pq.deinit(gpa);
        self.hnsw.deinit(gpa);
        if (self.ilists.owns_buffers) self.ilists.deinit(gpa);
        self.* = undefined;
    }

    pub fn docLayout(self: *const Index, d: u32) []const u8 {
        const lo: usize = @intCast(self.doc_index[d]);
        const hi: usize = @intCast(self.doc_index[d + 1]);
        return self.doc_payload[lo..hi];
    }

    /// Half-open `[lo, hi)` global token range of doc `d`. Mirrors the
    /// `TokenDump.doc_offsets` semantics. Use to slice `residual_norms`
    /// (and any future per-token side table) for a candidate doc in O(1).
    pub fn docTokenRange(self: *const Index, d: u32) [2]u64 {
        return .{ self.doc_token_offsets[d], self.doc_token_offsets[d + 1] };
    }
};

/// All the pieces a builder hands to `serialise`. The builder (task #15)
/// assembles these from token_dump → tac → pq → hnsw → inverted_list, then
/// calls `serialise` to materialise a single byte image.
pub const BuildOutput = struct {
    dim: u32,
    kappa: u32,
    sub_dim: u32,
    n_docs: u64,
    n_tokens: u64,
    centroids: []const f32, // kappa * dim
    /// `[]u32` token-→-centroid assignments, length `n_tokens`. Used to emit
    /// the per-doc Pass 1 region. Same source slice that fed inverted_list.
    assignments: []const u32,
    /// `[]u8` flat PQ codes, length `n_tokens * PQ_M` (M codes per token).
    /// Used to emit the per-doc Pass 2 region.
    pq_codes: []const u8,
    /// Doc offsets (CSR): `doc_offsets[d+1] - doc_offsets[d]` = n_d.
    /// Persisted as the `doc_token_offsets` section so refine can slice
    /// `residual_norms` per candidate doc in O(1).
    doc_offsets: []const u64,
    pq: *const pq_mod.PQ,
    hnsw: *const hnsw_mod.Hnsw,
    ilists: *const inverted_list.InvertedLists,
    residual_norms: []const f32, // n_tokens floats, paper §4 side data
};

inline fn alignUp(x: u64, a: u64) u64 {
    return (x + a - 1) / a * a;
}

// ---------------------------------------------------------------------------
// END-TO-END BUILD (paper §3 + §4) — task #15.
// ---------------------------------------------------------------------------

const tac = @import("../tac/tac.zig");
const token_dump = @import("../io/token_dump.zig");
const vec_mod = @import("../util/vec.zig");

pub const BuildParams = struct {
    kappa_total: u32,
    seed: u64,
    /// HNSW construction params — paper-strict defaults.
    hnsw: hnsw_mod.BuildParams = .{},
    /// TAC defaults pulled from constants. Tests can override for tiny
    /// fixtures where 39 vectors/centroid is impossibly demanding.
    mu: u32 = constants.TAC_MU,
    tau: u32 = constants.TAC_TAU,
    epsilon: u32 = constants.TAC_EPSILON,
    theta: u32 = constants.TAC_THETA,
    /// Worker threads for the per-token Lloyd loop (paper §3 parallelism).
    /// `1` is serial; >1 dispatches via std.Thread.spawn with static chunks.
    n_threads: u32 = 1,
    /// When true, stage timings are printed to stderr.
    verbose: bool = false,
};

/// Monotonic ns timestamp (Zig 0.16: std.time.Timer is gone behind std.Io).
fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn msSince(start_ns: u64) f64 {
    return @as(f64, @floatFromInt(nowNs() - start_ns)) / 1e6;
}

/// Static-chunk parallel for-loop over [0, n_tokens). Each thread runs
/// `worker(ctx, lo, hi)` for a disjoint chunk. n_threads <= 1 falls
/// through to a serial call.
fn parallelTokenLoop(
    n_threads: u32,
    n_tokens: u64,
    comptime worker: anytype,
    ctx: anytype,
    gpa: Allocator,
) !void {
    if (n_threads <= 1 or n_tokens < 1024) {
        try worker(ctx, 0, n_tokens);
        return;
    }
    const ChunkErr = struct { e: ?@typeInfo(@typeInfo(@TypeOf(worker)).@"fn".return_type.?).error_union.error_set };
    const Run = struct {
        fn run(c: @TypeOf(ctx), lo: u64, hi: u64, out: *ChunkErr) void {
            worker(c, lo, hi) catch |err| {
                out.e = err;
            };
        }
    };
    const T = @min(n_threads, @as(u32, @intCast(n_tokens)));
    const errs = try gpa.alloc(ChunkErr, T);
    defer gpa.free(errs);
    @memset(errs, .{ .e = null });

    const threads = try gpa.alloc(std.Thread, T);
    defer gpa.free(threads);
    const chunk = (n_tokens + T - 1) / T;
    for (0..T) |t| {
        const lo: u64 = @as(u64, t) * chunk;
        const hi: u64 = @min(lo + chunk, n_tokens);
        threads[t] = try std.Thread.spawn(.{}, Run.run, .{ ctx, lo, hi, &errs[t] });
    }
    for (threads) |th| th.join();
    for (errs) |c| if (c.e) |err| return err;
}

const ResidualsCtx = struct {
    residuals: []f32,
    residual_norms: []f32,
    vectors: []const f32,
    centroids: []const f32,
    assignments: []const u32,
    dim: u32,
};

fn residualsAndNormsChunk(c: ResidualsCtx, lo: u64, hi: u64) !void {
    const dim = c.dim;
    var i = lo;
    while (i < hi) : (i += 1) {
        const cid: u32 = c.assignments[@intCast(i)];
        const v = c.vectors[@intCast(i * @as(u64, dim))..][0..dim];
        const cent = c.centroids[@as(usize, cid) * dim ..][0..dim];
        const r = c.residuals[@intCast(i * @as(u64, dim))..][0..dim];
        var sum_sq: f32 = 0.0;
        var dd: usize = 0;
        while (dd < dim) : (dd += 1) {
            const diff = v[dd] - cent[dd];
            r[dd] = diff;
            sum_sq += diff * diff;
        }
        const norm: f32 = @sqrt(sum_sq);
        c.residual_norms[@intCast(i)] = norm;
        if (norm > 0.0) {
            const inv: f32 = 1.0 / norm;
            dd = 0;
            while (dd < dim) : (dd += 1) r[dd] *= inv;
        }
    }
}

const PqEncodeCtx = struct {
    pq: *const pq_mod.PQ,
    residuals: []const f32,
    pq_codes: []u8,
    dim: u32,
};

fn pqEncodeChunk(c: PqEncodeCtx, lo: u64, hi: u64) !void {
    const dim = c.dim;
    var i = lo;
    while (i < hi) : (i += 1) {
        const r = c.residuals[@intCast(i * @as(u64, dim))..][0..dim];
        const slot = c.pq_codes[@intCast(i * @as(u64, constants.PQ_M))..][0..constants.PQ_M];
        try c.pq.encode(r, slot);
    }
}

/// In-memory bundle returned by `build`. The caller writes `bytes` to disk
/// (or hands it to `parse` for a same-process round-trip). All transient
/// build artefacts are released before return — only the byte image survives.
pub const BuiltImage = struct {
    bytes: []align(8) u8,

    pub fn deinit(self: *BuiltImage, gpa: Allocator) void {
        gpa.free(self.bytes);
        self.* = undefined;
    }
};

/// End-to-end Index.build (paper §3 + §4). Steps:
///   1. (paper §3) TAC clusters the corpus → centroids + assignments.
///   2. (paper §4) Residuals = vec − centroid; normalise; norms saved.
///   3. (paper §4) PQ.train on residuals → 32-byte codes per vector.
///   4. (paper §4) Encode every residual → flat code stream.
///   5. (paper §4) HNSW.build over centroids (centroids re-normalised so
///      inner product = cosine, per paper §11 convention).
///   6. (paper §4) Inverted lists from (assignments, doc_offsets).
///   7. Serialise: header + sections + footer crc32 → byte image.
///
/// paper-gap: TAC produces centroids that are means of unit-norm tokens but
/// are NOT themselves unit-norm. We L2-normalise centroids before HNSW so
/// dot product agrees with cosine at query time. The original (un-norm)
/// centroids are NOT preserved — Tachiom only ever consumes ⟨q, c⟩ at
/// query time, where re-norming would be redundant.
pub fn build(
    td: *const token_dump.TokenDump,
    params: BuildParams,
    gpa: Allocator,
) (StorageError || tac.ClusteringError || pq_mod.PqError || hnsw_mod.HnswError ||
    inverted_list.InvertedListError)!BuiltImage {
    if (td.dim == 0 or td.dim % constants.PQ_M != 0) return error.InvalidIndex;

    const dim: u32 = td.dim;
    const sub_dim: u32 = dim / constants.PQ_M;
    const n_tokens: u64 = td.n_tokens;
    const n_docs: u64 = td.n_docs;

    const verbose = params.verbose;

    // ---- 1. TAC clustering (paper §3). ----
    const t1_0 = nowNs();
    var clu = try tac.clusterFlat(td.token_ids, td.vectors, dim, .{
        .kappa_total = params.kappa_total,
        .mu = params.mu,
        .tau = params.tau,
        .epsilon = params.epsilon,
        .theta = params.theta,
        .seed = params.seed,
        .n_threads = params.n_threads,
    }, gpa);
    defer clu.deinit(gpa);
    if (verbose) std.debug.print("    [stage] tac.clusterFlat:  {d:8.1} ms\n", .{msSince(t1_0)});

    const kappa: u32 = params.kappa_total;

    // ---- 2. Residuals + per-vector norms (paper §4). ----
    const t2_0 = nowNs();
    const residuals = try gpa.alloc(f32, @intCast(n_tokens * @as(u64, dim)));
    defer gpa.free(residuals);
    const residual_norms = try gpa.alloc(f32, @intCast(n_tokens));
    errdefer gpa.free(residual_norms);

    try parallelTokenLoop(
        params.n_threads,
        n_tokens,
        residualsAndNormsChunk,
        ResidualsCtx{
            .residuals = residuals,
            .residual_norms = residual_norms,
            .vectors = td.vectors,
            .centroids = clu.centroids,
            .assignments = clu.assignments,
            .dim = dim,
        },
        gpa,
    );
    if (verbose) std.debug.print("    [stage] residuals + norms: {d:8.1} ms\n", .{msSince(t2_0)});

    // ---- 3. PQ training (paper §4). ----
    const t3_0 = nowNs();
    var pq = try pq_mod.train(residuals, dim, params.seed +% 0xa1b2c3d4, params.n_threads, gpa);
    errdefer pq.deinit(gpa);
    if (verbose) std.debug.print("    [stage] pq.train:          {d:8.1} ms\n", .{msSince(t3_0)});

    // ---- 4. Encode every residual into a flat M·n_tokens code stream. ----
    const t4_0 = nowNs();
    const total_codes: u64 = n_tokens * @as(u64, constants.PQ_M);
    const pq_codes = try gpa.alloc(u8, @intCast(total_codes));
    errdefer gpa.free(pq_codes);
    try parallelTokenLoop(
        params.n_threads,
        n_tokens,
        pqEncodeChunk,
        PqEncodeCtx{
            .pq = &pq,
            .residuals = residuals,
            .pq_codes = pq_codes,
            .dim = dim,
        },
        gpa,
    );
    if (verbose) std.debug.print("    [stage] pq.encode:         {d:8.1} ms\n", .{msSince(t4_0)});

    // ---- 5. HNSW over (re-normalised) centroids (paper §4 + §11). ----
    // paper-gap §3: TAC outputs raw means. We need unit-norm for HNSW dot ≡
    // cosine. Make a working copy so we don't disturb `clu.centroids` for
    // any downstream caller.
    const t5_0 = nowNs();
    const centroids_norm = try gpa.alloc(f32, kappa * dim);
    errdefer gpa.free(centroids_norm);
    @memcpy(centroids_norm, clu.centroids);
    try vec_mod.normalizeRowsInPlace(centroids_norm, dim);

    var hnsw_g = try hnsw_mod.build(
        centroids_norm,
        dim,
        params.seed +% 0xdeadbeef,
        params.hnsw,
        gpa,
    );
    errdefer hnsw_g.deinit(gpa);
    if (verbose) std.debug.print("    [stage] hnsw.build:        {d:8.1} ms\n", .{msSince(t5_0)});

    // ---- 6. Inverted lists (paper §4, doc-level grain). ----
    const t6_0 = nowNs();
    var ilists = try inverted_list.build(clu.assignments, td.doc_offsets, kappa, gpa);
    errdefer ilists.deinit(gpa);
    if (verbose) std.debug.print("    [stage] inverted_list:     {d:8.1} ms\n", .{msSince(t6_0)});

    // ---- 7. Serialise. ----
    const t7_0 = nowNs();
    const out = BuildOutput{
        .dim = dim,
        .kappa = kappa,
        .sub_dim = sub_dim,
        .n_docs = n_docs,
        .n_tokens = n_tokens,
        .centroids = centroids_norm,
        .assignments = clu.assignments,
        .pq_codes = pq_codes,
        .doc_offsets = td.doc_offsets,
        .pq = &pq,
        .hnsw = &hnsw_g,
        .ilists = &ilists,
        .residual_norms = residual_norms,
    };

    const total = computeSize(out);
    const buf = try gpa.alignedAlloc(u8, .@"8", @intCast(total));
    errdefer gpa.free(buf);

    _ = try serialise(out, buf);
    if (verbose) std.debug.print("    [stage] serialise:         {d:8.1} ms\n", .{msSince(t7_0)});

    // Drop transient build state — image is self-contained.
    ilists.deinit(gpa);
    hnsw_g.deinit(gpa);
    pq.deinit(gpa);
    gpa.free(centroids_norm);
    gpa.free(pq_codes);
    gpa.free(residual_norms);

    return BuiltImage{ .bytes = buf };
}

// ---------------------------------------------------------------------------
// LE writers / readers
// ---------------------------------------------------------------------------

inline fn writeU32Le(buf: []u8, off: usize, v: u32) void {
    std.mem.writeInt(u32, buf[off..][0..4], v, .little);
}
inline fn writeU64Le(buf: []u8, off: usize, v: u64) void {
    std.mem.writeInt(u64, buf[off..][0..8], v, .little);
}
inline fn readU32Le(buf: []const u8, off: usize) u32 {
    return std.mem.readInt(u32, buf[off..][0..4], .little);
}
inline fn readU64Le(buf: []const u8, off: usize) u64 {
    return std.mem.readInt(u64, buf[off..][0..8], .little);
}

/// Compute the total byte size needed to serialise `out`. Caller allocates
/// a buffer of this size before calling `serialise`. Sizes are computed
/// section by section, each padded up to 8-byte alignment.
pub fn computeSize(out: BuildOutput) u64 {
    var off: u64 = header_region_size;
    // centroids
    off += @as(u64, out.kappa) * @as(u64, out.dim) * @sizeOf(f32);
    off = alignUp(off, 8);
    // hnsw — see writeHnsw
    off += hnswSize(out.hnsw);
    off = alignUp(off, 8);
    // pq codebooks
    off += @as(u64, constants.PQ_M) * @as(u64, constants.PQ_CENTROIDS) *
        @as(u64, out.sub_dim) * @sizeOf(f32);
    off = alignUp(off, 8);
    // inverted lists
    off += (@as(u64, out.kappa) + 1) * @sizeOf(u64);
    off += out.ilists.offsets[out.ilists.kappa] * @sizeOf(u32);
    off = alignUp(off, 8);
    // per-doc layouts: doc_index [n_docs+1]u64 + payload
    off += (out.n_docs + 1) * @sizeOf(u64);
    off += docPayloadSize(out);
    off = alignUp(off, 8);
    // doc_token_offsets [n_docs+1]u64 (CSR token range per doc, paper §4 + §5.3)
    off += (out.n_docs + 1) * @sizeOf(u64);
    off = alignUp(off, 8);
    // residual norms
    off += out.n_tokens * @sizeOf(f32);
    off = alignUp(off, 8);
    // footer crc32 + 4 bytes pad
    off += 8;
    return off;
}

fn hnswSize(g: *const hnsw_mod.Hnsw) u64 {
    // header: max_level u8 + pad7 + entry_point u32 + pad4 + n u32 + pad4 = 24 bytes
    var bytes: u64 = 24;
    // node_levels [n]u8 padded to 8
    bytes += @as(u64, g.n);
    bytes = alignUp(bytes, 8);
    // per layer: offsets [n+1]u32 padded, payload [last]u32 padded
    var L: usize = 0;
    while (L < g.layer_offsets.len) : (L += 1) {
        bytes += @as(u64, g.n + 1) * @sizeOf(u32);
        bytes = alignUp(bytes, 8);
        const last_off: u32 = g.layer_offsets[L][g.n];
        bytes += @as(u64, last_off) * @sizeOf(u32);
        bytes = alignUp(bytes, 8);
    }
    return bytes;
}

fn docPayloadSize(out: BuildOutput) u64 {
    // for each doc: u32 n_d + u32[n_d] centroids + u8[n_d * PQ_M] codes,
    // each doc padded to 8.
    var total: u64 = 0;
    var d: u64 = 0;
    while (d < out.n_docs) : (d += 1) {
        const n_d: u64 = out.doc_offsets[d + 1] - out.doc_offsets[d];
        var bytes: u64 = @sizeOf(u32) + n_d * @sizeOf(u32) + n_d * @as(u64, constants.PQ_M);
        bytes = alignUp(bytes, 8);
        total += bytes;
    }
    return total;
}

// ---------------------------------------------------------------------------
// SERIALISE
// ---------------------------------------------------------------------------

/// Write the whole index image into `buf`. Caller-owned buffer; size must
/// equal `computeSize(out)` (see test for expected usage). Returns the
/// final image size for convenience.
pub fn serialise(out: BuildOutput, buf: []u8) StorageError!u64 {
    const total: u64 = computeSize(out);
    if (buf.len < total) return error.BufferTooSmall;

    // Reserve header region; we fill it last after offsets are known.
    @memset(buf[0..header_region_size], 0);
    var cur: u64 = header_region_size;

    // ---- Centroids ----
    const centroids_off = cur;
    {
        const n: u64 = @as(u64, out.kappa) * @as(u64, out.dim);
        const bytes = std.mem.sliceAsBytes(out.centroids[0..@intCast(n)]);
        std.mem.copyForwards(u8, buf[@intCast(cur)..][0..bytes.len], bytes);
        cur += @as(u64, bytes.len);
        cur = alignUp(cur, 8);
    }

    // ---- HNSW ----
    const hnsw_off = cur;
    cur += try writeHnsw(out.hnsw, buf[@intCast(cur)..]);
    cur = alignUp(cur, 8);

    // ---- PQ codebooks ----
    const pq_off = cur;
    {
        const bytes = std.mem.sliceAsBytes(out.pq.codebooks);
        std.mem.copyForwards(u8, buf[@intCast(cur)..][0..bytes.len], bytes);
        cur += @as(u64, bytes.len);
        cur = alignUp(cur, 8);
    }

    // ---- Inverted lists ----
    const ilist_off = cur;
    {
        const offs_bytes = std.mem.sliceAsBytes(out.ilists.offsets);
        std.mem.copyForwards(u8, buf[@intCast(cur)..][0..offs_bytes.len], offs_bytes);
        cur += @as(u64, offs_bytes.len);

        const payload_bytes = std.mem.sliceAsBytes(out.ilists.payload);
        std.mem.copyForwards(u8, buf[@intCast(cur)..][0..payload_bytes.len], payload_bytes);
        cur += @as(u64, payload_bytes.len);
        cur = alignUp(cur, 8);
    }

    // ---- Per-doc layouts ----
    const doc_off = cur;
    cur += try writeDocLayouts(out, buf[@intCast(cur)..]);
    cur = alignUp(cur, 8);

    // ---- Doc token offsets (paper §5.3 refine slicing) ----
    const doc_tok_off = cur;
    {
        const dto_bytes = std.mem.sliceAsBytes(out.doc_offsets);
        std.mem.copyForwards(u8, buf[@intCast(cur)..][0..dto_bytes.len], dto_bytes);
        cur += @as(u64, dto_bytes.len);
        cur = alignUp(cur, 8);
    }

    // ---- Residual norms ----
    const norms_off = cur;
    {
        const norms_bytes = std.mem.sliceAsBytes(out.residual_norms);
        std.mem.copyForwards(u8, buf[@intCast(cur)..][0..norms_bytes.len], norms_bytes);
        cur += @as(u64, norms_bytes.len);
        cur = alignUp(cur, 8);
    }

    const footer_off = cur;

    // ---- Header ----
    const hdr = Header{
        .magic = constants.INDEX_MAGIC,
        .version = constants.INDEX_VERSION,
        .dim = out.dim,
        .kappa = out.kappa,
        .pq_M = constants.PQ_M,
        .pq_bits = @intCast(constants.PQ_BITS),
        ._pad0 = .{ 0, 0, 0 },
        .sub_dim = out.sub_dim,
        .n_docs = out.n_docs,
        .n_tokens = out.n_tokens,
        .centroids_off = centroids_off,
        .hnsw_off = hnsw_off,
        .pq_off = pq_off,
        .ilist_off = ilist_off,
        .doc_off = doc_off,
        .doc_tok_off = doc_tok_off,
        .norms_off = norms_off,
        .footer_off = footer_off,
    };
    const hdr_bytes = std.mem.asBytes(&hdr);
    std.mem.copyForwards(u8, buf[0..hdr_bytes.len], hdr_bytes);
    // bytes [hdr_bytes.len .. header_region_size) are already zero.

    // ---- Footer crc32 ----
    var hasher = std.hash.Crc32.init();
    hasher.update(buf[0..@intCast(footer_off)]);
    const crc: u32 = hasher.final();
    writeU32Le(buf, @intCast(footer_off), crc);
    writeU32Le(buf, @intCast(footer_off + 4), 0);
    cur = footer_off + 8;

    std.debug.assert(cur == total);
    return total;
}

fn writeHnsw(g: *const hnsw_mod.Hnsw, buf: []u8) StorageError!u64 {
    if (buf.len < 24) return error.BufferTooSmall;
    buf[0] = g.max_level;
    @memset(buf[1..8], 0);
    writeU32Le(buf, 8, g.entry_point);
    writeU32Le(buf, 12, 0);
    writeU32Le(buf, 16, g.n);
    writeU32Le(buf, 20, 0);
    var cur: u64 = 24;

    // node_levels
    if (buf.len < cur + g.n) return error.BufferTooSmall;
    @memcpy(buf[@intCast(cur)..][0..g.n], g.node_levels);
    cur += @as(u64, g.n);
    const padded = alignUp(cur, 8);
    @memset(buf[@intCast(cur)..@intCast(padded)], 0);
    cur = padded;

    // per layer
    var L: usize = 0;
    while (L < g.layer_offsets.len) : (L += 1) {
        const offs = g.layer_offsets[L];
        const offs_bytes = std.mem.sliceAsBytes(offs);
        if (buf.len < cur + offs_bytes.len) return error.BufferTooSmall;
        @memcpy(buf[@intCast(cur)..][0..offs_bytes.len], offs_bytes);
        cur += @as(u64, offs_bytes.len);
        cur = alignUp(cur, 8);

        const pay = g.layer_neighbours[L];
        const pay_bytes = std.mem.sliceAsBytes(pay);
        if (buf.len < cur + pay_bytes.len) return error.BufferTooSmall;
        @memcpy(buf[@intCast(cur)..][0..pay_bytes.len], pay_bytes);
        cur += @as(u64, pay_bytes.len);
        cur = alignUp(cur, 8);
    }
    return cur;
}

fn writeDocLayouts(out: BuildOutput, buf: []u8) StorageError!u64 {
    const idx_bytes: u64 = (out.n_docs + 1) * @sizeOf(u64);
    if (buf.len < idx_bytes) return error.BufferTooSmall;
    var cur: u64 = idx_bytes;
    // First write doc_index (offsets into the payload region — relative to
    // the start of the payload, which is the byte right after the index).
    var d: u64 = 0;
    var payload_off: u64 = 0;
    while (d < out.n_docs) : (d += 1) {
        writeU64Le(buf, @intCast(d * 8), payload_off);
        const n_d: u64 = out.doc_offsets[d + 1] - out.doc_offsets[d];
        const tok_lo: u64 = out.doc_offsets[d];
        // Pass 1: centroid IDs
        const pass1_bytes: u64 = n_d * @sizeOf(u32);
        const pass2_bytes: u64 = n_d * @as(u64, constants.PQ_M);
        const this_doc_bytes_unpadded: u64 = @sizeOf(u32) + pass1_bytes + pass2_bytes;
        const this_doc_bytes: u64 = alignUp(this_doc_bytes_unpadded, 8);
        if (buf.len < cur + this_doc_bytes) return error.BufferTooSmall;

        // Write n_d (u32) at cur, then pass1, then pass2, then zero-pad.
        writeU32Le(buf, @intCast(cur), @intCast(n_d));
        var write_off: u64 = cur + 4;

        // Pass 1: centroid IDs from assignments[tok_lo .. tok_lo + n_d]
        const assigns_slice = out.assignments[@intCast(tok_lo) .. @intCast(tok_lo + n_d)];
        const a_bytes = std.mem.sliceAsBytes(assigns_slice);
        @memcpy(buf[@intCast(write_off)..][0..a_bytes.len], a_bytes);
        write_off += @as(u64, a_bytes.len);

        // Pass 2: PQ codes from pq_codes[tok_lo*M .. (tok_lo+n_d)*M]
        const codes_lo: u64 = tok_lo * @as(u64, constants.PQ_M);
        const codes_hi: u64 = (tok_lo + n_d) * @as(u64, constants.PQ_M);
        const codes_slice = out.pq_codes[@intCast(codes_lo)..@intCast(codes_hi)];
        @memcpy(buf[@intCast(write_off)..][0..codes_slice.len], codes_slice);
        write_off += @as(u64, codes_slice.len);

        // Zero-pad up to this_doc_bytes from cur.
        @memset(buf[@intCast(write_off)..@intCast(cur + this_doc_bytes)], 0);

        payload_off += this_doc_bytes;
        cur += this_doc_bytes;
    }
    writeU64Le(buf, @intCast(out.n_docs * 8), payload_off);
    return cur;
}

// ---------------------------------------------------------------------------
// PARSE
// ---------------------------------------------------------------------------

/// Parse a serialised index back into structured form. Allocates copies of
/// the PQ codebooks, HNSW CSR arrays, and inverted-list buffers because the
/// downstream types own their memory — that keeps mmap unmaps decoupled
/// from the lifetime of those structs. Doc-index, doc-payload, centroids,
/// and residual-norms slices alias `bytes` (no copy) since they're plain
/// arrays read at query time.
///
/// `bytes` must be 8-byte aligned at offset 0 — both `gpa.alloc(u8, ...)`
/// and any mmap region trivially are. With `header_region_size = 128` and
/// every section padded to 8B, the natural alignments line up.
pub fn parse(bytes: []align(8) const u8, gpa: Allocator) StorageError!Index {
    if (bytes.len < header_region_size + 8) return error.InvalidIndex;

    // Read header.
    var hdr: Header = undefined;
    @memcpy(std.mem.asBytes(&hdr), bytes[0..@sizeOf(Header)]);

    if (!std.mem.eql(u8, &hdr.magic, &constants.INDEX_MAGIC)) {
        return error.InvalidIndex;
    }
    if (hdr.version != constants.INDEX_VERSION) return error.UnsupportedVersion;
    if (hdr.pq_M != constants.PQ_M) return error.InvalidIndex;
    if (hdr.pq_bits != @as(u8, @intCast(constants.PQ_BITS))) return error.InvalidIndex;
    if (hdr.sub_dim * hdr.pq_M != hdr.dim) return error.InvalidIndex;
    if (hdr.dim == 0 or hdr.kappa == 0) return error.InvalidIndex;

    // Section offsets must be monotonic and within bounds.
    const offs = [_]u64{
        hdr.centroids_off, hdr.hnsw_off,    hdr.pq_off,
        hdr.ilist_off,     hdr.doc_off,     hdr.doc_tok_off,
        hdr.norms_off,     hdr.footer_off,
    };
    if (offs[0] != header_region_size) return error.InvalidIndex;
    var i: usize = 1;
    while (i < offs.len) : (i += 1) {
        if (offs[i] < offs[i - 1]) return error.InvalidIndex;
        if (offs[i] > bytes.len) return error.InvalidIndex;
    }
    if (hdr.footer_off + 8 > bytes.len) return error.InvalidIndex;

    // CRC check.
    const stored_crc: u32 = readU32Le(bytes, @intCast(hdr.footer_off));
    var hasher = std.hash.Crc32.init();
    hasher.update(bytes[0..@intCast(hdr.footer_off)]);
    if (hasher.final() != stored_crc) return error.IndexCorrupt;

    // ---- Centroids (alias) ----
    const centroids_bytes = bytes[@intCast(hdr.centroids_off)..@intCast(hdr.hnsw_off)];
    const centroids_n: usize = @intCast(@as(u64, hdr.kappa) * @as(u64, hdr.dim));
    if (centroids_bytes.len < centroids_n * @sizeOf(f32)) return error.InvalidIndex;
    // The buffer is 8-byte aligned (parameter constraint) and centroids_off
    // = 128, also 8-aligned, so the resulting f32 slice is 4-aligned. We
    // use @alignCast to communicate this to the type system.
    const centroids_aligned: []align(@alignOf(f32)) const u8 = @alignCast(centroids_bytes[0 .. centroids_n * @sizeOf(f32)]);
    const centroids_slice = std.mem.bytesAsSlice(f32, centroids_aligned);

    // ---- HNSW (heap-copy CSR, borrow centroids) ----
    var hnsw_view = try parseHnsw(bytes[@intCast(hdr.hnsw_off)..@intCast(hdr.pq_off)], hdr.dim, centroids_slice, gpa);
    errdefer hnsw_view.deinit(gpa);

    // ---- PQ (heap-copy codebooks) ----
    var pq_view = try parsePq(bytes[@intCast(hdr.pq_off)..@intCast(hdr.ilist_off)], hdr.dim, hdr.sub_dim, gpa);
    errdefer pq_view.deinit(gpa);

    // ---- Inverted lists (heap-copy so deinit is uniform) ----
    var ilists_view = try parseInvertedLists(
        bytes[@intCast(hdr.ilist_off)..@intCast(hdr.doc_off)],
        hdr.kappa,
        gpa,
    );
    errdefer if (ilists_view.owns_buffers) ilists_view.deinit(gpa);

    // ---- Per-doc index + payload (alias slices) ----
    const doc_section = bytes[@intCast(hdr.doc_off)..@intCast(hdr.doc_tok_off)];
    const doc_index_bytes_len: usize = @intCast((hdr.n_docs + 1) * @sizeOf(u64));
    if (doc_section.len < doc_index_bytes_len) return error.InvalidIndex;
    const doc_index_aligned: []align(@alignOf(u64)) const u8 = @alignCast(doc_section[0..doc_index_bytes_len]);
    const doc_index_slice = std.mem.bytesAsSlice(u64, doc_index_aligned);
    const doc_payload_slice = doc_section[doc_index_bytes_len..];

    // ---- Doc token offsets (alias) ----
    const doc_tok_bytes = bytes[@intCast(hdr.doc_tok_off)..@intCast(hdr.norms_off)];
    const doc_tok_bytes_len: usize = @intCast((hdr.n_docs + 1) * @sizeOf(u64));
    if (doc_tok_bytes.len < doc_tok_bytes_len) return error.InvalidIndex;
    const doc_tok_aligned: []align(@alignOf(u64)) const u8 = @alignCast(doc_tok_bytes[0..doc_tok_bytes_len]);
    const doc_token_offsets_slice = std.mem.bytesAsSlice(u64, doc_tok_aligned);

    // ---- Residual norms (alias) ----
    const norms_bytes = bytes[@intCast(hdr.norms_off)..@intCast(hdr.footer_off)];
    const norms_n: usize = @intCast(hdr.n_tokens);
    if (norms_bytes.len < norms_n * @sizeOf(f32)) return error.InvalidIndex;
    const norms_aligned: []align(@alignOf(f32)) const u8 = @alignCast(norms_bytes[0 .. norms_n * @sizeOf(f32)]);
    const norms_slice = std.mem.bytesAsSlice(f32, norms_aligned);

    return Index{
        .header = hdr,
        .centroids = centroids_slice,
        .pq = pq_view,
        .hnsw = hnsw_view,
        .ilists = ilists_view,
        .doc_index = doc_index_slice,
        .doc_payload = doc_payload_slice,
        .doc_token_offsets = doc_token_offsets_slice,
        .residual_norms = norms_slice,
    };
}

fn parseHnsw(
    section: []const u8,
    dim: u32,
    centroids: []const f32,
    gpa: Allocator,
) StorageError!hnsw_mod.Hnsw {
    if (section.len < 24) return error.InvalidIndex;
    const max_level: u8 = section[0];
    const entry_point: u32 = readU32Le(section, 8);
    const n: u32 = readU32Le(section, 16);
    var cur: u64 = 24;

    if (n == 0) return error.InvalidIndex;
    if (section.len < cur + n) return error.InvalidIndex;
    const node_levels = try gpa.alloc(u8, n);
    errdefer gpa.free(node_levels);
    @memcpy(node_levels, section[@intCast(cur)..][0..n]);
    cur += @as(u64, n);
    cur = alignUp(cur, 8);

    const total_layers: usize = @as(usize, max_level) + 1;
    const layer_offsets = try gpa.alloc([]u32, total_layers);
    errdefer gpa.free(layer_offsets);
    const layer_neighbours = try gpa.alloc([]u32, total_layers);
    errdefer gpa.free(layer_neighbours);
    var allocated: usize = 0;
    errdefer {
        var k: usize = 0;
        while (k < allocated) : (k += 1) {
            gpa.free(layer_offsets[k]);
            gpa.free(layer_neighbours[k]);
        }
    }

    var L: usize = 0;
    while (L < total_layers) : (L += 1) {
        const offs_bytes_len: usize = @intCast(@as(u64, n + 1) * @sizeOf(u32));
        if (section.len < cur + offs_bytes_len) return error.InvalidIndex;
        const offs = try gpa.alloc(u32, n + 1);
        errdefer gpa.free(offs);
        @memcpy(std.mem.sliceAsBytes(offs), section[@intCast(cur)..][0..offs_bytes_len]);
        cur += @as(u64, offs_bytes_len);
        cur = alignUp(cur, 8);

        const last_off: u32 = offs[n];
        const pay_bytes_len: usize = @intCast(@as(u64, last_off) * @sizeOf(u32));
        if (section.len < cur + pay_bytes_len) return error.InvalidIndex;
        const pay = try gpa.alloc(u32, last_off);
        errdefer gpa.free(pay);
        @memcpy(std.mem.sliceAsBytes(pay), section[@intCast(cur)..][0..pay_bytes_len]);
        cur += @as(u64, pay_bytes_len);
        cur = alignUp(cur, 8);

        layer_offsets[L] = offs;
        layer_neighbours[L] = pay;
        allocated = L + 1;
    }

    return hnsw_mod.Hnsw{
        .dim = dim,
        .n = n,
        .entry_point = entry_point,
        .max_level = max_level,
        .node_levels = node_levels,
        .layer_offsets = layer_offsets,
        .layer_neighbours = layer_neighbours,
        .centroids = centroids,
    };
}

fn parsePq(
    section: []const u8,
    dim: u32,
    sub_dim: u32,
    gpa: Allocator,
) StorageError!pq_mod.PQ {
    const n_floats: usize = @as(usize, constants.PQ_M) *
        @as(usize, constants.PQ_CENTROIDS) *
        @as(usize, sub_dim);
    const need: usize = n_floats * @sizeOf(f32);
    if (section.len < need) return error.InvalidIndex;
    const codebooks = try gpa.alloc(f32, n_floats);
    errdefer gpa.free(codebooks);
    @memcpy(std.mem.sliceAsBytes(codebooks), section[0..need]);
    return pq_mod.PQ{
        .dim = dim,
        .sub_dim = sub_dim,
        .codebooks = codebooks,
    };
}

fn parseInvertedLists(
    section: []const u8,
    kappa: u32,
    gpa: Allocator,
) StorageError!inverted_list.InvertedLists {
    const offs_bytes_len: usize = @intCast(@as(u64, kappa + 1) * @sizeOf(u64));
    if (section.len < offs_bytes_len) return error.InvalidIndex;
    const offsets = try gpa.alloc(u64, kappa + 1);
    errdefer gpa.free(offsets);
    @memcpy(std.mem.sliceAsBytes(offsets), section[0..offs_bytes_len]);
    const total: u64 = offsets[kappa];
    const pay_bytes_len: usize = @intCast(total * @sizeOf(u32));
    if (section.len < offs_bytes_len + pay_bytes_len) return error.InvalidIndex;
    const payload = try gpa.alloc(u32, @intCast(total));
    errdefer gpa.free(payload);
    @memcpy(
        std.mem.sliceAsBytes(payload),
        section[offs_bytes_len .. offs_bytes_len + pay_bytes_len],
    );
    return inverted_list.InvertedLists{
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
const rng_mod = @import("../util/rng.zig");
const vec = @import("../util/vec.zig");

test "Header layout sanity" {
    // v2 adds doc_tok_off u64 between doc_off and norms_off → +8 vs v1's 104.
    try testing.expectEqual(@as(usize, 112), @sizeOf(Header));
}

test "magic + version constants match paper repo conventions" {
    try testing.expectEqualSlices(u8, "TAC_IDX1", &constants.INDEX_MAGIC);
    try testing.expectEqual(@as(u32, 2), constants.INDEX_VERSION);
}

/// Build a tiny fixture (paper-strict M=32 forces dim multiples of 32).
/// Returns owned slices the test must free, plus an Index-friendly bundle.
const Fixture = struct {
    centroids: []f32,
    assignments: []u32,
    pq_codes: []u8,
    doc_offsets: []u64,
    residual_norms: []f32,
    pq: pq_mod.PQ,
    hnsw: hnsw_mod.Hnsw,
    ilists: inverted_list.InvertedLists,

    fn init(gpa: Allocator) !Fixture {
        const dim: u32 = 32; // sub_dim = 1
        const kappa: u32 = 8;
        const n_docs: u64 = 4;
        // Each doc gets 3 tokens → n_tokens = 12.
        const n_tokens: u64 = 12;

        // ---- Centroids: 8 random unit vectors, dim=32. ----
        var rng = rng_mod.Rng.init(2026);
        const centroids = try gpa.alloc(f32, kappa * dim);
        errdefer gpa.free(centroids);
        for (centroids) |*x| x.* = (rng.nextFloat() * 2.0) - 1.0;
        var c_idx: usize = 0;
        while (c_idx < kappa) : (c_idx += 1) {
            try vec.normalizeInPlace(centroids[c_idx * dim ..][0..dim]);
        }

        // ---- Assignments: round-robin over centroids. ----
        const assignments = try gpa.alloc(u32, @intCast(n_tokens));
        errdefer gpa.free(assignments);
        for (assignments, 0..) |*a, idx| a.* = @intCast(idx % kappa);

        // ---- doc_offsets: 4 docs of 3 tokens. ----
        const doc_offsets = try gpa.alloc(u64, @intCast(n_docs + 1));
        errdefer gpa.free(doc_offsets);
        var d: usize = 0;
        while (d <= n_docs) : (d += 1) doc_offsets[d] = @as(u64, d * 3);

        // ---- Train PQ on synthetic residuals. ----
        const residuals = try gpa.alloc(f32, @intCast(n_tokens * dim));
        defer gpa.free(residuals);
        for (residuals) |*x| x.* = (rng.nextFloat() * 0.2) - 0.1;
        var t: usize = 0;
        while (t < n_tokens) : (t += 1) {
            try vec.normalizeInPlace(residuals[t * dim ..][0..dim]);
        }
        var pq = try pq_mod.train(residuals, dim, 7, 1, gpa);
        errdefer pq.deinit(gpa);

        // ---- Encode each token to PQ codes. ----
        const pq_codes = try gpa.alloc(u8, @intCast(n_tokens * @as(u64, constants.PQ_M)));
        errdefer gpa.free(pq_codes);
        var ti: usize = 0;
        while (ti < n_tokens) : (ti += 1) {
            const slot = pq_codes[ti * constants.PQ_M ..][0..constants.PQ_M];
            try pq.encode(residuals[ti * dim ..][0..dim], slot);
        }

        // ---- Residual norms: arbitrary positive values. ----
        const residual_norms = try gpa.alloc(f32, @intCast(n_tokens));
        errdefer gpa.free(residual_norms);
        for (residual_norms, 0..) |*v, idx| v.* = 1.0 + @as(f32, @floatFromInt(idx)) * 0.1;

        // ---- HNSW. ----
        var hnsw_g = try hnsw_mod.build(centroids, dim, 11, .{
            .ef_construction = 32,
            .m = 4,
        }, gpa);
        errdefer hnsw_g.deinit(gpa);

        // ---- Inverted lists. ----
        var ilists = try inverted_list.build(assignments, doc_offsets, kappa, gpa);
        errdefer ilists.deinit(gpa);

        return Fixture{
            .centroids = centroids,
            .assignments = assignments,
            .pq_codes = pq_codes,
            .doc_offsets = doc_offsets,
            .residual_norms = residual_norms,
            .pq = pq,
            .hnsw = hnsw_g,
            .ilists = ilists,
        };
    }

    fn deinit(self: *Fixture, gpa: Allocator) void {
        self.ilists.deinit(gpa);
        self.hnsw.deinit(gpa);
        self.pq.deinit(gpa);
        gpa.free(self.residual_norms);
        gpa.free(self.pq_codes);
        gpa.free(self.assignments);
        gpa.free(self.doc_offsets);
        gpa.free(self.centroids);
    }

    fn buildOutput(self: *const Fixture) BuildOutput {
        return .{
            .dim = 32,
            .kappa = 8,
            .sub_dim = 1,
            .n_docs = 4,
            .n_tokens = 12,
            .centroids = self.centroids,
            .assignments = self.assignments,
            .pq_codes = self.pq_codes,
            .doc_offsets = self.doc_offsets,
            .pq = &self.pq,
            .hnsw = &self.hnsw,
            .ilists = &self.ilists,
            .residual_norms = self.residual_norms,
        };
    }
};

test "round-trip: serialise then parse recovers every section" {
    const a = std.testing.allocator;
    var fx = try Fixture.init(a);
    defer fx.deinit(a);

    const out = fx.buildOutput();
    const total = computeSize(out);
    const buf = try a.alignedAlloc(u8, .@"8", @intCast(total));
    defer a.free(buf);

    const written = try serialise(out, buf);
    try testing.expectEqual(total, written);

    var idx = try parse(buf, a);
    defer idx.deinit(a);

    try testing.expectEqual(out.dim, idx.header.dim);
    try testing.expectEqual(out.kappa, idx.header.kappa);
    try testing.expectEqual(out.n_docs, idx.header.n_docs);
    try testing.expectEqual(out.n_tokens, idx.header.n_tokens);
    try testing.expectEqualSlices(u8, &constants.INDEX_MAGIC, &idx.header.magic);

    // Centroids alias bytes; equal to original.
    try testing.expectEqualSlices(f32, fx.centroids, idx.centroids);

    // PQ codebooks copied → equal.
    try testing.expectEqualSlices(f32, fx.pq.codebooks, idx.pq.codebooks);

    // HNSW round-trip — entry_point, max_level, node_levels, every layer.
    try testing.expectEqual(fx.hnsw.entry_point, idx.hnsw.entry_point);
    try testing.expectEqual(fx.hnsw.max_level, idx.hnsw.max_level);
    try testing.expectEqual(fx.hnsw.n, idx.hnsw.n);
    try testing.expectEqualSlices(u8, fx.hnsw.node_levels, idx.hnsw.node_levels);
    try testing.expectEqual(fx.hnsw.layer_offsets.len, idx.hnsw.layer_offsets.len);
    var L: usize = 0;
    while (L < fx.hnsw.layer_offsets.len) : (L += 1) {
        try testing.expectEqualSlices(u32, fx.hnsw.layer_offsets[L], idx.hnsw.layer_offsets[L]);
        try testing.expectEqualSlices(u32, fx.hnsw.layer_neighbours[L], idx.hnsw.layer_neighbours[L]);
    }

    // Inverted lists.
    try testing.expectEqual(fx.ilists.kappa, idx.ilists.kappa);
    try testing.expectEqualSlices(u64, fx.ilists.offsets, idx.ilists.offsets);
    try testing.expectEqualSlices(u32, fx.ilists.payload, idx.ilists.payload);

    // Residual norms.
    try testing.expectEqualSlices(f32, fx.residual_norms, idx.residual_norms);

    // Doc token offsets round-trip and match TokenDump.doc_offsets shape.
    try testing.expectEqualSlices(u64, fx.doc_offsets, idx.doc_token_offsets);

    // Per-doc layout: each doc decodes to (n_d, centroid_ids, pq_codes).
    var d: u32 = 0;
    while (d < out.n_docs) : (d += 1) {
        const slice = idx.docLayout(d);
        const n_d_back: u32 = readU32Le(slice, 0);
        const tok_lo: u64 = fx.doc_offsets[d];
        const tok_hi: u64 = fx.doc_offsets[d + 1];
        try testing.expectEqual(@as(u32, @intCast(tok_hi - tok_lo)), n_d_back);
        // docTokenRange agrees with the source TokenDump CSR.
        const range = idx.docTokenRange(d);
        try testing.expectEqual(tok_lo, range[0]);
        try testing.expectEqual(tok_hi, range[1]);
        // Pass 1 region matches assignments slice byte-for-byte.
        const pass1 = slice[4 .. 4 + n_d_back * 4];
        const expected_pass1 = std.mem.sliceAsBytes(
            fx.assignments[@intCast(tok_lo)..@intCast(tok_hi)],
        );
        try testing.expectEqualSlices(u8, expected_pass1, pass1);
        // Pass 2 region matches pq_codes slice byte-for-byte.
        const pass2_lo: usize = 4 + @as(usize, n_d_back) * 4;
        const pass2_hi: usize = pass2_lo + @as(usize, n_d_back) * constants.PQ_M;
        const pass2 = slice[pass2_lo..pass2_hi];
        const codes_lo: u64 = tok_lo * constants.PQ_M;
        const codes_hi: u64 = tok_hi * constants.PQ_M;
        const expected_pass2 = fx.pq_codes[@intCast(codes_lo)..@intCast(codes_hi)];
        try testing.expectEqualSlices(u8, expected_pass2, pass2);
    }
}

test "parse: tampered magic → error.InvalidIndex" {
    const a = std.testing.allocator;
    var fx = try Fixture.init(a);
    defer fx.deinit(a);
    const out = fx.buildOutput();
    const total = computeSize(out);
    const buf = try a.alignedAlloc(u8, .@"8", @intCast(total));
    defer a.free(buf);
    _ = try serialise(out, buf);
    buf[0] = 'X';
    try testing.expectError(error.InvalidIndex, parse(buf, a));
}

test "parse: tampered footer crc → error.IndexCorrupt" {
    const a = std.testing.allocator;
    var fx = try Fixture.init(a);
    defer fx.deinit(a);
    const out = fx.buildOutput();
    const total = computeSize(out);
    const buf = try a.alignedAlloc(u8, .@"8", @intCast(total));
    defer a.free(buf);
    _ = try serialise(out, buf);
    // Flip a bit somewhere inside the centroids region — CRC will diverge.
    buf[header_region_size + 0] ^= 0x80;
    try testing.expectError(error.IndexCorrupt, parse(buf, a));
}

test "parse: too-small buffer returns InvalidIndex" {
    const a = std.testing.allocator;
    const tiny = try a.alignedAlloc(u8, .@"8", 16);
    defer a.free(tiny);
    @memset(tiny, 0);
    try testing.expectError(error.InvalidIndex, parse(tiny, a));
}

const synthetic_fixture = @import("../io/synthetic_fixture.zig");

test "Index.build: end-to-end on synthetic fixture, parse round-trips" {
    const a = std.testing.allocator;
    // dim=32 (PQ_M=32 → sub_dim=1), 50-doc fixture.
    var fx = try synthetic_fixture.build(a, .{
        .seed = 20260501,
        .n_docs = 50,
        .dim = 32,
        .vocab_size = 16,
        .avg_doc_len = 8,
    });
    defer fx.deinit(a);

    const td = token_dump.TokenDump{
        .dim = fx.dim,
        .n_docs = fx.n_docs,
        .n_tokens = fx.n_tokens,
        .doc_offsets = fx.doc_offsets,
        .token_ids = fx.token_ids,
        .vectors = fx.vectors,
    };

    // Tiny-fixture-sized TAC params: μ=4, τ=8, ε=1, θ=1 so the 16-vocab
    // synthetic distribution can satisfy `n_j/κ_j ≥ θ` at every token.
    // paper-gap: paper-strict (μ=128, τ=256, ε=4, θ=39) is impossible at
    // n_tokens ≈ 400. Annotated as a fixture override only.
    var img = try build(&td, .{
        .kappa_total = 32,
        .seed = 7,
        .hnsw = .{ .ef_construction = 32, .m = 4 },
        .mu = 4,
        .tau = 8,
        .epsilon = 1,
        .theta = 1,
    }, a);
    defer img.deinit(a);

    var idx = try parse(img.bytes, a);
    defer idx.deinit(a);

    try testing.expectEqual(@as(u32, 32), idx.header.dim);
    try testing.expectEqual(@as(u32, 32), idx.header.kappa);
    try testing.expectEqual(fx.n_docs, idx.header.n_docs);
    try testing.expectEqual(fx.n_tokens, idx.header.n_tokens);
    try testing.expectEqual(@as(u32, 32), idx.header.pq_M);
    try testing.expectEqual(@as(u32, 1), idx.header.sub_dim);

    // Centroids stored should be unit-normalised (we re-normalise pre-HNSW).
    var c: u32 = 0;
    while (c < idx.header.kappa) : (c += 1) {
        const cv = idx.centroids[c * idx.header.dim ..][0..idx.header.dim];
        var norm_sq: f32 = 0.0;
        for (cv) |x| norm_sq += x * x;
        try testing.expectApproxEqAbs(@as(f32, 1.0), norm_sq, 1e-4);
    }

    // HNSW must include every centroid as a node.
    try testing.expectEqual(@as(u32, 32), idx.hnsw.n);

    // Inverted lists union should cover every doc (every doc has tokens
    // mapping somewhere). Easy check: |union L_j| == n_docs.
    var seen_doc = try a.alloc(bool, @intCast(fx.n_docs));
    defer a.free(seen_doc);
    @memset(seen_doc, false);
    var j: u32 = 0;
    while (j < idx.ilists.kappa) : (j += 1) {
        for (idx.ilists.list(j)) |did| seen_doc[did] = true;
    }
    for (seen_doc) |s| try testing.expect(s);

    // Per-doc layout: total token count across all doc layouts equals n_tokens.
    var total_n: u64 = 0;
    var d: u32 = 0;
    while (d < idx.header.n_docs) : (d += 1) {
        const slice = idx.docLayout(d);
        total_n += @as(u64, readU32Le(slice, 0));
    }
    try testing.expectEqual(idx.header.n_tokens, total_n);

    // PQ codebooks were trained — codebook count is correct.
    try testing.expectEqual(
        @as(usize, constants.PQ_M) * @as(usize, constants.PQ_CENTROIDS) * 1,
        idx.pq.codebooks.len,
    );

    // Residual norms vector has one entry per token.
    try testing.expectEqual(@as(usize, @intCast(fx.n_tokens)), idx.residual_norms.len);
}
