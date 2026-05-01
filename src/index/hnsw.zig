//! src/index/hnsw.zig — HNSW over centroids (paper §4).
//!
//! Owner: indexer. Plan: 03-index-pq-hnsw-storage.plan.md.
//!
//! Paper-strict defaults: `M_hnsw=32`, `efc=1500`, runtime `ef_s = 1.5·κ_c`
//! (computed by `constants.hnswEfSearch`).
//!
//! Reference algorithm: Malkov & Yashunin 2020, "Efficient and robust
//! approximate nearest neighbor search using Hierarchical Navigable Small
//! World graphs" (TPAMI). We follow Algorithms 1–4 with the paper's own
//! parameter names — `M`, `M_max`, `M_max0`, `efConstruction`, `mL`.
//!
//! Distance metric is **similarity** (inner product). Tachiom centroids are
//! L2-normalised, so dot product is cosine, and "closer" means "higher sim".
//! Internal heaps:
//!   - `cands`:   max-heap on similarity → best-first frontier expansion
//!   - `result`:  min-heap on similarity → drop the worst when |result| > ef
//!
//! paper-gap: Tachiom §4 doesn't pin which Malkov variant is used for
//! neighbour selection. We use the heuristic (Algorithm 4) — standard choice
//! for cluster-rich graphs over centroid sets.

const std = @import("std");
const Allocator = std.mem.Allocator;
const constants = @import("../constants.zig");
const vec = @import("../util/vec.zig");
const rng_mod = @import("../util/rng.zig");

pub const HnswError = error{
    EmptyCorpus,
    DimMismatch,
    OutBufferTooSmall,
    OutOfMemory,
} || vec.VecError;

const NodeId = u32;

/// (similarity, node) pair used in both heaps. Similarity-ordering helpers
/// live below — `lessSim` for a min-heap (worst-first), `greaterSim` for a
/// max-heap (best-first).
const Cand = struct { sim: f32, node: NodeId };

fn lessSim(_: void, a: Cand, b: Cand) std.math.Order {
    return std.math.order(a.sim, b.sim);
}
fn greaterSim(_: void, a: Cand, b: Cand) std.math.Order {
    return std.math.order(b.sim, a.sim);
}

const MinHeap = std.PriorityQueue(Cand, void, lessSim);
const MaxHeap = std.PriorityQueue(Cand, void, greaterSim);

/// Frozen, query-ready HNSW. Adjacency is per-layer CSR for cache locality
/// and zero-copy mmap mapping later (storage.zig wires that up in #14).
///
/// Lifetime: `centroids` is BORROWED — caller (or storage layer) owns the
/// underlying buffer. The graph buffers (`node_levels`, `layer_offsets`,
/// `layer_neighbours`) are heap-owned by this Hnsw and freed in `deinit`.
pub const Hnsw = struct {
    dim: u32,
    n: u32,
    entry_point: NodeId,
    max_level: u8,
    /// Top level each node lives in (0 ≤ node_levels[i] ≤ max_level).
    node_levels: []u8,
    /// `layer_offsets[L]` has length `n + 1`. Neighbours of node `i` at
    /// layer `L` are `layer_neighbours[L][offsets[i] .. offsets[i+1]]`.
    /// A node not present at layer `L` (because its top level < L) has
    /// `offsets[i] == offsets[i+1]` (empty slice).
    layer_offsets: [][]u32,
    layer_neighbours: [][]u32,
    centroids: []const f32,

    pub fn deinit(self: *Hnsw, gpa: Allocator) void {
        gpa.free(self.node_levels);
        for (self.layer_offsets) |s| gpa.free(s);
        for (self.layer_neighbours) |s| gpa.free(s);
        gpa.free(self.layer_offsets);
        gpa.free(self.layer_neighbours);
        self.* = undefined;
    }

    inline fn centroid(self: *const Hnsw, id: NodeId) []const f32 {
        const start: usize = @as(usize, id) * @as(usize, self.dim);
        return self.centroids[start .. start + @as(usize, self.dim)];
    }

    inline fn neighbours(self: *const Hnsw, id: NodeId, layer: u8) []const u32 {
        const off = self.layer_offsets[layer];
        return self.layer_neighbours[layer][off[id] .. off[id + 1]];
    }
};

// ---------------------------------------------------------------------------
// BUILD
// ---------------------------------------------------------------------------

/// Per-node, per-layer growable adjacency. Used only during build; we freeze
/// to CSR inside `build` before returning.
const Adj = struct {
    /// Flat, indexed by `node_id * (max_level_so_far + 1) + L`. We grow the
    /// outer dimension as `max_level` grows — see `ensureLevel`.
    rows: std.ArrayList(std.ArrayList(NodeId)),
    n: u32,
    max_level: u8,
    gpa: Allocator,

    fn init(gpa: Allocator, n: u32) !Adj {
        var rows: std.ArrayList(std.ArrayList(NodeId)) = .empty;
        try rows.ensureTotalCapacity(gpa, @as(usize, n));
        var i: usize = 0;
        while (i < n) : (i += 1) {
            try rows.append(gpa, .empty);
        }
        return .{ .rows = rows, .n = n, .max_level = 0, .gpa = gpa };
    }

    fn deinit(self: *Adj) void {
        for (self.rows.items) |*r| r.deinit(self.gpa);
        self.rows.deinit(self.gpa);
    }

    /// Make sure `rows` has slots up to and including `level`. We store
    /// adjacency as `rows[node*levels_per_node + L]`, but levels grow rarely
    /// (geometric drop-off), so we just **reallocate row table** when the
    /// graph's max level grows. Simplicity > speed during build for now.
    fn ensureLevel(self: *Adj, level: u8) !void {
        if (level <= self.max_level) return;
        const new_per: usize = @as(usize, level) + 1;
        const old_per: usize = @as(usize, self.max_level) + 1;
        const new_total: usize = @as(usize, self.n) * new_per;
        const old_total: usize = self.rows.items.len;
        // Append empty ArrayLists, then permute so each node's old slots
        // remain contiguous in their new position. We do an in-place rebuild.
        try self.rows.ensureTotalCapacity(self.gpa, new_total);
        // Step 1: append fresh empty lists for the new tail size.
        while (self.rows.items.len < new_total) {
            try self.rows.append(self.gpa, .empty);
        }
        // Step 2: from the highest node down, move old rows into new layout.
        // Old layout: node n owns rows[n*old_per .. (n+1)*old_per].
        // New layout: node n owns rows[n*new_per .. (n+1)*new_per].
        var node: usize = self.n;
        while (node > 0) {
            node -= 1;
            const old_base = node * old_per;
            const new_base = node * new_per;
            // The new layers above old_per are already empty .empty entries
            // (appended above) — but they may currently sit at the end of
            // rows[]. We need to move our `old_per` legitimate rows into
            // [new_base .. new_base + old_per).
            if (new_base != old_base) {
                var l: usize = old_per;
                while (l > 0) {
                    l -= 1;
                    self.rows.items[new_base + l] = self.rows.items[old_base + l];
                    self.rows.items[old_base + l] = .empty;
                }
            }
            // Higher new levels [old_per .. new_per) for this node should be
            // empty. They currently are .empty (either freshly appended or
            // we just zeroed them above when shifting).
        }
        _ = old_total; // silence unused warning on debug builds
        self.max_level = level;
    }

    fn neighbours(self: *Adj, node: NodeId, level: u8) *std.ArrayList(NodeId) {
        const per: usize = @as(usize, self.max_level) + 1;
        return &self.rows.items[@as(usize, node) * per + @as(usize, level)];
    }
};

pub const BuildParams = struct {
    /// Construction-time efSearch — paper §4 uses 1500.
    ef_construction: u32 = constants.HNSW_EFC,
    /// Neighbours per node on layers ≥ 1 — paper §4 uses 32.
    m: u32 = constants.HNSW_M,
};

/// Build an HNSW over `n = centroids.len / dim` L2-normalised vectors.
///
/// Determinism: identical (centroids, dim, seed, params) → identical CSR.
pub fn build(
    centroids: []const f32,
    dim: u32,
    seed: u64,
    params: BuildParams,
    gpa: Allocator,
) HnswError!Hnsw {
    if (dim == 0) return error.DimMismatch;
    if (centroids.len % @as(usize, dim) != 0) return error.DimMismatch;
    const n_usize = centroids.len / @as(usize, dim);
    if (n_usize == 0) return error.EmptyCorpus;
    if (n_usize > std.math.maxInt(u32)) return error.OutOfMemory;
    const n: u32 = @intCast(n_usize);

    const m: u32 = params.m;
    const m_max: u32 = m;
    const m_max0: u32 = 2 * m;
    const ef_c: u32 = params.ef_construction;
    // mL = 1 / ln(M) — Malkov §4.1, Eq. 7.
    const m_l: f32 = 1.0 / @log(@as(f32, @floatFromInt(m)));

    var rng = rng_mod.Rng.init(seed);

    var levels = try gpa.alloc(u8, n);
    errdefer gpa.free(levels);
    @memset(levels, 0);

    var adj = try Adj.init(gpa, n);
    errdefer adj.deinit();

    // Bookkeeping for searchLayer / selectHeuristic. Reused across inserts.
    const visited_gen = try gpa.alloc(u32, n);
    defer gpa.free(visited_gen);
    @memset(visited_gen, 0);
    var generation: u32 = 0;

    var entry_point: NodeId = 0;
    var max_level: u8 = 0;
    levels[0] = pickLevel(&rng, m_l);
    max_level = levels[0];
    try adj.ensureLevel(max_level);

    var q: u32 = 1;
    while (q < n) : (q += 1) {
        const l_new = pickLevel(&rng, m_l);
        levels[q] = l_new;
        try adj.ensureLevel(@max(l_new, max_level));

        const q_vec = sliceVec(centroids, dim, q);

        // Phase A: greedy descent through layers above l_new.
        var cur: NodeId = entry_point;
        if (max_level > l_new) {
            var L: u8 = max_level;
            while (L > l_new) : (L -= 1) {
                cur = greedySearchLayer(centroids, dim, &adj, cur, q_vec, L);
                if (L == 0) break;
            }
        }

        // Phase B: at each layer ≤ min(max_level, l_new), do an ef-beam
        // search from `cur`, pick neighbours via the heuristic, connect.
        const top_layer: u8 = @min(max_level, l_new);
        var L_signed: i32 = top_layer;
        while (L_signed >= 0) : (L_signed -= 1) {
            const L: u8 = @intCast(L_signed);
            generation +%= 1;
            if (generation == 0) {
                @memset(visited_gen, 0);
                generation = 1;
            }

            var W = try searchLayer(
                centroids,
                dim,
                &adj,
                cur,
                q_vec,
                ef_c,
                L,
                visited_gen,
                generation,
                gpa,
            );
            defer W.deinit(gpa);

            // Drain W into a Cand buffer, sorted ascending — that's the
            // candidate set for selectHeuristic. Also seed `cur` for the
            // next layer down (best of W).
            var cands_buf = try gpa.alloc(Cand, W.count());
            defer gpa.free(cands_buf);
            var idx: usize = 0;
            while (W.pop()) |c| : (idx += 1) cands_buf[idx] = c;

            // `cur` for next layer = best (highest-sim) candidate.
            if (cands_buf.len > 0) {
                var best_idx: usize = 0;
                var best_sim: f32 = cands_buf[0].sim;
                var k: usize = 1;
                while (k < cands_buf.len) : (k += 1) {
                    if (cands_buf[k].sim > best_sim) {
                        best_sim = cands_buf[k].sim;
                        best_idx = k;
                    }
                }
                cur = cands_buf[best_idx].node;
            }

            const cap_q: u32 = if (L == 0) m_max0 else m_max;
            const chosen = try selectHeuristic(
                centroids,
                dim,
                q_vec,
                cands_buf,
                m,
                gpa,
            );
            defer gpa.free(chosen);

            // Wire q → chosen. Bidirectional with cap pruning on the other
            // end via the heuristic.
            const q_neighbours = adj.neighbours(q, L);
            try q_neighbours.ensureTotalCapacity(gpa, chosen.len);
            for (chosen) |nbr_id| q_neighbours.appendAssumeCapacity(nbr_id);

            for (chosen) |r_id| {
                const r_neighbours = adj.neighbours(r_id, L);
                try r_neighbours.append(gpa, q);
                if (r_neighbours.items.len > cap_q) {
                    try shrinkNeighbours(
                        centroids,
                        dim,
                        r_id,
                        r_neighbours,
                        cap_q,
                        gpa,
                    );
                }
            }
        }

        if (l_new > max_level) {
            max_level = l_new;
            entry_point = q;
        }
    }

    // Freeze adjacency to per-layer CSR.
    const total_layers: usize = @as(usize, max_level) + 1;
    var layer_offsets = try gpa.alloc([]u32, total_layers);
    errdefer gpa.free(layer_offsets);
    var layer_payload = try gpa.alloc([]u32, total_layers);
    errdefer gpa.free(layer_payload);
    var allocated: usize = 0;
    errdefer {
        var i: usize = 0;
        while (i < allocated) : (i += 1) {
            gpa.free(layer_offsets[i]);
            gpa.free(layer_payload[i]);
        }
    }

    var L: usize = 0;
    while (L < total_layers) : (L += 1) {
        var off = try gpa.alloc(u32, @as(usize, n) + 1);
        errdefer gpa.free(off);
        off[0] = 0;
        var i: usize = 0;
        while (i < n) : (i += 1) {
            const list = adj.neighbours(@intCast(i), @intCast(L));
            const len: u32 = @intCast(list.items.len);
            off[i + 1] = off[i] + len;
        }
        const total = off[n];
        var pay = try gpa.alloc(u32, total);
        errdefer gpa.free(pay);
        i = 0;
        while (i < n) : (i += 1) {
            const list = adj.neighbours(@intCast(i), @intCast(L));
            @memcpy(pay[off[i]..off[i + 1]], list.items);
        }
        layer_offsets[L] = off;
        layer_payload[L] = pay;
        allocated = L + 1;
    }

    adj.deinit();

    return Hnsw{
        .dim = dim,
        .n = n,
        .entry_point = entry_point,
        .max_level = max_level,
        .node_levels = levels,
        .layer_offsets = layer_offsets,
        .layer_neighbours = layer_payload,
        .centroids = centroids,
    };
}

/// Geometric layer assignment, Malkov §4.1 Eq. 7.
fn pickLevel(rng: *rng_mod.Rng, m_l: f32) u8 {
    // u ∈ (0, 1]. nextFloat returns [0,1) so flip 1 - u to dodge log(0).
    const u: f32 = 1.0 - rng.nextFloat();
    const lvl_f: f32 = @floor(-@log(u) * m_l);
    if (lvl_f < 0.0) return 0;
    if (lvl_f > 31.0) return 31; // u8 fits; in practice <10 for any realistic n
    return @intFromFloat(lvl_f);
}

inline fn sliceVec(buf: []const f32, dim: u32, id: NodeId) []const f32 {
    const start: usize = @as(usize, id) * @as(usize, dim);
    return buf[start .. start + @as(usize, dim)];
}

fn similarity(centroids: []const f32, dim: u32, a: []const f32, b_id: NodeId) f32 {
    const b = sliceVec(centroids, dim, b_id);
    return vec.dot(a, b) catch unreachable; // we validated dim at entry
}

/// Greedy 1-step descent within a single layer. Returns the local maximum
/// reachable from `entry`. Deterministic: ties resolve to lower node id
/// because we only switch when strictly better.
fn greedySearchLayer(
    centroids: []const f32,
    dim: u32,
    adj: *Adj,
    entry: NodeId,
    q_vec: []const f32,
    layer: u8,
) NodeId {
    var cur: NodeId = entry;
    var cur_sim: f32 = similarity(centroids, dim, q_vec, cur);
    while (true) {
        var improved = false;
        const list = adj.neighbours(cur, layer);
        for (list.items) |e| {
            const s = similarity(centroids, dim, q_vec, e);
            if (s > cur_sim) {
                cur_sim = s;
                cur = e;
                improved = true;
            }
        }
        if (!improved) break;
    }
    return cur;
}

/// Malkov Algorithm 2 — beam search at a single layer. Returns a max-heap
/// of size up to `ef`; caller must free.
fn searchLayer(
    centroids: []const f32,
    dim: u32,
    adj: *Adj,
    entry: NodeId,
    q_vec: []const f32,
    ef: u32,
    layer: u8,
    visited_gen: []u32,
    generation: u32,
    gpa: Allocator,
) HnswError!MaxHeap {
    var cands: MaxHeap = .empty;
    errdefer cands.deinit(gpa);
    var W: MinHeap = .empty;
    errdefer W.deinit(gpa);

    const entry_sim = similarity(centroids, dim, q_vec, entry);
    visited_gen[entry] = generation;
    try cands.push(gpa, .{ .sim = entry_sim, .node = entry });
    try W.push(gpa, .{ .sim = entry_sim, .node = entry });

    while (cands.pop()) |c| {
        // peek worst-of-W to prune
        const worst_w = W.peek() orelse break;
        if (c.sim < worst_w.sim and W.count() >= ef) {
            break;
        }
        const list = adj.neighbours(c.node, layer);
        for (list.items) |e| {
            if (visited_gen[e] == generation) continue;
            visited_gen[e] = generation;
            const s = similarity(centroids, dim, q_vec, e);
            const w_worst = W.peek();
            if (W.count() < ef or (w_worst != null and s > w_worst.?.sim)) {
                try cands.push(gpa, .{ .sim = s, .node = e });
                try W.push(gpa, .{ .sim = s, .node = e });
                if (W.count() > ef) _ = W.pop();
            }
        }
    }

    cands.deinit(gpa);

    // Convert W (min-heap) → max-heap on similarity for the caller. We do
    // this by draining and re-pushing — cheap (≤ ef items). W still holds
    // an allocated buffer after the drain so we must free it explicitly.
    var out: MaxHeap = .empty;
    errdefer out.deinit(gpa);
    while (W.pop()) |c| try out.push(gpa, c);
    W.deinit(gpa);
    return out;
}

/// Malkov Algorithm 4 — heuristic neighbour selection. Pick from `cands`
/// (already in any order) the M closest-to-q nodes that are *also* closer
/// to q than to any already-chosen neighbour. Returned slice is heap-owned.
fn selectHeuristic(
    centroids: []const f32,
    dim: u32,
    q_vec: []const f32,
    cands: []const Cand,
    m: u32,
    gpa: Allocator,
) HnswError![]u32 {
    _ = q_vec; // candidates already carry sim(q, cand); no extra recomputation needed
    // Sort cands descending by similarity. Avoid mutating caller's slice.
    const sorted = try gpa.dupe(Cand, cands);
    defer gpa.free(sorted);
    std.mem.sort(Cand, sorted, {}, candDescSim);

    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(gpa);
    try out.ensureTotalCapacity(gpa, m);

    for (sorted) |cand| {
        if (out.items.len >= m) break;
        // Keep cand iff it's closer to q than to every already-chosen r.
        var keep = true;
        const cand_vec = sliceVec(centroids, dim, cand.node);
        for (out.items) |r_id| {
            const r_vec = sliceVec(centroids, dim, r_id);
            const sim_cr = vec.dot(cand_vec, r_vec) catch unreachable;
            if (sim_cr > cand.sim) {
                keep = false;
                break;
            }
        }
        if (keep) try out.append(gpa, cand.node);
    }

    return out.toOwnedSlice(gpa);
}

fn candDescSim(_: void, a: Cand, b: Cand) bool {
    return a.sim > b.sim;
}

/// When a node's adjacency overflows `cap`, prune via the heuristic.
fn shrinkNeighbours(
    centroids: []const f32,
    dim: u32,
    node: NodeId,
    list: *std.ArrayList(NodeId),
    cap: u32,
    gpa: Allocator,
) HnswError!void {
    const node_vec = sliceVec(centroids, dim, node);
    var cands = try gpa.alloc(Cand, list.items.len);
    defer gpa.free(cands);
    for (list.items, 0..) |id, i| {
        const s = vec.dot(node_vec, sliceVec(centroids, dim, id)) catch unreachable;
        cands[i] = .{ .sim = s, .node = id };
    }
    const kept = try selectHeuristic(centroids, dim, node_vec, cands, cap, gpa);
    defer gpa.free(kept);
    list.clearRetainingCapacity();
    try list.appendSlice(gpa, kept);
}

// ---------------------------------------------------------------------------
// SEARCH
// ---------------------------------------------------------------------------

/// Search for the top-`k` most similar centroid IDs to `query`, with ef-beam
/// width `ef` at layer 0. Returns the count actually written (≤ `out.len`).
/// Output is sorted by descending similarity.
pub fn search(
    self: *const Hnsw,
    query: []const f32,
    k: u32,
    ef: u32,
    out: []NodeId,
    gpa: Allocator,
) HnswError!u32 {
    if (query.len != @as(usize, self.dim)) return error.DimMismatch;
    if (out.len < k) return error.OutBufferTooSmall;
    if (k == 0) return 0;

    const ef_eff: u32 = @max(ef, k);

    var cur: NodeId = self.entry_point;
    if (self.max_level > 0) {
        var L: u8 = self.max_level;
        while (L > 0) : (L -= 1) {
            cur = greedySearchLayerCsr(self, cur, query, L);
        }
    }

    const visited_gen = try gpa.alloc(u32, self.n);
    defer gpa.free(visited_gen);
    @memset(visited_gen, 0);

    var W = try searchLayerCsr(self, cur, query, ef_eff, 0, visited_gen, 1, gpa);
    defer W.deinit(gpa);

    // W is max-heap on sim; drain to get descending-sim order.
    const want: u32 = @intCast(@min(@as(usize, k), W.count()));
    var written: u32 = 0;
    while (written < want) : (written += 1) {
        const c = W.pop() orelse break;
        out[written] = c.node;
    }
    return written;
}

fn greedySearchLayerCsr(
    self: *const Hnsw,
    entry: NodeId,
    q_vec: []const f32,
    layer: u8,
) NodeId {
    var cur: NodeId = entry;
    var cur_sim: f32 = vec.dot(q_vec, self.centroid(cur)) catch unreachable;
    while (true) {
        var improved = false;
        for (self.neighbours(cur, layer)) |e| {
            const s = vec.dot(q_vec, self.centroid(e)) catch unreachable;
            if (s > cur_sim) {
                cur_sim = s;
                cur = e;
                improved = true;
            }
        }
        if (!improved) break;
    }
    return cur;
}

fn searchLayerCsr(
    self: *const Hnsw,
    entry: NodeId,
    q_vec: []const f32,
    ef: u32,
    layer: u8,
    visited_gen: []u32,
    generation: u32,
    gpa: Allocator,
) HnswError!MaxHeap {
    var cands: MaxHeap = .empty;
    errdefer cands.deinit(gpa);
    var W: MinHeap = .empty;
    errdefer W.deinit(gpa);

    const entry_sim = vec.dot(q_vec, self.centroid(entry)) catch unreachable;
    visited_gen[entry] = generation;
    try cands.push(gpa, .{ .sim = entry_sim, .node = entry });
    try W.push(gpa, .{ .sim = entry_sim, .node = entry });

    while (cands.pop()) |c| {
        const w_worst = W.peek() orelse break;
        if (c.sim < w_worst.sim and W.count() >= ef) break;

        for (self.neighbours(c.node, layer)) |e| {
            if (visited_gen[e] == generation) continue;
            visited_gen[e] = generation;
            const s = vec.dot(q_vec, self.centroid(e)) catch unreachable;
            const w_now = W.peek();
            if (W.count() < ef or (w_now != null and s > w_now.?.sim)) {
                try cands.push(gpa, .{ .sim = s, .node = e });
                try W.push(gpa, .{ .sim = s, .node = e });
                if (W.count() > ef) _ = W.pop();
            }
        }
    }
    cands.deinit(gpa);

    // Convert min-heap to max-heap on sim. W still owns its backing buffer
    // after the drain loop, so we free it explicitly.
    var out: MaxHeap = .empty;
    errdefer out.deinit(gpa);
    while (W.pop()) |c| try out.push(gpa, c);
    W.deinit(gpa);
    return out;
}

// ---------------------------------------------------------------------------
// TESTS
// ---------------------------------------------------------------------------

const testing = std.testing;

test "HNSW defaults match paper §4" {
    try testing.expectEqual(@as(u32, 32), constants.HNSW_M);
    try testing.expectEqual(@as(u32, 1500), constants.HNSW_EFC);
    try testing.expectEqual(@as(u32, 60), constants.hnswEfSearch(40));
}

test "build: smoke on 4 unit vectors, dim=2" {
    const a = std.testing.allocator;
    // Four unit vectors at 0°, 90°, 180°, 270°.
    const c = [_]f32{ 1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0 };
    var idx = try build(&c, 2, 7, .{ .ef_construction = 16, .m = 4 }, a);
    defer idx.deinit(a);
    try testing.expectEqual(@as(u32, 4), idx.n);
    try testing.expect(idx.entry_point < 4);
    try testing.expect(idx.max_level >= 0);
}

test "search: returns k descending-sim items on a small graph" {
    const a = std.testing.allocator;
    const c = [_]f32{
        1.0, 0.0,
        0.9, 0.4359, // sim with q below ~0.97
        0.0, 1.0,
        -1.0, 0.0,
        0.0, -1.0,
    };
    var idx = try build(&c, 2, 1, .{ .ef_construction = 32, .m = 4 }, a);
    defer idx.deinit(a);

    const q = [_]f32{ 1.0, 0.0 };
    var out: [3]u32 = undefined;
    const written = try search(&idx, &q, 3, 16, &out, a);
    try testing.expectEqual(@as(u32, 3), written);
    // Top-1 should be node 0 (perfect match).
    try testing.expectEqual(@as(u32, 0), out[0]);
    // Output is descending in similarity → enforce by recomputing.
    var prev: f32 = std.math.inf(f32);
    for (out[0..written]) |id| {
        const s = try vec.dot(&q, c[@as(usize, id) * 2 ..][0..2]);
        try testing.expect(s <= prev + 1e-6);
        prev = s;
    }
}

test "recall@10 ≥ 0.9 on 1024 random unit vectors, dim=64" {
    const a = std.testing.allocator;
    const n: u32 = 1024;
    const dim: u32 = 64;
    const buf = try a.alloc(f32, @as(usize, n) * @as(usize, dim));
    defer a.free(buf);

    var rng = rng_mod.Rng.init(20260501);
    // Generate random unit vectors.
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const row = buf[i * dim .. (i + 1) * dim];
        for (row) |*x| x.* = (rng.nextFloat() * 2.0) - 1.0;
        try vec.normalizeInPlace(row);
    }

    var idx = try build(buf, dim, 42, .{}, a);
    defer idx.deinit(a);

    // 16 random queries; recall@10 against brute-force argmax.
    const n_queries: u32 = 16;
    const k: u32 = 10;
    const ef: u32 = constants.hnswEfSearch(40); // generous
    var hit_count: u32 = 0;
    var total_recall: u32 = 0;

    const distances = try a.alloc(f32, n);
    defer a.free(distances);

    var q_idx: u32 = 0;
    while (q_idx < n_queries) : (q_idx += 1) {
        const query_buf = try a.alloc(f32, dim);
        defer a.free(query_buf);
        for (query_buf) |*x| x.* = (rng.nextFloat() * 2.0) - 1.0;
        try vec.normalizeInPlace(query_buf);

        // Brute force top-k by similarity.
        for (distances, 0..) |*d, j| {
            d.* = -try vec.dot(query_buf, buf[j * dim .. (j + 1) * dim]);
        }
        var bf_top = try a.alloc(u32, k);
        defer a.free(bf_top);
        var taken: [1024]bool = undefined;
        @memset(taken[0..n], false);
        var t: u32 = 0;
        while (t < k) : (t += 1) {
            var best: usize = std.math.maxInt(usize);
            var best_d: f32 = std.math.inf(f32);
            var jj: usize = 0;
            while (jj < n) : (jj += 1) {
                if (taken[jj]) continue;
                if (distances[jj] < best_d) {
                    best_d = distances[jj];
                    best = jj;
                }
            }
            taken[best] = true;
            bf_top[t] = @intCast(best);
        }

        var ann_top: [10]u32 = undefined;
        const got = try search(&idx, query_buf, k, ef, &ann_top, a);
        try testing.expectEqual(k, got);

        // Count overlap.
        var overlap: u32 = 0;
        for (ann_top[0..got]) |x| {
            for (bf_top) |y| if (x == y) {
                overlap += 1;
                break;
            };
        }
        total_recall += overlap;
        hit_count += k;
    }

    const recall: f64 = @as(f64, @floatFromInt(total_recall)) /
        @as(f64, @floatFromInt(hit_count));
    try testing.expect(recall >= 0.9);
}

test "build: same seed → same CSR" {
    const a = std.testing.allocator;
    const n: u32 = 128;
    const dim: u32 = 16;
    const buf = try a.alloc(f32, @as(usize, n) * @as(usize, dim));
    defer a.free(buf);

    var rng = rng_mod.Rng.init(2026);
    for (buf) |*x| x.* = (rng.nextFloat() * 2.0) - 1.0;
    var i: usize = 0;
    while (i < n) : (i += 1) try vec.normalizeInPlace(buf[i * dim .. (i + 1) * dim]);

    var a1 = try build(buf, dim, 999, .{ .ef_construction = 64, .m = 8 }, a);
    defer a1.deinit(a);
    var a2 = try build(buf, dim, 999, .{ .ef_construction = 64, .m = 8 }, a);
    defer a2.deinit(a);

    try testing.expectEqual(a1.entry_point, a2.entry_point);
    try testing.expectEqual(a1.max_level, a2.max_level);
    try testing.expectEqualSlices(u8, a1.node_levels, a2.node_levels);
    try testing.expectEqual(a1.layer_offsets.len, a2.layer_offsets.len);
    var L: usize = 0;
    while (L < a1.layer_offsets.len) : (L += 1) {
        try testing.expectEqualSlices(u32, a1.layer_offsets[L], a2.layer_offsets[L]);
        try testing.expectEqualSlices(u32, a1.layer_neighbours[L], a2.layer_neighbours[L]);
    }
}
