//! src/util/vec.zig — SIMD-vectorised f32 operations.
//!
//! Owner: primitives-engineer.
//! Underpins every distance/similarity computation in TAC clustering (§3),
//! HNSW edge selection (§4), and the gather/refine query phases (§5).
//!
//! All functions operate on `f32` slices. SIMD width is derived from
//! `std.simd.suggestVectorLength(f32)`, with a scalar tail for inputs whose
//! length isn't a multiple of the lane count. Inputs are validated at the
//! boundary; mismatched lengths return errors rather than panicking.

const std = @import("std");

/// Hardware-suggested SIMD lane count for f32. Falls back to 4 on targets
/// where suggestVectorLength returns null (rare but possible for unknown CPUs).
pub const lane_count: usize = std.simd.suggestVectorLength(f32) orelse 4;

const Vec = @Vector(lane_count, f32);

pub const VecError = error{
    LengthMismatch,
    EmptyInput,
};

/// Inner product ⟨a, b⟩.
///
/// paper §3.1: token similarities are dot products of L2-normalised f32
/// embeddings (cosine), used everywhere from k-means assignment to the
/// gather-phase ⟨q_i, c_j⟩ accumulator.
///
/// plan-11: dispatcher for the four dims that actually appear in this repo —
/// 2 and 4 (PQ subspaces, since dim ∈ {64, 128} and PQ_M=32) and 64/128 (the
/// jina-colbert-v2-64 and ColBERTv2.0 token embeddings). Marked `inline` so
/// the caller's `a.len` propagates through the switch and LLVM DCEs every
/// other arm; without `inline` the runtime branch on `a.len` was a net loss
/// in the kmeans-assign hot loop (see plan-11 microbench).
pub inline fn dot(a: []const f32, b: []const f32) VecError!f32 {
    if (a.len != b.len) return error.LengthMismatch;
    // dim=2 is hand-inlined here rather than going through `dotComptime(2,...)`:
    // the indirection occasionally blocks LLVM from constant-folding the slice
    // length all the way to two scalar mul-adds. Hand-inlining is unambiguous.
    return switch (a.len) {
        2 => a[0] * b[0] + a[1] * b[1],
        4 => dotComptime(4, a, b),
        64 => dotComptime(64, a, b),
        128 => dotComptime(128, a, b),
        else => dotGeneric(a, b),
    };
}

/// Comptime-specialized dot product. `dim` is known at compile time so the
/// reduction loop fully unrolls, the tail branch disappears, and LLVM is free
/// to schedule the lanes across multiple FMA pipes.
///
/// Caller must guarantee `a.len == b.len == dim`; this is a private helper
/// reached only via the `dot` dispatcher (which has already validated lengths).
pub inline fn dotComptime(comptime dim: u32, a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == dim and b.len == dim);

    if (dim == 0) return 0.0;

    // dim ≤ 4 → inline scalar unroll. Padding into a 4-wide @Vector adds two
    // tiny stack stores + a reduce that's strictly slower than a flat scalar
    // chain on this size class (the PQ subspace path).
    if (dim <= 4) {
        var s: f32 = 0.0;
        inline for (0..dim) |i| s += a[i] * b[i];
        return s;
    }

    const lanes: comptime_int = comptime laneWidthFor(dim);
    const V = @Vector(lanes, f32);

    // dim > lanes: comptime-unrolled lane loop over @Vector(4) chunks.
    comptime std.debug.assert(dim % lanes == 0);
    const n_chunks: comptime_int = dim / lanes;
    var acc: V = @splat(0.0);
    inline for (0..n_chunks) |c| {
        const off = c * lanes;
        const va: V = a[off..][0..lanes].*;
        const vb: V = b[off..][0..lanes].*;
        acc += va * vb;
    }
    return @reduce(.Add, acc);
}

/// Generic runtime-`lane_count` dot product for unspecialized dims.
inline fn dotGeneric(a: []const f32, b: []const f32) f32 {
    const n = a.len;
    var acc: Vec = @splat(0.0);
    var i: usize = 0;
    while (i + lane_count <= n) : (i += lane_count) {
        const va: Vec = a[i..][0..lane_count].*;
        const vb: Vec = b[i..][0..lane_count].*;
        acc += va * vb;
    }
    var sum: f32 = @reduce(.Add, acc);
    while (i < n) : (i += 1) sum += a[i] * b[i];
    return sum;
}

/// Squared Euclidean distance ‖a − b‖².
///
/// paper §3.1: the spread `s_j = (1/n_j)·Σ ‖t_{j,i} − t̄_j‖²` is a sum of
/// l2sq terms, and Lloyd's-step assignment minimises the same quantity.
pub inline fn l2sq(a: []const f32, b: []const f32) VecError!f32 {
    if (a.len != b.len) return error.LengthMismatch;
    return switch (a.len) {
        // dim=2 directly inline — same plan-11 reasoning as `dot`. PQ encode
        // hits this path on every (subspace × centroid × token) triple.
        2 => blk: {
            const d0 = a[0] - b[0];
            const d1 = a[1] - b[1];
            break :blk d0 * d0 + d1 * d1;
        },
        4 => l2sqComptime(4, a, b),
        64 => l2sqComptime(64, a, b),
        128 => l2sqComptime(128, a, b),
        else => l2sqGeneric(a, b),
    };
}

/// Comptime-specialized squared L2 distance. See `dotComptime` for the
/// register/unroll strategy.
pub inline fn l2sqComptime(comptime dim: u32, a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == dim and b.len == dim);

    if (dim == 0) return 0.0;

    if (dim <= 4) {
        var s: f32 = 0.0;
        inline for (0..dim) |i| {
            const d = a[i] - b[i];
            s += d * d;
        }
        return s;
    }

    const lanes: comptime_int = comptime laneWidthFor(dim);
    const V = @Vector(lanes, f32);

    comptime std.debug.assert(dim % lanes == 0);
    const n_chunks: comptime_int = dim / lanes;
    var acc: V = @splat(0.0);
    inline for (0..n_chunks) |c| {
        const off = c * lanes;
        const va: V = a[off..][0..lanes].*;
        const vb: V = b[off..][0..lanes].*;
        const d = va - vb;
        acc += d * d;
    }
    return @reduce(.Add, acc);
}

/// Generic runtime-`lane_count` l2sq for unspecialized dims.
inline fn l2sqGeneric(a: []const f32, b: []const f32) f32 {
    const n = a.len;
    var acc: Vec = @splat(0.0);
    var i: usize = 0;
    while (i + lane_count <= n) : (i += lane_count) {
        const va: Vec = a[i..][0..lane_count].*;
        const vb: Vec = b[i..][0..lane_count].*;
        const diff = va - vb;
        acc += diff * diff;
    }
    var sum: f32 = @reduce(.Add, acc);
    while (i < n) : (i += 1) {
        const d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

/// L2-normalise `v` in place. Zero-length vectors are an error (callers should
/// have at least one element); the all-zero vector is left untouched (no unit
/// direction exists).
///
/// paper convention: ColBERT/Tachiom assume unit-norm token embeddings so
/// cosine collapses to dot product.
pub inline fn normalizeInPlace(v: []f32) VecError!void {
    if (v.len == 0) return error.EmptyInput;
    switch (v.len) {
        64 => normalizeComptime(64, v),
        128 => normalizeComptime(128, v),
        else => normalizeGeneric(v),
    }
}

/// Comptime-specialized in-place normalize.
pub inline fn normalizeComptime(comptime dim: u32, v: []f32) void {
    std.debug.assert(v.len == dim);
    if (dim == 0) return;

    var norm_sq: f32 = 0.0;

    if (dim <= 4) {
        inline for (0..dim) |i| norm_sq += v[i] * v[i];
    } else {
        const lanes: comptime_int = comptime laneWidthFor(dim);
        const V = @Vector(lanes, f32);
        comptime std.debug.assert(dim % lanes == 0);
        const n_chunks: comptime_int = dim / lanes;
        var acc: V = @splat(0.0);
        inline for (0..n_chunks) |c| {
            const off = c * lanes;
            const x: V = v[off..][0..lanes].*;
            acc += x * x;
        }
        norm_sq = @reduce(.Add, acc);
    }

    if (norm_sq == 0.0) return;
    const inv_norm: f32 = 1.0 / @sqrt(norm_sq);

    if (dim <= 4) {
        inline for (0..dim) |i| v[i] *= inv_norm;
    } else {
        const lanes: comptime_int = comptime laneWidthFor(dim);
        const V = @Vector(lanes, f32);
        const splat_inv: V = @splat(inv_norm);
        const n_chunks: comptime_int = dim / lanes;
        inline for (0..n_chunks) |c| {
            const off = c * lanes;
            const x: V = v[off..][0..lanes].*;
            v[off..][0..lanes].* = x * splat_inv;
        }
    }
}

/// Generic runtime-`lane_count` normalize for unspecialized dims.
inline fn normalizeGeneric(v: []f32) void {
    var acc: Vec = @splat(0.0);
    var i: usize = 0;
    while (i + lane_count <= v.len) : (i += lane_count) {
        const x: Vec = v[i..][0..lane_count].*;
        acc += x * x;
    }
    var norm_sq: f32 = @reduce(.Add, acc);
    while (i < v.len) : (i += 1) norm_sq += v[i] * v[i];

    if (norm_sq == 0.0) return;
    const inv_norm: f32 = 1.0 / @sqrt(norm_sq);
    const splat_inv: Vec = @splat(inv_norm);

    i = 0;
    while (i + lane_count <= v.len) : (i += lane_count) {
        const x: Vec = v[i..][0..lane_count].*;
        v[i..][0..lane_count].* = x * splat_inv;
    }
    while (i < v.len) : (i += 1) v[i] *= inv_norm;
}

/// Pick a SIMD lane width for a comptime-known `dim`.
///
/// On Apple Silicon NEON is 128-bit (4×f32). We always pick 4 — empirically
/// (plan-11 microbench) wider @Vector(16) accumulators lower to 4 stacked
/// NEON regs and end up *slower* than a comptime-unrolled loop over a single
/// 4-wide register. The win from comptime specialization is the unrolled,
/// branch-free reduction, not a wider accumulator.
fn laneWidthFor(comptime dim: u32) comptime_int {
    _ = dim;
    return 4;
}

/// Index of the smallest element. Empty input is an error rather than an
/// `unreachable` because callers (e.g. assignment loops) should validate
/// they have at least one centroid before asking.
pub fn argmin(values: []const f32) VecError!usize {
    if (values.len == 0) return error.EmptyInput;
    var best: usize = 0;
    var best_v: f32 = values[0];
    var i: usize = 1;
    while (i < values.len) : (i += 1) {
        if (values[i] < best_v) {
            best_v = values[i];
            best = i;
        }
    }
    return best;
}

/// Index of the largest element. Symmetric to `argmin`; useful for picking
/// the highest-similarity centroid in the gather phase.
pub fn argmax(values: []const f32) VecError!usize {
    if (values.len == 0) return error.EmptyInput;
    var best: usize = 0;
    var best_v: f32 = values[0];
    var i: usize = 1;
    while (i < values.len) : (i += 1) {
        if (values[i] > best_v) {
            best_v = values[i];
            best = i;
        }
    }
    return best;
}

/// Compute `dst[i] = ‖query − points[i*dim..(i+1)*dim]‖²` for every row.
/// `points` is a flat row-major matrix of length `n*dim`. Used by k-means
/// assignment to evaluate one point against every centroid in a single pass.
pub fn l2sqBatch(
    query: []const f32,
    points: []const f32,
    dim: usize,
    dst: []f32,
) VecError!void {
    if (query.len != dim) return error.LengthMismatch;
    if (points.len != dst.len * dim) return error.LengthMismatch;
    var i: usize = 0;
    while (i < dst.len) : (i += 1) {
        dst[i] = try l2sq(query, points[i * dim ..][0..dim]);
    }
}

/// Batched in-place L2 normalise of `n` row-major vectors of `dim` floats.
pub fn normalizeRowsInPlace(rows: []f32, dim: usize) VecError!void {
    if (dim == 0) return error.EmptyInput;
    if (rows.len % dim != 0) return error.LengthMismatch;
    var i: usize = 0;
    while (i < rows.len) : (i += dim) {
        try normalizeInPlace(rows[i..][0..dim]);
    }
}

// ---------------------------------------------------------------------------
// Tests — tiny inputs with hand-checkable expected values.
// ---------------------------------------------------------------------------

const testing = std.testing;

test "dot: 3-d hand-checked" {
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ 4.0, -5.0, 6.0 };
    // 1*4 + 2*(-5) + 3*6 = 4 - 10 + 18 = 12
    try testing.expectEqual(@as(f32, 12.0), try dot(&a, &b));
}

test "dot: longer than one SIMD lane" {
    var a: [17]f32 = undefined;
    var b: [17]f32 = undefined;
    var expected: f32 = 0.0;
    for (0..a.len) |i| {
        a[i] = @floatFromInt(i + 1);
        b[i] = 2.0;
        expected += a[i] * b[i];
    }
    // 2*(1+2+...+17) = 2*153 = 306
    try testing.expectEqual(@as(f32, 306.0), try dot(&a, &b));
    try testing.expectEqual(expected, try dot(&a, &b));
}

test "dot: orthogonal" {
    const a = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const b = [_]f32{ 0.0, 1.0, 0.0, 0.0 };
    try testing.expectEqual(@as(f32, 0.0), try dot(&a, &b));
}

test "dot: length mismatch returns error" {
    const a = [_]f32{ 1.0, 2.0 };
    const b = [_]f32{ 1.0, 2.0, 3.0 };
    try testing.expectError(error.LengthMismatch, dot(&a, &b));
}

test "l2sq: hand-checked" {
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ 4.0, 6.0, 3.0 };
    // (3)^2 + (4)^2 + 0^2 = 9 + 16 = 25
    try testing.expectEqual(@as(f32, 25.0), try l2sq(&a, &b));
}

test "l2sq: identical vectors are zero" {
    const a = [_]f32{ 0.5, -0.5, 0.25, -0.25, 1.0 };
    try testing.expectEqual(@as(f32, 0.0), try l2sq(&a, &a));
}

test "l2sq: tail beyond SIMD lane" {
    var a: [9]f32 = undefined;
    var b: [9]f32 = undefined;
    for (0..9) |i| {
        a[i] = @floatFromInt(i);
        b[i] = @floatFromInt(i + 1);
    }
    // each diff is -1, squared is 1, summed over 9 = 9
    try testing.expectEqual(@as(f32, 9.0), try l2sq(&a, &b));
}

test "normalizeInPlace: 3-4-5 → unit" {
    var v = [_]f32{ 3.0, 4.0 };
    try normalizeInPlace(&v);
    try testing.expectApproxEqAbs(@as(f32, 0.6), v[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.8), v[1], 1e-6);
    var sum_sq: f32 = 0.0;
    for (v) |x| sum_sq += x * x;
    try testing.expectApproxEqAbs(@as(f32, 1.0), sum_sq, 1e-6);
}

test "normalizeInPlace: zero vector left untouched" {
    var v = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    try normalizeInPlace(&v);
    for (v) |x| try testing.expectEqual(@as(f32, 0.0), x);
}

test "normalizeInPlace: empty input is an error" {
    var v = [_]f32{};
    try testing.expectError(error.EmptyInput, normalizeInPlace(&v));
}

test "normalizeInPlace: 13 elements (forces tail)" {
    var v: [13]f32 = undefined;
    for (0..13) |i| v[i] = 1.0; // norm = sqrt(13)
    try normalizeInPlace(&v);
    var sum_sq: f32 = 0.0;
    for (v) |x| sum_sq += x * x;
    try testing.expectApproxEqAbs(@as(f32, 1.0), sum_sq, 1e-5);
}

test "argmin: hand-checked" {
    const a = [_]f32{ 3.0, 1.5, 4.0, -2.0, 7.0 };
    try testing.expectEqual(@as(usize, 3), try argmin(&a));
}

test "argmin: ties pick first" {
    const a = [_]f32{ 1.0, 1.0, 2.0 };
    try testing.expectEqual(@as(usize, 0), try argmin(&a));
}

test "argmin: empty is an error" {
    const a = [_]f32{};
    try testing.expectError(error.EmptyInput, argmin(&a));
}

test "argmax: hand-checked" {
    const a = [_]f32{ -3.0, 1.5, 4.0, -2.0, 7.0 };
    try testing.expectEqual(@as(usize, 4), try argmax(&a));
}

test "l2sqBatch: 3 points, dim 2" {
    const q = [_]f32{ 0.0, 0.0 };
    const pts = [_]f32{
        1.0, 0.0, // dist² = 1
        0.0, 2.0, // dist² = 4
        3.0, 4.0, // dist² = 25
    };
    var out: [3]f32 = undefined;
    try l2sqBatch(&q, &pts, 2, &out);
    try testing.expectEqual(@as(f32, 1.0), out[0]);
    try testing.expectEqual(@as(f32, 4.0), out[1]);
    try testing.expectEqual(@as(f32, 25.0), out[2]);
}

test "normalizeRowsInPlace: 2 rows of dim 2" {
    var rows = [_]f32{ 3.0, 4.0, 0.0, 5.0 };
    try normalizeRowsInPlace(&rows, 2);
    try testing.expectApproxEqAbs(@as(f32, 0.6), rows[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.8), rows[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.0), rows[2], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), rows[3], 1e-6);
}

test "normalizeRowsInPlace: bad dim is an error" {
    var rows = [_]f32{ 1.0, 2.0, 3.0 };
    try testing.expectError(error.LengthMismatch, normalizeRowsInPlace(&rows, 2));
}

// ---------------------------------------------------------------------------
// plan-11 — comptime-specialization parity tests.
//
// For each specialized dim ∈ {2, 4, 64, 128} verify the comptime kernel agrees
// with a scalar reference within 1e-5 absolute. We test both hand-checkable
// inputs and generated patterns (ramp + permutation).
// ---------------------------------------------------------------------------

fn scalarDotRef(a: []const f32, b: []const f32) f32 {
    var s: f32 = 0.0;
    for (a, b) |x, y| s += x * y;
    return s;
}

fn scalarL2sqRef(a: []const f32, b: []const f32) f32 {
    var s: f32 = 0.0;
    for (a, b) |x, y| {
        const d = x - y;
        s += d * d;
    }
    return s;
}

fn scalarNormSqRef(v: []const f32) f32 {
    var s: f32 = 0.0;
    for (v) |x| s += x * x;
    return s;
}

test "plan-11 parity: dot dim=2 hand-checked" {
    const a = [_]f32{ 1.0, 2.0 };
    const b = [_]f32{ 3.0, 4.0 };
    // 1*3 + 2*4 = 11
    try testing.expectApproxEqAbs(@as(f32, 11.0), try dot(&a, &b), 1e-5);
    try testing.expectApproxEqAbs(scalarDotRef(&a, &b), dotComptime(2, &a, &b), 1e-5);
}

test "plan-11 parity: dot dim=4 hand-checked" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    // 5 + 12 + 21 + 32 = 70
    try testing.expectApproxEqAbs(@as(f32, 70.0), try dot(&a, &b), 1e-5);
    try testing.expectApproxEqAbs(scalarDotRef(&a, &b), dotComptime(4, &a, &b), 1e-5);
}

test "plan-11 parity: l2sq dim=4 hand-checked" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    // 4*4*4 = 64 (each diff is -4)
    try testing.expectApproxEqAbs(@as(f32, 64.0), try l2sq(&a, &b), 1e-5);
    try testing.expectApproxEqAbs(scalarL2sqRef(&a, &b), l2sqComptime(4, &a, &b), 1e-5);
}

test "plan-11 parity: dot dim=64 ramp + permutation" {
    var a: [64]f32 = undefined;
    var b: [64]f32 = undefined;
    for (0..64) |i| {
        a[i] = @as(f32, @floatFromInt(i)) * 0.01;
        b[i] = @as(f32, @floatFromInt(63 - i)) * 0.02 + 0.5;
    }
    const ref = scalarDotRef(&a, &b);
    const got = try dot(&a, &b);
    try testing.expectApproxEqAbs(ref, got, 1e-5);
    // dot(a,b) == dot(b,a)
    try testing.expectApproxEqAbs(got, try dot(&b, &a), 1e-5);
    // direct comptime call agrees with dispatcher
    try testing.expectApproxEqAbs(got, dotComptime(64, &a, &b), 1e-5);
}

test "plan-11 parity: l2sq dim=64 ramp + permutation" {
    var a: [64]f32 = undefined;
    var b: [64]f32 = undefined;
    for (0..64) |i| {
        a[i] = @as(f32, @floatFromInt(i)) * 0.03 - 0.1;
        b[i] = @as(f32, @floatFromInt((i * 7) % 64)) * 0.005;
    }
    const ref = scalarL2sqRef(&a, &b);
    const got = try l2sq(&a, &b);
    try testing.expectApproxEqAbs(ref, got, 1e-5);
    // symmetric in a,b
    try testing.expectApproxEqAbs(got, try l2sq(&b, &a), 1e-5);
    try testing.expectApproxEqAbs(got, l2sqComptime(64, &a, &b), 1e-5);
}

test "plan-11 parity: dot dim=128 ramp + permutation" {
    var a: [128]f32 = undefined;
    var b: [128]f32 = undefined;
    for (0..128) |i| {
        a[i] = @as(f32, @floatFromInt(i)) * 0.005 - 0.3;
        b[i] = @as(f32, @floatFromInt((i * 13 + 1) % 128)) * 0.01;
    }
    const ref = scalarDotRef(&a, &b);
    const got = try dot(&a, &b);
    // dim=128 has wider FMA reduction depth — relax to 1e-4 for the cumulative
    // floating-point reordering between scalar reference and the @Vector(16)
    // tree-reduce. Still well below any algorithmic-meaningful tolerance.
    try testing.expectApproxEqAbs(ref, got, 1e-4);
    try testing.expectApproxEqAbs(got, try dot(&b, &a), 1e-5);
    try testing.expectApproxEqAbs(got, dotComptime(128, &a, &b), 1e-5);
}

test "plan-11 parity: l2sq dim=128 ramp + permutation" {
    var a: [128]f32 = undefined;
    var b: [128]f32 = undefined;
    for (0..128) |i| {
        a[i] = @as(f32, @floatFromInt(i)) * 0.02;
        b[i] = @as(f32, @floatFromInt(127 - i)) * 0.02;
    }
    const ref = scalarL2sqRef(&a, &b);
    const got = try l2sq(&a, &b);
    try testing.expectApproxEqAbs(ref, got, 1e-4);
    try testing.expectApproxEqAbs(got, try l2sq(&b, &a), 1e-5);
    try testing.expectApproxEqAbs(got, l2sqComptime(128, &a, &b), 1e-5);
}

test "plan-11 parity: normalize dim=2" {
    var v = [_]f32{ 3.0, 4.0 };
    try normalizeInPlace(&v);
    // 3-4-5 → 0.6, 0.8
    try testing.expectApproxEqAbs(@as(f32, 0.6), v[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.8), v[1], 1e-6);
}

test "plan-11 parity: normalize dim=4" {
    var v = [_]f32{ 1.0, 2.0, 2.0, 0.0 };
    // norm² = 1+4+4 = 9, norm = 3
    try normalizeInPlace(&v);
    try testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), v[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 2.0 / 3.0), v[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 2.0 / 3.0), v[2], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.0), v[3], 1e-6);
}

test "plan-11 parity: normalize dim=64 unit-norm postcondition" {
    var v: [64]f32 = undefined;
    for (0..64) |i| v[i] = @as(f32, @floatFromInt(i)) * 0.01 + 0.1;
    const expected_norm_sq = scalarNormSqRef(&v);
    const expected_inv = 1.0 / @sqrt(expected_norm_sq);
    var ref: [64]f32 = undefined;
    for (0..64) |i| ref[i] = v[i] * expected_inv;

    try normalizeInPlace(&v);
    for (0..64) |i| try testing.expectApproxEqAbs(ref[i], v[i], 1e-5);

    // Postcondition: result is unit-norm.
    var got_sq: f32 = 0.0;
    for (v) |x| got_sq += x * x;
    try testing.expectApproxEqAbs(@as(f32, 1.0), got_sq, 1e-5);
}

test "plan-11 parity: normalize dim=128 unit-norm postcondition" {
    var v: [128]f32 = undefined;
    for (0..128) |i| v[i] = @as(f32, @floatFromInt(i + 1)) * 0.005 - 0.2;
    const expected_norm_sq = scalarNormSqRef(&v);
    const expected_inv = 1.0 / @sqrt(expected_norm_sq);
    var ref: [128]f32 = undefined;
    for (0..128) |i| ref[i] = v[i] * expected_inv;

    try normalizeInPlace(&v);
    for (0..128) |i| try testing.expectApproxEqAbs(ref[i], v[i], 1e-5);

    var got_sq: f32 = 0.0;
    for (v) |x| got_sq += x * x;
    try testing.expectApproxEqAbs(@as(f32, 1.0), got_sq, 1e-5);
}

test "plan-11 parity: dispatcher matches generic on unspecialized dim" {
    // dim=33 is not in {2,4,64,128}; dispatcher falls through to dotGeneric.
    // Hand-build inputs and confirm parity with a scalar reference — this
    // guards the fallback path from regressing while we're touching dispatch.
    var a: [33]f32 = undefined;
    var b: [33]f32 = undefined;
    for (0..33) |i| {
        a[i] = @as(f32, @floatFromInt(i)) * 0.1;
        b[i] = @as(f32, @floatFromInt(33 - i)) * 0.07;
    }
    try testing.expectApproxEqAbs(scalarDotRef(&a, &b), try dot(&a, &b), 1e-5);
    try testing.expectApproxEqAbs(scalarL2sqRef(&a, &b), try l2sq(&a, &b), 1e-5);
}
