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
pub fn dot(a: []const f32, b: []const f32) VecError!f32 {
    if (a.len != b.len) return error.LengthMismatch;
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
pub fn l2sq(a: []const f32, b: []const f32) VecError!f32 {
    if (a.len != b.len) return error.LengthMismatch;
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

/// L2-normalise `v` in place. Zero-length vectors are left untouched (the
/// only reasonable choice — there is no unit direction for the zero vector).
///
/// paper convention: ColBERT/Tachiom assume unit-norm token embeddings so
/// cosine collapses to dot product.
pub fn normalizeInPlace(v: []f32) VecError!void {
    if (v.len == 0) return error.EmptyInput;

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
