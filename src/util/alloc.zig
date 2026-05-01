//! src/util/alloc.zig — allocator helpers, alignment, slice-of-slices builders.
//!
//! Owner: primitives-engineer.
//!
//! Two collaborators across this codebase need shared allocator scaffolding:
//!  - clusterer (paper §3) builds a CSR-style partition (per-token vector
//!    rows grouped together) that's awkward to express without a builder.
//!  - everyone touching SIMD wants explicit alignment for `@Vector(N, f32)`
//!    loads to avoid the unaligned slow path.
//!
//! Nothing here owns memory long-term — every helper takes a
//! `std.mem.Allocator` explicitly and the caller frees what it asked for.

const std = @import("std");
const vec = @import("vec.zig");

/// SIMD-friendly alignment for f32 vector loads. `@Vector(N, f32)` wants
/// `N * 4`-byte alignment for the fast path on x86/ARM SIMD ISAs.
pub const simd_f32_alignment: std.mem.Alignment = blk: {
    const bytes = vec.lane_count * @sizeOf(f32);
    break :blk std.mem.Alignment.fromByteUnits(bytes);
};

/// Allocate `n` f32 values aligned for SIMD vector loads. Caller frees with
/// `freeSimdF32`.
pub fn allocSimdF32(allocator: std.mem.Allocator, n: usize) ![]align(simd_f32_alignment.toByteUnits()) f32 {
    return allocator.alignedAlloc(f32, simd_f32_alignment, n);
}

pub fn freeSimdF32(allocator: std.mem.Allocator, slice: []align(simd_f32_alignment.toByteUnits()) f32) void {
    allocator.free(slice);
}

/// Round `x` up to the next multiple of `align_to`. `align_to` must be > 0.
pub fn alignUp(x: usize, align_to: usize) usize {
    std.debug.assert(align_to > 0);
    return ((x + align_to - 1) / align_to) * align_to;
}

/// CSR-like builder for a partition of `n_items` items into `n_groups`.
///
/// Two-pass usage:
///   1. Construct with `.init(allocator, n_groups, n_items)`.
///   2. Call `addCount(group_id)` for every (item, group) pair → records
///      the per-group counts.
///   3. Call `finalizeOffsets()` to convert counts → CSR offsets.
///   4. Call `place(group_id, item)` for every (item, group) pair → writes
///      items into the right CSR row in O(1) using a per-group cursor.
///   5. Call `view()` to get a `Csr` that lets you iterate `groupSlice(g)`.
///   6. Call `deinit()` when done.
///
/// Used by the clusterer to bucket vector rows by token id (paper §3:
/// per-token sub-corpora) and by the indexer for inverted-list construction.
pub fn CsrBuilder(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        n_groups: usize,
        n_items: usize,
        offsets: []usize, // length n_groups + 1
        cursor: []usize, // length n_groups, write pointer per group
        items: []T,
        offsets_ready: bool,

        pub fn init(
            allocator: std.mem.Allocator,
            n_groups: usize,
            n_items: usize,
        ) !Self {
            const offsets = try allocator.alloc(usize, n_groups + 1);
            errdefer allocator.free(offsets);
            const cursor = try allocator.alloc(usize, n_groups);
            errdefer allocator.free(cursor);
            const items = try allocator.alloc(T, n_items);
            @memset(offsets, 0);
            @memset(cursor, 0);
            return .{
                .allocator = allocator,
                .n_groups = n_groups,
                .n_items = n_items,
                .offsets = offsets,
                .cursor = cursor,
                .items = items,
                .offsets_ready = false,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.offsets);
            self.allocator.free(self.cursor);
            self.allocator.free(self.items);
            self.* = undefined;
        }

        /// Pass 1: bump the count for `group_id`.
        pub fn addCount(self: *Self, group_id: usize) void {
            std.debug.assert(group_id < self.n_groups);
            std.debug.assert(!self.offsets_ready);
            // Use offsets[group+1] as the per-group counter so we can
            // exclusive-scan into final offsets later.
            self.offsets[group_id + 1] += 1;
        }

        /// Convert counts to exclusive-prefix-sum offsets. After this,
        /// `offsets[g]..offsets[g+1]` is the slot range for group g.
        pub fn finalizeOffsets(self: *Self) !void {
            std.debug.assert(!self.offsets_ready);
            var total: usize = 0;
            self.offsets[0] = 0;
            var g: usize = 0;
            while (g < self.n_groups) : (g += 1) {
                const c = self.offsets[g + 1];
                total += c;
                self.offsets[g + 1] = total;
                self.cursor[g] = self.offsets[g];
            }
            if (total != self.n_items) return error.CountsMismatchTotal;
            self.offsets_ready = true;
        }

        /// Pass 2: write `item` into the next slot for `group_id`.
        pub fn place(self: *Self, group_id: usize, item: T) void {
            std.debug.assert(self.offsets_ready);
            std.debug.assert(group_id < self.n_groups);
            const idx = self.cursor[group_id];
            std.debug.assert(idx < self.offsets[group_id + 1]);
            self.items[idx] = item;
            self.cursor[group_id] = idx + 1;
        }

        pub fn view(self: *const Self) Csr(T) {
            std.debug.assert(self.offsets_ready);
            return .{ .offsets = self.offsets, .items = self.items };
        }
    };
}

/// Read-only CSR view returned by `CsrBuilder.view()`.
pub fn Csr(comptime T: type) type {
    return struct {
        offsets: []const usize,
        items: []T,

        pub fn nGroups(self: @This()) usize {
            return self.offsets.len - 1;
        }

        pub fn groupSlice(self: @This(), group_id: usize) []T {
            std.debug.assert(group_id < self.nGroups());
            return self.items[self.offsets[group_id]..self.offsets[group_id + 1]];
        }
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "alignUp: hand-checked" {
    try testing.expectEqual(@as(usize, 0), alignUp(0, 8));
    try testing.expectEqual(@as(usize, 8), alignUp(1, 8));
    try testing.expectEqual(@as(usize, 8), alignUp(8, 8));
    try testing.expectEqual(@as(usize, 16), alignUp(9, 8));
    try testing.expectEqual(@as(usize, 32), alignUp(17, 16));
}

test "allocSimdF32: returns an f32 slice with correct length and alignment" {
    const a = std.testing.allocator;
    const buf = try allocSimdF32(a, 32);
    defer freeSimdF32(a, buf);
    try testing.expectEqual(@as(usize, 32), buf.len);
    const addr = @intFromPtr(buf.ptr);
    try testing.expectEqual(@as(usize, 0), addr % simd_f32_alignment.toByteUnits());
}

test "CsrBuilder: bucket numbers by parity" {
    const a = std.testing.allocator;
    var b = try CsrBuilder(u32).init(a, 2, 6);
    defer b.deinit();

    const data = [_]u32{ 1, 2, 3, 4, 5, 6 };
    for (data) |x| b.addCount(@intCast(x % 2)); // group 0 = even, 1 = odd
    try b.finalizeOffsets();
    for (data) |x| b.place(@intCast(x % 2), x);

    const v = b.view();
    try testing.expectEqual(@as(usize, 2), v.nGroups());

    const evens = v.groupSlice(0);
    const odds = v.groupSlice(1);
    try testing.expectEqual(@as(usize, 3), evens.len);
    try testing.expectEqual(@as(usize, 3), odds.len);

    // place() preserves arrival order within a group.
    try testing.expectEqualSlices(u32, &.{ 2, 4, 6 }, evens);
    try testing.expectEqualSlices(u32, &.{ 1, 3, 5 }, odds);
}

test "CsrBuilder: empty group is fine" {
    const a = std.testing.allocator;
    var b = try CsrBuilder(u32).init(a, 3, 2);
    defer b.deinit();

    b.addCount(0);
    b.addCount(2);
    try b.finalizeOffsets();
    b.place(0, 100);
    b.place(2, 300);

    const v = b.view();
    try testing.expectEqualSlices(u32, &.{100}, v.groupSlice(0));
    try testing.expectEqual(@as(usize, 0), v.groupSlice(1).len);
    try testing.expectEqualSlices(u32, &.{300}, v.groupSlice(2));
}

test "CsrBuilder: counts/total mismatch is an error" {
    const a = std.testing.allocator;
    var b = try CsrBuilder(u32).init(a, 2, 4);
    defer b.deinit();
    b.addCount(0);
    b.addCount(1); // only 2 counts, but n_items = 4
    try testing.expectError(error.CountsMismatchTotal, b.finalizeOffsets());
}

test "simd_f32_alignment: matches lane_count * sizeof(f32)" {
    try testing.expectEqual(
        vec.lane_count * @sizeOf(f32),
        simd_f32_alignment.toByteUnits(),
    );
}
