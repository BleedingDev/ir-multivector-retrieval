//! tac — CLI entry point.
//!
//! Subcommands:
//!   index   Build an index from a token-dump file
//!   search  Run a query against an index
//!   eval    Run evaluation harness (MRR@10 / Success@k)
//!   bench   Run latency benchmarks
//!
//! The lead owns this file; teammates wire their subsystems in by adding a
//! function call here and a corresponding `cmd_*.zig` if needed.

const std = @import("std");
const tac = @import("tac");

pub fn main(m: std.process.Init.Minimal) !void {
    // Zig 0.16 args API: take a process.Init.Minimal, iterate via Args.Iterator.
    // initAllocator is the portable path (POSIX no-op, Windows/WASI need it).
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    var iter = try std.process.Args.Iterator.initAllocator(m.args, gpa.allocator());
    defer iter.deinit();
    _ = iter.next(); // program name

    const cmd = iter.next() orelse {
        printUsage();
        return;
    };

    if (std.mem.eql(u8, cmd, "index")) {
        std.debug.print("tac index: not yet wired\n", .{});
    } else if (std.mem.eql(u8, cmd, "search")) {
        std.debug.print("tac search: not yet wired\n", .{});
    } else if (std.mem.eql(u8, cmd, "eval")) {
        std.debug.print("tac eval: not yet wired\n", .{});
    } else if (std.mem.eql(u8, cmd, "bench")) {
        std.debug.print("tac bench: not yet wired (use `zig build run-bench_msmarco -- ...` for the per-dataset harnesses)\n", .{});
    } else if (std.mem.eql(u8, cmd, "--help") or std.mem.eql(u8, cmd, "-h")) {
        printUsage();
    } else {
        std.debug.print("unknown subcommand: {s}\n\n", .{cmd});
        printUsage();
        return error.UnknownSubcommand;
    }
}

fn printUsage() void {
    std.debug.print(
        \\tac — Tachiom (arxiv 2604.28142v1) reimplementation in Zig.
        \\
        \\Usage: tac <subcommand> [options]
        \\
        \\Subcommands:
        \\  index   Build an index from a token-dump file
        \\  search  Run a query against an index
        \\  eval    Run evaluation harness (MRR@10 / Success@k)
        \\  bench   Run latency benchmarks
        \\
        \\Defaults (paper-strict): see src/constants.zig.
        \\
    , .{});
}

test "constants are accessible from main module" {
    try std.testing.expectEqual(@as(u32, 128), tac.constants.TAC_MU);
    try std.testing.expectEqual(@as(u32, 32), tac.constants.PQ_M);
}
