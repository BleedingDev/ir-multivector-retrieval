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

pub fn main() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        printUsage();
        return;
    }

    const cmd = args[1];
    if (std.mem.eql(u8, cmd, "index")) {
        std.debug.print("tac index: not yet wired (waiting on indexer)\n", .{});
    } else if (std.mem.eql(u8, cmd, "search")) {
        std.debug.print("tac search: not yet wired (waiting on retriever)\n", .{});
    } else if (std.mem.eql(u8, cmd, "eval")) {
        std.debug.print("tac eval: not yet wired (waiting on retriever)\n", .{});
    } else if (std.mem.eql(u8, cmd, "bench")) {
        std.debug.print("tac bench: not yet wired (waiting on retriever)\n", .{});
    } else if (std.mem.eql(u8, cmd, "--help") or std.mem.eql(u8, cmd, "-h")) {
        printUsage();
    } else {
        std.debug.print("unknown subcommand: {s}\n\n", .{cmd});
        printUsage();
        std.process.exit(2);
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
