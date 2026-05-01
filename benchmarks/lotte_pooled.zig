//! benchmarks/lotte_pooled.zig — LoTTE-pooled (search/dev) harness entry.
//!
//! Owner: retriever. Target operating point (paper Table 1):
//!   - Success@5 = 67.5 → 11 ms/query
//!
//! Same shape as msmarco_v1.zig, but uses Success@5 as the metric. See
//! benchmarks/README.md for the runbook and `src/retrieval/bench.zig` for
//! the sweep core.

const std = @import("std");
const tac = @import("tac");

const Args = struct {
    index_path: []const u8,
    queries_path: []const u8,
    qrels_path: []const u8,
    out_path: []const u8,
};

fn parseArgs(gpa: std.mem.Allocator) !Args {
    const argv = try std.process.argsAlloc(gpa);
    defer std.process.argsFree(gpa, argv);

    var args = Args{
        .index_path = "",
        .queries_path = "",
        .qrels_path = "",
        .out_path = "",
    };
    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const a = argv[i];
        if (std.mem.eql(u8, a, "--index") and i + 1 < argv.len) {
            args.index_path = try gpa.dupe(u8, argv[i + 1]);
            i += 1;
        } else if (std.mem.eql(u8, a, "--queries") and i + 1 < argv.len) {
            args.queries_path = try gpa.dupe(u8, argv[i + 1]);
            i += 1;
        } else if (std.mem.eql(u8, a, "--qrels") and i + 1 < argv.len) {
            args.qrels_path = try gpa.dupe(u8, argv[i + 1]);
            i += 1;
        } else if (std.mem.eql(u8, a, "--out") and i + 1 < argv.len) {
            args.out_path = try gpa.dupe(u8, argv[i + 1]);
            i += 1;
        } else {
            std.debug.print("unknown arg: {s}\n", .{a});
            return error.BadArgs;
        }
    }
    if (args.index_path.len == 0 or args.queries_path.len == 0 or
        args.qrels_path.len == 0 or args.out_path.len == 0)
    {
        std.debug.print(
            \\lotte_pooled: missing required args.
            \\  --index PATH    serialised .tac index file
            \\  --queries PATH  encoded queries.bin (token_dump format)
            \\  --qrels PATH    qrels.tsv
            \\  --out PATH      output CSV
            \\
        , .{});
        return error.BadArgs;
    }
    return args;
}

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();

    const args = try parseArgs(gpa);
    defer {
        gpa.free(args.index_path);
        gpa.free(args.queries_path);
        gpa.free(args.qrels_path);
        gpa.free(args.out_path);
    }

    std.debug.print(
        \\lotte_pooled harness: parsed args.
        \\  index   = {s}
        \\  queries = {s}
        \\  qrels   = {s}
        \\  out     = {s}
        \\
        \\Real-data loaders not yet implemented — see benchmarks/README.md
        \\for the runbook. The sweep core itself (`tac.retrieval.bench`) is
        \\ready and unit-tested on the synthetic fixture.
        \\
    , .{ args.index_path, args.queries_path, args.qrels_path, args.out_path });

    _ = tac.retrieval.bench.kappa_c_grid;
    _ = tac.retrieval.bench.alpha_grid;

    return error.NotImplemented;
}
