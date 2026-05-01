//! benchmarks/msmarco_v1.zig — MS MARCO-v1 (dev.small) harness entry.
//!
//! Owner: retriever. Target operating points (paper Table 1):
//!   - MRR@10 = 39.0 → 10 ms/query
//!   - MRR@10 = 39.3 → 14 ms/query
//!
//! Single-thread (paper §9). Sweeps κ_c × κ_d × α over the paper §6 grid via
//! `tac.retrieval.bench.runSweep` and writes one CSV row per cell.
//!
//! Build: `zig build run-bench_msmarco -Doptimize=ReleaseFast -- <args>`.
//!
//! Inputs (produced by `tools/encode.py`):
//!   --index    path/to/msmarco.tac        (tac.index.storage.serialise output)
//!   --queries  path/to/queries.bin        (encoded with --mode queries)
//!   --qids     path/to/queries.bin.qids   (sidecar, u32 LE per query)
//!   --qrels    path/to/qrels.dev.small.tsv  (TREC-style)
//!   --out      path/to/results.csv

const std = @import("std");
const tac = @import("tac");
const runner = @import("common/runner.zig");

fn die(msg: []const u8) noreturn {
    std.debug.print("{s}\n", .{msg});
    std.process.exit(2);
}

fn parseArgs(m: std.process.Init.Minimal, gpa: std.mem.Allocator) !runner.RunArgs {
    var iter = try std.process.Args.Iterator.initAllocator(m.args, gpa);
    defer iter.deinit();
    _ = iter.next(); // program name

    var index_path: []const u8 = "";
    var queries_path: []const u8 = "";
    var qids_path: []const u8 = "";
    var qrels_path: []const u8 = "";
    var out_csv: []const u8 = "";
    var git_sha: []const u8 = "unknown";

    while (iter.next()) |a| {
        if (std.mem.eql(u8, a, "--index")) {
            const v = iter.next() orelse die("--index needs a path");
            index_path = try gpa.dupe(u8, v);
        } else if (std.mem.eql(u8, a, "--queries")) {
            const v = iter.next() orelse die("--queries needs a path");
            queries_path = try gpa.dupe(u8, v);
        } else if (std.mem.eql(u8, a, "--qids")) {
            const v = iter.next() orelse die("--qids needs a path");
            qids_path = try gpa.dupe(u8, v);
        } else if (std.mem.eql(u8, a, "--qrels")) {
            const v = iter.next() orelse die("--qrels needs a path");
            qrels_path = try gpa.dupe(u8, v);
        } else if (std.mem.eql(u8, a, "--out")) {
            const v = iter.next() orelse die("--out needs a path");
            out_csv = try gpa.dupe(u8, v);
        } else if (std.mem.eql(u8, a, "--git-sha")) {
            const v = iter.next() orelse die("--git-sha needs a value");
            git_sha = try gpa.dupe(u8, v);
        } else {
            std.debug.print("unknown arg: {s}\n", .{a});
            die("unknown arg; required: --index --queries --qids --qrels --out [--git-sha]");
        }
    }
    if (index_path.len == 0 or queries_path.len == 0 or qids_path.len == 0 or
        qrels_path.len == 0 or out_csv.len == 0)
    {
        die(
            \\bench_msmarco: missing required args.
            \\  --index PATH    .tac index file
            \\  --queries PATH  encoded queries.bin (token_dump format)
            \\  --qids PATH     queries.bin.qids sidecar (u32 LE per query)
            \\  --qrels PATH    qrels.tsv (qid<TAB>iter<TAB>doc_id<TAB>rel)
            \\  --out PATH      output CSV
            \\  [--git-sha SHA] traceability tag (defaults to "unknown")
        );
    }
    return .{
        .dataset = "msmarco-v1",
        .index_path = index_path,
        .queries_path = queries_path,
        .qids_path = qids_path,
        .qrels_path = qrels_path,
        .out_csv_path = out_csv,
        .git_sha = git_sha,
        .metric = .mrr_at_10,
        .min_rel = 1,
        .pin_core = 0,
    };
}

pub fn main(m: std.process.Init.Minimal) !void {
    var gpa_state: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();

    const args = try parseArgs(m, gpa);
    defer {
        gpa.free(args.index_path);
        gpa.free(args.queries_path);
        gpa.free(args.qids_path);
        gpa.free(args.qrels_path);
        gpa.free(args.out_csv_path);
        if (!std.mem.eql(u8, args.git_sha, "unknown")) gpa.free(args.git_sha);
    }
    // Surface that the linker pulls in the public sweep API.
    _ = tac.retrieval.bench.csv_header;

    try runner.runDatasetSweep(args, gpa);
}
