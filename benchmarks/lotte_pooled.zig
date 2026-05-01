//! benchmarks/lotte_pooled.zig — LoTTE-pooled (search/dev) harness entry.
//!
//! Owner: retriever. Target operating point (paper Table 1):
//!   - Success@5 = 67.5 → 11 ms/query
//!
//! Same shape as msmarco_v1.zig, but uses Success@5 as the metric.
//! See benchmarks/README.md for the runbook.

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
    _ = iter.next();

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
            \\bench_lotte: missing required args.
            \\  --index PATH    .tac index file
            \\  --queries PATH  encoded queries.bin (token_dump format)
            \\  --qids PATH     queries.bin.qids sidecar
            \\  --qrels PATH    qrels.tsv
            \\  --out PATH      output CSV
            \\  [--git-sha SHA] traceability tag
        );
    }
    return .{
        .dataset = "lotte-pooled",
        .index_path = index_path,
        .queries_path = queries_path,
        .qids_path = qids_path,
        .qrels_path = qrels_path,
        .out_csv_path = out_csv,
        .git_sha = git_sha,
        .metric = .success_at_5,
        .min_rel = 1,
        // Apple silicon doesn't expose pin_to_core via std; pinToCore is a
        // no-op on macOS. Leave null to avoid implying a guarantee we don't
        // have. On Linux the runner still pins when this is non-null.
        .pin_core = if (@import("builtin").os.tag == .linux) @as(?u32, 0) else null,
    };
}

pub fn main(m: std.process.Init.Minimal) !void {
    // SmpAllocator: thread-safe, hot-path-friendly. DebugAllocator's
    // bookkeeping inflates per-query latency variance and is unsafe to
    // publish as headline numbers — see L10 in audit-fixes-master-plan.md.
    const gpa = std.heap.smp_allocator;

    const args = try parseArgs(m, gpa);
    defer {
        gpa.free(args.index_path);
        gpa.free(args.queries_path);
        gpa.free(args.qids_path);
        gpa.free(args.qrels_path);
        gpa.free(args.out_csv_path);
        if (!std.mem.eql(u8, args.git_sha, "unknown")) gpa.free(args.git_sha);
    }
    _ = tac.retrieval.bench.csv_header;

    const protocol = runner.protocolFromEnv(m.environ);
    try runner.runDatasetSweep(args, protocol, gpa);
}
