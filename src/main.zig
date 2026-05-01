//! tac — CLI entry point.
//!
//! Subcommands:
//!   index   Build a Tachiom index from a tokens.bin file
//!   search  Run queries through the gather→prune→refine pipeline
//!   eval    Score a (queries.bin, qrels.tsv) pair on a built index
//!   bench   Run the paper §6 (κ_c × κ_d × α) sweep and emit CSV
//!
//! Lead-owned. Per-dataset benchmark binaries (bench_msmarco / bench_lotte)
//! live under benchmarks/ and are produced by the same `zig build install`.

const std = @import("std");
const tac = @import("tac");

const Allocator = std.mem.Allocator;

pub fn main(m: std.process.Init.Minimal) !void {
    // Production CLI hot path: smp_allocator is the ReleaseFast allocator
    // and is honest about peak/headline numbers; DebugAllocator was masking
    // real allocator overhead in `tac index` and skewing benchmarks. Tests
    // (`zig build test`) still go through std.testing.allocator, which
    // preserves leak detection on test exit — see finding #10 in
    // .codex/deep-research-zig-2026-05-01.txt.
    const gpa = std.heap.smp_allocator;

    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const cwd = std.Io.Dir.cwd();

    var iter = try std.process.Args.Iterator.initAllocator(m.args, gpa);
    defer iter.deinit();
    _ = iter.next(); // program name

    const cmd = iter.next() orelse {
        printUsage();
        return;
    };

    if (std.mem.eql(u8, cmd, "index")) {
        try cmdIndex(&iter, gpa, io, cwd);
    } else if (std.mem.eql(u8, cmd, "search")) {
        try cmdSearch(&iter, gpa, io, cwd);
    } else if (std.mem.eql(u8, cmd, "eval")) {
        try cmdEval(&iter, gpa, io, cwd);
    } else if (std.mem.eql(u8, cmd, "bench")) {
        try cmdBench(&iter, gpa, io, cwd);
    } else if (std.mem.eql(u8, cmd, "--help") or std.mem.eql(u8, cmd, "-h")) {
        printUsage();
    } else {
        std.debug.print("unknown subcommand: {s}\n\n", .{cmd});
        printUsage();
        return error.UnknownSubcommand;
    }
}

// ---------------------------------------------------------------------------
// `tac index <tokens.bin> <out.tac> [--kappa N] [--seed N]`
// ---------------------------------------------------------------------------

fn cmdIndex(iter: *std.process.Args.Iterator, gpa: Allocator, io: std.Io, cwd: std.Io.Dir) !void {
    var tokens_path: ?[]const u8 = null;
    var out_path: ?[]const u8 = null;
    var kappa: u32 = 4096;
    var seed: u64 = 42;
    var mu: u32 = tac.constants.TAC_MU;
    var tau: u32 = tac.constants.TAC_TAU;
    var epsilon: u32 = tac.constants.TAC_EPSILON;
    var theta: u32 = tac.constants.TAC_THETA;
    var n_threads: u32 = 1;
    var positional: u32 = 0;

    while (iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--kappa")) {
            kappa = try std.fmt.parseInt(u32, iter.next() orelse return error.MissingArg, 10);
        } else if (std.mem.eql(u8, arg, "--seed")) {
            seed = try std.fmt.parseInt(u64, iter.next() orelse return error.MissingArg, 10);
        } else if (std.mem.eql(u8, arg, "--threads")) {
            n_threads = try std.fmt.parseInt(u32, iter.next() orelse return error.MissingArg, 10);
        } else if (std.mem.eql(u8, arg, "--mu")) {
            mu = try std.fmt.parseInt(u32, iter.next() orelse return error.MissingArg, 10);
        } else if (std.mem.eql(u8, arg, "--tau")) {
            tau = try std.fmt.parseInt(u32, iter.next() orelse return error.MissingArg, 10);
        } else if (std.mem.eql(u8, arg, "--epsilon")) {
            epsilon = try std.fmt.parseInt(u32, iter.next() orelse return error.MissingArg, 10);
        } else if (std.mem.eql(u8, arg, "--theta")) {
            theta = try std.fmt.parseInt(u32, iter.next() orelse return error.MissingArg, 10);
        } else if (std.mem.startsWith(u8, arg, "--")) {
            std.debug.print("unknown flag: {s}\n", .{arg});
            return error.UnknownFlag;
        } else if (positional == 0) {
            tokens_path = arg;
            positional += 1;
        } else if (positional == 1) {
            out_path = arg;
            positional += 1;
        } else return error.TooManyPositional;
    }

    const tp = tokens_path orelse return usage("tac index <tokens.bin> <out.tac> [--kappa N] [--seed N]");
    const op = out_path orelse return usage("tac index <tokens.bin> <out.tac> [--kappa N] [--seed N]");

    std.debug.print("tac index: tokens={s} out={s} kappa={d} seed={d}\n", .{ tp, op, kappa, seed });

    const t_load_0 = nowNs();
    const td_raw = try slurpAligned8(io, cwd, tp, gpa);
    defer gpa.free(td_raw);
    const td_bytes: []align(8) const u8 = @alignCast(td_raw);
    const td = try tac.io.token_dump.parseBytes(td_bytes);
    const t_load_1 = nowNs();
    std.debug.print(
        "  loaded {d} docs / {d} tokens / dim={d} in {d:.1} ms\n",
        .{ td.n_docs, td.n_tokens, td.dim, msBetween(t_load_0, t_load_1) },
    );

    if (mu != tac.constants.TAC_MU or tau != tac.constants.TAC_TAU or
        epsilon != tac.constants.TAC_EPSILON or theta != tac.constants.TAC_THETA)
    {
        std.debug.print(
            "  paper-relax: μ={d} (default {d}), τ={d} (default {d}), ε={d} (default {d}), θ={d} (default {d})\n",
            .{ mu, tac.constants.TAC_MU, tau, tac.constants.TAC_TAU, epsilon, tac.constants.TAC_EPSILON, theta, tac.constants.TAC_THETA },
        );
    }

    if (n_threads > 1) {
        std.debug.print("  parallel build: {d} threads\n", .{n_threads});
    }

    const t_build_0 = nowNs();
    var image = try tac.index.storage.build(&td, .{
        .kappa_total = kappa,
        .seed = seed,
        .mu = mu,
        .tau = tau,
        .epsilon = epsilon,
        .theta = theta,
        .n_threads = n_threads,
        .verbose = true,
    }, gpa);
    defer image.deinit(gpa);
    const t_build_1 = nowNs();
    std.debug.print(
        "  built index in {d:.1} ms: {d} bytes ({d:.2} MB)\n",
        .{
            msBetween(t_build_0, t_build_1),
            image.bytes.len,
            @as(f64, @floatFromInt(image.bytes.len)) / (1024.0 * 1024.0),
        },
    );

    const t_write_0 = nowNs();
    try writeFile(io, cwd, op, image.bytes, gpa);
    const t_write_1 = nowNs();
    std.debug.print("  wrote {s} in {d:.1} ms\n", .{ op, msBetween(t_write_0, t_write_1) });
}

// ---------------------------------------------------------------------------
// `tac search <index.tac> <queries.bin> [--kappa-c N] [--kappa-d N] [--alpha F] [--top-k N]`
// ---------------------------------------------------------------------------

fn cmdSearch(iter: *std.process.Args.Iterator, gpa: Allocator, io: std.Io, cwd: std.Io.Dir) !void {
    var index_path: ?[]const u8 = null;
    var queries_path: ?[]const u8 = null;
    var kappa_c: u32 = 80;
    var kappa_d: u32 = 1000;
    var alpha: ?f32 = null;
    var top_k: u32 = 10;
    var positional: u32 = 0;

    while (iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--kappa-c")) {
            kappa_c = try std.fmt.parseInt(u32, iter.next() orelse return error.MissingArg, 10);
        } else if (std.mem.eql(u8, arg, "--kappa-d")) {
            kappa_d = try std.fmt.parseInt(u32, iter.next() orelse return error.MissingArg, 10);
        } else if (std.mem.eql(u8, arg, "--alpha")) {
            alpha = try std.fmt.parseFloat(f32, iter.next() orelse return error.MissingArg);
        } else if (std.mem.eql(u8, arg, "--top-k")) {
            top_k = try std.fmt.parseInt(u32, iter.next() orelse return error.MissingArg, 10);
        } else if (std.mem.startsWith(u8, arg, "--")) {
            std.debug.print("unknown flag: {s}\n", .{arg});
            return error.UnknownFlag;
        } else if (positional == 0) {
            index_path = arg;
            positional += 1;
        } else if (positional == 1) {
            queries_path = arg;
            positional += 1;
        } else return error.TooManyPositional;
    }

    const ip = index_path orelse return usage("tac search <index.tac> <queries.bin> [--kappa-c N] [--kappa-d N] [--alpha F] [--top-k N]");
    const qp = queries_path orelse return usage("tac search <index.tac> <queries.bin> [--kappa-c N] [--kappa-d N] [--alpha F] [--top-k N]");

    const idx_raw = try slurpAligned8(io, cwd, ip, gpa);
    defer gpa.free(idx_raw);
    const idx_bytes: []align(8) u8 = @alignCast(idx_raw);
    var index = try tac.index.storage.parse(idx_bytes, gpa);
    defer index.deinit(gpa);
    std.debug.print("loaded index: dim={d}\n", .{index.header.dim});

    const q_raw = try slurpAligned8(io, cwd, qp, gpa);
    defer gpa.free(q_raw);
    const q_bytes: []align(8) const u8 = @alignCast(q_raw);
    const queries = try tac.io.token_dump.parseBytes(q_bytes);
    std.debug.print("loaded queries: n={d} dim={d}\n", .{ queries.n_docs, queries.dim });

    if (queries.dim != index.header.dim) {
        std.debug.print("dim mismatch: queries dim={d} != index dim={d}\n", .{ queries.dim, index.header.dim });
        return error.DimMismatch;
    }

    const sp: tac.retrieval.search.SearchParams = .{
        .kappa_c = kappa_c,
        .kappa_d = kappa_d,
        .alpha = alpha,
        .top_k = top_k,
    };

    var total_ns: u64 = 0;
    var n_run: u32 = 0;

    var d: u64 = 0;
    while (d < queries.n_docs) : (d += 1) {
        const range = queries.docTokenRange(d);
        const n_q: u32 = @intCast(range[1] - range[0]);
        if (n_q == 0) continue;
        const start_off: usize = @intCast(range[0] * queries.dim);
        const end_off: usize = @intCast(range[1] * queries.dim);
        const query_tokens = queries.vectors[start_off..end_off];

        const t0 = nowNs();
        const hits = try tac.retrieval.search.search(&index, query_tokens, n_q, sp, gpa);
        const t1 = nowNs();
        defer gpa.free(hits);

        const elapsed_ms = msBetween(t0, t1);
        total_ns += (t1 - t0);
        n_run += 1;

        std.debug.print("q[{d}] n_q={d} t={d:.2}ms hits={d}: ", .{ d, n_q, elapsed_ms, hits.len });
        const show: usize = @min(hits.len, @as(usize, top_k));
        for (hits[0..show]) |h| {
            std.debug.print("({d}, {d:.4}) ", .{ h.doc_id, h.score });
        }
        std.debug.print("\n", .{});
    }

    if (n_run > 0) {
        const avg_ms: f64 =
            @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(n_run)) / 1e6;
        std.debug.print(
            "\nsummary: n={d} avg={d:.3}ms params(kappa_c={d}, kappa_d={d}, top_k={d})\n",
            .{ n_run, avg_ms, kappa_c, kappa_d, top_k },
        );
    }
}

// ---------------------------------------------------------------------------
// `tac eval <index.tac> <queries.bin> <qrels.tsv>
//           [--metric {mrr,success}] [--k N]
//           [--kappa-c N] [--kappa-d N] [--alpha F]`
// ---------------------------------------------------------------------------

const MetricChoice = enum { mrr, success };

fn cmdEval(iter: *std.process.Args.Iterator, gpa: Allocator, io: std.Io, cwd: std.Io.Dir) !void {
    var index_path: ?[]const u8 = null;
    var queries_path: ?[]const u8 = null;
    var qrels_path: ?[]const u8 = null;
    var metric: MetricChoice = .mrr;
    var k: u32 = 10;
    var kappa_c: u32 = 80;
    var kappa_d: u32 = 1000;
    var alpha: ?f32 = null;
    var min_rel: i32 = 1;
    var positional: u32 = 0;

    while (iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--metric")) {
            const v = iter.next() orelse return error.MissingArg;
            if (std.mem.eql(u8, v, "mrr")) {
                metric = .mrr;
            } else if (std.mem.eql(u8, v, "success")) {
                metric = .success;
            } else {
                std.debug.print("unknown metric: {s} (expected mrr|success)\n", .{v});
                return error.UnknownMetric;
            }
        } else if (std.mem.eql(u8, arg, "--k")) {
            k = try std.fmt.parseInt(u32, iter.next() orelse return error.MissingArg, 10);
        } else if (std.mem.eql(u8, arg, "--kappa-c")) {
            kappa_c = try std.fmt.parseInt(u32, iter.next() orelse return error.MissingArg, 10);
        } else if (std.mem.eql(u8, arg, "--kappa-d")) {
            kappa_d = try std.fmt.parseInt(u32, iter.next() orelse return error.MissingArg, 10);
        } else if (std.mem.eql(u8, arg, "--alpha")) {
            alpha = try std.fmt.parseFloat(f32, iter.next() orelse return error.MissingArg);
        } else if (std.mem.eql(u8, arg, "--min-rel")) {
            min_rel = try std.fmt.parseInt(i32, iter.next() orelse return error.MissingArg, 10);
        } else if (std.mem.startsWith(u8, arg, "--")) {
            std.debug.print("unknown flag: {s}\n", .{arg});
            return error.UnknownFlag;
        } else if (positional == 0) {
            index_path = arg;
            positional += 1;
        } else if (positional == 1) {
            queries_path = arg;
            positional += 1;
        } else if (positional == 2) {
            qrels_path = arg;
            positional += 1;
        } else return error.TooManyPositional;
    }

    const ip = index_path orelse return usage(eval_usage_line);
    const qp = queries_path orelse return usage(eval_usage_line);
    const rp = qrels_path orelse return usage(eval_usage_line);

    // Load index.
    const idx_raw = try slurpAligned8(io, cwd, ip, gpa);
    defer gpa.free(idx_raw);
    const idx_bytes: []align(8) u8 = @alignCast(idx_raw);
    var index = try tac.index.storage.parse(idx_bytes, gpa);
    defer index.deinit(gpa);

    // Load queries (token dump). For `tac eval` we map each query by its
    // 0-based index in the dump → qid = index+1, mirroring the tac search
    // output convention. The qids sidecar is optional here; if it exists we
    // honour it, otherwise we fall back to 1-indexed sequential qids.
    const q_raw = try slurpAligned8(io, cwd, qp, gpa);
    defer gpa.free(q_raw);
    const q_bytes: []align(8) const u8 = @alignCast(q_raw);
    const queries = try tac.io.token_dump.parseBytes(q_bytes);

    if (queries.dim != index.header.dim) {
        std.debug.print(
            "dim mismatch: queries dim={d} != index dim={d}\n",
            .{ queries.dim, index.header.dim },
        );
        return error.DimMismatch;
    }

    const qids = try loadOrSyntheticQids(io, cwd, qp, queries.n_docs, gpa);
    defer gpa.free(qids);

    // Load qrels (CLI variant: 2/3/4 columns).
    const qrels_raw = try slurpFileBytes(io, cwd, rp, gpa);
    defer gpa.free(qrels_raw);
    var qrels_tbl = try tac.eval.qrels.parseBytes(qrels_raw, min_rel, gpa);
    defer qrels_tbl.deinit(gpa);

    // Run retrieval per query, score with the chosen metric.
    const sp: tac.retrieval.search.SearchParams = .{
        .kappa_c = kappa_c,
        .kappa_d = kappa_d,
        .alpha = alpha,
        .top_k = k,
    };

    var n_scored: u32 = 0;
    var n_skipped_no_qrels: u32 = 0;
    var score_sum: f64 = 0.0;
    const t0 = nowNs();

    var d: u64 = 0;
    while (d < queries.n_docs) : (d += 1) {
        const range = queries.docTokenRange(d);
        const n_q: u32 = @intCast(range[1] - range[0]);
        if (n_q == 0) continue;

        const qid = qids[@intCast(d)];
        const rels = qrels_tbl.forQuery(qid);
        if (rels.len == 0) {
            n_skipped_no_qrels += 1;
            continue;
        }

        const start_off: usize = @intCast(range[0] * queries.dim);
        const end_off: usize = @intCast(range[1] * queries.dim);
        const query_tokens = queries.vectors[start_off..end_off];

        const hits = try tac.retrieval.search.search(&index, query_tokens, n_q, sp, gpa);
        defer gpa.free(hits);

        const want: usize = @min(hits.len, @as(usize, k));
        const ranking = try gpa.alloc(u32, want);
        defer gpa.free(ranking);
        for (hits[0..want], 0..) |h, i| ranking[i] = h.doc_id;

        const m: f32 = switch (metric) {
            .mrr => tac.eval.metrics.mrrAt(ranking, rels, k),
            .success => tac.eval.metrics.successAt(ranking, rels, k),
        };
        score_sum += m;
        n_scored += 1;
    }

    const t1 = nowNs();
    const elapsed_ms = msBetween(t0, t1);

    const metric_label: []const u8 = switch (metric) {
        .mrr => "mrr",
        .success => "success",
    };

    if (n_scored == 0) {
        std.debug.print(
            "tac eval: no queries scored — checked {d} queries, none had matching qrels (qids in queries.bin.qids vs qrels.tsv don't overlap?)\n",
            .{queries.n_docs},
        );
        return error.NoOverlap;
    }

    const avg: f64 = score_sum / @as(f64, @floatFromInt(n_scored));
    std.debug.print(
        "tac eval: n_queries={d} metric={s}@{d} score={d:.4} elapsed={d:.1}ms (skipped_no_qrels={d}, kappa_c={d}, kappa_d={d})\n",
        .{ n_scored, metric_label, k, avg, elapsed_ms, n_skipped_no_qrels, kappa_c, kappa_d },
    );
}

const eval_usage_line: []const u8 =
    "tac eval <index.tac> <queries.bin> <qrels.tsv> [--metric mrr|success] [--k N] [--kappa-c N] [--kappa-d N] [--alpha F]";

// ---------------------------------------------------------------------------
// `tac bench <index.tac> <queries.bin> <qrels.tsv> [--metric mrr|success] [--out PATH]`
// ---------------------------------------------------------------------------

fn cmdBench(iter: *std.process.Args.Iterator, gpa: Allocator, io: std.Io, cwd: std.Io.Dir) !void {
    var index_path: ?[]const u8 = null;
    var queries_path: ?[]const u8 = null;
    var qrels_path: ?[]const u8 = null;
    var out_path: ?[]const u8 = null;
    var metric: MetricChoice = .mrr;
    var dataset_label: []const u8 = "tac-bench";
    var min_rel: i32 = 1;
    var positional: u32 = 0;

    while (iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--out")) {
            out_path = iter.next() orelse return error.MissingArg;
        } else if (std.mem.eql(u8, arg, "--metric")) {
            const v = iter.next() orelse return error.MissingArg;
            if (std.mem.eql(u8, v, "mrr")) {
                metric = .mrr;
            } else if (std.mem.eql(u8, v, "success")) {
                metric = .success;
            } else {
                std.debug.print("unknown metric: {s} (expected mrr|success)\n", .{v});
                return error.UnknownMetric;
            }
        } else if (std.mem.eql(u8, arg, "--dataset")) {
            dataset_label = iter.next() orelse return error.MissingArg;
        } else if (std.mem.eql(u8, arg, "--min-rel")) {
            min_rel = try std.fmt.parseInt(i32, iter.next() orelse return error.MissingArg, 10);
        } else if (std.mem.startsWith(u8, arg, "--")) {
            std.debug.print("unknown flag: {s}\n", .{arg});
            return error.UnknownFlag;
        } else if (positional == 0) {
            index_path = arg;
            positional += 1;
        } else if (positional == 1) {
            queries_path = arg;
            positional += 1;
        } else if (positional == 2) {
            qrels_path = arg;
            positional += 1;
        } else return error.TooManyPositional;
    }

    const ip = index_path orelse return usage(bench_usage_line);
    const qp = queries_path orelse return usage(bench_usage_line);
    const rp = qrels_path orelse return usage(bench_usage_line);

    // Load index.
    const idx_raw = try slurpAligned8(io, cwd, ip, gpa);
    defer gpa.free(idx_raw);
    const idx_bytes: []align(8) u8 = @alignCast(idx_raw);
    var index = try tac.index.storage.parse(idx_bytes, gpa);
    defer index.deinit(gpa);

    // Load queries.
    const q_raw = try slurpAligned8(io, cwd, qp, gpa);
    defer gpa.free(q_raw);
    const q_bytes: []align(8) const u8 = @alignCast(q_raw);
    const queries = try tac.io.token_dump.parseBytes(q_bytes);

    if (queries.dim != index.header.dim) {
        std.debug.print(
            "dim mismatch: queries dim={d} != index dim={d}\n",
            .{ queries.dim, index.header.dim },
        );
        return error.DimMismatch;
    }

    const qids = try loadOrSyntheticQids(io, cwd, qp, queries.n_docs, gpa);
    defer gpa.free(qids);

    // Load qrels.
    const qrels_raw = try slurpFileBytes(io, cwd, rp, gpa);
    defer gpa.free(qrels_raw);
    var qrels_tbl = try tac.eval.qrels.parseBytes(qrels_raw, min_rel, gpa);
    defer qrels_tbl.deinit(gpa);

    // Build a bench.QueryPack from the queries with matching qrels.
    var pack_n_q: std.ArrayList(u32) = .empty;
    defer pack_n_q.deinit(gpa);
    var pack_q_offsets: std.ArrayList(u64) = .empty;
    defer pack_q_offsets.deinit(gpa);
    var pack_tokens: std.ArrayList(f32) = .empty;
    defer pack_tokens.deinit(gpa);
    var pack_qrels: std.ArrayList(u32) = .empty;
    defer pack_qrels.deinit(gpa);
    var pack_qrels_lo: std.ArrayList(u64) = .empty;
    defer pack_qrels_lo.deinit(gpa);

    try pack_q_offsets.append(gpa, 0);
    try pack_qrels_lo.append(gpa, 0);

    var d: u64 = 0;
    while (d < queries.n_docs) : (d += 1) {
        const range = queries.docTokenRange(d);
        const n_q: u32 = @intCast(range[1] - range[0]);
        if (n_q == 0) continue;
        const qid = qids[@intCast(d)];
        const rels = qrels_tbl.forQuery(qid);
        if (rels.len == 0) continue;

        const start_off: usize = @intCast(range[0] * queries.dim);
        const end_off: usize = @intCast(range[1] * queries.dim);

        try pack_n_q.append(gpa, n_q);
        try pack_tokens.appendSlice(gpa, queries.vectors[start_off..end_off]);
        try pack_q_offsets.append(gpa, @as(u64, pack_tokens.items.len));
        try pack_qrels.appendSlice(gpa, rels);
        try pack_qrels_lo.append(gpa, @as(u64, pack_qrels.items.len));
    }

    if (pack_n_q.items.len == 0) {
        std.debug.print(
            "tac bench: no queries scored — qids in queries.bin.qids vs qrels.tsv don't overlap\n",
            .{},
        );
        return error.NoOverlap;
    }

    const pack: tac.retrieval.bench.QueryPack = .{
        .n_q = pack_n_q.items,
        .q_offsets = pack_q_offsets.items,
        .tokens = pack_tokens.items,
        .qrels = pack_qrels.items,
        .qrels_lo = pack_qrels_lo.items,
    };

    const metric_kind: tac.retrieval.bench.MetricKind = switch (metric) {
        .mrr => .mrr_at_10,
        .success => .success_at_5,
    };

    std.debug.print(
        "tac bench: dataset={s} n_queries={d} metric={s} grid={d}×{d}×{d} cells\n",
        .{
            dataset_label,
            pack.n_q.len,
            switch (metric) {
                .mrr => "mrr@10",
                .success => "success@5",
            },
            tac.retrieval.bench.kappa_c_grid.len,
            tac.retrieval.bench.kappa_d_grid.len,
            tac.retrieval.bench.alpha_grid.len,
        },
    );

    // Open output sink: file if --out given, else stdout (still goes through
    // std.Io's writer abstraction via the file descriptor).
    if (out_path) |op| {
        var out_file = try cwd.createFile(io, op, .{ .truncate = true });
        defer out_file.close(io);
        var write_buf: [16 * 1024]u8 = undefined;
        var fw = out_file.writer(io, &write_buf);
        const w = &fw.interface;
        try runBenchSweep(&index, pack, metric_kind, dataset_label, w, gpa);
        try w.flush();
        std.debug.print("tac bench: wrote {s}\n", .{op});
    } else {
        // No --out → format to stdout via std.debug.print line-by-line, since
        // the std.Io writer abstraction here isn't wired to stdout the way
        // the file path is. The CSV is still well-formed.
        try runBenchSweepStdout(&index, pack, metric_kind, dataset_label, gpa);
    }
}

const bench_usage_line: []const u8 =
    "tac bench <index.tac> <queries.bin> <qrels.tsv> [--metric mrr|success] [--out results.csv] [--dataset NAME]";

const SweepCtx = struct {
    writer: *std.Io.Writer,
    dataset: []const u8,
    git_sha: []const u8,
    n_done: u32,
};

fn sweepRowToWriter(row: tac.retrieval.bench.SweepRow, ctx: *anyopaque) anyerror!void {
    const c: *SweepCtx = @ptrCast(@alignCast(ctx));
    try tac.retrieval.bench.formatCsvRow(c.writer, c.dataset, c.git_sha, row);
    try c.writer.writeByte('\n');
    try c.writer.flush();
    c.n_done += 1;
    std.debug.print(
        "  cell {d}: kappa_c={d} kappa_d={d} alpha={d} q={d:.4} avg={d:.2}ms\n",
        .{
            c.n_done,
            row.kappa_c,
            row.kappa_d,
            row.alpha_x100,
            row.quality,
            row.avg_total_ms,
        },
    );
}

fn runBenchSweep(
    index: *const tac.index.storage.Index,
    pack: tac.retrieval.bench.QueryPack,
    metric: tac.retrieval.bench.MetricKind,
    dataset_label: []const u8,
    writer: *std.Io.Writer,
    gpa: Allocator,
) !void {
    try writer.writeAll(tac.retrieval.bench.csv_header);
    try writer.writeByte('\n');
    var ctx: SweepCtx = .{
        .writer = writer,
        .dataset = dataset_label,
        .git_sha = "unknown",
        .n_done = 0,
    };
    try tac.retrieval.bench.runSweep(index, pack, metric, sweepRowToWriter, &ctx, gpa);
}

const StdoutCtx = struct {
    dataset: []const u8,
    n_done: u32,
};

fn sweepRowToStdout(row: tac.retrieval.bench.SweepRow, ctx: *anyopaque) anyerror!void {
    const c: *StdoutCtx = @ptrCast(@alignCast(ctx));
    var buf: [512]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try tac.retrieval.bench.formatCsvRow(&w, c.dataset, "unknown", row);
    std.debug.print("{s}\n", .{w.buffered()});
    c.n_done += 1;
}

fn runBenchSweepStdout(
    index: *const tac.index.storage.Index,
    pack: tac.retrieval.bench.QueryPack,
    metric: tac.retrieval.bench.MetricKind,
    dataset_label: []const u8,
    gpa: Allocator,
) !void {
    std.debug.print("{s}\n", .{tac.retrieval.bench.csv_header});
    var ctx: StdoutCtx = .{ .dataset = dataset_label, .n_done = 0 };
    try tac.retrieval.bench.runSweep(index, pack, metric, sweepRowToStdout, &ctx, gpa);
}

/// Resolve qids for a queries.bin: prefer `<queries_path>.qids` (little-endian
/// u32 sidecar emitted by tools/encode.py --mode queries), fall back to
/// 1-indexed sequential qids when the sidecar is absent. The fallback matches
/// the convention used by the tac search debug output and the user-built
/// fixtures (`1\t<doc_id>` for the first query).
fn loadOrSyntheticQids(
    io: std.Io,
    cwd: std.Io.Dir,
    queries_path: []const u8,
    n_queries: u64,
    gpa: Allocator,
) ![]u32 {
    const sidecar_path = try std.fmt.allocPrint(gpa, "{s}.qids", .{queries_path});
    defer gpa.free(sidecar_path);

    const file_or_err = cwd.openFile(io, sidecar_path, .{});
    if (file_or_err) |f_| {
        var f = f_;
        defer f.close(io);
        const stat = try f.stat(io);
        const size: usize = std.math.cast(usize, stat.size) orelse return error.FileTooLarge;
        const expected: usize = @intCast(n_queries * @sizeOf(u32));
        if (size != expected) {
            std.debug.print(
                "qids sidecar {s}: size={d} bytes but queries.n_docs={d} (expected {d})\n",
                .{ sidecar_path, size, n_queries, expected },
            );
            return error.QidCountMismatch;
        }
        const buf = try gpa.alloc(u8, size);
        defer gpa.free(buf);
        var read_buf: [16 * 1024]u8 = undefined;
        var fr = f.reader(io, &read_buf);
        const reader = &fr.interface;
        try reader.readSliceAll(buf);
        const qids = try gpa.alloc(u32, @intCast(n_queries));
        var i: usize = 0;
        while (i < qids.len) : (i += 1) {
            qids[i] = std.mem.readInt(u32, buf[i * 4 ..][0..4], .little);
        }
        return qids;
    } else |err| switch (err) {
        error.FileNotFound => {
            const qids = try gpa.alloc(u32, @intCast(n_queries));
            var i: u32 = 0;
            while (i < qids.len) : (i += 1) qids[i] = i + 1;
            return qids;
        },
        else => return err,
    }
}

/// Slurp a file as a fresh byte buffer (no alignment requirement). Used for
/// qrels.tsv where the parser walks the bytes by hand.
fn slurpFileBytes(io: std.Io, dir: std.Io.Dir, path: []const u8, gpa: Allocator) ![]u8 {
    var f = try dir.openFile(io, path, .{});
    defer f.close(io);
    const stat = try f.stat(io);
    const size: usize = std.math.cast(usize, stat.size) orelse return error.FileTooLarge;
    const buf = try gpa.alloc(u8, size);
    errdefer gpa.free(buf);
    var read_buf: [16 * 1024]u8 = undefined;
    var fr = f.reader(io, &read_buf);
    const reader = &fr.interface;
    try reader.readSliceAll(buf);
    return buf;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Read a file into an 8-byte-aligned buffer. Matches the pattern proven
/// in benchmarks/common/runner.zig::slurp under Zig 0.16's std.Io.
fn slurpAligned8(io: std.Io, dir: std.Io.Dir, path: []const u8, gpa: Allocator) ![]align(8) u8 {
    var f = try dir.openFile(io, path, .{});
    defer f.close(io);
    const stat = try f.stat(io);
    const size: usize = std.math.cast(usize, stat.size) orelse return error.FileTooLarge;
    const buf = try gpa.alignedAlloc(u8, .@"8", size);
    errdefer gpa.free(buf);
    var read_buf: [16 * 1024]u8 = undefined;
    var fr = f.reader(io, &read_buf);
    const reader = &fr.interface;
    try reader.readSliceAll(buf);
    return buf;
}

/// Atomic write: spool to `<path>.tmp` in the same directory, fsync the
/// payload, close, then `rename` over the final path. The rename is atomic
/// on POSIX (same filesystem), so a crash mid-write leaves either the
/// previous version intact or no file at all — never a half-written `<path>`.
/// The fsync ensures the bytes are durable on disk before the rename, so a
/// power loss right after rename can't surface a renamed-but-empty inode.
/// On any error path the temp file is deleted via `errdefer`.
fn writeFile(io: std.Io, dir: std.Io.Dir, path: []const u8, bytes: []const u8, gpa: Allocator) !void {
    const tmp_path = try std.fmt.allocPrint(gpa, "{s}.tmp", .{path});
    defer gpa.free(tmp_path);

    // If a previous run died mid-write the stale .tmp may still be there;
    // creating with truncate=true overwrites it, so we don't need to unlink first.
    var f = try dir.createFile(io, tmp_path, .{ .truncate = true });
    var closed = false;
    errdefer {
        if (!closed) f.close(io);
        // Best-effort cleanup of the temp file on any error path so we
        // don't leak `<path>.tmp` artifacts after a failed write.
        dir.deleteFile(io, tmp_path) catch {};
    }

    {
        var write_buf: [16 * 1024]u8 = undefined;
        var fw = f.writer(io, &write_buf);
        const writer = &fw.interface;
        try writer.writeAll(bytes);
        try writer.flush();
    }

    try f.sync(io);
    f.close(io);
    closed = true;

    try dir.rename(tmp_path, dir, path, io);
}

/// Monotonic ns timestamp via posix clock_gettime(CLOCK_MONOTONIC).
/// Zig 0.16 moved std.time.Timer/Instant behind the std.Io abstraction;
/// going direct to libc avoids threading an Io instance through the CLI.
fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 +
        @as(u64, @intCast(ts.nsec));
}

fn msBetween(start_ns: u64, end_ns: u64) f64 {
    return @as(f64, @floatFromInt(end_ns - start_ns)) / 1e6;
}

fn usage(line: []const u8) error{MissingArg} {
    std.debug.print("usage: {s}\n", .{line});
    return error.MissingArg;
}

fn printUsage() void {
    std.debug.print(
        \\tac — Tachiom (arxiv 2604.28142v1) reimplementation in Zig.
        \\
        \\Usage:
        \\  tac index  <tokens.bin> <out.tac> [--kappa N] [--seed N]
        \\                                    [--mu N] [--tau N] [--epsilon N] [--theta N]
        \\  tac search <index.tac> <queries.bin> [--kappa-c N] [--kappa-d N] [--alpha F] [--top-k N]
        \\  tac eval   <index.tac> <queries.bin> <qrels.tsv>
        \\                                       [--metric mrr|success] [--k N]
        \\                                       [--kappa-c N] [--kappa-d N] [--alpha F]
        \\  tac bench  <index.tac> <queries.bin> <qrels.tsv> [--metric mrr|success]
        \\                                       [--out results.csv] [--dataset NAME]
        \\
        \\Defaults: kappa=4096, seed=42, kappa-c=80, kappa-d=1000, top-k=10,
        \\metric=mrr, k=10. `tac bench` runs the paper §6 (κ_c × κ_d × α) sweep.
        \\
        \\qrels.tsv format (any of):
        \\  qid<TAB>doc_id
        \\  qid<TAB>doc_id<TAB>rel
        \\  qid<TAB>iter<TAB>doc_id<TAB>rel        (TREC; iter ignored)
        \\
        \\qids: tac eval/bench reads <queries.bin>.qids (little-endian u32
        \\sidecar) when present; otherwise falls back to 1-indexed sequential
        \\qids matching hand-built smoke fixtures.
        \\
        \\Paper-strict μ=128, τ=256, ε=4, θ=39 (override only when corpus
        \\vocabulary diversity exceeds n_tokens/θ — e.g. Czech multilingual).
        \\
    , .{});
}

test "constants are accessible from main module" {
    try std.testing.expectEqual(@as(u32, 128), tac.constants.TAC_MU);
    try std.testing.expectEqual(@as(u32, 32), tac.constants.PQ_M);
}

test "writeFile: final path contains complete bytes after rename, no .tmp leak" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.testing.io;

    const payload: []const u8 = "TAC_TKN1\x02\x00\x00\x00abcdefghijklmnop";
    try writeFile(io, tmp.dir, "out.bin", payload, std.testing.allocator);

    var f = try tmp.dir.openFile(io, "out.bin", .{});
    defer f.close(io);
    const stat = try f.stat(io);
    try std.testing.expectEqual(@as(u64, payload.len), stat.size);

    var buf: [64]u8 = undefined;
    var read_buf: [128]u8 = undefined;
    var fr = f.reader(io, &read_buf);
    try fr.interface.readSliceAll(buf[0..payload.len]);
    try std.testing.expectEqualSlices(u8, payload, buf[0..payload.len]);

    // After a successful write the temp file must be gone — no leak.
    try std.testing.expectError(
        error.FileNotFound,
        tmp.dir.openFile(io, "out.bin.tmp", .{}),
    );
}

test "writeFile: existing final path is replaced atomically" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.testing.io;

    try writeFile(io, tmp.dir, "out.bin", "old-content-123", std.testing.allocator);
    try writeFile(io, tmp.dir, "out.bin", "new-content-XYZW", std.testing.allocator);

    var f = try tmp.dir.openFile(io, "out.bin", .{});
    defer f.close(io);
    const stat = try f.stat(io);
    try std.testing.expectEqual(@as(u64, 16), stat.size);

    var buf: [16]u8 = undefined;
    var read_buf: [32]u8 = undefined;
    var fr = f.reader(io, &read_buf);
    try fr.interface.readSliceAll(&buf);
    try std.testing.expectEqualSlices(u8, "new-content-XYZW", &buf);
}

test "writeFile: rename failure leaves prior content intact and cleans up tmp" {
    // Simulate a write failure by pre-creating a non-empty directory at the
    // final path: createFile on `<path>.tmp` succeeds but the final rename
    // (regular file over a non-empty directory) fails on POSIX. The
    // errdefer must delete `<path>.tmp` so we don't leak the artifact.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.testing.io;

    // Pre-existing directory at the would-be final path with a sentinel
    // file inside so the rename is unambiguously rejected.
    {
        var blocker = try tmp.dir.createDirPathOpen(io, "out.bin", .{});
        defer blocker.close(io);
        var sentinel = try blocker.createFile(io, "sentinel", .{ .truncate = true });
        defer sentinel.close(io);
        var write_buf: [32]u8 = undefined;
        var sw = sentinel.writer(io, &write_buf);
        try sw.interface.writeAll("do-not-clobber");
        try sw.interface.flush();
    }

    const result = writeFile(io, tmp.dir, "out.bin", "fresh-payload", std.testing.allocator);
    try std.testing.expect(std.meta.isError(result));

    // Sentinel inside the blocking dir must still be there: no clobber,
    // no half-written final artifact.
    var blocker2 = try tmp.dir.openDir(io, "out.bin", .{});
    defer blocker2.close(io);
    var s = try blocker2.openFile(io, "sentinel", .{});
    s.close(io);

    // Temp artifact must be cleaned up despite the rename failure.
    try std.testing.expectError(
        error.FileNotFound,
        tmp.dir.openFile(io, "out.bin.tmp", .{}),
    );
}
