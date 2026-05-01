//! tac — CLI entry point.
//!
//! Subcommands:
//!   index   Build a Tachiom index from a tokens.bin file
//!   search  Run queries through the gather→prune→refine pipeline
//!
//! Lead-owned. Per-dataset benchmark binaries (bench_msmarco / bench_lotte)
//! live under benchmarks/ and are produced by the same `zig build install`.

const std = @import("std");
const tac = @import("tac");

const Allocator = std.mem.Allocator;

pub fn main(m: std.process.Init.Minimal) !void {
    var gpa_state: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();

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
    try writeFile(io, cwd, op, image.bytes);
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

fn writeFile(io: std.Io, dir: std.Io.Dir, path: []const u8, bytes: []const u8) !void {
    var f = try dir.createFile(io, path, .{ .truncate = true });
    defer f.close(io);
    var write_buf: [16 * 1024]u8 = undefined;
    var fw = f.writer(io, &write_buf);
    const writer = &fw.interface;
    try writer.writeAll(bytes);
    try writer.flush();
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
        \\  tac index <tokens.bin> <out.tac> [--kappa N] [--seed N]
        \\                                   [--mu N] [--tau N] [--epsilon N] [--theta N]
        \\  tac search <index.tac> <queries.bin> [--kappa-c N] [--kappa-d N] [--alpha F] [--top-k N]
        \\
        \\Defaults: kappa=4096, seed=42, kappa-c=80, kappa-d=1000, top-k=10.
        \\Paper-strict μ=128, τ=256, ε=4, θ=39 (override only when corpus
        \\vocabulary diversity exceeds n_tokens/θ — e.g. Czech multilingual).
        \\
    , .{});
}

test "constants are accessible from main module" {
    try std.testing.expectEqual(@as(u32, 128), tac.constants.TAC_MU);
    try std.testing.expectEqual(@as(u32, 32), tac.constants.PQ_M);
}
