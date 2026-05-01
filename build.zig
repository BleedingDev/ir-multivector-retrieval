const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Public library module — re-exports everything.
    const lib_mod = b.addModule("tac", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Executable module wires the CLI to the library.
    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe_mod.addImport("tac", lib_mod);

    const exe = b.addExecutable(.{
        .name = "tac",
        .root_module = exe_mod,
    });
    b.installArtifact(exe);

    // `zig build run -- <args>`
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("run", "Run the tac binary");
    run_step.dependOn(&run_cmd.step);

    // `zig build test` — runs every test block reachable from root.zig + main.zig.
    const lib_tests = b.addTest(.{ .root_module = lib_mod });
    const run_lib_tests = b.addRunArtifact(lib_tests);

    const exe_tests = b.addTest(.{ .root_module = exe_mod });
    const run_exe_tests = b.addRunArtifact(exe_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_tests.step);
    test_step.dependOn(&run_exe_tests.step);

    // Live tests gate Python/torch/pylate-dependent harnesses behind a build
    // option so the default `zig build test` stays Python-free for any
    // contributor who hasn't provisioned tools/.venv. Enable explicitly:
    //
    //   zig build test -Dlive=true
    //
    // Live test sources live under tests/live/ and import the public `tac`
    // module just like in-tree unit tests. Owner: primitives-engineer (task #24).
    const live = b.option(
        bool,
        "live",
        "Include live torch+pylate-dependent tests under tests/live/",
    ) orelse false;
    if (live) {
        const live_mod = b.createModule(.{
            .root_source_file = b.path("tests/live/vocab_aliasing.zig"),
            .target = target,
            .optimize = optimize,
        });
        live_mod.addImport("tac", lib_mod);
        const live_tests = b.addTest(.{ .root_module = live_mod });
        const run_live_tests = b.addRunArtifact(live_tests);
        test_step.dependOn(&run_live_tests.step);
    }

    // Per-dataset benchmark executables (paper §6 Table 1 reproduction harness).
    // Each consumes pre-encoded tokens.bin + qrels.tsv at runtime; not part of
    // `zig build test`. Invoke with e.g. `zig build run-bench_msmarco -- <args>`.
    // .zig sources are retriever-owned; build.zig wiring is lead-owned (#28).
    inline for (.{
        .{ "bench_msmarco", "benchmarks/msmarco_v1.zig" },
        .{ "bench_lotte", "benchmarks/lotte_pooled.zig" },
    }) |entry| {
        const bench_mod = b.createModule(.{
            .root_source_file = b.path(entry[1]),
            .target = target,
            .optimize = optimize,
        });
        bench_mod.addImport("tac", lib_mod);
        const bench_exe = b.addExecutable(.{
            .name = entry[0],
            .root_module = bench_mod,
        });
        b.installArtifact(bench_exe);
        const run_bench = b.addRunArtifact(bench_exe);
        run_bench.step.dependOn(b.getInstallStep());
        if (b.args) |args| run_bench.addArgs(args);
        const bench_step = b.step("run-" ++ entry[0], "Run the " ++ entry[0] ++ " harness");
        bench_step.dependOn(&run_bench.step);
    }
}
