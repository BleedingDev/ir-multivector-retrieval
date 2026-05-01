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
}
