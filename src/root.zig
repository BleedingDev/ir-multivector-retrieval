//! tac — Tachiom (arxiv 2604.28142v1) reimplementation, public surface.
//!
//! Module map (file ownership in AGENTS.md):
//!   util/   — primitives-engineer
//!   io/     — primitives-engineer
//!   tac/    — clusterer
//!   index/  — indexer
//!   retrieval/, eval/ — retriever
//!   constants.zig, root.zig, main.zig — lead

pub const constants = @import("constants.zig");

pub const util = struct {
    pub const vec = @import("util/vec.zig");
    pub const rng = @import("util/rng.zig");
    pub const alloc = @import("util/alloc.zig");
};

pub const io = struct {
    pub const token_dump = @import("io/token_dump.zig");
};

pub const tac = @import("tac/tac.zig");
pub const kmeans = @import("tac/kmeans.zig");

pub const index = struct {
    pub const pq = @import("index/pq.zig");
    pub const hnsw = @import("index/hnsw.zig");
    pub const inverted_list = @import("index/inverted_list.zig");
    pub const storage = @import("index/storage.zig");
};

pub const retrieval = struct {
    pub const gather = @import("retrieval/gather.zig");
    pub const prune = @import("retrieval/prune.zig");
    pub const refine = @import("retrieval/refine.zig");
};

pub const eval = struct {
    pub const metrics = @import("eval/metrics.zig");
    pub const latency = @import("eval/latency.zig");
};

test {
    // refAllDecls + nested struct namespaces — recursively reach every test block.
    const std = @import("std");
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(util);
    std.testing.refAllDecls(io);
    std.testing.refAllDecls(index);
    std.testing.refAllDecls(retrieval);
    std.testing.refAllDecls(eval);
}
