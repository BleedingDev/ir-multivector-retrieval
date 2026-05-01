//! src/io/token_dump.zig — flat-binary token-dump reader (mmap-backed).
//!
//! Owner: primitives-engineer.
//! Format spec: docs/token-dump-format.md (created by primitives-engineer).
//! See plan 01-primitives-and-io.plan.md.
//!
//! Required surface:
//!   pub const TokenDump = struct {
//!       dim: u32,
//!       n_docs: u64,
//!       n_tokens: u64,
//!       doc_offsets: []const u64,   // CSR-style, len = n_docs + 1
//!       token_ids:   []const u32,
//!       vectors:     []const f32,   // n_tokens * dim
//!   };
//!   pub fn open(path: []const u8) !TokenDump;
//!   pub fn close(td: *TokenDump) void;

const std = @import("std");

test "placeholder" {
    try std.testing.expect(true);
}
