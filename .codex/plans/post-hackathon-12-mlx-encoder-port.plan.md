---
name: MLX Encoder Port (parity-tested)
overview: tools/encode.py uses pylate ColBERTv2 on PyTorch MPS. Apple's MLX framework is native to Apple Silicon and frequently 2-4× faster on the same model class. Port the encoder forward path to MLX as a NEW file (`tools/encode_mlx.py`) — keep encode.py untouched. Land alongside an extensive parity test suite that proves byte-equal token_ids and ≥0.998 cosine similarity per token-row vs the PyTorch path, on a fixture corpus. Only when parity is rock-solid does the README mention MLX as an option.
todos:
  - id: dependency-add
    content: "Add `mlx-lm` and `mlx` to tools/requirements.txt (pinned versions). Document in tools/LIVE_ENCODER_NOTES.md the new venv setup. Verify `import mlx.core` + `import mlx_lm` work in tools/.venv."
    status: pending
  - id: weight-export
    content: "Write `tools/export_colbert_to_mlx.py` — loads ColBERTv2.0 (or jina-colbert-v2-64) via pylate, walks the state_dict, saves MLX-compatible weights to `tools/.cache/<model>.mlx.safetensors`. Reuse safetensors so HF's huggingface_hub can read on either side. ColBERT = BERT base + a final linear projection layer; both have direct MLX equivalents."
    status: pending
  - id: forward-port
    content: "tools/encode_mlx.py — implement the forward pass using mlx.nn building blocks: token embeddings → BERT encoder layers (multi-head attention + FFN) → final linear projection → L2 normalization. Reuse HuggingFace AutoTokenizer for the tokenize step (MLX has no native tokenizer; HF runs on CPU and is fast enough). Output shape must match pylate exactly."
    status: pending
  - id: skiplist-mask
    content: "Reproduce the pylate skiplist + attention-mask combination at parity. Read `pylate.models.colbert.encode` lines ~688-720 and mirror the keep-mask construction byte-for-byte. The token_ids[] in tokens.bin must be byte-identical to encode.py's output for the same input — token IDs are deterministic from the tokenizer."
    status: pending
  - id: tokens-bin-output
    content: "Reuse the existing tokens.bin format (TAC_TKN1 v2). Write the same header layout, doc_offsets CSR, token_ids u32, vectors f32. Borrow the writer code from encode.py — refactor to a small library if needed (`tools/_tokens_bin.py`) so both encoders share the I/O path."
    status: pending
  - id: parity-fixture
    content: "Make tests/fixtures/live/mlx_parity_docs.jsonl with 50 mixed-language docs (the existing 100-doc fixture is fine to reuse if it has parity-friendly content)."
    status: pending
  - id: parity-test-tokenids
    content: "tests/live/mlx_parity.py — encode the fixture with both encode.py (--device mps --dtype fp16) and encode_mlx.py (--dtype fp16), parse both tokens.bin files, assert `token_ids` arrays are BYTE-EQUAL across the two outputs. (HF tokenizer is deterministic; this is a hard equality check, not approx.)"
    status: pending
  - id: parity-test-embeddings
    content: "Same harness: for each token row, compute cosine similarity between the corresponding pylate row and MLX row. Assert per-token cosine ≥ 0.998. Min cosine across all tokens must be ≥ 0.99 (tighter than the FP16 internal parity floor since both are FP16). Report mean + min + p1 cosine in the test output."
    status: pending
  - id: parity-test-downstream
    content: "Bigger gate: encode the same 100 docs with both, run TAC clustering on each token_dump (`tac.clusterFlat` from Zig — call via the test as a subprocess or via a small benchmark binary). Assert `kappa_per_token` arrays match (same vocab id → same cluster budget) and centroid means agree within 1e-3 cosine. This is the 'does the downstream pipeline see the same thing?' check that the user explicitly asked for."
    status: pending
  - id: parity-test-determinism
    content: "Run the MLX encoder twice on the same input — assert byte-equal output (or at most 1 ULP per float). MLX defaults are deterministic on Apple Silicon; if not, surface the env vars needed to make it so."
    status: pending
  - id: parity-test-build-flag
    content: "Wire all parity tests behind a `--mlx-parity` flag analogous to `-Dlive=true` so the default `zig build test` and pytest don't depend on MLX (different machines may not have it). New test command: `tools/.venv/bin/python tests/live/mlx_parity.py` runs everything end-to-end."
    status: pending
  - id: bench
    content: "Run encode_mlx.py on the existing 100-doc fixture and the 1000-doc Jira subset that encode-fp16 used. Record wall time for both vs the encode.py FP16 baseline (already measured: 100-doc 9.6s, 1000-doc 90.2s). Honest reporting — if MLX delivers <1.5× over PyTorch FP16 on this hardware/model, surface that and don't claim 2-4× anywhere user-visible."
    status: pending
  - id: readme-update
    content: "tools/LIVE_ENCODER_NOTES.md gets a new section 'MLX path' with: setup instructions (deps, weight export), parity test results (token_ids equal, cosine numbers, downstream cluster overlap), measured speedup (honestly), and recommendation (when to use MLX vs PyTorch). README's main perf table is unchanged unless MLX is genuinely headline-worthy."
    status: pending
  - id: commit
    content: "Commit + push to origin/main. Granular commits OK: deps + export script (1 commit), encode_mlx.py + shared writer (1 commit), parity tests (1 commit), README/notes (1 commit). Use git author Petr Glaser <petr@glaser.cz>."
    status: pending
isProject: false
---

# MLX Encoder Port (parity-tested)

## Execution Notes

Repo: ir-multivector-retrieval. Working directory: /Users/satan/side/experiments/ir-multivector-retrieval.

This is a **substantial lane** (~1 day of careful work) and **the user explicitly asked for heavy test coverage**. Treat parity testing as the contract — implement it FIRST against a stub MLX encoder, then build out the real encoder until parity passes. That's the right order for a port.

Approach pyramid:

1. **Token-id parity** is the cheap gate. HF tokenizer is fully deterministic; if MLX path emits different token_ids, the bug is upstream of any model code (probably in how we wrap `model.tokenize`).
2. **Per-token cosine ≥0.998** is the meaningful gate. FP16 introduces small numerical error; ColBERT's L2-normalized output collapses most of that. Below 0.99 = real bug, not numerical drift.
3. **Downstream cluster overlap** is the user-asked gate. Two encoders are "equivalent" iff TAC produces the same clustering on the same corpus. If kappa_per_token differs, something about the fp16/MLX rounding is shifting the spread enough to matter.

What MLX is good at:
- Native Metal Performance Shaders, optimized for the unified memory architecture
- No CPU↔GPU transfer overhead (the per-tensor `.cpu()` syncs that bit us in Wave 1 don't exist)
- Smaller graph compile + execution overhead than PyTorch eager mode

What MLX struggles with:
- No tokenizer (we use HF's, fine)
- Smaller op coverage than PyTorch — some BERT-specific ops may need manual workarounds
- Determinism story is good but not universal — pin RNG seeds and check.

## Constraints

- DO NOT modify or remove existing tools/encode.py — port is **additive** until parity is proven.
- `tokens.bin` format is FROZEN at v2; both encoders must produce identical-format output.
- All parity tests gated behind a flag so the default test path doesn't require MLX.
- 189 existing zig tests stay green.
- If parity fails at any of the 4 gates (token-ids / cosine / determinism / cluster), DO NOT claim victory. Report the failure clearly + leave encode_mlx.py marked experimental.

## Output

`tools/encode_mlx.py` (new), `tools/export_colbert_to_mlx.py` (new), `tools/_tokens_bin.py` (new — shared writer extracted from encode.py for reuse), `tests/live/mlx_parity.py` (new), updates to `tools/requirements.txt` and `tools/LIVE_ENCODER_NOTES.md`. encode.py untouched.
