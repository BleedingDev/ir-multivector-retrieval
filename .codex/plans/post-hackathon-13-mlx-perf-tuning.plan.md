---
name: MLX Encoder Perf Tuning (post-consult)
overview: Plan-12 landed MLX parity (3.47×/4.19× on 1000-doc Jira) but only 1.44× on 100-doc fixture, suggesting fixed costs and sub-optimal hot path. GPT-5.5 high-reasoning consult identified 10 prioritized opportunities. This plan acts on the top 4 highest-ROI/lowest-risk items: fused attention, skip-weight-recast, larger batch + length sort, and fast keep-mask gather. Each rec must keep all 4 parity gates green or be reverted. encode.py untouched. Skip M/L-effort items (#1 remove pylate load, #6 producer thread, #7 mx.compile) — those are a follow-up wave.
todos:
  - id: rec-03-fused-sdpa
    content: "tools/encode_mlx.py BertSelfAttention.__call__ (lines 110-128): replace manual q@k.T → softmax → @v with mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=attn_bias). Keep transpose/reshape + output projection + residual + LN unchanged. Re-run tests/live/mlx_parity.py — all 4 gates must stay green. Bench delta on 100-doc and 1000-doc fixtures, record honestly."
    status: pending
  - id: rec-02-skip-weight-recast
    content: "tools/export_colbert_to_mlx.py: write dtype-specific weight files (mlx_weights.fp16.safetensors when --dtype fp16, mlx_weights.fp32.safetensors when fp32). tools/encode_mlx.py: load the matching dtype file, skip the per-tensor astype loop when cfg dtype matches runtime dtype. Keep an astype fallback path for mixed cases. Re-run parity. Bench delta."
    status: pending
  - id: rec-04-batch-and-sort
    content: "tools/encode_mlx.py: bump default --encoder-batch-size to 128 (fp16) / 64 (fp32). Add --sort-by-length flag with correct output-order restoration (DO NOT copy encode.py's incomplete sort block; write a clean stable-perm + inverse-perm using numpy.argsort + numpy.empty index-restore). Bench on 1000-doc Jira where padding waste actually shows up."
    status: pending
  - id: rec-05-fast-keep-gather
    content: "tools/encode_mlx.py forward_batch (lines 398-417 area): build keep mask on device, gather kept rows once (mx.compress or boolean_mask) before host transfer instead of looping the [B,T,dim] padded tensor in Python. tools/_tokens_bin.py write_tokens_bin: add a fast path that takes a contiguous numpy float32 buffer + per-doc offsets and writes via fp.write(buf.tobytes()) instead of struct.pack-per-float."
    status: pending
  - id: parity-regression-gate
    content: "After EACH rec lands, run tests/live/mlx_parity.py end-to-end. All 4 gates (token-id byte-equal / cosine ≥ 0.998 mean & ≥ 0.99 min / two-run determinism / TAC kappa overlap) must stay green. If any gate fails: revert that rec, document why in LIVE_ENCODER_NOTES, move on."
    status: pending
  - id: bench-and-notes
    content: "Re-bench 100-doc and 1000-doc fixtures after all landed recs. Update tools/LIVE_ENCODER_NOTES.md perf table with new numbers (replace the post-plan-12 row). Honest reporting — if combined wins are <10% on real workload, surface that. Cite which recs delivered which wins."
    status: pending
  - id: commit-and-push
    content: "Granular commits per rec (rec-03, rec-02, rec-04, rec-05 = 4 commits). Final commit: bench/notes update. Push to origin/main. Use git author Petr Glaser <petr@glaser.cz>."
    status: pending
isProject: false
---

# MLX Encoder Perf Tuning (post-consult)

## Execution Notes

Repo: ir-multivector-retrieval. Working directory: /Users/satan/side/experiments/ir-multivector-retrieval.

Source of recommendations: `.codex/plans/post-hackathon-13-mlx-perf-consult.txt` (GPT-5.5 high reasoning, 81k tokens, evidence-grounded with file:line cites).

**Consult's recommended ordering**: 1, 3, 2, 4, 6, 5, 7, 8, 9, 10. We're skipping #1 (remove pylate load, M/L effort, high parity risk), #6 (producer thread, M effort), and #7 (mx.compile, M effort, shape-bucket complexity) for a follow-up wave. We're taking #3, #2, #4, #5 — the S-to-M effort wins that compound.

**Honest impact estimate** (sum of realistic ranges from consult):
- Fused SDPA: 5-12% on 1000-doc, 1-5% on 100-doc
- Skip weight recast: 3-8% on 100-doc (mostly amortized away on 1000-doc)
- Batch + sort: 5-20% on 1000-doc (heterogeneous lengths help here)
- Fast gather + bulk writer: 3-10% on large outputs

Stack favorably: ~15-30% on 1000-doc Jira workload, possibly 10-15% on 100-doc. Wins are NOT additive — some of these compete (gather + writer share the same Amdahl envelope as fused SDPA).

## Constraints

- All 4 parity gates from plan-12 must remain green. tests/live/mlx_parity.py is the regression gate.
- 201/201 zig tests stay green (lane is Python-only; should be trivial).
- encode.py is BYTE-IDENTICAL — no changes to PyTorch baseline.
- Bench numbers must be from the same hardware/state as plan-12's measurements (Apple Silicon, ReleaseFast equivalent for Python = `python -O` is irrelevant; use the same env).
- If a rec fails parity or shows ≤1% wall delta: revert and document. Do not ship dead code.

## Output

- Updated tools/encode_mlx.py with fused SDPA + skip-cast + larger batch defaults + sort-by-length + fast keep-mask gather
- Updated tools/export_colbert_to_mlx.py with dtype-specific weight outputs
- Updated tools/_tokens_bin.py with fast bulk-write path
- Updated tools/LIVE_ENCODER_NOTES.md perf table
- 4-5 commits on origin/main
- All parity gates green, bench numbers reported honestly

DO NOT TOUCH: tools/encode.py, src/**, tests/** (other than re-running mlx_parity.py).
