---
name: Encoder FP16 + Larger Batches
overview: tools/encode.py runs pylate ColBERTv2 on Metal MPS but at FP32 with default batch sizes. The README's optimization-story section names this as the single biggest absolute lever (95% of ingestion cost is encoder; FP16 + bigger batches typically gives 4-6× on Apple Silicon). Add explicit dtype + batch-size flags + sane MPS defaults.
todos:
  - id: fp16-flag
    content: "Add `--dtype {fp32,fp16}` to tools/encode.py. Default fp32 on cpu/cuda, fp16 on mps (since Apple GPU is dramatically faster at fp16). When fp16, call `model.half()` after `pylate.models.ColBERT(...)` and renormalise embeddings to f32 just before writing to tokens.bin (the binary format is f32-only)."
    status: pending
  - id: batch-size-flag
    content: "Expose `--encoder-batch-size N` (separate from the existing pipeline batching). Default 32 today via pylate; bump to 64 on cpu / 256 on mps. Larger batches saturate the GPU better."
    status: pending
  - id: warmup
    content: "Add a 4-doc warmup pass before timing — Metal compiles kernels on first use. Without warmup, the first batch dominates the wall-clock and skews bench numbers."
    status: pending
  - id: live-bench
    content: "Run `python tools/encode.py --mode docs --docs <100-doc-fixture> --device mps --dtype fp16 --encoder-batch-size 256` and record wall time vs the existing fp32-default path on the same fixture. Expect 4-6× speedup. Update README + tools/LIVE_ENCODER_NOTES.md."
    status: pending
  - id: parity-check
    content: "Cosine similarity check: encode the same 10 docs with fp32 and fp16, verify avg cosine ≥ 0.998 between corresponding token vectors. If parity drops below 0.99 we revisit (recall would too)."
    status: pending
  - id: commit
    content: "Commit + push to origin/main. Update tools/LIVE_ENCODER_NOTES.md with the new defaults and the parity number."
    status: pending
isProject: false
---

# Encoder FP16 + Larger Batches

## Execution Notes

Repo: ir-multivector-retrieval. Working directory: /Users/satan/side/experiments/ir-multivector-retrieval.

Current state:
- `tools/encode.py` uses pylate's high-level ColBERT with default args. No `--dtype` flag, no explicit `--encoder-batch-size`.
- README's optimization-story names FP16 + bigger batches as a 4-6× lever on the 48-min full-corpus encode.
- Sibling ir-expo's `services/warp-service/server.py` already runs `WARP_ENCODER_FP16=1` (default) + `WARP_ENCODER_BATCH_SIZE=256` defaults — proven on the same Apple Silicon machine. Replicate that defaulting here.

Why this is high-value:
- Encoder is 95% of total ingestion wall-clock for any user re-encoding their corpus.
- The existing Wave 1 work in ir-expo nailed the inference path on Apple Silicon — just port the defaults to the standalone tools/encode.py.
- No code in the Zig pipeline cares about encoder dtype since residuals get .cpu().float()-renormalised at write time anyway.

Numerical concern (FP16 parity): ColBERT vectors are L2-normalised; FP16 introduces small rounding error per coordinate. Empirically the cosine similarity between FP32 and FP16 outputs stays >0.998 on real text. The parity-check todo gates this at 0.99.

## Constraints

- Default behavior unchanged: `--dtype fp32 --encoder-batch-size 32` reproduces today's exact output. Only opt-in flags get the speedup.
- New defaults kick in only when `--device mps`. CPU-side stays FP32 (FP16 on x86 CPU is *slower* than FP32, not faster).
- The on-disk tokens.bin format stays f32 — no version bump needed.

## Output

`tools/encode.py` with `--dtype` + `--encoder-batch-size` flags. `tools/LIVE_ENCODER_NOTES.md` updated with new defaults + measured speedup + parity number. README perf row added if the speedup is the headline.
