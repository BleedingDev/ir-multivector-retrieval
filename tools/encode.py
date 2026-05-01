#!/usr/bin/env python3
"""tools/encode.py — ColBERTv2 → token-dump binary.

Owner: primitives-engineer.
See plan 01-primitives-and-io.plan.md and docs/token-dump-format.md (to be written).

Usage (target):
    python tools/encode.py \
        --docs path/to/docs.jsonl \
        --model colbert-ir/colbertv2.0 \
        --out  data/msmarco_v1/tokens.bin

Reads docs.jsonl with {"doc_id": ..., "text": ...} per line, encodes each doc
with ColBERTv2 (via pylate), and writes the flat binary specified in
docs/token-dump-format.md plus a sidecar tokens.meta.json.
"""

# TODO(primitives-engineer): implement after format spec is finalised in
# docs/token-dump-format.md.

if __name__ == "__main__":
    raise SystemExit("encode.py not yet implemented — see plan 01-primitives-and-io.")
