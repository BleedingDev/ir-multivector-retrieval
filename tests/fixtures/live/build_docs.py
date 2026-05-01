#!/usr/bin/env python3
"""Generate a 100-doc English JSONL with deliberate cross-doc vocabulary
overlap so paper §3 TAC has actual aliasing to find.

The corpus is a mix of three small thematic clusters (cooking, weather,
software releases) — each draws from a vocabulary that overlaps heavily
with the others on stopwords ("the", "of", "and", "is", "with", "to") and
on a handful of mid-frequency content words ("recipe", "release",
"version", "report", "weather", "system"). That guarantees ≫50% of
distinct vocab IDs end up with n_j ≥ 2 across the corpus.

Output: tests/fixtures/live/docs.jsonl (one {"doc_id", "text"} per line).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent / "docs.jsonl"

TEMPLATES_COOK = [
    "The recipe of {dish} is simple and uses {ing1} with {ing2}.",
    "Today the {dish} recipe was tested and the report shows good results.",
    "Cooking {dish} with {ing1} and {ing2} is the standard recipe.",
    "A new version of the {dish} recipe replaces {ing1} with {ing2}.",
]
DISHES = ["soup", "stew", "curry", "pasta", "salad", "bread", "rice", "porridge"]
INGS = ["onion", "garlic", "pepper", "tomato", "carrot", "cumin", "basil", "thyme"]

TEMPLATES_WEATHER = [
    "The weather report says it is sunny with light wind in the morning.",
    "Today the weather is cold and the system reports rain by evening.",
    "A storm system is moving over the bay and the weather is shifting.",
    "The morning weather is foggy with the wind picking up by noon.",
]

TEMPLATES_RELEASE = [
    "The new release of the system version {ver} fixes many bugs.",
    "Version {ver} of the package replaces the old version with a faster build.",
    "The release notes of version {ver} mention the recipe of the new index.",
    "A patch release of version {ver} of the system is ready and tested.",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-docs", type=int, default=100)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    rng = random.Random(args.seed)
    docs = []
    for i in range(args.n_docs):
        bucket = i % 3
        if bucket == 0:
            text = rng.choice(TEMPLATES_COOK).format(
                dish=rng.choice(DISHES),
                ing1=rng.choice(INGS),
                ing2=rng.choice(INGS),
            )
        elif bucket == 1:
            text = rng.choice(TEMPLATES_WEATHER)
        else:
            text = rng.choice(TEMPLATES_RELEASE).format(
                ver=f"{rng.randint(1, 5)}.{rng.randint(0, 9)}.{rng.randint(0, 9)}",
            )
        docs.append({"doc_id": f"doc-{i:03d}", "text": text})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"wrote {len(docs)} docs to {OUT}")


if __name__ == "__main__":
    main()
