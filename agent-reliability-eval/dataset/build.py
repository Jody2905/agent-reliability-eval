"""Build the full question set: the 15 hand-written seed questions plus templated ones.

    python dataset/build.py

Writes dataset/questions.jsonl (and refreshes seed_questions.jsonl). Sampling uses
a fixed random seed per template, so the output is identical on every run, and
adding a new template doesn't reshuffle the others.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # allow running as a script

import build_seed  # noqa: E402
from templates import TEMPLATES  # noqa: E402

SEED = 2905
HERE = Path(__file__).parent
FIELD_ORDER = [
    "id", "category", "difficulty", "template", "params", "question", "answer_format", "answer_type",
    "gold_answer", "acceptable_answers", "gold_source_ids", "min_tool_calls", "tools_expected", "notes", "snapshot",
]


def build() -> list[dict]:
    store = build_seed.store
    items = build_seed.seed_items()
    seen = {item["question"] for item in items}
    for name, template in TEMPLATES.items():
        rng = random.Random(f"{SEED}:{name}")
        made = 0
        for params in template.pick(store, rng):
            if made == template.count:
                break
            item = {"acceptable_answers": [], "notes": "", **template.build(store, params)}
            if item["question"] in seen:
                continue
            seen.add(item["question"])
            item.update(category=template.category, difficulty=template.difficulty, template=name, params=params,
                        min_tool_calls=template.min_tool_calls, tools_expected=template.tools_expected,
                        snapshot=store.manifest["kev_catalog_version"])
            items.append(item)
            made += 1
        if made < template.count:
            print(f"warning: {name} made {made}/{template.count} questions (not enough candidates)")
    for n, item in enumerate(items, start=1):
        item["id"] = f"q{n:03d}"
    return [{k: item[k] for k in FIELD_ORDER} for item in items]


def main():
    items = build()
    build_seed.main()
    with open(HERE / "questions.jsonl", "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(items)} questions to {HERE / 'questions.jsonl'}\n")
    table = Counter((i["category"], i["difficulty"]) for i in items)
    print(f"{'category':14}{'easy':>6}{'medium':>8}{'hard':>6}{'total':>7}")
    for cat in ("single_hop", "multi_hop", "comparison", "unanswerable"):
        row = [table[(cat, d)] for d in ("easy", "medium", "hard")]
        print(f"{cat:14}{row[0]:>6}{row[1]:>8}{row[2]:>6}{sum(row):>7}")


if __name__ == "__main__":
    main()
