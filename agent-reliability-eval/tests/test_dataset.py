"""The dataset is reproducible and balanced."""

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

DATASET = Path(__file__).parents[1] / "dataset"


def load(name):
    return [json.loads(l) for l in (DATASET / name).read_text(encoding="utf-8").splitlines() if l.strip()]


def test_build_is_deterministic(tmp_path):
    """Rebuilding from the snapshot gives byte-identical questions."""
    before = (DATASET / "questions.jsonl").read_bytes()
    subprocess.run([sys.executable, str(DATASET / "build.py")], check=True, capture_output=True)
    assert (DATASET / "questions.jsonl").read_bytes() == before


def test_balance():
    items = load("questions.jsonl")
    assert len(items) == 150
    assert Counter(i["category"] for i in items) == {"single_hop": 40, "multi_hop": 40, "comparison": 40, "unanswerable": 30}
    assert len({i["question"] for i in items}) == 150


def test_seed_questions_are_the_first_fifteen():
    full, seed = load("questions.jsonl"), load("seed_questions.jsonl")
    assert [q["question"] for q in full[:15]] == [q["question"] for q in seed]
