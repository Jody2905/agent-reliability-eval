import json
from pathlib import Path

import pytest

from reliability.common import parse_reply
from reliability.scoring import match_set, match_value, score_answer, score_citations

QUESTIONS = {
    q["id"]: q
    for q in map(json.loads, (Path(__file__).parents[1] / "dataset" / "seed_questions.jsonl").read_text().splitlines())
}


def pred(answer, citations=(), parse_error=False):
    return {"answer": answer, "citations": list(citations), "parse_error": parse_error}


# ---- parse_reply ----------------------------------------------------------

def test_parse_reply_with_prose_and_fence():
    text = 'The score is 10.\n```json\n{"answer": 10, "citations": ["cve:CVE-2021-44228"]}\n```'
    assert parse_reply(text) == {"answer": 10, "citations": ["cve:CVE-2021-44228"], "parse_error": False}


def test_parse_reply_takes_the_last_json_object():
    text = 'Example: {"answer": 1}. Final: {"answer": {"vendor": "X", "kev_entry_count": 2}, "citations": []}'
    assert parse_reply(text)["answer"] == {"vendor": "X", "kev_entry_count": 2}


def test_parse_reply_without_json():
    assert parse_reply("I don't know.")["parse_error"] is True


# ---- value matching -------------------------------------------------------

@pytest.mark.parametrize("gold,got,ok", [
    (10, "10.0", True), (10, 9.8, False), (5.5, "5.5 (MEDIUM)", True),
    ("2023-06-02", "June 2, 2023", True), ("2023-06-02", "2023-06-03", False),
    (False, "No", True), (True, "false", False),
    ("Fortinet", "fortinet", True), ("CVE-2021-26855", "cve-2021-26855", True),
])
def test_match_value(gold, got, ok):
    assert match_value(gold, got) is ok


def test_match_set_rules():
    assert match_set(["initial-access"], ["Initial Access"], ["Initial Access"])
    assert match_set(["CWE-89"], "SQL injection", ["SQL injection"])
    assert not match_set(["CWE-89"], ["CWE-89", "CWE-20"], [])  # extra, unsupported item
    assert not match_set(["CWE-89"], [], [])


# ---- outcomes -------------------------------------------------------------

def test_hallucination_vs_abstention():
    assert score_answer(QUESTIONS["q015"], pred("T1190"))["outcome"] == "hallucinated"
    assert score_answer(QUESTIONS["q015"], pred(None))["outcome"] == "correct"
    assert score_answer(QUESTIONS["q001"], pred(None))["outcome"] == "abstained"


def test_memory_answer_for_zerologon_gets_partial_credit_only():
    result = score_answer(QUESTIONS["q009"], pred({"cve_id": "CVE-2020-1472", "cvss": 10.0, "severity": "CRITICAL"}))
    assert result["outcome"] == "wrong" and result["partial"] == pytest.approx(1 / 3)


def test_citation_scores_and_fabrication():
    c = score_citations(QUESTIONS["q001"], pred(10, ["cve:CVE-2021-44228", "kev:CVE-2021-44228", "cve:CVE-9999-0001"]))
    assert c["citation_recall"] == 1.0
    assert c["citation_precision"] == pytest.approx(1 / 3)
    assert c["fabricated"] == ["cve:CVE-9999-0001"]


FULL_SET = [json.loads(l) for l in (Path(__file__).parents[1] / "dataset" / "questions.jsonl").read_text().splitlines()]


def test_gold_answers_score_perfectly():
    """Feeding back each gold answer and its sources must score 100% - dataset and scorer agree."""
    for q in FULL_SET:
        assert score_answer(q, pred(q["gold_answer"], q["gold_source_ids"]))["correct"], q["id"]
        c = score_citations(q, pred(q["gold_answer"], q["gold_source_ids"]))
        assert c["fabricated"] == [] and c["citation_recall"] in (None, 1.0), q["id"]
