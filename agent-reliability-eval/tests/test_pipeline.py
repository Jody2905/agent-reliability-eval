"""End-to-end: run the RAG system through the real MCP server with fake models (no API key)."""

import asyncio
import json
from pathlib import Path

import pytest

from reliability.llm import FakeLLM, cost_usd
from reliability.run import run
from reliability.scoring import summarize

QUESTIONS = [
    json.loads(line)
    for line in (Path(__file__).parents[1] / "dataset" / "seed_questions.jsonl").read_text().splitlines()
]
BY_QUESTION = {q["question"]: q for q in QUESTIONS}


def oracle(system, messages):
    """Replies with the gold answer, but only after checking the RAG context really was built."""
    prompt = messages[-1]["content"]
    assert prompt.startswith("Records:") and "Answer format:" in prompt
    question = prompt.split("Question: ", 1)[1].split("\n\nAnswer format:", 1)[0]
    q = BY_QUESTION[question]
    return "Answer below.\n" + json.dumps({"answer": q["gold_answer"], "citations": q["gold_source_ids"]})


def run_with(llm, questions=QUESTIONS):
    import reliability.run as run_module

    run_module.fake_llm = lambda: llm  # inject our fake model
    return asyncio.run(run("rag", "fake", questions, use_fake=True))


def test_rag_with_oracle_model_scores_100_percent():
    rows = run_with(FakeLLM(oracle))
    summary = summarize(rows)
    assert summary["accuracy"] == 1.0 and summary["errors"] == 0
    assert all(r["tool_calls"] >= 1 for r in rows)  # retrieval happened through MCP


def test_rag_retrieval_finds_the_right_record_for_a_lookup():
    rows = run_with(FakeLLM(oracle), [q for q in QUESTIONS if q["id"] == "q001"])
    assert "cve:CVE-2021-44228" in rows[0]["prediction"]["retrieved"]


def test_always_abstaining_model_only_gets_unanswerables_right():
    rows = run_with(FakeLLM(lambda s, m: '{"answer": null, "citations": []}'))
    summary = summarize(rows)
    assert summary["by_category"]["unanswerable"] == 1.0
    assert summary["accuracy"] == pytest.approx(2 / 15, abs=1e-3) and summary["wrong_abstention_rate"] == 1.0


def test_cost_tiers():
    assert cost_usd("claude-haiku-5-5", 1_000_000 // 10, 1000) == pytest.approx(0.0105)
    assert cost_usd("claude-haiku-5-5", 200_000, 0) == pytest.approx(0.1)  # long prompt -> higher tier for all tokens
    assert cost_usd("unknown-model", 10, 10) is None
