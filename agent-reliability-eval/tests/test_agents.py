"""Agent loops tested offline with scripted fake models, through the real MCP server."""

import asyncio
import json
from pathlib import Path

from reliability.llm import FakeLLM
from reliability.run import run
from reliability.scoring import summarize
from reliability.systems import agent, agent_verify

QUESTIONS = [
    json.loads(line)
    for line in (Path(__file__).parents[1] / "dataset" / "questions.jsonl").read_text().splitlines()
]
BY_ID = {q["id"]: q for q in QUESTIONS}
THINKING = {"type": "thinking", "thinking": "Let me look this up.", "signature": "sig-123"}


def question_of(messages):
    first = messages[0]["content"]
    text = first.split("Question: ", 1)[1].split("\n\nAnswer format:", 1)[0]
    return next(q for q in QUESTIONS if q["question"] == text)


def tool_results(messages):
    return [b for m in messages if m["role"] == "user" and isinstance(m["content"], list)
            for b in m["content"] if b.get("type") == "tool_result"]


def final(answer, citations):
    return [{"type": "text", "text": json.dumps({"answer": answer, "citations": citations})}]


def run_system(system, llm, ids):
    import reliability.run as run_module

    run_module.fake_llm = lambda: llm
    return asyncio.run(run(system, "fake", [BY_ID[i] for i in ids], use_fake=True))


# ---- System 2 ---------------------------------------------------------------

def oracle_agent(system, messages, tools):
    """Fetch each gold source with get_source, then answer with the gold answer."""
    q = question_of(messages)
    results = tool_results(messages)
    if not results and q["gold_source_ids"]:
        assert tools, "agent must be offered tools"
        calls = [{"type": "tool_use", "id": f"t{i}", "name": "get_source", "input": {"source_id": s}}
                 for i, s in enumerate(q["gold_source_ids"][:3])]
        return [THINKING, *calls]
    # The thinking block from the tool-calling turn must come back unchanged.
    if q["gold_source_ids"]:
        assert messages[1]["content"][0] == THINKING
        assert all(json.loads(r["content"])["found"] for r in results)
    return final(q["gold_answer"], q["gold_source_ids"])


def test_agent_with_oracle_model_scores_100_percent():
    rows = run_system("agent", FakeLLM(oracle_agent), list(BY_ID))
    assert summarize(rows)["accuracy"] == 1.0
    answerable = [r for r in rows if r["category"] != "unanswerable"]
    assert all(r["tool_calls"] >= 1 and r["prediction"]["tool_rounds"] == 1 for r in answerable)


def test_agent_sees_tool_errors_and_recovers():
    def model(system, messages, tools):
        results = tool_results(messages)
        if not results:
            return [{"type": "tool_use", "id": "a", "name": "lookup_cve", "input": {"cve_id": "Log4Shell"}}]
        if len(results) == 1:
            assert results[0].get("is_error") and "not a valid CVE id" in results[0]["content"]
            return [{"type": "tool_use", "id": "b", "name": "lookup_cve", "input": {"cve_id": "CVE-2021-44228"}}]
        return final(json.loads(results[1]["content"])["cvss"]["base_score"], ["cve:CVE-2021-44228"])

    row = run_system("agent", FakeLLM(model), ["q001"])[0]
    assert row["score"]["correct"] and row["tool_calls"] == 2
    assert [entry["error"] for entry in row["tool_log"]] == [True, False]


def test_step_budget_stops_a_runaway_agent():
    offered = []

    def model(system, messages, tools):
        offered.append(tools is not None)
        if tools:
            return [{"type": "tool_use", "id": f"s{len(offered)}", "name": "search_corpus", "input": {"query": "log4j"}}]
        assert messages[-1]["content"][-1]["text"] == agent.BUDGET_MESSAGE
        return final(None, [])

    row = run_system("agent", FakeLLM(model), ["q001"])[0]
    assert row["prediction"]["budget_exhausted"] and row["tool_calls"] == agent.MAX_STEPS
    assert offered == [True] * agent.MAX_STEPS + [False]  # tools switched off for the last call


# ---- System 3 ---------------------------------------------------------------

def fact_checking_verifier(messages):
    """Compares a claimed CVSS score with the score in the cited CVE record."""
    prompt = messages[0]["content"]
    claimed = json.loads(prompt.split("Proposed answer: ", 1)[1].split("\n\n", 1)[0])
    record = json.loads(prompt.split("[cve:CVE-2020-1472]\n", 1)[1].split("\n\n", 1)[0])
    if claimed["cvss"] != record["cvss"]["base_score"]:
        return json.dumps({"supported": False, "problems": [
            f"The record gives CVSS {record['cvss']['base_score']}, not {claimed['cvss']}."]})
    return '{"supported": true, "problems": []}'


def test_verifier_catches_an_answer_from_memory():
    """The classic failure: the agent 'knows' Zerologon is 10.0. The verifier makes it check the record."""

    def model(system, messages, tools):
        if system == agent_verify.VERIFIER_PROMPT:
            return fact_checking_verifier(messages)
        feedback = [m for m in messages if m["role"] == "user" and isinstance(m["content"], str)
                    and "reviewer" in m["content"]]
        if not feedback:
            return final({"cve_id": "CVE-2020-1472", "cvss": 10.0, "severity": "CRITICAL"}, ["cve:CVE-2020-1472"])
        assert "The record gives CVSS 5.5" in feedback[0]["content"]
        return final({"cve_id": "CVE-2020-1472", "cvss": 5.5, "severity": "MEDIUM"}, ["cve:CVE-2020-1472"])

    row = run_system("agent_verify", FakeLLM(model), ["q009"])[0]
    p = row["prediction"]
    assert row["score"]["correct"] and p["revisions"] == 1 and p["accepted_by_verifier"]
    assert p["verification_rounds"][0]["answer"]["cvss"] == 10.0  # the wrong draft is kept for analysis


def test_fabricated_citation_is_caught_without_a_model_call():
    verifier_calls = []

    def model(system, messages, tools):
        if system == agent_verify.VERIFIER_PROMPT:
            verifier_calls.append(1)
            return '{"supported": true, "problems": []}'
        if any(isinstance(m["content"], str) and "does not exist" in m["content"] for m in messages):
            return final(10, ["cve:CVE-2021-44228"])
        return final(10, ["cve:CVE-2021-99999"])

    row = run_system("agent_verify", FakeLLM(model), ["q001"])[0]
    assert row["score"]["correct"] and row["prediction"]["revisions"] == 1
    assert row["citations"]["fabricated"] == []
    assert len(verifier_calls) == 1  # the fabricated round was rejected by code, not by the verifier


def test_revisions_are_capped():
    def model(system, messages, tools):
        if system == agent_verify.VERIFIER_PROMPT:
            return '{"supported": false, "problems": ["Not supported."]}'
        return final(9.9, ["cve:CVE-2021-44228"])

    row = run_system("agent_verify", FakeLLM(model), ["q001"])[0]
    p = row["prediction"]
    assert p["revisions"] == agent_verify.MAX_REVISIONS and not p["accepted_by_verifier"]
    assert row["score"]["outcome"] == "wrong"


def test_abstentions_skip_the_verifier():
    def model(system, messages, tools):
        assert system != agent_verify.VERIFIER_PROMPT
        return final(None, [])

    row = run_system("agent_verify", FakeLLM(model), ["q015"])[0]
    assert row["score"]["correct"] and row["prediction"]["revisions"] == 0
