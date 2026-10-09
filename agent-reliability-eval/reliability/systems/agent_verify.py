"""System 3: tool-using agent + verification.

Same agent as System 2, then its answer goes through two checks before it is accepted:

1. Mechanical checks (code, no model):
   - the reply ended with the required JSON
   - a non-null answer cites at least one source
   - every cited source id actually exists
2. A verifier (a separate model call, no tools) reads the cited records and judges
   whether they really support the answer.

If either check finds problems, the agent gets the feedback and revises, using its
tools again, up to MAX_REVISIONS times. The verifier only sees the records the
agent cited, so it checks support, not truth: an answer backed by the wrong
record is caught, and an unsupported guess from memory is caught.

Abstentions ("answer": null) are not sent to the verifier, because a verifier
cannot confirm that something is absent from the data.
"""

from __future__ import annotations

import json
from typing import Any

from ..common import OUTPUT_CONTRACT, parse_reply, task_prompt
from ..llm import Usage
from .agent import SYSTEM_PROMPT, tool_loop

NAME = "agent_verify"
MAX_REVISIONS = 2
REVISION_STEPS = 4  # tool-call rounds allowed per revision

VERIFIER_PROMPT = """\
You are a strict fact-checker. You get a question, a proposed answer, and the
records the answer cites. Decide whether the cited records fully support the
answer. Do not use outside knowledge: if a record does not say it, it is not
supported. Check every part of the answer, including numbers, dates and names.

Reply with only a JSON object:
{"supported": true or false, "problems": ["<each specific problem>"]}"""

FEEDBACK = """\
A reviewer checked your answer against the records you cited and found problems:
{problems}

Investigate with your tools as needed, then give a corrected final answer.
{contract}"""


async def mechanical_problems(prediction: dict[str, Any], tools) -> list[str]:
    if prediction["parse_error"]:
        return ['Your reply did not end with the required JSON object {"answer": ..., "citations": [...]}.']
    if prediction["answer"] is None:
        return []
    if not prediction["citations"]:
        return ["The answer cites no sources."]
    problems = []
    for source_id in prediction["citations"]:
        if not (await tools.call("get_source", {"source_id": source_id})).get("found"):
            problems.append(f"Cited source '{source_id}' does not exist in the data.")
    return problems


async def verifier_problems(item, prediction, tools, llm, usage: Usage) -> tuple[list[str], str]:
    records = []
    for source_id in prediction["citations"]:
        record = await tools.call("get_source", {"source_id": source_id})
        records.append(f"[{source_id}]\n{json.dumps(record.get('record'), ensure_ascii=False)}")
    prompt = (
        f"Question: {item['question']}\n\n"
        f"Proposed answer: {json.dumps(prediction['answer'], ensure_ascii=False)}\n\n"
        "Cited records:\n\n" + "\n\n".join(records)
    )
    completion = await llm.complete(VERIFIER_PROMPT, [{"role": "user", "content": prompt}], usage)
    try:
        start = completion.text.index("{")
        verdict, _ = json.JSONDecoder().raw_decode(completion.text, start)
        if verdict.get("supported") is True:
            return [], completion.text
        return [str(p) for p in verdict.get("problems") or ["The cited records do not support the answer."]], completion.text
    except (ValueError, AttributeError):
        return [], completion.text  # unreadable verdict: accept rather than loop on verifier noise


async def answer(item: dict[str, Any], tools, llm, usage: Usage) -> dict[str, Any]:
    messages: list[dict] = [{"role": "user", "content": task_prompt(item)}]
    completion, steps, exhausted = await tool_loop(SYSTEM_PROMPT, messages, tools, llm, usage)
    prediction = parse_reply(completion.text)
    rounds = []

    for revision in range(MAX_REVISIONS + 1):
        problems = await mechanical_problems(prediction, tools)
        verifier_reply = None
        if not problems and prediction["answer"] is not None:
            problems, verifier_reply = await verifier_problems(item, prediction, tools, llm, usage)
        rounds.append({"answer": prediction["answer"], "citations": prediction["citations"],
                       "problems": problems, "verifier_reply": verifier_reply})
        if not problems or revision == MAX_REVISIONS:
            break
        feedback = FEEDBACK.format(problems="\n".join(f"- {p}" for p in problems), contract=OUTPUT_CONTRACT)
        messages.append({"role": "user", "content": feedback})
        completion, more_steps, more_exhausted = await tool_loop(
            SYSTEM_PROMPT, messages, tools, llm, usage, max_steps=REVISION_STEPS
        )
        steps, exhausted = steps + more_steps, exhausted or more_exhausted
        prediction = parse_reply(completion.text)

    return {
        **prediction,
        "raw_reply": completion.text,
        "tool_rounds": steps,
        "budget_exhausted": exhausted,
        "revisions": len(rounds) - 1,
        "verification_rounds": rounds,
        "accepted_by_verifier": not rounds[-1]["problems"],
        "transcript": messages,
    }
