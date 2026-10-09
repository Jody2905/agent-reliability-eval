"""System 2: tool-using agent.

The model gets the five MCP tools and decides itself what to call, in a loop:

    model -> tool calls -> results -> model -> ... -> final answer

The loop stops when the model answers without calling a tool, or after
MAX_STEPS rounds of tool calls. At that point the model is told to answer
with what it has, and tools are switched off (tool_choice "none").
"""

from __future__ import annotations

import json
from typing import Any

from ..common import OUTPUT_CONTRACT, parse_reply, task_prompt
from ..llm import Completion, Usage

NAME = "agent"
MAX_STEPS = 8
MAX_RESULT_CHARS = 30_000  # keep a huge tool result from flooding the context

SYSTEM_PROMPT = f"""\
You are a cybersecurity research agent. You have tools over a frozen snapshot of
CVE records, the CISA KEV catalog and MITRE ATT&CK techniques. Use them to find
the answer; plan which lookups you need, and check your facts before answering.
Every tool result includes source ids you can cite.

{OUTPUT_CONTRACT}"""

BUDGET_MESSAGE = "You have used all your tool calls. Give your final answer now, using only what you have found."


def _tool_result(call: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    text = json.dumps(output, ensure_ascii=False)
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + " ... [truncated]"
    block = {"type": "tool_result", "tool_use_id": call["id"], "content": text}
    if set(output) == {"error"}:
        block["is_error"] = True
    return block


async def tool_loop(
    system: str, messages: list[dict], tools, llm, usage: Usage, max_steps: int = MAX_STEPS
) -> tuple[Completion, int, bool]:
    """Run model <-> tools until the model answers. Mutates `messages` (the transcript).

    Returns (final completion, rounds of tool calls used, whether the budget ran out).
    """
    definitions = await tools.definitions()
    for step in range(max_steps):
        completion = await llm.complete(system, messages, usage, tools=definitions)
        # Pass every block back unchanged - the API rejects edited or dropped thinking blocks.
        messages.append({"role": "assistant", "content": completion.content})
        if not completion.tool_calls:
            return completion, step, False
        results = [_tool_result(call, await tools.call(call["name"], call.get("input") or {}))
                   for call in completion.tool_calls]
        messages.append({"role": "user", "content": results})

    # Budget used up: add the instruction to the last (tool result) message and switch tools off.
    messages[-1]["content"].append({"type": "text", "text": BUDGET_MESSAGE})
    completion = await llm.complete(system, messages, usage, tools=definitions, tool_choice={"type": "none"})
    messages.append({"role": "assistant", "content": completion.content})
    return completion, max_steps, True


async def answer(item: dict[str, Any], tools, llm, usage: Usage) -> dict[str, Any]:
    messages: list[dict] = [{"role": "user", "content": task_prompt(item)}]
    completion, steps, exhausted = await tool_loop(SYSTEM_PROMPT, messages, tools, llm, usage)
    return {
        **parse_reply(completion.text),
        "raw_reply": completion.text,
        "tool_rounds": steps,
        "budget_exhausted": exhausted,
        "transcript": messages,
    }
