"""LLM wrapper: one place for model calls, token counting and cost.

All three systems call the model through `LLM.complete`, so cost and token
accounting is identical across them.
"""

from __future__ import annotations

import inspect
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

# USD per million tokens (input, output), from the Claude pricing page
# (platform.claude.com/docs/en/about-claude/pricing), checked 2026-10-08.
# Haiku 5.5 is billed at a higher rate when a single prompt exceeds 100K tokens.
PRICING = {
    "claude-haiku-5-5": {"tiers": [(100_000, 0.10, 0.50), (None, 0.50, 2.50)]},
    "claude-sonnet-5-5": {"tiers": [(None, 2.00, 10.00)]},
    "claude-opus-5-5": {"tiers": [(None, 4.00, 20.00)]},
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Cost of one request. Returns None for models missing from PRICING."""
    if model not in PRICING:
        return None
    for limit, price_in, price_out in PRICING[model]["tiers"]:
        if limit is None or input_tokens <= limit:
            return (input_tokens * price_in + output_tokens * price_out) / 1_000_000
    raise AssertionError("unreachable")


@dataclass
class Usage:
    """Accumulated over every model call made while answering one question."""

    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = 0.0
    llm_seconds: float = 0.0

    def add(self, model: str, input_tokens: int, output_tokens: int, seconds: float) -> None:
        self.llm_calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.llm_seconds += seconds
        cost = cost_usd(model, input_tokens, output_tokens)
        self.cost_usd = None if cost is None or self.cost_usd is None else self.cost_usd + cost


@dataclass
class Completion:
    text: str  # all text blocks joined
    content: list[dict] = field(default_factory=list)  # every content block as a dict, in order
    stop_reason: str | None = None

    @property
    def tool_calls(self) -> list[dict]:
        return [b for b in self.content if b.get("type") == "tool_use"]


class LLM:
    """Calls the Anthropic Messages API. Reads ANTHROPIC_API_KEY from the environment.

    Notes for Claude 5.5 models (from the model docs, checked 2026-10-08):
    * Setting temperature/top_p/top_k returns a 400 error, so we send none of them.
      Outputs are therefore not deterministic: run each experiment several times
      and report the spread.
    * Adaptive thinking is on by default. Thinking tokens count as output tokens
      (and cost), so max_tokens must leave room for thinking plus the answer.
    * In a tool loop, thinking blocks must be passed back complete and unmodified.
      `Completion.content` keeps every block, in order, for exactly that purpose.
    """

    def __init__(self, model: str = "claude-haiku-5-5", max_tokens: int = 8192):
        import anthropic  # imported here so tests with FakeLLM don't need a key

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Put it in a .env file (see .env.example).")
        self.client = anthropic.AsyncAnthropic(max_retries=5)
        self.model, self.max_tokens = model, max_tokens

    async def complete(
        self, system: str, messages: list[dict], usage: Usage,
        tools: list[dict] | None = None, tool_choice: dict | None = None,
    ) -> Completion:
        kwargs: dict[str, Any] = dict(model=self.model, max_tokens=self.max_tokens, system=system, messages=messages)
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        start = time.perf_counter()
        response = await self.client.messages.create(**kwargs)
        usage.add(self.model, response.usage.input_tokens, response.usage.output_tokens, time.perf_counter() - start)
        content = [block.model_dump(mode="json", exclude_none=True) for block in response.content]
        text = "".join(b["text"] for b in content if b["type"] == "text")
        return Completion(text=text, content=content, stop_reason=response.stop_reason)


class FakeLLM:
    """Offline stand-in for tests and dry runs: no API key, no cost.

    `respond(system, messages, tools)` returns either reply text, or a list of
    content blocks (dicts) - e.g. tool_use blocks to test agent loops.
    """

    model = "fake"

    def __init__(self, respond: Callable[..., str | list[dict]]):
        self.respond = respond

    async def complete(
        self, system: str, messages: list[dict], usage: Usage,
        tools: list[dict] | None = None, tool_choice: dict | None = None,
    ) -> Completion:
        offered = tools if (tool_choice or {}).get("type") != "none" else None
        if len(inspect.signature(self.respond).parameters) >= 3:
            reply = self.respond(system, messages, offered)
        else:  # simple fakes take (system, messages)
            reply = self.respond(system, messages)
        content = [{"type": "text", "text": reply}] if isinstance(reply, str) else reply
        text = "".join(b["text"] for b in content if b["type"] == "text")
        stop = "tool_use" if any(b["type"] == "tool_use" for b in content) else "end_turn"
        prompt_chars = len(system) + sum(len(json.dumps(m["content"], default=str)) for m in messages)
        usage.add(self.model, prompt_chars // 4, len(json.dumps(content)) // 4, 0.0)  # rough token estimate
        return Completion(text=text, content=content, stop_reason=stop)
