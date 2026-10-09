"""System 1: basic retrieval-augmented generation.

1. Retrieve: BM25-search the corpus with the question text, then fetch the full
   record for each hit. This is fixed code, not model decisions.
2. Generate: one LLM call with the retrieved records as context.

No tools, no loop, no second chances. That is the point of a baseline.
"""

from __future__ import annotations

import json
from typing import Any

from ..common import OUTPUT_CONTRACT, parse_reply, task_prompt
from ..llm import Usage

NAME = "rag"
TOP_K = 8

SYSTEM_PROMPT = f"""\
You answer cybersecurity research questions using only the records provided.
Each record is labelled with its source id.

{OUTPUT_CONTRACT}"""


async def answer(item: dict[str, Any], tools, llm, usage: Usage) -> dict[str, Any]:
    hits = (await tools("search_corpus", query=item["question"], limit=TOP_K))["results"]
    records = []
    for hit in hits:
        full = await tools("get_source", source_id=hit["source_id"])
        records.append(f"[{hit['source_id']}]\n{json.dumps(full['record'], ensure_ascii=False)}")

    context = "\n\n".join(records) if records else "(no records found)"
    prompt = f"Records:\n\n{context}\n\n{task_prompt(item)}"
    completion = await llm.complete(SYSTEM_PROMPT, [{"role": "user", "content": prompt}], usage)

    return {
        **parse_reply(completion.text),
        "raw_reply": completion.text,
        "retrieved": [h["source_id"] for h in hits],
    }
