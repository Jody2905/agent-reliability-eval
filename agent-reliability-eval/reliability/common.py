"""Pieces shared by all three systems, so the only thing that differs is the architecture."""

from __future__ import annotations

import json
from typing import Any

OUTPUT_CONTRACT = """\
Finish your reply with a JSON object on its own, in exactly this shape:
{"answer": <the answer, in the requested format>, "citations": [<source ids that support the answer>]}

Rules:
- Use only the provided data. Do not answer from memory.
- Cite the source ids (like "cve:CVE-2021-44228" or "kev:CVE-2021-44228") that support your answer.
- If the data does not contain the answer, use {"answer": null, "citations": []}."""


def task_prompt(item: dict[str, Any]) -> str:
    return f"Question: {item['question']}\n\nAnswer format: {item['answer_format']}"


_DECODER = json.JSONDecoder()


def parse_reply(text: str) -> dict[str, Any]:
    """Pull {"answer", "citations"} out of a reply. Tolerates prose and code fences around it.

    Scans from the end, because the contract asks for the JSON object last.
    """
    for start in reversed([i for i, ch in enumerate(text) if ch == "{"]):
        try:
            obj, _ = _DECODER.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "answer" in obj:
            citations = obj.get("citations") or []
            if not isinstance(citations, list):
                citations = [citations]
            return {"answer": obj["answer"], "citations": [str(c) for c in citations], "parse_error": False}
    return {"answer": None, "citations": [], "parse_error": True}
