"""Run one system over the question set and score it.

    python -m reliability.run --system rag                    # Claude Haiku 5.5 (default)
    python -m reliability.run --system rag --model claude-sonnet-5-5
    python -m reliability.run --system rag --ids q001 q009    # just a few questions
    python -m reliability.run --system rag --fake             # offline dry run: no API key, no cost

Writes results/<system>__<model>__<timestamp>.jsonl (one row per question,
including the raw model reply) plus a matching _summary.json.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from mcp import Client

from cyber_intel_mcp.server import mcp

from .common import parse_reply
from .llm import LLM, FakeLLM, Usage
from .scoring import print_report, score_answer, score_citations, summarize
from .tools import Tools

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ["rag", "agent", "agent_verify"]


def fake_llm() -> FakeLLM:
    """Always abstains. Exercises the whole pipeline without spending anything."""
    return FakeLLM(lambda system, messages: 'I could not determine this.\n{"answer": null, "citations": []}')


async def run(system_name: str, model: str, questions: list[dict], use_fake: bool) -> list[dict]:
    system = importlib.import_module(f"reliability.systems.{system_name}")
    llm = fake_llm() if use_fake else LLM(model)
    rows = []
    async with Client(mcp) as client:
        for item in questions:
            tools, usage = Tools(client), Usage()
            start = time.perf_counter()
            error = None
            try:
                prediction = await system.answer(item, tools, llm, usage)
            except Exception as err:  # noqa: BLE001 - a crash is a result too; record it and move on
                error = f"{type(err).__name__}: {err}"
                prediction = {**parse_reply(""), "raw_reply": ""}
            row = {
                "id": item["id"], "category": item["category"], "system": system_name, "model": llm.model,
                "item": item, "prediction": prediction, "error": error,
                "seconds": round(time.perf_counter() - start, 3),
                "usage": vars(usage), "tool_calls": tools.calls, "tool_log": tools.log,
            }
            row["score"] = score_answer(item, prediction)
            row["citations"] = score_citations(item, prediction)
            print(f"  {item['id']}: {row['score']['outcome']}" + (f"  ({error})" if error else ""), flush=True)
            rows.append(row)
    return rows


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--model", default="claude-haiku-5-5")
    parser.add_argument("--questions", type=Path, default=ROOT / "dataset" / "questions.jsonl")
    parser.add_argument("--ids", nargs="+", help="only run these question ids")
    parser.add_argument("--fake", action="store_true", help="offline dry run with a fake model")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    questions = [json.loads(l) for l in args.questions.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.ids:
        questions = [q for q in questions if q["id"] in set(args.ids)]
    model = "fake" if args.fake else args.model
    print(f"Running {args.system} with {model} on {len(questions)} questions...")

    rows = asyncio.run(run(args.system, args.model, questions, args.fake))
    summary = summarize(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.system}__{model}__{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    with open(args.out_dir / f"{stem}.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    (args.out_dir / f"{stem}_summary.json").write_text(
        json.dumps({"system": args.system, "model": model, "questions": str(args.questions), **summary}, indent=2)
    )
    print_report(summary, rows, f"{args.system} / {model}")
    print(f"\nSaved results/{stem}.jsonl")


if __name__ == "__main__":
    main()
