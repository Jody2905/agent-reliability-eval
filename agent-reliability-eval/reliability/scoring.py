"""Score system outputs against the gold answers.

    python -m reliability.scoring results/<run>.jsonl

Per question it records:
  outcome            correct | wrong | abstained | hallucinated | parse_error
  correct            bool
  partial            fraction of parts correct (multi-part answers; else 0 or 1)
  citation_precision cited ids that are gold sources / all cited ids
  citation_recall    gold sources cited / all gold sources
  fabricated         cited ids that don't exist in the snapshot at all

Scoring is deterministic (no LLM judge), so anyone can re-run it and get the
same numbers. The rules per answer type are documented in dataset/SCHEMA.md.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from cyber_intel_mcp.server import get_store

_ABSTAIN_WORDS = {"", "null", "none", "n/a", "unknown", "not available", "not found", "no data"}


# ------------------------------------------------------------- normalizing


def norm(value: Any) -> str:
    text = str(value).lower().strip()
    text = re.sub(r"[-_/]", " ", text)
    text = re.sub(r"[^\w. ]", "", text)
    return re.sub(r"\s+", " ", text).strip(" .")


def as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group()) if match else None


def as_date(value: Any) -> str | None:
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass
    match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    return match.group() if match else None


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = norm(value)
    if text in {"true", "yes", "y"}:
        return True
    if text in {"false", "no", "n"}:
        return False
    return None


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [part for part in re.split(r"[,;]| and ", value) if part.strip()]
    return [value]


def is_abstention(value: Any) -> bool:
    return value is None or (isinstance(value, str) and norm(value) in _ABSTAIN_WORDS)


# ------------------------------------------------------------- comparing


def match_value(gold: Any, got: Any, acceptable: list[str] = ()) -> bool:
    """Compare one value, choosing the rule from the gold value's type."""
    if isinstance(gold, bool):
        return as_bool(got) is gold
    if isinstance(gold, (int, float)):
        n = as_number(got)
        return n is not None and abs(n - gold) < 0.05
    if isinstance(gold, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", gold):
        return as_date(got) == gold
    return norm(got) in {norm(gold), *(norm(a) for a in acceptable)}


def match_set(gold: list, got: Any, acceptable: list[str]) -> bool:
    """Every gold item named (directly or by an accepted alias), nothing else added."""
    answer = {norm(x) for x in as_list(got)}
    gold_n = {norm(g) for g in gold}
    alias_n = {norm(a) for a in acceptable}
    if not answer or not answer <= gold_n | alias_n:
        return False
    return gold_n <= answer or bool(answer & alias_n)


def score_answer(item: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    gold, got, kind = item["gold_answer"], prediction.get("answer"), item["answer_type"]
    acceptable = item.get("acceptable_answers", [])

    if prediction.get("parse_error"):
        return {"outcome": "parse_error", "correct": False, "partial": 0.0}
    if kind == "abstain":
        ok = is_abstention(got)
        return {"outcome": "correct" if ok else "hallucinated", "correct": ok, "partial": float(ok)}
    if is_abstention(got):
        return {"outcome": "abstained", "correct": False, "partial": 0.0}

    if kind == "multi":
        if isinstance(got, str):
            try:
                got = json.loads(got)
            except json.JSONDecodeError:
                got = {}
        got = got if isinstance(got, dict) else {}
        parts = {key: key in got and match_value(value, got[key]) for key, value in gold.items()}
        partial = sum(parts.values()) / len(parts)
        return {"outcome": "correct" if partial == 1 else "wrong", "correct": partial == 1,
                "partial": partial, "parts": parts}

    ok = match_set(gold, got, acceptable) if kind == "set" else match_value(gold, got, acceptable)
    return {"outcome": "correct" if ok else "wrong", "correct": ok, "partial": float(ok)}


def score_citations(item: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    store = get_store()
    cited = list(dict.fromkeys(c.strip() for c in prediction.get("citations", [])))
    gold = set(item["gold_source_ids"])
    fabricated = [c for c in cited if store.get_source(c) is None]
    hits = [c for c in cited if c in gold]
    return {
        "n_cited": len(cited),
        "citation_precision": len(hits) / len(cited) if cited and gold else None,
        "citation_recall": len(hits) / len(gold) if gold else None,
        "fabricated": fabricated,
    }


# ------------------------------------------------------------- summarizing


def _mean(values):
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 4) if values else None


def _percentile(values, pct):
    values = sorted(values)
    if not values:
        return None
    k = (len(values) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return round(values[lo] + (values[hi] - values[lo]) * (k - lo), 3)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [r for r in rows if r["category"] != "unanswerable"]
    unanswerable = [r for r in rows if r["category"] == "unanswerable"]
    correct = sum(r["score"]["correct"] for r in rows)
    costs = [r["usage"]["cost_usd"] for r in rows]
    total_cost = None if any(c is None for c in costs) else round(sum(costs), 6)
    seconds = [r["seconds"] for r in rows]
    return {
        "n": len(rows),
        "accuracy": round(correct / len(rows), 4) if rows else None,
        "partial_credit": _mean(r["score"]["partial"] for r in rows),
        "by_category": {
            cat: round(sum(r["score"]["correct"] for r in group) / len(group), 4)
            for cat in ("single_hop", "multi_hop", "comparison", "unanswerable")
            if (group := [r for r in rows if r["category"] == cat])
        },
        "hallucination_rate": _mean(float(r["score"]["outcome"] == "hallucinated") for r in unanswerable),
        "wrong_abstention_rate": _mean(float(r["score"]["outcome"] == "abstained") for r in answerable),
        "parse_errors": sum(r["score"]["outcome"] == "parse_error" for r in rows),
        "errors": sum(bool(r.get("error")) for r in rows),
        "citation_precision": _mean(r["citations"]["citation_precision"] for r in answerable),
        "citation_recall": _mean(r["citations"]["citation_recall"] for r in answerable),
        "fabricated_citations": sum(len(r["citations"]["fabricated"]) for r in rows),
        "total_cost_usd": total_cost,
        "cost_per_correct_usd": round(total_cost / correct, 6) if total_cost is not None and correct else None,
        "total_tokens": {"input": sum(r["usage"]["input_tokens"] for r in rows),
                         "output": sum(r["usage"]["output_tokens"] for r in rows)},
        "mean_llm_calls": _mean(r["usage"]["llm_calls"] for r in rows),
        "mean_tool_calls": _mean(r["tool_calls"] for r in rows),
        "budget_exhausted": sum(bool(r["prediction"].get("budget_exhausted")) for r in rows),
        "mean_revisions": _mean(r["prediction"].get("revisions") for r in rows),
        "revised_and_fixed": sum(
            1 for r in rows if r["prediction"].get("revisions") and r["score"]["correct"]
        ),
        "latency_p50_s": _percentile(seconds, 50),
        "latency_p95_s": _percentile(seconds, 95),
    }


def print_report(summary: dict[str, Any], rows: list[dict[str, Any]], title: str = "") -> None:
    if title:
        print(f"\n=== {title} ===")
    print(f"{'id':6} {'category':13} {'outcome':13} {'cite P/R':10} {'$':>9} {'sec':>6}")
    for r in rows:
        c = r["citations"]
        pr = "-" if c["citation_recall"] is None else f"{c['citation_precision'] or 0:.2f}/{c['citation_recall']:.2f}"
        cost = "-" if r["usage"]["cost_usd"] is None else f"{r['usage']['cost_usd']:.6f}"
        print(f"{r['id']:6} {r['category']:13} {r['score']['outcome']:13} {pr:10} {cost:>9} {r['seconds']:6.2f}")
    s = summary
    print(f"\nAccuracy {s['accuracy']:.0%} ({s['n']} questions) | by category: "
          + ", ".join(f"{k} {v:.0%}" for k, v in s["by_category"].items()))
    print(f"Hallucination rate (unanswerable): {s['hallucination_rate']} | wrong abstentions: {s['wrong_abstention_rate']}")
    print(f"Citations: precision {s['citation_precision']}, recall {s['citation_recall']}, fabricated {s['fabricated_citations']}")
    if s["mean_revisions"] is not None:
        print(f"Verification: mean revisions {s['mean_revisions']}, revised and ended correct {s['revised_and_fixed']}")
    print(f"Tool calls per question {s['mean_tool_calls']} | step budget hit {s['budget_exhausted']} times")
    print(f"Cost: total ${s['total_cost_usd']}, per correct answer ${s['cost_per_correct_usd']} | "
          f"latency p50 {s['latency_p50_s']}s, p95 {s['latency_p95_s']}s")


def rescore(results_path: Path) -> None:
    """Re-score a saved results file, e.g. after changing a scoring rule."""
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        row["score"] = score_answer(row["item"], row["prediction"])
        row["citations"] = score_citations(row["item"], row["prediction"])
    print_report(summarize(rows), rows, results_path.name)


if __name__ == "__main__":
    rescore(Path(sys.argv[1]))
