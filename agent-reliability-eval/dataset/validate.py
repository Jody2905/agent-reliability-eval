"""Validate the question set.

    python dataset/validate.py [path/to/questions.jsonl]

1. Schema: every item has the required fields with sensible values.
2. Sources: every gold_source_id resolves through the MCP server's get_source tool.
3. Reachability: a scripted *reference solution* answers each question using only
   the MCP tools, and must reproduce the gold answer. This proves the question is
   answerable through the same interface the agents use (or, for "unanswerable"
   items, that the tools really don't contain the answer).

The reference solutions are also a useful "oracle" baseline for the paper: they
show the minimum tool calls a perfect agent would need.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import date
from pathlib import Path

from mcp import Client

from cyber_intel_mcp.server import mcp

DEFAULT = Path(__file__).parent / "seed_questions.jsonl"
CATEGORIES = {"single_hop", "multi_hop", "comparison", "unanswerable"}
ANSWER_TYPES = {"number", "date", "text", "boolean", "set", "multi", "abstain"}
REQUIRED = {
    "id", "category", "question", "answer_format", "answer_type", "gold_answer", "acceptable_answers",
    "gold_source_ids", "min_tool_calls", "tools_expected", "notes", "snapshot",
}


class Tools:
    """Thin wrapper that calls the real MCP server and counts calls."""

    def __init__(self, client: Client):
        self.client, self.calls = client, 0

    async def __call__(self, name: str, **args):
        self.calls += 1
        result = await self.client.call_tool(name, args)
        if result.is_error:
            raise RuntimeError(f"{name}({args}) failed: {result.content[0].text}")
        return result.structured_content


def kev_count(result, vendor):
    """search_kev matches vendor substrings; questions mean the exact vendor name."""
    assert not result["truncated"], "raise the limit"
    return [e for e in result["entries"] if e["vendor"] == vendor]


# ---- reference solutions: one per question, using only MCP tools ----------

async def q001(t): return (await t("lookup_cve", cve_id="CVE-2021-44228"))["cvss"]["base_score"]
async def q002(t): return (await t("lookup_cve", cve_id="CVE-2023-34362"))["kev_date_added"]
async def q003(t): return (await t("get_attack_technique", technique_id="T1190"))["tactics"]
async def q004(t): return (await t("lookup_cve", cve_id="CVE-2023-34362"))["cwes"]
async def q005(t): return (await t("lookup_cve", cve_id="CVE-2021-45105"))["in_kev"]


async def q006(t):
    r = await t("lookup_cve", cve_id="CVE-2021-45046")
    return (date.fromisoformat(r["kev_date_added"]) - date.fromisoformat(r["date_published"])).days


async def q007(t):
    vendor = (await t("get_source", source_id="kev:CVE-2024-3400"))["record"]["vendor"]
    entries = kev_count(await t("search_kev", vendor=vendor, limit=100), vendor)
    return {"vendor": vendor, "kev_entry_count": len(entries)}


async def q008(t):
    a = await t("lookup_cve", cve_id="CVE-2023-46805")
    b = await t("lookup_cve", cve_id="CVE-2024-21887")
    higher = max((a, b), key=lambda r: r["cvss"]["base_score"])["cve_id"]
    return {"higher_cvss": higher, "same_kev_date": a["kev_date_added"] == b["kev_date_added"]}


async def q009(t):
    hits = (await t("search_corpus", query="Netlogon elevation of privilege", doc_types=["cve"]))["results"]
    cve_id = hits[0]["source_id"].split(":", 1)[1]
    r = await t("lookup_cve", cve_id=cve_id)
    return {"cve_id": cve_id, "cvss": r["cvss"]["base_score"], "severity": r["cvss"]["severity"]}


async def q010(t):
    return len(kev_count(await t("search_kev", vendor="Ivanti", ransomware_only=True, limit=100), "Ivanti"))


async def q011(t):
    ids = ["CVE-2021-26855", "CVE-2021-26857", "CVE-2021-26858", "CVE-2021-27065"]
    scores = {i: (await t("lookup_cve", cve_id=i))["cvss"]["base_score"] for i in ids}
    return max(scores, key=scores.get)


async def q012(t):
    return (await t("search_kev", added_after="2024-01-01", added_before="2024-01-31", limit=1))["total_matches"]


async def q013(t):
    counts = {}
    for v in ["Citrix", "Fortinet", "Ivanti"]:
        counts[v] = len(kev_count(await t("search_kev", vendor=v, ransomware_only=True, limit=100), v))
    return max(counts, key=counts.get)


async def q014(t):
    r = await t("lookup_cve", cve_id="CVE-2015-5477")
    return None if not r["found"] else r["cvss"]["base_score"]


async def q015(t):
    r = await t("lookup_cve", cve_id="CVE-2021-44228")
    # Neither the CVE record nor any tool exposes an ATT&CK mapping.
    assert not any(k.startswith("attack") or "technique" in k for k in r), "data now has a mapping!"
    return None


# ---------------------------------------------------------------------------

def check_schema(item) -> list[str]:
    problems = [f"missing field {f}" for f in REQUIRED - item.keys()]
    if item.get("category") not in CATEGORIES:
        problems.append(f"bad category {item.get('category')}")
    if item.get("answer_type") not in ANSWER_TYPES:
        problems.append(f"bad answer_type {item.get('answer_type')}")
    if item.get("category") == "unanswerable" and (item.get("answer_type") != "abstain" or item.get("gold_answer") is not None):
        problems.append("unanswerable items need answer_type=abstain and gold_answer=null")
    if item.get("category") != "unanswerable" and not item.get("gold_source_ids"):
        problems.append("answerable items need at least one gold_source_id")
    return problems


def same(gold, got) -> bool:
    if isinstance(gold, list):
        return sorted(gold) == sorted(got or [])
    if isinstance(gold, float) or isinstance(got, float):
        return abs(float(gold) - float(got)) < 1e-9
    return gold == got


async def main(path: Path) -> int:
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    failures = 0
    async with Client(mcp) as client:
        for item in items:
            problems = check_schema(item)
            for sid in item.get("gold_source_ids", []):
                r = await client.call_tool("get_source", {"source_id": sid})
                if not r.structured_content["found"]:
                    problems.append(f"source {sid} not found")
            solver = globals().get(item["id"])
            if solver is None:
                problems.append("no reference solution")
            else:
                tools = Tools(client)
                try:
                    got = await solver(tools)
                    if not same(item["gold_answer"], got):
                        problems.append(f"reference got {got!r}, gold is {item['gold_answer']!r}")
                    if tools.calls > item["min_tool_calls"]:
                        problems.append(f"reference needed {tools.calls} calls, min_tool_calls says {item['min_tool_calls']}")
                except Exception as err:  # noqa: BLE001 - report and keep going
                    problems.append(f"reference solution crashed: {err}")
            status = "ok  " if not problems else "FAIL"
            print(f"{status} {item['id']} [{item['category']}] gold={json.dumps(item['gold_answer'])}")
            for p in problems:
                print(f"       - {p}")
            failures += bool(problems)
    print(f"\n{len(items) - failures}/{len(items)} questions valid")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT)))
