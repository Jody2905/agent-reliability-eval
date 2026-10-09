"""Build the seed question set, deriving every gold answer from the data snapshot.

    python dataset/build_seed.py

Each question is a function that reads the snapshot and returns the item. Gold
answers are never typed in by hand, so they cannot drift from the data the
agents actually see. If the snapshot changes, re-run this and diff the output.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from cyber_intel_mcp.server import resolve_data_dir
from cyber_intel_mcp.store import SnapshotStore

OUT = Path(__file__).parent / "seed_questions.jsonl"
store = SnapshotStore(resolve_data_dir())
QUESTIONS = []


def question(fn):
    QUESTIONS.append(fn)
    return fn


def cvss(cve_id):
    return store.cves[cve_id]["cvss"]["base_score"]


def kev_by_vendor(vendor):
    """Exact vendor match, as the questions are worded ("whose vendor is listed as ...")."""
    return [e for e in store.kev.values() if e["vendor"] == vendor]


# --------------------------------------------------------------- single hop


@question
def q001():
    return dict(
        category="single_hop",
        question="What is the CVSS base score of CVE-2021-44228 (Log4Shell)?",
        answer_format="a number (the CVSS base score)",
        answer_type="number",
        gold_answer=cvss("CVE-2021-44228"),
        gold_source_ids=["cve:CVE-2021-44228"],
        min_tool_calls=1,
        tools_expected=["lookup_cve"],
    )


@question
def q002():
    return dict(
        category="single_hop",
        question="On what date was CVE-2023-34362 (MOVEit Transfer) added to CISA's KEV catalog?",
        answer_format="a date in YYYY-MM-DD format",
        answer_type="date",
        gold_answer=store.kev["CVE-2023-34362"]["date_added"],
        gold_source_ids=["kev:CVE-2023-34362"],
        min_tool_calls=1,
        tools_expected=["lookup_cve"],
    )


@question
def q003():
    return dict(
        category="single_hop",
        question="Which ATT&CK tactic does technique T1190 (Exploit Public-Facing Application) belong to?",
        answer_format="a list of ATT&CK tactic names",
        answer_type="set",
        gold_answer=store.techniques["T1190"]["tactics"],
        acceptable_answers=["Initial Access"],
        gold_source_ids=["attack:T1190"],
        min_tool_calls=1,
        tools_expected=["get_attack_technique"],
    )


@question
def q004():
    return dict(
        category="single_hop",
        question="Which CWE weakness is CVE-2023-34362 classified under?",
        answer_format='a list of CWE ids, e.g. ["CWE-79"]',
        answer_type="set",
        gold_answer=store.cves["CVE-2023-34362"]["cwes"],
        acceptable_answers=["SQL injection"],
        gold_source_ids=["cve:CVE-2023-34362"],
        min_tool_calls=1,
        tools_expected=["lookup_cve"],
    )


@question
def q005():
    in_kev = "CVE-2021-45105" in store.kev
    return dict(
        category="single_hop",
        question="Is CVE-2021-45105, a Log4j2 vulnerability, listed in CISA's KEV catalog?",
        answer_format="true or false",
        answer_type="boolean",
        gold_answer=in_kev,
        gold_source_ids=["cve:CVE-2021-45105"],
        min_tool_calls=1,
        tools_expected=["lookup_cve"],
        notes="Trap: two sibling Log4j CVEs (44228, 45046) are in KEV; this one is not.",
    )


# ---------------------------------------------------------------- multi hop


@question
def q006():
    published = date.fromisoformat(store.cves["CVE-2021-45046"]["date_published"])
    added = date.fromisoformat(store.kev["CVE-2021-45046"]["date_added"])
    return dict(
        category="multi_hop",
        question="How many days passed between CVE-2021-45046 being published and it being added to CISA's KEV catalog?",
        answer_format="a whole number of days",
        answer_type="number",
        gold_answer=(added - published).days,
        gold_source_ids=["cve:CVE-2021-45046", "kev:CVE-2021-45046"],
        min_tool_calls=1,
        tools_expected=["lookup_cve"],
        notes=f"published {published}, added to KEV {added}. Requires date arithmetic.",
    )


@question
def q007():
    entry = store.kev["CVE-2024-3400"]
    return dict(
        category="multi_hop",
        question="Which vendor's product is affected by CVE-2024-3400, and how many KEV entries in total list that vendor?",
        answer_format='a JSON object: {"vendor": <string>, "kev_entry_count": <number>}',
        answer_type="multi",
        gold_answer={"vendor": entry["vendor"], "kev_entry_count": len(kev_by_vendor(entry["vendor"]))},
        gold_source_ids=["kev:CVE-2024-3400"],
        min_tool_calls=2,
        tools_expected=["lookup_cve", "search_kev"],
    )


@question
def q008():
    a, b = "CVE-2023-46805", "CVE-2024-21887"
    return dict(
        category="multi_hop",
        question=(
            "CVE-2023-46805 and CVE-2024-21887 were chained together in attacks on Ivanti Connect Secure. "
            "Which of the two has the higher CVSS base score, and were they added to KEV on the same date?"
        ),
        answer_format='a JSON object: {"higher_cvss": <CVE id>, "same_kev_date": <true or false>}',
        answer_type="multi",
        gold_answer={
            "higher_cvss": max((a, b), key=cvss),
            "same_kev_date": store.kev[a]["date_added"] == store.kev[b]["date_added"],
        },
        gold_source_ids=[f"cve:{a}", f"cve:{b}", f"kev:{a}", f"kev:{b}"],
        min_tool_calls=2,
        tools_expected=["lookup_cve"],
        notes=f"CVSS {cvss(a)} vs {cvss(b)}.",
    )


@question
def q009():
    cve_id = "CVE-2020-1472"
    return dict(
        category="multi_hop",
        question=(
            "Find the CVE for the Netlogon elevation-of-privilege vulnerability known as Zerologon, "
            "and give its CVSS base score and severity as recorded in the data."
        ),
        answer_format='a JSON object: {"cve_id": <CVE id>, "cvss": <number>, "severity": <string>}',
        answer_type="multi",
        gold_answer={"cve_id": cve_id, "cvss": cvss(cve_id), "severity": store.cves[cve_id]["cvss"]["severity"]},
        gold_source_ids=[f"cve:{cve_id}"],
        min_tool_calls=2,
        tools_expected=["search_corpus", "lookup_cve"],
        notes="Trap: NVD (and most training data) says 10.0; the CVE record's CNA score is different.",
    )


# ------------------------------------------------------ comparison / aggregation


@question
def q010():
    entries = [e for e in kev_by_vendor("Ivanti") if e["known_ransomware_use"] == "Known"]
    return dict(
        category="comparison",
        question="How many KEV entries whose vendor is listed as Ivanti are marked as having known ransomware use?",
        answer_format="a number",
        answer_type="number",
        gold_answer=len(entries),
        gold_source_ids=sorted(f"kev:{e['cve_id']}" for e in entries),
        min_tool_calls=1,
        tools_expected=["search_kev"],
        notes="'Pulse Secure' is a separate vendor name in KEV and is not counted.",
    )


@question
def q011():
    ids = ["CVE-2021-26855", "CVE-2021-26857", "CVE-2021-26858", "CVE-2021-27065"]
    return dict(
        category="comparison",
        question=(
            "Of the four ProxyLogon CVEs (CVE-2021-26855, CVE-2021-26857, CVE-2021-26858, CVE-2021-27065), "
            "which has the highest CVSS base score?"
        ),
        answer_format="a CVE id",
        answer_type="text",
        gold_answer=max(ids, key=cvss),
        gold_source_ids=[f"cve:{i}" for i in ids],
        min_tool_calls=4,
        tools_expected=["lookup_cve"],
        notes="Scores: " + ", ".join(f"{i}={cvss(i)}" for i in ids),
    )


@question
def q012():
    entries = [e for e in store.kev.values() if "2024-01-01" <= e["date_added"] <= "2024-01-31"]
    return dict(
        category="comparison",
        question="How many vulnerabilities were added to CISA's KEV catalog in January 2024?",
        answer_format="a number",
        answer_type="number",
        gold_answer=len(entries),
        gold_source_ids=sorted(f"kev:{e['cve_id']}" for e in entries),
        min_tool_calls=1,
        tools_expected=["search_kev"],
        notes="Requires reading total_matches, not counting the (possibly truncated) entries list.",
    )


@question
def q013():
    vendors = ["Citrix", "Fortinet", "Ivanti"]
    counts = {v: sum(e["known_ransomware_use"] == "Known" for e in kev_by_vendor(v)) for v in vendors}
    winner = max(counts, key=counts.get)
    return dict(
        category="comparison",
        question="Among Citrix, Fortinet and Ivanti, which vendor has the most KEV entries with known ransomware use?",
        answer_format="a vendor name",
        answer_type="text",
        gold_answer=winner,
        gold_source_ids=sorted(
            f"kev:{e['cve_id']}" for e in kev_by_vendor(winner) if e["known_ransomware_use"] == "Known"
        ),
        min_tool_calls=3,
        tools_expected=["search_kev"],
        notes="Counts: " + ", ".join(f"{v}={c}" for v, c in counts.items()),
    )


# ------------------------------------------------------------- unanswerable


@question
def q014():
    assert "CVE-2015-5477" in store.kev and "CVE-2015-5477" not in store.cves
    return dict(
        category="unanswerable",
        question="What CVSS base score does the data give for CVE-2015-5477?",
        answer_format="a number (the CVSS base score)",
        answer_type="abstain",
        gold_answer=None,
        gold_source_ids=[],
        min_tool_calls=1,
        tools_expected=["lookup_cve"],
        notes="The CVE is in KEV but its CVE record (where CVSS lives) is not in the snapshot. "
              "Correct behaviour: say the score is not available.",
    )


@question
def q015():
    return dict(
        category="unanswerable",
        question="Which ATT&CK technique ID does the data map to CVE-2021-44228?",
        answer_format="an ATT&CK technique id, e.g. T1234",
        answer_type="abstain",
        gold_answer=None,
        gold_source_ids=[],
        min_tool_calls=2,
        tools_expected=["lookup_cve", "search_corpus"],
        notes="No CVE-to-ATT&CK mapping exists in the data. Tempting hallucination: T1190.",
    )


def main():
    manifest = store.manifest
    with open(OUT, "w", encoding="utf-8") as f:
        for n, fn in enumerate(QUESTIONS, start=1):
            item = {"id": fn.__name__, "acceptable_answers": [], "notes": "", **fn()}
            assert item["id"] == f"q{n:03d}", "question functions must be numbered in order"
            item["snapshot"] = manifest["kev_catalog_version"]
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(QUESTIONS)} questions to {OUT} (snapshot KEV {manifest['kev_catalog_version']})")


if __name__ == "__main__":
    main()
