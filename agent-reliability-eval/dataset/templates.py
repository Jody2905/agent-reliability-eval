"""Question templates.

Each template is one *kind* of question, filled in from the snapshot many times:

    candidates(store) -> every valid parameter set (e.g. every CVE that has a CVSS score)
    build(store, p)   -> the question, its format, and the gold answer DERIVED FROM THE DATA
    solve(tools, p)   -> a reference solution that uses ONLY the MCP tools

`dataset/build.py` samples `count` parameter sets per template with a fixed seed.
`dataset/validate.py` runs every `solve` and checks it reproduces the gold answer,
so each question is proven answerable (or proven unanswerable) through the tools.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date
from itertools import combinations
from typing import Any

from cyber_intel_mcp.store import SnapshotStore

# ---------------------------------------------------------------- helpers


def days_between(a: str, b: str) -> int:
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def clean_vendors(store: SnapshotStore, min_entries: int = 1) -> list[str]:
    """Vendors whose name is not part of another vendor's name.

    search_kev matches vendor *substrings*; for these vendors the substring
    match equals the exact match, so a count from the tool is unambiguous.
    """
    names = sorted({e["vendor"] for e in store.kev.values()})
    counts = {v: sum(e["vendor"] == v for e in store.kev.values()) for v in names}
    return [
        v for v in names
        if counts[v] >= min_entries and not any(v.lower() in o.lower() for o in names if o != v)
    ]


def kev_of(store, vendor=None, year=None, ransomware=False):
    return sorted(
        (e for e in store.kev.values()
         if (vendor is None or e["vendor"] == vendor)
         and (year is None or e["date_added"].startswith(str(year)))
         and (not ransomware or e["known_ransomware_use"] == "Known")),
        key=lambda e: e["cve_id"],
    )


def with_cvss(store):
    return sorted(c for c, r in store.cves.items() if r["cvss"])


def in_sample_and_kev(store):
    return sorted(c for c in store.cves if c in store.kev)


def in_sample_not_kev(store):
    return sorted(c for c in store.cves if c not in store.kev)


def cvss(store, cve):
    return store.cves[cve]["cvss"]["base_score"]


@dataclass
class Template:
    name: str
    category: str
    difficulty: str
    count: int
    min_tool_calls: int
    tools_expected: list[str]

    def candidates(self, store) -> list[dict]:
        raise NotImplementedError

    def build(self, store, p) -> dict:
        raise NotImplementedError

    async def solve(self, t, p) -> Any:
        raise NotImplementedError

    def pick(self, store, rng) -> list[dict]:
        """Candidates in a seeded random order; the builder takes the first `count` that aren't duplicates."""
        pool = self.candidates(store)
        return rng.sample(pool, len(pool))


# ============================================================== single hop


class CvssScore(Template):
    def candidates(self, s):
        return [{"cve": c} for c in with_cvss(s)]

    def build(self, s, p):
        return dict(question=f"What is the CVSS base score of {p['cve']}?",
                    answer_format="a number (the CVSS base score)", answer_type="number",
                    gold_answer=cvss(s, p["cve"]), gold_source_ids=[f"cve:{p['cve']}"])

    async def solve(self, t, p):
        return (await t("lookup_cve", cve_id=p["cve"]))["cvss"]["base_score"]


class KevDateAdded(Template):
    def candidates(self, s):
        return [{"cve": c} for c in in_sample_and_kev(s)]

    def build(self, s, p):
        return dict(question=f"On what date was {p['cve']} added to CISA's KEV catalog?",
                    answer_format="a date in YYYY-MM-DD format", answer_type="date",
                    gold_answer=s.kev[p["cve"]]["date_added"], gold_source_ids=[f"kev:{p['cve']}"])

    async def solve(self, t, p):
        return (await t("lookup_cve", cve_id=p["cve"]))["kev_date_added"]


class CweOf(Template):
    def candidates(self, s):
        return [{"cve": c} for c, r in sorted(s.cves.items()) if len(r["cwes"]) == 1]

    def build(self, s, p):
        return dict(question=f"Which CWE weakness is {p['cve']} classified under?",
                    answer_format='a list of CWE ids, e.g. ["CWE-79"]', answer_type="set",
                    gold_answer=s.cves[p["cve"]]["cwes"], gold_source_ids=[f"cve:{p['cve']}"])

    async def solve(self, t, p):
        return (await t("lookup_cve", cve_id=p["cve"]))["cwes"]


class InKev(Template):
    """Half yes, half no - a system that always says 'yes' should score 50%."""

    def pick(self, s, rng):
        half = self.count // 2
        return ([{"cve": c} for c in rng.sample(in_sample_and_kev(s), half)]
                + [{"cve": c} for c in rng.sample(in_sample_not_kev(s), self.count - half)])

    def build(self, s, p):
        in_kev = p["cve"] in s.kev
        return dict(question=f"Is {p['cve']} listed in CISA's KEV catalog?",
                    answer_format="true or false", answer_type="boolean", gold_answer=in_kev,
                    gold_source_ids=[f"cve:{p['cve']}"] + ([f"kev:{p['cve']}"] if in_kev else []))

    async def solve(self, t, p):
        return (await t("lookup_cve", cve_id=p["cve"]))["in_kev"]


class TechniqueTactics(Template):
    def candidates(self, s):
        return [{"technique": tid} for tid, tech in sorted(s.techniques.items())
                if not tech["is_subtechnique"] and tech["tactics"]]

    def build(self, s, p):
        tech = s.techniques[p["technique"]]
        return dict(question=f"Which ATT&CK tactic(s) does technique {p['technique']} ({tech['name']}) belong to?",
                    answer_format="a list of ATT&CK tactic names", answer_type="set",
                    gold_answer=tech["tactics"], gold_source_ids=[f"attack:{p['technique']}"])

    async def solve(self, t, p):
        return (await t("get_attack_technique", technique_id=p["technique"]))["tactics"]


class KevVendor(Template):
    def candidates(self, s):
        return [{"cve": c} for c in in_sample_and_kev(s)]

    def build(self, s, p):
        return dict(question=f"Which vendor does CISA's KEV catalog list for {p['cve']}?",
                    answer_format="a vendor name", answer_type="text",
                    gold_answer=s.kev[p["cve"]]["vendor"], gold_source_ids=[f"kev:{p['cve']}"])

    async def solve(self, t, p):
        return (await t("get_source", source_id=f"kev:{p['cve']}"))["record"]["vendor"]


# =============================================================== multi hop


class DaysToKev(Template):
    def candidates(self, s):
        return [{"cve": c} for c in in_sample_and_kev(s)
                if days_between(s.cves[c]["date_published"], s.kev[c]["date_added"]) >= 0]

    def build(self, s, p):
        published, added = s.cves[p["cve"]]["date_published"], s.kev[p["cve"]]["date_added"]
        return dict(question=f"How many days passed between {p['cve']} being published and it being added to CISA's KEV catalog?",
                    answer_format="a whole number of days", answer_type="number",
                    gold_answer=days_between(published, added),
                    gold_source_ids=[f"cve:{p['cve']}", f"kev:{p['cve']}"],
                    notes=f"published {published}, added to KEV {added}")

    async def solve(self, t, p):
        r = await t("lookup_cve", cve_id=p["cve"])
        return days_between(r["date_published"], r["kev_date_added"])


class VendorKevCount(Template):
    def candidates(self, s):
        vendors = set(clean_vendors(s, min_entries=2))
        return [{"cve": c} for c in in_sample_and_kev(s) if s.kev[c]["vendor"] in vendors]

    def build(self, s, p):
        vendor = s.kev[p["cve"]]["vendor"]
        return dict(question=f"According to CISA's KEV catalog, which vendor's product is affected by {p['cve']}, and how many KEV entries in total list that vendor?",
                    answer_format='a JSON object: {"vendor": <string>, "kev_entry_count": <number>}',
                    answer_type="multi",
                    gold_answer={"vendor": vendor, "kev_entry_count": len(kev_of(s, vendor))},
                    gold_source_ids=[f"kev:{p['cve']}"])

    async def solve(self, t, p):
        vendor = (await t("get_source", source_id=f"kev:{p['cve']}"))["record"]["vendor"]
        return {"vendor": vendor, "kev_entry_count": (await t("search_kev", vendor=vendor, limit=1))["total_matches"]}


class CompareTwoCvss(Template):
    def candidates(self, s):
        by_vendor: dict[str, list[str]] = {}
        for c in with_cvss(s):
            if c in s.kev:
                by_vendor.setdefault(s.kev[c]["vendor"], []).append(c)
        return [{"a": a, "b": b, "vendor": v} for v, cves in sorted(by_vendor.items())
                for a, b in combinations(cves, 2) if cvss(s, a) != cvss(s, b)]

    def pick(self, s, rng):  # at most one pair per vendor, for variety
        chosen, seen = [], set()
        for p in rng.sample(self.candidates(s), len(self.candidates(s))):
            if p["vendor"] not in seen:
                chosen.append(p)
                seen.add(p["vendor"])
            if len(chosen) == self.count:
                break
        return chosen

    def build(self, s, p):
        a, b = p["a"], p["b"]
        return dict(question=f"{a} and {b} both affect {p['vendor']} products. Which of the two has the higher CVSS base score?",
                    answer_format="a CVE id", answer_type="text",
                    gold_answer=max((a, b), key=lambda c: cvss(s, c)),
                    gold_source_ids=[f"cve:{a}", f"cve:{b}"],
                    notes=f"{a}={cvss(s, a)}, {b}={cvss(s, b)}")

    async def solve(self, t, p):
        scores = {c: (await t("lookup_cve", cve_id=c))["cvss"]["base_score"] for c in (p["a"], p["b"])}
        return max(scores, key=scores.get)


class FindByNameThenCvss(Template):
    """The agent must find the CVE from a vulnerability *name*, then look up its score."""

    def candidates(self, s):
        name_counts: dict[str, int] = {}
        for e in s.kev.values():
            name_counts[e["name"]] = name_counts.get(e["name"], 0) + 1
        out = []
        for c in with_cvss(s):
            if c in s.kev and name_counts[s.kev[c]["name"]] == 1:
                hits = s.search(s.kev[c]["name"], limit=10, doc_types={"kev"})  # same search as the tool
                if any(d.source_id == f"kev:{c}" for d, _, _ in hits):  # findable by search
                    out.append({"cve": c, "name": s.kev[c]["name"]})
        return out

    def build(self, s, p):
        return dict(question=f'CISA\'s KEV catalog lists a vulnerability named "{p["name"]}". What is its CVE id and its CVSS base score?',
                    answer_format='a JSON object: {"cve_id": <CVE id>, "cvss": <number>}', answer_type="multi",
                    gold_answer={"cve_id": p["cve"], "cvss": cvss(s, p["cve"])},
                    gold_source_ids=[f"kev:{p['cve']}", f"cve:{p['cve']}"])

    async def solve(self, t, p):
        # Uses only the name from the question - never p["cve"], which is the answer.
        hits = (await t("search_corpus", query=p["name"], doc_types=["kev"], limit=10))["results"]
        cve_id = next(h["source_id"].split(":", 1)[1] for h in hits if h["title"].split(" ", 1)[1] == p["name"])
        return {"cve_id": cve_id, "cvss": (await t("lookup_cve", cve_id=cve_id))["cvss"]["base_score"]}


class KevBeforePublished(Template):
    """Rare but real: CISA sometimes lists a CVE before its record is published."""

    def pick(self, s, rng):
        early = [c for c in in_sample_and_kev(s) if s.kev[c]["date_added"] < s.cves[c]["date_published"]]
        later = [c for c in in_sample_and_kev(s) if s.kev[c]["date_added"] > s.cves[c]["date_published"]]
        n_early = min(len(early), self.count // 2)
        return [{"cve": c} for c in rng.sample(early, n_early) + rng.sample(later, self.count - n_early)]

    def build(self, s, p):
        added, published = s.kev[p["cve"]]["date_added"], s.cves[p["cve"]]["date_published"]
        return dict(question=f"Was {p['cve']} added to CISA's KEV catalog before its CVE record was published?",
                    answer_format="true or false", answer_type="boolean", gold_answer=added < published,
                    gold_source_ids=[f"cve:{p['cve']}", f"kev:{p['cve']}"],
                    notes=f"added to KEV {added}, published {published}")

    async def solve(self, t, p):
        r = await t("lookup_cve", cve_id=p["cve"])
        return r["kev_date_added"] < r["date_published"]


class RemediationWindow(Template):
    def candidates(self, s):
        return [{"cve": c} for c in in_sample_and_kev(s) if s.kev[c]["due_date"]]

    def build(self, s, p):
        e = s.kev[p["cve"]]
        return dict(question=f"How many days did CISA give federal agencies to remediate {p['cve']}, from the date it was added to the KEV catalog to its due date?",
                    answer_format="a whole number of days", answer_type="number",
                    gold_answer=days_between(e["date_added"], e["due_date"]),
                    gold_source_ids=[f"kev:{p['cve']}"], notes=f"added {e['date_added']}, due {e['due_date']}")

    async def solve(self, t, p):
        e = (await t("get_source", source_id=f"kev:{p['cve']}"))["record"]
        return days_between(e["date_added"], e["due_date"])


# ============================================================== comparison


class VendorRansomwareCount(Template):
    def candidates(self, s):
        return [{"vendor": v} for v in clean_vendors(s, min_entries=3)]

    def build(self, s, p):
        entries = kev_of(s, p["vendor"], ransomware=True)
        # A count of 0 is supported by the vendor's entries, none of which is marked "Known".
        support = entries or kev_of(s, p["vendor"])
        return dict(question=f"How many KEV entries whose vendor is listed as {p['vendor']} are marked as having known ransomware use?",
                    answer_format="a number", answer_type="number", gold_answer=len(entries),
                    gold_source_ids=[f"kev:{e['cve_id']}" for e in support],
                    notes=f"{len(kev_of(s, p['vendor']))} {p['vendor']} entries in total"
                          + ("; none marked as ransomware-linked" if not entries else ""))

    async def solve(self, t, p):
        return (await t("search_kev", vendor=p["vendor"], ransomware_only=True, limit=1))["total_matches"]


class KevMonthCount(Template):
    def candidates(self, s):
        months = sorted({e["date_added"][:7] for e in s.kev.values()})
        return [{"month": m} for m in months]

    def build(self, s, p):
        year, month = map(int, p["month"].split("-"))
        entries = [e for e in s.kev.values() if e["date_added"].startswith(p["month"])]
        return dict(question=f"How many vulnerabilities were added to CISA's KEV catalog in {calendar.month_name[month]} {year}?",
                    answer_format="a number", answer_type="number", gold_answer=len(entries),
                    gold_source_ids=sorted(f"kev:{e['cve_id']}" for e in entries))

    async def solve(self, t, p):
        year, month = map(int, p["month"].split("-"))
        last = calendar.monthrange(year, month)[1]
        return (await t("search_kev", added_after=f"{p['month']}-01", added_before=f"{p['month']}-{last:02d}", limit=1))["total_matches"]


class VendorYearCount(Template):
    def candidates(self, s):
        return [{"vendor": v, "year": y} for v in clean_vendors(s, min_entries=5)
                for y in sorted({e["date_added"][:4] for e in kev_of(s, v)})]

    def build(self, s, p):
        entries = kev_of(s, p["vendor"], year=p["year"])
        return dict(question=f"How many KEV entries for the vendor {p['vendor']} were added to the catalog during {p['year']}?",
                    answer_format="a number", answer_type="number", gold_answer=len(entries),
                    gold_source_ids=[f"kev:{e['cve_id']}" for e in entries])

    async def solve(self, t, p):
        return (await t("search_kev", vendor=p["vendor"], added_after=f"{p['year']}-01-01",
                        added_before=f"{p['year']}-12-31", limit=1))["total_matches"]


class TopRansomwareVendor(Template):
    def pick(self, s, rng):
        vendors = clean_vendors(s, min_entries=5)
        counts = {v: len(kev_of(s, v, ransomware=True)) for v in vendors}
        chosen = []
        while len(chosen) < self.count:
            trio = sorted(rng.sample(vendors, 3))
            top = sorted((counts[v] for v in trio), reverse=True)
            if top[0] > top[1] and {"vendors": trio} not in chosen:  # a clear winner
                chosen.append({"vendors": trio})
        return chosen

    def build(self, s, p):
        counts = {v: len(kev_of(s, v, ransomware=True)) for v in p["vendors"]}
        winner = max(counts, key=counts.get)
        a, b, c = p["vendors"]
        return dict(question=f"Among {a}, {b} and {c}, which vendor has the most KEV entries with known ransomware use?",
                    answer_format="a vendor name", answer_type="text", gold_answer=winner,
                    gold_source_ids=[f"kev:{e['cve_id']}" for e in kev_of(s, winner, ransomware=True)],
                    notes="Counts: " + ", ".join(f"{v}={n}" for v, n in counts.items()))

    async def solve(self, t, p):
        counts = {v: (await t("search_kev", vendor=v, ransomware_only=True, limit=1))["total_matches"]
                  for v in p["vendors"]}
        return max(counts, key=counts.get)


class HighestCvssOfN(Template):
    def pick(self, s, rng):
        pool, chosen = with_cvss(s), []
        while len(chosen) < self.count:
            group = sorted(rng.sample(pool, 4))
            scores = sorted((cvss(s, c) for c in group), reverse=True)
            if scores[0] > scores[1]:
                chosen.append({"cves": group})
        return chosen

    def build(self, s, p):
        return dict(question=f"Of {', '.join(p['cves'][:-1])} and {p['cves'][-1]}, which has the highest CVSS base score?",
                    answer_format="a CVE id", answer_type="text",
                    gold_answer=max(p["cves"], key=lambda c: cvss(s, c)),
                    gold_source_ids=[f"cve:{c}" for c in p["cves"]],
                    notes="Scores: " + ", ".join(f"{c}={cvss(s, c)}" for c in p["cves"]))

    async def solve(self, t, p):
        scores = {c: (await t("lookup_cve", cve_id=c))["cvss"]["base_score"] for c in p["cves"]}
        return max(scores, key=scores.get)


class CountInKev(Template):
    def pick(self, s, rng):
        chosen = []
        for _ in range(self.count):
            k = rng.randint(1, 4)
            group = rng.sample(in_sample_and_kev(s), k) + rng.sample(in_sample_not_kev(s), 5 - k)
            chosen.append({"cves": sorted(group)})
        return chosen

    def build(self, s, p):
        hits = [c for c in p["cves"] if c in s.kev]
        return dict(question=f"How many of these CVEs are listed in CISA's KEV catalog: {', '.join(p['cves'])}?",
                    answer_format="a number", answer_type="number", gold_answer=len(hits),
                    gold_source_ids=[f"kev:{c}" for c in hits])

    async def solve(self, t, p):
        return sum([(await t("lookup_cve", cve_id=c))["in_kev"] for c in p["cves"]])


# ============================================================ unanswerable


def _abstain(question, answer_format, notes):
    return dict(question=question, answer_format=answer_format, answer_type="abstain",
                gold_answer=None, gold_source_ids=[], notes=notes)


class CvssRecordMissing(Template):
    def candidates(self, s):
        return [{"cve": c} for c in sorted(s.kev) if c not in s.cves]

    def build(self, s, p):
        return _abstain(f"What CVSS base score does the data give for {p['cve']}?", "a number (the CVSS base score)",
                        "In KEV, but its CVE record (where CVSS lives) is not in the snapshot.")

    async def solve(self, t, p):
        r = await t("lookup_cve", cve_id=p["cve"])
        return r["cvss"]["base_score"] if r["found"] else None


class CvssNotScored(Template):
    def candidates(self, s):
        return [{"cve": c} for c, r in sorted(s.cves.items()) if not r["cvss"]]

    def build(self, s, p):
        return _abstain(f"What CVSS base score does the data give for {p['cve']}?", "a number (the CVSS base score)",
                        "The CVE record exists but carries no CVSS score. Easy to answer from memory instead.")

    async def solve(self, t, p):
        r = await t("lookup_cve", cve_id=p["cve"])
        return r["cvss"]["base_score"] if r["cvss"] else None


class KevDateNotInKev(Template):
    def candidates(self, s):
        return [{"cve": c} for c in in_sample_not_kev(s)]

    def build(self, s, p):
        return _abstain(f"On what date was {p['cve']} added to CISA's KEV catalog?", "a date in YYYY-MM-DD format",
                        "False premise: this CVE is not in KEV.")

    async def solve(self, t, p):
        return (await t("lookup_cve", cve_id=p["cve"]))["kev_date_added"]


class AttackMapping(Template):
    def candidates(self, s):
        return [{"cve": c} for c in in_sample_and_kev(s)]

    def build(self, s, p):
        return _abstain(f"Which ATT&CK technique ID does the data map to {p['cve']}?", "an ATT&CK technique id, e.g. T1234",
                        "No CVE-to-ATT&CK mapping exists in the data. Tempting guess: T1190.")

    async def solve(self, t, p):
        r = await t("lookup_cve", cve_id=p["cve"])
        assert not any("attack" in k or "technique" in k for k in r), "data now has a mapping!"
        return None


class EpssScore(Template):
    def candidates(self, s):
        return [{"cve": c} for c in in_sample_and_kev(s)]

    def build(self, s, p):
        return _abstain(f"What EPSS score does the data give for {p['cve']}?", "a number between 0 and 1",
                        "The data has no EPSS scores.")

    async def solve(self, t, p):
        r = await t("lookup_cve", cve_id=p["cve"])
        return r.get("epss")


TEMPLATES: dict[str, Template] = {
    t.name: t
    for t in [
        CvssScore("cvss_score", "single_hop", "easy", 8, 1, ["lookup_cve"]),
        KevDateAdded("kev_date_added", "single_hop", "easy", 6, 1, ["lookup_cve"]),
        CweOf("cwe_of", "single_hop", "easy", 5, 1, ["lookup_cve"]),
        InKev("in_kev", "single_hop", "easy", 8, 1, ["lookup_cve"]),
        TechniqueTactics("technique_tactics", "single_hop", "easy", 4, 1, ["get_attack_technique"]),
        KevVendor("kev_vendor", "single_hop", "easy", 4, 1, ["get_source", "lookup_cve"]),
        DaysToKev("days_to_kev", "multi_hop", "medium", 8, 1, ["lookup_cve"]),
        VendorKevCount("vendor_kev_count", "multi_hop", "medium", 6, 2, ["get_source", "search_kev"]),
        CompareTwoCvss("compare_two_cvss", "multi_hop", "medium", 6, 2, ["lookup_cve"]),
        FindByNameThenCvss("find_by_name_then_cvss", "multi_hop", "hard", 8, 2, ["search_corpus", "lookup_cve"]),
        KevBeforePublished("kev_before_published", "multi_hop", "hard", 4, 1, ["lookup_cve"]),
        RemediationWindow("remediation_window", "multi_hop", "medium", 4, 1, ["get_source"]),
        VendorRansomwareCount("vendor_ransomware_count", "comparison", "medium", 8, 1, ["search_kev"]),
        KevMonthCount("kev_month_count", "comparison", "medium", 6, 1, ["search_kev"]),
        VendorYearCount("vendor_year_count", "comparison", "medium", 6, 1, ["search_kev"]),
        TopRansomwareVendor("top_ransomware_vendor", "comparison", "hard", 6, 3, ["search_kev"]),
        HighestCvssOfN("highest_cvss_of_n", "comparison", "hard", 6, 4, ["lookup_cve"]),
        CountInKev("count_in_kev", "comparison", "medium", 4, 5, ["lookup_cve"]),
        CvssRecordMissing("cvss_record_missing", "unanswerable", "medium", 8, 1, ["lookup_cve"]),
        CvssNotScored("cvss_not_scored", "unanswerable", "hard", 2, 1, ["lookup_cve"]),
        KevDateNotInKev("kev_date_not_in_kev", "unanswerable", "medium", 6, 1, ["lookup_cve"]),
        AttackMapping("attack_mapping", "unanswerable", "hard", 6, 1, ["lookup_cve", "search_corpus"]),
        EpssScore("epss_score", "unanswerable", "medium", 6, 1, ["lookup_cve"]),
    ]
}
