"""E13 ask-directly control: one unique Wikidata cue per entity (pre-registered 2026-11-09,
notes/decisions.md).

A cue is a question whose only answer is the entity:
  us_state_capitals       the state           "What is the capital of {cue}?"
  bengali_film_directors  a Bengali-language film whose only director is the person
                                              "Who directed the Bengali film {cue}?"
  japanese_novelists, tamil_poets, assamese_writers
                          a notable work (P800) whose only author (P50) is the person
                                              "Who wrote {cue}?"
"Only" is checked on Wikidata: the state has exactly one capital (P36), the film exactly
one director (P57), the work exactly one author (P50). The cue needs an English label.
Among an entity's valid cues the one with most sitelinks wins, then the lowest QID number.
Entities without a valid cue are skipped and counted, with the reason, in build_report.json.

One exclusion beyond the pre-registration (flagged in the build report): a cue whose label
contains one of the entity's own names (e.g. a film titled after its director) gives the
answer away, so it is skipped as "cue_contains_answer".

  python -m mb5.cues --data data/tailcat/pilot_v2 --out data/tailcat/pilot_v2_cues
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .common import read_jsonl, write_json, write_jsonl
from .prompts import BASE_GEN_SHOTS, render
from .sample_check import norm
from .tailcat import cached_sparql

QUESTIONS = {
    "capital": "What is the capital of {cue}?",
    "director": "Who directed the Bengali film {cue}?",
    "work": "Who wrote {cue}?",
}
CUE_INSTRUCTION = "Reply with only the name."  # the same instruction as GEN_TEMPLATES[0]

CATEGORY_KIND = {
    "us_state_capitals": "capital",
    "bengali_film_directors": "director",
    "japanese_novelists": "work",
    "tamil_poets": "work",
    "assamese_writers": "work",
}

# ?item is the TailCat entity; ?cue the cue entity; ?other every answer the cue has on the
# same property (uniqueness = exactly one).
_LINK = {
    "capital": "?cue wdt:P36 ?item ; wdt:P31 wd:Q35657 .\n  ?cue wdt:P36 ?other .",
    "director": "?cue wdt:P57 ?item ; wdt:P31 wd:Q11424 ; wdt:P364 wd:Q9610 .\n  ?cue wdt:P57 ?other .",
    "work": "?item wdt:P800 ?cue .\n  ?cue wdt:P50 ?item .\n  ?cue wdt:P50 ?other .",
}

CUE_QUERY = """
SELECT ?item ?cue ?cueLabel ?cueSitelinks (COUNT(DISTINCT ?other) AS ?nAnswers) WHERE {{
  VALUES ?item {{ {values} }}
  {link}
  ?cue wikibase:sitelinks ?cueSitelinks .
  OPTIONAL {{ ?cue rdfs:label ?cueLabel . FILTER(LANG(?cueLabel) = "en") }}
}} GROUP BY ?item ?cue ?cueLabel ?cueSitelinks
"""


def cue_query(kind: str, qids: list[str]) -> str:
    return CUE_QUERY.format(values=" ".join(f"wd:{q}" for q in sorted(qids)), link=_LINK[kind])


def question(kind: str, cue_label: str) -> str:
    return QUESTIONS[kind].format(cue=cue_label)


def render_cue_prompt(tok, q: str, chat: bool) -> str:
    """The cued generation prompt: the cue question plus the template-0 instruction, in the
    model's chat template (base models: the same Question/Answer frame and shot as run_model)."""
    return render(tok, f"{q} {CUE_INSTRUCTION}", chat, "" if chat else BASE_GEN_SHOTS)


def _qnum(q: str) -> int:
    return int(q[1:]) if q[1:].isdigit() else 10**12


def choose_cues(entity: dict, rows: list[dict]) -> tuple[dict | None, str | None, dict]:
    """rows: query rows for this entity. Returns (chosen cue row or None, skip reason, counts)."""
    names = [n for n in [entity["label"]] + list(entity.get("aliases", [])) if len(norm(n)) >= 4]
    cands = {}
    for r in rows:
        cands.setdefault(r["cue"], r)  # one row per cue (GROUP BY includes the label)
    counts = {"n_candidate_cues": len(cands)}
    if not cands:
        return None, "no_candidate", counts
    unique = [r for r in cands.values() if int(r["nAnswers"]) == 1]
    counts["n_unique_cues"] = len(unique)
    if not unique:
        return None, "not_unique", counts
    labelled = [r for r in unique if r.get("cueLabel")]
    if not labelled:
        return None, "no_english_label", counts
    fair = [r for r in labelled if not any(norm(n) in norm(r["cueLabel"]) for n in names)]
    if not fair:
        return None, "cue_contains_answer", counts
    best = sorted(fair, key=lambda r: (-int(r["cueSitelinks"]), _qnum(r["cue"])))[0]
    return best, None, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="TailCat split, e.g. data/tailcat/pilot_v2")
    ap.add_argument("--out", required=True, help="e.g. data/tailcat/pilot_v2_cues")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    ents = read_jsonl(data / "entities.jsonl")
    by_cat = defaultdict(list)
    for e in ents:
        if e["category"] in CATEGORY_KIND:
            by_cat[e["category"]].append(e)
    rows_out, report = [], {"source_split": str(data).replace("\\", "/"), "categories": {},
                            "rule": "notes/decisions.md 2026-11-09 (E13); module docstring for the cue definitions"}
    for cid, kind in CATEGORY_KIND.items():
        es = by_cat.get(cid, [])
        if not es:
            report["categories"][cid] = {"entities": 0, "kept": 0, "note": "category not in the split"}
            continue
        print(f"[cues] {cid} ({kind}, {len(es)} entities)", flush=True)
        raw = cached_sparql(out, f"{cid}.cues", cue_query(kind, [e["qid"] for e in es]), refresh=args.refresh)
        per = defaultdict(list)
        for r in raw:
            per[r["item"]].append(r)
        skipped = defaultdict(int)
        kept = 0
        for e in sorted(es, key=lambda e: _qnum(e["qid"])):
            best, why, counts = choose_cues(e, per.get(e["qid"], []))
            if best is None:
                skipped[why] += 1
                continue
            kept += 1
            rows_out.append({"qid": e["qid"], "category": cid, "label": e["label"], "kind": kind,
                             "cue_qid": best["cue"], "cue_label": best["cueLabel"],
                             "cue_sitelinks": int(best["cueSitelinks"]), "question": question(kind, best["cueLabel"]),
                             **counts})
        report["categories"][cid] = {"kind": kind, "entities": len(es), "kept": kept,
                                     "skipped": dict(sorted(skipped.items())),
                                     "query_rows": len(raw)}
        print(f"  kept {kept} of {len(es)}; skipped {dict(skipped)}", flush=True)
    report["total_kept"] = len(rows_out)
    write_jsonl(out / "cues.jsonl", rows_out)
    write_json(out / "build_report.json", report)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
