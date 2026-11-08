"""Sensitivity of the invented-name rate to answer matching, and the hand-audit sheet (P5, docs/02).

The pre-registered automatic rule (mb5.frontier.classify) counts as "invented" every non-empty answer
that matches no member and no foil. Inspection shows that class mixes (a) real members written another
way ("Everest" for "Mount Everest", "Periyar" for "Periyar River") or missing from Wikidata's membership,
(b) wrong answers (a Bengali poet named as a Tamil poet) and (c) made-up or garbled names. This module
(1) recomputes the rate with a looser, still conservative match: an answer also counts as valid if, after
dropping generic type words, its word set equals or is contained in a member name's word set (or the
reverse), with at least one shared word of five or more letters; and (2) writes a sheet of 200
"invented" answers (stratified over samplers) for a human to label as real member / wrong answer /
made up / not a name. The pre-registered rate stays the primary G3 reading; this is reported beside it.

  python -m mb5.invented_audit --results results/P5 --data data/tailcat/pilot_v2 \
      --models Qwen2.5-7B-Instruct,OLMo-2-1124-7B-Instruct
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .common import read_jsonl, write_json
from .metrics import write_csv
from .sample_check import norm

TYPE_WORDS = {"mount", "mt", "lake", "river", "temple", "university", "the", "of", "sri", "shri", "shree",
              "arulmigu", "kovil", "koil", "himal", "peak", "järvi", "jarvi", "national", "federal", "state"}


def words(s: str) -> frozenset:
    return frozenset(w for w in re.findall(r"[\w']+", norm(s)) if w not in TYPE_WORDS)


def loose_valid(text: str, names: list[frozenset]) -> bool:
    a = words(text)
    if not a:
        return False
    for b in names:
        if b and (a <= b or b <= a) and any(len(w) >= 5 for w in a & b):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--n-audit", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    data = Path(args.data)
    valid = json.loads((data / "valid_names.json").read_text(encoding="utf-8"))
    name_words = {c: [words(n) for n in ns] for c, ns in valid.items()}
    rows, audit_pool = [], []
    for m in args.models.split(","):
        mdir = Path(args.results) / m
        samples = read_jsonl(mdir / "samples_frontier.jsonl.gz")
        by = defaultdict(Counter)
        for s in samples:
            c = s["class"]
            if c == "invented" and loose_valid(s["text"], name_words.get(s["category"], [])):
                c = "valid_loose"
            by[s["spec"]][c] += 1
            if c == "invented":
                audit_pool.append({"model": m, "spec": s["spec"], "category": s["category"], "answer": s["text"]})
        for spec, cnt in by.items():
            ne = sum(cnt.values()) - cnt["empty"]
            rows.append({"model": m, "spec": spec, "invented_rate_preregistered": (cnt["invented"] + cnt["valid_loose"]) / ne,
                         "invented_rate_loose": cnt["invented"] / ne, "valid_loose_share": cnt["valid_loose"] / ne,
                         "n_nonempty": ne})
        ref = next(r for r in rows if r["model"] == m and r["spec"] == "reference")
        for r in rows:
            if r["model"] == m:
                r["loose_delta_vs_reference"] = r["invented_rate_loose"] - ref["invented_rate_loose"]
    out = Path(args.results) / "audit"
    write_csv(out / "invented_rate_sensitivity.csv", rows)
    # audit sheet: distinct answers, stratified over (model, spec), seeded
    rng = random.Random(args.seed)
    distinct = {}
    for a in audit_pool:
        distinct.setdefault((a["model"], a["spec"], a["category"], a["answer"]), a)
    strata = defaultdict(list)
    for a in distinct.values():
        strata[(a["model"], a["spec"])].append(a)
    keys = sorted(strata)
    sheet = []
    while len(sheet) < args.n_audit and any(strata.values()):
        for k in keys:
            if strata[k] and len(sheet) < args.n_audit:
                sheet.append(strata[k].pop(rng.randrange(len(strata[k]))))
    for a in sheet:
        a["human_label"] = ""  # real_member | wrong_answer | made_up | not_a_name
        a["notes"] = ""
    write_csv(out / "invented_audit_sheet.csv", sheet)
    write_json(out / "README.json", {"what": "invented-name audit sheet for a human; labels: real_member, "
                                     "wrong_answer, made_up, not_a_name", "n": len(sheet), "seed": args.seed})
    for r in rows:
        if r["spec"] in ("reference", "relax_first", "untruncated_T1"):
            print(f"[audit] {r['model']} {r['spec']:14s} pre-registered {r['invented_rate_preregistered']:.3f} "
                  f"loose {r['invented_rate_loose']:.3f} (delta vs reference {r['loose_delta_vs_reference']:+.3f})")


if __name__ == "__main__":
    main()
