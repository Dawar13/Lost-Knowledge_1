"""Multilingual invented-name metric (notes/decisions.md, 2026-11-09, fixed before any G3b data).

The pre-registered P5 classification (mb5.frontier.classify) matches a sampled answer against English
labels and aliases only: capped members (entities.jsonl), the uncapped member names (valid_names.json)
and foil labels (foils.jsonl). About 6% of Qwen's answers are written in Chinese, Japanese or Korean
script, and some of them are real members (Japanese novelists in kanji), which that rule counts as
invented. The multilingual rule keeps the English rule as it is and adds every label and alias that
Wikidata holds in any language for the same entities:

  member   the answer equals any-language label or alias of a capped member (the English index first,
           so English matches keep their member QID)
  valid    ... of any uncapped member (the raw member lists), or an English uncapped name
  foil     ... of a foil
  invented otherwise

Matching uses the answer candidates of mb5.frontier (first line, sample_check.norm: NFKC, casefold,
outer punctuation stripped; the part before ", " or " (" and without a leading "the "). Two CJK rules:
text containing CJK script is compared without whitespace, and a CJK answer is also cut at its first
"(" (NFKC turns a full-width parenthesis into "(" with no space before it, e.g. a name followed by a
note). Names shorter than two characters after normalisation are not indexed. A name that belongs
to two different capped members of one category is ambiguous: it gives no member QID, but as the
name of real members it still counts as valid.

  python -m mb5.invented_ml fetch --data data/tailcat/pilot_v2 --out data/tailcat/pilot_v2_labels_ml
  python -m mb5.invented_ml analyse --data data/tailcat/pilot_v2 --labels data/tailcat/pilot_v2_labels_ml \
      --p5 results/P5 --models Qwen2.5-7B-Instruct,OLMo-2-1124-7B-Instruct --out results/analysis_cpu/invented_ml

Wikidata output is data: names are only compared as strings, never executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from .common import code_version, read_jsonl, write_json, write_jsonl
from .sample_check import norm

# ---------------------------------------------------------------- normalisation
_CJK = re.compile("[ᄀ-ᇿ぀-ヿ㄰-㆏㐀-䶿一-鿿가-힯豈-﫿]")
_WS = re.compile(r"\s+")


def is_cjk(s: str) -> bool:
    return bool(_CJK.search(s))


def ck(x: str) -> str:
    """Comparison key of an already normalised string: CJK text without whitespace."""
    return _WS.sub("", x) if is_cjk(x) else x


def script(text: str) -> str:
    """Coarse script of an answer's first line: cjk if any CJK character, else latin if every letter
    is Latin, else other (Tamil, Bengali, Cyrillic, ...), or none if it has no letters."""
    t = norm(text)
    if is_cjk(t):
        return "cjk"
    letters = [c for c in t if c.isalpha()]
    if not letters:
        return "none"
    return "latin" if all(unicodedata.name(c, "").startswith("LATIN") for c in letters) else "other"


def candidates(t: str) -> list[str]:
    """mb5.frontier._candidates, plus the part before the first "(" for CJK text."""
    from .frontier import _candidates
    c = _candidates(t)
    if is_cjk(t) and "(" in t:
        head = t.split("(")[0].strip()
        if head and head not in c:
            c.append(head)
    return c


# ---------------------------------------------------------------- Wikidata fetch
LABEL_QUERY = """SELECT ?item ?l ?lang ?kind WHERE {{
  VALUES ?item {{ {values} }}
  {{ ?item rdfs:label ?l . BIND("label" AS ?kind) }} UNION {{ ?item skos:altLabel ?l . BIND("alias" AS ?kind) }}
  BIND(LANG(?l) AS ?lang)
}}"""


def split_qids(data: Path) -> dict[str, set]:
    """The QIDs whose names the metric uses: capped members, foils and uncapped members (raw lists)."""
    ents = read_jsonl(data / "entities.jsonl")
    foils = read_jsonl(data / "foils.jsonl")
    raw = set()
    for p in sorted((data / "raw").glob("*.members.*.json")):
        raw |= {r["qid"] for r in json.loads(p.read_text(encoding="utf-8"))["rows"]}
    return {"members": {e["qid"] for e in ents}, "foils": {f["qid"] for f in foils}, "uncapped": raw}


def fetch(data: Path, out: Path, batch: int = 200) -> dict:
    from .tailcat import cached_sparql
    q = split_qids(data)
    qids = sorted(set().union(*q.values()), key=lambda x: int(x[1:]))
    labels: dict[str, list] = defaultdict(list)
    for i in range(0, len(qids), batch):
        part = qids[i:i + batch]
        rows = cached_sparql(out, f"labels_{i // batch:04d}", LABEL_QUERY.format(values=" ".join("wd:" + x for x in part)))
        for r in rows:
            labels[r["item"]].append([r.get("lang", ""), r["kind"], r["l"]])
    recs = [{"qid": x, "names": sorted({tuple(n) for n in labels.get(x, [])})} for x in qids]
    write_jsonl(out / "labels.jsonl.gz", recs)
    langs = Counter(n[0] for r in recs for n in r["names"])
    rep = {"source_split": str(data).replace("\\", "/"), "qids": len(qids), "qids_with_names": sum(bool(r["names"]) for r in recs),
           "names": sum(len(r["names"]) for r in recs), "languages": len(langs),
           "top_languages": langs.most_common(20), "by_set": {k: len(v) for k, v in q.items()},
           "batch": batch, "code": code_version()}
    write_json(out / "build_report.json", rep)
    return rep


def load_labels(path: Path) -> dict[str, list[str]]:
    """qid -> every label and alias text, any language."""
    return {r["qid"]: [n[2] for n in r["names"]] for r in read_jsonl(path / "labels.jsonl.gz")}


# ---------------------------------------------------------------- matcher
class Matcher:
    """English indices exactly as mb5.frontier builds them, plus optional any-language indices."""

    def __init__(self, data: Path, ml_names: dict[str, list[str]] | None = None):
        ents = read_jsonl(data / "entities.jsonl")
        foils = read_jsonl(data / "foils.jsonl")
        self.member = defaultdict(dict)
        for e in ents:
            for form in [e["label"]] + e.get("aliases", []):
                self.member[e["category"]][norm(form)] = e["qid"]
        self.valid = {c: {norm(n) for n in names}
                      for c, names in json.loads((data / "valid_names.json").read_text(encoding="utf-8")).items()}
        self.foil = defaultdict(set)
        for f in foils:
            self.foil[f["category"]].add(norm(f["label"]))
        self.ml = ml_names is not None
        self.ambiguous = Counter()
        self.ml_member, self.ml_valid, self.ml_foil = defaultdict(dict), defaultdict(set), defaultdict(set)
        if not self.ml:
            return

        def keys(qid):
            out = set()
            for n in ml_names.get(qid, []):
                k = ck(norm(n))
                if len(k) >= 2 and not k.isdigit() and not re.fullmatch(r"q\d+", k):
                    out.add(k)
            return out

        seen = defaultdict(dict)  # category -> key -> set of qids
        for e in ents:
            for k in keys(e["qid"]) | {ck(norm(f)) for f in [e["label"]] + e.get("aliases", [])}:
                seen[e["category"]].setdefault(k, set()).add(e["qid"])
        for cat, d in seen.items():
            for k, qs in d.items():
                if len(qs) == 1:
                    self.ml_member[cat][k] = next(iter(qs))
                else:
                    self.ambiguous[cat] += 1
        for p in sorted((data / "raw").glob("*.members.*.json")):
            cat = p.name.split(".")[0]
            for r in json.loads(p.read_text(encoding="utf-8"))["rows"]:
                self.ml_valid[cat] |= keys(r["qid"])
        for f in foils:
            self.ml_foil[f["category"]] |= keys(f["qid"]) | {ck(norm(f["label"]))}

    def classify(self, text: str, category: str) -> tuple[str, str | None]:
        from .frontier import classify
        if not self.ml:
            return classify(text, self.member[category], self.valid.get(category, set()), self.foil[category])
        t = norm(text)
        if not t:
            return "empty", None
        cands = candidates(t)
        mem, mlm = self.member[category], self.ml_member[category]
        for x in cands:
            if x in mem:
                return "member", mem[x]
        for x in cands:
            if ck(x) in mlm:
                return "member", mlm[ck(x)]
        val, mlv = self.valid.get(category, set()), self.ml_valid[category]
        if any(x in val or ck(x) in mlv for x in cands):
            return "valid", None
        fo, mlf = self.foil[category], self.ml_foil[category]
        if any(x in fo or ck(x) in mlf for x in cands):
            return "foil", None
        return "invented", None


# ---------------------------------------------------------------- analysis
CLASSES = ("member", "valid", "foil", "invented", "empty")


def rates(rows: list[dict], key: str, qkey: str) -> dict:
    cnt = Counter(r[key] for r in rows)
    ne = sum(cnt.values()) - cnt["empty"]
    return {"n_nonempty": ne, **{f"{c}_rate": (cnt[c] / ne if ne else None) for c in ("member", "valid", "foil", "invented")},
            "distinct_members": len({(r["category"], r[qkey]) for r in rows if r[key] == "member"})}


def reclassify(samples: list[dict], m_en: Matcher, m_ml: Matcher) -> tuple[list[dict], int]:
    """Per sample: the English class (must equal the stored P5 class) and the multilingual class."""
    out, mismatches = [], 0
    for r in samples:
        c_en, q_en = m_en.classify(r["text"], r["category"])
        if (c_en, q_en) != (r["class"], r["member_qid"]):
            mismatches += 1
        c_ml, q_ml = m_ml.classify(r["text"], r["category"])
        out.append({"spec": r["spec"], "category": r["category"], "i": r["i"], "class_en": c_en,
                    "member_qid_en": q_en, "class_ml": c_ml, "member_qid_ml": q_ml, "script": script(r["text"])})
    return out, mismatches


def audit_sheet(pool: list[dict], n: int, seed: int) -> list[dict]:
    """The rule of mb5.invented_audit: distinct (model, spec, category, answer), stratified over
    (model, spec) in sorted order, one random draw per stratum per round, seeded."""
    rng = random.Random(seed)
    distinct = {}
    for a in pool:
        distinct.setdefault((a["model"], a["spec"], a["category"], a["answer"]), a)
    strata = defaultdict(list)
    for a in distinct.values():
        strata[(a["model"], a["spec"])].append(a)
    keys = sorted(strata)
    sheet = []
    while len(sheet) < n and any(strata.values()):
        for k in keys:
            if strata[k] and len(sheet) < n:
                sheet.append(strata[k].pop(rng.randrange(len(strata[k]))))
    return [dict(a, human_label="", notes="") for a in sheet]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def analyse(data: Path, labels: Path, p5: Path, models: list[str], out: Path, n_audit: int = 200, seed: int = 0) -> dict:
    from .metrics import write_csv
    ml_names = load_labels(labels)
    m_en, m_ml = Matcher(data), Matcher(data, ml_names)
    table, g3, reclass, scripts, inputs = [], {}, [], [], {}
    pool = []
    for m in models:
        f = p5 / m / "samples_frontier.jsonl.gz"
        inputs[str(f).replace("\\", "/")] = _sha(f)
        samples = read_jsonl(f)
        rows, mism = reclassify(samples, m_en, m_ml)
        if mism:
            raise SystemExit(f"{m}: the English matcher disagrees with the stored P5 class on {mism} samples")
        write_jsonl(out / f"samples_classes_{m}.jsonl.gz", rows)
        texts = {(r["spec"], r["category"], r["i"]): r["text"] for r in samples}
        by_spec = defaultdict(list)
        for r in rows:
            by_spec[r["spec"]].append(r)
        ref_en = rates(by_spec["reference"], "class_en", "member_qid_en")
        ref_ml = rates(by_spec["reference"], "class_ml", "member_qid_ml")
        fr = {x["spec"]: x for x in json.loads((p5 / m / "frontier.json").read_text())["rows"]}
        for s, rs in by_spec.items():
            en, ml = rates(rs, "class_en", "member_qid_en"), rates(rs, "class_ml", "member_qid_ml")
            ne = [r for r in rs if r["class_en"] != "empty"]
            table.append({
                "model": m, "spec": s, "n_nonempty": en["n_nonempty"],
                "rbu_robust": fr[s]["rbu_robust"], "recovered_vs_reference": fr[s]["recovered_vs_reference"],
                "invented_rate_en": en["invented_rate"], "invented_rate_ml": ml["invented_rate"],
                "invented_delta_vs_reference_en": en["invented_rate"] - ref_en["invented_rate"],
                "invented_delta_vs_reference_ml": ml["invented_rate"] - ref_ml["invented_rate"],
                "member_rate_en": en["member_rate"], "member_rate_ml": ml["member_rate"],
                "valid_rate_en": en["valid_rate"], "valid_rate_ml": ml["valid_rate"],
                "foil_rate_en": en["foil_rate"], "foil_rate_ml": ml["foil_rate"],
                "distinct_members_en": en["distinct_members"], "distinct_members_ml": ml["distinct_members"],
                "cjk_share": sum(r["script"] == "cjk" for r in ne) / len(ne) if ne else None,
                "other_script_share": sum(r["script"] == "other" for r in ne) / len(ne) if ne else None,
                "frontier_invented_rate_reproduced": abs(en["invented_rate"] - fr[s]["invented_rate"]) < 1e-12,
            })
        relax = next(t for t in table if t["model"] == m and t["spec"] == "relax_first")
        g3[m] = {"recovered": relax["recovered_vs_reference"],
                 "invented_delta_en": relax["invented_delta_vs_reference_en"],
                 "invented_delta_ml": relax["invented_delta_vs_reference_ml"],
                 "passes_en": bool(relax["recovered_vs_reference"] >= 0.5 and relax["invented_delta_vs_reference_en"] <= 0.02),
                 "passes_ml": bool(relax["recovered_vs_reference"] >= 0.5 and relax["invented_delta_vs_reference_ml"] <= 0.02)}
        ne_all = [r for r in rows if r["class_en"] != "empty"]
        moved = Counter((r["script"], r["class_ml"]) for r in ne_all if r["class_en"] == "invented" and r["class_ml"] != "invented")
        for (sc, c), n in sorted(moved.items()):
            reclass.append({"model": m, "script": sc, "from": "invented", "to": c, "answers": n,
                            "distinct_answers": len({(r["category"], ck(norm(texts[(r["spec"], r["category"], r["i"])])))
                                                     for r in ne_all if r["class_en"] == "invented"
                                                     and r["class_ml"] == c and r["script"] == sc})})
        sc_all = Counter(r["script"] for r in ne_all)
        sc_inv_en = Counter(r["script"] for r in ne_all if r["class_en"] == "invented")
        sc_inv_ml = Counter(r["script"] for r in ne_all if r["class_ml"] == "invented")
        for sc in sorted(sc_all):
            scripts.append({"model": m, "script": sc, "answers": sc_all[sc], "share_of_nonempty": sc_all[sc] / len(ne_all),
                            "invented_en": sc_inv_en[sc], "invented_ml": sc_inv_ml[sc]})
        for r in ne_all:
            if r["class_ml"] == "invented":
                pool.append({"model": m, "spec": r["spec"], "category": r["category"],
                             "answer": texts[(r["spec"], r["category"], r["i"])]})
    write_csv(out / "frontier_ml.csv", table)
    write_csv(out / "reclassified.csv", reclass)
    write_csv(out / "scripts.csv", scripts)
    write_json(out / "g3_ml.json", {"g3": g3, "note": "recovery does not depend on the invented-name metric; "
                                    "G3 keeps its decision (notes/decisions.md 2026-11-09)"})
    sheet = audit_sheet(pool, n_audit, seed)
    write_csv(out / "invented_audit_sheet_ml.csv", sheet)
    write_json(out / "invented_audit_README_ml.json", {
        "what": "invented-name audit sheet for a human, drawn from answers that are invented under the multilingual "
                "metric; labels: real_member, wrong_answer, made_up, not_a_name",
        "n": len(sheet), "seed": seed, "rule": "mb5.invented_audit stratification over (model, spec), distinct answers"})
    for p in [labels / "labels.jsonl.gz", data / "entities.jsonl", data / "foils.jsonl", data / "valid_names.json"]:
        inputs[str(p).replace("\\", "/")] = _sha(p)
    amb = {k: v for k, v in m_ml.ambiguous.items()}
    write_json(out / "manifest.json", {"inputs": inputs, "code": code_version(), "ambiguous_ml_member_names": amb,
                                       "models": models})
    return {"table": table, "g3": g3, "reclassified": reclass, "scripts": scripts}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--data", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--batch", type=int, default=200)
    a = sub.add_parser("analyse")
    a.add_argument("--data", required=True)
    a.add_argument("--labels", required=True)
    a.add_argument("--p5", required=True)
    a.add_argument("--models", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--n-audit", type=int, default=200)
    a.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "fetch":
        rep = fetch(Path(args.data), Path(args.out), args.batch)
        print(json.dumps({k: v for k, v in rep.items() if k != "code"}, ensure_ascii=False, indent=1))
    else:
        res = analyse(Path(args.data), Path(args.labels), Path(args.p5), args.models.split(","), Path(args.out),
                      args.n_audit, args.seed)
        for t in res["table"]:
            if t["spec"] in ("reference", "relax_first", "untruncated_T1", "k100_T1"):
                print(f"{t['model']:26s} {t['spec']:14s} invented en {t['invented_rate_en']:.3f} ml {t['invented_rate_ml']:.3f} "
                      f"delta en {t['invented_delta_vs_reference_en']:+.3f} ml {t['invented_delta_vs_reference_ml']:+.3f} "
                      f"members en {t['distinct_members_en']} ml {t['distinct_members_ml']}")
        print(json.dumps(res["g3"], indent=1))


if __name__ == "__main__":
    main()
