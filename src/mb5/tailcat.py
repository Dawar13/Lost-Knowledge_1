"""Build TailCat: category member lists (with aliases and popularity) and
hard foils from Wikidata.

Each category in the YAML config gives:
  id, singular ("a river in Kerala"), plural ("rivers in Kerala"),
  where      SPARQL body binding ?item to members
  foil_where SPARQL body binding ?item to near-miss non-members (sibling categories)

Members are capped per category with stratification over popularity
(Wikipedia sitelink count) so the head and the tail are both present.
Foils that are also members (by QID or by label) are removed.

Usage:
  python -m mb5.tailcat --config configs/pilot_categories.yaml --out data/tailcat/pilot
Always read the printed sanity report: if a category returns 0 rows or junk,
the QID in the config is wrong. Fix it before running any model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path

import requests
import yaml

from .common import write_json, write_jsonl

ENDPOINT = "https://query.wikidata.org/sparql"
HEADERS = {"User-Agent": "MB5-reachability-research/0.1 (academic research script)",
           "Accept": "application/sparql-results+json"}

# hint:runFirst forces Blazegraph to evaluate the member subquery before the
# label/sitelink joins. Without it the optimiser reorders the joins and even
# tiny categories hit the 60 s gateway timeout (tested 2026-11-08: 504 after
# 65 s without the hint, 146 rows in 3 s with it).
QUERY = """
SELECT ?item ?label ?sitelinks (GROUP_CONCAT(DISTINCT ?alias; separator="|") AS ?aliases) WHERE {{
  {{ SELECT DISTINCT ?item WHERE {{ {where} }} LIMIT {limit} }} hint:Prior hint:runFirst true .
  ?item wikibase:sitelinks ?sitelinks .
  ?item rdfs:label ?label . FILTER(LANG(?label) = "{lang}")
  OPTIONAL {{ ?item skos:altLabel ?alias . FILTER(LANG(?alias) = "{lang}") }}
}} GROUP BY ?item ?label ?sitelinks
"""


def run_query(where: str, lang: str = "en", limit: int = 20000, retries: int = 4) -> list[dict]:
    q = QUERY.format(where=where, lang=lang, limit=limit)
    for attempt in range(retries):
        try:
            r = requests.get(ENDPOINT, params={"query": q}, headers=HEADERS, timeout=180)
            if r.status_code == 429:
                time.sleep(30 * (attempt + 1))
                continue
            r.raise_for_status()
            rows = []
            for b in r.json()["results"]["bindings"]:
                aliases = [a for a in b.get("aliases", {}).get("value", "").split("|") if a]
                rows.append({
                    "qid": b["item"]["value"].rsplit("/", 1)[-1],
                    "label": b["label"]["value"],
                    "aliases": aliases,
                    "sitelinks": int(b["sitelinks"]["value"]),
                })
            return rows
        except requests.RequestException as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            print(f"  query failed ({code or type(e).__name__}); attempt {attempt + 1}/{retries}", flush=True)
            if code == 504 and attempt >= 1:
                break  # the server's 60 s limit: a second timeout means the query plan is too slow
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("Wikidata query failed repeatedly; rewrite the WHERE clause (P131* and P279* paths "
                       "time out; use hint:Prior hint:runFirst on the most selective pattern).")


def cached_query(out: Path, cid: str, kind: str, where: str, lang: str, refresh: bool = False) -> list[dict]:
    """run_query with a per-category raw cache in <out>/raw/. The cache is keyed
    by the exact query text, so editing a category's SPARQL invalidates it. The
    raw files are also the provenance record: query text and UTC fetch time."""
    q = QUERY.format(where=where, lang=lang, limit=20000)
    key = hashlib.sha256(q.encode()).hexdigest()[:16]
    path = out / "raw" / f"{cid}.{kind}.{lang}.json"
    if not refresh and path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("key") == key:
            print(f"  {kind}: {len(d['rows'])} rows from cache ({d['fetched_utc']})", flush=True)
            return d["rows"]
    t = time.time()
    rows = run_query(where, lang)
    print(f"  {kind}: {len(rows)} rows in {time.time() - t:.1f}s", flush=True)
    write_json(path, {"key": key, "query": q, "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "rows": rows})
    time.sleep(2)  # be polite to the public endpoint
    return rows


_QID = re.compile(r"^Q\d+$")


def clean(rows: list[dict], max_aliases: int) -> list[dict]:
    out = []
    for r in rows:
        lab = r["label"].strip()
        if not lab or _QID.match(lab) or len(lab) > 60 or lab.isdigit():
            continue
        al = []
        for a in r["aliases"]:
            a = a.strip()
            if a and a.casefold() != lab.casefold() and not _QID.match(a) and len(a) <= 60 and a not in al:
                al.append(a)
        r = dict(r, label=lab, aliases=al[:max_aliases])
        out.append(r)
    return out


def stratified_cap(rows: list[dict], cap: int, bins: int, rng: random.Random) -> list[dict]:
    if len(rows) <= cap:
        return rows
    rows = sorted(rows, key=lambda r: (r["sitelinks"], r["qid"]))
    per = cap // bins
    out = []
    for b in range(bins):
        chunk = rows[b * len(rows) // bins : (b + 1) * len(rows) // bins]
        out.extend(rng.sample(chunk, min(per, len(chunk))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--cap", type=int, default=None, help="override per-category member cap")
    ap.add_argument("--only", default=None, help="comma-separated category ids")
    ap.add_argument("--refresh", action="store_true", help="ignore cached raw Wikidata results")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    s = cfg.get("settings", {})
    cap = args.cap or s.get("member_cap", 400)
    foil_cap = s.get("foil_cap", 200)
    bins = s.get("strata", 5)
    max_aliases = s.get("max_aliases", 3)
    seed = s.get("seed", 0)
    only = set(args.only.split(",")) if args.only else None

    out = Path(args.out)
    cats, members_all, foils_all, report, failed = [], [], [], [], []
    valid_names: dict[str, list[str]] = {}
    for c in cfg["categories"]:
        if only and c["id"] not in only:
            continue
        print(f"[tailcat] {c['id']}", flush=True)
        # One RNG per category: a category's sample does not depend on which
        # other categories were built, skipped or failed.
        rng = random.Random(f"{seed}:{c['id']}")
        try:
            mem = clean(cached_query(out, c["id"], "members", c["where"], args.lang, args.refresh), max_aliases)
            foil = clean(cached_query(out, c["id"], "foils", c["foil_where"], args.lang, args.refresh), 0)
        except RuntimeError as e:
            print(f"  FAILED: {e}", flush=True)
            failed.append(c["id"])
            continue
        mem_q = {m["qid"] for m in mem}
        mem_names = {m["label"].casefold() for m in mem} | {a.casefold() for m in mem for a in m["aliases"]}
        foil = [f for f in foil if f["qid"] not in mem_q and f["label"].casefold() not in mem_names]
        n_total = len(mem)
        # Uncapped name list: used to judge whether a generated name is a real member
        # (coverage in P4, invented-name rate in P5). Never judge validity against the capped list.
        valid_names[c["id"]] = sorted(mem_names)
        mem = stratified_cap(mem, cap, bins, rng)
        foil = rng.sample(foil, min(foil_cap, len(foil)))
        for m in mem:
            members_all.append({**m, "category": c["id"], "lang": args.lang})
        for f in foil:
            foils_all.append({"qid": f["qid"], "label": f["label"], "sitelinks": f["sitelinks"],
                              "category": c["id"], "lang": args.lang})
        cats.append({k: c[k] for k in ("id", "singular", "plural")} | {"control": bool(c.get("control", False)),
                                                                     "n_members_total": n_total})
        sl = sorted(m["sitelinks"] for m in mem) or [0]
        report.append({
            "category": c["id"], "members_total": n_total, "members_kept": len(mem), "foils_kept": len(foil),
            "sitelinks_median": sl[len(sl) // 2], "sitelinks_max": sl[-1],
            "example_members": [m["label"] for m in rng.sample(mem, min(6, len(mem)))],
            "example_foils": [f["label"] for f in rng.sample(foil, min(6, len(foil)))],
        })
    write_json(out / "categories.json", cats)
    write_jsonl(out / "entities.jsonl", members_all)
    write_jsonl(out / "foils.jsonl", foils_all)
    write_json(out / "build_report.json", report)
    write_json(out / "valid_names.json", valid_names)
    print("\n=== SANITY REPORT (read this) ===")
    for r in report:
        flag = "" if r["members_kept"] >= 30 and r["foils_kept"] >= 30 else "   <-- TOO SMALL: fix the QID/query"
        print(f"{r['category']:28s} members {r['members_kept']:4d}/{r['members_total']:5d}  foils {r['foils_kept']:4d}"
              f"  median sitelinks {r['sitelinks_median']}{flag}")
        print(f"    e.g. {r['example_members']}\n    foils {r['example_foils']}")
    for cid in failed:
        print(f"{cid:28s} FAILED: Wikidata query did not complete; rewrite its query (see build log)")
    print(f"\nWrote {len(members_all)} members, {len(foils_all)} foils to {out}")
    if failed:
        raise SystemExit(f"{len(failed)} categories failed: {failed}. Fix their queries and re-run "
                         "(successful categories are cached under raw/).")


if __name__ == "__main__":
    main()
