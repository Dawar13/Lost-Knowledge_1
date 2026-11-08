"""Gate G2 (P3): generality and who loses. A framing gate (docs/02), decided by the agent under
delegation with the rules fixed in notes/decisions.md before the runs.

  python -m mb5.g2 --full results/P3 --robust results/P3_robust \
      --models Qwen2.5-7B-Instruct,Qwen2.5-7B,OLMo-2-1124-7B-Instruct,OLMo-2-1124-7B

Writes results/gates/G2.md and <full>/g2/*.csv|json:
  popularity_gee.csv   GEE logistic (binomial, exchangeable within category) of robust unreachability at
                       p0.90_T1 on log10(1 + sitelinks) for recognised entities (all templates as rows),
                       and of recognition on log10(1 + sitelinks) for all members (E9)
  by_region.csv, by_type.csv   robust RBU at p0.90_T1 (and shipped where present) per region / entity type
  wording.csv          per-template robust RBU and CV on pilot_v2 (5 templates)
  temperature.csv      robust RBU at top-p 0.9 for T = 1.0, 0.7, 0.5
  validity.csv         recognition AUROC, recognised count, sample-check violations at scale
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .analysis import fmt, form_reachable, recognised_set, summarise_model
from .common import read_jsonl, write_json
from .metrics import write_csv
from .tailcat_grid import COUNTRIES


def cat_meta(cid: str) -> tuple[str, str]:
    """(entity type, region) from a generated category id 'type__country' or 'writers_lang__x'."""
    t, _, c = cid.partition("__")
    if t == "writers_lang":
        return "writers_by_language", "south_asia"
    return t, COUNTRIES.get(c, (None, None, "other"))[2]


def gee_logit(y, x, groups):
    import statsmodels.api as sm

    X = sm.add_constant(np.asarray(x, dtype=float))
    model = sm.GEE(np.asarray(y, dtype=float), X, groups=np.asarray(groups), family=sm.families.Binomial(),
                   cov_struct=sm.cov_struct.Exchangeable())
    r = model.fit()
    b, se, p = float(r.params[1]), float(r.bse[1]), float(r.pvalues[1])
    return {"slope_per_decade": b, "se": se, "p": p, "odds_ratio_per_decade": math.exp(b),
            "or_ci": (math.exp(b - 1.96 * se), math.exp(b + 1.96 * se)), "n_rows": len(y),
            "n_clusters": len(set(groups))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True)
    ap.add_argument("--robust", default=None)
    ap.add_argument("--models", required=True)
    ap.add_argument("--primary", default="Qwen2.5-7B-Instruct")
    args = ap.parse_args()
    full = Path(args.full)
    models = args.models.split(",")
    out = full / "g2"
    pop_rows, region_rows, type_rows, val_rows, temp_rows, word_rows = [], [], [], [], [], []
    for m in models:
        mdir = full / m
        reach = read_jsonl(mdir / "reach.jsonl.gz")
        recog = read_jsonl(mdir / "recog.jsonl.gz")
        rec, validity = recognised_set(recog)
        s, _, _, _ = summarise_model(mdir)
        # entity-template reachability, robust, any form
        tab = defaultdict(lambda: defaultdict(bool))
        sl = {}
        specs = [x for x in ("p0.90_T1", "shipped_hf", "p0.90_T0.7", "p0.90_T0.5", "k50_T1", "minp0.10_T1")
                 if any(f"reach__{x}" in r for r in reach[:20])]
        for r in reach:
            k = (r["qid"] + "|" + r["category"], r["template"])
            sl[k[0]] = r["sitelinks"]
            for x in specs:
                tab[k][x] |= form_reachable(r, x, "robust")
        # popularity: unreachability among recognised (all templates as rows) and recognition among members
        y, xs, g = [], [], []
        for (ek, t), d in tab.items():
            if rec.get(ek):
                y.append(int(not d["p0.90_T1"]))
                xs.append(math.log10(1 + sl[ek]))
                g.append(ek.split("|", 1)[1])
        pr = gee_logit(y, xs, g) | {"model": m, "outcome": "unreachable_p0.90_T1 | recognised"}
        yr = [int(bool(v)) for v in rec.values()]
        xr = [math.log10(1 + sl.get(k, 0)) for k in rec]
        gr = [k.split("|", 1)[1] for k in rec]
        rr = gee_logit(yr, xr, gr) | {"model": m, "outcome": "recognised | member"}
        pop_rows += [pr, rr]
        # region and type breakdowns
        for key_fn, sink, name in ((lambda c: cat_meta(c)[1], region_rows, "region"),
                                   (lambda c: cat_meta(c)[0], type_rows, "type")):
            grp = defaultdict(lambda: defaultdict(list))
            for (ek, t), d in tab.items():
                if rec.get(ek):
                    for x in specs:
                        grp[key_fn(ek.split("|", 1)[1])][x].append(int(not d[x]))
            for gk, dd in sorted(grp.items()):
                sink.append({"model": m, name: gk, "n_rows": len(dd["p0.90_T1"]),
                             **{f"rbu_{x}": float(np.mean(v)) for x, v in dd.items()}})
        sc = mdir / "sample_check.json"
        viol = bound = nsamp = None
        if sc.exists():
            rows = json.loads(sc.read_text())
            viol = sum(r["violations"] for r in rows)
            bound = sum(r.get("boundary_hits", 0) for r in rows)
            nsamp = sum(r["n_samples"] for r in rows)
        val_rows.append({"model": m, "auroc_yesno": validity["auroc_yesno"], "mc_accuracy": validity["mc_accuracy"],
                         "n_members": validity["n_members"], "n_recognised": validity["n_recognised"],
                         "n_categories": len({ek.split("|", 1)[1] for ek in rec}),
                         "sample_check_samples": nsamp, "violations": viol, "boundary_hits": bound,
                         **{f"rbu_{x}": s["specs"][x]["rbu_mean"] for x in specs if x in s["specs"]},
                         **{f"rbu_{x}_ci": s["specs"][x]["rbu_mean_ci"] for x in specs if x in s["specs"]}})
        temp_rows.append({"model": m, **{f"T{t}": s["specs"].get(x, {}).get("rbu_mean")
                                         for t, x in (("1.0", "p0.90_T1"), ("0.7", "p0.90_T0.7"),
                                                      ("0.5", "p0.90_T0.5"))}})
        if args.robust and (Path(args.robust) / m).exists():
            rs, _, _, _ = summarise_model(Path(args.robust) / m)
            from .metrics import Model, rbu_from_rows  # per-template RBU on the robustness run
            M = Model(Path(args.robust) / m)
            ent = M.entity_rows()
            cats = None
            pt = []
            for t in M.templates:
                per = rbu_from_rows(ent, "p0.90_T1", "robust", [t])
                cats = cats or list(per)
                pt.append(sum(per[c][1] for c in per) / sum(per[c][0] for c in per))
            word_rows.append({"model": m, **{f"template{t}": v for t, v in zip(M.templates, pt)},
                              "mean": float(np.mean(pt)), "cv": float(np.std(pt) / np.mean(pt))})

    write_csv(out / "popularity_gee.csv", pop_rows)
    write_csv(out / "by_region.csv", region_rows)
    write_csv(out / "by_type.csv", type_rows)
    write_csv(out / "validity.csv", val_rows)
    write_csv(out / "temperature.csv", temp_rows)
    write_csv(out / "wording.csv", word_rows)

    prim = next(r for r in pop_rows if r["model"] == args.primary and r["outcome"].startswith("unreachable"))
    tail_deleted = prim["slope_per_decade"] < 0 and prim["p"] < 0.01
    wording_robust = bool(word_rows) and all(r["cv"] < 0.25 for r in word_rows)
    v1_ok = all(r["violations"] in (None, 0) for r in val_rows)
    summary = {"tail_deleted_headline": tail_deleted, "wording_robust": wording_robust, "V1_at_scale": v1_ok,
               "multilingual": "not run in this pass (provisional translations only; see notes/decisions.md)",
               "popularity_primary": prim}
    write_json(out / "g2_summary.json", summary)

    pop_verdict = ('LEAD WITH "the long tail is deleted, not sampled less"' if tail_deleted
                   else "no popularity headline")
    L = ["# Gate G2 (framing): decided by the agent under delegation", "",
         f"- **Popularity:** {pop_verdict} "
         f"(primary {args.primary}: odds ratio of being unreachable per tenfold more sitelinks "
         f"{prim['odds_ratio_per_decade']:.2f}, 95% CI {prim['or_ci'][0]:.2f} to {prim['or_ci'][1]:.2f}, p {prim['p']:.2g}).",
         f"- **Wording:** {'wording-robust (CV < 0.25 for every model)' if wording_robust else 'report the spread across wordings'}.",
         f"- **V1 at scale:** {'0 robust violations' if v1_ok else 'VIOLATIONS: investigate before using these numbers'}.",
         "- **Multilingual:** not run in this pass; no multilingual headline.", "",
         "## Validity and headline RBU (robust, mean over templates, 95% cluster-bootstrap CI)", "",
         "| model | categories | recognised | AUROC | RBU p0.90_T1 | RBU shipped | RBU k50 | samples checked | violations | boundary hits |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in val_rows:
        ci = r.get("rbu_p0.90_T1_ci", (float("nan"),) * 2)
        L.append(f"| {r['model']} | {r['n_categories']} | {r['n_recognised']} | {r['auroc_yesno']:.3f} | "
                 f"{fmt(r.get('rbu_p0.90_T1'))} ({fmt(ci[0])} to {fmt(ci[1])}) | {fmt(r.get('rbu_shipped_hf'))} | "
                 f"{fmt(r.get('rbu_k50_T1'))} | {r['sample_check_samples'] or '-'} | {r['violations'] if r['violations'] is not None else '-'} | "
                 f"{r['boundary_hits'] if r['boundary_hits'] is not None else '-'} |")
    L += ["", "## Popularity (GEE logistic, clustered by category; x = log10(1 + sitelinks))", "",
          "| model | outcome | odds ratio per decade | 95% CI | p | rows | categories |", "|---|---|---|---|---|---|---|"]
    for r in pop_rows:
        L.append(f"| {r['model']} | {r['outcome']} | {r['odds_ratio_per_decade']:.3f} | {r['or_ci'][0]:.3f} to "
                 f"{r['or_ci'][1]:.3f} | {r['p']:.2g} | {r['n_rows']} | {r['n_clusters']} |")
    L += ["", "## By region (robust RBU at p0.90_T1, entity-template rows)", "", "| model | region | rows | RBU p0.90_T1 |",
          "|---|---|---|---|"]
    L += [f"| {r['model']} | {r['region']} | {r['n_rows']} | {fmt(r['rbu_p0.90_T1'])} |" for r in region_rows]
    L += ["", "## Temperature sweep (robust RBU at top-p 0.9)", "", "| model | T=1.0 | T=0.7 | T=0.5 |", "|---|---|---|---|"]
    L += [f"| {r['model']} | {fmt(r['T1.0'])} | {fmt(r['T0.7'])} | {fmt(r['T0.5'])} |" for r in temp_rows]
    if word_rows:
        L += ["", "## Wording (pilot_v2, robust RBU at p0.90_T1 per template)", "",
              "| model | " + " | ".join(f"t{i}" for i in range(5)) + " | CV |", "|---|" + "---|" * 6]
        L += [f"| {r['model']} | " + " | ".join(fmt(r.get(f'template{i}')) for i in range(5)) + f" | {r['cv']:.3f} |"
              for r in word_rows]
    L += ["", "Tables: results/P3/g2/. Rules: docs/02 (P3) as read in notes/decisions.md.", "",
          "Sign-off: ____________________"]
    Path("results/gates").mkdir(parents=True, exist_ok=True)
    Path("results/gates/G2.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
