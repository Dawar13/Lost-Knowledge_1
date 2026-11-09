"""Region composition analysis (CPU only; POST HOC and exploratory).

G2 (results/gates/G2.md) found robust RBU at p0.90_T1 lower for South Asian, African and Southeast Asian
categories than for European and North American ones. This was not predicted. The agent's guess was
"composition": Western categories have more recognised members competing for the same few first pieces,
so a larger share of them cannot fit. This module tests that guess, and the alternatives we can test from
stored data, after the pattern was seen. Nothing here is pre-registered and no gate depends on it.

  python -m mb5.region --full results/P3 --data data/tailcat/full_v1_s150 --out results/analysis_cpu/region

Rows: one per (recognised entity, template), outcome = robustly unreachable (all surface forms cut even with
the target raised by the margin; the headline mode). GEE logistic, binomial, exchangeable within category
(the setup of mb5.g2). Region dummies with Europe as reference. The "gap" is the contrast
mean(south_asia, africa, southeast_asia) - mean(europe, north_america) on the log-odds scale.

Models (same rows throughout, so coefficients are comparable apart from non-collapsibility, see notes):
  M0  region only
  Mc  + log(number of recognised members in the category, this model)        composition
  Mf  + first-piece concentration of the category prompt at that template      shortlist width
      (top-1 mass and log kept pieces under top-p 0.9, from first_token.json)
  Me  + entity level: log10(1 + sitelinks), within-category sitelink percentile, label token count,
      number of surface forms
  Mall  all of the above + entity-type fixed effects

Outputs (<out>/): gee_gap.csv, gee_region_terms.csv, by_region.csv, stratified.csv, categories.csv,
paired_region.csv, summary.json, manifest.json.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .analysis import auroc, form_reachable, recognised_set
from .common import code_version, read_jsonl, write_json
from .metrics import sha256, write_csv

REF = "europe"
WEST = ("europe", "north_america")
SOUTH = ("south_asia", "africa", "southeast_asia")
REGIONS = ("europe", "north_america", "east_asia", "west_asia", "latin_america", "south_asia", "southeast_asia",
           "africa")
ENTITY_COVS = ("log_sitelinks", "sitelink_pct", "label_tokens", "n_forms")
FIRST_COVS = ("top1_mass", "log_kept_p90")
STEPS = (("M0", ()), ("Mc", ("log_n_recognised",)), ("Mf", FIRST_COVS), ("Me", ENTITY_COVS),
         ("Mall", ("log_n_recognised", *FIRST_COVS, *ENTITY_COVS, "TYPE_FE")))
DEFAULT_MODELS = {"Qwen2.5-7B-Instruct": ("p0.90_T1", "shipped_hf"), "Qwen2.5-7B": ("p0.90_T1",),
                  "OLMo-2-1124-7B-Instruct": ("p0.90_T1",), "OLMo-2-1124-7B": ("p0.90_T1",)}
PAIRS = (("Qwen2.5-7B-Instruct", "Qwen2.5-7B"), ("OLMo-2-1124-7B-Instruct", "OLMo-2-1124-7B"))


# ---------------------------------------------------------------- loading (streamed: reach files are large)
def load_reach(path: Path, specs) -> tuple[dict, dict]:
    """(entity key, template) -> {spec: reachable (robust, any form)}; and -> {label_tokens, n_forms}."""
    reach = defaultdict(lambda: {s: False for s in specs})
    info = defaultdict(lambda: {"label_tokens": None, "n_forms": 0})
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            k = (r["qid"] + "|" + r["category"], r["template"])
            d = reach[k]
            for s in specs:
                d[s] = d[s] or form_reachable(r, s, "robust")
            info[k]["n_forms"] += 1
            if r.get("is_label"):
                info[k]["label_tokens"] = r["L"]
    return dict(reach), dict(info)


def first_piece(path: Path) -> dict:
    """(category, template) -> {top1_mass, log_kept_p90}; empty if the file is absent."""
    if not path.exists():
        return {}
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = d["rows"] if isinstance(d, dict) else d
    return {(r["category"], r["template"]): {"top1_mass": r["top1_mass"], "log_kept_p90": math.log(r["kept_top_p_0.9"])}
            for r in rows}


def sitelink_percentiles(ents: list[dict]) -> dict:
    """entity key -> mid-rank percentile of sitelinks among the category's members in the split (ties averaged)."""
    by = defaultdict(list)
    for e in ents:
        by[e["category"]].append((e["sitelinks"], e["qid"]))
    out = {}
    for cat, xs in by.items():
        vals = np.array([v for v, _ in xs], dtype=float)
        order = vals.argsort(kind="mergesort")
        ranks = np.empty(len(vals))
        ranks[order] = np.arange(1, len(vals) + 1)
        for v in np.unique(vals):  # average ties
            m = vals == v
            ranks[m] = ranks[m].mean()
        for (v, q), rk in zip(xs, ranks):
            out[q + "|" + cat] = (rk - 0.5) / len(vals)
    return out


# ---------------------------------------------------------------- statistics
def gap_contrast(names: list[str]) -> np.ndarray:
    """Contrast vector over design columns: mean of SOUTH region dummies minus mean of WEST (Europe = 0)."""
    c = np.zeros(len(names))
    for r in SOUTH:
        c[names.index(f"region[{r}]")] += 1 / len(SOUTH)
    for r in WEST:
        if r != REF:
            c[names.index(f"region[{r}]")] -= 1 / len(WEST)
    return c


def design(rows: list[dict], covs, types_ref: str | None = None) -> tuple[np.ndarray, list[str]]:
    """Columns: const, region dummies (reference Europe), centred numeric covariates, type dummies if TYPE_FE."""
    cols, names = [np.ones(len(rows))], ["const"]
    for r in REGIONS:
        if r != REF:
            cols.append(np.array([float(x["region"] == r) for x in rows]))
            names.append(f"region[{r}]")
    for cv in covs:
        if cv == "TYPE_FE":
            types = sorted({x["type"] for x in rows})
            ref = types_ref if types_ref in types else types[0]
            for t in types:
                if t != ref:
                    cols.append(np.array([float(x["type"] == t) for x in rows]))
                    names.append(f"type[{t}]")
            continue
        v = np.array([x[cv] for x in rows], dtype=float)
        cols.append(v - v.mean())
        names.append(cv)
    return np.column_stack(cols), names


def fit_gee(y, X, groups, names, cov: str = "exchangeable") -> dict:
    """GEE logistic clustered by category with robust (sandwich) SEs. cov="exchangeable" is the mb5.g2 setup,
    started from the GLM fit; cov="independence" is the sensitivity that always converges (point estimates =
    GLM, SEs still cluster-robust). Non-finite exchangeable fits are reported as not converged."""
    import statsmodels.api as sm
    from scipy.stats import norm

    y = np.asarray(y, dtype=float)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        start = sm.GLM(y, X, family=sm.families.Binomial()).fit().params
        cs = sm.cov_struct.Exchangeable() if cov == "exchangeable" else sm.cov_struct.Independence()
        model = sm.GEE(y, X, groups=np.asarray(groups), family=sm.families.Binomial(), cov_struct=cs)
        r = model.fit(maxiter=100, start_params=start)
    b, V = np.asarray(r.params), np.asarray(r.cov_params())
    c = gap_contrast(names)
    est, se = float(c @ b), float(math.sqrt(max(c @ V @ c, 0.0))) if np.all(np.isfinite(V)) else float("nan")
    ok = math.isfinite(est) and math.isfinite(se)
    dep = r.cov_struct.dep_params if cov == "exchangeable" else None
    out = {"converged": ok, "gap_logodds": est if ok else float("nan"), "gap_se": se if ok else float("nan"),
           "gap_or": math.exp(est) if ok else float("nan"),
           "gap_or_ci": (math.exp(est - 1.96 * se), math.exp(est + 1.96 * se)) if ok else (float("nan"),) * 2,
           "gap_p": float(2 * norm.sf(abs(est / se))) if ok and se > 0 else float("nan"),
           "terms": {n: {"coef": float(bi), "se": float(math.sqrt(max(V[i, i], 0.0))) if np.isfinite(V[i, i])
                         else float("nan"), "p": float(pi)}
                     for i, (n, bi, pi) in enumerate(zip(names, b, np.asarray(r.pvalues)))},
           "warnings": sorted({str(x.message)[:120] for x in w}),
           "dep_param": float(np.asarray(dep).ravel()[0]) if dep is not None else None}
    return out


# ---------------------------------------------------------------- per model
def model_rows(mdir: Path, spec: str, cats: dict, pct: dict, reach: dict, info: dict, rec: dict, fp: dict):
    """Rows for recognised (entity, template) pairs with every covariate; None-covariate rows are dropped."""
    n_rec = Counter(ek.split("|", 1)[1] for ek, ok in rec.items() if ok)
    rows, dropped = [], Counter()
    for (ek, t), d in reach.items():
        if not rec.get(ek):
            continue
        cat = ek.split("|", 1)[1]
        meta = cats[cat]
        f = fp.get((cat, t))
        lt = info[(ek, t)]["label_tokens"]
        if f is None:
            dropped["no_first_piece"] += 1
            continue
        if lt is None:
            dropped["no_label_row"] += 1
            continue
        rows.append({"key": ek, "template": t, "category": cat, "region": meta["region"], "type": meta["type"],
                     "y": int(not d[spec]), "log_sitelinks": math.log10(1 + d["_sitelinks"]),
                     "sitelink_pct": pct[ek], "label_tokens": lt, "n_forms": info[(ek, t)]["n_forms"],
                     "log_n_recognised": math.log(n_rec[cat]), "n_recognised": n_rec[cat], **f})
    return rows, dropped


def by_region_table(model: str, spec: str, rows: list[dict], recog: list[dict], rec: dict, members: dict,
                    info: dict, fp: dict, cats: dict) -> list[dict]:
    out = []
    for reg in REGIONS:
        rr = [r for r in rows if r["region"] == reg]
        cs = sorted({c for c, m in cats.items() if m["region"] == reg})
        mem_v = [x["V"] for x in recog if x["kind"] == "member" and cats[x["category"]]["region"] == reg]
        foil_v = [x["V"] for x in recog if x["kind"] == "foil" and cats[x["category"]]["region"] == reg]
        n_mem = sum(members[c] for c in cs)
        n_recd = sum(1 for ek, ok in rec.items() if ok and cats[ek.split("|", 1)[1]]["region"] == reg)
        per_cat = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # cat -> template -> [recognised, reachable]
        for r in rr:
            per_cat[r["category"]][r["template"]][0] += 1
            per_cat[r["category"]][r["template"]][1] += 1 - r["y"]
        n_rec_cat = [np.mean([v[0] for v in tt.values()]) for tt in per_cat.values()]
        n_reach_cat = [np.mean([v[1] for v in tt.values()]) for tt in per_cat.values()]
        fps = [v for (c, t), v in fp.items() if c in cs]
        out.append({"model": model, "spec": spec, "region": reg, "n_categories": len(cs),
                    "n_categories_with_recognised": len(per_cat), "rows": len(rr),
                    "rbu": float(np.mean([r["y"] for r in rr])) if rr else float("nan"),
                    "members": n_mem, "recognised": n_recd, "recognition_rate": n_recd / n_mem if n_mem else float("nan"),
                    "auroc_yesno": auroc(mem_v, foil_v) if mem_v and foil_v else float("nan"),
                    "median_recognised_per_category": float(np.median(n_rec_cat)) if n_rec_cat else float("nan"),
                    "mean_recognised_per_category": float(np.mean(n_rec_cat)) if n_rec_cat else float("nan"),
                    "median_reachable_per_category": float(np.median(n_reach_cat)) if n_reach_cat else float("nan"),
                    "mean_reachable_per_category": float(np.mean(n_reach_cat)) if n_reach_cat else float("nan"),
                    "mean_label_tokens": float(np.mean([r["label_tokens"] for r in rr])) if rr else float("nan"),
                    "share_with_aliases": float(np.mean([r["n_forms"] > 1 for r in rr])) if rr else float("nan"),
                    "mean_n_forms": float(np.mean([r["n_forms"] for r in rr])) if rr else float("nan"),
                    "mean_log_sitelinks": float(np.mean([r["log_sitelinks"] for r in rr])) if rr else float("nan"),
                    "mean_sitelink_pct": float(np.mean([r["sitelink_pct"] for r in rr])) if rr else float("nan"),
                    "first_piece_top1_mass_mean": float(np.mean([v["top1_mass"] for v in fps])) if fps else float("nan"),
                    "first_piece_kept_p90_median": float(np.median([math.exp(v["log_kept_p90"]) for v in fps]))
                    if fps else float("nan")})
    return out


def category_table(model: str, spec: str, rows: list[dict], cats: dict, members: dict, fp: dict) -> list[dict]:
    per = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        per[r["category"]][r["template"]][0] += 1
        per[r["category"]][r["template"]][1] += 1 - r["y"]
    out = []
    for cat, tt in sorted(per.items()):
        fps = [fp[(cat, t)] for t in tt if (cat, t) in fp]
        n_rec = np.mean([v[0] for v in tt.values()])
        n_reach = np.mean([v[1] for v in tt.values()])
        out.append({"model": model, "spec": spec, "category": cat, "region": cats[cat]["region"],
                    "type": cats[cat]["type"], "members": members[cat], "n_members_total": cats[cat].get("n_members_total"),
                    "recognised": float(n_rec), "reachable_recognised_mean": float(n_reach),
                    "rbu": float(1 - n_reach / n_rec) if n_rec else float("nan"),
                    "top1_mass_mean": float(np.mean([v["top1_mass"] for v in fps])) if fps else float("nan"),
                    "kept_p90_median": float(np.median([math.exp(v["log_kept_p90"]) for v in fps])) if fps else float("nan")})
    return out


def stratified(model: str, spec: str, cat_rows: list[dict], edges) -> list[dict]:
    """Row-weighted RBU of WEST vs SOUTH categories within bins of recognised members per category."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        for grp, regs in (("west", WEST), ("south", SOUTH)):
            cs = [c for c in cat_rows if c["region"] in regs and lo <= c["recognised"] < hi]
            n = sum(c["recognised"] for c in cs)
            unreach = sum(c["recognised"] - c["reachable_recognised_mean"] for c in cs)
            out.append({"model": model, "spec": spec, "bin": f"[{lo:g}, {hi:g})", "group": grp, "categories": len(cs),
                        "recognised_rows_mean": n, "rbu": unreach / n if n else float("nan"),
                        "mean_reachable_per_category": float(np.mean([c["reachable_recognised_mean"] for c in cs]))
                        if cs else float("nan")})
    return out


def paired(inst: str, base: str, R: dict, spec: str, cats: dict) -> list[dict]:
    """Among (entity, template) recognised by both models: RBU of each, and the post-training loss rate
    P(instruct unreachable | base reachable), per region."""
    ri, rb = R[inst], R[base]
    agg = defaultdict(lambda: Counter())
    for k, di in ri["reach"].items():
        ek = k[0]
        if not (ri["rec"].get(ek) and rb["rec"].get(ek)) or k not in rb["reach"]:
            continue
        reg = cats[ek.split("|", 1)[1]]["region"]
        bi, bb = di[spec], rb["reach"][k][spec]
        for g in (reg, "WEST" if reg in WEST else "SOUTH" if reg in SOUTH else "OTHER", "ALL"):
            a = agg[g]
            a["n"] += 1
            a["inst_unreach"] += int(not bi)
            a["base_unreach"] += int(not bb)
            a["base_reach"] += int(bb)
            a["lost"] += int(bb and not bi)
            a["gained"] += int(bi and not bb)
    out = []
    for g in (*REGIONS, "WEST", "SOUTH", "OTHER", "ALL"):
        a = agg.get(g)
        if not a:
            continue
        out.append({"pair": f"{inst} vs {base}", "spec": spec, "region": g, "rows_recognised_by_both": a["n"],
                    "rbu_instruct": a["inst_unreach"] / a["n"], "rbu_base": a["base_unreach"] / a["n"],
                    "loss_rate_given_base_reachable": a["lost"] / a["base_reach"] if a["base_reach"] else float("nan"),
                    "gained_rows": a["gained"]})
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default="results/P3")
    ap.add_argument("--data", default="data/tailcat/full_v1_s150")
    ap.add_argument("--out", default="results/analysis_cpu/region")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    args = ap.parse_args()
    full, data, out = Path(args.full), Path(args.data), Path(args.out)
    cats = {c["id"]: c for c in json.loads((data / "categories.json").read_text(encoding="utf-8"))}
    ents = read_jsonl(data / "entities.jsonl")
    members = Counter(e["category"] for e in ents)
    sl = {e["qid"] + "|" + e["category"]: e["sitelinks"] for e in ents}
    pct = sitelink_percentiles(ents)

    inputs = {str(data / "categories.json"): sha256(data / "categories.json"),
              str(data / "entities.jsonl"): sha256(data / "entities.jsonl")}
    R, gap_rows, term_rows, region_rows, cat_rows_all, strat_rows, checks = {}, [], [], [], [], [], {}
    summary = {"post_hoc": True, "reference_region": REF, "west": WEST, "south": SOUTH, "models": {}}
    g2_region = {}
    g2p = full / "g2" / "by_region.csv"
    if g2p.exists():
        import csv

        for r in csv.DictReader(open(g2p, encoding="utf-8")):
            g2_region[(r["model"], r["region"])] = r
        inputs[str(g2p)] = sha256(g2p)

    for m in args.models.split(","):
        specs = DEFAULT_MODELS.get(m, ("p0.90_T1",))
        mdir = full / m
        for fn in ("reach.jsonl.gz", "recog.jsonl.gz", "first_token.json"):
            inputs[str(mdir / fn)] = sha256(mdir / fn)
        print(f"[region] loading {m}", flush=True)
        reach, info = load_reach(mdir / "reach.jsonl.gz", specs)
        for k, d in reach.items():
            d["_sitelinks"] = sl[k[0]]
        recog = read_jsonl(mdir / "recog.jsonl.gz")
        rec, validity = recognised_set(recog)
        fp = first_piece(mdir / "first_token.json")
        R[m] = {"reach": reach, "rec": rec}
        summary["models"][m] = {"first_token_json": bool(fp), "n_recognised": validity["n_recognised"], "specs": {}}
        for spec in specs:
            rows, dropped = model_rows(mdir, spec, cats, pct, reach, info, rec, fp)
            y = [r["y"] for r in rows]
            groups = [r["category"] for r in rows]
            reg_tab = by_region_table(m, spec, rows, recog, rec, members, info, fp, cats)
            region_rows += reg_tab
            # reproduction check against G2's by_region table (same rows, same verdicts)
            col = f"rbu_{spec}"
            diffs = [abs(t["rbu"] - float(g2_region[(m, t["region"])][col])) for t in reg_tab
                     if (m, t["region"]) in g2_region and g2_region[(m, t["region"])].get(col)]
            checks[f"{m}|{spec}|by_region_matches_g2"] = (max(diffs) < 1e-6) if diffs else None
            crow = category_table(m, spec, rows, cats, members, fp)
            cat_rows_all += crow
            q = np.quantile([c["recognised"] for c in crow], [0.25, 0.5, 0.75])
            edges = [0, *[float(x) for x in q], float("inf")]
            strat_rows += stratified(m, spec, crow, edges)
            fits, fits_ind = {}, {}
            for name, covs in STEPS:
                X, names = design(rows, covs, types_ref="rivers")
                print(f"[region] {m} {spec} {name}: {X.shape}", flush=True)
                f = fit_gee(y, X, groups, names)
                fi = fit_gee(y, X, groups, names, cov="independence")
                fits[name] = f
                fits_ind[name] = fi
                gap_rows.append({"model": m, "spec": spec, "step": name, "covariates": "+".join(covs) or "none",
                                 "rows": len(rows), "categories": len(set(groups)), "gap_logodds": f["gap_logodds"],
                                 "gap_se": f["gap_se"], "gap_or": f["gap_or"], "gap_or_ci_lo": f["gap_or_ci"][0],
                                 "gap_or_ci_hi": f["gap_or_ci"][1], "gap_p": f["gap_p"],
                                 "share_of_M0_gap": f["gap_logodds"] / fits["M0"]["gap_logodds"]
                                 if fits["M0"]["gap_logodds"] else float("nan"),
                                 "converged": f["converged"], "exchangeable_rho": f["dep_param"],
                                 "indep_gap_or": fi["gap_or"], "indep_gap_or_ci_lo": fi["gap_or_ci"][0],
                                 "indep_gap_or_ci_hi": fi["gap_or_ci"][1], "indep_gap_p": fi["gap_p"],
                                 "indep_share_of_M0_gap": fi["gap_logodds"] / fits_ind["M0"]["gap_logodds"],
                                 "warnings": " | ".join(f["warnings"])})
                for n, t in f["terms"].items():
                    if n.startswith("region[") or n in ("log_n_recognised", *FIRST_COVS, *ENTITY_COVS):
                        term_rows.append({"model": m, "spec": spec, "step": name, "term": n, "coef": t["coef"],
                                          "se": t["se"], "or": math.exp(t["coef"]),
                                          "or_ci_lo": math.exp(t["coef"] - 1.96 * t["se"]),
                                          "or_ci_hi": math.exp(t["coef"] + 1.96 * t["se"]), "p": t["p"]})
            summary["models"][m]["specs"][spec] = {
                "rows": len(rows), "dropped": dict(dropped),
                "gap": {k: {"or": v["gap_or"], "ci": v["gap_or_ci"], "p": v["gap_p"], "converged": v["converged"],
                            "rho": v["dep_param"],
                            "share_of_M0": v["gap_logodds"] / fits["M0"]["gap_logodds"]} for k, v in fits.items()},
                "gap_independence": {k: {"or": v["gap_or"], "ci": v["gap_or_ci"], "p": v["gap_p"],
                                         "share_of_M0": v["gap_logodds"] / fits_ind["M0"]["gap_logodds"]}
                                     for k, v in fits_ind.items()},
                "composition_term_Mc": fits["Mc"]["terms"]["log_n_recognised"]}
    for inst, base in PAIRS:
        if inst in R and base in R:
            pr = paired(inst, base, R, "p0.90_T1", cats)
            summary.setdefault("paired", {})[f"{inst} vs {base}"] = {r["region"]: r for r in pr}
            paired_rows = summary.setdefault("_paired_rows", [])
            paired_rows += pr
    write_csv(out / "gee_gap.csv", gap_rows)
    write_csv(out / "gee_region_terms.csv", term_rows)
    write_csv(out / "by_region.csv", region_rows)
    write_csv(out / "categories.csv", cat_rows_all)
    write_csv(out / "stratified.csv", strat_rows)
    write_csv(out / "paired_region.csv", summary.pop("_paired_rows", []))
    summary["checks"] = checks
    write_json(out / "summary.json", summary)
    import statsmodels

    write_json(out / "manifest.json", {"inputs_sha256": inputs, "code": code_version(),
                                       "statsmodels": statsmodels.__version__, "numpy": np.__version__,
                                       "command": "python -m mb5.region " + " ".join(
                                           f"--{k} {v}" for k, v in vars(args).items())})
    print(json.dumps({"checks": checks}, indent=1))


if __name__ == "__main__":
    main()
