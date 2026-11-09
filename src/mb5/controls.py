"""Controls requested by internal review round 1 (notes/drafts/review_round1.md): CPU only, POST HOC.

  python -m mb5.controls --full results/P3 --data data/tailcat/full_v1_s150 \
      --mechanism results/G1/mechanism.csv --region results/analysis_cpu/region --out results/analysis_cpu/controls

(1) popularity_controls.csv: the G2 popularity headline (robust unreachability at the sampler, among
    recognised entities, all templates as rows, GEE logistic, binomial, exchangeable within category, as in
    mb5.g2) re-fitted with label token count and entity-type fixed effects added; P4 also adds region.
    Steps: P0 log10(1 + sitelinks) only (reproduces results/P3/g2/popularity_gee.csv); P1 + label tokens;
    P2 + type fixed effects; P3 + label tokens + type fixed effects; P4 = P3 + region fixed effects.
(2) mechanism_controls.csv: the first-token mechanism correlation (first-token top-1 mass of the category
    prompt against robust RBU at p0.90_T1), pooled 0.817 in results/G1/mechanism.csv over six models, broken
    down: Spearman within each model; mean of the within-model Spearman; pooled Spearman of within-model ranks;
    Pearson of within-model demeaned values (a model fixed effect); OLS slope with model fixed effects; and,
    for the OLMo-2 stages, within-category correlations across stages. 95% CIs by bootstrap over categories
    (categories resampled with replacement, kept paired across models). At scale the same within-model
    Spearman uses the 277 P3 categories (results/analysis_cpu/region/categories.csv; top-1 mass averaged over
    templates; RBU over recognised entity-template rows).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from .analysis import recognised_set
from .common import code_version, read_jsonl, write_json
from .metrics import sha256, write_csv
from .region import DEFAULT_MODELS, REGIONS, load_reach

B = 2000
SEED = 0


# ---------------------------------------------------------------- (1) popularity with controls
TYPE_REF = "rivers"  # the largest type, as in mb5.region


def pop_design(rows, label: bool, types: bool, region: bool):
    """Numeric columns are centred (slopes unchanged; better conditioning for the GEE iterations)."""
    s = np.array([r["log_sitelinks"] for r in rows], dtype=float)
    cols, names = [np.ones(len(rows)), s - s.mean()], ["const", "log10_sitelinks"]
    if label:
        v = np.array([r["label_tokens"] for r in rows], dtype=float)
        cols.append(v - v.mean())
        names.append("label_tokens")
    if types:
        ts = sorted({r["type"] for r in rows})
        ref = TYPE_REF if TYPE_REF in ts else ts[0]
        for t in [t for t in ts if t != ref]:
            cols.append(np.array([float(r["type"] == t) for r in rows]))
            names.append(f"type[{t}]")
    if region:
        for g in REGIONS[1:]:
            cols.append(np.array([float(r["region"] == g) for r in rows]))
            names.append(f"region[{g}]")
    return np.column_stack(cols), names


def gee_slope(y, X, groups, names, term="log10_sitelinks"):
    """GEE as in mb5.g2 (exchangeable within category, robust SEs), started from the GLM fit, plus the
    independence working correlation as a sensitivity that always converges (point estimate = GLM, SEs still
    cluster-robust). A non-finite exchangeable fit is reported as not converged, never silently replaced."""
    import statsmodels.api as sm

    y = np.asarray(y, float)
    start = sm.GLM(y, X, family=sm.families.Binomial()).fit().params
    i = names.index(term)
    out = {}
    for lab, cs in (("", sm.cov_struct.Exchangeable()), ("indep_", sm.cov_struct.Independence())):
        r = sm.GEE(y, X, groups=np.asarray(groups), family=sm.families.Binomial(), cov_struct=cs).fit(
            maxiter=100, start_params=start)
        b, se, p = float(r.params[i]), float(r.bse[i]), float(r.pvalues[i])
        ok = math.isfinite(b) and math.isfinite(se)
        out |= {f"{lab}converged": ok, f"{lab}coef": b if ok else float("nan"), f"{lab}se": se if ok else float("nan"),
                f"{lab}p": p if ok else float("nan"), f"{lab}or": math.exp(b) if ok else float("nan"),
                f"{lab}or_ci_lo": math.exp(b - 1.96 * se) if ok else float("nan"),
                f"{lab}or_ci_hi": math.exp(b + 1.96 * se) if ok else float("nan")}
        if not lab:
            dep = r.cov_struct.dep_params
            out["exchangeable_rho"] = float(np.asarray(dep).ravel()[0]) if dep is not None and ok else float("nan")
        if "label_tokens" in names and ok:
            j = names.index("label_tokens")
            out |= {f"{lab}label_tokens_or": math.exp(float(r.params[j])), f"{lab}label_tokens_p": float(r.pvalues[j])}
    return out


POP_STEPS = (("P0", False, False, False), ("P1", True, False, False), ("P2", False, True, False),
             ("P3", True, True, False), ("P4", True, True, True))


def popularity(full: Path, data: Path, models, inputs) -> list[dict]:
    cats = {c["id"]: c for c in json.loads((data / "categories.json").read_text(encoding="utf-8"))}
    sl = {e["qid"] + "|" + e["category"]: e["sitelinks"] for e in read_jsonl(data / "entities.jsonl")}
    g2 = {}
    g2p = full / "g2" / "popularity_gee.csv"
    if g2p.exists():
        inputs[str(g2p)] = sha256(g2p)
        for r in csv.DictReader(open(g2p, encoding="utf-8")):
            if r["outcome"].startswith("unreachable"):
                g2[r["model"]] = float(r["odds_ratio_per_decade"])
    out = []
    for m in models:
        specs = DEFAULT_MODELS.get(m, ("p0.90_T1",))
        mdir = full / m
        for fn in ("reach.jsonl.gz", "recog.jsonl.gz"):
            inputs[str(mdir / fn)] = sha256(mdir / fn)
        print(f"[controls] popularity {m}", flush=True)
        reach, info = load_reach(mdir / "reach.jsonl.gz", specs)
        rec, _ = recognised_set(read_jsonl(mdir / "recog.jsonl.gz"))
        for spec in specs:
            rows, no_label = [], 0
            for (ek, t), d in reach.items():
                if not rec.get(ek):
                    continue
                if info[(ek, t)]["label_tokens"] is None:
                    no_label += 1
                    continue
                cat = ek.split("|", 1)[1]
                rows.append({"y": int(not d[spec]), "log_sitelinks": math.log10(1 + sl[ek]),
                             "label_tokens": info[(ek, t)]["label_tokens"], "type": cats[cat]["type"],
                             "region": cats[cat]["region"], "category": cat})
            y, g = [r["y"] for r in rows], [r["category"] for r in rows]
            for step, lab, typ, reg in POP_STEPS:
                X, names = pop_design(rows, lab, typ, reg)
                f = gee_slope(y, X, g, names)
                out.append({"model": m, "spec": spec, "step": step, "label_tokens": lab, "type_fe": typ,
                            "region_fe": reg, "rows": len(rows), "rows_without_label_form": no_label,
                            "categories": len(set(g)), **f,
                            "g2_reference_or": g2.get(m) if spec == "p0.90_T1" else None})
                print(f"  {m} {spec} {step}: OR {f['or']:.3f} ({f['or_ci_lo']:.3f} to {f['or_ci_hi']:.3f}); "
                      f"independence {f['indep_or']:.3f}", flush=True)
    return out


# ---------------------------------------------------------------- (2) mechanism
def _within_stats(tab: dict, models: list[str], cats: list[str]) -> dict:
    """tab[model][cat] = (x, y). Statistics with a model fixed effect."""
    per = {}
    for m in models:
        xs = [tab[m][c][0] for c in cats if c in tab[m]]
        ys = [tab[m][c][1] for c in cats if c in tab[m]]
        per[m] = float(spearmanr(xs, ys).statistic) if len(xs) >= 3 and np.std(xs) > 0 and np.std(ys) > 0 else float("nan")
    rx, ry, dx, dy, X, Y, M = [], [], [], [], [], [], []
    for m in models:
        xs = np.array([tab[m][c][0] for c in cats if c in tab[m]])
        ys = np.array([tab[m][c][1] for c in cats if c in tab[m]])
        if len(xs) < 2:
            continue
        rx += list(rankdata(xs) / len(xs))
        ry += list(rankdata(ys) / len(ys))
        dx += list(xs - xs.mean())
        dy += list(ys - ys.mean())
        X += list(xs)
        Y += list(ys)
        M += [m] * len(xs)
    within_rank = float(np.corrcoef(rx, ry)[0, 1])
    within_pearson = float(np.corrcoef(dx, dy)[0, 1])
    slope = float(np.dot(dx, dy) / np.dot(dx, dx))  # OLS slope with model fixed effects
    pooled = float(spearmanr(X, Y).statistic)
    vals = [v for v in per.values() if not math.isnan(v)]
    return {"pooled_spearman": pooled, "per_model": per, "mean_within_spearman": float(np.mean(vals)),
            "within_rank_corr": within_rank, "within_pearson_model_fe": within_pearson, "fe_slope": slope}


def _stage_stats(tab: dict, stages: list[str], cats: list[str]) -> dict:
    """Within-category association across stages (category fixed effect)."""
    per_cat, dx, dy = [], [], []
    for c in cats:
        xs = np.array([tab[s][c][0] for s in stages if c in tab[s]])
        ys = np.array([tab[s][c][1] for s in stages if c in tab[s]])
        if len(xs) < 3:
            continue
        if np.std(xs) > 0 and np.std(ys) > 0:
            per_cat.append(float(spearmanr(xs, ys).statistic))
        dx += list(xs - xs.mean())
        dy += list(ys - ys.mean())
    return {"mean_within_category_spearman": float(np.mean(per_cat)) if per_cat else float("nan"),
            "share_categories_positive": float(np.mean([v > 0 for v in per_cat])) if per_cat else float("nan"),
            "n_categories": len(per_cat),
            "within_category_pearson": float(np.corrcoef(dx, dy)[0, 1]) if len(dx) > 2 else float("nan")}


def _boot(fn, tab, keys, cats, b=B, seed=SEED) -> dict:
    """Bootstrap over categories (paired across models): returns key -> (lo, hi)."""
    rng = np.random.default_rng(seed)
    acc = defaultdict(list)
    for _ in range(b):
        draw = list(rng.choice(len(cats), size=len(cats), replace=True))
        # duplicate categories get distinct names so they count as separate rows
        names = [f"{cats[i]}#{j}" for j, i in enumerate(draw)]
        t2 = {k: {n: tab[k][cats[i]] for n, i in zip(names, draw) if cats[i] in tab[k]} for k in keys}
        try:
            res = fn(t2, keys, names)
        except Exception:  # degenerate draw (e.g. constant ranks)
            continue
        for k, v in _flatten(res).items():
            if not (isinstance(v, float) and math.isnan(v)):
                acc[k].append(v)
    return {k: (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) for k, v in acc.items()}


def _flatten(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out |= _flatten(v, f"{pre}{k}.")
        else:
            out[f"{pre}{k}"] = v
    return out


def mechanism_pilot(path: Path) -> tuple[list[dict], dict]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    tab = defaultdict(dict)
    for r in rows:
        tab[f"{r['family']}:{r['stage']}"][r["category"]] = (float(r["top1_mass"]), float(r["rbu_p0.90_T1"]))
    models = sorted(tab)
    cats = sorted({r["category"] for r in rows})
    est = _within_stats(tab, models, cats)
    ci = _boot(_within_stats, tab, models, cats)
    olmo = [f"olmo2:{s}" for s in ("base", "sft", "dpo", "instruct") if f"olmo2:{s}" in tab]
    st = _stage_stats(tab, olmo, cats)
    st_ci = _boot(lambda t, k, c: _stage_stats(t, k, c), tab, olmo, cats)
    qwen = [m for m in models if m.startswith("qwen2.5:")]
    qw = _within_stats(tab, qwen, cats) if len(qwen) == 2 else None
    out = [_row("pilot_v2 (results/G1/mechanism.csv), 6 models x 11 categories", "pooled Spearman (as reported)",
                est["pooled_spearman"], ci.get("pooled_spearman"), len(rows))]
    for m in models:
        out.append(_row("pilot_v2", f"within {m}: Spearman over 11 categories", est["per_model"][m],
                        ci.get(f"per_model.{m}"), len(tab[m])))
    for k, lab in (("mean_within_spearman", "mean of the six within-model Spearman"),
                   ("within_rank_corr", "pooled correlation of within-model ranks (model fixed effect)"),
                   ("within_pearson_model_fe", "Pearson of within-model demeaned values (model fixed effect)"),
                   ("fe_slope", "OLS slope of RBU on top-1 mass with model fixed effects")):
        out.append(_row("pilot_v2", lab, est[k], ci.get(k), len(rows)))
    for k, lab in (("mean_within_category_spearman", "OLMo-2 stages: mean within-category Spearman over 4 stages"),
                   ("share_categories_positive", "OLMo-2 stages: share of categories with a positive association"),
                   ("within_category_pearson", "OLMo-2 stages: Pearson with category fixed effects")):
        out.append(_row("pilot_v2", lab, st[k], st_ci.get(k), st["n_categories"]))
    return out, {"pilot": est, "pilot_ci": ci, "olmo_stages": st, "olmo_stages_ci": st_ci, "qwen_pair": qw}


def mechanism_scale(cat_csv: Path) -> tuple[list[dict], dict]:
    rows = [r for r in csv.DictReader(open(cat_csv, encoding="utf-8")) if r["spec"] == "p0.90_T1"]
    tab = defaultdict(dict)
    for r in rows:
        x, y = float(r["top1_mass_mean"]), float(r["rbu"])
        if not (math.isnan(x) or math.isnan(y)):
            tab[r["model"]][r["category"]] = (x, y)
    models = sorted(tab)
    cats = sorted({c for m in models for c in tab[m]})
    est = _within_stats(tab, models, cats)
    ci = _boot(_within_stats, tab, models, cats, b=1000)
    out = [_row("full TailCat P3 (277 categories), 4 models", "pooled Spearman", est["pooled_spearman"],
                ci.get("pooled_spearman"), sum(len(v) for v in tab.values()))]
    for m in models:
        out.append(_row("full TailCat P3", f"within {m}: Spearman over categories", est["per_model"][m],
                        ci.get(f"per_model.{m}"), len(tab[m])))
    for k, lab in (("mean_within_spearman", "mean of the within-model Spearman"),
                   ("within_rank_corr", "pooled correlation of within-model ranks (model fixed effect)"),
                   ("within_pearson_model_fe", "Pearson of within-model demeaned values (model fixed effect)"),
                   ("fe_slope", "OLS slope of RBU on top-1 mass with model fixed effects")):
        out.append(_row("full TailCat P3", lab, est[k], ci.get(k), sum(len(v) for v in tab.values())))
    return out, {"scale": est, "scale_ci": ci}


def _row(data, stat, v, ci, n):
    ci = ci or (float("nan"), float("nan"))
    return {"data": data, "statistic": stat, "value": v, "ci_lo": ci[0], "ci_hi": ci[1], "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default="results/P3")
    ap.add_argument("--data", default="data/tailcat/full_v1_s150")
    ap.add_argument("--mechanism", default="results/G1/mechanism.csv")
    ap.add_argument("--region", default="results/analysis_cpu/region")
    ap.add_argument("--out", default="results/analysis_cpu/controls")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--skip-popularity", action="store_true")
    ap.add_argument("--skip-mechanism", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    inputs = {}
    summary = {"post_hoc": True, "requested_by": "notes/drafts/review_round1.md (internal review round 1)"}
    if not args.skip_mechanism:
        mp = Path(args.mechanism)
        inputs[str(mp)] = sha256(mp)
        mrows, msum = mechanism_pilot(mp)
        cc = Path(args.region) / "categories.csv"
        if cc.exists():
            inputs[str(cc)] = sha256(cc)
            srows, ssum = mechanism_scale(cc)
            mrows += srows
            msum |= ssum
        write_csv(out / "mechanism_controls.csv", mrows)
        summary["mechanism"] = msum
    if not args.skip_popularity:
        prow = popularity(Path(args.full), Path(args.data), args.models.split(","), inputs)
        write_csv(out / "popularity_controls.csv", prow)
        summary["popularity"] = prow
    write_json(out / "summary.json", summary)
    import statsmodels

    write_json(out / "manifest.json", {"inputs_sha256": inputs, "code": code_version(),
                                       "statsmodels": statsmodels.__version__, "bootstrap": {"B": B, "seed": SEED},
                                       "command": "python -m mb5.controls " + " ".join(
                                           f"--{k} {v}" for k, v in vars(args).items())})


if __name__ == "__main__":
    main()
