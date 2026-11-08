"""Turn run outputs into RBU numbers and a written gate decision.

RBU(spec) = share of RECOGNISED entities that are UNREACHABLE under spec.
  recognised: yes/no score V >= tau_c (tau_c = 95th percentile of V over the
              category's foils, so at most 5% of foils pass) AND the 4-way
              choice picks the entity.
  unreachable (for a template): no surface form (label or alias) is reachable.
  RBU_mean: averaged over templates.  RBU_all: unreachable under EVERY template.
CIs: cluster bootstrap over categories (categories are the unit that varies).

Usage:
  python -m mb5.analysis --results results/G0 --gate G0 \
      --instruct Qwen2.5-7B-Instruct --base Qwen2.5-7B
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import read_jsonl, write_json

HEADLINE_SPECS = ["shipped_hf", "shipped_json", "p0.80_T1", "p0.90_T1", "p0.95_T1", "p0.90_T0.7",
                  "k20_T1", "k50_T1", "minp0.05_T1", "minp0.10_T1", "untruncated_T1"]


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def recognised_set(recog):
    foil_v = defaultdict(list)
    for r in recog:
        if r["kind"] == "foil":
            foil_v[r["category"]].append(r["V"])
    all_foil = [v for vs in foil_v.values() for v in vs]
    glob_tau = float(np.quantile(all_foil, 0.95)) if all_foil else 0.5
    tau = {c: (float(np.quantile(vs, 0.95)) if len(vs) >= 30 else glob_tau) for c, vs in foil_v.items()}
    rec = {}
    for r in recog:
        if r["kind"] == "member":
            ok = r["V"] >= tau.get(r["category"], glob_tau) and r.get("mc_correct", 0) == 1
            rec[r["qid"] + "|" + r["category"]] = ok
    mem_v = [r["V"] for r in recog if r["kind"] == "member"]
    mc = [r["mc_correct"] for r in recog if r["kind"] == "member"]
    validity = {"auroc_yesno": auroc(mem_v, all_foil), "mc_accuracy": float(np.mean(mc)) if mc else float("nan"),
                "n_members": len(mem_v), "n_foils": len(all_foil), "n_recognised": int(sum(rec.values())),
                "tau_per_category": tau}
    return rec, validity


# How a surface form counts as reachable, per verdict mode (see reachability.score_items):
#   robust  unreachable only if cut even with the target raised by the margin (conservative RBU, the headline)
#   exact   the verdict on our own logits, no margin
#   upper   reachable only if it survives with the target lowered by the margin (upper bound on RBU)
MODES = ("robust", "exact", "upper")


def form_reachable(r, s, mode):
    exact = bool(r.get(f"reach__{s}", True))
    if mode == "exact":
        return exact
    if mode == "robust":
        return not bool(r.get(f"robust_unreach__{s}", not exact))
    return bool(r.get(f"robust_reach__{s}", exact))


SENS_EPS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def rbu_sensitivity(reach, rec, spec, templates, eps_list=SENS_EPS):
    """RBU-mean as a function of the margin, from the stored signed margins d*:
    a form is unreachable at eps iff d* < -eps; an entity iff all its forms are.
    Returns None if the run did not record d* for this spec."""
    if not any(f"dstar__{spec}" in r for r in reach):
        return None
    dmax = defaultdict(lambda: -1e9)  # (entity, template) -> max over forms of d*
    for r in reach:
        key = (r["qid"] + "|" + r["category"], r["template"])
        dmax[key] = max(dmax[key], r[f"dstar__{spec}"])
    out = {}
    for eps in eps_list:
        tab = defaultdict(dict)
        for (ek, t), d in dmax.items():
            tab[(ek, t)][spec] = not (d < -eps)
        per = rbu_by_category(tab, rec, spec, templates)
        out[str(eps)] = pooled(per, list(per), 1)
    return out


def entity_table(reach, specs, mode="robust"):
    """(entity, template) -> spec -> reachable via any surface form; plus failure info for the label."""
    tab = defaultdict(lambda: defaultdict(bool))
    lab = {}
    for r in reach:
        key = (r["qid"] + "|" + r["category"], r["template"])
        for s in specs:
            tab[key][s] = tab[key][s] or form_reachable(r, s, mode)
        if r.get("is_label"):
            lab[key] = r
    return tab, lab


def rbu_by_category(tab, rec, spec, templates):
    """category -> (n_recognised, mean-over-templates unreachable count, all-templates unreachable count)."""
    per = defaultdict(lambda: [0, 0.0, 0])
    ents = defaultdict(dict)
    for (ek, t), d in tab.items():
        ents[ek][t] = d[spec]
    for ek, tr in ents.items():
        if not rec.get(ek):
            continue
        cat = ek.split("|", 1)[1]
        ts = [tr[t] for t in templates if t in tr]
        if not ts:
            continue
        per[cat][0] += 1
        per[cat][1] += sum(1 for x in ts if not x) / len(ts)
        per[cat][2] += int(not any(ts))
    return per


def pooled(per, cats, idx):
    n = sum(per[c][0] for c in cats)
    return (sum(per[c][idx] for c in cats) / n) if n else float("nan")


def cluster_ci(per, idx, B=2000, seed=0):
    cats = [c for c in per if per[c][0] > 0]
    if not cats:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    vals = []
    for _ in range(B):
        s = [rng.choice(cats) for _ in cats]
        v = pooled(per, s, idx)
        if not math.isnan(v):
            vals.append(v)
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))


def summarise_model(mdir: Path):
    reach = read_jsonl(mdir / "reach.jsonl.gz")
    recog = read_jsonl(mdir / "recog.jsonl.gz")
    meta = json.loads((mdir / "meta.json").read_text())
    specs = [s for s in HEADLINE_SPECS if s in meta["specs"]]
    templates = sorted({r["template"] for r in reach})
    rec, validity = recognised_set(recog)
    tabs = {m: entity_table(reach, specs, m) for m in MODES}
    tab, lab = tabs["robust"]
    out = {"model": meta["model"], "validity": validity, "margin": meta.get("margin", 0.0), "specs": {}}
    for s in specs:
        per = rbu_by_category(tab, rec, s, templates)
        cats = list(per)
        fp = f"failpos_robust__{s}" if any(f"failpos_robust__{s}" in r for r in lab.values()) else f"failpos__{s}"
        fail0 = [lab[k][fp] == 0 for k in lab if rec.get(k[0]) and lab[k].get(fp, -1) >= 0]
        out["specs"][s] = {
            "rbu_mean": pooled(per, cats, 1), "rbu_mean_ci": cluster_ci(per, 1),
            "rbu_all": pooled(per, cats, 2), "rbu_all_ci": cluster_ci(per, 2),
            "share_fail_at_first_token": float(np.mean(fail0)) if fail0 else float("nan"),
            "per_category": {c: {"n_rec": per[c][0], "rbu_mean": per[c][1] / per[c][0] if per[c][0] else None}
                             for c in cats},
        }
        out["specs"][s]["rbu_mean_by_margin"] = rbu_sensitivity(reach, rec, s, templates)
        for m in ("exact", "upper"):
            pm = rbu_by_category(tabs[m][0], rec, s, templates)
            cm = list(pm)
            out["specs"][s][m] = {"rbu_mean": pooled(pm, cm, 1), "rbu_mean_ci": cluster_ci(pm, 1),
                                  "rbu_all": pooled(pm, cm, 2), "rbu_all_ci": cluster_ci(pm, 2)}
    # p* quantiles over recognised labels, template 0
    ps = [lab[k]["p_star_T1.0"] for k in lab if rec.get(k[0]) and k[1] == 0]
    out["p_star_T1_quantiles_recognised"] = {q: float(np.quantile(ps, q)) for q in (0.5, 0.75, 0.9, 0.95)} if ps else {}
    sc = mdir / "sample_check.json"
    if sc.exists():
        rows = json.loads(sc.read_text())
        keys = ("violations", "boundary_hits", "exact_violations", "noncanonical", "n_samples", "n_valid_text",
                "n_id_prefix_hits", "distinct_id_prefix_hits")
        by = defaultdict(lambda: {k: 0 for k in keys})
        for r in rows:
            for k in keys:
                by[r["spec"]][k] += r.get(k, 0)
        out["sample_check"] = dict(by)
    return out, rec, tab, templates


def paired_test(rec_a, tab_a, rec_b, tab_b, spec, templates):
    """Entities recognised by both models, template-0 reachability: instruct-unreachable & base-reachable (b)
    vs the reverse (c). Sign test via normal approximation."""
    b = c = 0
    for (ek, t), d in tab_a.items():
        if t != templates[0] or not (rec_a.get(ek) and rec_b.get(ek)):
            continue
        da = d.get(spec)
        db = tab_b.get((ek, t), {}).get(spec)
        if da is None or db is None:
            continue
        if (not da) and db:
            b += 1
        elif da and (not db):
            c += 1
    z = (b - c) / math.sqrt(b + c) if b + c else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))
    return {"instruct_only_unreachable": b, "base_only_unreachable": c, "z": z, "p_two_sided": p}


def g0_decision(inst, base, paired, mode="robust"):
    v = inst["validity"]
    sc = inst.get("sample_check", {})
    checks = {
        "V1_no_sampling_violations": all(x["violations"] == 0 for x in sc.values()) if sc else None,
        "V2_recognition_auroc_ge_0.85": v["auroc_yesno"] >= 0.85,
        "V3_at_least_300_recognised": v["n_recognised"] >= 300,
    }
    blocked = [k for k, ok in checks.items() if ok is False]
    S = inst["specs"] if mode == "robust" else {s: d[mode] for s, d in inst["specs"].items()}

    def get(s, key="rbu_mean"):
        return S.get(s, {}).get(key, float("nan"))

    def lo(s):
        return S.get(s, {}).get("rbu_mean_ci", (float("nan"),))[0]

    verdict, why = "KILL", []
    if blocked:
        verdict = "BLOCKED"
        why.append(f"validity checks failed: {blocked}. Fix these before reading any RBU number.")
    else:
        sh = "shipped_hf" if "shipped_hf" in S else None
        strong = (sh and get(sh) >= 0.10 and lo(sh) >= 0.02 and get(sh, "rbu_all") >= 0.5 * get(sh)) or \
                 (get("p0.90_T1") >= 0.05 and lo("p0.90_T1") >= 0.02)
        std = [s for s in ("shipped_hf", "p0.90_T1", "p0.95_T1") if s in S]
        go = any(get(s) >= 0.02 and lo(s) >= 0.01 for s in std)
        other = any(get(s) >= 0.05 for s in ("k20_T1", "k50_T1", "minp0.10_T1") if s in S)
        if strong:
            verdict = "GO (strong)"
        elif go:
            verdict = "GO"
            why.append("Effect is real but modest; the paper needs G1 (post-training mechanism) to carry it.")
        elif other:
            verdict = "REFRAME"
            why.append("Top-p barely matters but top-k/min-p does: reframe around the truncation that bites.")
        else:
            why.append("RBU < 2% under every standard sampler: knowledge collapse is not a decoding artefact here.")
        if base is not None and "p0.90_T1" in S and "p0.90_T1" in base["specs"]:
            bs = base["specs"]["p0.90_T1"] if mode == "robust" else base["specs"]["p0.90_T1"][mode]
            ratio = get("p0.90_T1") / max(bs["rbu_mean"], 1e-9)
            why.append(f"instruct/base RBU ratio at p0.90_T1 = {ratio:.2f} (G1 needs >= 1.5 with paired p < 0.01).")
    return {"verdict": verdict, "checks": checks, "notes": why, "paired_p0.90_T1": paired}


def fmt(x):
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--gate", default="G0")
    ap.add_argument("--instruct", required=True, help="model tag (folder name) of the instruct model")
    ap.add_argument("--base", default=None)
    args = ap.parse_args()
    root = Path(args.results)
    inst, rec_i, tab_i, tpl = summarise_model(root / args.instruct)
    base = paired = None
    if args.base and (root / args.base).exists():
        base, rec_b, tab_b, _ = summarise_model(root / args.base)
        paired = paired_test(rec_i, tab_i, rec_b, tab_b, "p0.90_T1", tpl)
    dec = g0_decision(inst, base, paired)
    dec_exact = g0_decision(inst, base, None, mode="exact")
    write_json(root / "summary.json", {"instruct": inst, "base": base, "decision": dec,
                                       "decision_if_exact_rbu": dec_exact["verdict"]})

    lines = [f"# Gate {args.gate} decision: **{dec['verdict']}**", "",
             f"Instruct model: `{inst['model']}`" + (f"; base model: `{base['model']}`" if base else ""), "",
             f"Verdict computed on robust RBU (margin {inst['margin']} raw logits: an entity counts as unreachable "
             f"only if it stays cut with its logit raised by the margin; docs/04 section 2, notes/decisions.md). "
             f"The same rules on exact RBU give: **{dec_exact['verdict']}**.", "",
             "## Validity checks", ""]
    lines += [f"- {k}: {v}" for k, v in dec["checks"].items()]
    v = inst["validity"]
    lines += [f"- recognition AUROC (yes/no, members vs foils): {v['auroc_yesno']:.3f}",
              f"- 4-way accuracy on members: {v['mc_accuracy']:.3f}",
              f"- recognised entities: {v['n_recognised']} of {v['n_members']}", ""]
    if "sample_check" in inst:
        lines += ["## Sampling sanity check", ""]
        for s, d in inst["sample_check"].items():
            lines.append(f"- {s}: {d['n_samples']} samples, {d['n_valid_text']} valid, "
                         f"{d['n_id_prefix_hits']} id hits on {d['distinct_id_prefix_hits']} distinct forms; "
                         f"violations (robustly unreachable yet sampled) **{d['violations']}**, boundary hits "
                         f"{d['boundary_hits']}, exact-verdict violations {d['exact_violations']}, "
                         f"non-canonical paths {d['noncanonical']}")
        lines.append("")
    lines += ["## Recognised-but-unreachable rate (RBU)", "",
              "RBU mean = robust (headline); exact and upper use no margin and the opposite margin.", "",
              "| sampler | RBU mean | 95% CI | RBU all templates | fail at first token | exact | upper |"
              + (" base RBU mean |" if base else ""),
              "|---|---|---|---|---|---|---|" + ("---|" if base else "")]
    for s, d in inst["specs"].items():
        ci = d["rbu_mean_ci"]
        row = (f"| {s} | {fmt(d['rbu_mean'])} | {fmt(ci[0])} to {fmt(ci[1])} | {fmt(d['rbu_all'])} | "
               f"{fmt(d['share_fail_at_first_token'])} | {fmt(d['exact']['rbu_mean'])} | {fmt(d['upper']['rbu_mean'])} |")
        if base:
            row += f" {fmt(base['specs'].get(s, {}).get('rbu_mean'))} |"
        lines.append(row)
    sens = {s: d["rbu_mean_by_margin"] for s, d in inst["specs"].items() if d.get("rbu_mean_by_margin")}
    if sens:
        lines += ["", "## Sensitivity of RBU to the margin (RBU mean; margin in raw logits)", "",
                  "| sampler | " + " | ".join(f"eps {e}" for e in SENS_EPS) + " |",
                  "|---|" + "---|" * len(SENS_EPS)]
        for s, d in sens.items():
            lines.append(f"| {s} | " + " | ".join(fmt(d[str(e)]) for e in SENS_EPS) + " |")
    lines += ["", "## Notes", ""] + [f"- {n}" for n in dec["notes"]]
    if paired:
        lines.append(f"- paired test at p0.90_T1: {paired}")
    lines += ["", "Decision rules: docs/02_phases_and_gates.md. A human must sign off below before Phase 2.", "",
              "Sign-off: ____________________"]
    gate_dir = Path("results/gates")
    gate_dir.mkdir(parents=True, exist_ok=True)
    (gate_dir / f"{args.gate}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
