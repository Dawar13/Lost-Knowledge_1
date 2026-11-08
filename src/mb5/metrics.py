"""Compute and store every metric the paper may need from one run directory.

No GPU. Reads the outputs of run_model (and selfcheck, if present) and writes
<results>/metrics/ with one CSV per table plus a manifest of inputs and code.
Every number used in docs, notes or the paper should be read from these files
(CLAUDE.md rule 5).

  python -m mb5.metrics --results results/G0_5090 --data data/tailcat/pilot_v2 \
      --instruct Qwen2.5-7B-Instruct --base Qwen2.5-7B

Tables (per model unless noted; "mode" is robust / exact / upper, docs/04 s.2):
  entities_<m>.csv         one row per (entity, template): recognition, sitelinks, label length,
                           critical values over surface forms, and per spec reachability
  validity_<m>.csv         per category: members, foils, AUROC, 4-way accuracy, tau, recognised, foil pass rate
  rbu_<m>.csv              per spec x mode: RBU mean / all templates with cluster-bootstrap CIs,
                           RBU per template and its coefficient of variation, label-only RBU
  rbu_by_category_<m>.csv  per spec x category: recognised, RBU mean (robust, exact)
  rbu_curves_<m>.csv       exact single-truncation curves: RBU against top-p (T=1, 0.7), top-k, min-p
  failpos_<m>.csv          per spec: histogram of the first failing answer position (robust verdict)
  popularity_<m>.csv       per sitelink quintile: recognition rate and RBU per spec
  length_<m>.csv           per label token length: recognition rate and RBU per spec
  reachable_set_<m>.csv    per spec x category x template: members reachable (all, recognised)
  sample_check_<m>.csv     per spec x category: samples, valid, distinct valid, violations, boundary hits,
                           and coverage of the reachable set
  calibration_<m>.json     generate-vs-scorer logit gap, margins needed, epsilon rule
  paired.csv               (instruct vs base) per spec: discordant pairs, sign test, RBU ratio
  headline.json            the numbers a reader looks for first, with their source table
  manifest.json            sha256 of every input, code version, epsilon, time
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .analysis import (MODES, auroc, cluster_ci, form_reachable, pooled, rbu_by_category, recognised_set)
from .common import code_version, read_jsonl, write_json

P_GRID = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]
K_GRID = [1, 2, 5, 10, 20, 40, 50, 100, 200, 500, 1000]
MINP_GRID = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]


# ---------------------------------------------------------------- helpers
def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in keys})


def _fmt(x):
    if isinstance(x, float):
        return "" if math.isnan(x) else round(x, 6)
    if isinstance(x, (list, tuple)):
        return json.dumps(x)
    return x


def sign_test(b: int, c: int) -> tuple[float, float]:
    z = (b - c) / math.sqrt(b + c) if b + c else 0.0
    return z, math.erfc(abs(z) / math.sqrt(2))


def quintile_edges(values):
    v = np.asarray(sorted(values))
    return [float(np.quantile(v, q)) for q in (0.2, 0.4, 0.6, 0.8)]


def bin_of(x, edges):
    return sum(x > e for e in edges)


# ---------------------------------------------------------------- per model
class Model:
    def __init__(self, mdir: Path):
        self.dir = mdir
        self.reach = read_jsonl(mdir / "reach.jsonl.gz")
        self.recog = read_jsonl(mdir / "recog.jsonl.gz")
        self.meta = json.loads((mdir / "meta.json").read_text())
        self.specs = [s for s in self.meta["specs"]]
        self.templates = sorted({r["template"] for r in self.reach})
        self.rec, self.validity = recognised_set(self.recog)
        # (entity key, template) -> list of form records
        self.forms = defaultdict(list)
        for r in self.reach:
            self.forms[(r["qid"] + "|" + r["category"], r["template"])].append(r)
        self.members = {r["qid"] + "|" + r["category"]: r for r in self.recog if r["kind"] == "member"}

    def entity_rows(self):
        rows = []
        for (ek, t), fs in sorted(self.forms.items()):
            lab = next((f for f in fs if f.get("is_label")), fs[0])
            m = self.members.get(ek, {})
            row = {"entity": ek, "qid": ek.split("|")[0], "category": ek.split("|")[1], "template": t,
                   "label": m.get("label", lab.get("form")), "sitelinks": lab.get("sitelinks"),
                   "n_forms": len(fs), "label_tokens": lab.get("L"), "boundary_ok": lab.get("boundary_ok"),
                   "V": m.get("V"), "mc_correct": m.get("mc_correct"), "mc_p": m.get("mc_p"),
                   "recognised": bool(self.rec.get(ek)),
                   # entity reachable iff ANY form is: take the most favourable form's critical value
                   "p_star_T1": min(f["p_star_T1.0"] for f in fs),
                   "p_star_T0.7": min(f["p_star_T0.7"] for f in fs),
                   "k_star": min(f["k_star"] for f in fs),
                   "minp_star_T1": max(f["minp_star_T1.0"] for f in fs),
                   "label_p_star_pos_T1": lab.get("p_star_pos_T1.0"), "label_k_star_pos": lab.get("k_star_pos")}
            for s in self.specs:
                for mode in MODES:
                    row[f"reach_{mode}__{s}"] = any(form_reachable(f, s, mode) for f in fs)
                row[f"label_reach_robust__{s}"] = form_reachable(lab, s, "robust")
                row[f"label_failpos_robust__{s}"] = lab.get(f"failpos_robust__{s}", lab.get(f"failpos__{s}"))
                if any(f"dstar__{s}" in f for f in fs):
                    row[f"dstar__{s}"] = max(f.get(f"dstar__{s}", -99.0) for f in fs)
            rows.append(row)
        return rows


def rbu_from_rows(rows, spec, mode, templates, key=None):
    """per category [n_rec, mean-over-templates unreachable, all-templates unreachable]"""
    key = key or f"reach_{mode}__{spec}"
    ents = defaultdict(dict)
    for r in rows:
        if r["recognised"]:
            ents[r["entity"]][r["template"]] = r[key]
    per = defaultdict(lambda: [0, 0.0, 0])
    for ek, tr in ents.items():
        ts = [tr[t] for t in templates if t in tr]
        if not ts:
            continue
        c = ek.split("|", 1)[1]
        per[c][0] += 1
        per[c][1] += sum(not x for x in ts) / len(ts)
        per[c][2] += int(not any(ts))
    return per


def model_tables(M: Model, rows, out: Path, tag: str, data: Path | None):
    T = M.templates
    # validity per category
    mem, foil, mc = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in M.recog:
        (mem if r["kind"] == "member" else foil)[r["category"]].append(r["V"])
        if r["kind"] == "member":
            mc[r["category"]].append(r["mc_correct"])
    tau = M.validity["tau_per_category"]
    val = []
    for c in sorted(mem):
        val.append({"category": c, "n_members": len(mem[c]), "n_foils": len(foil[c]),
                    "auroc_yesno": auroc(mem[c], foil[c]), "mc_accuracy": float(np.mean(mc[c])),
                    "tau": tau.get(c), "members_V_ge_tau": float(np.mean([v >= tau.get(c, 0.5) for v in mem[c]])),
                    "foils_V_ge_tau": float(np.mean([v >= tau.get(c, 0.5) for v in foil[c]])) if foil[c] else None,
                    "n_recognised": sum(1 for k, v in M.rec.items() if v and k.split("|", 1)[1] == c)})
    val.append({"category": "ALL (pooled)", "n_members": sum(len(v) for v in mem.values()),
                "n_foils": sum(len(v) for v in foil.values()), "auroc_yesno": M.validity["auroc_yesno"],
                "mc_accuracy": M.validity["mc_accuracy"], "n_recognised": M.validity["n_recognised"]})
    write_csv(out / f"validity_{tag}.csv", val)

    # RBU per spec x mode, with per-template RBU and CV, and label-only
    rbu, bycat = [], []
    for s in M.specs:
        for mode in MODES:
            per = rbu_from_rows(rows, s, mode, T)
            cats = list(per)
            pt = [pooled(rbu_from_rows(rows, s, mode, [t]), cats, 1) for t in T]
            mean_pt = float(np.mean(pt)) if pt else float("nan")
            row = {"spec": s, "mode": mode, "n_recognised": sum(per[c][0] for c in cats),
                   "rbu_mean": pooled(per, cats, 1), "rbu_mean_ci": cluster_ci(per, 1),
                   "rbu_all_templates": pooled(per, cats, 2), "rbu_all_ci": cluster_ci(per, 2),
                   "rbu_cv_over_templates": float(np.std(pt) / mean_pt) if mean_pt else float("nan")}
            for t, v in zip(T, pt):
                row[f"rbu_template{t}"] = v
            if mode == "robust":
                pl = rbu_from_rows(rows, s, mode, T, key=f"label_reach_robust__{s}")
                row["rbu_mean_label_only"] = pooled(pl, list(pl), 1)
            rbu.append(row)
        per_r = rbu_from_rows(rows, s, "robust", T)
        per_e = rbu_from_rows(rows, s, "exact", T)
        for c in sorted(per_r):
            bycat.append({"spec": s, "category": c, "n_recognised": per_r[c][0],
                          "rbu_mean_robust": per_r[c][1] / per_r[c][0] if per_r[c][0] else None,
                          "rbu_all_robust": per_r[c][2] / per_r[c][0] if per_r[c][0] else None,
                          "rbu_mean_exact": per_e[c][1] / per_e[c][0] if per_e[c][0] else None})
    write_csv(out / f"rbu_{tag}.csv", rbu)
    write_csv(out / f"rbu_by_category_{tag}.csv", bycat)

    # exact single-truncation curves over recognised entities (mean over templates)
    recog_rows = [r for r in rows if r["recognised"]]

    def curve(name, grid, unreachable):
        out_rows = []
        for g in grid:
            per = defaultdict(lambda: [0, 0.0, 0])
            ents = defaultdict(list)
            for r in recog_rows:
                ents[r["entity"]].append(unreachable(r, g))
            for ek, us in ents.items():
                c = ek.split("|", 1)[1]
                per[c][0] += 1
                per[c][1] += sum(us) / len(us)
                per[c][2] += int(all(us))
            out_rows.append({"family": name, "value": g, "rbu_mean": pooled(per, list(per), 1),
                             "rbu_mean_ci": cluster_ci(per, 1, B=500), "rbu_all_templates": pooled(per, list(per), 2)})
        return out_rows

    curves = (curve("top_p_T1", P_GRID, lambda r, p: r["p_star_T1"] >= p)
              + curve("top_p_T0.7", P_GRID, lambda r, p: r["p_star_T0.7"] >= p)
              + curve("top_k", K_GRID, lambda r, k: r["k_star"] > k)
              + curve("min_p_T1", MINP_GRID, lambda r, m: r["minp_star_T1"] < m))
    write_csv(out / f"rbu_curves_{tag}.csv", curves)

    # first failing position (label, robust verdict), recognised entities
    fp = []
    for s in M.specs:
        h = Counter(r[f"label_failpos_robust__{s}"] for r in recog_rows
                    if r.get(f"label_failpos_robust__{s}") not in (None, -1))
        n = sum(h.values())
        for pos in sorted(h):
            fp.append({"spec": s, "fail_position": pos, "count": h[pos], "share": h[pos] / n if n else None})
    write_csv(out / f"failpos_{tag}.csv", fp)

    # popularity quintiles (over all members) and label length
    edges = quintile_edges([r["sitelinks"] for r in rows if r["template"] == T[0]])

    def strata(key_fn, name):
        out_rows = []
        groups = defaultdict(list)
        for r in rows:
            groups[key_fn(r)].append(r)
        for g in sorted(groups):
            rs = groups[g]
            ents = {r["entity"] for r in rs}
            rec_ents = {r["entity"] for r in rs if r["recognised"]}
            row = {name: g, "n_entities": len(ents), "recognition_rate": len(rec_ents) / len(ents) if ents else None,
                   "sitelinks_median": float(np.median([r["sitelinks"] for r in rs]))}
            for s in M.specs:
                rr = [r for r in rs if r["recognised"]]
                row[f"rbu_robust__{s}"] = (sum(not r[f"reach_robust__{s}"] for r in rr) / len(rr)) if rr else None
            out_rows.append(row)
        return out_rows

    write_csv(out / f"popularity_{tag}.csv",
              [dict(r, quintile_edges=edges) for r in strata(lambda r: bin_of(r["sitelinks"], edges), "sitelink_quintile")])
    write_csv(out / f"length_{tag}.csv", strata(lambda r: min(r["label_tokens"] or 0, 8), "label_tokens_capped8"))

    # reachable set sizes
    rs_rows = []
    for s in M.specs:
        for c in sorted({r["category"] for r in rows}):
            for t in T:
                cr = [r for r in rows if r["category"] == c and r["template"] == t]
                rs_rows.append({"spec": s, "category": c, "template": t, "members": len(cr),
                                "reachable_robust": sum(bool(r[f"reach_robust__{s}"]) for r in cr),
                                "reachable_exact": sum(bool(r[f"reach_exact__{s}"]) for r in cr),
                                "recognised": sum(r["recognised"] for r in cr),
                                "recognised_reachable_robust": sum(r["recognised"] and r[f"reach_robust__{s}"]
                                                                   for r in cr)})
    write_csv(out / f"reachable_set_{tag}.csv", rs_rows)

    # sample check with coverage of the reachable set (template 0)
    sc_path = M.dir / "sample_check.json"
    if sc_path.exists():
        sc_rows = []
        for r in json.loads(sc_path.read_text()):
            s, c = r["spec"], r["category"]
            reach0 = {x["qid"] for x in rows if x["category"] == c and x["template"] == T[0]
                      and x.get(f"reach_robust__{s}")}
            sampled = set(r.get("distinct_valid_qids", []))
            sc_rows.append({"spec": s, "category": c, "n_samples": r["n_samples"], "n_valid_text": r["n_valid_text"],
                            "distinct_valid": r["distinct_valid"], "n_id_prefix_hits": r["n_id_prefix_hits"],
                            "distinct_id_prefix_hits": r.get("distinct_id_prefix_hits"),
                            "violations": r["violations"], "boundary_hits": r.get("boundary_hits"),
                            "exact_violations": r.get("exact_violations"), "noncanonical": r["noncanonical"],
                            "reachable_set_t0": len(reach0),
                            "sampled_within_reachable_t0": len(sampled & reach0),
                            "coverage_of_reachable_t0": len(sampled & reach0) / len(reach0) if reach0 else None})
        write_csv(out / f"sample_check_{tag}.csv", sc_rows)
    return val, rbu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--instruct", required=True)
    ap.add_argument("--base", default=None)
    args = ap.parse_args()
    root = Path(args.results)
    out = root / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    models = {"instruct": args.instruct} | ({"base": args.base} if args.base else {})
    loaded, ent_rows, headline = {}, {}, {}
    for role, tag in models.items():
        M = Model(root / tag)
        rows = M.entity_rows()
        loaded[role], ent_rows[role] = M, rows
        write_csv(out / f"entities_{tag}.csv", rows)
        val, rbu = model_tables(M, rows, out, tag, Path(args.data) if args.data else None)
        cal = root / "calibration" / tag / "selfcheck.json"
        if cal.exists():
            d = json.loads(cal.read_text())
            write_json(out / f"calibration_{tag}.json",
                       {k: d.get(k) for k in ("total", "required_margin", "epsilon_rule", "epsilon_grid", "logit_gap")}
                       | {"seconds": d["meta"].get("seconds"), "gpu": d["meta"].get("gpu")})
        hl = {r["spec"]: {"rbu_mean": r["rbu_mean"], "ci": r["rbu_mean_ci"], "rbu_all": r["rbu_all_templates"]}
              for r in rbu if r["mode"] == "robust"}
        headline[role] = {"model": M.meta["model"], "model_commit": M.meta.get("commit"), "gpu": M.meta.get("gpu"),
                          "margin": M.meta.get("margin"), "recognition": {k: v for k, v in M.validity.items()
                                                                           if k != "tau_per_category"},
                          "rbu_robust": hl, "source": f"rbu_{tag}.csv"}

    if "base" in loaded:
        I, B = loaded["instruct"], loaded["base"]
        ri = {(r["entity"], r["template"]): r for r in ent_rows["instruct"]}
        rb = {(r["entity"], r["template"]): r for r in ent_rows["base"]}
        paired = []
        for s in [s for s in I.specs if s in B.specs]:
            for t in ["all"] + I.templates:
                b = c = both = 0
                for key, x in ri.items():
                    if t != "all" and key[1] != t:
                        continue
                    y = rb.get(key)
                    if y is None or not (x["recognised"] and y["recognised"]):
                        continue
                    both += 1
                    ui, ub = not x[f"reach_robust__{s}"], not y[f"reach_robust__{s}"]
                    b += ui and not ub
                    c += ub and not ui
                z, p = sign_test(b, c)
                ri_rbu = headline["instruct"]["rbu_robust"].get(s, {}).get("rbu_mean")
                rb_rbu = headline["base"]["rbu_robust"].get(s, {}).get("rbu_mean")
                paired.append({"spec": s, "template": t, "pairs_recognised_by_both": both,
                               "instruct_only_unreachable": b, "base_only_unreachable": c, "z": z, "p_two_sided": p,
                               "rbu_mean_instruct": ri_rbu, "rbu_mean_base": rb_rbu,
                               "rbu_ratio_instruct_over_base": (ri_rbu / rb_rbu) if ri_rbu is not None and rb_rbu
                               else None})
        write_csv(out / "paired.csv", paired)
        headline["paired_p0.90_T1_all_templates"] = next((p for p in paired if p["spec"] == "p0.90_T1"
                                                          and p["template"] == "all"), None)

    write_json(out / "headline.json", headline)
    inputs = {}
    for role, tag in models.items():
        for f in ("reach.jsonl.gz", "recog.jsonl.gz", "meta.json", "sample_check.json"):
            inputs[f"{tag}/{f}"] = sha256(root / tag / f)
        inputs[f"calibration/{tag}/selfcheck.json"] = sha256(root / "calibration" / tag / "selfcheck.json")
    if args.data:
        for f in ("categories.json", "entities.jsonl", "foils.jsonl", "valid_names.json", "build_report.json"):
            inputs[f"data:{f}"] = sha256(Path(args.data) / f)
    write_json(out / "manifest.json", {"inputs_sha256": inputs, "code": code_version(),
                                       "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                       "args": vars(args), "tables": sorted(p.name for p in out.iterdir())})
    print(f"[metrics] wrote {len(list(out.iterdir()))} files to {out}")


if __name__ == "__main__":
    main()
