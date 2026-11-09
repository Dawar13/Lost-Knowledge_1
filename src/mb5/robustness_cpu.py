"""Robustness of the headline RBU to the recognition test (CPU only, from the metric tables).

Answers internal review R2.3 / R2.10 (notes/drafts/review_round1.md): the paper says the main results
hold on the categories whose yes/no probe passes V2 on its own (per-category AUROC >= 0.85) and under a
second probe wording, and docs/04 names RBU-all (unreachable under every wording) the conservative
headline. This module computes all three from the stored per-(entity, template) tables:

  <run>/metrics/entities_<model>.csv   recognised, reach_robust__<spec> per (entity, template)
  <run>/metrics/validity_<model>.csv   per-category AUROC of the yes/no probe

The category subset is fixed by the INSTRUCT model's per-category AUROC (V2 was judged on the instruct
model) and applied to both models of a pair, so paired comparisons use the same categories. RBU-mean and
RBU-all are pooled over categories exactly as mb5.analysis (rbu_by_category / pooled / cluster_ci); the
all-category values are checked against the stored rbu_<model>.csv (field `check`).

  python -m mb5.robustness_cpu --out results/analysis_cpu/robustness
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from .analysis import cluster_ci, pooled
from .common import code_version, write_json
from .metrics import sha256

R = Path("results")
V2_BAR = 0.85

# run name -> (metrics dir, instruct tag, base tag or None, specs)
RUNS = {
    "pilot_qwen": (R / "G0_5090/metrics", "Qwen2.5-7B-Instruct", "Qwen2.5-7B", ("p0.90_T1", "shipped_hf")),
    "pilot_qwen_probe2": (R / "G0_a3_5090/metrics", "Qwen2.5-7B-Instruct", "Qwen2.5-7B", ("p0.90_T1", "shipped_hf")),
    "pilot_olmo": (R / "G1/metrics/olmo2_instruct_vs_base", "OLMo-2-1124-7B-Instruct", "OLMo-2-1124-7B", ("p0.90_T1",)),
    "full_qwen": (R / "P3/metrics/qwen2.5", "Qwen2.5-7B-Instruct", "Qwen2.5-7B", ("p0.90_T1", "shipped_hf")),
    "full_olmo": (R / "P3/metrics/olmo2", "OLMo-2-1124-7B-Instruct", "OLMo-2-1124-7B", ("p0.90_T1",)),
}


def _csv(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _true(x) -> bool:
    return str(x).strip().lower() in ("true", "1", "1.0")


def per_category(rows: list[dict], spec: str) -> dict:
    """category -> [n_recognised, mean-over-templates unreachable count, all-templates unreachable count]
    (the layout of analysis.rbu_by_category)."""
    col = f"reach_robust__{spec}"
    if rows and col not in rows[0]:
        return {}
    ents = defaultdict(dict)
    rec = {}
    for r in rows:
        k = r["qid"] + "|" + r["category"]
        ents[k][int(r["template"])] = _true(r[col])
        rec[k] = _true(r["recognised"])
    per = defaultdict(lambda: [0, 0.0, 0])
    for k, tr in ents.items():
        if not rec.get(k):
            continue
        cat = k.split("|", 1)[1]
        ts = list(tr.values())
        per[cat][0] += 1
        per[cat][1] += sum(1 for x in ts if not x) / len(ts)
        per[cat][2] += int(not any(ts))
    return per


def summarise(per: dict, cats: list[str]) -> dict:
    sub = {c: per[c] for c in cats if c in per and per[c][0] > 0}
    if not sub:
        return {"n_categories": 0, "n_recognised": 0}
    return {"n_categories": len(sub), "n_recognised": int(sum(v[0] for v in sub.values())),
            "rbu_mean": pooled(sub, list(sub), 1), "rbu_mean_ci": list(cluster_ci(sub, 1)),
            "rbu_all": pooled(sub, list(sub), 2), "rbu_all_ci": list(cluster_ci(sub, 2))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/analysis_cpu/robustness")
    args = ap.parse_args()
    out = Path(args.out)
    res, inputs, flat = {}, {}, []
    for run, (mdir, inst, base, specs) in RUNS.items():
        vfile = mdir / f"validity_{inst}.csv"
        # per-category rows only (the table ends with an "ALL (pooled)" row)
        val = {r["category"]: float(r["auroc_yesno"]) for r in _csv(vfile) if not r["category"].startswith("ALL")}
        passing = sorted(c for c, a in val.items() if a >= V2_BAR)
        inputs[str(vfile)] = sha256(vfile)
        res[run] = {"v2_bar": V2_BAR, "categories": len(val), "passing_categories": passing, "models": {}}
        for tag in (inst, base):
            efile = mdir / f"entities_{tag}.csv"
            if not efile.exists():
                continue
            inputs[str(efile)] = sha256(efile)
            rows = _csv(efile)
            stored = {}
            rfile = mdir / f"rbu_{tag}.csv"
            if rfile.exists():
                inputs[str(rfile)] = sha256(rfile)
                stored = {r["spec"]: r for r in _csv(rfile) if r["mode"] == "robust"}
            mres = {}
            for spec in specs:
                per = per_category(rows, spec)
                if not per:
                    continue
                allc = summarise(per, list(per))
                pas = summarise(per, passing)
                chk = None
                if spec in stored:
                    chk = {"rbu_mean_matches": abs(allc["rbu_mean"] - float(stored[spec]["rbu_mean"])) < 5e-6,
                           "rbu_all_matches": abs(allc["rbu_all"] - float(stored[spec]["rbu_all_templates"])) < 5e-6}
                mres[spec] = {"all_categories": allc, "v2_passing_categories": pas, "check": chk}
                for subset, d in (("all", allc), ("v2_passing", pas)):
                    flat.append({"run": run, "model": tag, "spec": spec, "subset": subset,
                                 "n_categories": d["n_categories"], "n_recognised": d["n_recognised"],
                                 "rbu_mean": d.get("rbu_mean"), "rbu_mean_ci": json.dumps(d.get("rbu_mean_ci")),
                                 "rbu_all": d.get("rbu_all"), "rbu_all_ci": json.dumps(d.get("rbu_all_ci")),
                                 "matches_stored": None if chk is None or subset != "all"
                                 else chk["rbu_mean_matches"] and chk["rbu_all_matches"]})
            res[run]["models"][tag] = mres
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "robustness.json", res)
    with open(out / "robustness.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(flat)
    write_json(out / "manifest.json", {"inputs": inputs, "code": code_version(), "module": "mb5.robustness_cpu"})
    bad = [r for r in flat if r["matches_stored"] is False]
    print(f"[robustness_cpu] {len(flat)} rows; {len(bad)} mismatches with stored rbu tables")
    for r in flat:
        print(f"  {r['run']:18s} {r['model']:26s} {r['spec']:11s} {r['subset']:10s} cats {r['n_categories']:3d} "
              f"rec {r['n_recognised']:5d} mean {r['rbu_mean']:.3f} all {r['rbu_all']:.3f}")


if __name__ == "__main__":
    main()
