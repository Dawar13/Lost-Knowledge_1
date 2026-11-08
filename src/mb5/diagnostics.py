"""Per-category diagnostics for a gate run (no GPU). Reads the outputs of
run_model and writes <results>/diagnostics.{json,md}. Diagnostics inform the
human at a gate; they do not change any gate rule (docs/02).

  python -m mb5.diagnostics --results results/G0 --models Qwen2.5-7B-Instruct,Qwen2.5-7B
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .analysis import auroc, entity_table, rbu_by_category, recognised_set
from .common import read_jsonl, write_json

SPECS = ("shipped_hf", "p0.90_T1", "k50_T1")


def per_category(mdir: Path) -> dict:
    reach = read_jsonl(mdir / "reach.jsonl.gz")
    recog = read_jsonl(mdir / "recog.jsonl.gz")
    rec, validity = recognised_set(recog)
    templates = sorted({r["template"] for r in reach})
    specs = [s for s in SPECS if any(f"reach__{s}" in r for r in reach[:50])]
    tab, _ = entity_table(reach, specs, "robust")
    mem, foil, mc = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in recog:
        (mem if r["kind"] == "member" else foil)[r["category"]].append(r["V"])
        if r["kind"] == "member":
            mc[r["category"]].append(r["mc_correct"])
    out = {}
    for c in sorted(mem):
        d = {"n_members": len(mem[c]), "n_foils": len(foil[c]), "auroc_yesno": auroc(mem[c], foil[c]),
             "mc_accuracy": float(np.mean(mc[c])), "tau": validity["tau_per_category"].get(c),
             "n_recognised": sum(1 for k, v in rec.items() if v and k.split("|", 1)[1] == c)}
        for s in specs:
            per = rbu_by_category(tab, rec, s, templates)
            d[f"rbu_{s}"] = per[c][1] / per[c][0] if per.get(c) and per[c][0] else None
            # members (recognised or not) robustly reachable under template 0: the size of the reachable set
            d[f"reachable_members_t0_{s}"] = sum(1 for (ek, t), v in tab.items()
                                                 if t == 0 and ek.split("|", 1)[1] == c and v[s])
        out[c] = d
    cats = list(out)
    summary = {"auroc_pooled": validity["auroc_yesno"],
               "auroc_mean_over_categories": float(np.mean([out[c]["auroc_yesno"] for c in cats])),
               "auroc_median_over_categories": float(np.median([out[c]["auroc_yesno"] for c in cats])),
               "n_categories_auroc_ge_0.85": sum(out[c]["auroc_yesno"] >= 0.85 for c in cats),
               "n_categories": len(cats)}
    return {"summary": summary, "per_category": out}


def fmt(x, pct=False):
    if x is None:
        return "n/a"
    return f"{100 * x:.1f}%" if pct else (f"{x:.3f}" if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--models", required=True)
    args = ap.parse_args()
    root = Path(args.results)
    res = {m: per_category(root / m) for m in args.models.split(",")}
    write_json(root / "diagnostics.json", res)
    lines = []
    for m, d in res.items():
        s = d["summary"]
        lines += [f"## {m}", "",
                  f"AUROC pooled {s['auroc_pooled']:.3f}; mean over categories {s['auroc_mean_over_categories']:.3f}; "
                  f"median {s['auroc_median_over_categories']:.3f}; categories at or above 0.85: "
                  f"{s['n_categories_auroc_ge_0.85']} of {s['n_categories']}", "",
                  "| category | members | foils | AUROC | 4-way acc | recognised | RBU shipped | RBU p0.90 | RBU k50 "
                  "| reachable members t0 (shipped / p0.90 / k50) |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for c, x in d["per_category"].items():
            lines.append(
                f"| {c} | {x['n_members']} | {x['n_foils']} | {fmt(x['auroc_yesno'])} | {fmt(x['mc_accuracy'])} | "
                f"{x['n_recognised']} | {fmt(x.get('rbu_shipped_hf'), True)} | {fmt(x.get('rbu_p0.90_T1'), True)} | "
                f"{fmt(x.get('rbu_k50_T1'), True)} | {x.get('reachable_members_t0_shipped_hf', 'n/a')} / "
                f"{x.get('reachable_members_t0_p0.90_T1', 'n/a')} / {x.get('reachable_members_t0_k50_T1', 'n/a')} |")
        lines.append("")
    (root / "diagnostics.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
