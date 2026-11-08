"""Gate G1: where does unreachability come from? Many models, families, stages.

  python -m mb5.analysis_multi --gate G1 --out results/G1 \
      --model qwen2.5:base=results/G0_5090/Qwen2.5-7B \
      --model qwen2.5:instruct=results/G0_5090/Qwen2.5-7B-Instruct \
      --model olmo2:base=results/G1/OLMo-2-1124-7B --model olmo2:sft=results/G1/OLMo-2-1124-7B-SFT \
      --model olmo2:dpo=results/G1/OLMo-2-1124-7B-DPO --model olmo2:instruct=results/G1/OLMo-2-1124-7B-Instruct

Within a family, stages are taken in the order given (first = base, last = final post-trained model).
Rules (docs/02, P2; interpretation fixed in notes/decisions.md before the runs):
  H2 holds if, in at least two families, RBU(last)/RBU(base) at p0.90_T1 is at least 1.5 with paired
     sign test p < 0.01 (template 0, entities recognised by both), AND RBU is non-decreasing across the
     OLMo stages except for one allowed dip.
  H3 holds if, for every family's final model, at least 60% of unreachable recognised entities fail at
     answer position 0 (robust verdict, label form, p0.90_T1).
RBU here is the robust (headline) RBU of docs/04. Writes <out>/g1_summary.json, <out>/stages.csv,
<out>/mechanism.csv and results/gates/<gate>.md.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np

from .analysis import fmt, paired_test, summarise_model
from .common import write_json
from .metrics import write_csv

SPECS = ("p0.90_T1", "k50_T1", "minp0.10_T1", "shipped_hf")


def spearman(x, y):
    if len(x) < 3:
        return float("nan")
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", default="G1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", action="append", required=True, help="family:stage=run_dir/model_tag")
    args = ap.parse_args()

    fams: "OrderedDict[str, OrderedDict[str, Path]]" = OrderedDict()
    for m in args.model:
        key, path = m.split("=", 1)
        fam, stage = key.split(":", 1)
        fams.setdefault(fam, OrderedDict())[stage] = Path(path)

    S, REC, TAB, TPL, FT = {}, {}, {}, {}, {}
    for fam, stages in fams.items():
        for stage, p in stages.items():
            k = (fam, stage)
            S[k], REC[k], TAB[k], TPL[k] = summarise_model(p)
            ft = p / "first_token.json"
            FT[k] = json.loads(ft.read_text())["rows"] if ft.exists() else None

    stage_rows, paired_rows, mech_rows = [], [], []
    for (fam, stage), s in S.items():
        row = {"family": fam, "stage": stage, "model": s["model"], "auroc_yesno": s["validity"]["auroc_yesno"],
               "n_recognised": s["validity"]["n_recognised"]}
        for sp in SPECS:
            d = s["specs"].get(sp)
            if d:
                row[f"rbu_{sp}"] = d["rbu_mean"]
                row[f"rbu_{sp}_ci"] = d["rbu_mean_ci"]
                row[f"fail_first_token_{sp}"] = d["share_fail_at_first_token"]
        if FT[(fam, stage)]:
            r = FT[(fam, stage)]
            row["first_token_top1_mass_mean"] = float(np.mean([x["top1_mass"] for x in r]))
            row["first_token_entropy_mean"] = float(np.mean([x["entropy_nats"] for x in r]))
            row["first_token_kept_top_p_0.9_median"] = float(np.median([x["kept_top_p_0.9"] for x in r]))
            per_cat = defaultdict(list)
            for x in r:
                per_cat[x["category"]].append(x["top1_mass"])
            pc = s["specs"].get("p0.90_T1", {}).get("per_category", {})
            for c, v in per_cat.items():
                if c in pc and pc[c]["rbu_mean"] is not None:
                    mech_rows.append({"family": fam, "stage": stage, "category": c, "top1_mass": float(np.mean(v)),
                                      "rbu_p0.90_T1": pc[c]["rbu_mean"], "n_rec": pc[c]["n_rec"]})
        stage_rows.append(row)

    # paired tests: adjacent stages and base vs final, per family
    h2_fam = {}
    for fam, stages in fams.items():
        names = list(stages)
        pairs = list(zip(names, names[1:]))
        if len(names) > 2:
            pairs.append((names[0], names[-1]))
        for a, b in pairs:
            ka, kb = (fam, a), (fam, b)
            for sp in ("p0.90_T1", "k50_T1"):
                if sp not in S[ka]["specs"] or sp not in S[kb]["specs"]:
                    continue
                pt = paired_test(REC[kb], TAB[kb], REC[ka], TAB[ka], sp, TPL[kb])  # later stage as "instruct"
                ra, rb = S[ka]["specs"][sp]["rbu_mean"], S[kb]["specs"][sp]["rbu_mean"]
                paired_rows.append({"family": fam, "from": a, "to": b, "spec": sp, "rbu_from": ra, "rbu_to": rb,
                                    "ratio": rb / ra if ra else float("nan"),
                                    "to_only_unreachable": pt["instruct_only_unreachable"],
                                    "from_only_unreachable": pt["base_only_unreachable"], "z": pt["z"],
                                    "p_two_sided": pt["p_two_sided"]})
        first, last = names[0], names[-1]
        r = next((x for x in paired_rows if x["family"] == fam and x["from"] == first and x["to"] == last
                  and x["spec"] == "p0.90_T1"), None)
        h2_fam[fam] = bool(r and r["ratio"] >= 1.5 and r["p_two_sided"] < 0.01 and r["rbu_to"] > r["rbu_from"])

    olmo = [x for x in stage_rows if x["family"].startswith("olmo")]
    dips = sum(1 for a, b in zip(olmo, olmo[1:]) if b.get("rbu_p0.90_T1", 0) < a.get("rbu_p0.90_T1", 0))
    olmo_ok = (len(olmo) >= 2 and dips <= 1) if olmo else None
    h2 = sum(h2_fam.values()) >= 2 and (olmo_ok is not False)
    h3_fam = {}
    for fam, stages in fams.items():
        last = list(stages)[-1]
        share = S[(fam, last)]["specs"].get("p0.90_T1", {}).get("share_fail_at_first_token", float("nan"))
        h3_fam[fam] = {"model": S[(fam, last)]["model"], "share": share, "holds": bool(share >= 0.60)}
    h3 = all(v["holds"] for v in h3_fam.values())
    mech = {"spearman_top1_mass_vs_rbu": spearman([m["top1_mass"] for m in mech_rows],
                                                  [m["rbu_p0.90_T1"] for m in mech_rows]),
            "n_points": len(mech_rows)}

    if h2 and h3:
        story = "Both hold: alignment concentrates first-token mass; truncation deletes the tail. P5 fix targets the first token."
    elif not h2 and h3:
        story = "H2 fails: lead with 'decoding defaults alone delete recognised knowledge'; drop the alignment claim."
    elif h2 and not h3:
        story = "H3 fails: the P5 fix switches from first-token relaxation to min-p or an entity-aware rule."
    else:
        story = "Neither holds: report decoding defaults alone; P5 fix uses min-p or an entity-aware rule."
    out = Path(args.out)
    write_csv(out / "stages.csv", stage_rows)
    write_csv(out / "paired.csv", paired_rows)
    write_csv(out / "mechanism.csv", mech_rows)
    summary = {"H2": h2, "H2_per_family": h2_fam, "olmo_stage_dips": dips, "olmo_monotone_ok": olmo_ok,
               "H3": h3, "H3_per_family": h3_fam, "mechanism": mech, "story": story,
               "models": {f"{f}:{s}": str(p) for f, st in fams.items() for s, p in st.items()}}
    write_json(out / "g1_summary.json", summary)

    L = [f"# Gate {args.gate}: H2 {'HOLDS' if h2 else 'FAILS'}, H3 {'HOLDS' if h3 else 'FAILS'}", "",
         f"**Story:** {story}", "",
         "RBU = robust RBU (docs/04 s.2), mean over templates, cluster-bootstrap 95% CI over categories. "
         "Recognition uses the a-priori probe (option B, notes/decisions.md).", "",
         "## RBU by family and stage", "",
         "| family | stage | AUROC | recognised | RBU p0.90_T1 (CI) | RBU k50_T1 | RBU min-p 0.10 | RBU shipped | "
         "fail at first token (p0.90) | first-token top-1 mass |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in stage_rows:
        ci = r.get("rbu_p0.90_T1_ci", (float("nan"), float("nan")))
        shipped = fmt(r["rbu_shipped_hf"]) if "rbu_shipped_hf" in r else "none shipped"
        ft = f"{r['first_token_top1_mass_mean']:.3f}" if "first_token_top1_mass_mean" in r else "n/a"
        L.append(f"| {r['family']} | {r['stage']} | {r['auroc_yesno']:.3f} | {r['n_recognised']} | "
                 f"{fmt(r.get('rbu_p0.90_T1'))} ({fmt(ci[0])} to {fmt(ci[1])}) | {fmt(r.get('rbu_k50_T1'))} | "
                 f"{fmt(r.get('rbu_minp0.10_T1'))} | {shipped} | {fmt(r.get('fail_first_token_p0.90_T1'))} | {ft} |")
    L += ["", "## Paired tests (template 0, entities recognised by both stages)", "",
          "| family | from | to | sampler | RBU from | RBU to | ratio | only-to unreachable | only-from unreachable | p |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in paired_rows:
        L.append(f"| {r['family']} | {r['from']} | {r['to']} | {r['spec']} | {fmt(r['rbu_from'])} | {fmt(r['rbu_to'])} | "
                 f"{r['ratio']:.2f} | {r['to_only_unreachable']} | {r['from_only_unreachable']} | {r['p_two_sided']:.2g} |")
    L += ["", "## Rules", "",
          f"- H2 per family (ratio at least 1.5, p < 0.01, base to final): {h2_fam}; OLMo dips across stages: {dips} "
          f"(at most 1 allowed) -> **H2 {'holds' if h2 else 'fails'}**",
          f"- H3 per family (share failing at position 0 at least 60%, final model): "
          f"{ {k: round(v['share'], 3) for k, v in h3_fam.items()} } -> **H3 {'holds' if h3 else 'fails'}**",
          f"- Mechanism (exploratory): Spearman correlation of first-token top-1 mass with RBU across "
          f"(model, category) points: {mech['spearman_top1_mass_vs_rbu']:.3f} (n = {mech['n_points']})", "",
          "Decision rules: docs/02_phases_and_gates.md (P2). A human must sign off below before Phase 3.", "",
          "Sign-off: ____________________"]
    gates = Path("results/gates")
    gates.mkdir(parents=True, exist_ok=True)
    (gates / f"{args.gate}.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
