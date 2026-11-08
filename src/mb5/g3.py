"""Gate G3 report from mb5.frontier outputs.

  python -m mb5.g3 --results results/P5 --models Qwen2.5-7B-Instruct,OLMo-2-1124-7B-Instruct

Rule (notes/decisions.md, fixed before the runs): first-token relaxation recovers at least 50% of the
reference sampler's unreachable recognised entities (robust RBU) and raises the invented-name rate by at
most 2 points. Reading: G3 passes if it holds for every model. Decided by the agent under delegation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import fmt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--models", required=True)
    args = ap.parse_args()
    root = Path(args.results)
    res = {m: json.loads((root / m / "frontier.json").read_text()) for m in args.models.split(",")}
    passes = all(r["g3"]["passes"] for r in res.values())
    L = [f"# Gate G3 decision: **{'PASS' if passes else 'FAIL'}** (decided by the agent under delegation; "
         f"provisional until the hand audit of invented names)", "",
         "Rule: first-token relaxation recovers at least 50% of the reference sampler's unreachable recognised "
         "entities (robust RBU) with the invented-name rate at most 2 points above the reference "
         "(notes/decisions.md). Data: pilot_v2, template 0, 1,000 samples per category per sampler.", ""]
    for m, r in res.items():
        g = r["g3"]
        L += [f"## {m} (reference: {g['reference']})", "",
              f"- recovered: **{fmt(g['recovered'])}** of the reference's unreachable recognised entities; "
              f"invented-name rate change: **{100 * g['invented_delta']:+.1f} points** -> "
              f"**{'passes' if g['passes'] else 'fails'}**", "",
              "| sampler | RBU robust | RBU exact | recovered vs reference | distinct members sampled | "
              "valid rate | foil rate | invented rate | change vs reference |",
              "|---|---|---|---|---|---|---|---|---|"]
        for x in r["rows"]:
            L.append(f"| {x['spec']} | {fmt(x['rbu_robust'])} | {fmt(x['rbu_exact'])} | "
                     f"{fmt(x['recovered_vs_reference'])} | {x['distinct_members_sampled']} | {fmt(x['valid_rate'])} | "
                     f"{fmt(x['foil_rate'])} | {fmt(x['invented_rate'])} | "
                     f"{100 * x['invented_delta_vs_reference']:+.1f} pts |")
        L.append("")
    L += ["Notes: 'invented' is any non-empty answer matching no member of the category in the uncapped "
          "Wikidata list and no foil; it includes real names Wikidata misses and non-name text. The 200-answer "
          "hand audit (docs/02) is pending. Samples: `samples_frontier.jsonl.gz` per model.", "",
          "Sign-off: ____________________"]
    gates = Path("results/gates")
    gates.mkdir(parents=True, exist_ok=True)
    (gates / "G3.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
