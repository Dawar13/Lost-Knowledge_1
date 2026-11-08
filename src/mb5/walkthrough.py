"""Per-piece walk-through of real entities: where along the name the sampler cuts.

Picks entities recognised by both models of a pair where the post-trained model
is cut at the first answer token and the base model only later (or any other
pattern via --pattern), and prints every token with the mass strictly ahead of
it (G, T=1) and its rank, for template 0. Explanation and appendix figure.
Writes <results>/examples/walkthrough_<spec>.{txt,json}. No GPU.

  python -m mb5.walkthrough --results results/G0_5090 --instruct Qwen2.5-7B-Instruct --base Qwen2.5-7B
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import recognised_set
from .common import read_jsonl, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--instruct", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--spec", default="p0.90_T1")
    ap.add_argument("--top-p", type=float, default=0.9, help="threshold used to print KEPT/CUT from G")
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    root = Path(args.results)
    meta = json.loads((root / args.instruct / "meta.json").read_text())
    tok = AutoTokenizer.from_pretrained(meta["model"])
    rec = {}
    lab = {}
    for role, tag in (("instruct", args.instruct), ("base", args.base)):
        rec[role], _ = recognised_set(read_jsonl(root / tag / "recog.jsonl.gz"))
        lab[role] = {(r["qid"], r["category"]): r for r in read_jsonl(root / tag / "reach.jsonl.gz")
                     if r["template"] == 0 and r["is_label"]}
    f = f"failpos__{args.spec}"
    cands = []
    for k, a in lab["instruct"].items():
        b = lab["base"].get(k)
        ek = k[0] + "|" + k[1]
        if b and rec["instruct"].get(ek) and rec["base"].get(ek) and a[f] == 0 and b[f] >= 1:
            cands.append((a["sitelinks"], k, a, b))
    cands.sort(key=lambda x: (-x[0], x[1]))
    lines = [f"{len(cands)} entities recognised by both models, cut at the first token by {args.instruct} and later "
             f"by {args.base} under {args.spec} (template 0). Most popular {args.n}:", ""]
    rows = []
    for sl, k, a, b in cands[: args.n]:
        lines.append(f"=== {a['form']}  ({k[1]}, {sl} Wikipedia languages)")
        ex = {"qid": k[0], "category": k[1], "label": a["form"], "sitelinks": sl}
        for role, r in (("chat", a), ("base", b)):
            pieces = [tok.decode([t]) for t in r["answer_ids"]]
            ex[role] = [{"piece": p, "G_T1": g, "rank": rk, "kept_by_G": g < args.top_p}
                        for p, g, rk in zip(pieces, r["G_T1.0_pos"], r["rank_pos"])]
            lines.append(f"  {role}: pieces {pieces}  (exact verdict: first failing position {r[f]})")
            for j, x in enumerate(ex[role]):
                lines.append(f"     piece {j} {x['piece']!r:16s} mass ahead {x['G_T1']:.3f}  rank {x['rank']:6d}  -> "
                             f"{'KEPT' if x['kept_by_G'] else 'CUT'} at top-p {args.top_p}")
        rows.append(ex)
        lines.append("")
    out = root / "examples"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"walkthrough_{args.spec}.txt").write_text("\n".join(lines), encoding="utf-8")
    write_json(out / f"walkthrough_{args.spec}.json", {"n_candidates": len(cands), "examples": rows,
                                                       "args": vars(args)})
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()
