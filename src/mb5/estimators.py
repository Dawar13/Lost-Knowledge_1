"""P4 (E10): sample-based richness estimates against the exact reachable set.

From the sample check of a run (samples.jsonl.gz), per sampler and category:
  N         samples that name a sampled member
  S_obs     distinct members named
  f1, f2    members named exactly once / twice
  Chao1     S_obs + f1^2 / (2 f2)   (S_obs + f1 (f1 - 1) / 2 when f2 = 0)
  ACE       abundance-based coverage estimator, rare threshold 10 (Chao and Lee 1992)
  coverage  Good-Turing sample coverage 1 - f1 / N
and the exact sets for the same sampler and template 0 (robust verdict, any surface form):
  reachable members, recognised members, recognised and reachable members.
docs/04 s.4: any estimator fitted to samples targets the reachable support, so its shortfall against
the recognised set should be about RBU x |recognised|. Also writes rarefaction curves (distinct members
among the first n samples). No GPU.

  python -m mb5.estimators --results results/G0_a3_5090 --model Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from .analysis import form_reachable, recognised_set
from .common import read_jsonl, write_json
from .metrics import write_csv

RAREFY = (10, 20, 50, 100, 200, 500, 1000)


def chao1(counts: Counter) -> float:
    s = len(counts)
    f1 = sum(1 for v in counts.values() if v == 1)
    f2 = sum(1 for v in counts.values() if v == 2)
    return s + (f1 * f1 / (2 * f2) if f2 else f1 * (f1 - 1) / 2)


def ace(counts: Counter, rare: int = 10) -> float:
    s_abund = sum(1 for v in counts.values() if v > rare)
    rare_c = {k: v for k, v in counts.items() if v <= rare}
    s_rare = len(rare_c)
    n_rare = sum(rare_c.values())
    f1 = sum(1 for v in rare_c.values() if v == 1)
    if n_rare == 0:
        return float(s_abund)
    c_ace = 1 - f1 / n_rare
    if c_ace == 0:
        return float("nan")
    gamma2 = max((s_rare / c_ace) * sum(i * (i - 1) * sum(1 for v in rare_c.values() if v == i)
                                        for i in range(1, rare + 1)) / (n_rare * (n_rare - 1)) - 1, 0) \
        if n_rare > 1 else 0.0
    return s_abund + s_rare / c_ace + f1 / c_ace * gamma2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--model", required=True, help="model tag (folder) with samples.jsonl.gz")
    ap.add_argument("--metrics-dir", default="metrics")
    args = ap.parse_args()
    root = Path(args.results)
    mdir = root / args.model
    samples = read_jsonl(mdir / "samples.jsonl.gz")
    reach = read_jsonl(mdir / "reach.jsonl.gz")
    rec, _ = recognised_set(read_jsonl(mdir / "recog.jsonl.gz"))
    specs = sorted({s["spec"] for s in samples})
    reach_t0 = defaultdict(lambda: defaultdict(bool))  # (category, qid) -> spec -> reachable via any form
    for r in reach:
        if r["template"] == 0:
            for s in specs:
                reach_t0[(r["category"], r["qid"])][s] |= form_reachable(r, s, "robust")
    by = defaultdict(list)
    for s in samples:
        by[(s["spec"], s["category"])].append(s)
    rows, rare_rows = [], []
    for (spec, cat), ss in sorted(by.items()):
        ss = sorted(ss, key=lambda x: x["i"])
        named = [x["text_match_qid"] for x in ss if x["text_match_qid"]]
        cnt = Counter(named)
        members = [q for (c, q) in reach_t0 if c == cat]
        reachable = {q for q in members if reach_t0[(cat, q)][spec]}
        recog = {q for q in members if rec.get(f"{q}|{cat}")}
        f1 = sum(1 for v in cnt.values() if v == 1)
        rows.append({"spec": spec, "category": cat, "n_samples": len(ss), "N_member_samples": len(named),
                     "S_obs": len(cnt), "f1": f1, "f2": sum(1 for v in cnt.values() if v == 2),
                     "chao1": chao1(cnt), "ace": ace(cnt), "coverage": 1 - f1 / len(named) if named else None,
                     "members": len(members), "reachable": len(reachable), "recognised": len(recog),
                     "recognised_and_reachable": len(recog & reachable),
                     "sampled_outside_reachable": len(set(cnt) - reachable)})
        for n in RAREFY:
            sub = [x["text_match_qid"] for x in ss[:n] if x["text_match_qid"]]
            rare_rows.append({"spec": spec, "category": cat, "n": n, "distinct_members": len(set(sub))})
    pooled = {}
    for spec in specs:
        rs = [r for r in rows if r["spec"] == spec]
        pooled[spec] = {k: sum(r[k] for r in rs) for k in ("S_obs", "chao1", "ace", "members", "reachable",
                                                            "recognised", "recognised_and_reachable",
                                                            "sampled_outside_reachable")}
        p = pooled[spec]
        p["chao1_over_reachable"] = p["chao1"] / p["reachable"] if p["reachable"] else None
        p["chao1_over_recognised"] = p["chao1"] / p["recognised"] if p["recognised"] else None
        p["shortfall_vs_recognised"] = p["recognised"] - p["chao1"]
        p["rbu_times_recognised"] = p["recognised"] - p["recognised_and_reachable"]
    out = root / args.metrics_dir
    write_csv(out / f"estimators_{args.model}.csv", rows)
    write_csv(out / f"rarefaction_{args.model}.csv", rare_rows)
    write_json(out / f"estimators_{args.model}.json", {"pooled": pooled, "note": "template 0, robust verdict"})
    for spec, p in pooled.items():
        print(f"[estimators] {args.model} {spec}: S_obs {p['S_obs']}, Chao1 {p['chao1']:.0f}, ACE {p['ace']:.0f}, "
              f"reachable {p['reachable']}, recognised {p['recognised']}, recognised&reachable "
              f"{p['recognised_and_reachable']}; sampled outside reachable {p['sampled_outside_reachable']}")


if __name__ == "__main__":
    main()
