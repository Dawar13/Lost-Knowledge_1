"""P6 measures (CPU only) for E13, E14 and E15 / gate G3b, exactly as pre-registered on
2026-11-09 (notes/decisions.md), from the outputs of mb5.p6 and the stored open-prompt and P5 runs.

  python -m mb5.p6_analysis --results results/P6 \
      --model "Qwen2.5-7B-Instruct:open=results/G0_5090/Qwen2.5-7B-Instruct,p5=results/P5/Qwen2.5-7B-Instruct,base=Qwen/Qwen2.5-7B" \
      --model "OLMo-2-1124-7B-Instruct:open=results/G1/OLMo-2-1124-7B-Instruct,p5=results/P5/OLMo-2-1124-7B-Instruct,base=allenai/OLMo-2-1124-7B"

Definitions (recognised, tau, cluster bootstrap) are those of mb5.analysis. Verdicts:
  open prompt   template 0 of the open-prompt run; an entity is robustly unreachable if every surface
                form is cut even with the target raised by the margin (the RBU headline mode).
  E13           population: recognised cued entities robustly unreachable under the open prompt.
                Main measure: share ROBUSTLY REACHABLE under the cue (some form survives with the target
                lowered by the margin: robust_reach). Also: share not robustly unreachable (lenient) and
                share answered greedily (every answer token ranked first). H5: main share > 0.5.
  E14           robust RBU over recognised members other than the pre-filled ones, list prompt against
                open prompt on the same entities. H6: list RBU < open RBU and > base model's open RBU at
                p0.90_T1 (results/G1/stages.csv).
  E15 / G3b     filter: keep an answer if V >= tau_c (tau at the 95th foil percentile; 90th and 99th as
                sensitivity). Members use their stored V; other answers the V from mb5.p6 e15. Invented
                rate = invented / non-empty among kept answers; recovery = (RBU_ref - RBU_filtered) / RBU_ref
                (the G3 formula), where a recognised entity counts as reachable under the filtered wide
                sampler if it is reachable under that sampler and its V passes the filter. A wide sampler
                passes G3b for a model if recovery >= 50% and the filtered invented rate is at most the
                unfiltered reference's plus 2 points.
                Reading of "both models must pass" with two wide samplers (stated here, flagged to the
                human): G3b passes if at least one wide sampler passes for every model; that sampler is
                the one the paper may recommend.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .analysis import cluster_ci, fmt, form_reachable, pooled, recognised_set
from .common import read_jsonl, write_json
from .p6 import REFERENCE, WIDE_SPECS
from .sample_check import norm

QUANTILES = (0.95, 0.90, 0.99)
MAIN_Q = 0.95
NAN = float("nan")


# ---------------------------------------------------------------- shared helpers
def taus(recog: list[dict], q: float = 0.95) -> tuple[dict, float]:
    """Per-category yes/no threshold at foil quantile q, with the same fallback as
    analysis.recognised_set (global quantile when a category has fewer than 30 foils)."""
    foil_v = defaultdict(list)
    for r in recog:
        if r["kind"] == "foil":
            foil_v[r["category"]].append(r["V"])
    all_foil = [v for vs in foil_v.values() for v in vs]
    glob = float(np.quantile(all_foil, q)) if all_foil else 0.5
    return {c: (float(np.quantile(vs, q)) if len(vs) >= 30 else glob) for c, vs in foil_v.items()}, glob


def ent_reach(rows: list[dict], spec: str, mode: str = "robust", template: int | None = None) -> dict[str, bool]:
    """entity key -> reachable via any surface form. mode robust: not robustly unreachable
    (analysis.form_reachable); mode strict: robustly reachable (survives the margin against it)."""
    out: dict[str, bool] = {}
    for r in rows:
        if template is not None and r.get("template", template) != template:
            continue
        k = r["qid"] + "|" + r["category"]
        ok = bool(r[f"robust_reach__{spec}"]) if mode == "strict" else form_reachable(r, spec, "robust")
        out[k] = out.get(k, False) or ok
    return out


def greedy_set(rows: list[dict]) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for r in rows:
        k = r["qid"] + "|" + r["category"]
        out[k] = out.get(k, False) or all(x == 0 for x in r["rank_pos"])
    return out


def _pool(per: dict, idx: int) -> dict:
    cats = [c for c in per if per[c][0] > 0]
    v = pooled(per, cats, idx) if cats else NAN
    return {"share": v, "ci": list(cluster_ci(per, idx)) if cats else [NAN, NAN]}


# ---------------------------------------------------------------- E13
def e13_summary(open_rows, recog, cue_rows, specs) -> dict:
    rec, validity = recognised_set(recog)
    cued = sorted({r["qid"] + "|" + r["category"] for r in cue_rows})
    greedy = greedy_set(cue_rows)
    res = {"n_cued_entities": len(cued), "n_cued_recognised": sum(bool(rec.get(k)) for k in cued), "specs": {}}
    for s in specs:
        open_ok = ent_reach(open_rows, s, "robust", template=0)
        strict = ent_reach(cue_rows, s, "strict")
        lenient = ent_reach(cue_rows, s, "robust")
        per = defaultdict(lambda: [0, 0, 0, 0])  # population, strict, lenient, greedy
        ctx = defaultdict(lambda: [0, 0, 0])  # all recognised cued: n, reachable under cue (strict), open
        missing = 0
        for k in cued:
            if not rec.get(k):
                continue
            if k not in open_ok:
                missing += 1
                continue
            cat = k.split("|", 1)[1]
            ctx[cat][0] += 1
            ctx[cat][1] += strict.get(k, False)
            ctx[cat][2] += open_ok[k]
            if open_ok[k]:
                continue
            per[cat][0] += 1
            per[cat][1] += strict.get(k, False)
            per[cat][2] += lenient.get(k, False)
            per[cat][3] += greedy.get(k, False)
        main = _pool(per, 1)
        res["specs"][s] = {
            "population": sum(v[0] for v in per.values()),
            "reachable_under_cue": main["share"], "reachable_under_cue_ci": main["ci"],
            "reachable_under_cue_lenient": _pool(per, 2)["share"],
            "answered_greedily": _pool(per, 3)["share"], "answered_greedily_ci": _pool(per, 3)["ci"],
            "h5_holds": bool(main["share"] > 0.5) if not math.isnan(main["share"]) else None,
            "per_category": {c: {"population": v[0], "reachable_under_cue": v[1], "lenient": v[2], "greedy": v[3]}
                             for c, v in sorted(per.items())},
            "context_all_recognised_cued": {
                "n": sum(v[0] for v in ctx.values()),
                "reachable_under_cue": _pool(ctx, 1)["share"], "reachable_under_open_prompt": _pool(ctx, 2)["share"]},
            "recognised_without_open_verdict": missing,
        }
    res["recognition_validity"] = {k: v for k, v in validity.items() if k != "tau_per_category"}
    return res


# ---------------------------------------------------------------- E14
def e14_summary(open_rows, recog, list_rows, specs, base_rbu: float | None, prefill: dict | None = None) -> dict:
    rec, _ = recognised_set(recog)
    listed = sorted({r["qid"] + "|" + r["category"] for r in list_rows})
    res = {"n_scored_entities": len(listed), "base_rbu_p0.90_T1": base_rbu, "specs": {},
           "prefill_sizes": {c: len(v) for c, v in sorted((prefill or {}).items())}}
    for s in specs:
        open_ok = ent_reach(open_rows, s, "robust", template=0)
        list_ok = ent_reach(list_rows, s, "robust")
        per = defaultdict(lambda: [0, 0, 0, 0, 0])  # n, list unreachable, open unreachable, rescued, lost
        for k in listed:
            if not rec.get(k) or k not in open_ok:
                continue
            cat = k.split("|", 1)[1]
            per[cat][0] += 1
            per[cat][1] += not list_ok.get(k, False)
            per[cat][2] += not open_ok[k]
            per[cat][3] += (not open_ok[k]) and list_ok.get(k, False)
            per[cat][4] += open_ok[k] and not list_ok.get(k, False)
        lst, opn = _pool(per, 1), _pool(per, 2)
        h6 = None
        if not (math.isnan(lst["share"]) or math.isnan(opn["share"]) or base_rbu is None):
            h6 = bool(lst["share"] < opn["share"] and lst["share"] > base_rbu)
        res["specs"][s] = {
            "n_recognised": sum(v[0] for v in per.values()),
            "rbu_list": lst["share"], "rbu_list_ci": lst["ci"], "rbu_open": opn["share"], "rbu_open_ci": opn["ci"],
            "rescued_by_list": sum(v[3] for v in per.values()), "lost_in_list": sum(v[4] for v in per.values()),
            "h6_holds": h6,
            "per_category": {c: {"n": v[0], "list_unreachable": v[1], "open_unreachable": v[2]}
                             for c, v in sorted(per.items())},
        }
    return res


# ---------------------------------------------------------------- E15 / G3b
def _rates(rows, keep=None) -> dict:
    ne = [r for r in rows if r["class"] != "empty"]
    kept = [r for r in ne if keep is None or keep(r)]
    n = len(kept)

    def rate(c):
        return sum(r["class"] == c for r in kept) / n if n else NAN

    return {"n_nonempty": len(ne), "n_kept": n, "kept_share": n / len(ne) if ne else NAN,
            "invented_rate": rate("invented"), "member_rate": rate("member"), "valid_rate": rate("valid"),
            "foil_rate": rate("foil"),
            "distinct_members": len({(r["category"], r["member_qid"]) for r in kept if r["class"] == "member"})}


def e15_summary(samples, recog, scores, p5_reach, wide=WIDE_SPECS, ref=REFERENCE, quantiles=QUANTILES,
                main_q=MAIN_Q, frontier: dict | None = None) -> dict:
    member_v = {r["qid"] + "|" + r["category"]: r["V"] for r in recog if r["kind"] == "member"}
    score_v = {(r["category"], r["key"]): r["V"] for r in scores}
    rec, _ = recognised_set(recog)

    def v_of(r):
        if r["class"] == "member":
            return member_v[r["member_qid"] + "|" + r["category"]]
        k = (r["category"], norm(r["text"]))
        if k not in score_v:
            raise KeyError(f"no e15 score for {k}: run mb5.p6 --steps e15 first")
        return score_v[k]

    by_spec = defaultdict(list)
    for r in samples:
        if r["spec"] in (*wide, ref):
            by_spec[r["spec"]].append(r)
    ref_ok = ent_reach(p5_reach, ref, "robust")
    recd = sorted(k for k in ref_ok if rec.get(k))
    rbu_ref = sum(not ref_ok[k] for k in recd) / len(recd) if recd else NAN
    ref_unf = _rates(by_spec[ref])
    res = {"reference": ref, "reference_unfiltered": ref_unf, "rbu_reference": rbu_ref, "n_recognised": len(recd),
           "quantiles": {}, "checks": {}}
    for q in quantiles:
        tau, glob = taus(recog, q)

        def keep(r, tau=tau, glob=glob):
            return v_of(r) >= tau.get(r["category"], glob)

        rows = {}
        for s in (*wide, ref):
            unf, fil = _rates(by_spec[s]), _rates(by_spec[s], keep)
            ne = [r for r in by_spec[s] if r["class"] != "empty"]
            rem_members = [r for r in ne if r["class"] == "member" and not keep(r)]
            rem_valid = [r for r in ne if r["class"] == "valid" and not keep(r)]
            s_ok = ent_reach(p5_reach, s, "robust")
            passed = {k: s_ok.get(k, False) and member_v[k] >= tau.get(k.split("|", 1)[1], glob) for k in recd}
            rbu_f = sum(not passed[k] for k in recd) / len(recd) if recd else NAN
            rbu_u = sum(not s_ok.get(k, False) for k in recd) / len(recd) if recd else NAN
            unreach_ref = [k for k in recd if not ref_ok[k]]
            rows[s] = {
                "unfiltered": unf, "filtered": fil,
                "invented_delta_vs_reference": fil["invented_rate"] - ref_unf["invented_rate"],
                "rbu_unfiltered": rbu_u, "rbu_filtered": rbu_f,
                "recovery": (rbu_ref - rbu_f) / rbu_ref if rbu_ref else NAN,
                "recovery_unfiltered": (rbu_ref - rbu_u) / rbu_ref if rbu_ref else NAN,
                "recovered_share_direct": (sum(passed[k] for k in unreach_ref) / len(unreach_ref)
                                           if unreach_ref else NAN),
                "member_answers_removed": len(rem_members),
                "distinct_members_removed": len({(r["category"], r["member_qid"]) for r in rem_members}),
                "valid_answers_removed": len(rem_valid),
            }
            if s in wide:
                rows[s]["passes"] = bool(rows[s]["recovery"] >= 0.5 and rows[s]["invented_delta_vs_reference"] <= 0.02)
        res["quantiles"][str(q)] = {"tau_per_category": tau, "rows": rows}
    if frontier:  # the unfiltered numbers must reproduce mb5.frontier's
        fr = {r["spec"]: r for r in frontier["rows"]}
        for s in (*wide, ref):
            r0 = res["quantiles"][str(main_q)]["rows"][s]
            res["checks"][s] = {"invented_rate_matches_frontier": abs(r0["unfiltered"]["invented_rate"]
                                                                      - fr[s]["invented_rate"]) < 1e-9,
                                "rbu_matches_frontier": abs(r0["rbu_unfiltered"] - fr[s]["rbu_robust"]) < 1e-9}
    main = res["quantiles"][str(main_q)]["rows"]
    res["g3b"] = {"main_quantile": main_q, "by_sampler": {s: main[s]["passes"] for s in wide}}
    return res


def apply_classes(samples: list[dict], classes: list[dict]) -> list[dict]:
    """The P5 samples with class and member_qid replaced by the multilingual ones (mb5.invented_ml
    samples_classes_<model>.jsonl.gz), joined on (spec, category, i). The file must have been made
    from the same samples: its English class must equal the stored class on every row."""
    idx = {(c["spec"], c["category"], c["i"]): c for c in classes}
    out = []
    for r in samples:
        c = idx.get((r["spec"], r["category"], r["i"]))
        if c is None or c["class_en"] != r["class"]:
            raise ValueError(f"classes file does not match the samples at {(r['spec'], r['category'], r['i'])}")
        out.append(dict(r, **{"class": c["class_ml"], "member_qid": c["member_qid_ml"]}))
    if len(idx) != len(samples):
        raise ValueError(f"classes file has {len(idx)} rows for {len(samples)} samples")
    return out


def g3b_decision(e15: dict[str, dict], wide=WIDE_SPECS) -> dict:
    by_sampler = {s: all(m["g3b"]["by_sampler"][s] for m in e15.values()) for s in wide}
    return {"by_sampler": by_sampler, "passes": any(by_sampler.values()),
            "recommended": [s for s, ok in by_sampler.items() if ok]}


def g3b_report(e15: dict[str, dict], wide=WIDE_SPECS, english: dict[str, dict] | None = None) -> str:
    """english: the same summaries under the English-only invented-name metric, reported beside the
    decisive multilingual ones (notes/decisions.md 2026-11-09). None: e15 itself is English-only."""
    d = g3b_decision(e15, wide)
    L = [f"# Gate G3b decision: **{'PASS' if d['passes'] else 'FAIL'}** (sample wide, then self-check; "
         "provisional until the hand audit of kept invented answers)", "",
         "Rule (notes/decisions.md, 2026-11-09, fixed before the run): a wide sampler passes for a model if it "
         "recovers at least 50% of the reference sampler's unreachable recognised entities (robust RBU) and, after "
         "keeping only answers whose own yes/no score V reaches the category threshold (95th foil percentile), "
         "the invented-name rate is at most 2 points above the unfiltered reference. Reading: G3b passes if at "
         "least one wide sampler passes for every model. Data: the stored P5 samples (pilot_v2, template 0, "
         "1,000 per category per sampler).", "",
         "| wide sampler | passes for every model |", "|---|---|"]
    L += [f"| {s} | {'yes' if ok else 'no'} |" for s, ok in d["by_sampler"].items()]
    L.append("")
    for m, r in e15.items():
        ru = r["reference_unfiltered"]
        L += [f"## {m} (reference: {r['reference']}, unfiltered invented rate {fmt(ru['invented_rate'])}, "
              f"robust RBU {fmt(r['rbu_reference'])})", ""]
        for q, block in r["quantiles"].items():
            L += [f"Threshold at the {round(100 * float(q))}th foil percentile"
                  f"{' (decisive)' if float(q) == MAIN_Q else ' (sensitivity)'}:", "",
                  "| sampler | recovery | invented rate kept | change vs reference | kept share | distinct members "
                  "kept | member answers removed | valid answers removed | passes |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for s, x in block["rows"].items():
                f = x["filtered"]
                ok = x.get("passes")
                L.append(f"| {s}{' (filtered)' if s == r['reference'] else ''} | {fmt(x['recovery'])} | "
                         f"{fmt(f['invented_rate'])} | {100 * x['invented_delta_vs_reference']:+.1f} pts | "
                         f"{fmt(f['kept_share'])} | {f['distinct_members']} | {x['member_answers_removed']} | "
                         f"{x['valid_answers_removed']} | {'-' if ok is None else ('yes' if ok else 'no')} |")
            L.append("")
        bad = {s: c for s, c in r.get("checks", {}).items() if not all(c.values())}
        if bad:
            L += [f"WARNING: unfiltered numbers do not reproduce frontier.json: {bad}", ""]
    if english is not None:
        de = g3b_decision(english, wide)
        L += ["## English-only invented-name metric (reported beside; not decisive)", "",
              "| wide sampler | passes for every model |", "|---|---|"]
        L += [f"| {s} | {'yes' if ok else 'no'} |" for s, ok in de["by_sampler"].items()]
        L += ["", "| model | sampler | invented rate kept | change vs reference | passes |", "|---|---|---|---|---|"]
        for m, r in english.items():
            for s, x in r["quantiles"][str(MAIN_Q)]["rows"].items():
                if s in wide:
                    L.append(f"| {m} | {s} | {fmt(x['filtered']['invented_rate'])} | "
                             f"{100 * x['invented_delta_vs_reference']:+.1f} pts | {'yes' if x['passes'] else 'no'} |")
        L.append("")
        L += ["Notes: 'invented' is the multilingual class of mb5.invented_ml (decisive, fixed before the run): the "
              "answer matches no label or alias, in any language Wikidata holds, of a member, an uncapped member or a "
              "foil. The English-only class of mb5.frontier is reported above. Both include wrong answers (real "
              "people of another category) and real names Wikidata misses. The hand audit is extended to 100 kept "
              "invented answers per model (notes/decisions.md).", "", "Sign-off: ____________________"]
        return "\n".join(L)
    L += ["Notes: 'invented' is mb5.frontier's automatic class (no member of the uncapped Wikidata list, no foil); "
          "it includes real names Wikidata misses, spelling variants and non-name text. The hand audit is extended "
          "to 100 kept invented answers per model (notes/decisions.md).", "", "Sign-off: ____________________"]
    return "\n".join(L)


# ---------------------------------------------------------------- report
def summary_md(e13: dict, e14: dict) -> str:
    L = ["# P6 summary: ask-directly control (E13) and list control (E14)", "",
         "Pre-registered 2026-11-09 (notes/decisions.md). Computed by mb5.p6_analysis.", "",
         "## E13: recognised names unreachable under the open prompt, asked directly", "",
         "| model | sampler | population | robustly reachable under cue (95% CI) | lenient | answered greedily | "
         "H5 (> 50%) |", "|---|---|---|---|---|---|---|"]
    for m, r in e13.items():
        for s, x in r["specs"].items():
            ci = x["reachable_under_cue_ci"]
            L.append(f"| {m} | {s} | {x['population']} | {fmt(x['reachable_under_cue'])} ({fmt(ci[0])} to "
                     f"{fmt(ci[1])}) | {fmt(x['reachable_under_cue_lenient'])} | {fmt(x['answered_greedily'])} | "
                     f"{x['h5_holds']} |")
    L += ["", "## E14: list prompt with the most frequent answers pre-filled", "",
          "| model | sampler | recognised scored | RBU list (95% CI) | RBU open | base RBU (p0.90, open) | rescued | "
          "lost | H6 |", "|---|---|---|---|---|---|---|---|---|"]
    for m, r in e14.items():
        for s, x in r["specs"].items():
            ci = x["rbu_list_ci"]
            L.append(f"| {m} | {s} | {x['n_recognised']} | {fmt(x['rbu_list'])} ({fmt(ci[0])} to {fmt(ci[1])}) | "
                     f"{fmt(x['rbu_open'])} | {fmt(r['base_rbu_p0.90_T1'])} | {x['rescued_by_list']} | "
                     f"{x['lost_in_list']} | {x['h6_holds']} |")
        L.append(f"\nPre-fill sizes for {m}: {r['prefill_sizes']}")
    return "\n".join(L) + "\n"


def _parse_model(s: str) -> tuple[str, dict]:
    tag, rest = s.split(":", 1)
    return tag, dict(kv.split("=", 1) for kv in rest.split(","))


def base_rbu_from_stages(path: Path, base_id: str | None) -> float | None:
    if not base_id or not path.exists():
        return None
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["model"] == base_id:
                return float(r["rbu_p0.90_T1"])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results/P6")
    ap.add_argument("--model", action="append", required=True,
                    help="TAG:open=DIR,p5=DIR,base=HF_ID (base optional; for H6)")
    ap.add_argument("--stages", default="results/G1/stages.csv")
    ap.add_argument("--gates", default="results/gates")
    ap.add_argument("--e14-variant", default="", help="suffix of a non-default E14 run, e.g. _member_or_valid")
    ap.add_argument("--classes-dir", default=None,
                    help="mb5.invented_ml output (samples_classes_<tag>.jsonl.gz): decide G3b under the multilingual "
                         "invented-name metric and report the English-only one beside it (notes/decisions.md "
                         "2026-11-09). Without it, the English-only P5 classes are used, as before.")
    args = ap.parse_args()
    root = Path(args.results)
    all13, all14, all15, all15_en = {}, {}, {}, {}
    for spec in args.model:
        tag, kv = _parse_model(spec)
        mdir, odir, p5 = root / tag, Path(kv["open"]), Path(kv["p5"])
        recog = read_jsonl(odir / "recog.jsonl.gz")
        open_rows = None
        if (mdir / "e13_reach.jsonl.gz").exists() or (mdir / f"e14{args.e14_variant}_reach.jsonl.gz").exists():
            open_rows = [r for r in read_jsonl(odir / "reach.jsonl.gz") if r.get("template") == 0]
        if (mdir / "e13_reach.jsonl.gz").exists():
            specs = list(json.loads((mdir / "e13_meta.json").read_text())["specs"])
            all13[tag] = e13_summary(open_rows, recog, read_jsonl(mdir / "e13_reach.jsonl.gz"), specs)
            write_json(mdir / "e13_summary.json", all13[tag])
        f14 = mdir / f"e14{args.e14_variant}_reach.jsonl.gz"
        if f14.exists():
            specs = list(json.loads((mdir / f"e14{args.e14_variant}_meta.json").read_text())["specs"])
            pre = json.loads((mdir / f"e14{args.e14_variant}_prompts.json").read_text(encoding="utf-8"))
            all14[tag] = e14_summary(open_rows, recog, read_jsonl(f14), specs,
                                     base_rbu_from_stages(Path(args.stages), kv.get("base")), pre["prefill"])
            write_json(mdir / f"e14{args.e14_variant}_summary.json", all14[tag])
        if (mdir / "e15_scores.jsonl.gz").exists():
            fr = p5 / "frontier.json"
            samples = read_jsonl(p5 / "samples_frontier.jsonl.gz")
            scores, p5_reach = read_jsonl(mdir / "e15_scores.jsonl.gz"), read_jsonl(p5 / "reach.jsonl.gz")
            en = e15_summary(samples, recog, scores, p5_reach,
                             frontier=json.loads(fr.read_text()) if fr.exists() else None)
            if args.classes_dir:
                ml = e15_summary(apply_classes(samples, read_jsonl(Path(args.classes_dir) /
                                                                   f"samples_classes_{tag}.jsonl.gz")),
                                 recog, scores, p5_reach)
                all15[tag], all15_en[tag] = ml | {"metric": "multilingual"}, en | {"metric": "english"}
                write_json(mdir / "e15_summary_en.json", all15_en[tag])
            else:
                all15[tag] = en
            write_json(mdir / "e15_summary.json", all15[tag])
    if all13 or all14:
        (root / "summary.md").write_text(summary_md(all13, all14), encoding="utf-8")
        print(summary_md(all13, all14))
    if all15 and len(all15) == len(args.model):
        rep = g3b_report(all15, english=all15_en or None)
        gates = Path(args.gates)
        gates.mkdir(parents=True, exist_ok=True)
        (gates / "G3b.md").write_text(rep, encoding="utf-8")
        dec = g3b_decision(all15)
        if all15_en:
            dec |= {"metric": "multilingual", "english_only": g3b_decision(all15_en)}
        write_json(root / "g3b.json", dec)
        print(rep)


if __name__ == "__main__":
    main()
