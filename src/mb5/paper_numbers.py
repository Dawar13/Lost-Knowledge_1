"""Generate paper/numbers.tex: every number in the paper as a LaTeX macro read from results/.

CLAUDE.md rule 5: numbers in the paper come from files in results/ produced by committed code. The
paper never types a number; it uses these macros. A missing result becomes \\missing{name} (renders as
a red marker) and is listed on stdout, so nothing is invented.

  python -m mb5.paper_numbers            # writes paper/numbers.tex and paper/numbers_sources.json
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

R = Path("results")
MACROS: dict[str, str] = {}
SOURCES: dict[str, str] = {}
MISSING: list[str] = []


def _csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def put(name: str, value, src: str, kind: str = "pct", digits: int = 1):
    if value is None or value == "" or (isinstance(value, float) and value != value):
        MACROS[name] = f"\\missing{{{name}}}"
        MISSING.append(name)
        return
    v = float(value) if kind in ("pct", "num", "p", "signed") else value
    if kind == "pct":
        s = f"{100 * v:.{digits}f}\\%"
    elif kind == "signed":  # a signed difference in points, typeset in maths so the minus is a minus sign
        s = f"${v:+.{digits}f}$"
    elif kind == "sci":  # m x 10^k with `digits` decimals in the mantissa
        import math

        k = math.floor(math.log10(abs(v)))
        mant = v / 10 ** k
        if round(mant, digits) >= 10:
            k, mant = k + 1, mant / 10
        s = f"${mant:.{digits}f}\\times10^{{{k}}}$"
    elif kind == "ord":
        n = int(value)
        suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        s = f"{n}{suf}"
    elif kind == "num":
        s = f"{v:,.{digits}f}".replace(",", "{,}")
    elif kind == "int":
        s = f"{int(value):,}".replace(",", "{,}")
    elif kind == "p":  # "<10^{k}" with the smallest k that is a true upper bound; plain value above 0.001
        import math

        if v < 1e-3:
            k = math.floor(math.log10(v)) + 1 if v > 0 else -300
            s = f"$<10^{{{k}}}$"
        else:
            s = f"$={v:.3f}$"
    else:
        s = str(value)
    MACROS[name] = s
    SOURCES[name] = src


def safe(fn, *names):
    try:
        fn()
    except (FileNotFoundError, StopIteration, KeyError, IndexError) as e:
        for n in names:
            if n not in MACROS:
                MACROS[n] = f"\\missing{{{n}}}"
                MISSING.append(n)
        print(f"[paper_numbers] missing input for {names}: {type(e).__name__} {e}")


def rbu_row(path, spec, mode="robust"):
    return next(r for r in _csv(path) if r["spec"] == spec and r["mode"] == mode)


def g0():
    src = "results/G0_5090/metrics/rbu_Qwen2.5-7B-Instruct.csv"
    for spec, tag in (("shipped_hf", "Shipped"), ("p0.90_T1", "Pnine"), ("k50_T1", "Kfifty"), ("minp0.10_T1", "Minp")):
        r = rbu_row(src, spec)
        put(f"gzQwenInst{tag}", r["rbu_mean"], src)
        lo, hi = json.loads(r["rbu_mean_ci"])
        put(f"gzQwenInst{tag}Lo", lo, src)
        put(f"gzQwenInst{tag}Hi", hi, src)
    srcb = "results/G0_5090/metrics/rbu_Qwen2.5-7B.csv"
    for spec, tag in (("p0.90_T1", "Pnine"), ("k50_T1", "Kfifty")):
        put(f"gzQwenBase{tag}", rbu_row(srcb, spec)["rbu_mean"], srcb)
    h = json.loads((R / "G0_5090/metrics/headline.json").read_text())
    put("gzQwenInstRecognised", h["instruct"]["recognition"]["n_recognised"], "headline.json", "int")
    put("gzQwenInstAuroc", h["instruct"]["recognition"]["auroc_yesno"], "headline.json", "num", 3)
    put("gzEntities", h["instruct"]["recognition"]["n_members"], "headline.json", "int")
    paired = next(r for r in _csv(R / "G0_5090/metrics/paired.csv") if r["spec"] == "p0.90_T1" and r["template"] == "0")
    put("gzPairedInstOnly", paired["instruct_only_unreachable"], "G0_5090/metrics/paired.csv", "int")
    put("gzPairedBaseOnly", paired["base_only_unreachable"], "G0_5090/metrics/paired.csv", "int")
    sc = json.loads((R / "G0_5090/Qwen2.5-7B-Instruct/sample_check.json").read_text())
    put("gzSamples", sum(r["n_samples"] for r in sc), "sample_check.json", "int")
    put("gzViolations", sum(r["violations"] for r in sc), "sample_check.json", "int")
    put("gzBoundaryHits", sum(r.get("boundary_hits", 0) for r in sc), "sample_check.json", "int")
    put("gzExactViolations", sum(r.get("exact_violations", 0) for r in sc), "sample_check.json", "int")
    cal = json.loads((R / "G0_5090/metrics/calibration_Qwen2.5-7B-Instruct.json").read_text())
    put("gzEps", cal["epsilon_rule"], "calibration", "num", 2)
    put("gzGapQnine", cal["logit_gap"]["rel_near"]["q0.999"], "calibration", "num", 2)
    put("gzGapMax", cal["logit_gap"]["rel_near"]["max"], "calibration", "num", 2)
    put("gzCalSeqs", cal["total"]["distinct"], "calibration", "int")
    put("gzCalExact", cal["total"]["unreachable"], "calibration", "int")
    fp = [r for r in _csv(R / "G0_5090/metrics/failpos_Qwen2.5-7B-Instruct.csv") if r["spec"] == "p0.90_T1"]
    put("gzQwenInstFirstTok", next(r["share"] for r in fp if r["fail_position"] == "0"), "failpos")
    fpb = [r for r in _csv(R / "G0_5090/metrics/failpos_Qwen2.5-7B.csv") if r["spec"] == "p0.90_T1"]
    put("gzQwenBaseFirstTok", next(r["share"] for r in fpb if r["fail_position"] == "0"), "failpos")
    ct = {r["category"]: r for r in _csv(R / "G0_5090/metrics/reachable_set_Qwen2.5-7B-Instruct.csv")
          if r["spec"] == "p0.90_T1" and r["template"] == "0" and r["category"] == "us_state_capitals"}
    put("gzCapitalsReachInst", ct["us_state_capitals"]["reachable_robust"], "reachable_set", "int")
    cb = {r["category"]: r for r in _csv(R / "G0_5090/metrics/reachable_set_Qwen2.5-7B.csv")
          if r["spec"] == "p0.90_T1" and r["template"] == "0" and r["category"] == "us_state_capitals"}
    put("gzCapitalsReachBase", cb["us_state_capitals"]["reachable_robust"], "reachable_set", "int")
    put("gzCapitalsMembers", cb["us_state_capitals"]["members"], "reachable_set", "int")


def g1():
    rows = {(r["family"], r["stage"]): r for r in _csv(R / "G1/stages.csv")}
    for (fam, stage), tag in ((("olmo2", "base"), "OlmoBase"), (("olmo2", "sft"), "OlmoSft"),
                              (("olmo2", "dpo"), "OlmoDpo"), (("olmo2", "instruct"), "OlmoInst")):
        r = rows[(fam, stage)]
        put(f"goRbu{tag}", r["rbu_p0.90_T1"], "G1/stages.csv")
        lo, hi = json.loads(r["rbu_p0.90_T1_ci"])
        put(f"goRbu{tag}Lo", lo, "G1/stages.csv")
        put(f"goRbu{tag}Hi", hi, "G1/stages.csv")
        put(f"goFirstTok{tag}", r["fail_first_token_p0.90_T1"], "G1/stages.csv")
        put(f"goTopOne{tag}", r["first_token_top1_mass_mean"], "G1/stages.csv", "num", 2)
        put(f"goKfifty{tag}", r["rbu_k50_T1"], "G1/stages.csv")
    for (fam, stage), tag in ((("qwen2.5", "base"), "QwenBase"), (("qwen2.5", "instruct"), "QwenInst")):
        put(f"goTopOne{tag}", rows[(fam, stage)]["first_token_top1_mass_mean"], "G1/stages.csv", "num", 2)
        put(f"goKfifty{tag}", rows[(fam, stage)]["rbu_k50_T1"], "G1/stages.csv")
    pr = {(r["family"], r["from"], r["to"], r["spec"]): r for r in _csv(R / "G1/paired.csv")}
    r = pr[("olmo2", "base", "instruct", "p0.90_T1")]
    put("goPairedInstOnly", r["to_only_unreachable"], "G1/paired.csv", "int")
    put("goPairedBaseOnly", r["from_only_unreachable"], "G1/paired.csv", "int")
    put("goRatioOlmo", r["ratio"], "G1/paired.csv", "num", 2)
    put("goRatioQwen", pr[("qwen2.5", "base", "instruct", "p0.90_T1")]["ratio"], "G1/paired.csv", "num", 2)
    # sign tests, base against final model (normal approximation, two-sided). The stored p underflows to
    # 0.0, so p is recomputed from the stored z with erfc, which stays exact far into the tail.
    import math

    for fam, tg in (("olmo2", "Olmo"), ("qwen2.5", "Qwen")):
        z = float(pr[(fam, "base", "instruct", "p0.90_T1")]["z"])
        put(f"goPairedZ{tg}", z, "G1/paired.csv", "num", 1)
        put(f"goPairedP{tg}", math.erfc(abs(z) / math.sqrt(2)), "G1/paired.csv (p from z)", "p")
    # Holm over the pre-registered tests that carry p-values: the two H2 sign tests (G1) and the four G2
    # popularity slopes for unreachability (P3); every gate decision rule required p < 0.01.
    ps = [math.erfc(abs(float(pr[(f, "base", "instruct", "p0.90_T1")]["z"])) / math.sqrt(2)) for f in ("olmo2", "qwen2.5")]
    for g in _csv(R / "P3/g2/popularity_gee.csv"):
        if g["outcome"].startswith("unreachable"):
            ps.append(math.erfc(abs(float(g["slope_per_decade"]) / float(g["se"])) / math.sqrt(2)))
    ps.sort()
    m, run, adj = len(ps), 0.0, []
    for i, p in enumerate(ps):
        run = max(run, min(1.0, (m - i) * p))
        adj.append(run)
    put("goHolmTests", m, "G1/paired.csv, P3/g2/popularity_gee.csv", "int")
    put("goHolmMaxP", max(adj), "G1/paired.csv, P3/g2/popularity_gee.csv (Holm)", "p")
    s = json.loads((R / "G1/g1_summary.json").read_text())
    put("goSpearman", s["mechanism"]["spearman_top1_mass_vs_rbu"], "g1_summary.json", "num", 2)
    put("goSpearmanN", s["mechanism"]["n_points"], "g1_summary.json", "int")
    n = v = 0
    for m in ("OLMo-2-1124-7B-SFT", "OLMo-2-1124-7B-DPO", "OLMo-2-1124-7B-Instruct"):
        sc = json.loads((R / f"G1/{m}/sample_check.json").read_text())
        n += sum(x["n_samples"] for x in sc)
        v += sum(x["violations"] for x in sc)
    put("goSamples", n, "G1 sample_check.json", "int")
    put("goViolations", v, "G1 sample_check.json", "int")


def robust():
    for pair, tag, tg in (("qwen2.5", "Qwen2.5-7B-Instruct", "QwenInst"), ("qwen2.5", "Qwen2.5-7B", "QwenBase"),
                          ("olmo2", "OLMo-2-1124-7B-Instruct", "OlmoInst"), ("olmo2", "OLMo-2-1124-7B", "OlmoBase")):
        src = R / f"P3_robust/metrics/{pair}/rbu_{tag}.csv"
        r = rbu_row(src, "p0.90_T1")
        put(f"rbCv{tg}", r["rbu_cv_over_templates"], str(src), "num", 3)
        for spec, t in (("p0.90_T1", "Tone"), ("p0.90_T0.7", "Tseven"), ("p0.90_T0.5", "Tfive")):
            put(f"rbTemp{tg}{t}", rbu_row(src, spec)["rbu_mean"], str(src))
        put(f"rbLabelOnly{tg}", r["rbu_mean_label_only"], str(src))


def p5():
    for m, tg in (("Qwen2.5-7B-Instruct", "Qwen"), ("OLMo-2-1124-7B-Instruct", "Olmo")):
        f = json.loads((R / f"P5/{m}/frontier.json").read_text())
        rows = {r["spec"]: r for r in f["rows"]}
        src = f"results/P5/{m}/frontier.json"
        put(f"pfRecovered{tg}", f["g3"]["recovered"], src)
        put(f"pfInvDelta{tg}", 100 * f["g3"]["invented_delta"], src, "signed", 1)
        for spec, t in (("reference", "Ref"), ("relax_first", "Relax"), ("untruncated_T1", "Untr"),
                        ("p0.99_T1", "Pnn"), ("k100_T1", "Khund"), ("eps3e-4_T1", "Eps")):
            r = rows[spec]
            put(f"pf{tg}{t}Rbu", r["rbu_robust"], src)
            put(f"pf{tg}{t}Distinct", r["distinct_members_sampled"], src, "int")
            put(f"pf{tg}{t}Inv", r["invented_rate"], src)
    for r in _csv(R / "P5/audit/invented_rate_sensitivity.csv"):
        tg = "Qwen" if r["model"].startswith("Qwen") else "Olmo"
        if r["spec"] == "relax_first":
            put(f"pfLooseDelta{tg}", 100 * float(r["loose_delta_vs_reference"]), "invented_rate_sensitivity.csv", "signed", 1)
        if r["spec"] == "reference":
            put(f"pfLooseRef{tg}", r["invented_rate_loose"], "invented_rate_sensitivity.csv")
            # how much looser matching lowers the reference sampler's invented-name rate, in points
            put(f"pfLooseDrop{tg}", 100 * (float(r["invented_rate_preregistered"]) - float(r["invented_rate_loose"])),
                "invented_rate_sensitivity.csv", "num", 1)


def p3():
    val = {r["model"]: r for r in _csv(R / "P3/g2/validity.csv")}
    src = "results/P3/g2/validity.csv"
    for m, tg in (("Qwen2.5-7B-Instruct", "QwenInst"), ("Qwen2.5-7B", "QwenBase"),
                  ("OLMo-2-1124-7B-Instruct", "OlmoInst"), ("OLMo-2-1124-7B", "OlmoBase")):
        r = val[m]
        put(f"ptRbu{tg}", r.get("rbu_p0.90_T1"), src)
        if r.get("rbu_p0.90_T1_ci"):
            lo, hi = json.loads(r["rbu_p0.90_T1_ci"])
            put(f"ptRbu{tg}Lo", lo, src)
            put(f"ptRbu{tg}Hi", hi, src)
        put(f"ptRecognised{tg}", r["n_recognised"], src, "int")
        put(f"ptAuroc{tg}", r["auroc_yesno"], src, "num", 3)
        if r.get("rbu_shipped_hf"):
            put(f"ptShipped{tg}", r["rbu_shipped_hf"], src)
        if r.get("sample_check_samples"):
            put(f"ptSamples{tg}", r["sample_check_samples"], src, "int")
            put(f"ptViolations{tg}", r["violations"], src, "int")
            put(f"ptBoundary{tg}", r["boundary_hits"], src, "int")
    # non-canonical token paths at scale: distinct entities ever produced only through another tokenisation
    import gzip
    from collections import defaultdict

    nc, nsamp = defaultdict(set), defaultdict(int)
    with gzip.open(R / "P3/Qwen2.5-7B-Instruct/samples.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            if s["id_hit_qid"] is None and s["text_match_qid"]:
                nc[s["spec"]].add((s["category"], s["text_match_qid"]))
                nsamp[s["spec"]] += 1
    put("ptNoncanonEntShipped", len(nc["shipped_hf"]), "P3 samples.jsonl.gz", "int")
    put("ptNoncanonEntPnine", len(nc["p0.90_T1"]), "P3 samples.jsonl.gz", "int")
    put("ptCategories", val["Qwen2.5-7B-Instruct"]["n_categories"], src, "int")
    put("ptMembers", val["Qwen2.5-7B-Instruct"]["n_members"], src, "int")
    for r in _csv(R / "P3/g2/popularity_gee.csv"):
        tg = {"Qwen2.5-7B-Instruct": "QwenInst", "Qwen2.5-7B": "QwenBase", "OLMo-2-1124-7B-Instruct": "OlmoInst",
              "OLMo-2-1124-7B": "OlmoBase"}[r["model"]]
        kind = "Unreach" if r["outcome"].startswith("unreachable") else "Recog"
        put(f"ptOr{kind}{tg}", r["odds_ratio_per_decade"], "popularity_gee.csv", "num", 2)
        lo, hi = json.loads(r["or_ci"])
        put(f"ptOr{kind}{tg}Lo", lo, "popularity_gee.csv", "num", 2)
        put(f"ptOr{kind}{tg}Hi", hi, "popularity_gee.csv", "num", 2)
    pr = next(r for r in _csv(R / "P3/metrics/qwen2.5/paired.csv") if r["spec"] == "p0.90_T1" and r["template"] == "all")
    put("ptPairedInstOnlyQwen", pr["instruct_only_unreachable"], "P3 paired.csv", "int")
    put("ptPairedBaseOnlyQwen", pr["base_only_unreachable"], "P3 paired.csv", "int")
    pr = next(r for r in _csv(R / "P3/metrics/olmo2/paired.csv") if r["spec"] == "p0.90_T1" and r["template"] == "all")
    put("ptPairedInstOnlyOlmo", pr["instruct_only_unreachable"], "P3 paired.csv", "int")
    put("ptPairedBaseOnlyOlmo", pr["base_only_unreachable"], "P3 paired.csv", "int")


def p4():
    for m, tg in (("Qwen2.5-7B-Instruct", "Qwen"), ("OLMo-2-1124-7B-Instruct", "Olmo")):
        p = json.loads((R / f"P3/metrics/estimators_{m}.json").read_text())["pooled"]
        src = f"results/P3/metrics/estimators_{m}.json"
        for spec, t in (("shipped_hf", "Shipped"), ("p0.90_T1", "Pnine")):
            if spec not in p:
                continue
            x = p[spec]
            put(f"pe{tg}{t}Sobs", x["S_obs"], src, "int")
            put(f"pe{tg}{t}Chao", x["chao1"], src, "num", 0)
            put(f"pe{tg}{t}Ace", x["ace"], src, "num", 0)
            put(f"pe{tg}{t}Reach", x["reachable"], src, "int")
            put(f"pe{tg}{t}Recog", x["recognised"], src, "int")
            put(f"pe{tg}{t}RecogReach", x["recognised_and_reachable"], src, "int")
            put(f"pe{tg}{t}Outside", x["sampled_outside_reachable"], src, "int")


def walkthrough_intro():
    """The US-state-capitals example of the introduction and Figure 1. G0_5090 is the main analysis.
    The sampled answers with their names come from attempt 3 (G0_a3_5090), whose generation and sample
    check are identical to attempt 2's (results/INDEX.md); the counts are checked against attempt 2."""
    import gzip

    cat, inst, base = "us_state_capitals", "Qwen2.5-7B-Instruct", "Qwen2.5-7B"
    ft = {}
    for m in (inst, base):
        d = json.loads((R / f"G0_5090/{m}/first_token.json").read_text(encoding="utf-8"))
        ft[m] = next(r for r in d["rows"] if r["category"] == cat and r["template"] == 0)
    src = "results/G0_5090/<model>/first_token.json (us_state_capitals, template 0)"
    for m, tg in ((inst, "Inst"), (base, "Base")):
        put(f"cw{tg}KeptPnine", ft[m]["kept_top_p_0.9"], src, "int")
        top = ft[m]["top_tokens"][0]
        put(f"cw{tg}TopPiece", tex_escape(top["token"].strip()), src, "str")
        put(f"cw{tg}TopPieceP", top["p"], src)
    sc = json.loads((R / f"G0_5090/{inst}/sample_check.json").read_text())
    x = next(r for r in sc if r["category"] == cat and r["spec"] == "shipped_hf")
    put("cwShippedSamples", x["n_samples"], "results/G0_5090/<inst>/sample_check.json", "int")
    put("cwShippedDistinct", x["distinct_valid"], "results/G0_5090/<inst>/sample_check.json", "int")
    labels = {}
    with open("data/tailcat/pilot_v2/entities.jsonl", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if e["category"] == cat:
                labels[e["qid"]] = e["label"]
    ta = sorted((r for r in _csv(R / f"G0_a3_5090/metrics/top_answers_{inst}.csv")
                 if r["spec"] == "shipped_hf" and r["category"] == cat), key=lambda r: int(r["rank"]))
    if len(ta) != x["distinct_valid"] or sum(int(r["count"]) for r in ta) != x["n_samples"]:
        raise KeyError("attempt-3 answers do not reproduce attempt 2's sample check")
    for r, tg in zip(ta, "ABCD"):
        put(f"cwAnsName{tg}", tex_escape(labels[r["matched_member_qid"]]), "results/G0_a3_5090/metrics/top_answers", "str")
        put(f"cwAnsCount{tg}", r["count"], "results/G0_a3_5090/metrics/top_answers", "int")
    val = next(r for r in _csv(R / f"G0_5090/metrics/validity_{inst}.csv") if r["category"] == cat)
    put("cwCapsMc", val["mc_accuracy"], "results/G0_5090/metrics/validity", "pct")
    rec = {r["qid"]: r["recognised"] == "True" for r in _csv(R / f"G0_5090/metrics/entities_{inst}.csv")
           if r["category"] == cat and r["template"] == "0"}
    rows = []
    with gzip.open(R / f"G0_5090/{inst}/reach.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["category"] == cat and r["template"] == 0 and r.get("is_label"):
                rows.append(r)
    srcr = "results/G0_5090/<inst>/reach.jsonl.gz (labels, template 0, robust verdict at top-p 0.9)"
    cut = [r for r in rows if rec.get(r["qid"]) and r["robust_unreach__p0.90_T1"]]
    # rule: the recognised, robustly unreachable capital with most sitelinks that is cut at the first token
    first = max((r for r in cut if r["failpos_robust__p0.90_T1"] == 0), key=lambda r: (r["sitelinks"], r["qid"]))
    put("cwFirstExName", tex_escape(first["form"]), srcr, "str")
    put("cwFirstExRank", first["rank_pos"][0] + 1, srcr, "ord")
    put("cwFirstExMass", first["G_T1.0_pos"][0], srcr)
    # rule: the same, cut at a later token; its rival is the reachable member sharing the tokens before the cut
    later = max((r for r in cut if r["failpos_robust__p0.90_T1"] > 0), key=lambda r: (r["sitelinks"], r["qid"]))
    j = later["failpos_robust__p0.90_T1"]
    rival = next(r for r in rows if r["qid"] != later["qid"] and not r["robust_unreach__p0.90_T1"]
                 and r["answer_ids"][:j] == later["answer_ids"][:j] and r["rank_pos"][j] == 0)
    if later["rank_pos"][j] != 1 or not later["robust_unreach__p0.95_T1"]:
        raise KeyError("walk-through wording assumes one token ahead and a cut at top-p 0.95 too")
    put("cwLaterExName", tex_escape(later["form"]), srcr, "str")
    put("cwLaterExRival", tex_escape(rival["form"]), srcr, "str")
    put("cwLaterExMass", later["G_T1.0_pos"][j], srcr)
    # batch jitter: the same first token after the same prompt, scored in two rows of the batched pass
    put("cwJitterA", later["G_T1.0_pos"][0], srcr, "num", 3)
    put("cwJitterB", rival["G_T1.0_pos"][0], srcr, "num", 3)


def zt():
    """Zero against tiny (results/analysis_cpu/support, mb5.support): how likely deleted names would be
    without truncation, the reachable ceiling on discovery, and the calibration of the log-probability reading.
    Main setting: full TailCat, template 0, recognised entities; shipped-sampler values omit the repetition
    penalty and are approximate."""
    h = json.loads((R / "analysis_cpu/support/headline.json").read_text(encoding="utf-8"))
    src = "results/analysis_cpu/support/headline.json"
    qi = h["scale|Qwen2.5-7B-Instruct|p0.90_T1"]
    t0, cv = qi["template0"], qi["curve_t0"]
    put("ztMedQQI", t0["median_q_max"], src, "sci", 1)
    put("ztMedSightQI", t0["median_samples_to_first_sight_q_max"], src, "sci", 0)
    put("ztSeenKQI", t0["share_seen_q_max"]["1000"], src)
    put("ztSeenMQI", t0["share_seen_q_max"]["1000000"], src)
    put("ztQmilliQI", t0["share_q_max_at_least"]["0.001"], src)
    put("ztMassQI", t0["deleted_mass_per_prompt_mean"], src, "pct", 2)
    put("ztUnreachQI", t0["n_unreachable"], src, "int")
    for key, tg in (("scale|Qwen2.5-7B|p0.90_T1", "QB"), ("scale|OLMo-2-1124-7B-Instruct|p0.90_T1", "OI"),
                    ("scale|OLMo-2-1124-7B|p0.90_T1", "OB")):
        put(f"ztMedQ{tg}", h[key]["template0"]["median_q_max"], src, "sci", 1)
    put("ztMedRatioQ", h["scale|Qwen2.5-7B|p0.90_T1"]["template0"]["median_q_max"] / t0["median_q_max"], src, "num", 0)
    put("ztPlateauQI", cv["plateau_reachable"], src, "int")
    put("ztRecQI", cv["n_recognised"], src, "int")
    put("ztPromptsQI", cv["n_prompts"], src, "int")
    put("ztUntrKQI", cv["1000"]["E_untruncated_qmax"], src, "num", 0)
    put("ztTruncKLoQI", cv["1000"]["E_truncated_lo"], src, "num", 0)
    put("ztTruncKHiQI", cv["1000"]["E_truncated_hi"], src, "num", 0)
    put("ztUntrMQI", cv["1000000"]["E_untruncated_qmax"], src, "num", 0)
    put("ztTruncMHiQI", cv["1000000"]["E_truncated_hi"], src, "num", 0)
    put("ztNstarQI", round(cv["n_star_qmax_exceeds_plateau"], -3), src, "num", 0)
    sh = h["scale|Qwen2.5-7B-Instruct|shipped_hf"]["curve_t0"]
    put("ztShipPlateauQI", sh["plateau_reachable"], src, "int")
    put("ztShipNstarQI", sh["n_star_qmax_exceeds_plateau"], src, "num", 0)
    put("ztShipUntrKQI", sh["1000"]["E_untruncated_qmax"], src, "num", 0)
    put("ztMassPilotQI", h["pilot|Qwen2.5-7B-Instruct|p0.90_T1"]["template0"]["deleted_mass_per_prompt_mean"], src, "pct", 2)
    for m, tg in (("Qwen2.5-7B-Instruct", "QI"), ("OLMo-2-1124-7B-Instruct", "OI")):
        c = h["calibration"][m]["untruncated_T1"]
        put(f"ztCalSamples{tg}", c["samples"], src, "int")
        put(f"ztCalObs{tg}", c["member_samples_observed"], src, "int")
        put(f"ztCalPred{tg}", c["member_samples_predicted_any"], src, "num", 0)
        put(f"ztCalDistObs{tg}", c["distinct_members_observed"], src, "int")
        put(f"ztCalDistPred{tg}", c["distinct_members_predicted_any"], src, "num", 1)


def region_controls():
    """Post hoc analyses (results/analysis_cpu/region and controls, mb5.region and mb5.controls): the
    South-West region gap with controls, the popularity effect with label-length and type controls, and the
    mechanism correlation within models."""
    A = R / "analysis_cpu"
    gap = {(r["model"], r["spec"], r["step"]): r for r in _csv(A / "region/gee_gap.csv")}
    src = "results/analysis_cpu/region/gee_gap.csv"

    def gap_put(name, key, ci=True):
        r = gap[key]
        put(name, r["gap_or"], src, "num", 2)
        if ci:
            put(name + "Lo", r["gap_or_ci_lo"], src, "num", 2)
            put(name + "Hi", r["gap_or_ci_hi"], src, "num", 2)
            put(name + "CI", f"{float(r['gap_or_ci_lo']):.2f} to {float(r['gap_or_ci_hi']):.2f}", src, "str")

    gap_put("rgGapRawQwen", ("Qwen2.5-7B-Instruct", "p0.90_T1", "M0"))
    gap_put("rgGapAdjQwen", ("Qwen2.5-7B-Instruct", "p0.90_T1", "Mall"))
    gap_put("rgGapAdjQwenShipped", ("Qwen2.5-7B-Instruct", "shipped_hf", "Mall"), ci=False)
    gap_put("rgGapAdjOlmo", ("OLMo-2-1124-7B-Instruct", "p0.90_T1", "Mall"))
    gap_put("rgGapAdjQwenBase", ("Qwen2.5-7B", "p0.90_T1", "Mall"), ci=False)
    gap_put("rgGapRawQwenBase", ("Qwen2.5-7B", "p0.90_T1", "M0"), ci=False)
    gap_put("rgGapRawOlmoBase", ("OLMo-2-1124-7B", "p0.90_T1", "M0"), ci=False)
    put("rgCompositionShareLeft", gap[("Qwen2.5-7B-Instruct", "p0.90_T1", "Mc")]["share_of_M0_gap"], src, "pct", 0)
    terms = {(r["model"], r["spec"], r["step"], r["term"]): r for r in _csv(A / "region/gee_region_terms.csv")}
    t = terms[("Qwen2.5-7B-Instruct", "p0.90_T1", "Mc", "log_n_recognised")]
    put("rgCompositionOR", t["or"], "results/analysis_cpu/region/gee_region_terms.csv", "num", 2)
    pr = {(r["pair"].split(" vs ")[0], r["region"]): r for r in _csv(A / "region/paired_region.csv")}
    for m, tg in (("Qwen2.5-7B-Instruct", "Qwen"), ("OLMo-2-1124-7B-Instruct", "Olmo")):
        for reg, rt in (("SOUTH", "South"), ("WEST", "West")):
            put(f"rgLoss{rt}{tg}", pr[(m, reg)]["loss_rate_given_base_reachable"], "region/paired_region.csv")
    br = {r["region"]: r for r in _csv(A / "region/by_region.csv")
          if r["model"] == "Qwen2.5-7B-Instruct" and r["spec"] == "p0.90_T1"}
    for reg, rt in (("europe", "Europe"), ("north_america", "NorthAmerica"), ("south_asia", "SouthAsia"),
                    ("southeast_asia", "SoutheastAsia"), ("africa", "Africa")):
        put(f"rgRecRate{rt}", br[reg]["recognition_rate"], "region/by_region.csv")
        put(f"rgRbu{rt}", br[reg]["rbu"], "region/by_region.csv")
    pc = {(r["model"], r["spec"], r["step"]): r for r in _csv(A / "controls/popularity_controls.csv")}
    srcp = "results/analysis_cpu/controls/popularity_controls.csv"
    r = pc[("Qwen2.5-7B-Instruct", "p0.90_T1", "P3")]
    put("ctPopOrQwenControlled", r["or"], srcp, "num", 2)
    put("ctPopOrQwenControlledLo", r["or_ci_lo"], srcp, "num", 2)
    put("ctPopOrQwenControlledHi", r["or_ci_hi"], srcp, "num", 2)
    put("ctPopOrQwenControlledCI", f"{float(r['or_ci_lo']):.2f} to {float(r['or_ci_hi']):.2f}", srcp, "str")
    put("ctLabelTokenOR", r["label_tokens_or"], srcp, "num", 2)
    put("ctPopOrOlmoControlled", pc[("OLMo-2-1124-7B-Instruct", "p0.90_T1", "P3")]["or"], srcp, "num", 2)
    put("ctPopOrOlmoBaseControlled", pc[("OLMo-2-1124-7B", "p0.90_T1", "P3")]["or"], srcp, "num", 2)
    put("ctPopOrQwenShipped", pc[("Qwen2.5-7B-Instruct", "shipped_hf", "P0")]["or"], srcp, "num", 3)
    put("ctPopOrQwenShippedControlled", pc[("Qwen2.5-7B-Instruct", "shipped_hf", "P3")]["or"], srcp, "num", 3)
    tok = [float(x["label_tokens_or"]) for x in pc.values() if x["step"] == "P3" and x["converged"] == "True"
           and x["label_tokens_or"]]
    put("ctLabelTokenPctMin", min(tok) - 1, srcp, "pct", 0)
    put("ctLabelTokenPctMax", max(tok) - 1, srcp, "pct", 0)
    mc = _csv(A / "controls/mechanism_controls.csv")
    srcm = "results/analysis_cpu/controls/mechanism_controls.csv"

    def mrow(data_prefix, stat):
        return next(x for x in mc if x["data"].startswith(data_prefix) and x["statistic"] == stat)

    within = [float(x["value"]) for x in mc if x["data"] == "full TailCat P3" and x["statistic"].startswith("within ")]
    put("ctMechWithinScaleMin", min(within), srcm, "num", 2)
    put("ctMechWithinScaleMax", max(within), srcm, "num", 2)
    for name, pre, stat in (("ctMechFEScale", "full TailCat P3", "Pearson of within-model demeaned values (model fixed effect)"),
                            ("ctMechStagesFE", "pilot_v2", "OLMo-2 stages: Pearson with category fixed effects"),
                            ("ctMechWithinPilotMean", "pilot_v2", "mean of the six within-model Spearman"),
                            ("ctMechPooledScale", "full TailCat P3 (277", "pooled Spearman")):
        x = mrow(pre, stat)
        put(name, x["value"], srcm, "num", 2)
        put(name + "Lo", x["ci_lo"], srcm, "num", 2)
        put(name + "Hi", x["ci_hi"], srcm, "num", 2)
        put(name + "CI", f"{float(x['ci_lo']):.2f} to {float(x['ci_hi']):.2f}", srcm, "str")


def split_figures():
    """paper/figures/figures.tex (written by mb5.figures, one block per figure) -> paper/tables/figure_<name>.tex,
    so the paper can place each generated block in its own section without copying it by hand."""
    import re

    src = Path("paper/figures/figures.tex")
    if not src.exists():
        return
    blocks = re.split(r"(?m)^% Figure \d+ \(", src.read_text(encoding="utf-8"))[1:]
    for b in blocks:
        name = b.split(")", 1)[0]
        body = b.split("\n", 1)[1].strip()
        Path(f"paper/tables/figure_{name}.tex").write_text("% GENERATED by mb5.paper_numbers from paper/figures/"
                                                          "figures.tex (mb5.figures)\n" + body + "\n", encoding="utf-8")


def robustness():
    """Recognition-robustness numbers (results/analysis_cpu/robustness, mb5.robustness_cpu)."""
    d = json.loads((R / "analysis_cpu/robustness/robustness.json").read_text())
    src = "results/analysis_cpu/robustness/robustness.json"
    runs = {"pilot_qwen": ("PilotQwen", "Qwen2.5-7B-Instruct", "Qwen2.5-7B"),
            "pilot_olmo": ("PilotOlmo", "OLMo-2-1124-7B-Instruct", "OLMo-2-1124-7B"),
            "full_qwen": ("FullQwen", "Qwen2.5-7B-Instruct", "Qwen2.5-7B"),
            "full_olmo": ("FullOlmo", "OLMo-2-1124-7B-Instruct", "OLMo-2-1124-7B")}
    for run, (tg, inst, base) in runs.items():
        x = d[run]
        put(f"rc{tg}PassN", len(x["passing_categories"]), src, "int")
        put(f"rc{tg}Cats", x["categories"], src, "int")
        for m, mt in ((inst, "Inst"), (base, "Base")):
            s = x["models"][m]["p0.90_T1"]
            put(f"rc{tg}{mt}Pass", s["v2_passing_categories"]["rbu_mean"], src)
            put(f"rc{tg}{mt}PassRec", s["v2_passing_categories"]["n_recognised"], src, "int")
            put(f"rcAll{tg}{mt}", s["all_categories"]["rbu_all"], src)
        if "shipped_hf" in x["models"][inst]:
            s = x["models"][inst]["shipped_hf"]
            put(f"rc{tg}InstShippedPass", s["v2_passing_categories"]["rbu_mean"], src)
            put(f"rcAll{tg}InstShipped", s["all_categories"]["rbu_all"], src)
    p2 = d["pilot_qwen_probe2"]["models"]
    put("rcProbeTwoQwenInst", p2["Qwen2.5-7B-Instruct"]["p0.90_T1"]["all_categories"]["rbu_mean"], src)
    put("rcProbeTwoQwenInstShipped", p2["Qwen2.5-7B-Instruct"]["shipped_hf"]["all_categories"]["rbu_mean"], src)
    put("rcProbeTwoQwenBase", p2["Qwen2.5-7B"]["p0.90_T1"]["all_categories"]["rbu_mean"], src)
    put("rcProbeTwoRecognised", p2["Qwen2.5-7B-Instruct"]["p0.90_T1"]["all_categories"]["n_recognised"], src, "int")
    # range of the yes/no AUROC over every main model (pilot Qwen pair, OLMo-2 stages, full TailCat)
    au = [float(r["auroc_yesno"]) for r in _csv(R / "G1/stages.csv")]
    au += [float(r["auroc_yesno"]) for r in _csv(R / "P3/g2/validity.csv")]
    put("rcAurocMin", min(au), "G1/stages.csv, P3/g2/validity.csv", "num", 2)
    put("rcAurocMax", max(au), "G1/stages.csv, P3/g2/validity.csv", "num", 2)
    rows = _csv(R / "analysis_cpu/robustness/robustness.csv")
    lines = [r"\begin{table*}[t]\centering\small", r"\begin{tabular}{lllrrrr}", r"\toprule",
             r"run & model & sampler & categories & recognised & RBU-mean & RBU-all \\ \midrule"]
    names = {"pilot_qwen": "pilot", "pilot_qwen_probe2": "pilot, second probe", "pilot_olmo": "pilot",
             "full_qwen": "full TailCat", "full_olmo": "full TailCat"}
    for r in rows:
        sub = "all" if r["subset"] == "all" else "AUROC $\\geq$ 0.85"
        nrec = f"{int(r['n_recognised']):,}".replace(",", "{,}")
        lines.append(f"{names[r['run']]} ({sub}) & {tex_escape(r['model'])} & {tex_escape(r['spec'])} & "
                     f"{r['n_categories']} & {nrec} & {pct(r['rbu_mean'])} & {pct(r['rbu_all'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Robust RBU on all categories and on the categories where the instruct model's yes/no probe "
              r"reaches AUROC 0.85 on its own (the same categories for both models of a pair), and under a second "
              r"probe wording with worked examples. RBU-mean averages over prompt wordings; RBU-all counts an "
              r"entity only if it is unreachable under every wording.}",
              r"\label{tab:robustness}\end{table*}"]
    TABLES["robustness"] = "\n".join(lines)


def extras():
    """Smaller numbers the text needs: calibration sample size, full-scale V1 totals, estimator split."""
    d = json.loads((R / "G0_5090/calibration/Qwen2.5-7B-Instruct/selfcheck.json").read_text(encoding="utf-8"))
    put("gzCalSamples", sum(r["n_samples"] for r in d["rows"]), "G0_5090 calibration selfcheck.json", "int")
    put("gzCalPerCategory", d["meta"]["args"]["n_samples"], "G0_5090 calibration selfcheck.json", "int")
    val = {r["model"]: r for r in _csv(R / "P3/g2/validity.csv")}
    n = sum(int(float(r["sample_check_samples"])) for r in val.values() if r.get("sample_check_samples"))
    v = sum(int(float(r["violations"])) for r in val.values() if r.get("sample_check_samples"))
    put("ptSamplesTotal", n, "results/P3/g2/validity.csv", "int")
    put("ptViolationsTotal", v, "results/P3/g2/validity.csv", "int")
    p = json.loads((R / "P3/metrics/estimators_Qwen2.5-7B-Instruct.json").read_text())["pooled"]["shipped_hf"]
    src = "results/P3/metrics/estimators_Qwen2.5-7B-Instruct.json"
    put("peQwenShippedTrunc", p["rbu_times_recognised"], src, "int")
    put("peQwenShippedChaoShare", p["chao1_over_reachable"], src)
    put("peQwenShippedShortfall", p["shortfall_vs_recognised"], src, "int")


def tex_escape(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


def pct(x) -> str:
    return "--" if x in (None, "") else f"{100 * float(x):.1f}"


TABLES: dict[str, str] = {}


def tables():
    # recognition validity per category, pilot (G0 attempt 2, a-priori probe)
    rows = []
    for m, tg in (("Qwen2.5-7B-Instruct", "instruct"), ("Qwen2.5-7B", "base")):
        for r in _csv(R / f"G0_5090/metrics/validity_{m}.csv"):
            rows.append((r["category"], tg, r))
    lines = [r"\begin{table}[h]\centering\small", r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"category & model & AUROC & 4-way acc. & recognised & foils passing \\ \midrule"]
    for cat, tg, r in rows:
        lines.append(f"{tex_escape(cat)} & {tg} & {float(r['auroc_yesno']):.3f} & {float(r['mc_accuracy']):.3f} & "
                     f"{r['n_recognised']} & {pct(r.get('foils_V_ge_tau'))} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Recognition validity per pilot category (Qwen2.5-7B pair, main probe). Foils passing: share "
              r"of foils at or above the category threshold.}",
              r"\end{table}"]
    TABLES["recognition"] = "\n".join(lines)

    # walk-through examples: first two per family
    out = []
    for run, fam in (("G0_5090", "Qwen2.5-7B"), ("G1", "OLMo-2-7B")):
        d = json.loads((R / f"{run}/examples/walkthrough_p0.90_T1.json").read_text(encoding="utf-8"))
        for ex in d["examples"][:2]:
            out.append(r"\paragraph{" + tex_escape(f"{ex['label']} ({ex['category']}, {fam})") + "}")
            out.append(r"\begin{tabular}{lll}\toprule model & token & mass ahead (G, $T=1$) / kept at $p=0.9$ \\ \midrule")
            for role in ("chat", "base"):
                for j, x in enumerate(ex[role]):
                    out.append(f"{role if j == 0 else ''} & \\texttt{{{tex_escape(repr(x['piece']))}}} & "
                               f"{x['G_T1']:.3f} / {'kept' if x['kept_by_G'] else 'cut'} \\\\")
            out.append(r"\bottomrule\end{tabular}")
    TABLES["walkthrough"] = "\n".join(out)

    # frontier, both instruct models
    lines = [r"\begin{table*}[t]\centering\small", r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"model & sampler & RBU & recovered & distinct members & invented & $\Delta$ invented (pts) \\ \midrule"]
    for m in ("Qwen2.5-7B-Instruct", "OLMo-2-1124-7B-Instruct"):
        f = json.loads((R / f"P5/{m}/frontier.json").read_text())
        for r in f["rows"]:
            lines.append(f"{tex_escape(m)} & {tex_escape(r['spec'])} & {pct(r['rbu_robust'])} & "
                         f"{pct(r['recovered_vs_reference'])} & {r['distinct_members_sampled']} & {pct(r['invented_rate'])} & "
                         f"{100 * r['invented_delta_vs_reference']:+.1f} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\caption{Mitigation frontier on the pilot (template 0, 1{,}000 samples per category per "
              r"sampler). Reference: shipped sampler (Qwen2.5), top-$p$ 0.9 (OLMo-2, which ships none). Invented: "
              r"automatic English-label matching (Section~\ref{sec:fix}).}",
              r"\label{tab:frontier}\end{table*}"]
    TABLES["frontier"] = "\n".join(lines)


def tables_p3():
    for kind, key in (("region", "region"), ("type", "type")):
        rows = _csv(R / f"P3/g2/by_{kind}.csv")
        models = sorted({r["model"] for r in rows})
        cats = sorted({r[key] for r in rows})
        idx = {(r["model"], r[key]): r for r in rows}
        lines = [r"\begin{table}[h]\centering\small", r"\begin{tabular}{l" + "r" * len(models) + "}", r"\toprule",
                 f"{kind} & " + " & ".join(tex_escape(m.replace("-Instruct", "-Inst")) for m in models) + r" \\ \midrule"]
        for c in cats:
            lines.append(tex_escape(str(c)) + " & " + " & ".join(pct(idx.get((m, c), {}).get("rbu_p0.90_T1")) for m in models) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}",
                  f"\\caption{{Robust RBU (\\%) at top-$p$ 0.9 by {kind}, full TailCat.}}",
                  r"\end{table}"]
        TABLES[f"by_{kind}"] = "\n".join(lines)


def tables_repro():
    """Reproducibility appendix: every main run's model snapshot, GPU, margin and time. The anonymous
    version (tables/repro.tex) leaves out our code commits, which could identify the repository; the
    camera-ready version (tables/repro_camera.tex) adds them."""
    rows = []
    for run in ("G0_5090", "G1", "P3_robust", "P3", "P3_scale"):
        for meta in sorted((R / run).glob("*/meta.json")):
            m = json.loads(meta.read_text())
            rows.append((run, m, m.get("margin", "--")))
    for f in sorted((R / "P5").glob("*/frontier.json")):
        m = json.loads(f.read_text())["meta"]
        rows.append(("P5", m, m["args"]["margin"]))
    total = sum(float(r["gpu_hours"]) for r in _csv("logs/gpu_ledger.csv"))
    put("reproGpuHours", total, "logs/gpu_ledger.csv", "num", 2)
    for name, with_code in (("repro", False), ("repro_camera", True)):
        lines = [r"\begin{table*}[t]\centering\small", r"\begin{tabular}{llllrr" + ("l" if with_code else "") + "}",
                 r"\toprule", r"run & model & snapshot & GPU & $\epsilon$ & minutes" + (" & code" if with_code else "")
                 + r" \\ \midrule"]
        for run, m, margin in rows:
            code = ((m.get("code") or {}).get("git_head") or "--")[:7]
            lines.append(f"{tex_escape(run)} & {tex_escape(m['model'].split('/')[-1])} & "
                         f"\\texttt{{{(m.get('commit') or '--')[:7]}}} & {tex_escape(m.get('gpu', '').replace('NVIDIA ', ''))} & "
                         f"{margin} & {m['seconds'] / 60:.1f}" + (f" & \\texttt{{{code}}}" if with_code else "") + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}",
                  r"\caption{Runs behind every number: model snapshot (Hugging Face commit), GPU, numeric margin and "
                  r"wall time" + (" and code commit" if with_code else "") + r". Data, outputs, metric tables and the "
                  r"decision log are in the supplementary material; total GPU time for the project \reproGpuHours{} "
                  r"hours.}", r"\label{tab:repro}\end{table*}"]
        TABLES[name] = "\n".join(lines)


def fit(body: str) -> str:
    """Scale every tabular to its float's width (column for table, page for table*)."""
    import re

    width = r"\textwidth" if "table*" in body else r"\columnwidth"
    return re.sub(r"(\\begin\{tabular\}.*?\\end\{tabular\})",
                  lambda m: r"\resizebox{" + width + "}{!}{%\n" + m.group(1) + "}", body, flags=re.S)


def write_tables():
    Path("paper/tables").mkdir(parents=True, exist_ok=True)
    for name in ("recognition", "walkthrough", "frontier", "by_region", "by_type", "repro", "repro_camera", "robustness"):
        body = TABLES.get(name, "\\missing{table " + name.replace("_", " ") + "}")
        body = fit(body)
        Path(f"paper/tables/{name}.tex").write_text("% GENERATED by mb5.paper_numbers\n" + body + "\n", encoding="utf-8")


def scale():
    for tag, tg in (("Qwen2.5-0.5B-Instruct", "Half"), ("Qwen2.5-1.5B-Instruct", "OneHalf"), ("Qwen2.5-3B-Instruct", "Three")):
        src = R / f"P3_scale/metrics/{tag}/rbu_{tag}.csv"
        put(f"scShipped{tg}", rbu_row(src, "shipped_hf")["rbu_mean"], str(src))
        put(f"scPnine{tg}", rbu_row(src, "p0.90_T1")["rbu_mean"], str(src))
        h = json.loads((R / f"P3_scale/metrics/{tag}/headline.json").read_text())["instruct"]["recognition"]
        put(f"scRecognised{tg}", h["n_recognised"], "headline.json", "int")
        put(f"scAuroc{tg}", h["auroc_yesno"], "headline.json", "num", 2)


def main():
    safe(scale, "scShippedHalf")
    safe(tables, "tblMissingPilot")
    safe(tables_p3, "tblMissingFull")
    safe(tables_repro, "reproGpuHours")
    write_tables()
    safe(g0, "gzQwenInstShipped")
    safe(g1, "goRbuOlmoBase")
    safe(robust, "rbCvQwenInst")
    safe(p5, "pfRecoveredQwen")
    safe(p3, "ptRbuQwenInst")
    safe(p4, "peQwenPnineChao")
    safe(walkthrough_intro, "cwInstKeptPnine")
    safe(robustness, "rcPilotQwenPassN")
    safe(extras, "gzCalSamples")
    safe(zt, "ztMedQQI")
    safe(region_controls, "rgGapAdjQwen")
    write_tables()  # again: robustness() adds a table
    split_figures()
    # Every number macro the paper uses must exist: undefined ones become visible markers. The prefix pd
    # marks results still pending (GPU runs, other analyses); they render as red markers until filled.
    import re

    used = set(re.findall(r"\\((?:gz|go|rb|pf|pt|pe|sc|cw|rc|zt|rg|ct|pd|repro)[A-Z][A-Za-z]*)",
                          Path("paper/main.tex").read_text(encoding="utf-8")))
    for name in sorted(used - set(MACROS)):
        MACROS[name] = f"\\missing{{{name}}}"
        MISSING.append(name)
    lines = ["% GENERATED by `python -m mb5.paper_numbers` from results/. Do not edit by hand.",
             "\\providecommand{\\missing}[1]{\\textcolor{red}{\\textbf{[#1]}}}"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(MACROS.items())]
    Path("paper").mkdir(exist_ok=True)
    Path("paper/numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path("paper/numbers_sources.json").write_text(json.dumps(SOURCES, indent=1), encoding="utf-8")
    print(f"[paper_numbers] {len(MACROS)} macros, {len(MISSING)} missing: {MISSING}")


if __name__ == "__main__":
    main()
