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
    v = float(value) if kind in ("pct", "num", "p") else value
    if kind == "pct":
        s = f"{100 * v:.{digits}f}\\%"
    elif kind == "num":
        s = f"{v:,.{digits}f}".replace(",", "{,}")
    elif kind == "int":
        s = f"{int(value):,}".replace(",", "{,}")
    elif kind == "p":
        s = "10^{%d}" % int(f"{v:.0e}".split("e")[1]) if v < 1e-3 else f"{v:.3f}"
        s = f"${'<' if v < 1e-3 else '='}{s}$" if v < 1e-3 else s
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
    pr = {(r["family"], r["from"], r["to"], r["spec"]): r for r in _csv(R / "G1/paired.csv")}
    r = pr[("olmo2", "base", "instruct", "p0.90_T1")]
    put("goPairedInstOnly", r["to_only_unreachable"], "G1/paired.csv", "int")
    put("goPairedBaseOnly", r["from_only_unreachable"], "G1/paired.csv", "int")
    put("goRatioOlmo", r["ratio"], "G1/paired.csv", "num", 2)
    put("goRatioQwen", pr[("qwen2.5", "base", "instruct", "p0.90_T1")]["ratio"], "G1/paired.csv", "num", 2)
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
        put(f"pfInvDelta{tg}", 100 * f["g3"]["invented_delta"], src, "num", 1)
        for spec, t in (("reference", "Ref"), ("relax_first", "Relax"), ("untruncated_T1", "Untr"),
                        ("p0.99_T1", "Pnn"), ("k100_T1", "Khund"), ("eps3e-4_T1", "Eps")):
            r = rows[spec]
            put(f"pf{tg}{t}Rbu", r["rbu_robust"], src)
            put(f"pf{tg}{t}Distinct", r["distinct_members_sampled"], src, "int")
            put(f"pf{tg}{t}Inv", r["invented_rate"], src)
    for r in _csv(R / "P5/audit/invented_rate_sensitivity.csv"):
        tg = "Qwen" if r["model"].startswith("Qwen") else "Olmo"
        if r["spec"] == "relax_first":
            put(f"pfLooseDelta{tg}", 100 * float(r["loose_delta_vs_reference"]), "invented_rate_sensitivity.csv", "num", 1)
        if r["spec"] == "reference":
            put(f"pfLooseRef{tg}", r["invented_rate_loose"], "invented_rate_sensitivity.csv")


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
              r"\caption{Recognition validity on the pilot (Qwen2.5-7B pair, a-priori probe). Source: results/G0\_5090/metrics/validity\_*.csv.}",
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
              r"sampler). Reference: shipped sampler (Qwen2.5), top-$p$ 0.9 (OLMo-2, which ships none). Source: results/P5.}",
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
                  f"\\caption{{Robust RBU (\\%) at top-$p$ 0.9 by {kind}, full TailCat. Source: results/P3/g2/by\\_{kind}.csv.}}",
                  r"\end{table}"]
        TABLES[f"by_{kind}"] = "\n".join(lines)


def write_tables():
    Path("paper/tables").mkdir(parents=True, exist_ok=True)
    for name in ("recognition", "walkthrough", "frontier", "by_region", "by_type"):
        body = TABLES.get(name, f"\\missing{{table {name}}}")
        Path(f"paper/tables/{name}.tex").write_text("% GENERATED by mb5.paper_numbers\n" + body + "\n", encoding="utf-8")


def main():
    safe(tables, "tblMissingPilot")
    safe(tables_p3, "tblMissingFull")
    write_tables()
    safe(g0, "gzQwenInstShipped")
    safe(g1, "goRbuOlmoBase")
    safe(robust, "rbCvQwenInst")
    safe(p5, "pfRecoveredQwen")
    safe(p3, "ptRbuQwenInst")
    safe(p4, "peQwenPnineChao")
    lines = ["% GENERATED by `python -m mb5.paper_numbers` from results/. Do not edit by hand.",
             "\\providecommand{\\missing}[1]{\\textcolor{red}{\\textbf{[#1]}}}"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(MACROS.items())]
    Path("paper").mkdir(exist_ok=True)
    Path("paper/numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path("paper/numbers_sources.json").write_text(json.dumps(SOURCES, indent=1), encoding="utf-8")
    print(f"[paper_numbers] {len(MACROS)} macros, {len(MISSING)} missing: {MISSING}")


if __name__ == "__main__":
    main()
