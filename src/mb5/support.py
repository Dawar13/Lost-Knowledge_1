"""Zero versus tiny (CPU only): how likely would a truncated-away name be without truncation?

A reviewer's objection: "truncation makes these names exactly zero, but without truncation they
would be practically never sampled anyway." This module answers it from stored reachability rows.

What is stored (reachability.score_items): for every surface form of an entity under a prompt,
logp_sum_T{T} = sum over the form's answer tokens of log softmax(z / T), where z are the model's
raw next-token logits under teacher forcing. No repetition penalty, no truncation, and no
end-of-answer token. So exp(logp_sum_T) is the probability that one untruncated sample at
temperature T BEGINS with exactly this token sequence (the canonical tokenisation of the form after
the prompt). It is stored for T = 1.0 and T = 0.7.

Per (entity, template):
  q_max  max over forms of exp(logp_sum_T). Main measure; conservative.
  q_any  sum over the prefix-minimal forms (forms whose token sequence has no other listed form's
         sequence as a prefix). Their "begins with" events are disjoint, so q_any is exactly the
         probability that a sample begins with some listed form (canonical tokenisations). It is an
         upper bound on the probability that the answer IS the entity (a sample can begin with a
         form and continue into a longer, different name).

The untruncated counterpart of a sampler S is its chain with every truncation removed: temperature
T_S only. For p0.90_T1 that is untruncated_T1 (T 1) and q is exact. For p0.90_T0.7 it is T 0.7,
exact. For Qwen's shipped_hf (repetition penalty 1.05, T 0.7, top-k 20, top-p 0.8) the counterpart
would keep the repetition penalty, which the stored values omit: shipped numbers use T 0.7 without
the penalty and are flagged approximate.

Populations, per sampler S and template: recognised entities (analysis.recognised_set) split into
U (robustly unreachable under S: every form cut even with the target raised by the margin; the RBU
numerator) and R (the rest; their count is the plateau |R|).

Truncated sampling. For e in R the truncated per-sample probability q_S(e) is not stored. Bounds:
  lower  q_lo(e) = max over forms surely kept (robust_reach) of q(f): a kept path's probability can
         only grow when the sampler renormalises over the kept set (exact for pure top-p; for
         shipped_hf approximate, see above).
  upper  (pure top-p samplers only) q_hi(e) = sum over prefix-minimal forms not robustly cut of
         q(f) / p^L(f): top-p keeps at least mass p at every step, so renormalising multiplies a
         path's probability by at most p^-L.
  and E_S[D(N)] <= |R| for every N: truncated sampling can never find more than the reachable set.

Expected distinct recognised entities after N samples per prompt (category x template):
  E[D(N)] = sum over prompts, sum over entities of 1 - (1 - q_e)^N.

Calibration: predicted member counts (n x q) against the stored P5 samples (pilot_v2, template 0,
1,000 per category) under untruncated_T1 (and, as bounds, p0.90_T1).

  python -m mb5.support --out results/analysis_cpu/support
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .analysis import form_reachable, recognised_set
from .common import code_version, write_json
from .metrics import sha256

N_GRID = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000,
          500_000, 1_000_000, 10_000_000, 100_000_000)
N_SEEN = (1_000, 10_000, 100_000, 1_000_000)
Q_LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9)
Q_THRESH = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
DEFAULT_SPECS = ("p0.90_T1", "p0.90_T0.7", "shipped_hf")
STORED_T = (1.0, 0.7)
N_STAR_MAX = 1e12

DEFAULT_RUNS = [
    ("pilot", "Qwen2.5-7B", "results/G0_5090/Qwen2.5-7B"),
    ("pilot", "Qwen2.5-7B-Instruct", "results/G0_5090/Qwen2.5-7B-Instruct"),
    ("pilot", "OLMo-2-1124-7B", "results/G1/OLMo-2-1124-7B"),
    ("pilot", "OLMo-2-1124-7B-SFT", "results/G1/OLMo-2-1124-7B-SFT"),
    ("pilot", "OLMo-2-1124-7B-DPO", "results/G1/OLMo-2-1124-7B-DPO"),
    ("pilot", "OLMo-2-1124-7B-Instruct", "results/G1/OLMo-2-1124-7B-Instruct"),
    ("scale", "Qwen2.5-7B", "results/P3/Qwen2.5-7B"),
    ("scale", "Qwen2.5-7B-Instruct", "results/P3/Qwen2.5-7B-Instruct"),
    ("scale", "OLMo-2-1124-7B", "results/P3/OLMo-2-1124-7B"),
    ("scale", "OLMo-2-1124-7B-Instruct", "results/P3/OLMo-2-1124-7B-Instruct"),
]
DEFAULT_CALIB = [
    ("Qwen2.5-7B-Instruct", "results/P5/Qwen2.5-7B-Instruct"),
    ("OLMo-2-1124-7B-Instruct", "results/P5/OLMo-2-1124-7B-Instruct"),
]


# ---------------------------------------------------------------- maths
def p_seen(q, n) -> np.ndarray:
    """Probability of at least one sight in n independent samples: 1 - (1 - q)^n, stable for tiny q."""
    q = np.clip(np.asarray(q, dtype=float), 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = -np.expm1(np.asarray(n, dtype=float) * np.log1p(-q))
    return np.where(q >= 1.0, 1.0, out)


def expected_distinct(qs, n) -> float:
    """E[D(n)] = sum_e 1 - (1 - q_e)^n."""
    return float(np.sum(p_seen(qs, n))) if len(qs) else 0.0


def prefix_minimal(seqs: list) -> list[int]:
    """Indices of the sequences with no other listed sequence as a prefix. Identical sequences
    count once (the first). The 'begins with' events of the returned sequences are disjoint."""
    first: dict[tuple, int] = {}
    for i, s in enumerate(seqs):
        first.setdefault(tuple(s), i)
    keys = list(first)
    out = [first[s] for s in keys if not any(len(o) < len(s) and s[: len(o)] == o for o in keys)]
    return sorted(out)


def q_entity(qs: list[float], seqs: list) -> tuple[float, float]:
    """(q_max, q_any) for one (entity, template) from its forms' probabilities and token ids."""
    if not qs:
        return 0.0, 0.0
    mins = prefix_minimal(seqs)
    return max(qs), min(1.0, sum(qs[i] for i in mins))


def q_trunc_bounds(qs: list[float], seqs: list, sure: list[bool], maybe: list[bool],
                   top_p: float | None) -> tuple[float, float | None]:
    """Bounds on the truncated per-sample probability of an entity (see the module docstring).
    sure: form kept even with the target lowered by the margin; maybe: form not robustly cut."""
    lo = max((q for q, s in zip(qs, sure) if s), default=0.0)
    if top_p is None:
        return lo, None
    idx = [i for i, m in enumerate(maybe) if m]
    mins = prefix_minimal([seqs[i] for i in idx])
    hi = sum(qs[idx[j]] / top_p ** len(seqs[idx[j]]) for j in mins)
    return lo, min(1.0, hi)


def n_star(qs_by_prompt: list[np.ndarray], target: float, n_max: float = N_STAR_MAX) -> float | None:
    """Smallest N (samples per prompt) with sum over prompts of E[D(N)] >= target; None if not by n_max."""
    f = lambda n: sum(expected_distinct(q, n) for q in qs_by_prompt)  # noqa: E731
    if f(n_max) < target:
        return None
    lo, hi = 0.0, math.log10(n_max)
    if f(1.0) >= target:
        return 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if f(10 ** mid) >= target:
            hi = mid
        else:
            lo = mid
    return 10 ** hi


def quantiles(x) -> dict:
    x = np.asarray(x, dtype=float)
    return {str(q): (float(np.quantile(x, q)) if len(x) else None) for q in Q_LEVELS}


# ---------------------------------------------------------------- per run (memory-lean: streamed, columns)
def stream_jsonl(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def spec_info(name: str, spec: dict) -> dict:
    T = float(spec.get("temperature", 1.0))
    pure_top_p = set(spec) <= {"top_p", "temperature"} and "top_p" in spec
    exact = not spec.get("repetition_penalty") or float(spec["repetition_penalty"]) == 1.0
    return {"name": name, "T": T, "top_p_bound": float(spec["top_p"]) if pure_top_p else None,
            "q_exact": exact and T in STORED_T, "stored_T": T in STORED_T, "spec": spec}


def load_groups(path: Path, names: list[str]) -> dict:
    """(entity key, template) -> compact forms (ids, logp_sum per stored T, is_label, sitelinks, maybe, sure);
    maybe/sure per spec: form not robustly cut / form kept even with the target lowered by the margin."""
    groups: dict[tuple, list] = defaultdict(list)
    for r in stream_jsonl(path):
        groups[(r["qid"] + "|" + r["category"], r["template"])].append((
            tuple(r["answer_ids"]), tuple(r[f"logp_sum_T{T}"] for T in STORED_T), bool(r.get("is_label")),
            r.get("sitelinks"), tuple(form_reachable(r, s, "robust") for s in names),
            tuple(form_reachable(r, s, "upper") for s in names)))
    return groups


def entity_columns(groups: dict, rec: dict, infos: list[dict]) -> dict[str, np.ndarray]:
    """One row per (entity, template), as columns: q at each stored T; per spec the robust entity
    verdict (unreachable iff every form is robustly cut) and the truncated-probability bounds."""
    cols: dict[str, list] = defaultdict(list)
    for (ek, t), forms in groups.items():
        seqs = [f[0] for f in forms]
        lab = next((f for f in forms if f[2]), forms[0])
        qid, cat = ek.split("|", 1)
        cols["qid"].append(qid)
        cols["category"].append(cat)
        cols["template"].append(t)
        cols["recognised"].append(bool(rec.get(ek)))
        cols["sitelinks"].append(float(lab[3]) if lab[3] is not None else float("nan"))
        cols["L_label"].append(len(lab[0]))
        cols["n_forms"].append(len(forms))
        qT = {}
        for ti, T in enumerate(STORED_T):
            qs = [math.exp(f[1][ti]) for f in forms]
            qT[T] = qs
            qm, qa = q_entity(qs, seqs)
            cols[f"q_max_T{T}"].append(qm)
            cols[f"q_any_T{T}"].append(qa)
        for si, inf in enumerate(infos):
            s = inf["name"]
            maybe = [f[4][si] for f in forms]
            sure = [f[5][si] for f in forms]
            cols[f"unreach__{s}"].append(not any(maybe))
            if inf["stored_T"]:
                lo, hi = q_trunc_bounds(qT[inf["T"]], seqs, sure, maybe, inf["top_p_bound"])
                cols[f"qlo__{s}"].append(lo)
                cols[f"qhi__{s}"].append(float("nan") if hi is None else hi)
    return {k: (np.array(v, dtype=object) if k in ("qid", "category") else np.array(v)) for k, v in cols.items()}


def summarise(E: dict[str, np.ndarray], inf: dict, templates: list[int]) -> dict:
    """Distribution, sight shares, deleted mass and curves for one model and one sampler."""
    s, T = inf["name"], inf["T"]
    res = {"spec": inf["spec"], "T_untruncated": T, "q_exact": inf["q_exact"],
           "top_p_bound": inf["top_p_bound"], "scopes": {}, "curves": {}}
    rec = E["recognised"].astype(bool)
    unr = E[f"unreach__{s}"].astype(bool)
    tmpl = E["template"]

    def pop_stats(m):
        st = {"n": int(m.sum())}
        for meas in ("q_max", "q_any"):
            q = E[f"{meas}_T{T}"][m].astype(float)
            st[meas] = {"quantiles": quantiles(q),
                        "samples_to_first_sight_quantiles": quantiles(1.0 / np.maximum(q, 1e-300)),
                        "mean_share_seen": {str(n): (float(np.mean(p_seen(q, n))) if len(q) else None)
                                            for n in N_SEEN},
                        "share_q_at_least": {f"{th:g}": (float(np.mean(q >= th)) if len(q) else None)
                                             for th in Q_THRESH}}
        return st

    for scope, tsel in (("template0", [0]), ("all_templates", templates)):
        pop = rec & np.isin(tmpl, tsel)
        U, R = pop & unr, pop & ~unr
        # untruncated probability mass, per prompt, of the recognised names the sampler deletes (q_any)
        mass = defaultdict(float)
        prompts = set(zip(E["category"][pop], tmpl[pop]))
        for c, t, q in zip(E["category"][U], tmpl[U], E[f"q_any_T{T}"][U]):
            mass[(c, t)] += float(q)
        mvals = np.array([mass[p] for p in prompts]) if prompts else np.array([])
        res["scopes"][scope] = {
            "n_recognised_pairs": int(pop.sum()), "n_unreachable_pairs": int(U.sum()),
            "rbu_pairs": float(U.sum() / pop.sum()) if pop.sum() else None,
            "unreachable": pop_stats(U), "reachable": pop_stats(R),
            "deleted_mass_per_prompt": {"n_prompts": len(prompts),
                                        "mean": float(mvals.mean()) if len(mvals) else None,
                                        "quantiles": quantiles(mvals)},
        }
    # curves per template: N samples per prompt (category), summed over categories
    for t in templates:
        pop = rec & (tmpl == t)
        cats = sorted(set(E["category"][pop]))
        byc = [pop & (E["category"] == c) for c in cats]
        qm = [E[f"q_max_T{T}"][m].astype(float) for m in byc]
        qa = [E[f"q_any_T{T}"][m].astype(float) for m in byc]
        ql = [E[f"qlo__{s}"][m & ~unr].astype(float) for m in byc]
        qh = [E[f"qhi__{s}"][m & ~unr].astype(float) for m in byc] if inf["top_p_bound"] is not None else None
        plateau = int((pop & ~unr).sum())
        rows = []
        for n in N_GRID:
            rows.append({"N": n, "E_untruncated_qmax": sum(expected_distinct(q, n) for q in qm),
                         "E_untruncated_qany": sum(expected_distinct(q, n) for q in qa),
                         "E_truncated_lo": sum(expected_distinct(q, n) for q in ql),
                         "E_truncated_hi": (min(plateau, sum(expected_distinct(q, n) for q in qh))
                                            if qh is not None else None),
                         "plateau_reachable": plateau, "n_recognised": int(pop.sum())})
        res["curves"][str(t)] = {
            "rows": rows,
            "n_star_qmax_exceeds_plateau": n_star(qm, plateau) if plateau else None,
            "n_star_qany_exceeds_plateau": n_star(qa, plateau) if plateau else None,
            "n_prompts": len(qm),
        }
    return res


RECOG_KEEP = ("qid", "category", "kind", "V", "mc_correct")


def run_one(scope: str, tag: str, mdir: Path, spec_names) -> tuple[dict, dict, list[Path]]:
    meta = json.loads((mdir / "meta.json").read_text())
    infos = [spec_info(s, meta["specs"][s]) for s in spec_names if s in meta["specs"]]
    groups = load_groups(mdir / "reach.jsonl.gz", [i["name"] for i in infos])
    recog = [{k: r[k] for k in RECOG_KEEP if k in r} for r in stream_jsonl(mdir / "recog.jsonl.gz")]
    rec, validity = recognised_set(recog)
    del recog
    templates = sorted({t for _, t in groups})
    E = entity_columns(groups, rec, infos)
    del groups
    out = {"scope": scope, "model": meta["model"], "tag": tag, "dir": mdir.as_posix(),
           "margin": meta.get("margin"), "n_recognised": validity["n_recognised"], "templates": templates,
           "specs": {}}
    for inf in infos:
        if inf["stored_T"]:
            out["specs"][inf["name"]] = summarise(E, inf, templates)
    return out, E, [mdir / "meta.json", mdir / "reach.jsonl.gz", mdir / "recog.jsonl.gz"]


# ---------------------------------------------------------------- calibration against P5 samples
CAL_BINS = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0, float("inf"))


def calibrate(tag: str, p5: Path, open_dir: Path | None = None) -> tuple[dict, list[dict], list[Path]]:
    meta = json.loads((p5 / "frontier.json").read_text())["meta"]
    t0 = meta["args"]["template"]
    keep = ("qid", "category", "form", "is_label", "answer_ids", "logp_sum_T1.0", "reach__p0.90_T1",
            "robust_unreach__p0.90_T1", "robust_reach__p0.90_T1")
    groups: dict[str, list] = defaultdict(list)
    for r in stream_jsonl(p5 / "reach.jsonl.gz"):  # the P5 pass scores one template and stores no template field
        if r.get("template", t0) == t0:
            groups[r["qid"] + "|" + r["category"]].append({k: r[k] for k in keep if k in r})
    n_cat: dict[tuple, Counter] = defaultdict(Counter)
    obs: dict[str, Counter] = defaultdict(Counter)
    for r in stream_jsonl(p5 / "samples_frontier.jsonl.gz"):
        n_cat[r["spec"]][r["category"]] += 1
        if r["class"] == "member":
            obs[r["spec"]][r["member_qid"] + "|" + r["category"]] += 1
    res = {"model": meta["model"], "template": t0, "n_members": len(groups), "specs": {}}
    bins_rows = []
    # untruncated_T1: exact prediction from q (T 1)
    for spec in ("untruncated_T1",):
        pred_any, pred_max, o, p_any_seen = [], [], [], []
        per_cat = defaultdict(lambda: {"pred_any": 0.0, "pred_max": 0.0, "obs": 0, "pred_distinct": 0.0,
                                       "obs_distinct": 0})
        surprises = []
        for ek, forms in groups.items():
            cat = ek.split("|", 1)[1]
            n = n_cat[spec][cat]
            qs = [math.exp(f["logp_sum_T1.0"]) for f in forms]
            qm, qa = q_entity(qs, [tuple(f["answer_ids"]) for f in forms])
            c = obs[spec].get(ek, 0)
            pred_any.append(n * qa)
            pred_max.append(n * qm)
            o.append(c)
            p_any_seen.append(float(p_seen(qa, n)))
            pc = per_cat[cat]
            pc["pred_any"] += n * qa
            pc["pred_max"] += n * qm
            pc["obs"] += c
            pc["pred_distinct"] += float(p_seen(qa, n))
            pc["obs_distinct"] += int(c > 0)
            if c > 0 and n * qa < 0.05:
                lab = next((f for f in forms if f.get("is_label")), forms[0])
                surprises.append({"entity": ek, "label": lab["form"], "observed": c, "predicted_any": n * qa})
        pred_any, pred_max, o = np.array(pred_any), np.array(pred_max), np.array(o)
        from scipy.stats import spearmanr
        rho = spearmanr(pred_any, o)
        pos = o > 0
        res["specs"][spec] = {
            "samples": int(sum(n_cat[spec].values())),
            "member_samples_observed": int(o.sum()), "member_samples_predicted_any": float(pred_any.sum()),
            "member_samples_predicted_max": float(pred_max.sum()),
            "distinct_members_observed": int(pos.sum()), "distinct_members_predicted_any": float(sum(p_any_seen)),
            "spearman_pred_any_vs_observed": {"rho": float(rho.statistic), "p": float(rho.pvalue)},
            "spearman_among_predicted_or_observed": None,
            "observed_with_predicted_below_0.05": {"n_entities": len(surprises),
                                                   "samples": int(sum(x["observed"] for x in surprises)),
                                                   "examples": sorted(surprises, key=lambda x: -x["observed"])[:15]},
            "per_category": dict(per_cat),
        }
        keep = (pred_any >= 0.01) | pos
        if keep.sum() > 2:
            r2 = spearmanr(pred_any[keep], o[keep])
            res["specs"][spec]["spearman_among_predicted_or_observed"] = {
                "rho": float(r2.statistic), "p": float(r2.pvalue), "n": int(keep.sum())}
        for lo, hi in zip(CAL_BINS[:-1], CAL_BINS[1:]):
            m = (pred_any >= lo) & (pred_any < hi)
            bins_rows.append({"model": tag, "spec": spec, "pred_count_bin": f"[{lo:g}, {hi:g})", "n_members": int(m.sum()),
                              "mean_predicted_any": float(pred_any[m].mean()) if m.any() else None,
                              "mean_observed": float(o[m].mean()) if m.any() else None,
                              "share_seen_predicted": float(np.mean(np.array(p_any_seen)[m])) if m.any() else None,
                              "share_seen_observed": float(np.mean(pos[m])) if m.any() else None})
    # p0.90_T1: truncated bounds and the V1 consistency check
    spec = "p0.90_T1"
    if spec in meta["specs"] and n_cat.get(spec):
        lo_tot, hi_tot, obs_tot, obs_unreach = 0.0, 0.0, 0, 0
        for ek, forms in groups.items():
            cat = ek.split("|", 1)[1]
            n = n_cat[spec][cat]
            qs = [math.exp(f["logp_sum_T1.0"]) for f in forms]
            seqs = [tuple(f["answer_ids"]) for f in forms]
            maybe = [form_reachable(f, spec, "robust") for f in forms]
            sure = [form_reachable(f, spec, "upper") for f in forms]
            lo, hi = q_trunc_bounds(qs, seqs, sure, maybe, 0.9)
            c = obs[spec].get(ek, 0)
            lo_tot += n * lo
            hi_tot += n * hi
            obs_tot += c
            if c and not any(maybe):
                obs_unreach += c
        res["specs"][spec] = {"samples": int(sum(n_cat[spec].values())), "member_samples_observed": obs_tot,
                              "member_samples_predicted_lo": lo_tot, "member_samples_predicted_hi": hi_tot,
                              "observed_samples_of_robustly_unreachable_entities": obs_unreach}
    inputs = [p5 / "frontier.json", p5 / "reach.jsonl.gz", p5 / "samples_frontier.jsonl.gz"]
    if open_dir is not None:  # the P5 scoring pass should reproduce the open-prompt run's log-probabilities
        lp = {(r["qid"], r["category"], r["form"]): r["logp_sum_T1.0"] for r in stream_jsonl(open_dir / "reach.jsonl.gz")
              if r.get("template", t0) == t0}
        diffs = [abs(r["logp_sum_T1.0"] - lp[(r["qid"], r["category"], r["form"])]) for rs in groups.values()
                 for r in rs if (r["qid"], r["category"], r["form"]) in lp]
        res["logp_agreement_with_open_run"] = {"open_run": open_dir.as_posix(), "n_forms": len(diffs),
                                               "max_abs_diff": float(max(diffs)) if diffs else None,
                                               "median_abs_diff": float(np.median(diffs)) if diffs else None}
        inputs.append(open_dir / "reach.jsonl.gz")
    return res, bins_rows, inputs


# ---------------------------------------------------------------- output
def write_csv(path: Path, rows: list[dict], gz: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=keys, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    data = buf.getvalue().encode("utf-8")
    if gz:
        with gzip.GzipFile(path, "wb", mtime=0) as f:
            f.write(data)
    else:
        path.write_bytes(data)


def write_columns_csv_gz(path: Path, E: dict[str, np.ndarray]):
    """Entity table, streamed row by row (no full copy in memory)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(E)
    with gzip.GzipFile(path, "wb", mtime=0) as gz, io.TextIOWrapper(gz, encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(keys)
        for i in range(len(E["qid"])):
            row = []
            for k in keys:
                v = E[k][i]
                if isinstance(v, (bool, np.bool_)):
                    row.append(int(v))
                elif isinstance(v, (float, np.floating)):
                    row.append("" if math.isnan(v) else repr(float(v)))
                else:
                    row.append(v)
            w.writerow(row)


def flat_rows(summ: dict) -> tuple[list[dict], list[dict]]:
    qrows, crows = [], []
    for sname, sd in summ["specs"].items():
        for scope, sc in sd["scopes"].items():
            for popname in ("unreachable", "reachable"):
                p = sc[popname]
                for meas in ("q_max", "q_any"):
                    if meas not in p:
                        continue
                    d = p[meas]
                    row = {"scope": summ["scope"], "model": summ["tag"], "spec": sname, "q_exact": sd["q_exact"],
                           "templates": scope, "population": popname, "measure": meas, "n": p["n"]}
                    row |= {f"q_{k}": v for k, v in d["quantiles"].items()}
                    row |= {f"first_sight_{k}": v for k, v in d["samples_to_first_sight_quantiles"].items()}
                    row |= {f"seen_N{k}": v for k, v in d["mean_share_seen"].items()}
                    row |= {f"q_ge_{k}": v for k, v in d["share_q_at_least"].items()}
                    qrows.append(row)
        for t, cv in sd["curves"].items():
            for r in cv["rows"]:
                crows.append({"scope": summ["scope"], "model": summ["tag"], "spec": sname, "q_exact": sd["q_exact"],
                              "template": int(t)} | r)
    return qrows, crows


def headline(summaries: list[dict], calib: dict) -> dict:
    """The numbers a paper paragraph would cite (p0.90_T1 and shipped_hf, unreachable recognised names)."""
    h = {}
    for su in summaries:
        for sname in ("p0.90_T1", "shipped_hf"):
            sd = su["specs"].get(sname)
            if not sd:
                continue
            key = f"{su['scope']}|{su['tag']}|{sname}"
            d = {"q_exact": sd["q_exact"]}
            for scope in ("template0", "all_templates"):
                sc = sd["scopes"][scope]
                u = sc["unreachable"]
                d[scope] = {
                    "n_unreachable": u["n"], "rbu_pairs": sc["rbu_pairs"],
                    "median_q_max": u["q_max"]["quantiles"]["0.5"], "median_q_any": u["q_any"]["quantiles"]["0.5"],
                    "q_max_quartiles": [u["q_max"]["quantiles"]["0.25"], u["q_max"]["quantiles"]["0.75"]],
                    "median_samples_to_first_sight_q_max": u["q_max"]["samples_to_first_sight_quantiles"]["0.5"],
                    "share_seen_q_max": u["q_max"]["mean_share_seen"],
                    "share_q_max_at_least": u["q_max"]["share_q_at_least"],
                    "deleted_mass_per_prompt_mean": sc["deleted_mass_per_prompt"]["mean"],
                    "reachable_median_q_max": sc["reachable"]["q_max"]["quantiles"]["0.5"],
                }
            cv = sd["curves"].get("0")
            if cv:
                at = {r["N"]: r for r in cv["rows"]}
                d["curve_t0"] = {str(n): {k: at[n][k] for k in ("E_untruncated_qmax", "E_untruncated_qany",
                                                                "E_truncated_lo", "E_truncated_hi")}
                                 for n in (1_000, 10_000, 100_000, 1_000_000)}
                d["curve_t0"]["plateau_reachable"] = cv["rows"][0]["plateau_reachable"]
                d["curve_t0"]["n_recognised"] = cv["rows"][0]["n_recognised"]
                d["curve_t0"]["n_prompts"] = cv["n_prompts"]
                d["curve_t0"]["n_star_qmax_exceeds_plateau"] = cv["n_star_qmax_exceeds_plateau"]
                d["curve_t0"]["n_star_qany_exceeds_plateau"] = cv["n_star_qany_exceeds_plateau"]
            h[key] = d
    h["calibration"] = {k: {s: {kk: vv for kk, vv in v.items() if kk not in ("per_category",)}
                            for s, v in c["specs"].items()} | {"logp_agreement_with_open_run":
                                                                 c.get("logp_agreement_with_open_run")}
                        for k, c in calib.items()}
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/analysis_cpu/support")
    ap.add_argument("--run", action="append", default=[], help="scope:tag=dir (default: pilot and scale runs)")
    ap.add_argument("--calib", action="append", default=[], help="tag=results/P5/<tag> (default: both P5 models)")
    ap.add_argument("--specs", default=",".join(DEFAULT_SPECS))
    args = ap.parse_args()
    out = Path(args.out)
    runs = [(a.split(":", 1)[0], a.split(":", 1)[1].split("=", 1)[0], a.split("=", 1)[1]) for a in args.run] \
        or DEFAULT_RUNS
    cal = [tuple(a.split("=", 1)) for a in args.calib] or DEFAULT_CALIB
    specs = [s for s in args.specs.split(",") if s]
    inputs: list[Path] = []
    summaries, qrows, crows = [], [], []
    for scope, tag, d in runs:
        t = time.time()
        summ, E, ins = run_one(scope, tag, Path(d), specs)
        inputs += ins
        summaries.append(summ)
        a, b = flat_rows(summ)
        qrows += a
        crows += b
        write_columns_csv_gz(out / "entities" / f"{scope}_{tag}.csv.gz", E)
        print(f"[support] {scope} {tag}: {len(E['qid'])} entity-template pairs in {time.time() - t:.0f}s", flush=True)
        del E
    calib, bins_rows = {}, []
    open_of = {tag: Path(d) for scope, tag, d in runs if scope == "pilot"}
    for tag, d in cal:
        c, b, ins = calibrate(tag, Path(d), open_of.get(tag))
        calib[tag] = c
        bins_rows += b
        inputs += ins
        print(f"[support] calibration {tag} done", flush=True)
    write_json(out / "summary.json", {"definitions": __doc__, "runs": summaries})
    write_json(out / "calibration.json", calib)
    write_csv(out / "quantiles.csv", qrows)
    write_csv(out / "curves.csv", crows)
    write_csv(out / "calibration_bins.csv", bins_rows)
    write_json(out / "headline.json", headline(summaries, calib))
    write_json(out / "manifest.json", {
        "code": code_version(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": "python -m mb5.support " + " ".join(f"--run {s}:{t}={d}" for s, t, d in runs)
                   + " " + " ".join(f"--calib {t}={d}" for t, d in cal) + f" --specs {','.join(specs)}",
        "inputs": {str(p).replace("\\", "/"): sha256(p) for p in dict.fromkeys(inputs)},
        "outputs": sorted(str(p.relative_to(out)).replace("\\", "/") for p in out.rglob("*")
                          if p.is_file() and p.name != "manifest.json"),
    })
    print(f"[support] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
