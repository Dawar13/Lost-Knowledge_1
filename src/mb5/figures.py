"""Paper figures, drawn from stored results only (CPU; CLAUDE.md rule 5: no number is typed by hand).

  PYTHONPATH=src python -m mb5.figures --out paper/figures            # all figures, PDF + PNG preview
  PYTHONPATH=src python -m mb5.figures --out paper/figures --only 1,6

Writes <out>/fig<N>_<name>.pdf (vector, fonts embedded as TrueType), a PNG preview of each,
<out>/figures.tex (one figure environment per figure; captions carry numbers computed here) and
<out>/sources.json (per figure: every input file with its sha256, and the code version).

Style: print paper (ACL, Times). Model identity is one hue per family (Qwen blue, OLMo orange; aqua is
reserved for a third family); base vs post-trained is open vs filled markers and dashed vs solid
lines, so every figure also reads in greyscale. Sampler families in the frontier use violet, aqua and
dark yellow (validated all-pairs with the dataviz validator) plus marker shapes and direct labels.

Figure 6 has a hook for gate G3b: if results/P6/<model>/e15_summary.json exists (mb5.p6_analysis), the
self-checked wide samplers are drawn as filled points with an arrow from their unfiltered position.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import matplotlib
import matplotlib.transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

from .common import ROOT, code_version  # noqa: E402

# ---------------------------------------------------------------- style
COL_W, FULL_W = 3.15, 6.5  # ACL column and text width, inches
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#9a9993", "#e6e5e1", "#ffffff"
FAMILY = {"qwen2.5": {"color": "#2a78d6", "marker": "o", "name": "Qwen2.5-7B"},
          "olmo2": {"color": "#eb6834", "marker": "s", "name": "OLMo-2-7B"}}
THIRD_FAMILY_COLOR = "#1baf7a"  # reserved (Llama-3.1 or Granite) so existing colours never move
SAMPLER_FAMILY = {"top-p": {"color": "#6250d6", "marker": "o"},
                  "top-k": {"color": "#1baf7a", "marker": "s"},
                  "min-p": {"color": "#c98500", "marker": "^"}}
CUT_FILL = "#d9d8d3"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.4, "axes.axisbelow": True, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "lines.linewidth": 1.2, "lines.markersize": 3.5, "legend.frameon": False, "legend.handlelength": 1.8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 220,
})
MONO = {"family": "monospace", "fontname": "Courier New"}


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}\\%"


# ---------------------------------------------------------------- inputs
class Inputs:
    """Records every file a figure reads (path relative to the repo, sha256)."""

    def __init__(self):
        self.files: dict[str, str] = {}

    def path(self, rel: str) -> Path:
        p = ROOT / rel
        if rel not in self.files:
            self.files[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
        return p

    def csv(self, rel: str) -> list[dict]:
        with open(self.path(rel), newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def json(self, rel: str):
        return json.loads(self.path(rel).read_text(encoding="utf-8"))

    def jsonl(self, rel: str, keep=None):
        out = []
        with gzip.open(self.path(rel), "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if keep is None or keep(r):
                    out.append(r)
        return out


def ci(s: str) -> tuple[float, float]:
    lo, hi = json.loads(s)
    return float(lo), float(hi)


def style_axes(ax, ygrid=True, xgrid=False):
    ax.grid(axis="y", visible=ygrid)
    ax.grid(axis="x", visible=xgrid)
    ax.tick_params(which="both", direction="out")


def place_labels(ax, items, avoid_points=None, avoid_lines=None, fontsize=6.4):
    """Greedy direct labelling: for each (x, y, text, kwargs) try offsets around the point and keep the
    first whose box overlaps no placed label, no marker, no drawn line and stays inside the axes."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ax_box = ax.get_window_extent(r)
    pts = [ax.transData.transform(p) for p in (avoid_points or [])]
    point_boxes = [matplotlib.transforms.Bbox.from_extents(px - 5, py - 5, px + 5, py + 5) for px, py in pts]
    for line in avoid_lines or []:  # densify each polyline in display space
        d = [ax.transData.transform(p) for p in line]
        for (x0, y0), (x1, y1) in zip(d[:-1], d[1:]):
            for t in [i / 12 for i in range(13)]:
                px, py = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
                point_boxes.append(matplotlib.transforms.Bbox.from_extents(px - 1.5, py - 1.5, px + 1.5, py + 1.5))
    placed = []
    cands = [(4, -2, "left", "top"), (4, 2, "left", "bottom"), (-4, 2, "right", "bottom"), (-4, -2, "right", "top"),
             (0, 6, "center", "bottom"), (0, -6, "center", "top"), (8, 0, "left", "center"), (-8, 0, "right", "center"),
             (0, 11, "center", "bottom"), (0, -11, "center", "top"), (10, 8, "left", "bottom"), (-10, 8, "right", "bottom"),
             (10, -8, "left", "top"), (-10, -8, "right", "top")]
    for x, y, text, kw in items:
        best, best_cost = None, None
        for dx, dy, ha, va in cands:
            a = ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy), ha=ha, va=va,
                            fontsize=kw.get("fontsize", fontsize), color=kw.get("color", INK2))
            bb = a.get_window_extent(r)
            inside = bb.x0 >= ax_box.x0 and bb.x1 <= ax_box.x1 and bb.y0 >= ax_box.y0 and bb.y1 <= ax_box.y1
            cost = (0 if inside else 100) + 10 * sum(bb.overlaps(o) for o in placed) + sum(bb.overlaps(o) for o in point_boxes)
            if best_cost is None or cost < best_cost:
                if best is not None:
                    best.remove()
                best, best_cost = a, cost
            else:
                a.remove()
            if cost == 0:
                break
        placed.append(best.get_window_extent(r))


def panel_title(ax, letter: str, text: str):
    ax.set_title(f"({letter}) {text}", loc="left", pad=4)


# ---------------------------------------------------------------- figure 1: mechanism walk-through
def fig_mechanism(inp: Inputs):
    cat, tmpl = "us_state_capitals", 0
    rows = {}
    for tag in ("Qwen2.5-7B-Instruct", "Qwen2.5-7B"):
        d = inp.json(f"results/G0_5090/{tag}/first_token.json")
        rows[tag] = next(r for r in d["rows"] if r["category"] == cat and r["template"] == tmpl)
    prompts = inp.json("results/G0_a3_5090/Qwen2.5-7B-Instruct/prompts.json")["generation"]
    question = prompts[f"{cat}|template{tmpl}"].split("<|im_start|>user\n")[1].split("<|im_end|>")[0]
    question = question.split(" Reply with")[0]

    # Montpelier's second step: reach rows of the same model and prompt (label forms, template 0)
    reach = inp.jsonl("results/G0_a3_5090/Qwen2.5-7B-Instruct/reach.jsonl.gz",
                      keep=lambda r: r["category"] == cat and r["template"] == tmpl and r["is_label"]
                      and r["form"] in ("Montpelier", "Montgomery"))
    mp = next(r for r in reach if r["form"] == "Montpelier")
    mg = next(r for r in reach if r["form"] == "Montgomery")
    assert mp["answer_ids"][0] == mg["answer_ids"][0], "Montpelier and Montgomery should share their first piece"
    assert mp["rank_pos"][1] == 1 and mg["rank_pos"][1] == 0, "expected 'pel' second and Montgomery's piece first"
    mass_above_pel = mp["G_T1.0_pos"][1]  # rank 1: the only piece above it is Montgomery's second piece
    try:  # decode piece ids with the cached tokenizer file (no torch import; offline)
        from huggingface_hub import try_to_load_from_cache
        from tokenizers import Tokenizer

        tok_file = try_to_load_from_cache("Qwen/Qwen2.5-7B-Instruct", "tokenizer.json")
        tok = Tokenizer.from_file(tok_file)
        inp.files["hf:Qwen/Qwen2.5-7B-Instruct/tokenizer.json"] = hashlib.sha256(Path(tok_file).read_bytes()).hexdigest()
        pieces_mp = [tok.decode([i]) for i in mp["answer_ids"]]
        top2 = tok.decode([mg["answer_ids"][1]])
    except Exception:  # tokenizer not cached: fall back to ids
        pieces_mp = [str(i) for i in mp["answer_ids"]]
        top2 = f"id {mg['answer_ids'][1]}"

    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.05), gridspec_kw={"width_ratios": [1, 1, 0.95], "wspace": 0.55})
    xmax = max(t["p"] for r in rows.values() for t in r["top_tokens"]) * 100
    xmax = math.ceil(xmax / 10) * 10
    info = {}
    for ax, tag, letter, label in ((axes[0], "Qwen2.5-7B-Instruct", "a", "chat model"),
                                   (axes[1], "Qwen2.5-7B", "b", "base model")):
        r = rows[tag]
        toks = r["top_tokens"]
        kept = int(r["kept_top_p_0.9"])
        ys = list(range(len(toks)))
        for y, t in zip(ys, toks):
            is_kept = y < kept
            ax.barh(-y, 100 * t["p"], height=0.72, color=FAMILY["qwen2.5"]["color"] if is_kept else CUT_FILL,
                    hatch=None if is_kept else "//////", edgecolor=SURFACE if is_kept else MUTED, linewidth=0.0 if is_kept else 0.3)
        ax.set_yticks([-y for y in ys])
        ax.set_yticklabels([repr(t["token"].strip())[1:-1] for t in toks], **MONO, fontsize=7)
        ax.set_xlim(0, xmax)
        ax.set_xlabel("probability of first piece (%)")
        style_axes(ax, ygrid=False, xgrid=True)
        cum_kept = sum(t["p"] for t in toks[:kept]) if kept <= len(toks) else None
        shown = sum(t["p"] for t in toks)
        if kept < len(toks):
            ax.axhline(-(kept - 0.5), color=INK, linewidth=0.8)
            ax.text(xmax * 0.97, -(kept - 0.5) + 0.25, f"top-p 0.9 keeps {kept} pieces\n(together {100 * cum_kept:.1f}%)",
                    ha="right", va="bottom", fontsize=7, color=INK)
            ax.text(xmax * 0.97, -(kept - 0.5) - 0.35, "cut: can never be written", ha="right", va="top",
                    fontsize=7, color=INK2)
        else:
            ax.text(xmax * 0.97, -(len(toks) - 1), f"top-p 0.9 keeps {kept} pieces;\nthe {len(toks)} shown hold {100 * shown:.1f}%",
                    ha="right", va="center", fontsize=7, color=INK)
        panel_title(ax, letter, f"{label}: first piece")
        info[tag] = {"kept": kept, "cum_kept": cum_kept, "shown": shown, "top": toks[0]["token"].strip(),
                     "top_p": toks[0]["p"]}

    # (c) after the shared first piece: one horizontal bar of the next-piece distribution
    ax = axes[2]
    ax.barh(0, 100 * mass_above_pel, height=0.5, color=FAMILY["qwen2.5"]["color"], edgecolor=SURFACE, linewidth=0)
    ax.barh(0, 100 * (1 - mass_above_pel), left=100 * mass_above_pel, height=0.5, color=CUT_FILL, hatch="//////",
            edgecolor=MUTED, linewidth=0.3)
    for p, yo in ((0.90, 0.40), (0.95, 0.62)):
        ax.plot([100 * p, 100 * p], [-0.34, yo - 0.02], color=INK, linewidth=0.8)
        ax.text(100 * p + (-1 if p < 0.93 else 1), yo, f"top-p {p:.2f}", ha="right" if p < 0.93 else "left",
                va="bottom", fontsize=7, color=INK)
    ax.text(4, 0, f"'{top2.strip()}' (Montgomery)\n{100 * mass_above_pel:.1f}%", ha="left", va="center",
            fontsize=7, color=SURFACE)
    ax.annotate(f"'{pieces_mp[1].strip()}' (Montpelier) and every\nother piece: {100 * (1 - mass_above_pel):.1f}% together",
                xy=(100 * (1 + mass_above_pel) / 2, -0.26), xytext=(97, -0.95), fontsize=7, color=INK, ha="right",
                va="center", arrowprops={"arrowstyle": "-", "color": INK2, "linewidth": 0.5})
    ax.set_xlim(0, 100)
    ax.set_ylim(-1.35, 0.95)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("probability of next piece (%)")
    style_axes(ax, ygrid=False, xgrid=False)
    panel_title(ax, "c", f"chat model, after '{pieces_mp[0].strip()}'")
    ax.title.set_fontfamily("serif")

    chat, base = info["Qwen2.5-7B-Instruct"], info["Qwen2.5-7B"]
    caption = (f"How a name becomes unreachable. Prompt: ``{question}'' "
               f"(a) The chat model's ten most likely first pieces. Top-p 0.9 keeps the {chat['kept']} pieces above "
               f"the line ({100 * chat['cum_kept']:.1f}\\% of the probability) and deletes the rest, so no capital "
               f"starting with a deleted piece can be written. "
               f"(b) The base model spreads its first step: its top piece has {pct(base['top_p'])} and top-p 0.9 "
               f"keeps {base['kept']} pieces. "
               f"(c) A name can also be cut later. After the shared piece `{pieces_mp[0].strip()}', the chat model "
               f"puts {pct(mass_above_pel)} on `{top2.strip()}' (Montgomery), so `{pieces_mp[1].strip()}' "
               f"(Montpelier) is deleted even at top-p 0.95. Hatched: deleted pieces.")
    return fig, caption, "full"


# ---------------------------------------------------------------- figure 2: RBU against truncation strength
CURVE_MODELS = [("qwen2.5", "instruct", "results/P3/metrics/qwen2.5/rbu_curves_Qwen2.5-7B-Instruct.csv"),
                ("qwen2.5", "base", "results/P3/metrics/qwen2.5/rbu_curves_Qwen2.5-7B.csv"),
                ("olmo2", "instruct", "results/P3/metrics/olmo2/rbu_curves_OLMo-2-1124-7B-Instruct.csv"),
                ("olmo2", "base", "results/P3/metrics/olmo2/rbu_curves_OLMo-2-1124-7B.csv")]


def series_style(fam: str, stage: str) -> dict:
    f = FAMILY[fam]
    post = stage != "base"
    return {"color": f["color"], "marker": f["marker"], "linestyle": "-" if post else (0, (3, 1.6)),
            "markerfacecolor": f["color"] if post else SURFACE, "markeredgecolor": f["color"],
            "markeredgewidth": 0.8}


def stage_label(fam: str, stage: str) -> str:
    return f"{FAMILY[fam]['name']} {'base' if stage == 'base' else 'instruct'}"


def fig_curves(inp: Inputs):
    panels = [("top_p_T1", "top-p", lambda v: 1 - v, True), ("top_k", "top-k", lambda v: v, False),
              ("min_p_T1", "min-p", lambda v: v, True)]
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.0), sharey=True, gridspec_kw={"wspace": 0.12})
    data = {}
    for fam, stage, rel in CURVE_MODELS:
        data[(fam, stage)] = inp.csv(rel)
    handles = []
    for ax, (family, title, tx, invert) in zip(axes, panels):
        for (fam, stage), rows in data.items():
            rr = [r for r in rows if r["family"] == family]
            xs = [tx(float(r["value"])) for r in rr]
            ys = [100 * float(r["rbu_mean"]) for r in rr]
            lo = [100 * ci(r["rbu_mean_ci"])[0] for r in rr]
            hi = [100 * ci(r["rbu_mean_ci"])[1] for r in rr]
            st = series_style(fam, stage)
            ax.fill_between(xs, lo, hi, color=st["color"], alpha=0.13, linewidth=0)
            h, = ax.plot(xs, ys, label=stage_label(fam, stage), markersize=2.8, **st)
            if ax is axes[0]:
                handles.append(h)
        ax.set_xscale("log")
        vals = sorted({float(r["value"]) for rows in data.values() for r in rows if r["family"] == family})
        if family == "top_p_T1":
            ticks = [v for v in vals if v in (0.5, 0.9, 0.99, 0.999)]
            ax.xaxis.set_major_locator(FixedLocator([1 - v for v in ticks]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{1 - x:g}"))
        elif family == "top_k":
            ax.xaxis.set_major_locator(FixedLocator([v for v in vals if v in (1, 10, 100, 1000)]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        else:
            ax.xaxis.set_major_locator(FixedLocator([v for v in vals if v in (0.001, 0.01, 0.1)]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
        if invert:
            ax.invert_xaxis()
        ax.set_ylim(0, 102)
        ax.set_xlabel({"top_p_T1": "top-p (T = 1)", "top_k": "top-k (T = 1)", "min_p_T1": "min-p (T = 1)"}[family])
        style_axes(ax)
        ax.set_title(title, loc="left", pad=3)
    axes[0].set_ylabel("recognised but unreachable (%)")
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06), columnspacing=1.4)
    q = data[("qwen2.5", "instruct")]
    qb = data[("qwen2.5", "base")]
    at = lambda rows, fam, v: float(next(r["rbu_mean"] for r in rows if r["family"] == fam and float(r["value"]) == v))  # noqa: E731
    caption = ("Share of recognised entities that are unreachable (RBU, exact verdict) as each truncation is "
               "loosened, full TailCat (277 categories), mean over three wordings with 95\\% cluster-bootstrap "
               "bands. Left to right within each panel the truncation gets weaker. Solid, filled: post-trained "
               "(instruct) models; dashed, open: their base models. At top-p 0.9 the Qwen instruct model leaves "
               f"{pct(at(q, 'top_p_T1', 0.9))} unreachable against {pct(at(qb, 'top_p_T1', 0.9))} for its base model.")
    return fig, caption, "full"


# ---------------------------------------------------------------- figure 3: OLMo-2 stages
def fig_stages(inp: Inputs):
    rows = inp.csv("results/G1/stages.csv")
    order = ["base", "sft", "dpo", "instruct"]
    names = {"base": "base", "sft": "SFT", "dpo": "DPO", "instruct": "instruct"}
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 1.85), gridspec_kw={"wspace": 0.42})
    metrics = [("first_token_kept_top_p_0.9_median", "first-step shortlist size\n(median pieces kept, top-p 0.9)", True),
               ("rbu_p0.90_T1", "recognised but unreachable\nat top-p 0.9 (%)", False),
               ("fail_first_token_p0.90_T1", "of those, lost at the\nfirst piece (%)", False)]
    handles = {}
    for ax, (col, ylabel, logy) in zip(axes, metrics):
        for fam in ("olmo2", "qwen2.5"):
            rr = {r["stage"]: r for r in rows if r["family"] == fam}
            st = [s for s in order if s in rr]
            xs = [order.index(s) + (0.12 if fam == "qwen2.5" else 0.0) for s in st]
            scale = 1 if logy else 100
            ys = [scale * float(rr[s][col]) for s in st]
            f = FAMILY[fam]
            ls = "-" if fam == "olmo2" else "none"
            h, = ax.plot(xs, ys, color=f["color"], marker=f["marker"], linestyle=ls, markersize=4,
                         markerfacecolor=f["color"], markeredgecolor=SURFACE, markeredgewidth=0.6,
                         label=FAMILY[fam]["name"] + (" (four stages)" if fam == "olmo2" else " (base and instruct only)"))
            handles[fam] = h
            if col == "rbu_p0.90_T1":
                lo = [100 * ci(rr[s]["rbu_p0.90_T1_ci"])[0] for s in st]
                hi = [100 * ci(rr[s]["rbu_p0.90_T1_ci"])[1] for s in st]
                ax.vlines(xs, lo, hi, color=f["color"], linewidth=0.8)
            if fam == "olmo2" and col != "rbu_p0.90_T1":
                for x, y in zip(xs, ys):
                    first = x == xs[0]
                    ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 6) if first else (-5, 4),
                                ha="center" if first else "right", fontsize=6.5, color=INK2)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([names[s] for s in order])
        ax.set_xlim(-0.35, len(order) - 0.65)
        if logy:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(FixedLocator([1, 3, 10, 30, 100, 300]))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.set_ylim(2, 300)
        else:
            ax.set_ylim(0, 105)
        ax.set_ylabel(ylabel)
        style_axes(ax)
    fig.legend(handles=[handles["olmo2"], handles["qwen2.5"]], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08))
    o = {r["stage"]: r for r in rows if r["family"] == "olmo2"}
    caption = ("Where unreachability comes from: OLMo-2-7B after each post-training stage (pilot, 11 categories; "
               "SFT: supervised fine-tuning; DPO: preference tuning), with the Qwen2.5-7B base and instruct pair "
               "for comparison. SFT alone shrinks the first-step shortlist from "
               f"{float(o['base']['first_token_kept_top_p_0.9_median']):.0f} to "
               f"{float(o['sft']['first_token_kept_top_p_0.9_median']):.0f} pieces. From base to instruct, the share "
               f"of recognised entities that are unreachable at top-p 0.9 rises from {pct(float(o['base']['rbu_p0.90_T1']))} "
               f"to {pct(float(o['instruct']['rbu_p0.90_T1']))}, and the share of those lost at the first piece from "
               f"{pct(float(o['base']['fail_first_token_p0.90_T1']), 0)} to "
               f"{pct(float(o['instruct']['fail_first_token_p0.90_T1']), 0)}. Bars: 95\\% cluster-bootstrap intervals.")
    return fig, caption, "full"


# ---------------------------------------------------------------- figure 4: who loses at scale
POP_MODELS = [("qwen2.5", "instruct", "results/P3/metrics/qwen2.5/popularity_Qwen2.5-7B-Instruct.csv"),
              ("qwen2.5", "base", "results/P3/metrics/qwen2.5/popularity_Qwen2.5-7B.csv"),
              ("olmo2", "instruct", "results/P3/metrics/olmo2/popularity_OLMo-2-1124-7B-Instruct.csv"),
              ("olmo2", "base", "results/P3/metrics/olmo2/popularity_OLMo-2-1124-7B.csv")]


def bin_labels(edges: list[float]) -> list[str]:
    e = [int(round(x)) for x in edges]
    labs = [f"$\\leq${e[0]}"]
    for a, b in zip(e[:-1], e[1:]):
        labs.append(f"{b}" if b == a + 1 else f"{a + 1}-{b}")
    labs.append(f"$\\geq${e[-1] + 1}")
    return labs


def fig_popularity(inp: Inputs):
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.0), sharex=True, gridspec_kw={"wspace": 0.55})
    handles = []
    edges = None
    for fam, stage, rel in POP_MODELS:
        rows = sorted(inp.csv(rel), key=lambda r: int(r["sitelink_quintile"]))
        e = json.loads(rows[0]["quintile_edges"])
        assert edges is None or e == edges, "all models must share the sitelink bins"
        edges = e
        xs = [int(r["sitelink_quintile"]) for r in rows]
        st = series_style(fam, stage)
        h, = axes[0].plot(xs, [100 * float(r["rbu_robust__p0.90_T1"]) for r in rows], label=stage_label(fam, stage), **st)
        axes[1].plot(xs, [100 * float(r["recognition_rate"]) for r in rows], **st)
        handles.append(h)
    for ax, yl in zip(axes, ("unreachable at top-p 0.9 (%)", "recognised (%)")):
        ax.set_xticks(range(len(edges) + 1))
        ax.set_xticklabels(bin_labels(edges), fontsize=6.8)
        ax.set_xlabel("Wikipedia sitelinks")
        ax.set_ylabel(yl)
        ax.set_ylim(0, 100)
        style_axes(ax)
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.53, 1.13), columnspacing=1.0,
               handlelength=2.2)
    q = sorted(inp.csv(POP_MODELS[0][2]), key=lambda r: int(r["sitelink_quintile"]))
    caption = ("Who loses, on the full TailCat (33,437 members). Entities are binned by Wikipedia sitelinks "
               "(popularity quintiles; ties make the bins unequal). Left: share of recognised entities that are "
               "unreachable at top-p 0.9 (robust verdict). Right: share recognised. For the Qwen instruct model, "
               f"unreachability falls from {pct(float(q[0]['rbu_robust__p0.90_T1']))} in the least linked bin to "
               f"{pct(float(q[-1]['rbu_robust__p0.90_T1']))} in the most linked bin.")
    return fig, caption, "column"


# ---------------------------------------------------------------- figure 5: sample-based estimators
def fig_estimators(inp: Inputs):
    cases = [("Qwen2.5-7B-Instruct", "shipped_hf", "Qwen2.5 instruct, shipped"),
             ("Qwen2.5-7B-Instruct", "p0.90_T1", "Qwen2.5 instruct, top-p 0.9"),
             ("OLMo-2-1124-7B-Instruct", "p0.90_T1", "OLMo-2 instruct, top-p 0.9")]
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.0), sharey=True, gridspec_kw={"wspace": 0.1})
    info = []
    for ax, (tag, spec, title) in zip(axes, cases):
        est = inp.json(f"results/P3/metrics/estimators_{tag}.json")["pooled"][spec]
        rar = inp.csv(f"results/P3/metrics/rarefaction_{tag}.csv")
        ns = sorted({int(r["n"]) for r in rar if r["spec"] == spec})
        tot = [sum(int(r["distinct_members"]) for r in rar if r["spec"] == spec and int(r["n"]) == n) for n in ns]
        fam = "qwen2.5" if tag.startswith("Qwen") else "olmo2"
        col = FAMILY[fam]["color"]
        ax.plot(ns, tot, color=col, marker=FAMILY[fam]["marker"], markersize=3, label="distinct members seen")
        ax.plot([ns[-1]], [est["chao1"]], marker="D", color=INK, markersize=3.5, linestyle="none", label="Chao1 at 1,000")
        for val, lab, yoff in ((est["reachable"], "reachable set", 1.18), (est["recognised"], "recognised", 1.18)):
            ax.axhline(val, color=INK if lab == "recognised" else INK2, linewidth=0.8)
            ax.text(ns[0], val * yoff, f"{lab}: {val:,}", fontsize=7, color=INK, va="bottom", ha="left")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(50, 60000)
        ax.set_xlabel("samples per category")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}"))
        style_axes(ax)
        ax.set_title(title, loc="left", pad=3)
        info.append((title, est))
    axes[0].set_ylabel("entities (all categories)")
    from matplotlib.lines import Line2D

    h1 = [Line2D([], [], color=INK2, linewidth=1.2, label="distinct member entities seen (Qwen blue, OLMo orange)"),
          Line2D([], [], color=INK, marker="D", markersize=3.5, linestyle="none", label="Chao1 estimate at 1,000")]
    fig.legend(handles=h1, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.07))
    t, e = info[0]
    caption = ("Sample-based knowledge estimates see the reachable set, not what the model recognises (full "
               "TailCat, template 0, 1,000 samples per category). Curves: distinct member entities seen as samples "
               "grow; diamond: the Chao1 richness estimate from all samples; lines: the reachable set and the "
               f"recognised set from the exact test. With the shipped sampler, Qwen yields {e['S_obs']:,} distinct "
               f"members and Chao1 estimates {e['chao1']:,.0f}, against {e['reachable']:,} reachable and "
               f"{e['recognised']:,} recognised.")
    return fig, caption, "full"


# ---------------------------------------------------------------- figure 6: mitigation frontier
def sampler_family(spec: str) -> str | None:
    if spec.startswith("p0") and spec.endswith("_T1"):
        return "top-p"
    if spec.startswith("k") and spec.endswith("_T1"):
        return "top-k"
    if spec.startswith("minp"):
        return "min-p"
    return None


def short_label(spec: str) -> str:
    if spec.startswith("p0"):
        return spec[1:].split("_")[0]
    if spec.startswith("k"):
        return spec[1:].split("_")[0]
    if spec.startswith("minp"):
        return spec[4:].split("_")[0]
    return spec


def fig_frontier(inp: Inputs, p6_root: str = "results/P6"):
    models = [("Qwen2.5-7B-Instruct", "Qwen2.5-7B instruct (reference: shipped sampler)"),
              ("OLMo-2-1124-7B-Instruct", "OLMo-2-7B instruct (reference: top-p 0.9)")]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.35), gridspec_kw={"wspace": 0.18})
    g3b_drawn = False
    notes = {}
    for ax, (tag, title) in zip(axes, models):
        rows = {r["spec"]: r for r in inp.csv(f"results/P5/{tag}/frontier.csv")}
        xy = {s: (100 * float(r["invented_rate"]), int(r["distinct_members_sampled"])) for s, r in rows.items()}
        labels, lines = [], []
        for fam, sty in SAMPLER_FAMILY.items():
            ss = [s for s in rows if sampler_family(s) == fam]
            ss.sort(key=lambda s: xy[s][1])
            ax.plot([xy[s][0] for s in ss], [xy[s][1] for s in ss], color=sty["color"], marker=sty["marker"],
                    markersize=3.6, markeredgecolor=SURFACE, markeredgewidth=0.5, linewidth=1.0, label=fam)
            labels += [(*xy[s], short_label(s), {}) for s in ss if not (s == "p0.90_T1" and xy[s] == xy.get("reference"))]
            lines.append([xy[s] for s in ss])
        specials = [("untruncated_T1", "*", "no truncation"), ("reference", "X", "reference" if tag.startswith("Qwen") else "reference (top-p 0.90)"),
                    ("relax_first", "D", "first-piece relaxation"), ("eps3e-4_T1", "P", "epsilon 3e-4")]
        for s, mk, lab in specials:
            if s not in xy:
                continue
            ax.plot(*xy[s], marker=mk, color=INK, markersize=5 if mk == "*" else 4, linestyle="none",
                    markeredgecolor=INK if mk in "*X" else SURFACE, markeredgewidth=0.5)
            labels.insert(0, (*xy[s], lab, {"color": INK, "fontsize": 6.6}))
        # gate G3b hook: self-checked wide samplers, if mb5.p6_analysis has run
        f15 = ROOT / p6_root / tag / "e15_summary.json"
        if f15.exists():
            try:
                e15 = inp.json(f"{p6_root}/{tag}/e15_summary.json")
                rr = e15["quantiles"]["0.95"]["rows"]
                for s in ("untruncated_T1", "k100_T1"):
                    if s in rr and s in xy:
                        fx = 100 * rr[s]["filtered"]["invented_rate"]
                        fy = rr[s]["filtered"]["distinct_members"]
                        ax.annotate("", xy=(fx, fy), xytext=xy[s],
                                    arrowprops={"arrowstyle": "->", "color": INK, "linewidth": 0.7})
                        ax.plot(fx, fy, marker="o", color=INK, markersize=4.5, linestyle="none")
                        xy[f"{s}+self-check"] = (fx, fy)
                        labels.insert(0, (fx, fy, f"{'no truncation' if s.startswith('untr') else 'top-k 100'} + self-check",
                                          {"color": INK, "fontsize": 6.6}))
                        g3b_drawn = True
            except (KeyError, TypeError) as e:  # layout of e15_summary.json changed: draw without G3b, say so
                print(f"  fig6: G3b hook skipped for {tag} ({type(e).__name__}: {e}); update fig_frontier", flush=True)
        xs_all = [v[0] for v in xy.values()]
        span = max(xs_all) - min(xs_all)
        ax.set_xlim(min(xs_all) - 0.12 * span, max(xs_all) + 0.42 * span)
        ax.set_ylim(0, max(v[1] for v in xy.values()) * 1.12)
        ref_x = xy["reference"][0]
        ax.axvspan(ref_x, ref_x + 2, color=GRID, alpha=0.7, linewidth=0, zorder=0)
        ax.text(ref_x + 1, 2, "+2 points", ha="center", va="bottom", fontsize=6.3, color=INK2)
        ax.set_title(title, loc="left", pad=3)
        ax.set_xlabel("invented answers (% of non-empty answers)")
        style_axes(ax, xgrid=True)
        place_labels(ax, labels, avoid_points=list(xy.values()), avoid_lines=lines)
        notes[tag] = {"untr": xy["untruncated_T1"], "ref": xy["reference"]}
    axes[0].set_ylabel("distinct real entities sampled\n(11 categories $\\times$ 1,000 samples)")
    hs, ls = axes[0].get_legend_handles_labels()
    fig.legend(hs, ls, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06), title=None)
    q = notes["Qwen2.5-7B-Instruct"]
    caption = ("The mitigation frontier (pilot, 11 categories, 1,000 samples per category and sampler). Each point "
               "is a sampler: distinct real member entities it produced against the share of answers that match no "
               "real member (``invented''; includes spelling variants and real names missing from Wikidata, see the "
               "hand audit). Lines join settings of one sampler family, labelled by their parameter. Shaded: within "
               "2 points of the reference, the band gate G3 allowed. For Qwen, removing truncation raises distinct "
               f"entities from {q['ref'][1]} to {q['untr'][1]} at {q['untr'][0] - q['ref'][0]:.1f} more points of "
               "invented answers.")
    if g3b_drawn:
        caption += " Filled black points: the wide samplers after the self-check filter (gate G3b)."
    return fig, caption, "full"


# ---------------------------------------------------------------- driver
FIGURES = {1: ("mechanism", fig_mechanism), 2: ("rbu_curves", fig_curves), 3: ("stages", fig_stages),
           4: ("popularity", fig_popularity), 5: ("estimators", fig_estimators), 6: ("frontier", fig_frontier)}


def tex_block(n: int, name: str, caption: str, width: str) -> str:
    env = "figure*" if width == "full" else "figure"
    gw = "\\textwidth" if width == "full" else "\\columnwidth"
    return (f"% Figure {n} ({name}): generated by mb5.figures; inputs and hashes in sources.json\n"
            f"\\begin{{{env}}}[t]\n  \\centering\n  \\includegraphics[width={gw}]{{figures/fig{n}_{name}.pdf}}\n"
            f"  \\caption{{{caption}}}\n  \\label{{fig:{name}}}\n\\end{{{env}}}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--only", default="", help="comma-separated figure numbers")
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args()
    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    only = {int(x) for x in args.only.split(",") if x}
    src_path = out / "sources.json"
    sources = json.loads(src_path.read_text(encoding="utf-8")) if src_path.exists() else {}
    tex_path = out / "figures.tex"
    blocks: dict[int, str] = {}
    if tex_path.exists():  # keep blocks of figures not regenerated now
        for chunk in tex_path.read_text(encoding="utf-8").split("% Figure ")[1:]:
            blocks[int(chunk.split(" ", 1)[0])] = "% Figure " + chunk.rstrip() + "\n"
    cv = code_version()
    for n, (name, fn) in FIGURES.items():
        if only and n not in only:
            continue
        inp = Inputs()
        fig, caption, width = fn(inp)
        stem = out / f"fig{n}_{name}"
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
        if not args.no_png:
            fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        blocks[n] = tex_block(n, name, caption, width)
        sources[f"fig{n}_{name}"] = {"inputs": inp.files, "code": cv, "width": width}
        print(f"fig{n}_{name}: {len(inp.files)} inputs", flush=True)
    header = ("% Generated by `PYTHONPATH=src python -m mb5.figures`; do not edit by hand.\n"
              "% Every number in a caption is computed from the inputs listed in sources.json.\n\n")
    tex_path.write_text(header + "\n".join(blocks[k] for k in sorted(blocks)), encoding="utf-8", newline="\n")
    src_path.write_text(json.dumps(sources, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
