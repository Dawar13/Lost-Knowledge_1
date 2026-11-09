"""P6 GPU steps for the experiments pre-registered on 2026-11-09 (notes/decisions.md):

  e13  ask-directly control: every cued entity's surface forms scored under its cue prompt
       ("{question} Reply with only the name.", mb5.cues) with the same samplers and margin as
       the open-prompt run.
  e14  list control: "List ten {plural}, one per line, names only." with the assistant turn
       pre-filled with the reference sampler's most frequent member answers ("1. A\\n2. B ... 6."),
       every other member scored as the next item.
  e15  self-check scores: the model's yes/no V (a-priori probe) for every distinct non-member
       answer in the stored P5 samples of the wide samplers and the reference.

  python -m mb5.p6 --model Qwen/Qwen2.5-7B-Instruct --data data/tailcat/pilot_v2 \
      --cues data/tailcat/pilot_v2_cues/cues.jsonl --open results/G0_5090/Qwen2.5-7B-Instruct \
      --p5 results/P5/Qwen2.5-7B-Instruct --out results/P6 --steps e13,e14,e15

Outputs in <out>/<model_tag>/: e13_reach.jsonl.gz, e13_prompts.json, e14_reach.jsonl.gz,
e14_prompts.json (with the pre-fill), e15_scores.jsonl.gz, and <step>_meta.json. A step whose
output exists is skipped (resumable). Measures and gate G3b: mb5.p6_analysis (CPU).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from .common import (append_gpu_ledger, estimate_need_gib, is_chat_model, load_model, model_tag, peak_gib, read_jsonl,
                     require_gpu_memory, run_meta, seed_all, write_json, write_jsonl)
from .cues import render_cue_prompt
from .prompts import BASE_GEN_SHOTS, answer_prefix, encode_pair, render
from .reachability import score_items
from .recognition import yesno_scores
from .sample_check import norm
from .samplers import GRID, shipped_specs

LIST_TEMPLATE = "List ten {plural}, one per line, names only."
LIST_N = 5
WIDE_SPECS = ("untruncated_T1", "k100_T1")
REFERENCE = "reference"  # the P5 name of each model's reference sampler (Qwen shipped_hf, OLMo p0.90_T1)


# ---------------------------------------------------------------- E13
def cue_items(tok, chat: bool, cues: list[dict], ents: dict[str, dict], aliases: bool = True):
    """Items for score_items (every surface form of every cued entity) and the exact prompts."""
    pre = answer_prefix(chat)
    items, prompts = [], {}
    for c in cues:
        key = c["qid"] + "|" + c["category"]
        e = ents.get(key)
        if e is None:
            continue
        p = render_cue_prompt(tok, c["question"], chat)
        prompts[key] = p
        forms = [e["label"]] + (list(e.get("aliases", [])) if aliases else [])
        for fi, form in enumerate(forms):
            p_ids, a_ids, ok = encode_pair(tok, p, pre + form, chat)
            if a_ids:
                items.append({"qid": e["qid"], "category": e["category"], "form": form, "is_label": fi == 0,
                              "sitelinks": e["sitelinks"], "boundary_ok": ok, "cue_qid": c["cue_qid"],
                              "question": c["question"], "prompt_ids": p_ids, "answer_ids": a_ids})
    return items, prompts


# ---------------------------------------------------------------- E14
def _first_line(text: str) -> str:
    return text.strip().split("\n")[0].strip()


def list_prefill(samples: list[dict], spec: str = REFERENCE, n: int = LIST_N, source: str = "member") -> dict:
    """category -> up to n entries {key, qid, count, text}: the answers sampled most often under
    `spec` (ties by key). source="member" is the pre-registered rule (member answers only);
    "member_or_valid" also admits real names from the uncapped Wikidata list (class "valid").
    The text is the model's own most frequent spelling (first line, stripped)."""
    counts: dict[str, Counter] = defaultdict(Counter)
    texts: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for r in samples:
        if r["spec"] != spec:
            continue
        if r["class"] == "member":
            key = r["member_qid"]
        elif source == "member_or_valid" and r["class"] == "valid":
            key = "valid:" + norm(r["text"])
        else:
            continue
        counts[r["category"]][key] += 1
        texts[r["category"]][key][_first_line(r["text"])] += 1
    out = {}
    for cat, c in counts.items():
        top = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
        out[cat] = [{"key": k, "qid": None if k.startswith("valid:") else k, "count": v,
                     "text": sorted(texts[cat][k].items(), key=lambda kv: (-kv[1], kv[0]))[0][0]} for k, v in top]
    return out


def list_prompt(tok, chat: bool, plural: str, prefill: list[dict]) -> str:
    """Prompt text up to and including the next item's number ("6."); the answer is scored as
    " " + name, so the tokens are the ones the model would emit after "6."."""
    user = LIST_TEMPLATE.format(plural=plural)
    head = render(tok, user, chat, "" if chat else BASE_GEN_SHOTS)
    if not chat:
        head += " "  # base frame: "Answer: 1. ..."
    body = "".join(f"{i + 1}. {p['text']}\n" for i, p in enumerate(prefill))
    return f"{head}{body}{len(prefill) + 1}."


def list_items(tok, chat: bool, cats: dict, ents: list[dict], prefill: dict, aliases: bool = True):
    items, prompts = [], {}
    excluded = {cat: {p["qid"] for p in pf if p["qid"]} for cat, pf in prefill.items()}
    for cid, c in cats.items():
        prompts[cid] = list_prompt(tok, chat, c["plural"], prefill.get(cid, []))
    for e in ents:
        if e["qid"] in excluded.get(e["category"], set()):
            continue
        forms = [e["label"]] + (list(e.get("aliases", [])) if aliases else [])
        for fi, form in enumerate(forms):
            p_ids, a_ids, ok = encode_pair(tok, prompts[e["category"]], " " + form, chat)
            if a_ids:
                items.append({"qid": e["qid"], "category": e["category"], "form": form, "is_label": fi == 0,
                              "sitelinks": e["sitelinks"], "boundary_ok": ok, "prompt_ids": p_ids,
                              "answer_ids": a_ids})
    return items, prompts, {k: sorted(v) for k, v in excluded.items()}


# ---------------------------------------------------------------- E15
def distinct_unmatched(samples: list[dict], specs=(*WIDE_SPECS, REFERENCE)) -> list[dict]:
    """Every distinct (category, normalised answer) among non-empty, non-member answers of the
    given specs, with the most frequent first-line spelling as the text to ask about."""
    texts: dict[tuple, Counter] = defaultdict(Counter)
    n_by_spec: dict[tuple, Counter] = defaultdict(Counter)
    for r in samples:
        if r["spec"] not in specs or r["class"] in ("member", "empty"):
            continue
        k = (r["category"], norm(r["text"]))
        if not k[1]:
            continue
        texts[k][_first_line(r["text"])] += 1
        n_by_spec[k][r["spec"]] += 1
    out = []
    for (cat, key) in sorted(texts):
        text = sorted(texts[(cat, key)].items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        out.append({"category": cat, "key": key, "text": text, "n_by_spec": dict(n_by_spec[(cat, key)])})
    return out


# ---------------------------------------------------------------- main
def _specs_for(model, chat: bool, open_meta: dict | None) -> dict:
    specs = {"p0.90_T1": dict(GRID["p0.90_T1"])}
    sh = shipped_specs(model.generation_config).get("shipped_hf") if chat else None
    if sh:
        specs["shipped_hf"] = sh
        if open_meta and open_meta.get("specs", {}).get("shipped_hf") not in (None, sh):
            raise SystemExit(f"shipped_hf differs from the open-prompt run: {sh} vs {open_meta['specs']['shipped_hf']}")
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--cues", required=True)
    ap.add_argument("--open", required=True, help="the open-prompt run dir of the same model (meta.json, recog)")
    ap.add_argument("--p5", required=True, help="results/P5/<model_tag> (samples_frontier.jsonl.gz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", default="e13,e14,e15")
    ap.add_argument("--margin", type=float, default=None, help="default: the open-prompt run's margin")
    ap.add_argument("--prefill-source", default="member", choices=["member", "member_or_valid"],
                    help="member is the pre-registered rule; member_or_valid is a flagged alternative")
    ap.add_argument("--prefill-spec", default=REFERENCE,
                    help="P5 sampler whose samples give the pre-fill; 'reference' is the pre-registered rule")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    tag = model_tag(args.model)
    out = Path(args.out) / tag
    steps = [s for s in args.steps.split(",") if s]
    # E14 variants other than the pre-registered one are written under their own names
    suffix = (("" if args.prefill_source == "member" else "_" + args.prefill_source)
              + ("" if args.prefill_spec == REFERENCE else "_" + args.prefill_spec))
    main_out = {"e13": out / "e13_reach.jsonl.gz", "e14": out / f"e14{suffix}_reach.jsonl.gz",
                "e15": out / "e15_scores.jsonl.gz"}
    todo = [s for s in steps if args.force or not main_out[s].exists()]
    for s in steps:
        if s not in todo:
            print(f"== skip (exists): {main_out[s]}")
    if not todo:
        return
    open_meta = json.loads((Path(args.open) / "meta.json").read_text())
    margin = open_meta["margin"] if args.margin is None else args.margin
    if args.margin is not None and abs(args.margin - open_meta["margin"]) > 1e-9:
        raise SystemExit(f"--margin {args.margin} differs from the open-prompt run's {open_meta['margin']}")

    seed_all(args.seed)
    if args.device == "cuda":
        require_gpu_memory(estimate_need_gib(args.model, args.batch_size))
    t_all = time.time()
    chat = is_chat_model(args.model)
    model, tok = load_model(args.model, device=args.device)
    pad_id = tok.pad_token_id
    specs = _specs_for(model, chat, open_meta)
    print(f"[p6] {tag} specs {specs} margin {margin}", flush=True)

    data = Path(args.data)
    cats = {c["id"]: c for c in json.loads((data / "categories.json").read_text())}
    ents_list = read_jsonl(data / "entities.jsonl")
    ents = {e["qid"] + "|" + e["category"]: e for e in ents_list}
    samples = read_jsonl(Path(args.p5) / "samples_frontier.jsonl.gz") if {"e14", "e15"} & set(todo) else []

    for step in todo:
        t0 = time.time()
        if step == "e13":
            cues = [c for c in read_jsonl(Path(args.cues)) if c["category"] in cats]
            items, prompts = cue_items(tok, chat, cues, ents)
            print(f"[e13] {len(items)} items for {len(prompts)} cued entities", flush=True)
            rows = score_items(model, items, specs, device=args.device, batch_size=args.batch_size, pad_id=pad_id,
                               margin=margin, dstar=False, keep_positions=True)
            write_jsonl(main_out[step], rows)
            write_json(out / "e13_prompts.json", {"cue": prompts, "cues_file": args.cues})
        elif step == "e14":
            prefill = list_prefill(samples, args.prefill_spec, LIST_N, args.prefill_source)
            items, prompts, excluded = list_items(tok, chat, cats, ents_list, prefill)
            print(f"[e14] {len(items)} items; pre-fill sizes "
                  f"{ {c: len(prefill.get(c, [])) for c in cats} }", flush=True)
            rows = score_items(model, items, specs, device=args.device, batch_size=args.batch_size, pad_id=pad_id,
                               margin=margin, dstar=False, keep_positions=True)
            write_jsonl(main_out[step], rows)
            write_json(out / f"e14{suffix}_prompts.json", {"list": prompts, "prefill": prefill, "excluded": excluded,
                                                           "prefill_source": args.prefill_source,
                                                           "prefill_spec": args.prefill_spec,
                                                           "template": LIST_TEMPLATE})
        elif step == "e15":
            qs = distinct_unmatched(samples)
            print(f"[e15] yes/no on {len(qs)} distinct non-member answers", flush=True)
            v = yesno_scores(model, tok, chat, [{"name": q["text"], "singular": cats[q["category"]]["singular"]}
                                                for q in qs], args.device, pad_id, args.batch_size, probe="apriori")
            write_jsonl(main_out[step], [q | x for q, x in zip(qs, v)])
        secs = time.time() - t0
        write_json(out / f"{step}{suffix if step == 'e14' else ''}_meta.json",
                   run_meta(model, args.model) | {"chat": chat, "specs": specs, "margin": margin, "seconds": secs,
                                                 "open_run": args.open, "p5": args.p5, "args": vars(args),
                                                 "peak_gpu_gib": peak_gib()})
        print(f"[p6] {step} done in {secs / 60:.1f} min", flush=True)
    secs = time.time() - t_all
    if args.device == "cuda":
        append_gpu_ledger("p6:" + "+".join(todo), args.model, secs, note=f"{args.out} peak {peak_gib():.1f} GiB")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
