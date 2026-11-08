"""One model, one TailCat split: reachability + recognition (+ optional sample check).

Example (G0 pilot):
  python -m mb5.run_model --model Qwen/Qwen2.5-7B-Instruct --data data/tailcat/pilot \
      --out results/G0 --templates 3 --sample-check

Outputs in <out>/<model_tag>/:
  reach.jsonl.gz        one row per (entity, surface form, template)
  recog.jsonl.gz        one row per member and per foil (yes/no V; 4-way for members)
  sample_check.json     (if --sample-check) violations and coverage counts
  meta.json             model commit, library versions, GPU, specs, wall time
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import torch

from .common import (append_gpu_ledger, estimate_need_gib, is_chat_model, load_model, model_tag, peak_gib, read_jsonl,
                     require_gpu_memory, run_meta, seed_all, write_json, write_jsonl)
from .prompts import BASE_GEN_SHOTS, GEN_TEMPLATES, answer_prefix, encode_pair, encode_prompt, render
from .reachability import score_items
from .recognition import mc_scores, yesno_scores
from .sample_check import check, generate_samples, norm
from .samplers import GRID, describe, extra_processor_fields, shipped_specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chat", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--templates", type=int, default=3)
    ap.add_argument("--limit-per-category", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dstar-specs", default="shipped_hf,p0.90_T1",
                    help="specs whose signed-margin curve (d*) is recorded; each costs ~12 chain evaluations")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="raw-logit margin for the robust verdicts (calibrate with mb5.selfcheck; see docs/04)")
    ap.add_argument("--no-aliases", action="store_true")
    ap.add_argument("--sample-check", action="store_true")
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--sample-specs", default="shipped_hf,p0.90_T1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_all(args.seed)
    if args.device == "cuda":
        require_gpu_memory(estimate_need_gib(args.model, args.batch_size))
    t0 = time.time()  # after any wait for memory: waiting is not GPU time
    chat = is_chat_model(args.model) if args.chat == "auto" else args.chat == "yes"
    model, tok = load_model(args.model, device=args.device)
    pad_id = tok.pad_token_id
    tag = model_tag(args.model)
    out = Path(args.out) / tag

    data = Path(args.data)
    cats = {c["id"]: c for c in __import__("json").loads((data / "categories.json").read_text())}
    ents = read_jsonl(data / "entities.jsonl")
    foils = read_jsonl(data / "foils.jsonl")
    if args.limit_per_category:
        seen = defaultdict(int)
        keep = []
        for e in ents:
            if seen[e["category"]] < args.limit_per_category:
                keep.append(e)
                seen[e["category"]] += 1
        ents = keep

    specs = dict(GRID)
    specs.update(shipped_specs(model.generation_config) if chat else {})
    print("[specs]", {k: describe(v) for k, v in specs.items()})
    extra = extra_processor_fields(model.generation_config)
    if extra:
        print(f"[WARNING] generation_config sets {extra}: generate() will add processors that the "
              "shipped_hf reachability does not model. Expect V1 violations; extend samplers.build_chain.")

    # ---------------- reachability items
    templates = GEN_TEMPLATES[: args.templates]
    shots = "" if chat else BASE_GEN_SHOTS
    pre = answer_prefix(chat)
    prompt_cache: dict[tuple, str] = {}
    items = []
    for e in ents:
        c = cats[e["category"]]
        forms = [e["label"]] + ([] if args.no_aliases else e.get("aliases", []))
        for ti, tpl in enumerate(templates):
            key = (c["id"], ti)
            if key not in prompt_cache:
                prompt_cache[key] = render(tok, tpl.format(singular=c["singular"], plural=c["plural"]), chat, shots)
            for fi, form in enumerate(forms):
                p_ids, a_ids, ok = encode_pair(tok, prompt_cache[key], pre + form, chat)
                if not a_ids:
                    continue
                items.append({"qid": e["qid"], "category": c["id"], "template": ti, "form": form,
                              "is_label": fi == 0, "sitelinks": e["sitelinks"], "boundary_ok": ok,
                              "prompt_ids": p_ids, "answer_ids": a_ids})
    print(f"[reach] {len(items)} items ({len(ents)} entities x forms x {len(templates)} templates)")
    reach = score_items(model, items, specs, device=args.device, batch_size=args.batch_size, pad_id=pad_id,
                        margin=args.margin, dstar=[s for s in args.dstar_specs.split(",") if s])
    write_jsonl(out / "reach.jsonl.gz", reach)

    # ---------------- recognition (members and foils)
    q_mem = [{"name": e["label"], "singular": cats[e["category"]]["singular"], "category": e["category"]} for e in ents]
    q_foil = [{"name": f["label"], "singular": cats[f["category"]]["singular"], "category": f["category"]} for f in foils]
    print(f"[recog] yes/no on {len(q_mem)} members and {len(q_foil)} foils")
    v_mem = yesno_scores(model, tok, chat, q_mem, args.device, pad_id, args.batch_size)
    v_foil = yesno_scores(model, tok, chat, q_foil, args.device, pad_id, args.batch_size)
    pool = defaultdict(list)
    for f in foils:
        pool[f["category"]].append(f["label"])
    mc = mc_scores(model, tok, chat, q_mem, pool, args.device, pad_id, seed=args.seed, batch_size=args.batch_size)
    recog = []
    for e, v, m in zip(ents, v_mem, mc):
        recog.append({"qid": e["qid"], "category": e["category"], "kind": "member", "label": e["label"],
                      "sitelinks": e["sitelinks"], "V": v["V"], "logp_yes": v["logp_yes"], "logp_no": v["logp_no"],
                      "mc_p": m["p_correct"], "mc_correct": m["correct"], "mc_options": m["options"],
                      "mc_answer": m["answer"], "mc_chosen": m["chosen"], "mc_letter_probs": m["letter_probs"],
                      "yesno_prompt": v["prompt"], "mc_prompt": m["prompt"]})
    for f, v in zip(foils, v_foil):
        recog.append({"qid": f["qid"], "category": f["category"], "kind": "foil", "label": f["label"],
                      "sitelinks": f["sitelinks"], "V": v["V"], "logp_yes": v["logp_yes"], "logp_no": v["logp_no"],
                      "yesno_prompt": v["prompt"]})
    write_jsonl(out / "recog.jsonl.gz", recog)
    # The exact generation prompts (one per category and wording), for the record and the appendix.
    write_json(out / "prompts.json", {"generation": {f"{cid}|template{ti}": p for (cid, ti), p in prompt_cache.items()},
                                      "templates": templates, "answer_prefix": pre})

    # ---------------- sampling sanity check (template 0 only)
    sc_results, sample_rows = [], []
    if args.sample_check:
        names = defaultdict(dict)
        for e in ents:
            for form in [e["label"]] + e.get("aliases", []):
                names[e["category"]][norm(form)] = e["qid"]
        for spec_name in [s for s in args.sample_specs.split(",") if s in specs]:
            for cid in cats:
                cat_items = [r for r in reach if r["category"] == cid and r["template"] == 0]
                if not cat_items:
                    continue
                p_ids = encode_prompt(tok, prompt_cache[(cid, 0)], chat)
                max_new = min(32, max(len(r["answer_ids"]) for r in cat_items) + 4)
                spec = None if spec_name == "shipped_hf" else specs[spec_name]
                samples = generate_samples(model, p_ids, args.n_samples, max_new, args.device, pad_id, spec,
                                           seed=args.seed)
                res = check(samples, tok, cat_items, spec_name, names[cid])
                res["category"] = cid
                for i, row in enumerate(res.pop("_samples")):  # every generated answer, kept verbatim
                    sample_rows.append({"spec": spec_name, "category": cid, "i": i} | row)
                sc_results.append(res)
                print(f"[sample] {spec_name:12s} {cid:28s} valid {res['n_valid_text']:4d} distinct "
                      f"{res['distinct_valid']:3d} violations {res['violations']} boundary {res['boundary_hits']} "
                      f"exact_viol {res['exact_violations']} noncanonical {res['noncanonical']}", flush=True)
        write_json(out / "sample_check.json", sc_results)
        write_jsonl(out / "samples.jsonl.gz", sample_rows)

    secs = time.time() - t0
    meta = run_meta(model, args.model) | {"chat": chat, "specs": specs, "templates": templates,
                                         "extra_processor_fields": extra, "margin": args.margin,
                                         "n_entities": len(ents), "n_foils": len(foils), "seconds": secs, "peak_gpu_gib": peak_gib(),
                                         "args": vars(args)}
    write_json(out / "meta.json", meta)
    if args.device == "cuda":
        append_gpu_ledger("run_model", args.model, secs, note=f"{args.out} peak {peak_gib():.1f} GiB")
    print(f"[done] {tag} in {secs / 60:.1f} min -> {out}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
