"""Dense self-consistency check of the reachability scorer on a real model.

The G0 sample check (sample_check.py) only tests answers that match a TailCat
entity, so on a small model it can be nearly vacuous. This check needs no
entity list: draw samples with generate(), then score every distinct sampled
continuation with score_items under the same spec. Every sampled sequence was
produced by the sampler, so every one of its tokens survived the processor
chain and must be scored reachable. Any failure means the teacher-forced
scorer and generate() disagree (for example bf16 numeric differences between
a batched full-sequence forward and KV-cached decoding at a top-p boundary).

Failures are reported with the target's G (mass strictly above it) and the
spec's p, so a borderline numeric flip can be told apart from a logic bug.

  python -m mb5.selfcheck --model Qwen/Qwen2.5-0.5B-Instruct --data data/tailcat/pilot \
      --out results/selfcheck --n-samples 500
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
import time
from pathlib import Path

import torch

from .common import (append_gpu_ledger, estimate_need_gib, is_chat_model, load_model, model_tag, peak_gib,
                     require_gpu_memory,
                     run_meta, seed_all, write_json, write_jsonl)
from .prompts import BASE_GEN_SHOTS, GEN_TEMPLATES, encode_prompt, render
from .reachability import _forward_rows, score_items
from .sample_check import generate_samples
from .samplers import GRID, shipped_specs


def _eos_ids(model) -> set[int]:
    e = model.generation_config.eos_token_id
    return set(e if isinstance(e, list) else ([e] if e is not None else []))


GAP_KEYS = ("rel_all", "rel_near", "dG_T1.0", "dG_T0.7", "drank")
EPS_GRID = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@torch.no_grad()
def logit_gap(model, p_ids: list[int], n: int, max_new: int, pad_id: int, eos: set[int], batch_size: int = 50,
              seed: int = 0, near: float = 2.0, tf_batch: int | None = None) -> dict[str, list[float]]:
    """How far apart are generate()'s raw logits and the scorer's teacher-forced
    logits for the same sequences? With d = z_generate - z_scorer and t the
    sampled token, per sampled position:
      rel_all   max over the whole vocabulary of |d_v - d_t|
      rel_near  the same over tokens the scorer puts above t or within `near`
                logits below it (the only tokens that can change t's rank)
      dG_T*     |G_generate - G_scorer|, G = mass strictly above t at that T
      drank     |rank_generate - rank_scorer|
    These calibrate the margin used to call a token robustly cut."""
    from .reachability import row_stats

    torch.manual_seed(seed)
    base = torch.tensor([p_ids], dtype=torch.long, device=model.device)
    gaps: dict[str, list[float]] = {k: [] for k in GAP_KEYS}
    done = 0
    while done < n:
        bs = min(batch_size, n - done)
        inp = base.repeat(bs, 1)
        g = model.generate(input_ids=inp, attention_mask=torch.ones_like(inp), do_sample=True,
                           max_new_tokens=max_new, pad_token_id=pad_id, return_dict_in_generate=True,
                           output_logits=True)
        seqs = g.sequences[:, inp.shape[1]:].tolist()
        raw = torch.stack(g.logits, 1).float()  # [bs, steps, V], before any processor
        batch, gen_rows = [], []
        for b, s in enumerate(seqs):
            cut = next((i for i, t in enumerate(s) if t in eos), None)
            s = s if cut is None else s[: cut + 1]
            if s:
                batch.append((p_ids, s))
                gen_rows.append(raw[b, : len(s)])
        step = tf_batch or len(batch)  # tf_batch=1: unpadded teacher forcing, to isolate padding effects
        tf = torch.cat([_forward_rows(model, batch[i: i + step], model.device, pad_id)
                        for i in range(0, len(batch), step)], 0)
        gen = torch.cat(gen_rows, 0)
        tgt = torch.tensor([t for _, s in batch for t in s], device=tf.device)
        for c0 in range(0, tf.shape[0], 256):
            sl = slice(c0, c0 + 256)
            a, b, t = tf[sl], gen[sl], tgt[sl]
            rel = ((b - a) - (b - a).gather(1, t[:, None])).abs()
            gaps["rel_all"].extend(rel.max(1).values.cpu().tolist())
            near_mask = a >= a.gather(1, t[:, None]) - near
            gaps["rel_near"].extend(rel.masked_fill(~near_mask, 0).max(1).values.cpu().tolist())
            sa, sb = row_stats(a, t), row_stats(b, t)
            for T in (1.0, 0.7):
                gaps[f"dG_T{T}"].extend((sb[f"G_T{T}"] - sa[f"G_T{T}"]).abs().cpu().tolist())
            gaps["drank"].extend((sb["rank"] - sa["rank"]).abs().float().cpu().tolist())
        done += bs
        del g, raw, tf, gen
    return gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--specs", default="shipped_hf,p0.90_T1")
    ap.add_argument("--n-samples", type=int, default=500)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gap-samples", type=int, default=100,
                    help="samples per category for the generate-vs-scorer logit gap (0 to skip)")
    ap.add_argument("--gap-only", action="store_true")
    ap.add_argument("--gap-tf-batch", type=int, default=None,
                    help="teacher-forcing batch size in the gap measurement (1 = no padding)")
    ap.add_argument("--margins", default="0.25,0.5,0.75,1.0,1.5,2.0,3.0",
                    help="margins at which to report how many sampled sequences would be called unreachable")
    ap.add_argument("--device", default="cuda", help="cpu only for testing the code path")
    ap.add_argument("--limit-categories", type=int, default=None, help="testing only")
    args = ap.parse_args()

    seed_all(args.seed)
    if args.device == "cuda":
        require_gpu_memory(estimate_need_gib(args.model, args.batch_size))
    t0 = time.time()
    chat = is_chat_model(args.model)
    model, tok = load_model(args.model, device=args.device)
    specs = dict(GRID) | (shipped_specs(model.generation_config) if chat else {})
    eos = _eos_ids(model)
    cats = json.loads((Path(args.data) / "categories.json").read_text())[: args.limit_categories]
    shots = "" if chat else BASE_GEN_SHOTS

    gap_stats = None
    if args.gap_samples > 0:
        import numpy as np

        gaps = {k: [] for k in GAP_KEYS}
        for c in cats:
            prompt = render(tok, GEN_TEMPLATES[0].format(singular=c["singular"], plural=c["plural"]), chat, shots)
            got = logit_gap(model, encode_prompt(tok, prompt, chat), args.gap_samples, args.max_new_tokens,
                            tok.pad_token_id, eos, seed=args.seed, tf_batch=args.gap_tf_batch)
            for k in GAP_KEYS:
                gaps[k] += got[k]
        gap_stats = {}
        for k in GAP_KEYS:
            g = np.asarray(gaps[k])
            gap_stats[k] = {"n_positions": int(g.size), "mean": float(g.mean()),
                            **{f"q{q}": float(np.quantile(g, q)) for q in (0.5, 0.9, 0.99, 0.999)},
                            "max": float(g.max())}
            print(f"[selfcheck] gap {k:8s} over {g.size} positions: " +
                  ", ".join(f"{q} {v:.4f}" for q, v in gap_stats[k].items() if q != "n_positions"), flush=True)

    margins = [float(m) for m in args.margins.split(",")]
    required: dict[str, float] = {}
    seq_rows: list[dict] = []
    rows, total = [], {"distinct": 0, "unreachable": 0, **{f"robust_unreachable@{m}": 0 for m in margins}}
    for spec_name in ([] if args.gap_only else [s for s in args.specs.split(",") if s in specs]):
        for c in cats:
            prompt = render(tok, GEN_TEMPLATES[0].format(singular=c["singular"], plural=c["plural"]), chat, shots)
            p_ids = encode_prompt(tok, prompt, chat)
            spec = None if spec_name == "shipped_hf" else specs[spec_name]
            samples = generate_samples(model, p_ids, args.n_samples, args.max_new_tokens, args.device,
                                       tok.pad_token_id, spec, seed=args.seed)
            seqs = Counter()
            for s in samples:
                cut = next((i for i, t in enumerate(s) if t in eos), None)
                seqs[tuple(s if cut is None else s[: cut + 1])] += 1  # keep the sampled EOS: it was kept too
            items = [{"prompt_ids": p_ids, "answer_ids": list(q)} for q in sorted(seqs) if q]
            recs = score_items(model, items, {spec_name: specs[spec_name]}, device=args.device, batch_size=args.batch_size,
                               pad_id=tok.pad_token_id, progress=False, dstar=True)
            for r in recs:  # every distinct sampled sequence, kept verbatim with its verdicts
                seq_rows.append({"spec": spec_name, "category": c["id"], "text": tok.decode(r["answer_ids"]),
                                 "ids": r["answer_ids"], "times_sampled": seqs[tuple(r["answer_ids"])],
                                 "dstar": r[f"dstar__{spec_name}"], "reach_exact": r[f"reach__{spec_name}"],
                                 "failpos_exact": r[f"failpos__{spec_name}"]})
            # Every sequence here was sampled, so at a valid margin eps none may have d* < -eps.
            need = [max(0.0, -r[f"dstar__{spec_name}"]) for r in recs]
            required[spec_name] = max([required.get(spec_name, 0.0)] + need)
            robust = {m: sum(x > m for x in need) for m in margins}
            for m in margins:
                total[f"robust_unreachable@{m}"] += robust[m]
            bad = [r for r in recs if not r[f"reach__{spec_name}"]]
            p = specs[spec_name].get("top_p")
            examples = []
            for r in bad[:10]:
                j = r[f"failpos__{spec_name}"]
                examples.append({"text": tok.decode(r["answer_ids"]), "failpos": j, "rank": r["rank_pos"][j],
                                 "G_T1": r["G_T1.0_pos"][j], "top_p": p})
            rows.append({"spec": spec_name, "category": c["id"], "n_samples": len(samples),
                         "distinct_sequences": len(items), "unreachable": len(bad),
                         "robust_unreachable": {str(m): v for m, v in robust.items()}, "examples": examples})
            total["distinct"] += len(items)
            total["unreachable"] += len(bad)
            print(f"[selfcheck] {spec_name:12s} {c['id']:26s} distinct {len(items):4d} exact-unreachable {len(bad):3d} "
                  "robust " + " ".join(f"@{m}:{v}" for m, v in robust.items()), flush=True)

    # Pre-registered rule (notes/decisions.md): epsilon = smallest grid value at least the
    # largest margin any sampled sequence needed, plus the bisection resolution.
    eps = None
    if required:
        need_max = max(required.values()) + 0.02
        eps = next((g for g in EPS_GRID if g >= need_max), None)
        print(f"[selfcheck] largest margin needed by a sampled sequence: " +
              ", ".join(f"{s} {v:.3f}" for s, v in required.items()) + f" -> epsilon rule gives {eps}", flush=True)
    secs = time.time() - t0
    out = Path(args.out) / model_tag(args.model)
    write_json(out / "selfcheck.json", {"total": total, "required_margin": required, "epsilon_rule": eps,
                                        "epsilon_grid": list(EPS_GRID), "logit_gap": gap_stats, "rows": rows,
                                        "meta": run_meta(model, args.model) | {"seconds": secs, "peak_gpu_gib": peak_gib(), "args": vars(args),
                                                                               "specs": specs}})
    if seq_rows:
        write_jsonl(out / "selfcheck_samples.jsonl.gz", seq_rows)
    if gap_stats is not None:  # raw values behind the summary statistics
        write_jsonl(out / "logit_gap_raw.jsonl.gz", [{"stat": k, "values": gaps[k]} for k in GAP_KEYS])
    if args.device == "cuda":
        append_gpu_ledger("selfcheck", args.model, secs, note=f"{args.out} peak {peak_gib():.1f} GiB")
    print(f"[selfcheck] TOTAL distinct {total['distinct']} exact-unreachable {total['unreachable']} robust " +
          " ".join(f"@{m}:{total[f'robust_unreachable@{m}']}" for m in margins) +
          f" in {secs / 60:.1f} min (a margin passes when its robust count is 0)", flush=True)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
