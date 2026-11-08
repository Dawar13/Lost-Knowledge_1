"""First-token concentration: the predictor of RBU in docs/04 section 5 (E5).

For each generation prompt (category x template) the distribution of the first
answer token at T=1: top-1 and top-5 mass, entropy (nats), how many tokens
top-p 0.9 / 0.95 and min-p 0.1 keep, and the ten most likely first tokens.
One forward pass per prompt. run_model computes this for every run; the CLI
fills it in for runs made before that (G0):

  python -m mb5.firsttoken --model Qwen/Qwen2.5-7B-Instruct --data data/tailcat/pilot_v2 \
      --out results/G0_5090 --templates 3
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from .common import (append_gpu_ledger, estimate_need_gib, is_chat_model, load_model, model_tag, peak_gib,
                     require_gpu_memory, run_meta, write_json)
from .prompts import BASE_GEN_SHOTS, GEN_TEMPLATES, encode_prompt, render


@torch.no_grad()
def first_token_stats(model, tok, prompts: dict[tuple, str], chat: bool, device, top: int = 10) -> list[dict]:
    """prompts: {(category, template): prompt text}. Returns one record per prompt."""
    out = []
    for (cid, ti), text in sorted(prompts.items()):
        ids = torch.tensor([encode_prompt(tok, text, chat)], device=device)
        logits = model(input_ids=ids).logits[0, -1].float()
        p = torch.softmax(logits, -1)
        ps, idx = torch.sort(p, descending=True)
        ahead = torch.cumsum(ps, 0) - ps  # mass strictly ahead of each token in sorted order
        ent = float(-(p * torch.log(p.clamp_min(1e-30))).sum())
        out.append({
            "category": cid, "template": ti, "top1_mass": float(ps[0]), "top5_mass": float(ps[:5].sum()),
            "entropy_nats": ent, "perplexity": math.exp(ent),
            "kept_top_p_0.9": int((ahead < 0.9).sum()), "kept_top_p_0.95": int((ahead < 0.95).sum()),
            "kept_min_p_0.1": int((ps >= 0.1 * ps[0]).sum()),
            "top_tokens": [{"token": tok.decode([int(i)]), "id": int(i), "p": float(q)}
                           for q, i in zip(ps[:top], idx[:top])],
        })
    return out


def generation_prompts(tok, cats: list[dict], templates: list[str], chat: bool) -> dict[tuple, str]:
    shots = "" if chat else BASE_GEN_SHOTS
    return {(c["id"], ti): render(tok, t.format(singular=c["singular"], plural=c["plural"]), chat, shots)
            for c in cats for ti, t in enumerate(templates)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True, help="run directory; writes <out>/<model_tag>/first_token.json")
    ap.add_argument("--templates", type=int, default=3)
    args = ap.parse_args()
    require_gpu_memory(estimate_need_gib(args.model, 1))
    t0 = time.time()
    chat = is_chat_model(args.model)
    model, tok = load_model(args.model)
    cats = json.loads((Path(args.data) / "categories.json").read_text())
    rows = first_token_stats(model, tok, generation_prompts(tok, cats, GEN_TEMPLATES[: args.templates], chat),
                             chat, "cuda")
    secs = time.time() - t0
    write_json(Path(args.out) / model_tag(args.model) / "first_token.json",
               {"rows": rows, "meta": run_meta(model, args.model) | {"seconds": secs, "args": vars(args)}})
    append_gpu_ledger("firsttoken", args.model, secs, note=f"{args.out} peak {peak_gib():.1f} GiB")
    print(f"[firsttoken] {len(rows)} prompts in {secs:.0f}s; mean top-1 mass "
          f"{sum(r['top1_mass'] for r in rows) / len(rows):.3f}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
