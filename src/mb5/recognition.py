"""Does the model recognise an entity as a member of the category?

Two independent probes:
  yes/no  V(e) = P(Yes) / (P(Yes) + P(No)) for 'Is "e" <category>?'
          (variants of the Yes/No first token are pooled with logsumexp)
  4-way   P(correct letter) when e is shown with three foils from the same
          neighbourhood (sibling categories), positions randomised.

analysis.py turns these into a binary "recognised" label using foils to set
the yes/no threshold (false-positive rate on foils <= 5%) and requiring the
4-way choice to be correct.
"""
from __future__ import annotations

import math
import random

import torch

from .prompts import BASE_MC_SHOTS, BASE_YESNO_SHOTS, MC, YESNO, encode_prompt, letter_tokens, render, yes_no_tokens


@torch.no_grad()
def next_token_logprobs(model, prompts_ids: list[list[int]], cand_ids: list[int], device, pad_id: int,
                        batch_size: int = 32) -> list[list[float]]:
    out: list[list[float]] = []
    cand = torch.tensor(cand_ids, dtype=torch.long, device=device)
    for b in range(0, len(prompts_ids), batch_size):
        chunk = prompts_ids[b : b + batch_size]
        S = max(len(s) for s in chunk)
        ids = torch.full((len(chunk), S), pad_id, dtype=torch.long)
        att = torch.zeros((len(chunk), S), dtype=torch.long)
        for i, s in enumerate(chunk):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            att[i, : len(s)] = 1
        logits = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits
        last = torch.tensor([len(s) - 1 for s in chunk], device=logits.device)
        rows = logits[torch.arange(len(chunk), device=logits.device), last].float()
        lp = torch.log_softmax(rows, dim=1)[:, cand]
        out.extend(lp.cpu().tolist())
    return out


def _lse(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def yesno_scores(model, tok, chat: bool, queries: list[dict], device, pad_id, batch_size=32) -> list[float]:
    """queries: dicts with 'name' and 'singular'. Returns V in [0,1]."""
    yes, no = yes_no_tokens(tok, chat)
    shots = "" if chat else BASE_YESNO_SHOTS
    prompts = [encode_prompt(tok, render(tok, YESNO.format(name=q["name"], singular=q["singular"]), chat, shots), chat)
               for q in queries]
    lps = next_token_logprobs(model, prompts, yes + no, device, pad_id, batch_size)
    out = []
    for lp in lps:
        ly, ln = _lse(lp[: len(yes)]), _lse(lp[len(yes):])
        out.append(1.0 / (1.0 + math.exp(ln - ly)))
    return out


def mc_scores(model, tok, chat: bool, queries: list[dict], foil_pool: dict[str, list[str]], device, pad_id,
              seed: int = 0, batch_size=32) -> list[dict]:
    """queries: dicts with 'name', 'singular', 'category'. Returns per query
    {'p_correct', 'correct'} where correct means the right letter is argmax."""
    rng = random.Random(seed)
    letters = letter_tokens(tok, chat)
    shots = "" if chat else BASE_MC_SHOTS
    prompts, answers = [], []
    for q in queries:
        pool = [f for f in foil_pool.get(q["category"], []) if f.casefold() != q["name"].casefold()]
        foils = rng.sample(pool, 3) if len(pool) >= 3 else (pool + ["(none)"] * 3)[:3]
        opts = foils + [q["name"]]
        rng.shuffle(opts)
        ans = opts.index(q["name"])
        text = MC.format(singular=q["singular"], a=opts[0], b=opts[1], c=opts[2], d=opts[3])
        prompts.append(encode_prompt(tok, render(tok, text, chat, shots), chat))
        answers.append(ans)
    lps = next_token_logprobs(model, prompts, letters, device, pad_id, batch_size)
    out = []
    for lp, a in zip(lps, answers):
        m = max(lp)
        ps = [math.exp(x - m) for x in lp]
        s = sum(ps)
        out.append({"p_correct": ps[a] / s, "correct": int(max(range(4), key=lp.__getitem__) == a)})
    return out
