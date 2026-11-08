"""Sampling sanity check (validity check V1 for gate G0).

Draw real samples and confirm the exact reachability computation:
  * an answer we computed as unreachable must never be produced with the same
    token ids (any such case is a bug in our code or in our reading of the
    sampler: stop and fix);
  * text matches to unreachable entities produced through a different
    tokenisation are counted separately ("non-canonical paths"); their rate
    bounds how much canonical-tokenisation analysis overstates unreachability.

Two modes:
  chain    generate() with a clean config plus our processor chain for a spec
  shipped  generate() with the model's own generation_config and nothing else
           (checks our reading of the shipped defaults, spec "shipped_hf")
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

import torch
from transformers import GenerationConfig

from .samplers import build_chain

_STRIP = re.compile(r"^[\s\"'*`(\[]+|[\s\"'*`.,;:!?)\]]+$")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).split("\n")[0]
    s = _STRIP.sub("", s).casefold()
    return re.sub(r"\s+", " ", s)


KV_BUDGET_GIB = 3.0  # cap on the KV cache of one generation batch


def kv_batch_size(model, seq_len: int, cap: int = 200, budget_gib: float = KV_BUDGET_GIB) -> int:
    """Largest batch (at most `cap`) whose KV cache fits in `budget_gib`. Models
    without grouped-query attention (e.g. OLMo-2-7B: 32 KV heads) need about 9
    times the cache of Qwen2.5-7B (4 KV heads), so a fixed batch of 200 does not
    fit next to the weights. Qwen2.5 keeps 200, so its G0 samples are unchanged."""
    c = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    layers = c.num_hidden_layers
    kv_heads = getattr(c, "num_key_value_heads", None) or c.num_attention_heads
    head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    per_token = 2 * layers * kv_heads * head_dim * 2  # keys and values, bf16
    return max(8, min(cap, int(budget_gib * 2**30 / (per_token * seq_len))))


@torch.no_grad()
def generate_samples(model, prompt_ids: list[int], n: int, max_new_tokens: int, device, pad_id: int,
                     spec: dict | None, batch_size: int | None = None, seed: int = 0) -> list[list[int]]:
    torch.manual_seed(seed)
    if batch_size is None:
        batch_size = kv_batch_size(model, len(prompt_ids) + max_new_tokens)
    outs: list[list[int]] = []
    base = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    eos = model.generation_config.eos_token_id
    while len(outs) < n:
        bs = min(batch_size, n - len(outs))
        inp = base.repeat(bs, 1)
        att = torch.ones_like(inp)
        if spec is None:  # shipped: the model's own generation_config
            gen = model.generate(input_ids=inp, attention_mask=att, do_sample=True,
                                 max_new_tokens=max_new_tokens, pad_token_id=pad_id)
        else:
            gc = GenerationConfig(do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                                  repetition_penalty=1.0, max_new_tokens=max_new_tokens,
                                  pad_token_id=pad_id, eos_token_id=eos)
            gen = model.generate(input_ids=inp, attention_mask=att, generation_config=gc,
                                 logits_processor=build_chain(spec))
        outs.extend(gen[:, inp.shape[1]:].tolist())
    return outs


def check(samples: list[list[int]], tok, items: list[dict], spec_name: str, name_index: dict[str, str]) -> dict:
    """items: reach records for ONE (category, template) with 'answer_ids',
    'qid', 'form' and f'reach__{spec_name}'. name_index: normalised surface form -> qid."""
    by_len: dict[int, dict[tuple, dict]] = {}
    for it in items:
        by_len.setdefault(len(it["answer_ids"]), {})[tuple(it["answer_ids"])] = it
    def robust_unreach(it):  # records scored without a margin fall back to the exact verdict
        return bool(it.get(f"robust_unreach__{spec_name}", not it[f"reach__{spec_name}"]))

    runreach_q: dict[str, bool] = {}
    for it in items:  # entity robustly unreachable only if EVERY surface form is
        runreach_q[it["qid"]] = runreach_q.get(it["qid"], True) and robust_unreach(it)
    violations, exact_viol, boundary, noncanon, rows = [], [], [], [], []
    id_hits, text_hits = Counter(), Counter()
    for s in samples:
        hit = None
        for L, table in by_len.items():
            it = table.get(tuple(s[:L]))
            if it is not None and (hit is None or L > len(hit["answer_ids"])):
                hit = it
        if hit is not None:
            id_hits[hit["qid"]] += 1
            ex = {"qid": hit["qid"], "form": hit.get("form"), "ids": s[: len(hit["answer_ids"])]}
            if robust_unreach(hit):
                violations.append(ex)  # V1: claimed unreachable with margin, yet sampled
            elif not hit[f"reach__{spec_name}"]:
                boundary.append(ex)  # exact verdict was "cut" but within the numeric margin
            if not hit[f"reach__{spec_name}"]:
                exact_viol.append(ex)
        raw = tok.decode(s, skip_special_tokens=True)
        text = norm(raw)
        q = name_index.get(text) or name_index.get(text.removeprefix("the "))
        if q is not None:
            text_hits[q] += 1
            if hit is None and runreach_q.get(q, False):
                noncanon.append({"qid": q, "text": text})
        rows.append({"text": raw, "norm_text": text, "ids": s, "text_match_qid": q,
                     "id_hit_qid": hit["qid"] if hit else None, "id_hit_form": hit.get("form") if hit else None,
                     "id_hit_reach_exact": bool(hit[f"reach__{spec_name}"]) if hit else None,
                     "id_hit_robust_unreach": robust_unreach(hit) if hit else None})
    return {
        "spec": spec_name,
        "margin": items[0].get("margin", 0.0) if items else None,
        "n_samples": len(samples),
        "n_valid_text": int(sum(text_hits.values())),
        "distinct_valid": len(text_hits),
        "n_id_prefix_hits": int(sum(id_hits.values())),
        "distinct_id_prefix_hits": len(id_hits),
        "violations": len(violations),
        "violation_examples": violations[:20],
        "boundary_hits": len(boundary),
        "boundary_examples": boundary[:20],
        "exact_violations": len(exact_viol),
        "noncanonical": len(noncanon),
        "noncanonical_examples": noncanon[:20],
        "distinct_valid_qids": sorted(text_hits),
        "valid_counts": dict(text_hits),
        "_samples": rows,  # per-sample records; run_model moves them to samples.jsonl.gz
    }
