"""Sampler specifications and the exact Hugging Face processor chain for each.

A "spec" is a plain dict with any of: temperature, top_k, top_p, min_p,
repetition_penalty. The chain is built in the same order that
transformers' generate() uses when do_sample=True:

    repetition penalty -> temperature -> top-k -> top-p -> min-p

The order matters. top-p is computed on the distribution that remains after
temperature and top-k; min-p is computed after top-p. tests/test_semantics.py
checks our analytic formulas against these processors, and the G0 sample
check verifies the whole pipeline against real model.generate() output.
"""
from __future__ import annotations

from transformers import LogitsProcessorList
from transformers.generation.logits_process import (
    MinPLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

# Standard grid. Pure settings (no repetition penalty) so curves are clean.
GRID: dict[str, dict] = {
    "untruncated_T1": {},  # control: everything is reachable here
    "p0.80_T1": {"top_p": 0.80},
    "p0.90_T1": {"top_p": 0.90},
    "p0.95_T1": {"top_p": 0.95},
    "p0.90_T0.7": {"temperature": 0.7, "top_p": 0.90},
    "k20_T1": {"top_k": 20},
    "k50_T1": {"top_k": 50},  # transformers' generate() default when a config sets nothing
    "minp0.05_T1": {"min_p": 0.05},
    "minp0.10_T1": {"min_p": 0.10},
}


def build_chain(spec: dict) -> LogitsProcessorList:
    chain = LogitsProcessorList()
    rp = spec.get("repetition_penalty")
    if rp is not None and float(rp) != 1.0:
        chain.append(RepetitionPenaltyLogitsProcessor(penalty=float(rp)))
    t = spec.get("temperature")
    if t is not None and float(t) != 1.0:
        chain.append(TemperatureLogitsWarper(float(t)))
    k = spec.get("top_k")
    if k is not None and int(k) > 0:
        chain.append(TopKLogitsWarper(top_k=int(k)))
    p = spec.get("top_p")
    if p is not None and float(p) < 1.0:
        chain.append(TopPLogitsWarper(top_p=float(p)))
    mp = spec.get("min_p")
    if mp is not None and float(mp) > 0.0:
        chain.append(MinPLogitsWarper(min_p=float(mp)))
    return chain


def shipped_specs(gen_config) -> dict[str, dict]:
    """Two readings of a model's shipped sampler.

    shipped_hf:   what transformers' generate() actually does with this
                  generation_config (library defaults such as top_k=50 fill
                  any field the json leaves unset).
    shipped_json: only the fields the model authors wrote into
                  generation_config.json (closer to vLLM / server defaults,
                  which do not add top_k=50).

    Returns {} if the config does not sample (do_sample False).
    """
    if gen_config is None or not getattr(gen_config, "do_sample", False):
        return {}
    keys = ("temperature", "top_k", "top_p", "min_p", "repetition_penalty")
    hf = {k: getattr(gen_config, k, None) for k in keys}
    hf = {k: v for k, v in hf.items() if v is not None}
    try:
        diff = gen_config.to_diff_dict()
    except Exception:  # pragma: no cover
        diff = {}
    js = {k: diff[k] for k in keys if k in diff and diff[k] is not None}
    out = {"shipped_hf": hf}
    if js != hf:
        out["shipped_json"] = js
    return out


# generation_config fields that make generate() add processors we do not model.
# Several of them renormalise the distribution before top-p (e.g. min_new_tokens
# masks EOS), which changes which tokens survive. If any is set, the exact
# reachability for "shipped" settings must be recomputed with that processor
# included, or the sample check (V1) will show violations.
EXTRA_FIELDS = ("min_length", "min_new_tokens", "suppress_tokens", "begin_suppress_tokens", "bad_words_ids",
                "no_repeat_ngram_size", "encoder_repetition_penalty", "sequence_bias", "forced_bos_token_id",
                "forced_eos_token_id", "exponential_decay_length_penalty", "renormalize_logits", "typical_p",
                "epsilon_cutoff", "eta_cutoff")


def extra_processor_fields(gen_config) -> dict:
    if gen_config is None:
        return {}
    try:
        diff = gen_config.to_diff_dict()
    except Exception:  # pragma: no cover
        return {}
    return {k: diff[k] for k in EXTRA_FIELDS if k in diff and diff[k] not in (None, 0, [], False)}


def describe(spec: dict) -> str:
    if not spec:
        return "no truncation, T=1"
    return ", ".join(f"{k}={v}" for k, v in spec.items())
