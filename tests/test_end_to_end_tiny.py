"""End-to-end on a tiny random Llama (CPU, no download).

1. Score candidate answers with score_items.
2. Sample from model.generate() with the same truncation.
3. Every sampled answer prefix must have been scored reachable, and no
   unreachable candidate may ever appear. This is the same logic as the G0
   sampling sanity check, on a model small enough to run in seconds.
"""
import torch
from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM

from mb5.reachability import score_items
from mb5.samplers import build_chain

V, L_ANS = 48, 3


def tiny_model():
    torch.manual_seed(1)
    cfg = LlamaConfig(vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64,
                      pad_token_id=0, bos_token_id=1, eos_token_id=2)
    m = LlamaForCausalLM(cfg).eval()
    # sharpen the output so truncation actually removes tokens
    with torch.no_grad():
        m.lm_head.weight.mul_(8.0)
    # No EOS anywhere: otherwise generate() would add processors (e.g. min_new_tokens
    # masking EOS) that renormalise the distribution before our chain. That exact
    # trap is why run_model.py records any extra processors a generation_config adds.
    m.config.eos_token_id = None
    m.generation_config.eos_token_id = None
    return m


def sample(model, prompt, n, spec):
    gc = GenerationConfig(do_sample=True, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0,
                          max_new_tokens=L_ANS, pad_token_id=0, eos_token_id=None)
    inp = torch.tensor([prompt] * n)
    out = model.generate(input_ids=inp, attention_mask=torch.ones_like(inp), generation_config=gc,
                         logits_processor=build_chain(spec))
    return [tuple(x) for x in out[:, len(prompt):].tolist()]


def test_reachability_matches_sampling():
    model = tiny_model()
    prompt = [1, 7, 11, 13, 5]
    torch.manual_seed(2)
    candidates = sorted(set(sample(model, prompt, 600, {})))  # untruncated: a wide candidate set
    specs = {"p0.5": {"top_p": 0.5}, "k3": {"top_k": 3},
             "shipped_like": {"repetition_penalty": 1.05, "temperature": 0.7, "top_k": 5, "top_p": 0.8}}
    items = [{"prompt_ids": prompt, "answer_ids": list(c), "cid": i} for i, c in enumerate(candidates)]
    recs = score_items(model, items, specs, device="cpu", batch_size=16, pad_id=0, progress=False)
    reach = {tuple(it["answer_ids"]): r for it, r in zip(items, recs)}
    assert any(not r["reach__p0.5"] for r in recs), "test is vacuous: nothing truncated"
    for name, spec in specs.items():
        torch.manual_seed(3)
        drawn = sample(model, prompt, 1500, spec)
        for d in drawn:
            if d in reach:
                assert reach[d][f"reach__{name}"], (name, d, reach[d])
        # analytic p* must agree with the exact chain for pure top-p
        if name == "p0.5":
            for r in recs:
                if abs(r["p_star_T1.0"] - 0.5) > 1e-5:
                    assert r["reach__p0.5"] == (r["p_star_T1.0"] < 0.5)
        if name == "k3":
            for r in recs:
                assert r["reach__k3"] == (r["k_star"] <= 3)


def test_margin_classification_is_ordered():
    """robust_unreach implies exact-unreachable implies not robust_reach."""
    model = tiny_model()
    prompt = [1, 7, 11, 13, 5]
    torch.manual_seed(2)
    candidates = sorted(set(sample(model, prompt, 600, {})))
    items = [{"prompt_ids": prompt, "answer_ids": list(c)} for c in candidates]
    specs = {"p0.5": {"top_p": 0.5}, "k3": {"top_k": 3}}
    recs = score_items(model, items, specs, device="cpu", batch_size=16, pad_id=0, progress=False, margin=0.5)
    for r in recs:
        for n in specs:
            if r[f"robust_unreach__{n}"]:
                assert not r[f"reach__{n}"]
            if r[f"robust_reach__{n}"]:
                assert r[f"reach__{n}"]
    assert any(r["robust_unreach__p0.5"] for r in recs) and any(r["robust_reach__p0.5"] for r in recs)


def test_first_token_relaxation_matches_sampling():
    """Everything the relaxed sampler draws is scored reachable under the
    relax_first verdict, and the relaxation rescues answers that the plain
    sampler cannot reach (otherwise the test is vacuous)."""
    from transformers import LogitsProcessorList

    from mb5.samplers import FirstTokenRelaxed

    model = tiny_model()
    prompt = [1, 7, 11, 13, 5]
    torch.manual_seed(2)
    candidates = sorted(set(sample(model, prompt, 600, {})))
    items = [{"prompt_ids": prompt, "answer_ids": list(c)} for c in candidates]
    specs = {"p0.5": {"top_p": 0.5}, "relax": {"top_p": 0.5, "relax_first": True}}
    recs = score_items(model, items, specs, device="cpu", batch_size=16, pad_id=0, progress=False, margin=0.3)
    reach = {tuple(it["answer_ids"]): r for it, r in zip(items, recs)}
    assert any(r["reach__relax"] and not r["reach__p0.5"] for r in recs), "relaxation rescues nothing"
    assert all(r["reach__relax"] or not r["reach__p0.5"] for r in recs)  # relaxing never removes reach
    gc = GenerationConfig(do_sample=True, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0,
                          max_new_tokens=L_ANS, pad_token_id=0, eos_token_id=None)
    torch.manual_seed(4)
    inp = torch.tensor([prompt] * 1500)
    out = model.generate(input_ids=inp, attention_mask=torch.ones_like(inp), generation_config=gc,
                         logits_processor=LogitsProcessorList([FirstTokenRelaxed(specs["relax"], len(prompt))]))
    for d in [tuple(x) for x in out[:, len(prompt):].tolist()]:
        if d in reach:
            assert reach[d]["reach__relax"], d
