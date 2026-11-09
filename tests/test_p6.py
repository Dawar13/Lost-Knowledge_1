"""P6 (E13 ask-directly control, E14 list control, E15 self-check filter): prompt construction,
filter logic on synthetic data, and the E13 and E14 scorers end to end on a tiny random Llama
(CPU, no download) with a character tokenizer."""
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from mb5.cues import CUE_INSTRUCTION, choose_cues, question, render_cue_prompt
from mb5.p6 import cue_items, distinct_unmatched, list_items, list_prefill, list_prompt
from mb5.p6_analysis import e13_summary, e14_summary, e15_summary, g3b_decision, greedy_set
from mb5.prompts import BASE_GEN_SHOTS, GEN_TEMPLATES, encode_pair, render
from mb5.reachability import score_items


class CharTok:
    """One token per character; prompt+answer tokenise as the concatenation, so the joint
    tokenisation always extends the prompt cleanly (boundary_ok)."""
    pad_token_id = 0
    CHARS = " abcdefghijklmnopqrstuvwxyz0123456789.?<>\n"  # ids 4..45; unknown 3

    def _id(self, ch):
        i = self.CHARS.find(ch.lower())
        return 4 + i if i >= 0 else 3

    def __call__(self, text, add_special_tokens=True):
        return SimpleNamespace(input_ids=([1] if add_special_tokens else []) + [self._id(c) for c in text])

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        s = "".join(f"<{m['role']}>{m['content']}\n" for m in msgs)
        return s + ("<assistant>" if add_generation_prompt else "")


def tiny_model():
    torch.manual_seed(1)
    cfg = LlamaConfig(vocab_size=48, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=512,
                      pad_token_id=0, bos_token_id=1, eos_token_id=2)
    m = LlamaForCausalLM(cfg).eval()
    with torch.no_grad():
        m.lm_head.weight.mul_(8.0)  # sharpen so truncation removes tokens
    return m


# ---------------------------------------------------------------- prompts
def test_cue_prompt_rendering():
    tok = CharTok()
    q = question("capital", "Vermont")
    assert q == "What is the capital of Vermont?"
    assert question("director", "Pather Panchali") == "Who directed the Bengali film Pather Panchali?"
    assert question("work", "Tiruppavai") == "Who wrote Tiruppavai?"
    chat = render_cue_prompt(tok, q, chat=True)
    assert chat == f"<user>{q} {CUE_INSTRUCTION}\n<assistant>"
    assert CUE_INSTRUCTION in GEN_TEMPLATES[0]  # same instruction as the open prompt
    base = render_cue_prompt(tok, q, chat=False)
    assert base == f"{BASE_GEN_SHOTS}Question: {q} {CUE_INSTRUCTION}\nAnswer:"


def test_choose_cues_rules():
    e = {"label": "Satyajit Ray", "aliases": ["Ray"]}
    row = lambda cue, n, lab, sl: {"cue": cue, "nAnswers": str(n), "cueLabel": lab, "cueSitelinks": str(sl)}  # noqa: E731
    assert choose_cues(e, [])[1] == "no_candidate"
    assert choose_cues(e, [row("Q1", 2, "A film", 9)])[1] == "not_unique"
    assert choose_cues(e, [{"cue": "Q1", "nAnswers": "1", "cueSitelinks": "3"}])[1] == "no_english_label"
    assert choose_cues(e, [row("Q1", 1, "The Satyajit Ray Story", 9)])[1] == "cue_contains_answer"
    best, why, counts = choose_cues(e, [row("Q9", 1, "Film B", 5), row("Q3", 1, "Film A", 5),
                                        row("Q2", 1, "Film C", 4), row("Q7", 3, "Film D", 50)])
    assert why is None and best["cue"] == "Q3"  # most sitelinks among unique cues, then lowest QID
    assert counts == {"n_candidate_cues": 4, "n_unique_cues": 3}


def _samples():
    S = []
    add = lambda spec, cat, text, cls, q=None, n=1: S.extend(  # noqa: E731
        {"spec": spec, "category": cat, "text": text, "class": cls, "member_qid": q} for _ in range(n))
    add("reference", "caps", "abo", "member", "Q1", 5)
    add("reference", "caps", "Abo.", "member", "Q1", 1)  # another spelling of the same member
    add("reference", "caps", "bac", "member", "Q2", 3)
    add("reference", "caps", "xyz", "valid", None, 4)
    add("reference", "caps", "qqq", "invented", None, 2)
    add("untruncated_T1", "caps", "qqq", "invented", None, 1)
    add("untruncated_T1", "caps", "", "empty", None, 1)
    return S


def test_list_prefill_and_prompt():
    tok = CharTok()
    S = _samples()
    pf = list_prefill(S, "reference", 5, "member")
    assert [p["qid"] for p in pf["caps"]] == ["Q1", "Q2"]  # pre-registered rule: members only, by frequency
    assert pf["caps"][0]["text"] == "abo" and pf["caps"][0]["count"] == 6
    pf2 = list_prefill(S, "reference", 5, "member_or_valid")
    assert [p["key"] for p in pf2["caps"]] == ["Q1", "valid:xyz", "Q2"] and pf2["caps"][1]["qid"] is None
    p = list_prompt(tok, True, "capitals", pf["caps"])
    assert p == "<user>List ten capitals, one per line, names only.\n<assistant>1. abo\n2. bac\n3."


def test_list_items_score_the_target_after_the_prefill():
    tok = CharTok()
    cats = {"caps": {"id": "caps", "singular": "a capital", "plural": "capitals"}}
    ents = [{"qid": f"Q{i}", "category": "caps", "label": lab, "aliases": al, "sitelinks": 10 * i}
            for i, (lab, al) in enumerate([("abo", []), ("bac", []), ("cab", ["kab"]), ("dab", [])], start=1)]
    pf = list_prefill(_samples(), "reference", 5, "member")
    items, prompts, excluded = list_items(tok, True, cats, ents, pf)
    assert excluded == {"caps": ["Q1", "Q2"]}
    assert {it["qid"] for it in items} == {"Q3", "Q4"}  # pre-filled answers are not scored
    for it in items:
        assert it["prompt_ids"] == tok(prompts["caps"], add_special_tokens=False).input_ids
        assert it["answer_ids"] == tok(" " + it["form"], add_special_tokens=False).input_ids
        assert it["boundary_ok"]
    assert len(items) == 3  # cab, its alias kab, dab


def test_distinct_unmatched():
    qs = distinct_unmatched(_samples())
    keys = [(q["category"], q["key"]) for q in qs]
    assert keys == [("caps", "qqq"), ("caps", "xyz")]  # members and empty answers excluded
    assert qs[0]["n_by_spec"] == {"reference": 2, "untruncated_T1": 1}


# ---------------------------------------------------------------- E15 filter on synthetic data
def _e15_data():
    recog = [{"kind": "foil", "category": "c", "V": i / 100, "qid": f"F{i}"} for i in range(30)]
    for q, v in [("m1", 0.9), ("m2", 0.8), ("m3", 0.1), ("m4", 0.95)]:
        recog.append({"kind": "member", "category": "c", "qid": q, "V": v, "mc_correct": 1})
    # P5 verdicts: reference reaches m1 only; untruncated everything; k100 all but m4
    reach = []
    for q in ("m1", "m2", "m3", "m4"):
        r = {"qid": q, "category": "c"}
        for s, ok in [("reference", q == "m1"), ("untruncated_T1", True), ("k100_T1", q != "m4")]:
            r[f"reach__{s}"] = ok
            r[f"robust_unreach__{s}"] = not ok
        reach.append(r)
    S = []
    add = lambda spec, text, cls, q=None, n=1: S.extend(  # noqa: E731
        {"spec": spec, "category": "c", "text": text, "class": cls, "member_qid": q} for _ in range(n))
    add("reference", "m1", "member", "m1", 6)
    add("reference", "zzz", "invented", None, 4)
    add("untruncated_T1", "m1", "member", "m1", 4)
    add("untruncated_T1", "m2", "member", "m2", 2)
    add("untruncated_T1", "m3", "member", "m3", 1)  # a member the model does not endorse: removed
    add("untruncated_T1", "yyy", "invented", None, 1)  # an invention the model endorses: kept
    add("untruncated_T1", "zzz", "invented", None, 2)
    add("k100_T1", "m1", "member", "m1", 5)
    add("k100_T1", "zzz", "invented", None, 5)
    scores = [{"category": "c", "key": "yyy", "V": 0.9}, {"category": "c", "key": "zzz", "V": 0.1}]
    return S, recog, scores, reach


def test_selfcheck_filter_logic():
    S, recog, scores, reach = _e15_data()
    r = e15_summary(S, recog, scores, reach)
    assert r["rbu_reference"] == pytest.approx(2 / 3)  # recognised m1, m2, m4; reference reaches m1
    assert r["reference_unfiltered"]["invented_rate"] == pytest.approx(0.4)
    u = r["quantiles"]["0.95"]["rows"]["untruncated_T1"]
    assert u["unfiltered"]["invented_rate"] == pytest.approx(3 / 10)
    assert u["filtered"]["n_kept"] == 7 and u["filtered"]["invented_rate"] == pytest.approx(1 / 7)
    assert u["invented_delta_vs_reference"] == pytest.approx(1 / 7 - 0.4)
    assert u["recovery"] == pytest.approx(1.0) and u["passes"]
    assert u["member_answers_removed"] == 1 and u["distinct_members_removed"] == 1
    k = r["quantiles"]["0.95"]["rows"]["k100_T1"]
    assert k["recovery"] == pytest.approx(0.5) and k["filtered"]["invented_rate"] == 0 and k["passes"]
    assert set(r["quantiles"]) == {"0.95", "0.9", "0.99"}
    d = g3b_decision({"A": r, "B": r})
    assert d["passes"] and d["recommended"] == ["untruncated_T1", "k100_T1"]


def test_selfcheck_filter_fails_on_invented_rate_and_needs_scores():
    S, recog, scores, reach = _e15_data()
    scores = [{"category": "c", "key": "yyy", "V": 0.9}, {"category": "c", "key": "zzz", "V": 0.99}]
    r = e15_summary(S, recog, scores, reach)  # the model endorses its own inventions: nothing removed
    u = r["quantiles"]["0.95"]["rows"]["untruncated_T1"]
    assert u["filtered"]["invented_rate"] == pytest.approx(3 / 9) and u["invented_delta_vs_reference"] < 0.02
    # raise the reference's precision so the wide sampler now loses on invented names
    S2 = [x for x in S if not (x["spec"] == "reference" and x["class"] == "invented")]
    r2 = e15_summary(S2, recog, scores, reach)
    assert not r2["quantiles"]["0.95"]["rows"]["untruncated_T1"]["passes"]
    with pytest.raises(KeyError):
        e15_summary(S, recog, scores[:1], reach)


# ---------------------------------------------------------------- E13 and E14 end to end (tiny model)
def test_e13_e14_end_to_end_tiny():
    tok, model = CharTok(), tiny_model()
    cats = {"caps": {"id": "caps", "singular": "a capital", "plural": "capitals"}}
    labels = ["abo", "bac", "cab", "dab", "eba", "fab", "gab", "hab"]
    ents = [{"qid": f"Q{i}", "category": "caps", "label": lab, "aliases": [], "sitelinks": i}
            for i, lab in enumerate(labels, start=1)]
    specs = {"p0.90_T1": {"top_p": 0.9}, "p0.50_T1": {"top_p": 0.5}}
    kw = dict(device="cpu", batch_size=8, pad_id=0, progress=False, margin=0.1, dstar=False, keep_positions=True)
    # open prompt, template 0
    open_p = render(tok, GEN_TEMPLATES[0].format(singular="a capital", plural="capitals"), True)
    items = []
    for e in ents:
        p_ids, a_ids, ok = encode_pair(tok, open_p, e["label"], True)
        items.append({"qid": e["qid"], "category": "caps", "template": 0, "form": e["label"], "is_label": True,
                      "sitelinks": e["sitelinks"], "prompt_ids": p_ids, "answer_ids": a_ids})
    open_rows = score_items(model, items, specs, **kw)
    recog = [{"kind": "foil", "category": "caps", "qid": f"F{i}", "V": 0.0} for i in range(30)]
    recog += [{"kind": "member", "category": "caps", "qid": e["qid"], "V": 1.0, "mc_correct": 1} for e in ents]
    # E13
    cues = [{"qid": e["qid"], "category": "caps", "cue_qid": f"S{i}", "question": f"What is the capital of s{i}?"}
            for i, e in enumerate(ents)]
    citems, cprompts = cue_items(tok, True, cues, {e["qid"] + "|caps": e for e in ents})
    assert len(citems) == len(ents) and len(cprompts) == len(ents)
    for it in citems:
        assert it["answer_ids"] == tok(it["form"], add_special_tokens=False).input_ids
    cue_rows = score_items(model, citems, specs, **kw)
    r13 = e13_summary(open_rows, recog, cue_rows, list(specs))
    g = greedy_set(cue_rows)
    for s in specs:
        x = r13["specs"][s]
        unreach_open = [r["qid"] for r in open_rows if r[f"robust_unreach__{s}"]]
        assert x["population"] == len(unreach_open)
        exp = [r["qid"] for r in cue_rows if r["qid"] in unreach_open and r[f"robust_reach__{s}"]]
        if unreach_open:
            assert x["reachable_under_cue"] == pytest.approx(len(exp) / len(unreach_open))
            assert x["answered_greedily"] == pytest.approx(
                sum(g[q + "|caps"] for q in unreach_open) / len(unreach_open))
    assert r13["specs"]["p0.50_T1"]["population"] > 0, "test is vacuous: nothing unreachable at top-p 0.5"
    # E14
    S = [{"spec": "reference", "category": "caps", "text": "abo", "class": "member", "member_qid": "Q1"}] * 3 + \
        [{"spec": "reference", "category": "caps", "text": "bac", "class": "member", "member_qid": "Q2"}]
    pf = list_prefill(S)
    litems, lprompts, excluded = list_items(tok, True, cats, ents, pf)
    assert lprompts["caps"].endswith("<assistant>1. abo\n2. bac\n3.")
    list_rows = score_items(model, litems, specs, **kw)
    assert all(r["L"] == len(r["form"]) + 1 for r in list_rows)  # " " + name, nothing of the pre-fill
    r14 = e14_summary(open_rows, recog, list_rows, list(specs), base_rbu=0.0, prefill=pf)
    for s in specs:
        x = r14["specs"][s]
        assert x["n_recognised"] == len(ents) - 2
        lu = sum(r[f"robust_unreach__{s}"] for r in list_rows)
        assert x["rbu_list"] == pytest.approx(lu / (len(ents) - 2))
    assert r14["prefill_sizes"] == {"caps": 2}
