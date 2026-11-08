"""P5 / gate G3: the mitigation frontier (E11).

For one post-trained model on one TailCat split (template 0), every sampler in
the sweep gets (a) exact reachability of every member (RBU over recognised
entities, robust and exact) and (b) real samples, scored for distinct valid
members and the invented-name rate. The sweep includes first-token
relaxation: the reference sampler with no truncation at the first answer token.

  python -m mb5.frontier --model Qwen/Qwen2.5-7B-Instruct --data data/tailcat/pilot_v2 \
      --recog results/G0_5090/Qwen2.5-7B-Instruct/recog.jsonl.gz --reference shipped_hf \
      --margin 0.75 --out results/P5

Answer classes (normalised text, also tried: text before the first comma or " (", without "the "):
  member    matches a sampled member's label or alias (the RBU population)
  valid     matches any member of the category in valid_names.json (uncapped Wikidata list)
  foil      matches a foil of the category (a near miss)
  invented  non-empty and none of the above (a name the category does not contain, or other text)
  empty     nothing after normalisation
Invented-name rate = invented / non-empty. A hand audit of 200 invented answers (docs/02) is pending.
G3 (notes/decisions.md): relaxation recovers at least 50% of the reference sampler's unreachable
recognised entities (robust RBU) with the invented-name rate at most 2 points above the reference.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from .analysis import recognised_set
from .common import (append_gpu_ledger, estimate_need_gib, is_chat_model, load_model, model_tag, peak_gib, read_jsonl,
                     require_gpu_memory, run_meta, seed_all, write_json, write_jsonl)
from .metrics import write_csv
from .prompts import BASE_GEN_SHOTS, GEN_TEMPLATES, answer_prefix, encode_pair, encode_prompt, render
from .reachability import score_items
from .sample_check import generate_samples, norm
from .samplers import shipped_specs

SWEEP = {
    "untruncated_T1": {},
    "p0.80_T1": {"top_p": 0.80}, "p0.90_T1": {"top_p": 0.90}, "p0.95_T1": {"top_p": 0.95},
    "p0.99_T1": {"top_p": 0.99},
    "k20_T1": {"top_k": 20}, "k50_T1": {"top_k": 50}, "k100_T1": {"top_k": 100},
    "minp0.02_T1": {"min_p": 0.02}, "minp0.05_T1": {"min_p": 0.05}, "minp0.10_T1": {"min_p": 0.10},
    "eps3e-4_T1": {"epsilon": 3e-4},
}


def _candidates(t: str) -> list[str]:
    c = [t, t.split(",")[0].strip(), t.split(" (")[0].strip()]
    return [x for x in dict.fromkeys(c + [x.removeprefix("the ") for x in c]) if x]


def classify(text: str, members: dict[str, str], valid: set[str], foils: set[str]) -> tuple[str, str | None]:
    t = norm(text)
    if not t:
        return "empty", None
    cands = _candidates(t)
    for x in cands:
        if x in members:
            return "member", members[x]
    if any(x in valid for x in cands):
        return "valid", None
    if any(x in foils for x in cands):
        return "foil", None
    return "invented", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--recog", required=True, help="recog.jsonl.gz of the same model and split (a-priori probe)")
    ap.add_argument("--reference", required=True, help="reference sampler: shipped_hf or a SWEEP name")
    ap.add_argument("--margin", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--template", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_all(args.seed)
    require_gpu_memory(estimate_need_gib(args.model, args.batch_size))
    t0 = time.time()
    chat = is_chat_model(args.model)
    model, tok = load_model(args.model)
    tag = model_tag(args.model)
    out = Path(args.out) / tag
    data = Path(args.data)
    cats = {c["id"]: c for c in json.loads((data / "categories.json").read_text())}
    ents = read_jsonl(data / "entities.jsonl")
    foils = read_jsonl(data / "foils.jsonl")
    valid_all = {c: {norm(n) for n in names}
                 for c, names in json.loads((data / "valid_names.json").read_text(encoding="utf-8")).items()}

    specs = dict(SWEEP)
    ref = shipped_specs(model.generation_config).get("shipped_hf") if args.reference == "shipped_hf" \
        else SWEEP[args.reference]
    if ref is None:
        raise SystemExit(f"{args.model} ships no sampler; choose a SWEEP reference")
    specs["reference"] = dict(ref)
    specs["relax_first"] = dict(ref) | {"relax_first": True}

    # ---- exact reachability, template 0, every member and surface form
    tpl = GEN_TEMPLATES[args.template]
    shots = "" if chat else BASE_GEN_SHOTS
    pre = answer_prefix(chat)
    prompts = {cid: render(tok, tpl.format(singular=c["singular"], plural=c["plural"]), chat, shots)
               for cid, c in cats.items()}
    items = []
    for e in ents:
        for fi, form in enumerate([e["label"]] + e.get("aliases", [])):
            p_ids, a_ids, ok = encode_pair(tok, prompts[e["category"]], pre + form, chat)
            if a_ids:
                items.append({"qid": e["qid"], "category": e["category"], "form": form, "is_label": fi == 0,
                              "sitelinks": e["sitelinks"], "prompt_ids": p_ids, "answer_ids": a_ids})
    print(f"[frontier] scoring {len(items)} items under {len(specs)} samplers", flush=True)
    reach = score_items(model, items, specs, batch_size=args.batch_size, pad_id=tok.pad_token_id,
                        margin=args.margin, dstar=["reference", "relax_first"], keep_positions=False)
    write_jsonl(out / "reach.jsonl.gz", reach)
    rec, validity = recognised_set(read_jsonl(Path(args.recog)))

    def rbu(spec, mode):
        ent = defaultdict(bool)  # entity reachable if any form is
        for r in reach:
            k = r["qid"] + "|" + r["category"]
            ok = r[f"reach__{spec}"] if mode == "exact" else not r[f"robust_unreach__{spec}"]
            ent[k] = ent[k] or ok
        recd = [k for k in ent if rec.get(k)]
        return sum(not ent[k] for k in recd) / len(recd) if recd else float("nan"), len(recd)

    # ---- samples per sampler and category
    member_index = defaultdict(dict)
    for e in ents:
        for form in [e["label"]] + e.get("aliases", []):
            member_index[e["category"]][norm(form)] = e["qid"]
    foil_index = defaultdict(set)
    for f in foils:
        foil_index[f["category"]].add(norm(f["label"]))
    sample_rows, rows, by_cat = [], [], []
    for spec_name, spec in specs.items():
        cls_total = Counter()
        members_seen, valid_seen = set(), set()
        for cid in cats:
            p_ids = encode_prompt(tok, prompts[cid], chat)
            lens = [len(r["answer_ids"]) for r in reach if r["category"] == cid]
            max_new = min(32, max(lens) + 4) if lens else 16
            samples = generate_samples(model, p_ids, args.n_samples, max_new, "cuda", tok.pad_token_id, spec,
                                       seed=args.seed)
            cls = Counter()
            for i, s in enumerate(samples):
                text = tok.decode(s, skip_special_tokens=True)
                c, q = classify(text, member_index[cid], valid_all.get(cid, set()), foil_index[cid])
                cls[c] += 1
                if c == "member":
                    members_seen.add(q)
                if c in ("member", "valid"):
                    valid_seen.add((cid, norm(text)))
                sample_rows.append({"spec": spec_name, "category": cid, "i": i, "text": text, "class": c,
                                    "member_qid": q})
            nonempty = sum(cls.values()) - cls["empty"]
            by_cat.append({"spec": spec_name, "category": cid, **{k: cls[k] for k in
                           ("member", "valid", "foil", "invented", "empty")},
                           "invented_rate": cls["invented"] / nonempty if nonempty else None})
            cls_total += cls
        ne = sum(cls_total.values()) - cls_total["empty"]
        rb, n_rec = rbu(spec_name, "robust")
        re_, _ = rbu(spec_name, "exact")
        rows.append({"spec": spec_name, "definition": json.dumps(spec), "rbu_robust": rb, "rbu_exact": re_,
                     "n_recognised": n_rec, "distinct_members_sampled": len(members_seen),
                     "distinct_valid_answers": len(valid_seen),
                     "member_rate": cls_total["member"] / ne if ne else None,
                     "valid_rate": (cls_total["member"] + cls_total["valid"]) / ne if ne else None,
                     "foil_rate": cls_total["foil"] / ne if ne else None,
                     "invented_rate": cls_total["invented"] / ne if ne else None,
                     "empty_rate": cls_total["empty"] / sum(cls_total.values())})
        print(f"[frontier] {spec_name:14s} RBU {rb:.3f} distinct members {len(members_seen):4d} "
              f"invented {rows[-1]['invented_rate']:.3f}", flush=True)

    refrow = next(r for r in rows if r["spec"] == "reference")
    for r in rows:
        r["recovered_vs_reference"] = ((refrow["rbu_robust"] - r["rbu_robust"]) / refrow["rbu_robust"]
                                       if refrow["rbu_robust"] else None)
        r["invented_delta_vs_reference"] = r["invented_rate"] - refrow["invented_rate"]
    relax = next(r for r in rows if r["spec"] == "relax_first")
    g3 = {"recovered": relax["recovered_vs_reference"], "invented_delta": relax["invented_delta_vs_reference"],
          "passes": bool(relax["recovered_vs_reference"] >= 0.5 and relax["invented_delta_vs_reference"] <= 0.02),
          "reference": args.reference, "note": "invented-name rate is automatic; the 200-answer hand audit is pending"}
    secs = time.time() - t0
    write_csv(out / "frontier.csv", rows)
    write_csv(out / "frontier_by_category.csv", by_cat)
    write_jsonl(out / "samples_frontier.jsonl.gz", sample_rows)
    write_json(out / "frontier.json", {"g3": g3, "rows": rows, "recognition_validity":
                                       {k: v for k, v in validity.items() if k != "tau_per_category"},
                                       "meta": run_meta(model, args.model) | {"seconds": secs, "peak_gpu_gib": peak_gib(),
                                                                              "specs": specs, "args": vars(args)}})
    append_gpu_ledger("frontier", args.model, secs, note=f"{args.out} peak {peak_gib():.1f} GiB")
    print(f"[frontier] G3 for {tag}: recovered {g3['recovered']:.3f}, invented delta {g3['invented_delta']:+.3f} -> "
          f"{'PASS' if g3['passes'] else 'FAIL'} ({secs / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
