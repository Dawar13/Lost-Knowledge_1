# MB5 phases and gates

Work runs in phases. Each phase ends at a gate with pre-registered rules. A
phase may not start until the previous gate's decision file exists in
`results/gates/` and a human has signed it. Times are wall-clock hours with
coding agents and the GPU-hours actually consumed on one shared 80 GB card.

Total GPU budget: **15 GPU-hours** (target 12, 3 in reserve). Log every GPU job
in `logs/gpu_ledger.csv` (run_model.py does this automatically).

| Phase | What | Wall clock | GPU-h | Gate |
|---|---|---|---|---|
| P0 | Setup, tests, TailCat pilot build, smoke test | 0.5 to 1 h | 0.05 | none (tests must pass) |
| P1 | Go/no-go pilot | 1 to 2 h | 0.5 | **G0** |
| P2 | Mechanism: post-training stages, first-token localisation, second family | 2 to 3 h | 2 | **G1** |
| P3 | Full TailCat, 4 languages, all families, robustness | 4 to 6 h | 6 | **G2** |
| P4 | Coverage and estimator bias | 1 to 2 h | 1.5 | none |
| P5 | Mitigation frontier | 1 to 2 h | 2 | **G3** |
| P6 | Figures, writing, internal review | rest | 0 | submission checklist |

## P0 Setup

1. `pip install -r requirements.txt`; set `HF_TOKEN` (Llama and Gemma are gated).
2. `PYTHONPATH=src python -m mb5.check_env`: note free memory; other people's jobs are on this card.
3. `PYTHONPATH=src python -m pytest -q tests`: must pass. These prove our keep-rules equal transformers' warpers.
4. `PYTHONPATH=src python -m mb5.tailcat --config configs/pilot_categories.yaml --out data/tailcat/pilot`. Read the sanity report; fix any category under 30 members or foils.
5. `SMOKE=1 bash scripts/run_g0.sh` on Qwen2.5-0.5B-Instruct: end-to-end in minutes; zero violations required.

## P1 and gate G0: go/no-go

Run `bash scripts/run_g0.sh`. Models: Qwen2.5-7B-Instruct (with sampling sanity check) and Qwen2.5-7B base. Data: TailCat pilot (10 long-tail categories, 1 head control, up to 400 entities each, 3 prompt wordings, label plus up to 3 aliases). `analysis.py` writes `results/gates/G0.md` with the verdict computed by these rules:

**Validity (all must hold, otherwise BLOCKED: fix, do not interpret):**
- V1: zero sampling violations under both `shipped_hf` (real `generate()` with the model's config) and `p0.90_T1`. Any violation is a bug.
- V2: recognition AUROC (yes/no, members against foils) at least 0.85 for the instruct model.
- V3: at least 300 recognised entities.
- V4 (manual): the head-control category `us_state_capitals` has RBU under 2% at `p0.90_T1`. If not, suspect the pipeline.

**Verdict (RBU = mean over templates; CI = cluster bootstrap over categories):**
- **GO (strong):** RBU at `shipped_hf` at least 10% with CI lower bound at least 2% and RBU-all-templates at least half of RBU-mean; or RBU at `p0.90_T1` at least 5% with CI lower bound at least 2%.
- **GO:** RBU at least 2% with CI lower bound at least 1% under any of `shipped_hf`, `p0.90_T1`, `p0.95_T1`. The paper then needs G1 to carry it.
- **REFRAME:** the above fail but `k20_T1`, `k50_T1` or `minp0.10_T1` gives RBU at least 5%. Reframe around the truncation that actually bites (top-k=50 is transformers' silent default).
- **KILL:** RBU under 2% everywhere. Knowledge collapse is not a decoding artefact for this model. Write `notes/postmortem.md`; offer the reachability tool to the sister paper (Price of Not Hallucinating) as its decoding arm.

**Read alongside the verdict:** instruct/base RBU ratio at `p0.90_T1`, the paired sign test, the share of failures at the first token, and the non-canonical path rate from the sample check (if non-canonical hits exceed 10% of valid samples, canonical tokenisation overstates RBU: extend scoring to alternative tokenisations before Phase 2).

Oral-grade signal: RBU at shipped defaults of 10% or more on recognised entities, with base clearly lower.

## P2 and gate G1: where does it come from?

Models: OLMo-2-1124-7B base, SFT, DPO, Instruct (same pilot data, 3 templates); add Llama-3.1-8B base and Instruct. About 2 GPU-hours.

Measure: RBU at `p0.90_T1`, `k50_T1` and shipped settings for each stage; failure position histogram; first-token G distribution for recognised tail entities.

Rules:
- **H2 (post-training) holds** if RBU(Instruct) is at least 1.5 times RBU(base) at `p0.90_T1` with paired sign test p < 0.01 in at least two families, and RBU is non-decreasing across OLMo stages except for one allowed dip.
- **H3 (first token) holds** if at least 60% of unreachable recognised entities fail at answer position 0.
- Both hold: the story is "alignment concentrates first-token mass; truncation deletes the tail". The fix in P5 targets the first token.
- H2 fails, G0 strong: the story is "decoding defaults alone delete recognised knowledge". Still a full paper; drop the alignment claim.
- H3 fails: the P5 fix switches from first-token relaxation to min-p or an entity-aware rule.

## P3 and gate G2: generality and who loses

Build full TailCat: about 200 categories, 30 to 2,000 members each, labels in English, Hindi, Tamil and Bengali (native speakers check the prompt translations, not the names). Run Qwen, Llama, Gemma and OLMo pairs, 5 templates, and the scale ladder if budget allows. Robustness: aliases, self-preamble context (prefix the model's own greedy opening words), temperature sweep, non-canonical tokenisation audit. Popularity: Wikipedia sitelinks and pageviews.

G2 is a framing gate, not a kill gate:
- If RBU in Indian-language prompts is at least 1.5 times English for the same entities, the multilingual result becomes a headline ("defaults erase non-English knowledge first").
- If RBU falls with popularity (logistic slope negative with p < 0.01), lead with "the long tail is deleted, not sampled less".
- If RBU is stable across five templates (coefficient of variation under 0.25), state it as wording-robust; otherwise report the spread honestly.

## P4 Coverage and estimator bias

Sample 1,000 answers per category under shipped settings; count distinct valid entities; fit Chao1 and coverage-based estimates; compare with |reachable recognised| and |recognised|. Expected: the estimator converges to the reachable set, and its shortfall against recognised knowledge equals the RBU mass. This is the "measurement correction" contribution.

## P5 and gate G3: the fix

Samplers compared: top-p, top-k, min-p, epsilon, eta, temperature sweep, and first-token relaxation (no truncation at the first content token, shipped truncation afterwards). For each: distinct valid entities, invented-name rate (generated names not in Wikidata membership; audit 200 by hand), and RBU.

- **G3 passes** if first-token relaxation recovers at least 50% of unreachable recognised entities while the invented-name rate rises by at most 2 points. Then it is the paper's recommended default.
- If it fails, report the frontier as the result (no single dominant setting) and keep the diagnostic as the contribution.

## P6 Writing

Paper outline: docs/06_paper_outline.md. Before submission: every number traces to a file in `results/`; every citation verified on arXiv or the ACL Anthology; collision search re-run (arXiv, OpenReview ICLR 2027, ACL Anthology) in the final 48 hours; decide arXiv or anonymous (non-publicised award slots); Responsible NLP checklist; limitations section.
