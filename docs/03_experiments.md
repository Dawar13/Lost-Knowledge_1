# MB5 experiments

## Status at 2026-11-09 (results registry: results/INDEX.md; reasoning: notes/decisions.md)

| experiment | status | where |
|---|---|---|
| E0 instrument | done; numeric margin added (docs/04 s.2); 15 unit tests | tests/, results/dev_* |
| E1 TailCat | pilot_v2 (11 categories) and full_v1 (277 kept of 346 built) | data/tailcat/ |
| E2, E3, E4 G0 | done; signed under option B (V2 AUROC 0.835 < 0.85) | results/G0_5090, gates/G0*.md |
| E5, E6 G1 | done; H2 and H3 hold; signed | results/G1, gates/G1.md |
| E7 robustness | 5 wordings and temperature sweep on pilot_v2 (wording-robust) | results/P3_robust |
| E7 self-preamble, non-canonical audit at scale | preamble not run; non-canonical counts come from the P3 sample check | results/P3 |
| E8 multilingual | not run (no native-checked translations) | future work |
| E9 popularity, G2 | running (P3) | results/P3/g2, gates/G2.md |
| E10 estimators | pilot check done; at scale from P3 samples | results/P3/metrics/estimators_* |
| E11 mitigation, G3 | done; G3 FAIL (frontier reported); hand audit pending | results/P5, gates/G3.md |
| E12 closed models | not run | |
| scale ladder | queued after P3 | results/P3_scale |

Status key: **ready** = code in `src/mb5` runs it; **to build** = an agent writes it in that phase, reusing the existing modules. Every experiment writes to `results/<phase>/...` and appends to `logs/gpu_ledger.csv`.

## E0 Correctness of the instrument (P0, ready)

- `tests/test_semantics.py`: our analytic keep-rules (top-p via mass strictly above the target, top-k via rank, min-p via logit gap) equal transformers' warpers on random logits, including top-k followed by top-p and the left-padding trick for repetition penalty.
- `tests/test_end_to_end_tiny.py`: tiny random Llama; every answer sampled by `generate()` under a truncation spec was scored reachable, and no unreachable answer is ever sampled.
- Lesson already learnt while writing these tests: any extra processor in `generate()` (e.g. `min_new_tokens` masking EOS) renormalises the distribution before top-p and changes which tokens survive. `run_model.py` warns if a model's generation_config sets such a field.

## E1 TailCat (P0 pilot ready; P3 full to build)

- Pilot: `configs/pilot_categories.yaml`, 10 long-tail categories plus the head control `us_state_capitals`; members capped at 400 with stratification over Wikipedia sitelinks; 200 hard foils per category from sibling categories; `valid_names.json` keeps the uncapped name list for validity judgements.
- Full (P3): about 200 categories, 30 to 2,000 members, spread over people, places, works, organisations and species; at least 40% from South Asia, Africa, Southeast Asia and Latin America. Labels in en, hi, ta, bn via `--lang`. Native speakers translate and check the **prompt templates** (not names). Add Wikimedia pageviews as a second popularity measure.
- Output: `data/tailcat/<split>/{categories.json, entities.jsonl, foils.jsonl, valid_names.json, build_report.json}`.

## E2 Exact reachability (P1 ready)

- For every (entity, surface form, template): teacher-forced forward pass; per position: rank, gap, G at T=1 and T=0.7; per answer: p\*(T), k\*, min-p\*(T), bottleneck position; exact reachability and first failing position for every spec in `samplers.GRID` plus `shipped_hf` and `shipped_json`.
- Entity counts as reachable for a template if any surface form is.
- Output: `reach.jsonl.gz`.

## E3 Recognition (P1 ready)

- Yes/no V(e) with Yes/No token variants pooled; few-shot frame for base models.
- 4-way choice against three foils from sibling categories, positions shuffled.
- Recognised = V(e) at or above the category's 95th foil percentile AND 4-way correct.
- Validity: AUROC members against foils at least 0.85; at least 300 recognised in the pilot.
- Output: `recog.jsonl.gz`.

## E4 Sampling sanity check (P1 ready)

- 1,000 samples per category (template 0) under `shipped_hf` (real `generate()` with the model's own config) and `p0.90_T1` (our chain).
- Violations (unreachable entity produced with identical ids) must be zero.
- Non-canonical paths (valid name produced by different token ids while its canonical form is unreachable) are counted; if above 10% of valid samples, extend reachability to alternative tokenisations before P2.
- Output: `sample_check.json`.
- Dense companion (added P0, `mb5.selfcheck`): the entity-matched check above is nearly vacuous on a small model (smoke: 227 hits on 2 distinct entities). The self-check scores every distinct sampled continuation (up to and including EOS) under the spec that produced it; every one must come out reachable. Failures report the target's rank and G so bf16 boundary flips (batched teacher-forced forward against KV-cached decoding) can be told apart from logic bugs. Output: `results/selfcheck/<model>/selfcheck.json`.

## E5 Post-training stages (P2, run with existing code; multi-model analysis to build)

- OLMo-2-1124-7B base, SFT, DPO, Instruct on the pilot (3 templates).
- Llama-3.1-8B base and Instruct as the second family (Qwen already done in G0).
- Analysis to build: `mb5/analysis_multi.py` producing RBU by stage with cluster-bootstrap CIs, paired sign tests between adjacent stages, and the first-token concentration statistic (top-1 mass and entropy of the first answer token, per category) as the predictor of RBU.
- GPU: about 1.5 h.

## E6 Where the cut happens (P2, analysis to build)

- Histogram of first failing position among unreachable recognised entities, per spec and model.
- Split by script and tokenisation length (C3): do longer tokenisations fail more often after position 0?
- No extra GPU.

## E7 Robustness (P3, partly ready)

- Templates: 5 wordings (`--templates 5`); report RBU-mean, RBU-all and the coefficient of variation.
- Aliases: on by default; report label-only versus any-form.
- Self-preamble context (to build): generate the model's greedy answer for each prompt, take the text before the first matched entity (e.g. "Sure! One example is"), append it to the prompt and re-score. This tests C4.
- Temperature: T in {0.5, 0.7, 1.0} with top-p 0.9.
- Non-canonical tokenisation audit from E4 at full scale.

## E8 Multilingual (P3, to build on existing modules)

- Same entities, prompts in hi, ta, bn (native-checked), answers in the label of that language if Wikidata has one, else the English label.
- Compare RBU per language on the same entity set; mixed-effects logistic regression: unreachable ~ language + log sitelinks + tokens per name + (1 | category).
- Framing gate G2: RBU(Indian language) at least 1.5 times RBU(English) makes this a headline.

## E9 Popularity (P3, analysis to build)

- Logistic regression of unreachable on log sitelinks (and log pageviews), with category random effects, for recognised entities.
- Compare the popularity slope of reachability with the slope of recognition: the claim "the tail is deleted, not unknown" needs reachability to fall with popularity faster than recognition does.

## E10 Coverage and estimator bias (P4, to build)

- 1,000 samples per category under shipped settings (vLLM allowed after a 1-category cross-check against HF `generate()` showing the same reachable set).
- Count distinct valid names using `valid_names.json`; fit Chao1 and coverage-based richness on the valid-name counts.
- Compare the estimates with |reachable and recognised| and |recognised|.
- Expected: estimates converge to the reachable set; their shortfall against recognised knowledge equals the RBU mass (docs/04, section 4).

## E11 Mitigation frontier (P5, to build)

- Samplers: top-p {0.8, 0.9, 0.95, 0.99}, top-k {20, 50, 100}, min-p {0.02, 0.05, 0.1}, epsilon and eta sampling, and **first-token relaxation** (a LogitsProcessor that skips truncation at the first answer token and applies the shipped chain afterwards).
- For each: distinct valid names per 1,000 samples, invented-name rate (not in `valid_names.json`; hand-audit 200 to measure how many "invented" names are real but missing from Wikidata), and RBU.
- Plot the frontier: distinct valid names against invented-name rate.
- Gate G3: first-token relaxation recovers at least 50% of unreachable recognised entities with at most +2 points invented-name rate.

## E12 Closed models (optional, P3)

API models do not expose full distributions, so exact reachability is impossible. Report only sampling coverage at their default settings on 20 categories, clearly labelled as a lower-bound comparison. Skip if budget is tight.

## Compute plan (GPU-hours on one 80 GB card)

| Item | GPU-h |
|---|---|
| G0 pilot (2 models, about 3,800 entities, 3 templates, sampling check) | 0.5 |
| E5 stages and second family (6 models) | 1.5 |
| E7 robustness on 2 models | 1.0 |
| Full TailCat x 8 models x 3 templates (about 60k entities) | 4.0 |
| E8 three extra languages on 2 models | 1.0 |
| E10 coverage sampling | 1.5 |
| E11 mitigation sweep | 2.0 |
| **Total** | **11.5** (reserve 3.5) |
