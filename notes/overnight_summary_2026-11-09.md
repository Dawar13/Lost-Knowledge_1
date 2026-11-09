# Overnight summary, 8 to 9 November 2026

Written for the human on waking. Full reasoning: `notes/decisions.md`. Every run: `results/INDEX.md`.

## What was done

| phase | result | decided by |
|---|---|---|
| G0 | BLOCKED on V2 twice (AUROC 0.835 and 0.817 < 0.85); closed under option B | human (sign-off name still needed) |
| G1 | H2 and H3 hold (Qwen and OLMo-2 stages) | human signed |
| P3 robustness | wording-robust (CV 0.009 to 0.106 over 5 wordings); lower temperature raises RBU | rule |
| G3 (P5) | FAIL: first-token relaxation recovers 12.3% (Qwen) and 23.7% (OLMo) | agent, under your delegation |
| G2 (P3, full TailCat) | popularity headline: odds ratio 0.15 per tenfold sitelinks (Qwen instruct, p 6e-77) | agent, under your delegation |
| P4 | Qwen instruct, shipped sampler: Chao1 141 vs 15,514 recognised | analysis |
| scale ladder | RBU stays high from 1.5B to 7B | exploratory |

Headline numbers (robust RBU, top-p 0.9, full TailCat, 277 categories, 33,437 entities):
Qwen2.5-7B-Instruct 91.9% (95% CI 90.4 to 93.4) against base 55.0%; OLMo-2-Instruct 91.2% against base 62.7%;
Qwen shipped sampler 98.8%. V1: 0 violations in 831,000 samples at scale.

## Surprises you should know about

1. **Region:** RBU is *lower* for South Asian, African and Southeast Asian categories than for Western ones. Untested hypothesis: composition. No regional "who loses" claim is made; the who-loses result is popularity.
2. **The fix failed** because names that fail at the first token usually fail again later under the same sampler. No standard sampler reaches 50% recovery within +2 points of invented answers.
3. **Lost in Sampling (arXiv 2605.27268)** already uses the same exact token-survival test and a base vs aligned comparison for words. The paper is repositioned: our novelty is recognised knowledge, the first-token mechanism across post-training stages, the estimator correction, the numeric margin and who loses.
4. **Recognition probe** stays below the 0.85 bar for most models (0.80 to 0.85). Reported as a limitation.

## What needs you

1. **Sign-off names** for G0 and G1 (`results/gates/G0_signoff.md`, `G1_signoff.md`); review G2 and G3 (decided under delegation) and overturn anything you disagree with.
2. **Hand audit** of 200 "invented" answers: `results/P5/audit/invented_audit_sheet.csv` (fill `human_label`).
3. **Llama access** (Meta form on the model page) and an `HF_TOKEN`, or approve the Granite 3.1 8B replacement (`scripts/run_granite.sh`, ready, about 0.3 GPU-hours). Needs a GPU again: the vast.ai instance stopped at about 21:30 UTC (credit).
4. **Manual JSTOR check** of Chao (1984); collision search again in the final 48 hours.
5. **Native speakers** for Hindi, Tamil and Bengali prompt translations if the multilingual analysis (E8) is wanted.
6. **Paper:** `paper/main.pdf` is a complete draft (8 pages with appendix; every number generated from results). It still needs figures (RBU curves, failure positions, stages, rarefaction, frontier), a careful read, and the ARR checklist.

GPU used: 5.28 of 15 GPU-hours.
