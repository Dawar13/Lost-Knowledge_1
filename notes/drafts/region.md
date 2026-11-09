# Region composition test, popularity controls, mechanism within models (fork B draft, 2026-11-09)

All three analyses are **post hoc**: chosen after seeing the G2 results and the internal review. None is
pre-registered and no gate depends on them. CPU only, from stored results; no GPU was used.

Code: `src/mb5/region.py`, `src/mb5/controls.py`; tests `tests/test_region.py`, `tests/test_controls.py`
(the full suite passes: 41 tests on 2026-11-09). Outputs: `results/analysis_cpu/region/` and
`results/analysis_cpu/controls/`, each with `manifest.json` (input sha256, code version, statsmodels
version; local statsmodels is 0.14.6). Reproduction checks: the raw RBU by region equals
`results/P3/g2/by_region.csv` for all five model and sampler pairs (`region/summary.json`, `checks`), and
step P0 of the popularity refit reproduces `results/P3/g2/popularity_gee.csv` (see part 2).

## 1. Region: the composition guess does not hold

**Question.** G2 found robust RBU at top-p 0.9 lower for South Asian, Southeast Asian and African categories
(about 86 to 87% for Qwen2.5-7B-Instruct) than for European and North American ones (96.8% and 99.5%). The
agent guessed "composition": Western categories have more recognised members competing for the same few
first pieces.

**Method.** One row per recognised (entity, template); outcome robustly unreachable; GEE logistic, binomial,
exchangeable within category (the mb5.g2 setup); 277 categories. Region dummies, Europe as reference.
The "gap" is the contrast mean(South Asia, Southeast Asia, Africa) minus mean(Europe, North America) on the
log-odds scale, reported as an odds ratio. Steps: M0 region only; Mc + log number of recognised members in
the category (composition); Mf + first-piece concentration of the category prompt (top-1 mass and log kept
pieces under top-p 0.9; stored in `results/P3/<model>/first_token.json` for all four models); Me + entity
covariates (log10(1 + sitelinks), within-category sitelink percentile, label token count, number of surface
forms); Mall all of these plus entity-type fixed effects.

**Result** (`region/gee_gap.csv`; odds ratio of unreachability, South against West, 95% CI):

| model, sampler | M0 | Mc (composition) | Mall (everything) | share of M0 gap left in Mall |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct, top-p 0.9 | 0.096 (0.043 to 0.214) | 0.112 (0.050 to 0.255) | 0.246 (0.111 to 0.549), p 0.0006 | 0.60 |
| Qwen2.5-7B-Instruct, shipped | 0.109 (0.050 to 0.236) | 0.140 (0.064 to 0.307) | 0.127 (0.051 to 0.315) | 0.93 |
| Qwen2.5-7B (base), top-p 0.9 | 0.409 (0.326 to 0.512) | 0.470 (0.381 to 0.580) | 0.431 (0.307 to 0.604) | 0.94 |
| OLMo-2-1124-7B-Instruct, top-p 0.9 | 0.098 (0.055 to 0.175) | 0.128 (0.070 to 0.232) | 0.150 (0.075 to 0.299) | 0.82 |
| OLMo-2-1124-7B (base), top-p 0.9 | 0.307 (0.230 to 0.409) | 0.374 (0.278 to 0.502) | 0.203 (0.129 to 0.321) | 1.35 |

- Composition is real but small. More recognised members per category does raise unreachability (Mc: odds
  ratio 1.40 per e-fold more recognised members for Qwen instruct, 95% CI 1.09 to 1.79; 1.41 to 1.79 across
  the five fits). But adding it leaves 83 to 93% of the gap (log-odds scale). The gap stays large and
  significant after every control in every model.
- Stratified view without a model (`region/stratified.csv`, Qwen instruct, top-p 0.9, quartile bins of
  recognised members per category): with 26 to 46 recognised members, West 98.5% (14 categories) against
  South 83.3% (41); with 47 to 78, 96.8% (16) against 83.1% (31); with 79 or more, 98.2% (17) against 96.3% (12).
  Only the smallest bin reverses (fewer than 26: West 79.5%, 6 categories; South 88.1%, 45). At equal
  category size, Southern categories still lose a smaller share.
- Caveat on "share of gap": logistic coefficients are not collapsible, so adding predictors can move the
  region coefficient even without confounding (it grows for OLMo base). Read the shares as rough; the
  stratified table is the model-free check.
- Estimation check: every exchangeable fit converged (within-category correlation 0.01 to 0.14,
  `gee_gap.csv` column `exchangeable_rho`). Sensitivity with the independence working correlation (same
  cluster-robust SEs): the gap after all controls is 0.145 (0.062 to 0.342) for Qwen instruct, 0.103 for Qwen
  shipped, 0.137 for OLMo instruct, 0.463 and 0.272 for the base models; all exclude 1. Same conclusion.

**What does differ** (`region/by_region.csv`, Qwen instruct, top-p 0.9):
- Southern prompts leave *more* names reachable in absolute terms, not only a smaller share: mean reachable
  recognised members per category 4.5 (South Asia), 7.6 (Southeast Asia), 4.2 (Africa) against 2.2 (Europe)
  and 0.3 (North America), although these categories have fewer recognised members (median 31, 54, 25
  against 59.5 and 44). Composition predicts similar reachable counts; we see more.
- The first-piece distribution is flatter for Southern prompts: median pieces kept under top-p 0.9 are 4, 3, 3
  against 3 and 2; mean top-1 mass 0.60, 0.62, 0.69 against 0.68 and 0.77. In the base model the contrast is
  larger: median kept pieces 233, 249, 332 against 124 and 165. A wider first shortlist predicts
  reachability strongly within the model (Mall: odds ratio 0.54 per e-fold more kept pieces, 0.48 to 0.60), but
  adding it explains only about 16% of the gap (Mf).
- The gap is already in the base models (M0 odds ratio 0.41 for Qwen base, 0.31 for OLMo base), so it is not
  created by post-training.
- Post-training removes a *smaller* share of base-reachable names in Southern categories
  (`region/paired_region.csv`, entity-template rows recognised by both models, P(instruct unreachable given
  base reachable)): Qwen 78.4% (South) against 92.7% (West); OLMo 71.6% against 87.3%.

**Alternatives checked** (all from stored data; none explains the gap):
- *Recognition rate.* Lower in South Asia (31.4%) and Africa (32.7%) than Europe (50.8%) for Qwen instruct
  (Southeast Asia 55.6%). This is the composition channel; see above.
- *Recognition noise.* Yes/no AUROC is lower in South Asia (0.78) and Africa (0.77) than Europe (0.85). Noisier
  recognition would add names the model does not know, which should be unreachable; that pushes RBU up, the
  opposite of the observed direction. It cannot explain a lower RBU.
- *Selection on popularity.* Recognised Southern members are relatively more popular: mean log10(1 + sitelinks)
  0.80 (South Asia), 0.92 (Africa) against 0.73 (Europe); mean within-category sitelink percentile 0.55 and 0.57
  against 0.51. Popularity strongly predicts reachability, so this explains part: entity covariates (Me)
  leave 74% of the gap for Qwen instruct and 83% for OLMo instruct, but none for the base models.
- *Name length.* Southern labels are slightly shorter (5.4 to 5.8 tokens against 6.2 for Europe), and more tokens
  raise unreachability (Mall: odds ratio 1.09 per token). North American labels are shortest (4.2) and have the
  highest RBU, so length does not track the regional pattern. Controlled in Me and Mall.
- *Aliases.* Southern entities more often have an alias (39 to 48% against 29% for Europe), and each extra surface
  form lowers unreachability (odds ratio 0.77 per form). Controlled in Me and Mall.
- *Entity type.* The language-writer categories are all South Asian; type fixed effects (Mall) do not remove
  the gap.

**Verdict.** The composition guess is rejected as the main explanation. After controlling for composition,
first-piece width, popularity (absolute and within category), name length, aliases and type, Southern
categories still have about a quarter of the Western odds of unreachability for Qwen instruct (odds ratio 0.25)
and about a seventh for OLMo instruct (0.15). The descriptive picture is that the models have weaker
favourites for non-Western categories, so more of these names fit on the shortlist; we do not know why, and
this analysis cannot tell (it may reflect training-data distribution, which we do not observe). The pattern
exists before post-training, and post-training removes a smaller share in these categories.

What it does **not** license: any claim that non-Western knowledge is safer. Fewer non-Western members are
recognised in the first place (31 to 33% against 51%), so the absolute number of recognised Southern names is
lower; the loss happens earlier, at recognition, which this paper does not measure.

### Draft paper paragraph (region)

> **Region.** Recognised entities from South Asian, Southeast Asian and African categories are less often
> unreachable than European and North American ones (Qwen2.5-7B-Instruct at top-p 0.9: about 86 to 87% against
> 97 and 99.5%). We did not predict this, and we tested one explanation after seeing it: Western categories have
> more recognised members competing for the same short list. That effect is real but small. With the number of
> recognised members, the first-token distribution, popularity, name length, aliases and entity type all
> controlled, the odds of unreachability for Southern categories remain about a quarter of the Western odds
> (odds ratio 0.25, 95% CI 0.11 to 0.55; OLMo-2-Instruct 0.15). The gap is already present in the base models, and
> post-training removes a smaller share of base-reachable names in these categories (78% against 93% for Qwen).
> The models simply have weaker favourites for these categories, so more names fit on the short list. This is not
> a sign that non-Western knowledge is safe: the models recognise fewer of these names to begin with (31 to 33% of
> members against 51% for Europe), a loss our test does not measure. We make no regional "who loses" claim.

## 2. Popularity headline with controls (review item 7)

**Question.** Does the G2 headline (odds ratio 0.149 per tenfold more sitelinks for Qwen instruct) survive
label token count and entity type? Rare names tend to be longer in tokens, so they have more chances to be cut.

**Method.** Same rows and model as mb5.g2 (recognised entity-template pairs, robust unreachability, GEE
logistic exchangeable within category, robust SEs), numeric covariates centred, started from the GLM fit.
P0 log10(1 + sitelinks) only; P1 + label tokens; P2 + type fixed effects; P3 + both; P4 = P3 + region fixed
effects. Independence working correlation reported for every step as a sensitivity.

**Result** (`controls/popularity_controls.csv`; odds ratio per tenfold more sitelinks, 95% CI):

| model, sampler | P0 (= G2) | P3: + label tokens + type | P4: + region | independence, P0 to P3 |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct, top-p 0.9 | 0.149 (0.122 to 0.183) | 0.141 (0.114 to 0.175) | 0.110 (0.084 to 0.143) | 0.183 to 0.158 |
| Qwen2.5-7B-Instruct, shipped | 0.019 (0.012 to 0.029) | 0.015 (0.009 to 0.024) | 0.009 (0.005 to 0.017) | 0.018 to 0.014 |
| Qwen2.5-7B (base), top-p 0.9 | 0.094 (0.081 to 0.110) | not converged | not converged | 0.115 to 0.104 (0.086 to 0.125) |
| OLMo-2-1124-7B-Instruct, top-p 0.9 | 0.103 (0.083 to 0.128) | 0.094 (0.074 to 0.121) | 0.068 (0.053 to 0.088) | 0.137 to 0.113 |
| OLMo-2-1124-7B (base), top-p 0.9 | 0.083 (0.070 to 0.098) | 0.079 (0.067 to 0.095) | not converged | 0.105 to 0.093 |

- P0 reproduces G2 exactly for all four models (column `g2_reference_or`).
- Label length matters a little: odds ratio 1.05 to 1.09 per extra token. Adding it moves the popularity odds
  ratio slightly towards 1 (0.149 to 0.158 for Qwen instruct, P1); adding type moves it away again.
- With length and type together (P3) the effect is as strong as or stronger than G2 in every model; with region
  added it is stronger still. The headline survives.
- Convergence: the exchangeable GEE did not give finite estimates for Qwen base with type fixed effects (P2 to
  P4) or OLMo base at P4 (the within-category correlation estimate drifts upward over iterations). These are
  recorded as not converged, not replaced. The independence fits, which converge, give the same picture.
- Shipped Qwen (odds ratio 0.019 at P0) is a new number: G2 reported top-p 0.9 only.

**Verdict.** The popularity headline does not depend on name length or entity type. The paper can say the
odds ratio is 0.14 (0.11 to 0.18) for Qwen instruct with both controls, and 0.08 to 0.14 at top-p 0.9 across
the models where that model converges (Qwen base: independence fit 0.10).

### Draft paper sentence (popularity)

> The popularity effect is not an artefact of name length or entity type: with the label's token count and
> entity-type fixed effects added, the odds of a recognised name being unreachable fall by a factor of 0.14
> (95% CI 0.11 to 0.18) per tenfold increase in Wikipedia sitelinks for Qwen2.5-7B-Instruct at top-p 0.9 (0.15
> without controls), and by 0.08 to 0.09 for the OLMo-2 models. Each extra token raises the odds by 5 to 9%.

## 3. Mechanism: does first-token concentration predict RBU within a model? (review item 6)

`results/G1/mechanism.csv` gives a pooled Spearman of 0.817 between the first-token top-1 mass of the category
prompt and robust RBU at top-p 0.9, over 66 points (11 pilot categories by six models: Qwen base and instruct,
OLMo base, SFT, DPO, instruct). Pooling mixes two things: base models have low mass and low RBU, chat models
high and high. Breakdown (`controls/mechanism_controls.csv`; 95% CIs by bootstrap over categories, paired across
models, 2,000 draws pilot, 1,000 at scale, seed 0):

| statistic | pilot (11 categories) | full TailCat (277 categories, 4 models) |
|---|---|---|
| pooled Spearman (as reported) | 0.817 (0.720 to 0.890) | 0.743 (0.714 to 0.771) |
| mean within-model Spearman (= pooled within-model ranks) | 0.629 (0.268 to 0.816) | 0.475 (0.416 to 0.533) |
| Pearson with a model fixed effect | 0.599 (0.460 to 0.746) | 0.442 (0.385 to 0.498) |
| OLS slope with model fixed effects (RBU per unit top-1 mass) | 0.672 (0.352 to 1.021) | 0.372 (0.323 to 0.422) |

Within each model:
- Pilot (11 categories each, so wide CIs): Qwen base 0.755 (0.261 to 0.972), Qwen instruct 0.791 (0.308 to 0.990),
  OLMo base 0.127 (-0.591 to 0.766), SFT 0.745 (0.139 to 1.000), DPO 0.655 (0.089 to 0.944), instruct 0.700
  (0.168 to 0.935).
- Full TailCat (277 categories each): Qwen base 0.337 (0.231 to 0.438), Qwen instruct 0.588 (0.509 to 0.663),
  OLMo base 0.455 (0.361 to 0.546), OLMo instruct 0.519 (0.426 to 0.600). All four exclude zero.

Within a category across the OLMo-2 stages (base, SFT, DPO, instruct; category fixed effect): Pearson 0.820
(0.708 to 0.935); mean within-category Spearman over the four stages 0.541 (0.109 to 0.882); 9 of 11 categories
positive.

**Verdict.** The association holds beyond the base-against-instruct gap, but it is weaker than 0.817
suggests. Within a model, categories whose prompt puts more mass on one first token lose more of their
recognised members (Spearman about 0.34 to 0.59 at scale, every CI above zero); within a category, the
stages that concentrate more lose more (Pearson 0.82 with category fixed effects). The pilot OLMo base model
alone shows no clear within-model association (0.13, CI spans zero). The paper should report the within-model
and within-category numbers next to, or instead of, the pooled 0.817, and describe the pooled figure as
mixing the between-model gap with the within-model association.

### Draft paper sentence (mechanism)

> First-token concentration predicts unreachability within a model, not only between base and chat models.
> Across the 277 categories, the Spearman correlation between the top-1 mass of the first answer token and RBU
> is 0.34 to 0.59 within each of the four models (all 95% CIs above zero), and across OLMo-2's training stages
> the correlation within a category is 0.82 (0.71 to 0.94). Pooling base and chat models gives 0.82 on the pilot,
> which mixes this association with the large gap between the two kinds of model.

## 4. Proposed macros (all from files written by this code; prefix `rg` region, `ct` controls)

| macro | value | source (file; row; column) |
|---|---|---|
| `\rgGapRawQwen` | 0.096 | region/gee_gap.csv; Qwen2.5-7B-Instruct, p0.90_T1, M0; gap_or |
| `\rgGapRawQwenCI` | 0.043 to 0.214 | same row; gap_or_ci_lo, gap_or_ci_hi |
| `\rgGapAdjQwen` | 0.246 | region/gee_gap.csv; Qwen2.5-7B-Instruct, p0.90_T1, Mall; gap_or |
| `\rgGapAdjQwenCI` | 0.11 to 0.55 | same row; gap_or_ci_lo, gap_or_ci_hi |
| `\rgGapAdjOlmo` | 0.150 | region/gee_gap.csv; OLMo-2-1124-7B-Instruct, p0.90_T1, Mall; gap_or |
| `\rgGapAdjQwenBase` | 0.431 | region/gee_gap.csv; Qwen2.5-7B, p0.90_T1, Mall; gap_or |
| `\rgCompositionOR` | 1.40 | region/gee_region_terms.csv; Qwen2.5-7B-Instruct, p0.90_T1, Mc, log_n_recognised; or |
| `\rgCompositionShareLeft` | 0.93 | region/gee_gap.csv; Qwen2.5-7B-Instruct, p0.90_T1, Mc; share_of_M0_gap |
| `\rgLossSouthQwen` | 78.4% | region/paired_region.csv; Qwen pair, SOUTH; loss_rate_given_base_reachable |
| `\rgLossWestQwen` | 92.7% | same file; Qwen pair, WEST |
| `\rgLossSouthOlmo` / `\rgLossWestOlmo` | 71.6% / 87.3% | same file; OLMo pair, SOUTH / WEST |
| `\rgRecRateEurope` | 50.8% | region/by_region.csv; Qwen2.5-7B-Instruct, p0.90_T1, europe; recognition_rate |
| `\rgRecRateSouthAsia` / `\rgRecRateAfrica` | 31.4% / 32.7% | same file; south_asia / africa |
| `\ctPopOrQwenControlled` | 0.141 | controls/popularity_controls.csv; Qwen2.5-7B-Instruct, p0.90_T1, P3; or |
| `\ctPopOrQwenControlledCI` | 0.114 to 0.175 | same row; or_ci_lo, or_ci_hi |
| `\ctPopOrOlmoControlled` | 0.094 | controls/popularity_controls.csv; OLMo-2-1124-7B-Instruct, p0.90_T1, P3; or |
| `\ctPopOrOlmoBaseControlled` | 0.079 | same file; OLMo-2-1124-7B, p0.90_T1, P3; or |
| `\ctLabelTokenOR` | 1.06 | controls/popularity_controls.csv; Qwen2.5-7B-Instruct, p0.90_T1, P3; label_tokens_or |
| `\ctMechWithinScaleMin` / `Max` | 0.34 / 0.59 | controls/mechanism_controls.csv; full TailCat, within Qwen2.5-7B / within Qwen2.5-7B-Instruct; value |
| `\ctMechFEScale` | 0.44 (0.39 to 0.50) | same file; full TailCat, Pearson of within-model demeaned values |
| `\ctMechStagesFE` | 0.82 (0.71 to 0.94) | same file; pilot_v2, OLMo-2 stages: Pearson with category fixed effects |
| `\ctMechWithinPilotMean` | 0.63 (0.27 to 0.82) | same file; pilot_v2, mean of the six within-model Spearman |

(Values here are rounded for reading; paper_numbers.py should read them from the CSVs, not from this table.)

## 5. Decisions entry (exact text for notes/decisions.md)

- 2026-11-09 (agent): **Post hoc analyses after G2 and internal review round 1 (no gate; CPU only; not
  pre-registered).** (1) *Region composition* (mb5.region, results/analysis_cpu/region): the agent's guess that
  Western categories lose a larger share because more recognised members compete for the same first pieces is
  rejected as the main explanation. More recognised members per category does raise unreachability (odds ratio
  1.40 per e-fold, Qwen instruct, top-p 0.9), but controlling for it leaves 93% of the South-West gap
  (log-odds), and with first-piece width, popularity, within-category percentile, label length, aliases and type
  also controlled the South-West odds ratio is 0.25 (95% CI 0.11 to 0.55; OLMo instruct 0.15; independence
  working correlation 0.15 and 0.14). The gap is present in the base models, post-training removes a smaller
  share of base-reachable names in Southern categories (Qwen 78% against 93%), and Southern prompts leave more
  names reachable in absolute terms. Paper: report the pattern, the failed explanation and the remaining gap;
  no regional "who loses" claim; note lower recognition rates there (31 to 33% against 51%). (2) *Popularity with
  controls* (mb5.controls, results/analysis_cpu/controls): adding label token count and entity-type fixed
  effects leaves the G2 headline intact (Qwen instruct 0.149 to 0.141, 95% CI 0.114 to 0.175; OLMo instruct 0.103
  to 0.094); exchangeable GEE did not converge for Qwen base with type effects and OLMo base with region effects,
  recorded as such, with independence fits reported instead. (3) *Mechanism within model*: the pooled Spearman
  0.817 (pilot, six models) mixes the base-chat gap with a within-model association; within models it is 0.63
  on the pilot (0.27 to 0.82) and 0.34 to 0.59 per model over the 277 full-TailCat categories (all CIs above 0);
  across OLMo-2 stages within a category, Pearson 0.82 (0.71 to 0.94). The paper reports the within-model and
  within-category values with the pooled one.
