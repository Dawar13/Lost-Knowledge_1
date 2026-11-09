# Draft: zero versus tiny (reviewer attack 3)

Source of every number: `results/analysis_cpu/support/` (code `src/mb5/support.py`; inputs and sha256 in
`manifest.json`). Main setting: full TailCat (results/P3), Qwen2.5-7B-Instruct, top-p 0.9 at T 1, template 0,
recognised entities only. "Untruncated" means the same sampler with truncation removed (T 1 only). Its
per-sample probabilities are exact functions of the stored log-probabilities (no repetition penalty is
involved at T 1 without truncation). Shipped-sampler numbers use T 0.7 without Qwen's repetition penalty 1.05
and are approximate; they are marked as such.

## Draft paragraph (Section 6 or the appendix)

**Zero versus tiny.** Truncation sets a deleted name's probability to exactly zero. Without truncation it
would be small but positive. How small? For each recognised entity we compute q, the probability that one
untruncated sample begins with the entity's name, from the stored teacher-forced log-probabilities. These
values are well calibrated. On the pilot, 11,000 untruncated samples per model contain 3,466 member answers
for Qwen2.5-7B-Instruct against 3,653 predicted, and 4,042 against 4,069 for OLMo-2-Instruct. The models
produce 75 and 78 distinct members against 77.0 and 83.5 predicted. For most deleted names the objection
holds. The median recognised name that top-p 0.9 deletes has q = 1.4e-11 under Qwen2.5-7B-Instruct. Seeing it
once would take about 7e10 samples. Only 1.3% of deleted names would appear in 1,000 untruncated samples, and
12.3% in a million. Truncation turns "astronomically unlikely" into "impossible". Most of the burial happens
before truncation, in the distribution itself. The same median is 2.1e-9 for the base model, about 150 times
larger. This is the concentration that post-training adds (Section 5). Truncation still sets a hard ceiling.
With top-p 0.9 the expected number of distinct recognised names found can never exceed the reachable set:
1,252 of 15,514 recognised names, over 277 prompts. At 1,000 samples per prompt, truncated sampling finds
between 284 and 417 of them (bounds). Untruncated sampling finds 575. At a million samples per prompt the
truncated count stays at or below 863, while untruncated sampling reaches 2,587. It passes the truncated
ceiling after about 24,000 samples per prompt. So the practical cost of truncation is a ceiling on discovery
that more sampling cannot lift. The deeper cause, a distribution that buries most recognised names at
probabilities near 1e-11, is not fixed by removing truncation either.

## Things this weakens (state them in the paper)

1. **Per name, the reviewer is mostly right.** The median deleted name is practically unsampleable even
   without truncation (Qwen-Instruct 1.4e-11; OLMo-Instruct 2.9e-10; scale, template 0). Wording such as
   "truncation deletes knowledge the model would otherwise express" overstates it for the median name. Safer:
   "post-training buries the long tail; truncation seals it".
2. **The estimator claim needs one sentence of care.** Unseen-species estimators applied to truncated samples
   estimate the reachable set, as the paper says. But untruncated samples of practical size would not
   recover the recognised set either: 2,587 of 15,514 after a million samples per prompt. P4's 141 versus
   15,514 gap is mostly concentration, partly truncation.
3. **The probability mass that truncation removes from recognised names is small:** 0.23% of untruncated
   samples per prompt on average (0.69% on the pilot). The case for truncation mattering rests on the
   ceiling and on the upper tail (0.8% of deleted names, about 118 at template 0, computed as
   14,262 x 0.0083, have q of at least 1e-3), not on mass.
4. **Shipped sampler (approximate):** the reachable ceiling is 194 names, and untruncated sampling at T 0.7
   passes it after about 49 samples per prompt; at 1,000 samples per prompt it finds 399. This is the
   strongest practical case, but it omits the repetition penalty, so it should be labelled approximate.

## Proposed macros

All values are rounded from the cited fields. "scale" = results/P3 via `support.py`; "QI" =
Qwen2.5-7B-Instruct, "QB" = Qwen2.5-7B, "OI" = OLMo-2-1124-7B-Instruct, "OB" = OLMo-2-1124-7B; spec
p0.90_T1, template 0, unless stated. The file is `results/analysis_cpu/support/headline.json`; the keys are
`scale|<model>|<spec>` and `calibration`.

| macro | value | field |
|---|---|---|
| `\ztMedQQI` | $1.4\times10^{-11}$ | scale QI p0.90_T1 template0.median_q_max (1.38e-11) |
| `\ztMedSightQI` | $7\times10^{10}$ | scale QI template0.median_samples_to_first_sight_q_max (7.26e10) |
| `\ztSeenKQI` | 1.3\% | scale QI template0.share_seen_q_max["1000"] (0.0133) |
| `\ztSeenMQI` | 12.3\% | scale QI template0.share_seen_q_max["1000000"] (0.1229) |
| `\ztQmilliQI` | 0.8\% | scale QI template0.share_q_max_at_least["0.001"] (0.0083) |
| `\ztMassQI` | 0.23\% | scale QI template0.deleted_mass_per_prompt_mean (0.0023) |
| `\ztMedQQB` | $2.1\times10^{-9}$ | scale QB template0.median_q_max (2.08e-9) |
| `\ztMedQOI` | $2.9\times10^{-10}$ | scale OI template0.median_q_max (2.91e-10) |
| `\ztMedQOB` | $4.7\times10^{-9}$ | scale OB template0.median_q_max (4.71e-9) |
| `\ztPlateauQI` | 1,252 | scale QI curve_t0.plateau_reachable |
| `\ztRecQI` | 15,514 | scale QI curve_t0.n_recognised |
| `\ztUntrKQI` | 575 | scale QI curve_t0["1000"].E_untruncated_qmax (575.1) |
| `\ztTruncKLoQI` | 284 | scale QI curve_t0["1000"].E_truncated_lo (283.8) |
| `\ztTruncKHiQI` | 417 | scale QI curve_t0["1000"].E_truncated_hi (417.3) |
| `\ztUntrMQI` | 2,587 | scale QI curve_t0["1000000"].E_untruncated_qmax (2586.5) |
| `\ztTruncMHiQI` | 863 | scale QI curve_t0["1000000"].E_truncated_hi (863.1) |
| `\ztNstarQI` | 24,000 | scale QI curve_t0.n_star_qmax_exceeds_plateau (24,046) |
| `\ztShipPlateauQI` | 194 | scale QI shipped_hf curve_t0.plateau_reachable (approximate) |
| `\ztShipNstarQI` | 49 | scale QI shipped_hf curve_t0.n_star_qmax_exceeds_plateau (48.7; approximate) |
| `\ztShipUntrKQI` | 399 | scale QI shipped_hf curve_t0["1000"].E_untruncated_qmax (399.3; approximate) |
| `\ztCalObsQI` | 3,466 | calibration.Qwen2.5-7B-Instruct.untruncated_T1.member_samples_observed |
| `\ztCalPredQI` | 3,653 | calibration...member_samples_predicted_any (3653.1) |
| `\ztCalObsOI` | 4,042 | calibration.OLMo-2-1124-7B-Instruct.untruncated_T1.member_samples_observed |
| `\ztCalPredOI` | 4,069 | calibration...member_samples_predicted_any (4068.7) |
| `\ztCalDistObsQI` | 75 | calibration QI distinct_members_observed |
| `\ztCalDistPredQI` | 77.0 | calibration QI distinct_members_predicted_any (76.97) |
| `\ztCalDistObsOI` | 78 | calibration OI distinct_members_observed |
| `\ztCalDistPredOI` | 83.5 | calibration OI distinct_members_predicted_any (83.46) |
| `\ztCalRhoQI` | 0.81 | calibration QI spearman_among_predicted_or_observed.rho (n 188) |
| `\ztCalRhoOI` | 0.75 | calibration OI spearman_among_predicted_or_observed.rho (n 228) |
| `\ztMassPilotQI` | 0.69\% | pilot QI p0.90_T1 template0.deleted_mass_per_prompt_mean (0.006889) |

The prefix `zt` is new: `paper_numbers.py` would need it added to its `\missing` fallback list.

## Checks recorded in the outputs

- RBU recomputed here matches the gate numbers: pilot QI 95.1% (all templates), QB 52.4%, OI 93.8%; scale QI
  91.9% (`rbu_pairs`).
- Under truncation, P5 observed 3,581 (Qwen) and 4,362 (OLMo) member answers at top-p 0.9, inside the bounds
  [3,232, 4,552] and [3,642, 5,354]; none from a robustly unreachable entity.
- The P5 scoring pass reproduces the open-prompt run's log-probabilities: median absolute difference 0 nats.
  The maximum is 1.18 (Qwen) and 0.74 (OLMo) nats, on a few forms (batch padding numerics in bf16).
- Calibration bins (`calibration_bins.csv`) track observed counts in every bin from 0.01 to over 100 expected
  samples.
