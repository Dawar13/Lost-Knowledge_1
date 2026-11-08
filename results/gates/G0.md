# Gate G0 decision: **BLOCKED**

Instruct model: `Qwen/Qwen2.5-7B-Instruct`; base model: `Qwen/Qwen2.5-7B`

Verdict computed on robust RBU (margin 0.75 raw logits: an entity counts as unreachable only if it stays cut with its logit raised by the margin; docs/04 section 2, notes/decisions.md). The same rules on exact RBU give: **BLOCKED**.

## Validity checks

- V1_no_sampling_violations: True
- V2_recognition_auroc_ge_0.85: False
- V3_at_least_300_recognised: True
- recognition AUROC (yes/no, members vs foils): 0.791
- 4-way accuracy on members: 0.705
- recognised entities: 1587 of 3225

## Sampling sanity check

- shipped_hf: 11000 samples, 2887 valid, 3075 id hits on 13 distinct forms; violations (robustly unreachable yet sampled) **0**, boundary hits 0, exact-verdict violations 0, non-canonical paths 0
- p0.90_T1: 11000 samples, 2897 valid, 3145 id hits on 34 distinct forms; violations (robustly unreachable yet sampled) **0**, boundary hits 0, exact-verdict violations 0, non-canonical paths 0

## Recognised-but-unreachable rate (RBU)

RBU mean = robust (headline); exact and upper use no margin and the opposite margin.

| sampler | RBU mean | 95% CI | RBU all templates | fail at first token | exact | upper | base RBU mean |
|---|---|---|---|---|---|---|---|
| shipped_hf | 99.3% | 98.2% to 99.7% | 99.1% | 88.2% | 99.5% | 99.7% | nan |
| p0.80_T1 | 98.5% | 97.0% to 99.3% | 98.1% | 85.6% | 99.1% | 99.3% | 81.8% |
| p0.90_T1 | 96.1% | 91.7% to 98.4% | 94.9% | 82.9% | 97.4% | 98.3% | 59.6% |
| p0.95_T1 | 91.3% | 81.6% to 96.8% | 88.3% | 77.5% | 93.2% | 95.1% | 40.6% |
| p0.90_T0.7 | 99.0% | 97.8% to 99.6% | 98.7% | 87.4% | 99.3% | 99.6% | 89.4% |
| k20_T1 | 83.2% | 73.5% to 90.2% | 80.8% | 66.1% | 87.9% | 91.2% | 71.8% |
| k50_T1 | 66.9% | 54.9% to 75.6% | 62.0% | 50.7% | 74.8% | 82.1% | 48.5% |
| minp0.05_T1 | 98.1% | 95.9% to 99.2% | 97.7% | 85.0% | 98.8% | 99.2% | 84.9% |
| minp0.10_T1 | 98.7% | 97.2% to 99.5% | 98.3% | 86.1% | 99.1% | 99.4% | 90.2% |
| untruncated_T1 | 0.0% | 0.0% to 0.0% | 0.0% | nan | 0.0% | 0.0% | 0.0% |

## Sensitivity of RBU to the margin (RBU mean; margin in raw logits)

| sampler | eps 0.0 | eps 0.25 | eps 0.5 | eps 1.0 | eps 2.0 | eps 4.0 |
|---|---|---|---|---|---|---|
| shipped_hf | 99.5% | 99.4% | 99.4% | 99.2% | 98.7% | 95.7% |
| p0.90_T1 | 97.4% | 96.8% | 96.4% | 95.8% | 93.0% | 84.9% |

## Notes

- validity checks failed: ['V2_recognition_auroc_ge_0.85']. Fix these before reading any RBU number.
- paired test at p0.90_T1: {'instruct_only_unreachable': 477, 'base_only_unreachable': 5, 'z': 21.49901088142587, 'p_two_sided': 1.5904147208013645e-102}

Decision rules: docs/02_phases_and_gates.md. A human must sign off below before Phase 2.

Sign-off: ____________________