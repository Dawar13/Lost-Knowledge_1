# Gate G0 decision: **BLOCKED**

Instruct model: `Qwen/Qwen2.5-7B-Instruct`; base model: `Qwen/Qwen2.5-7B`

Verdict computed on robust RBU (margin 0.75 raw logits: an entity counts as unreachable only if it stays cut with its logit raised by the margin; docs/04 section 2, notes/decisions.md). The same rules on exact RBU give: **BLOCKED**.

## Validity checks

- V1_no_sampling_violations: True
- V2_recognition_auroc_ge_0.85: False
- V3_at_least_300_recognised: True
- recognition AUROC (yes/no, members vs foils): 0.835
- 4-way accuracy on members: 0.767
- recognised entities: 1310 of 2312

## Sampling sanity check

- shipped_hf: 11000 samples, 3561 valid, 3561 id hits on 15 distinct forms; violations (robustly unreachable yet sampled) **0**, boundary hits 0, exact-verdict violations 0, non-canonical paths 0
- p0.90_T1: 11000 samples, 3581 valid, 3749 id hits on 44 distinct forms; violations (robustly unreachable yet sampled) **0**, boundary hits 1, exact-verdict violations 1, non-canonical paths 0

## Recognised-but-unreachable rate (RBU)

RBU mean = robust (headline); exact and upper use no margin and the opposite margin.

| sampler | RBU mean | 95% CI | RBU all templates | fail at first token | exact | upper | base RBU mean |
|---|---|---|---|---|---|---|---|
| shipped_hf | 98.9% | 97.3% to 99.6% | 98.6% | 89.3% | 99.3% | 99.6% | nan |
| p0.80_T1 | 97.8% | 94.5% to 99.1% | 97.1% | 85.7% | 98.4% | 98.9% | 76.3% |
| p0.90_T1 | 95.1% | 88.0% to 98.2% | 94.0% | 84.1% | 96.4% | 97.3% | 52.4% |
| p0.95_T1 | 90.8% | 78.7% to 96.0% | 87.4% | 80.3% | 92.8% | 94.5% | 33.3% |
| p0.90_T0.7 | 98.6% | 96.4% to 99.4% | 98.2% | 87.8% | 98.9% | 99.4% | 85.3% |
| k20_T1 | 79.5% | 64.7% to 87.8% | 76.4% | 66.5% | 84.5% | 88.1% | 64.6% |
| k50_T1 | 63.7% | 45.7% to 73.3% | 58.7% | 56.8% | 71.4% | 78.3% | 42.7% |
| minp0.05_T1 | 97.4% | 93.7% to 99.0% | 96.6% | 85.7% | 98.4% | 98.8% | 80.1% |
| minp0.10_T1 | 98.2% | 95.7% to 99.3% | 97.7% | 86.7% | 98.8% | 99.2% | 87.2% |
| untruncated_T1 | 0.0% | 0.0% to 0.0% | 0.0% | nan | 0.0% | 0.0% | 0.0% |

## Sensitivity of RBU to the margin (RBU mean; margin in raw logits)

| sampler | eps 0.0 | eps 0.25 | eps 0.5 | eps 1.0 | eps 2.0 | eps 4.0 |
|---|---|---|---|---|---|---|
| shipped_hf | 99.3% | 99.2% | 99.1% | 98.8% | 98.1% | 94.4% |
| p0.90_T1 | 96.4% | 96.0% | 95.5% | 94.8% | 92.1% | 84.8% |

## Notes

- validity checks failed: ['V2_recognition_auroc_ge_0.85']. Fix these before reading any RBU number.
- paired test at p0.90_T1: {'instruct_only_unreachable': 456, 'base_only_unreachable': 5, 'z': 21.005164120751267, 'p_two_sided': 5.882993993169263e-98}

Decision rules: docs/02_phases_and_gates.md. A human must sign off below before Phase 2.

Sign-off: ____________________