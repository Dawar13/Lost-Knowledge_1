# Gate SMOKE decision: **BLOCKED**

Instruct model: `Qwen/Qwen2.5-0.5B-Instruct`

Verdict computed on robust RBU (margin 0.0 raw logits: an entity counts as unreachable only if it stays cut with its logit raised by the margin; docs/04 section 2, notes/decisions.md). The same rules on exact RBU give: **BLOCKED**.

## Validity checks

- V1_no_sampling_violations: True
- V2_recognition_auroc_ge_0.85: False
- V3_at_least_300_recognised: False
- recognition AUROC (yes/no, members vs foils): 0.627
- 4-way accuracy on members: 0.361
- recognised entities: 50 of 440

## Sampling sanity check

- shipped_hf: 2200 samples, 78 valid, 78 id hits on 1 distinct forms; violations (robustly unreachable yet sampled) **0**, boundary hits 0, exact-verdict violations 0, non-canonical paths 0
- p0.90_T1: 2200 samples, 110 valid, 112 id hits on 7 distinct forms; violations (robustly unreachable yet sampled) **0**, boundary hits 0, exact-verdict violations 0, non-canonical paths 0

## Recognised-but-unreachable rate (RBU)

RBU mean = robust (headline); exact and upper use no margin and the opposite margin.

| sampler | RBU mean | 95% CI | RBU all templates | fail at first token | exact | upper |
|---|---|---|---|---|---|---|
| shipped_hf | 98.0% | 88.9% to 100.0% | 98.0% | 66.3% | 98.0% | 98.0% |
| p0.80_T1 | 91.0% | 76.0% to 96.1% | 88.0% | 41.8% | 91.0% | 91.0% |
| p0.90_T1 | 75.0% | 50.0% to 84.1% | 72.0% | 29.6% | 75.0% | 75.0% |
| p0.95_T1 | 46.0% | 25.0% to 56.8% | 40.0% | 18.9% | 46.0% | 46.0% |
| p0.90_T0.7 | 95.0% | 80.0% to 100.0% | 94.0% | 56.8% | 95.0% | 95.0% |
| k20_T1 | 82.0% | 58.1% to 92.7% | 80.0% | 44.3% | 82.0% | 82.0% |
| k50_T1 | 61.0% | 26.7% to 79.0% | 60.0% | 29.0% | 61.0% | 61.0% |
| minp0.05_T1 | 93.0% | 77.3% to 98.3% | 92.0% | 50.5% | 93.0% | 93.0% |
| minp0.10_T1 | 97.0% | 85.7% to 100.0% | 96.0% | 57.7% | 97.0% | 97.0% |
| untruncated_T1 | 0.0% | 0.0% to 0.0% | 0.0% | nan | 0.0% | 0.0% |

## Sensitivity of RBU to the margin (RBU mean; margin in raw logits)

| sampler | eps 0.0 | eps 0.25 | eps 0.5 | eps 1.0 | eps 2.0 | eps 4.0 |
|---|---|---|---|---|---|---|
| shipped_hf | 98.0% | 97.0% | 97.0% | 94.0% | 90.0% | 67.0% |
| p0.90_T1 | 75.0% | 73.0% | 69.0% | 58.0% | 35.0% | 25.0% |

## Notes

- validity checks failed: ['V2_recognition_auroc_ge_0.85', 'V3_at_least_300_recognised']. Fix these before reading any RBU number.

Decision rules: docs/02_phases_and_gates.md. A human must sign off below before Phase 2.

Sign-off: ____________________