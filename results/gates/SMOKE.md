# Gate SMOKE decision: **BLOCKED**

Instruct model: `Qwen/Qwen2.5-0.5B-Instruct`

Verdict computed on robust RBU (margin 0.0 raw logits: an entity counts as unreachable only if it stays cut with its logit raised by the margin; docs/04 section 2, notes/decisions.md). The same rules on exact RBU give: **BLOCKED**.

## Validity checks

- V1_no_sampling_violations: True
- V2_recognition_auroc_ge_0.85: False
- V3_at_least_300_recognised: False
- recognition AUROC (yes/no, members vs foils): 0.623
- 4-way accuracy on members: 0.336
- recognised entities: 45 of 440

## Sampling sanity check

- shipped_hf: 2200 samples, 107 valid, 107 id hits on 0 distinct forms; violations (robustly unreachable yet sampled) **0**, boundary hits 0, exact-verdict violations 0, non-canonical paths 0
- p0.90_T1: 2200 samples, 119 valid, 120 id hits on 0 distinct forms; violations (robustly unreachable yet sampled) **0**, boundary hits 0, exact-verdict violations 0, non-canonical paths 0

## Recognised-but-unreachable rate (RBU)

RBU mean = robust (headline); exact and upper use no margin and the opposite margin.

| sampler | RBU mean | 95% CI | RBU all templates | fail at first token | exact | upper |
|---|---|---|---|---|---|---|
| shipped_hf | 96.7% | 88.9% to 100.0% | 95.6% | 59.8% | 96.7% | 96.7% |
| p0.80_T1 | 94.4% | 81.5% to 100.0% | 91.1% | 34.1% | 94.4% | 94.4% |
| p0.90_T1 | 75.6% | 58.8% to 86.5% | 66.7% | 22.2% | 75.6% | 75.6% |
| p0.95_T1 | 55.6% | 34.4% to 72.7% | 53.3% | 13.2% | 55.6% | 55.6% |
| p0.90_T0.7 | 96.7% | 88.9% to 100.0% | 95.6% | 42.5% | 96.7% | 96.7% |
| k20_T1 | 80.0% | 58.6% to 95.0% | 77.8% | 32.4% | 80.0% | 80.0% |
| k50_T1 | 56.7% | 27.8% to 74.2% | 55.6% | 15.8% | 56.7% | 56.7% |
| minp0.05_T1 | 96.7% | 88.9% to 100.0% | 95.6% | 36.8% | 96.7% | 96.7% |
| minp0.10_T1 | 96.7% | 88.9% to 100.0% | 95.6% | 46.0% | 96.7% | 96.7% |
| untruncated_T1 | 0.0% | 0.0% to 0.0% | 0.0% | nan | 0.0% | 0.0% |

## Notes

- validity checks failed: ['V2_recognition_auroc_ge_0.85', 'V3_at_least_300_recognised']. Fix these before reading any RBU number.

Decision rules: docs/02_phases_and_gates.md. A human must sign off below before Phase 2.

Sign-off: ____________________