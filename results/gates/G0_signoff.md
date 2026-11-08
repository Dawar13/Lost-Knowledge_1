# Gate G0 sign-off

**Decision: proceed to Phase 2 under option B.** Recorded 2026-11-08.

- Human instruction (2026-11-08): "start the next phases as much as you can" after the attempt-3 report,
  which proposed option B. Recorded as sign-off; signer's name to be added below.
- Main analysis: G0 attempt 2 (`results/G0_5090`, report `G0_attempt2_5090.md`), recognition by the
  a-priori yes/no probe.
- Sensitivity analysis: attempt 3 (`results/G0_a3_5090`, report `G0.md`), worked-example probe.
- Validity: V1 holds (0 robust violations in 22,000 samples); V3 holds (1,310 recognised); V2 below the
  pre-registered bar (pooled AUROC 0.835 against 0.85): reported in the paper, with the main result shown
  also on the categories that pass V2 on their own; V4 judged on the base model (us_state_capitals RBU 0.0%
  at p0.90_T1) and untruncated RBU = 0; the instruct model's capitals result (88.9%) is a finding.
- Deciding numbers (attempt 2, robust RBU, mean over templates, 95% cluster-bootstrap CI):
  shipped_hf 98.9% (97.3 to 99.6); p0.90_T1 95.1% (88.0 to 98.2) instruct against 52.4% base;
  paired at p0.90_T1, template 0: 456 instruct-only against 5 base-only unreachable.
- GPU at sign-off: 1.317 of 15 GPU-hours.
- Full reasoning: `notes/decisions.md` (entries dated 2026-11-08).

Signed: ____________________ (name to be supplied)
