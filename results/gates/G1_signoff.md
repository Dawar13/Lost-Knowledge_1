# Gate G1 sign-off

**Decision: G1 passed (H2 holds, H3 holds); proceed to Phase 3.** Recorded 2026-11-08.

- Human instruction (2026-11-08): "Sign G1 and start phase 3 but keep all the relevant paper related data
  and artifacts intact, we will need them during reviews as well." Followed by: "complete all the
  important phases of this project as per the results because I will not be awake to sign off things."
- Gate report: `G1.md`; data and analysis: `results/G1/`; reasoning: `notes/decisions.md` (2026-11-08).
- Deciding numbers (robust RBU at p0.90_T1, 95% cluster-bootstrap CI): OLMo-2 base 54.4% (43.0 to 60.6),
  SFT 89.6%, DPO 93.8%, Instruct 93.8% (83.7 to 97.6); Qwen2.5 base 52.4%, Instruct 95.1% (88.0 to 98.2).
  Paired base to final: OLMo 348 vs 3, Qwen 456 vs 5. First-token failure share: OLMo Instruct 73.8%,
  Qwen Instruct 84.1%. Mechanism (exploratory): Spearman 0.817, first-token top-1 mass against RBU.
- Validity: 0 robust violations in 33,000 OLMo samples; recognition AUROC 0.80 to 0.82 for all OLMo stages
  (same probe limitation as G0, reported). Llama-3.1 not run (access pending); to be added as a third
  family when access arrives.
- GPU at sign-off: 1.574 of 15 GPU-hours.

Signed: ____________________ (name to be supplied; instruction given by the human as quoted above)

## Delegation for Phases 3 to 5

Because the human is unavailable to sign, the agent applies the pre-registered rules of
`docs/02_phases_and_gates.md` itself for G2 (a framing gate) and G3, records each verdict in
`results/gates/` and `notes/decisions.md` with "decided by the agent under delegation", and fixes every
interpretation of a rule in `notes/decisions.md` before the data it applies to exists. The human reviews
and may overturn any of these decisions on waking. No threshold is changed after seeing data.
