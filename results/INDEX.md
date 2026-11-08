# MB5 results index

Every number in docs, notes or the paper must come from a file listed here (CLAUDE.md rule 5).
Each run directory holds the raw outputs of `mb5.run_model` / `mb5.selfcheck` and, under `metrics/`,
the tables computed from them by `mb5.metrics` (no GPU; rerun any time). `metrics/manifest.json`
records the sha256 of every input and the code version that computed the tables.

Status keys: **gate** = counts for a pre-registered decision; **dev** = instrument development,
never a result; **superseded** = kept for the record, replaced by a later attempt.

## Data snapshots (TailCat pilot)

| split | built | categories | members / foils | entities.jsonl sha256 | notes |
|---|---|---|---|---|---|
| `data/tailcat/pilot` (v1) | 2026-11-08, code before d1055ed, config at d1055ed | 11 | 3,225 / 1,912 | `c7f37d52...7b1f63` | used by G0 attempt 1; superseded |
| `data/tailcat/pilot_v2` | 2026-11-08, code and config 9486a4b | 11 | 2,312 / 1,724 | `7cc454ad...ec122` | G0 audit fixes: min_sitelinks 2 for all categories; Bengali directors by citizenship; Assamese writers by writing language (notes/decisions.md) |

Each split also holds `raw/` (the Wikidata query results with query text and fetch time: the
snapshot record), `build_report.json` (sanity report) and `valid_names.json` (uncapped member names,
for judging generated names in P4 and P5). Full sha256 values: `metrics/manifest.json` of each run.

## Gate runs

| run | status | GPU | code | data | models (snapshot) | epsilon | verdict | GPU-h |
|---|---|---|---|---|---|---|---|---|
| `G0_a100` | superseded | A100-SXM4-80GB, shared | 0d09276 (calibration abdccc6, 6ce67a6) | pilot (v1) | Qwen2.5-7B-Instruct (a09a354), Qwen2.5-7B (d149729) | 0.75 | BLOCKED: V2 AUROC 0.791 | 0.822 incl. 0.201 stopped run |
| `G0_5090` | **gate, a-priori recognition probe** (proposed main analysis under option B) | RTX 5090 32 GB, dedicated | 9486a4b | pilot_v2 | same snapshots | 0.75 | BLOCKED: V2 AUROC 0.835 | 0.180 |
| `G0_a3_5090` | gate, option A probe (worked examples): failed fix; sensitivity analysis; full prompt and sample artefacts | RTX 5090 32 GB, dedicated | 7660274 | pilot_v2 | same snapshots | 0.75 | BLOCKED: V2 AUROC 0.817 | 0.188 |

| `G1` | **gate G1 (P2)**: OLMo-2 base, SFT, DPO, Instruct; analysis also uses the Qwen pair from `G0_5090` | RTX 5090 32 GB, dedicated | 7a50eb9 | pilot_v2 | OLMo-2-1124-7B (7df9a82), -SFT (1de02c0), -DPO (e34ea60), -Instruct (470b1fb) | 0.5 | H2 HOLDS, H3 HOLDS | 0.257 incl. 0.013 failed run |

| `P3_robust` | P3 robustness (E7): pilot_v2, 5 templates, temperature sweep (T 1.0 / 0.7 / 0.5 at top-p 0.9); Qwen and OLMo pairs | RTX 5090 | ff7fce2 | pilot_v2 | Qwen2.5-7B(-Instruct), OLMo-2-1124-7B(-Instruct) | Qwen 0.75, OLMo 0.5 | wording CV 0.009 to 0.106 (all below 0.25) | about 0.23 |
| `P5` | **gate G3**: mitigation frontier (14 samplers incl. first-token relaxation), pilot_v2, template 0, 1,000 samples per category per sampler; `audit/` holds the invented-name sensitivity and the 200-answer hand-audit sheet | RTX 5090 | a2be437 (frontier), ca604db (audit) | pilot_v2 | Qwen2.5-7B-Instruct, OLMo-2-1124-7B-Instruct | Qwen 0.75, OLMo 0.5 | G3 FAIL (recovered 12.3% and 23.7%) | about 0.42 |
| `P3` | **gate G2**: full TailCat capped split full_v1_s150 (277 categories, 33,437 members), 3 templates; sample check at scale for the instruct models; P4 estimator tables in `P3/metrics/` | RTX 5090 | 3053872 era (see meta.json) | full_v1_s150 | Qwen and OLMo pairs | Qwen 0.75, OLMo 0.5 | running at the time of writing | |
| `P3_scale` | scale ladder (exploratory): Qwen2.5-Instruct 0.5B / 1.5B / 3B on pilot_v2 | RTX 5090 | | pilot_v2 | Qwen2.5-{0.5,1.5,3}B-Instruct | per model | queued | |

Data split for P3: `data/tailcat/full_v1` (uncapped build, 346 categories; raw Wikidata cache as
`data/tailcat/full_v1_raw.tar.gz`, sha256 dae975aa57d7a5928955cff45b27e4b14f30b8480fbb46d4c3951d3467bb2372)
and `data/tailcat/full_v1_s150` (capped split used by the runs; copied in when P3 finishes).

G1 layout: `G1/<model>/` as for G0 (plus `first_token.json`, `prompts.json`, `samples.jsonl.gz` for the
post-trained stages); `G1/calibration/<model>/`; `G1/metrics/<model or pair>/` (mb5.metrics per model);
`G1/stages.csv`, `G1/paired.csv`, `G1/mechanism.csv`, `G1/g1_summary.json` (mb5.analysis_multi).
Gate report: `gates/G1.md`. First-token statistics for the G0 Qwen pair: `G0_5090/<model>/first_token.json`.

Attempts 2 and 3 have identical reachability verdicts and calibrations (verified row by row); they differ
only in the yes/no recognition prompt. So the prompts and sampled answers stored with attempt 3
(`prompts.json`, `samples.jsonl.gz`, `selfcheck_samples.jsonl.gz`) are also those of attempt 2's
reachability and sample check.

Gate reports: `gates/G0.md` (attempt 3, latest), `gates/G0_attempt2_5090.md`, `gates/G0_attempt1_a100.md`;
per-category diagnostics `gates/*_diagnostics.md`. GPU ledger: `logs/gpu_ledger.csv`.
Human sign-off pending (option B proposed).

Layout of a gate run directory:

```
<run>/<model>/meta.json            model snapshot, code version, GPU, specs (incl. shipped), epsilon, time, peak memory
<run>/<model>/reach.jsonl.gz       one row per (entity, surface form, template): per-position rank and G,
                                   p*, k*, min-p*, and per sampler exact / robust verdicts, failing positions, d*
<run>/<model>/recog.jsonl.gz       one row per member and foil: yes/no V; members also 4-way choice
<run>/<model>/sample_check.json    (instruct) V1 sampling check per sampler and category
<run>/calibration/<model>/selfcheck.json   epsilon calibration: logit gap, margins needed, rule value
<run>/summary.json, diagnostics.*  gate analysis and per-category diagnostics
<run>/metrics/                     every table below

From G0 attempt 3 onwards (not in attempts 1 and 2), every run also keeps what was asked and answered:
<run>/<model>/prompts.json         exact generation prompts, one per category and wording (chat template included)
<run>/<model>/recog.jsonl.gz       + exact yes/no and 4-way prompts, quiz options, correct and chosen letter,
                                   letter probabilities, raw log P(Yes) and log P(No)
<run>/<model>/samples.jsonl.gz     every answer generated in the sample check: text, token ids, matched entity,
                                   verdict of the matched answer
<run>/calibration/<model>/selfcheck_samples.jsonl.gz   every distinct calibration sample: text, ids,
                                   times sampled, signed margin d*, exact verdict
<run>/calibration/<model>/logit_gap_raw.jsonl.gz       raw generate-vs-scorer gap values
```

## Metric tables (`<run>/metrics/`) and where they go in the paper

| table | what it holds | paper use |
|---|---|---|
| `headline.json` | recognition validity, robust RBU per sampler with CI, paired test | abstract, Table 1 |
| `rbu_<m>.csv` | RBU per sampler x verdict mode (robust, exact, upper): mean, all-templates, cluster-bootstrap CIs, per template, CV over templates, label-only | Table 1; appendix robustness (wording, aliases, margin) |
| `rbu_curves_<m>.csv` | exact single-truncation curves: RBU against top-p (T=1, 0.7), top-k, min-p | Figure 1 |
| `failpos_<m>.csv` | first failing answer position, per sampler | Figure 2 (H3) |
| `paired.csv` | instruct vs base per sampler and template: discordant pairs, sign test, RBU ratio | Section 5 (H2) |
| `popularity_<m>.csv` | per sitelink quintile: recognition rate, RBU per sampler | Section 7 (who loses), Figure 6 |
| `length_<m>.csv` | per label token length: recognition rate, RBU | appendix (tokenisation, C3) |
| `reachable_set_<m>.csv` | per sampler x category x template: members reachable (all, recognised) | Section 4 and 6 (reachable set size) |
| `sample_check_<m>.csv` | per sampler x category: samples, valid, distinct, violations, boundary hits, coverage of the reachable set | Section 2 validation; Section 6 preview |
| `validity_<m>.csv` | per category: AUROC, 4-way accuracy, tau, foil pass rate, recognised | Section 3; appendix |
| `rbu_by_category_<m>.csv` | per sampler x category RBU (robust, exact) | appendix table |
| `calibration_<m>.json` | generate-vs-scorer logit gap, margin needed, epsilon rule | Section 2 and appendix (numeric margin) |
| `entities_<m>.csv` | master table: one row per (entity, template) with recognition, popularity, critical values, verdicts | any further analysis |
| `top_answers_<m>.csv` | (attempt 3 on) the 15 most frequent generated answers per sampler and category, counts, whether a real member | qualitative examples; P4 and P5 previews |

## Instrument development runs (dev: never cite as results)

| dir | what it tested | outcome |
|---|---|---|
| `dev_a100/smoke` | P0 smoke test, Qwen2.5-0.5B-Instruct, pilot v1, no margin | completes; V1 0 violations on 227 hits (2 entities); BLOCKED V2/V3 as expected |
| `dev_a100/selfcheck` | dense self-check, 0.5B, no margin | 180 of 4,802 sampled sequences scored unreachable: bf16 jitter and ties at the boundary (motivated the margin) |
| `dev_a100/selfcheck_gap` | generate-vs-scorer logit gap, 0.5B | rel_near median 0.0625, q0.999 0.69, max 2.78 raw logits |
| `dev_a100/selfcheck_margin` | margin sweep, 0.5B | robust failures 8 / 2 / 2 / 2 / 1 at margins 0.25 / 0.5 / 0.75 / 1.0 / 1.5 |
| `dev_a100/cputest_selfcheck` | same check on CPU in float32 | logit gap exactly 0, no failures: scorer logic matches generate() |
| `dev_5090/smoke` | smoke test on the RTX 5090, pilot_v2 | completes in 18 s; V1 holds; BLOCKED V2/V3 as expected |

Run logs (stdout of every job): `logs/archive/a100/`, `logs/archive/rtx5090/`.

## Not yet computed (needed later)

| metric | needs | phase |
|---|---|---|
| first-token concentration (top-1 mass of the first answer token per prompt) as the RBU predictor | one GPU forward per (category, template, model) | P2 (E5), cheap |
| RBU across OLMo-2 stages and a second family | GPU runs | P2 (G1) |
| sample-based richness estimates (Chao1, coverage) against the reachable and recognised sets | 1,000 samples per category per sampler | P4 (E10) |
| invented-name rate and hand audit of 200 generated names | generations + human audit | P5 (E11) |
| multilingual RBU, mixed-effects regressions with popularity | full TailCat, 4 languages | P3 (E8, E9) |
| human check of recognition on a sample of entities | human | before submission |
| data audit: implausible sitelink counts (e.g. "Federal University of Science and Innovation of Minas Gerais", 104 sitelinks, top of the OLMo walkthrough) | CPU | before P3 runs |
