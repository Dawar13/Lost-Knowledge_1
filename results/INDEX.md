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
| `G0_5090` | **gate (current)** | RTX 5090 32 GB, dedicated | 9486a4b | pilot_v2 | same snapshots | 0.75 | BLOCKED: V2 AUROC 0.835 | 0.180 |

Gate reports: `gates/G0.md` (attempt 2), `gates/G0_attempt1_a100.md`; per-category diagnostics
`gates/G0_diagnostics.md`, `gates/G0_attempt1_a100_diagnostics.md`. GPU ledger: `logs/gpu_ledger.csv`.

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
