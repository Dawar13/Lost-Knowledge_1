# Internal adversarial review, round 1

Draft reviewed: `paper/main.tex` and `paper/main.pdf` as committed at 06985ad (PDF compiled 2026-11-09 08:35), with `paper/numbers.tex` and `paper/tables/*.tex`. Written as three ARR reviewers would. Pending work (E13 ask directly, E14 list prompt, E15/G3b self-check fix, Llama-3.1, figures, novelty contrast, support and region analyses) is judged as absent, with a note on where each lands.

Scores use the ARR scales (1 to 5, half points allowed): Soundness, Excitement, Overall (3 = Findings, 4 = main conference, 5 = award level).

| reviewer | focus | Soundness | Excitement | Overall |
|---|---|---|---|---|
| R1 | novelty | 3.5 | 2.5 | 2.5 |
| R2 | method and validity | 2.5 | 3 | 2.5 |
| R3 | clarity, significance, theme | 3.5 | 3 | 3 |

Likely outcome as it stands: borderline Findings, with "resubmit" risk from R1 and R2. The idea and the exactness are liked; the paper loses points on novelty against Awad et al. (2026), on claims the text does not support, and on looking unfinished (no figures, about 4.6 of 8 pages used, rendering bugs).

---

## Reviewer 1: novelty sceptic

**Summary.** The paper applies a token-survival test (one teacher-forced pass; an answer is reachable iff every token survives truncation) to Wikidata entities that a model "recognises", reports that most recognised entities are unreachable for chat models, traces the rise across OLMo-2 post-training stages, and argues that sample-based knowledge estimators converge to the reachable set.

**Strengths.**
- Clean, exact framing; the zero-probability case is crisp and well motivated.
- Stage-resolved OLMo-2 analysis and the first-token shift are new relative to the closest work.
- The estimator argument (Section 6) is simple and useful.

**Weaknesses.**
1. The instrument is not new. The paper itself says the test is "also used for ordinary words by Awad et al. (2026)", and Related Work adds that they "compare base and aligned models and tabulate shipped defaults". So the test, the base vs aligned contrast and the shipped-defaults angle all predate this paper. What remains new (recognition pairing, stage resolution, first-token mechanism, estimator correction, popularity) is stated in one sentence of Related Work, not demonstrated.
2. Pairing an existing test with a yes/no plus four-way probe looks like a straightforward combination. The paper must show what the knowledge framing reveals that the word-level test cannot. Today nothing in the paper does this (E13 would).
3. The numeric margin is presented as a contribution; to me it is careful engineering, not a research contribution.
4. The estimator section reads as a remark: Li et al. (2025) already sample without nucleus truncation for this reason. The correction is never applied to a published estimate.
5. The fix fails; a negative result plus a frontier table is less exciting than a working remedy.

**Questions.** What exactly differs from Awad et al. in method, data and findings? Would their word-level analysis already predict your entity results?

**What would move me to 4 or more.**
- A side-by-side table: Awad et al. against this paper (unit of analysis, what counts as "known", models and stages, the mechanism, the estimator consequence, validation at scale).
- E13: show the same names are produced when cued directly but are unreachable under the open prompt. That is knowledge, not vocabulary, and is something the lexical test cannot show.
- Apply the correction to a concrete estimator in the Li et al. (2025) style on our samples, and report the corrected number.
- A remedy that passes G3b, or a sharper negative result explaining why no truncation sampler can satisfy both constraints.
- A third family (Llama-3.1, with its shipped T 0.6, top-p 0.9).

---

## Reviewer 2: methodology and validity

**Summary.** As R1.

**Strengths.**
- The exact test is validated against real `generate()` samples with zero robust violations.
- Cluster bootstrap CIs over categories; pre-registered thresholds; an honest region result and an honest negative fix.

**Weaknesses (each checked against the repository).**
1. **The introduction's example contradicts the paper's own definition.** "Kumaragurupara Desikar" (Q6443534) has V about 1.0 but fails the four-way choice (`mc_correct` False, `recognised` False in `results/G0_5090/metrics/entities_Qwen2.5-7B-Instruct.csv`), so the paper's test does not count it as recognised. The claim "it will usually say Thiruvalluvar" is also not what the data show. Under top-p 0.9 the most frequent answer for Tamil poets is "泰戈尔" (Tagore in Chinese characters, 36.5% of 1,000 samples); "tiruvalluvar" and "thiruvalluvar" together are 25.9% (`results/G0_a3_5090/metrics/top_answers_Qwen2.5-7B-Instruct.csv`).
2. **"Mass, not rank" is shown for OLMo only and is contradicted by Qwen.** Top-k 50 RBU for Qwen2.5 rises from 42.7% (base) to 63.7% (instruct) (`results/G1/stages.csv`), a 21-point rise. The OLMo-only reporting reads as selective.
3. **Unsupported robustness claim.** Section 3 and the Limitations say "the main results hold on the categories where it exceeds 0.85 on its own, and under a second probe wording (Appendix B)". Appendix B has only the validity table, with no RBU for the passing subset and no second-probe numbers. `numbers.tex` has no macro for either.
4. **The recognition proxy is below the paper's own bar** (pooled AUROC 0.835 pilot, 0.847 full). Four-way accuracy is near chance-plus for several categories (kerala_rivers 0.500, nepal_mountains 0.550, tamilnadu_hindu_temples 0.555; chance is 0.25). The word "knowledge" in the title and abstract rests on this test. E13 is the missing second test.
5. **Zero against tiny.** The obvious objection is that unreachable names would almost never be sampled even without truncation, so truncation changes little in practice. docs/04 section 5 lists "untruncated log-probabilities of the same entities stay finite" as a testable consequence, but the paper reports nothing on it. `logp_sum_T1.0` is stored per answer in `reach.jsonl.gz`, so this is CPU-only.
6. **The estimator claims are asymptotic but stated as fact.** "Every unseen-species estimator ... estimates the size of that support" fails at N = 1,000: Chao1 gives 141 while 231 members are reachable (Section 6). The shortfall against the recognised set has two parts: sampling (231 minus 141) and truncation (RBU times recognised). Only the second is "computed exactly". Also say that the species set is the capped member list (at most 150 per category).
7. **Popularity regression has no controls.** Rare names tend to have more tokens, and therefore more chances to be cut, and they differ in script and type. Add label token count (and type) to the GEE; `length_*.csv` and `entities_*.csv` already hold the covariates.
8. **The mechanism correlation is confounded.** Spearman 0.82 over 66 (model, category) points pools base and post-trained models; the between-model gap drives it. Report within-model or within-category associations, or a mixed model.
9. **The invented-name metric is of unknown validity, and Section 8's conclusions rest on it.**
   - 52.4% of the reference sampler's answers are "invented" for Qwen.
   - 6.5% of those reference answers are in CJK script (12.3% of the invented ones). Some are real Japanese novelists written in kanji, and some are Tagore in Chinese. OLMo has none (computed from `results/P5/*/samples_frontier.jsonl.gz`).
   - Match multilingual Wikidata labels and finish the hand audit before reading the frontier.
10. **Deviations from the pre-registered definitions are not reported** (CLAUDE.md rule 6, docs/04):
    - The headline uses RBU-mean, while docs/04 section 3 names RBU-all "the conservative headline".
    - Popularity uses GEE instead of the pre-registered mixed-effects logistic.
    - Holm correction (docs/04 section 6) is not mentioned.
    - G0 was closed by human sign-off under option B despite failing V2; the paper does not say so.
11. **The strongest validation is missing.** V1 at full scale (554,000 Qwen and 277,000 OLMo samples, 0 robust violations) is not in the text, although the macros `ptSamplesQwenInst`, `ptSamplesOlmoInst`, `ptViolations*` exist and are unused. Meanwhile "On CPU in float32 the two computations agree exactly" was shown only for Qwen2.5-0.5B (`results/dev_a100/cputest_selfcheck`), not for the 7B models, and must be qualified.
12. **Inconsistencies.**
    - "we sample about two thousand answers" (Section 2) against Appendix A: 500 per pilot category, so 5,500 samples and 1,685 distinct answers.
    - "94.1% with aliases" (Section 4, five wordings, P3_robust) against 95.1% in Table 1 (three wordings) for what reads as the same quantity.
    - "sign tests far below p = 0.01": give the p-values.
    - "We prove these rules equal the library's processors in unit tests": tests verify, they do not prove.
13. **Prompt realism.** Every result uses one-name answers. E14 (list prompt) is pending.
14. **"Shipped sampler" means Hugging Face `generate()` order.** vLLM, llama.cpp and SGLang order processors differently. State that the shipped claims hold for transformers 5.22.2.

**What would move me to 4.**
- Fixes 1 to 3 and 10 to 12 (writing only).
- Fixes 5 to 8 (CPU only).
- E13 and E14 run, and the invented-name metric validated (audit plus multilingual matching).

---

## Reviewer 3: clarity, significance, special theme

**Strengths.** A memorable title and hook, and a result that matters for the "homogenisation and knowledge collapse" debate: part of the collapse is a decoding artefact and can be measured exactly. The writing is plain and short.

**Weaknesses.**
1. **It looks unfinished.** There are no figures. The main body ends on page 5 (Limitations start at PDF line 396), so about 4.6 of 8 pages are used.
2. **Rendering bugs a reviewer will see on page 2 and page 6.**
   - "Appendix efapp:margin": main.tex lines 53 to 54, where the `\r` of `\ref` became a newline.
   - "extttresults/*/examples/": main.tex line 149, where `\t` became a tab.
   - "-0.8 points" uses a hyphen, not a minus sign.
3. **The bibliography prints internal notes.** Jiang et al. 2025 shows "Best Paper ... the NeurIPS site lists nine authors, without Alon Albalak"; Awad et al. and Garces Arias et al. show "(arXiv comment)".
4. **Artefacts are not cited.** `qwen2024qwen25`, `olmo2024olmo2` and `vrandecic2014wikidata` are in references.bib but never cited, so the models and Wikidata go uncredited (checklist B1). The KL-optimum claim in Section 5 has no citation (DPO). Top-k sampling has no origin citation.
5. **The theme link is thin.** The knowledge-collapse literature is not engaged. Candidates, each to verify before citing (rule 7): Peterson (2024) on AI and knowledge collapse; Shumailov et al. (2024) on model collapse; homogenisation-in-writing work such as Padmakumar and He (2024). The strongest theme argument is missing: names with probability exactly zero never enter LLM-generated corpora, so the tail cannot come back in the next generation of models.
6. **Overclaims.**
   - Abstract: "Any estimate of model knowledge built from samples therefore measures the sampler". It should be "built from truncated samples".
   - "four post-training stages": the base model is not a post-training stage.
   - "truncation by mass ... is what deletes the tail": see R2.2.
   - The title "Not Forgetting" rests on the recognition proxy: see R2.4.
7. **The head control is re-read without saying so.** "Even head knowledge collapses" presents US state capitals as a finding, but they were the pre-registered control V4, judged on the base model. Say so in one sentence.
8. **English prompts only.** For this theme the multilingual question is natural; it is acknowledged as future work. Fine, but expect the question.
9. **Region.** RBU is lower for Global South categories, against the framing of the data. It is honestly reported, but the composition hypothesis is untested (another fork is on it).

**What would move me to 4 or more.**
- Four or five figures. The capitals walk-through as Figure 1 (first-step shortlist, base 110 pieces against chat 7) is the clearest picture of the whole paper.
- Use the full 8 pages for the pending experiments.
- A paragraph tying RBU to synthetic-data feedback loops.
- A working or sharply explained fix.

---

## Responsible NLP checklist and submission rules

| item | status in draft | action |
|---|---|---|
| A1 limitations | present (Recognition, Tokenisation, Numerics, Scope) | add: single-answer prompts, transformers-only sampler order, invented-name metric validity, intro of E13/E14 outcomes |
| A2 risks | missing | short paragraph: wider sampling raises invented answers; TailCat lists public names from Wikidata, including living people |
| B1 cite artefacts | models and Wikidata uncited | cite Qwen2.5, OLMo-2 (and Llama-3.1 if run), Wikidata |
| B2/B3 licences and intended use | missing | state Wikidata licence and each model's licence after checking the model cards; TailCat release licence and intended use |
| B4/B5/B6 data documentation | partly (region and type tables) | add TailCat statistics: languages, regions, sizes, a datasheet in supplementary |
| C1 models and compute | GPU type in the repro table; total GPU-hours not stated (`\reproGpuHours` = 5.28 exists, unused) | state total GPU-hours and GPU type in text |
| C2/C3 settings and statistics | sampler settings and CIs present | add software versions (transformers 5.22.2, statsmodels 0.15.0) and the GEE details |
| D annotation | 200-answer audit pending; not described | describe annotators (authors), instructions, time, no payment, no personal data |
| E AI assistance | not addressed | this project's code, analyses and draft were produced with an AI agent; checklist question E1 must be answered truthfully under ACL's AI-assistance policy (human decision) |
| Anonymity | no names in the PDF | the repro table prints git commit hashes and its caption says everything is "in the repository". If github.com/Dawar13/Lost-Knowledge_1 is public, the hashes can be searched and de-anonymise the paper. Drop the hashes or replace them with anonymised IDs; ship code via an anonymised supplementary archive. The internal `results/...` source paths in captions are not identifying but clutter; move them to supplementary |
| Page limit | well under 8 pages | use the space |
| Numbers rule | all 95 macros used resolve (224 defined, none undefined) | hand-typed data-derived numbers remain: "about ten points" (Section 8), "AUROC around 0.8" (Limitations), "about two thousand answers" (Section 2, wrong), "far below p = 0.01" (Section 5). Make them macros or remove them |
| Style | British spelling throughout; no em or en dashes in source | "-0.8" hyphen used as a minus sign |

---

## Where the pending experiments land

| experiment | lands in | answers |
|---|---|---|
| E13 ask directly | Section 3 (second knowledge test), Limitations "Recognition", abstract "knowledge it holds" | R1.2, R2.4, R3.6 |
| E14 list prompt | Section 4, a new paragraph "not a quirk of the prompt format" | R2.13 |
| E15 / G3b self-check | Section 8, abstract last sentence, contribution 5 | R1.5, R3 fix |
| Llama-3.1 | Table 1, Section 5, Scope limitation | R1 third family |
| support (zero against tiny) | Section 6 or a new short section | R2.5 |
| region composition | Section 7 | R3.9 |
| figures, novelty table | throughout; Related Work | R3.1, R1.1 |

---

## Prioritised fix list

| # | fix | where | needs |
|---|---|---|---|
| 1 | Repair the `\ref` and `\texttt` corruption and the minus sign | main.tex l.53 to 54, l.149, Section 8 | writing |
| 2 | Replace the intro example with a recognised, robustly unreachable entity and the real most frequent answer (the Chinese-script Tagore answer is itself a vivid example) | Introduction | writing, CPU lookup |
| 3 | Report RBU on AUROC-passing categories and under the second probe, plus RBU-all; run E13 | Section 3, App. B, Limitations | CPU (existing G0_a3 and G0_5090 data), GPU for E13 |
| 4 | Report Qwen's top-k rise and soften "mass, not rank" | Section 5 | writing |
| 5 | Zero against tiny: untruncated probability of unreachable recognised names and expected samples to surface them | Section 6 | CPU |
| 6 | Popularity GEE with token count and type controls; within-model mechanism statistics | Sections 5 and 7 | CPU |
| 7 | Rewrite the estimator claim as asymptotic; decompose the shortfall; say the species set is the capped list | Section 6 | writing |
| 8 | Validate the invented-name metric: multilingual Wikidata labels, hand audit, then reread the frontier | Section 8, App. E | CPU, network, human |
| 9 | Cite Qwen2.5, OLMo-2, Wikidata, DPO, top-k; strip bib notes; add verified theme references and the synthetic-data argument | Related Work, Sections 1 and 5, references.bib | writing, verification |
| 10 | Checklist and anonymity: licences, total compute, annotation details, AI-assistance answer, remove commit hashes or the repository mention | new Ethics paragraph, App. F | writing, human decision |
| 11 | Report pre-registration deviations: RBU-mean headline, GEE, Holm, G0 option B sign-off, V4 re-read | Sections 3, 4 and 7 | writing |
| 12 | Add the full-scale V1 (554,000 and 277,000 samples, 0 violations); qualify the CPU float32 check to the 0.5B model; fix the "two thousand" and 94.1 against 95.1 inconsistencies; give p-values; "verify" not "prove" | Sections 2, 4 and 5 | writing |
| 13 | Figures (4 to 5) and use the 8 pages | throughout | writing (figures fork) |
| 14 | Novelty table against Awad et al. | Related Work or Introduction | writing (novelty fork) |
| 15 | Pending runs: E13, E14, E15/G3b, Llama-3.1 | Sections 3, 4, 5, 8 | GPU |
