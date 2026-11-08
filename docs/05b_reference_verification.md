# MB5 reference verification

**Date of checking:** 8 November 2026.

**How each item was checked.** For every arXiv item, the metadata (title, full
author list, comments, journal reference, abstract) was pulled from the arXiv
export API and cross-checked against the `citation_*` meta tags of the
`https://arxiv.org/abs/<ID>` page itself. Where a claim in
`docs/05_related_work.md` goes beyond the abstract (settings, data, defaults),
the arXiv HTML full text was searched for the relevant passage. Venue claims not
stated on the arXiv page were checked on the venue's own page (NeurIPS, ICLR,
PMLR, ACL Anthology). Non-arXiv items were checked against Crossref DOI records
and the publisher page where it could be opened.

**Verdict scale.** *Supported*: the abstract or full text says it. *Partly
supported*: the core is right but a detail differs (stated). *Not supported /
cannot confirm*: the source does not say it.

**Summary.** 26 of 26 references resolve and were verified (20 from
`docs/05`, 6 standard references). No arXiv ID failed to resolve and no title
differs materially from the name in `docs/05`, so there is **no MISMATCH**.
Every "what it does" claim is supported or partly supported. Three
"our gap" statements in `docs/05` are contradicted by the papers and need
rewording before they go into the paper (see section 4).

## 1. Closest work (docs/05, first table)

| Work as named in docs/05 | arXiv ID | Verified title | Authors | Year / venue | Claim check | Notes |
|---|---|---|---|---|---|---|
| Lost in Sampling: Word Coverage Score | 2605.27268 (v2) | Lost in Sampling: Assessing Lexical Reachability in LLMs via the Word Coverage Score (WCS) | Samer Awad, Javier Conde, Carlos Arriaga, et al. (6) | 2026; arXiv comment: "Accepted to Findings of EMNLP 2026" | **Supported** (what it does). **Gap column partly wrong.** | Full text confirms: multi-token word counts as reachable only if every sub-token survives at its step (product of indicators, top-k, top-p, min-p); contexts are 256-token prefixes from PG-19 books; Table 2 lists shipped or recommended defaults (Qwen2.5-14B-Instruct, Qwen3.5-9B, Llama-3.1-8B-Instruct, Gemma-4-E4B, DeepSeek-R1-Distill) with 22% to 57% of words erased. **But it does analyse post-training:** seven base vs aligned pairs; aligned models erase more words in most families (mean word-level reachability 0.740 to 0.728); a "distillation deficit" for DeepSeek-R1-Distill. So "no post-training analysis" must become "no mechanism for the post-training effect (no first-token mass analysis)". No knowledge, recognition test or estimator consequence: confirmed. Not yet located in the ACL Anthology. |
| The Truncation Blind Spot | 2603.18482 (v4) | The Truncation Blind Spot: How Decoding Strategies Systematically Exclude Human-Like Token Choices | Esteban Garces Arias, Nurzhan Sapargali, Christian Heumann, Matthias Aßenmacher | 2026; arXiv comment: "Accepted at INLG 2026" | **Supported** | Abstract: 8 to 18% of human-selected tokens fall outside common truncation boundaries, across five open models and three domains; content-word tokens excluded disproportionately; machine-text detection setting. Token-level, not word or entity level. Also reports that probability-floor samplers (epsilon, eta) narrow the blind spot, relevant to our fix. Not about knowledge or entities: confirmed. |
| Empty Shelves or Lost Keys? | 2602.14080 (v2) | Empty Shelves or Lost Keys? Recall Is the Bottleneck for Parametric Factuality | Nitay Calderon, Eyal Ben-David, Zorik Gekhman, Eran Ofek, Gal Yona | 2026; arXiv preprint, no venue stated | **Supported** (what it does). **Gap column partly supported.** | Abstract: encoding nearly saturated (GPT-5, Gemini-3 encode 95 to 98% of facts); recall is the bottleneck; failures disproportionately affect long-tail facts. Full text: "we generate eight responses per LLM with a temperature of 1". The paper does not state any truncation setting (no top-p, top-k or min-p mentioned), and many models are API models. Say "temperature 1, eight samples, truncation settings not reported", not "without truncation". No reachability analysis: confirmed. |
| Inside-Out | 2503.15299 (v4) | Inside-Out: Hidden Factual Knowledge in LLMs | Zorik Gekhman, Eyal Ben David, Hadas Orgad, et al. (8) | 2025; arXiv comment: "Accepted to COLM 2025" | **Supported** | Abstract: a model "can internally know an answer perfectly, yet fail to generate it even once, despite large-scale repeated sampling of 1,000 answers". Full text: in 9% of questions an internal probe scores the gold answer above every incorrect candidate; samples drawn at temperature 1; no truncation analysis. Note "internally" means a probe on hidden states, not the model's output probabilities. |
| KnowSum | 2506.02058 (v1) | Evaluating the Unseen Capabilities: How Many Theorems Do LLMs Know? | Xiang Li, Jiayi Xin, Qi Long, Weijie J. Su | 2025; arXiv preprint, no venue stated or found | **Supported** (what it does). **Gap column needs care.** | "KnowSum" is the framework name, not the title (not a mismatch). Full text: uses the smoothed Good-Turing (Efron-Thisted) estimator. **Important:** the paper already ablates decoding: temperature 0.7 vs 1 crossed with nucleus sampling (p = 0.9) on or off; truncation lowers both seen and estimated totals, so all main experiments use temperature 1 with nucleus sampling disabled. Top-k and API-side defaults are not mentioned. Our section 4 correction must engage with this ablation: their own result agrees that truncation lowers the estimate, and even untruncated the estimator only extrapolates what the sampling distribution reaches. |
| Long-tail knowledge survey | 2602.16201 (v1) | Long-Tail Knowledge in Large Language Models: Taxonomy, Mechanisms, Interventions and Implications | Sanket Badhe, Deep Shah, Nehal Kathrotia | 2026; arXiv preprint, no venue | **Supported** | Section 3.4.1 "Decoding-Time Probability Truncation": top-p and top-k "systematically silence the correct but rare answers". Cites Holtzman et al. and a factuality survey; no measurement. Not peer reviewed; cite only for the framing. |
| Artificial Hivemind | 2510.22954 (v2) | Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond) | Liwei Jiang, Yuanjun Chai, Margaret Li, et al. (10 on arXiv v2; 9 on the NeurIPS page) | 2025; NeurIPS 2025 Datasets and Benchmarks track, oral; **Best Paper (DB track)** | **Supported** (what it does). **Gap column partly wrong.** | Best Paper confirmed on neurips.cc/virtual/2025/poster/121421 ("Best Paper (DB track)") and the NeurIPS blog post of 26 November 2025. Write "NeurIPS 2025 Best Paper (Datasets and Benchmarks track)". Author list: arXiv v2 ("Camera-Ready Version") adds Alon Albalak; the NeurIPS pages list nine. Decoding: 50 samples per model at top-p 0.9, temperature 1; **it does test one alternative decoder** (min-p 0.1 at temperature 2) and finds homogeneity persists. So "does not separate decoding from model" overstates; say "fixed decoding settings plus one min-p check; no reachability". |
| Multilingual Artificial Hivemind | 2610.16755 (v1) | The Multilingual Artificial Hivemind: Do All LLMs Think Alike in Every Language? | Xaver Raphael Davey, Shubhra Mishra, Anton Adelöw, Siyao Liu | 2026; arXiv preprint, no venue | **Supported** | 23 open-weight models, 13 languages; homogenisation gap tracks language competence. One decoding setting throughout (temperature 1, top-p 0.9). No reachability, no knowledge: confirmed. |
| Verbalized Sampling | 2510.01171 (v4) | Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity | Jiayi Zhang, Simon Yu, Derek Chong, et al. (7) | 2026; **ICML 2026**, PMLR 306, pp. 160073-160152 (confirmed on proceedings.mlr.press/v306/zhang26hy.html) | **Supported** | Abstract: typicality bias in preference data drives mode collapse; a training-free prompt restores diversity (1.6 to 2.1x in creative writing). "Suggests suppressed mass survives" is our reading of "unlock pre-trained generative diversity"; fair, but flag it as interpretation. |
| Where does output diversity collapse in post-training? | 2604.16027 (v1) | Where does output diversity collapse in post-training? | Constantinos Karouzos, Xingwei Tan, Nikolaos Aletras | 2026; arXiv preprint, no venue | **Supported** | Three Olmo 3 lineages (Think, Instruct, RL-Zero), 15 tasks, four diversity metrics. Full text: sampling at temperature 0.6, top-p 0.95 throughout; one appendix compares temperature settings. Truncation is held fixed, never analysed. No knowledge: confirmed. |
| Tracing Multilingual Factual Knowledge Acquisition | 2505.14824 (v2) | Tracing Multilingual Factual Knowledge Acquisition in Pretraining | Yihong Liu, Mingyang Wang, Amir Hossein Kargaran, et al. (8) | 2025; Findings of EMNLP 2025, pp. 2121-2146 (ACL Anthology 2025.findings-emnlp.113) | **Supported** | OLMo-7B; recall driven mainly by fact frequency; some rare non-English facts recalled via crosslingual transfer. Full text: vLLM, greedy decoding, at most 10 output tokens. |

## 2. Theory and background (docs/05, second list)

| Work as named in docs/05 | arXiv ID | Verified title | Authors | Year / venue | Claim check | Notes |
|---|---|---|---|---|---|---|
| Kalavasis, Mehrotra, Velegkas | 2411.09642 (v3) | On the Limits of Language Generation: Trade-Offs Between Hallucination and Mode Collapse | Alkis Kalavasis, Anay Mehrotra, Grigoris Velegkas | 2024 preprint; arXiv comment: STOC 2025 | **Supported** | Consistency with breadth is impossible for a large class of models including next-token predictors; achievable with negative examples. docs/05 gave no venue; add STOC 2025. |
| Holtzman et al. | 1904.09751 (v2) | The Curious Case of Neural Text Degeneration | Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi | 2020; ICLR 2020 (arXiv comment) | **Supported** | Introduces nucleus sampling. Was marked "(memory)"; now checked. |
| Hewitt, Manning, Liang | 2210.15191 (v1) | Truncation Sampling as Language Model Desmoothing | John Hewitt, Christopher D. Manning, Percy Liang | 2022; Findings of EMNLP 2022, pp. 3414-3427 (ACL Anthology 2022.findings-emnlp.249) | **Supported** | Abstract names eta-sampling; full text section 4.3 also proposes epsilon-sampling ("(ours)"). Truncation framed as desmoothing, estimating a subset of the true support. |
| Nguyen et al., min-p sampling | 2407.01082 (v8) | Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs | Minh Nhat Nguyen, Andrew Baker, Clement Neo, et al. (6) | 2025; ICLR 2025, oral (arXiv journal-ref) | **Supported** | docs/05 gave no title or venue; both added here. |
| Lee et al., Factuality Enhanced LMs | 2206.04624 (v3) | Factuality Enhanced Language Models for Open-Ended Text Generation | Nayeon Lee, Wei Ping, Peng Xu, et al. (7) | 2022; NeurIPS 2022 (arXiv comment) | **Supported** | Abstract: top-p "can harm the factuality"; proposes factual-nucleus sampling. |
| Kandpal et al. | 2211.08411 (v2) | Large Language Models Struggle to Learn Long-Tail Knowledge | Nikhil Kandpal, Haikang Deng, Adam Roberts, Eric Wallace, Colin Raffel | 2023; ICML 2023 (arXiv comment: camera-ready) | **Supported** | QA accuracy tracks count of relevant pre-training documents. |
| Mallen et al. (PopQA) | 2212.10511 (v4) | When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories | Alex Mallen, Akari Asai, Victor Zhong, et al. (6) | 2023; ACL 2023 (Long), pp. 9802-9822 (ACL Anthology 2023.acl-long.546) | **Supported** | Introduces PopQA (14k questions); LMs struggle with less popular facts. |
| Kirk et al. | 2310.06452 (v3) | Understanding the Effects of RLHF on LLM Generalisation and Diversity | Robert Kirk, Ishita Mediratta, Christoforos Nalmpantis, et al. (7) | 2024; ICLR 2024, poster (confirmed on iclr.cc/virtual/2024/poster/18704) | **Supported** | RLHF generalises better than SFT but significantly reduces output diversity. Venue not on the arXiv page; confirmed on the ICLR site. |
| ECLeKTic | 2502.21228 (v3) | ECLeKTic: a Novel Challenge Set for Evaluation of Cross-Lingual Knowledge Transfer | Omer Goldman, Uri Shaham, Dan Malkin, et al. (14) | 2025; arXiv preprint, no venue stated or found | **Supported** | Closed-book QA in 12 languages built from Wikipedia article presence and absence. Cite as a preprint. |

## 3. Standard references

| Reference | Identifier | Verified title | Authors | Year / venue | Where confirmed | Notes |
|---|---|---|---|---|---|---|
| OLMo 2 technical report | arXiv 2501.00656 (v3) | 2 OLMo 2 Furious | Team OLMo, Pete Walsh, Luca Soldaini, et al. (Team OLMo plus 42 named authors) | 2024 (first submitted 31 Dec 2024); arXiv comment: "Shorter version accepted to COLM 2025" | https://arxiv.org/abs/2501.00656 | Lead ID correct. v3 adds 32B results. |
| Qwen2.5 technical report | arXiv 2412.15115 (v2) | Qwen2.5 Technical Report | Qwen: An Yang, Baosong Yang, Beichen Zhang, et al. (42 named authors) | 2024; arXiv preprint | https://arxiv.org/abs/2412.15115 | Lead ID correct. arXiv shows the author field as "Qwen: An Yang, ...". |
| Wikidata | DOI 10.1145/2629489 | Wikidata: A Free Collaborative Knowledgebase | Denny Vrandečić, Markus Krötzsch | 2014; Communications of the ACM 57(10): 78-85 | https://cacm.acm.org/research/wikidata/ (title, issue) and Crossref record for the DOI (pages) | Crossref stores the title as "Wikidata" only; the CACM page gives the full title. |
| Chao (1984) | JSTOR 4615964 (no DOI) | Nonparametric Estimation of the Number of Classes in a Population | Anne Chao | 1984; Scandinavian Journal of Statistics 11(4): 265-270 | OpenAlex record W1558982506, landing page https://www.jstor.org/stable/4615964 | The JSTOR page itself returned a bot challenge and could not be opened. Details agree across OpenAlex and the CRAN SPECIES documentation. Re-open the JSTOR page by hand before submission. |
| Chao and Lee (1992), ACE | DOI 10.1080/01621459.1992.10475194 | Estimating the Number of Classes via Sample Coverage | Anne Chao, Shen-Ming Lee | 1992; Journal of the American Statistical Association 87(417): 210-217 | Crossref record for the DOI | |
| Liang and Zeger (1986), GEE | DOI 10.1093/biomet/73.1.13 | Longitudinal Data Analysis Using Generalized Linear Models | Kung-Yee Liang, Scott L. Zeger | 1986; Biometrika 73(1): 13-22 | Crossref record for the DOI | |

## 4. Items that matter most

**MISMATCH:** none. All 20 arXiv IDs in `docs/05` resolve to the paper named.

**Claims in docs/05 that are not right as written (all in the "our gap" column):**

1. **Lost in Sampling, "no post-training analysis".** Wrong. The paper compares
   seven base and aligned pairs and reports that alignment erases more words in
   most families, plus a distillation deficit. What it lacks is a mechanism
   (first-token mass) and any knowledge, recognition or estimator angle. It also
   already uses the exact multi-token survival test and tabulates shipped
   defaults. It is the closest prior work by far; our novelty must rest on
   *recognised knowledge*, the post-training *mechanism*, the *estimator
   consequence*, *who loses*, and the *fix*, not on the test itself.
2. **KnowSum, "its estimates measure reachable support".** Defensible, but the
   paper already shows that nucleus sampling (p = 0.9) lowers its estimates and
   so disables nucleus sampling in all main runs. Our correction must cite that
   ablation and argue from it (and from the fact that top-k and API-side
   defaults are unstated), not present truncation sensitivity as unnoticed.
3. **Artificial Hivemind, "does not separate decoding from model".** Overstated.
   It checks one diversity-oriented decoder (min-p at temperature 2) and finds
   homogeneity persists. Reword to "fixed decoding settings with one min-p
   check; no reachability".

**Smaller corrections:**

- Empty Shelves: say "temperature 1, eight samples, truncation not reported",
  not "without truncation".
- Artificial Hivemind: the award is "Best Paper (Datasets and Benchmarks
  track)". The arXiv camera-ready lists ten authors (adds Alon Albalak); the
  NeurIPS site lists nine. The BibTeX follows arXiv v2.
- Venues to add: Kalavasis et al. (STOC 2025), min-p (ICLR 2025), Kirk et al.
  (ICLR 2024), Verbalized Sampling (ICML 2026), Lost in Sampling (Findings of
  EMNLP 2026, per arXiv comment only), Truncation Blind Spot (INLG 2026, per
  arXiv comment only), Inside-Out (COLM 2025, per arXiv comment only).
- No venue found for KnowSum, ECLeKTic, Empty Shelves, the long-tail survey,
  the Multilingual Hivemind or the Olmo 3 diversity paper; cite as preprints.

## 5. Collision search

**Scope.** About 10 minutes on 8 November 2026: arXiv API keyword and phrase
queries (unreachable, top-p, min-p, truncation sampling, knowledge, entities,
long-tail, unseen knowledge, Good-Turing, hidden knowledge, reachability,
generator-validator); general web search for 2025 and 2026 work on truncation
and knowledge, sampling-based knowledge estimators and truncation, and
ICLR 2027 submissions. Semantic Scholar's API rate-limited after one query, so
it was covered only through general web search. OpenReview ICLR 2027
submissions were not searched directly.

**Result.** Nothing found that measures whether truncation samplers make
*recognised knowledge* exactly unreachable, nor any work that corrects
sample-based knowledge estimators for truncation. The closest work remains the
two items already in `docs/05`.

| ID | Title | One-line summary | How close to our claim |
|---|---|---|---|
| 2605.27268 | Lost in Sampling: Assessing Lexical Reachability in LLMs via the Word Coverage Score (WCS) | Exact per-step survival test for multi-token words under top-k, top-p, min-p in PG-19 contexts; shipped defaults erase 22 to 57% of words; aligned models usually worse than base. | **Very close on method.** Same exactness argument and one-forward-pass logic, same samplers, shipped defaults, base vs aligned. Differs on object (ordinary words, not recognised entities), no recognition test, no estimator, no multilingual or popularity analysis. Already in docs/05. |
| 2603.18482 | The Truncation Blind Spot | 8 to 18% of human tokens fall outside common truncation sets; probability-floor samplers narrow the gap. | Close on mechanism, token-level, detection setting. Already in docs/05. |
| 2602.16201 | Long-Tail Knowledge in Large Language Models (survey) | Names decoding-time truncation as a mechanism that silences rare correct answers. | States our hypothesis without measuring it. Already in docs/05. |
| 2604.11012 | Min-k Sampling: Decoupling Truncation from Temperature Scaling via Relative Logit Dynamics (ACL 2026 per arXiv comment) | New truncation sampler. Its limitations section (checked in full text) says that for "extremely long-tail knowledge entities ... aggressive truncation (by any method) might inadvertently filter out the correct answer". | One-sentence acknowledgement of our effect; no measurement. Useful as evidence the risk is recognised but unmeasured. |
| 2310.01693 | Closing the Curious Case of Neural Text Degeneration (Finlayson et al.) | Proves threshold truncation guarantees non-zero true probability for sampled tokens but "necessarily discard[s] some tokens with nonzero true probability". | Theory background for why the zero set is unavoidable. Not a collision. Not yet in docs/05; worth adding after venue check (search results point to ICLR 2024; not opened). |
| 2609.15476 | Temperature Fragility and the Conditional Benefits of Truncation Sampling | 13 open models on GSM8K and MMLU-Pro; deployed defaults cluster between temperature 0.6 and 1.0; truncation helps only where high temperature degrades accuracy. | Adjacent: truncation at deployed defaults, but accuracy, not reachability or knowledge. |
| 2605.07153 | Beyond Reasoning: Reinforcement Learning Unlocks Parametric Knowledge in LLMs | RL moves correct answers from the low-probability tail into greedy outputs; answers never seen in 128 pre-RL samples drive most of the gain. | Adjacent to "knows but cannot say" (low probability, not zero); no truncation. |
| 2412.20846 | Are LLMs Really Not Knowledgeable? Mining the Submerged Knowledge in LLMs' Memory | Correct answers often sit among high-probability candidates without being selected; proposes Hits@k. | Adjacent to "knows but cannot say" via token distributions; no truncation or exactness. Search results point to ICLR 2026; not opened. |
| ACL Anthology 2026.findings-emnlp.2278 | Instruction-Tuned Language Models Cannot Sample from Distributions They Can Describe (Jang, Kim, Lee; Findings of EMNLP 2026) | Instruction-tuned models collapse to one answer when sampled but can describe the distribution ("KNOWS/DOES split"); links this to post-training. | Adjacent framing (knows vs does, post-training); about survey response distributions, no truncation, no facts. |
| 2610.22754 | Refusal Is Not Robustness: Measuring Harmful Reachability in Autoregressive Language Models | Exact probabilistic model checking of how much probability mass reaches harmful continuations beneath a sampled refusal. | Analogous "exact reachability versus sampled output" idea in safety; truncation not mentioned in the abstract; not about knowledge. |
| 2602.00377 | DecompressionLM: Deterministic, Diagnostic, and Zero-Shot Concept Graph Extraction from Language Models | Deterministic concept extraction that avoids decoding effects which "suppress long-tail concepts"; measures concept coverage. | Adjacent to the estimator angle (decoding-based probing under-counts the tail); not truncation-exact. |
| 2408.13586 | Balancing Diversity and Risk in LLM Sampling | Estimates the capacity of truncation methods against a prefix tree of human continuations. | Support-size view of truncation for open text; no knowledge. |

**Plain statement.** No 2025 or 2026 paper was found that (a) measures exact
unreachability of knowledge a model demonstrably recognises under shipped
truncation defaults, (b) separates "knows but cannot generate" from "does not
know" by an exact zero-probability test, or (c) corrects KnowSum-style
unseen-knowledge estimates for truncation. The nearest threat is Lost in
Sampling on method. Re-run this search in the final 48 hours before submission,
including OpenReview ICLR 2027 submissions and Semantic Scholar with an API key.

## 6. Notes for `paper/references.bib`

- Contains only the 26 items verified above. Collision-search hits are not
  included; add them only after their own check.
- Keys follow `firstauthorYEARword`. The year is the venue year where one is
  confirmed, otherwise the arXiv first-submission year.
- Booktitles for items whose venue comes only from an arXiv comment use the
  short form given there (for example "NeurIPS 2022"); add full proceedings
  names and pages from the proceedings pages before camera-ready.
- Compound surnames are braced as the papers use them: Garces Arias (the
  Truncation Blind Spot paper cites itself this way), Ben David (Inside-Out),
  Ben-David (Empty Shelves). Team OLMo and Qwen are braced as corporate authors.
