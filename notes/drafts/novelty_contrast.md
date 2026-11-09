# Novelty contrast: "Lost in Sampling" and two other overlaps (draft, fork D, 9 November 2026)

Sources read in full on 9 November 2026: arXiv 2605.27268v2 HTML (submitted 26 May 2026, v2 21 August 2026; arXiv comment "15 pages, 6 figures. Accepted to Findings of EMNLP 2026"); arXiv 2506.02058 HTML (KnowSum); arXiv 2510.22954 HTML (Artificial Hivemind). Our side: paper/main.tex (abstract, introduction, sections 2 to 7, related work), results/G1/stages.csv, results/G0_5090/Qwen2.5-7B-Instruct/meta.json, docs/05b_reference_verification.md. Section numbers below refer to the arXiv HTML of each paper.

## 1. Side by side: Awad et al. (LiS) against this paper

| dimension | Lost in Sampling (Awad et al., 2605.27268v2) | this paper |
|---|---|---|
| Object | "contextually appropriate human vocabulary": words ranked 10,000 to 40,000 in the Google Web Trillion Word Corpus, filtered by the Moby lists (s3.4) | named entities given as answers to knowledge prompts ("Name a river in Kerala"): 11 pilot categories; full TailCat 277 categories, 33,437 members |
| Contexts | 256-token prefixes from PG-19 books before 1919, checked by gemini-2.5-flash (s3.5); 100 words x 10 contexts per model pair (s4.2) | chat-templated question prompts (3 wordings at scale, 5 in the robustness run); instruct models with their own chat template |
| Test | "Forced-Path Audit": teacher-force the word, record rank and probability of each sub-token (s3.1); a word is covered "if, and only if, every constituent token t_i survives the filter at its respective step" (s3.2, Eqs. 1 to 3) | the same per-token survival logic. Differences: we run the library's own processor chain at every position for combined (shipped) samplers; we score up to three aliases; we record critical values p*, k*, min-p* |
| Samplers | top-p {0.7 to 0.99}, top-k {1 to 20}, min-p {0.01 to 0.1}, at T 0.7, 1, 1.5 (s4.3); combined filter for the default rows of Table 2 | 11 headline samplers plus each model's shipped chain; 14-sampler frontier in P5 |
| Shipped defaults | Table 2: Qwen2.5-14B-Instruct p 0.8, k 20, T 0.7; Llama-3.1-8B-Instruct p 0.9 with T 0.7 used as "the nearest evaluated temperature" to the documented 0.6 (Table 2 caption); no repetition penalty mentioned anywhere in the text | the exact generation_config chain, e.g. Qwen2.5-7B-Instruct: temperature 0.7, top-k 20, top-p 0.8, repetition penalty 1.05 (results/G0_5090/Qwen2.5-7B-Instruct/meta.json) |
| Knowledge or recognition | explicitly not tested: "Rather than assessing static knowledge, the WCS measures the lexical survival rate..." (Abstract); no recognition test (the text never mentions recognition or entities) | yes/no against near-miss foils at the foils' 95th percentile, plus a four-way choice; RBU is defined over recognised entities |
| Models | 7 base/aligned pairs under 20B: Llama-3.1-8B, Mistral-7B-v0.3, Qwen3.5-9B, Qwen2.5-14B, Gemma-3-12B, Gemma-4-E4B, DeepSeek-R1-Distill-Qwen-14B against Qwen2.5-14B (Table 1) | Qwen2.5-7B base and instruct; OLMo-2-1124-7B base, SFT, DPO, Instruct; Qwen2.5-Instruct 0.5B, 1.5B, 3B (exploratory); Llama-3.1-8B pending |
| Post-training stages | base against final aligned only; plus a "Distillation Deficit" for DeepSeek-R1-Distill (s4.1) | four OLMo-2 stages; most of the rise at SFT |
| Base vs aligned result | small on average: "mean word-level reachability decreases from 0.740 for base models to 0.728 for instruct models, while WCS decreases from 0.290 to 0.277"; Gemma-3-12B-it is an exception (s5) | large in the answer slot: robust RBU at top-p 0.9 on the pilot, Qwen base 0.524 against instruct 0.951; OLMo base 0.544 against Instruct 0.938 (results/G1/stages.csv) |
| Mechanism | none measured; the introduction attributes collapse to alignment (RLHF, DPO) by citation (s1) | first failing token moves to the first token across stages; first-token top-1 mass tracks RBU (Spearman 0.817); top-k RBU flat while top-p RBU rises ("mass, not rank") |
| Estimator or coverage consequence | none; s6.1 calls the binary WCS "a conservative lower bound on the true scale of word erosion" | unseen-species estimators converge to the reachable set; exact gap (Chao1 141 against 15,514 recognised, Qwen shipped, full TailCat) |
| Who loses | words from one frequency band only; no within-band frequency analysis (s3.4; Limitations "Boundary Bounds of Frequency Bands") | GEE on log sitelinks within 277 categories (odds ratio 0.149 per tenfold, Qwen instruct); region and type breakdowns |
| Fix | discussion only: alternative samplers, "semantic-guided decoders", training objectives with reachability penalties (s6.4); no experiment | first-token relaxation tested (G3 fail: 12.3% and 23.7% recovered); full frontier with invented-name rate; sample-then-self-check (G3b) pre-registered, not yet run |
| Numeric robustness and validation | none: the text never mentions numeric precision, ties or a check against real samples | calibrated margin epsilon; 831,000 real samples at scale with 0 robust violations |
| Language | English only (Limitations) | English prompts; categories from 24 countries and 12 Indian-language writer lists; multilingual prompts not run |
| Recursive-training argument | yes, for words: "future generations of language models will recursively ingest this flattened discourse" (s6.3) | the same argument for knowledge (the special theme) |
| Closed models | stated impossible: "this protocol cannot be extended to proprietary, closed-source models" (Limitations) | same limitation |

## 2. Verdict

**What survives.** The test itself is not ours, and LiS also shows that aligned models usually lose more reachability than base models. Our contribution survives on five points that LiS does not touch:

1. **Knowledge, not vocabulary.** We measure entities the model demonstrably recognises against near-miss foils. LiS says outright that it does not assess knowledge.
2. **Mechanism across stages.** LiS compares base and final models only and measures no mechanism. Our stage-wise OLMo-2 result (most of the rise at SFT, the cut moving to the first token, mass not rank) is new for reachability.
3. **Consequence for measurement.** No estimator argument in LiS.
4. **Who loses within the tail, at scale.** LiS fixes one frequency band.
5. **Exactness to the deployed chain, a numeric margin and sampling validation.** LiS approximates Llama's temperature, never mentions the repetition penalty and has no validation against real samples.

There is also a substantive contrast in size: for book continuations LiS reports a small average base-to-aligned drop, while we see RBU nearly double in the answer slot of knowledge prompts. The metrics and settings differ, so state the contrast qualitatively and do not compare the numbers on one scale.

**Claims we must NOT make:**
- That we introduce or are first with an exact, one-pass, sampling-free reachability test. LiS does this (s3.1 to s3.2), and Garces Arias et al. do it at token level.
- That we are first to show shipped defaults erase content, or first to tabulate them (LiS Table 2).
- That we are first to show aligned models make more content unreachable than base models. LiS shows it for words in most families. We can claim the stage-wise mechanism and the size in the knowledge setting.
- That we are first to link truncation to recursive homogenisation (LiS s6.3).
- That knowledge-estimation work has not noticed truncation. KnowSum ablates nucleus sampling and switches it off (see section 3).
- That post-training concentration at the answer slot is our discovery. Jang et al. (Findings of EMNLP 2026) report base-vs-instruct collapse to one answer and a top-two logit gap above 14 nats; Parikh (arXiv 2607.12796) reports convergence on "Name a tree."-type prompts across 105 models; Karouzos et al. locate diversity collapse at SFT for Olmo 3. Our claim is narrower: where the cut falls and how first-token concentration tracks exact unreachability of recognised knowledge across stages.
- That "sample, then verify" is a new idea. If G3b passes, the claim is that the model's own recognition can replace truncation for recall coverage, measured exactly; any prior generate-then-verify work must be found and verified before we write this.

## 3. KnowSum and Artificial Hivemind, rechecked with quotes

**KnowSum (Li, Xin, Long, Su; arXiv 2506.02058).**
- Section 5.2, "Effect of temperatures and sampling methods": "We consider two key factors: the temperature Temp in {0.7, 1} and the use of nucleus sampling ... which truncates the token distribution to the smallest set of tokens whose cumulative probability exceeds 0.9." Then: "using a higher temperature and disabling truncation yields the highest values for both N_seen and N_tot. Therefore, we adopt a temperature of 1 and disable nucleus sampling in all experiments to best capture each model's capabilities."
- Appendix B.8 repeats this for diseases.
- Caution: KnowSum's "truncation level k" (s5.2 and Appendix B.5) is the estimator's frequency cut-off, not top-k sampling. Do not cite it as a top-k result.
- Section 6 (Discussion): "a more ambitious direction is to move beyond estimation and toward extraction, actively surfacing knowledge that the model encodes but rarely generates." This is a good hook for our fix section.
- Appendix B.9: models are "prompted to list 20 theorem names over 30,000 runs". Their prompts are list prompts, which is relevant to our list control (E14).
- Overlap with our estimator claim: they show empirically that nucleus sampling lowers the estimate. We add the reason (estimators converge to the support of the sampling distribution), the exact size of the gap, and the point that top-k and shipped chains remain even when nucleus sampling is off. Our current related-work sentence already credits their ablation; keep it.

**Artificial Hivemind (Jiang et al.; arXiv 2510.22954; NeurIPS 2025 Datasets and Benchmarks).**
- Appendix C.1: "For the top-p setup, we use p=0.9, temperature=1.0, and a maximum generation length of 2048 tokens. For the minimum-p setup, we use p=1.0, min-p=0.1, temperature=2.0".
- Section 3: "while min-p reduces extreme repetition (fewer pairs above 0.9), 81% of response pairs still exceed 0.7 similarity and 61.2% exceed 0.8, revealing mode collapse even under diversity-oriented decoding."
- Overlap with our min-p claims: they test min-p 0.1 at temperature 2 for homogeneity of open-ended text, not reachability or knowledge. Our related-work sentence "find homogeneity persists under min-p" is accurate. Do not claim min-p has gone untested for homogeneity.

## 4. Proposed related-work paragraph (replaces "Truncation and what it removes")

> **Truncation and what it removes.** Nucleus sampling \citep{holtzman2020curious}, epsilon and eta sampling \citep{hewitt2022truncation} and min-$p$ \citep{nguyen2025minp} all give up the tail of the distribution for fluency, and top-$p$ can harm factuality \citep{lee2022factuality}. The closest work is \citet{awad2026lost}. They use the per-token survival test we also use. They ask which mid-frequency English words remain reachable in book continuations, compare seven base and aligned pairs, and tabulate shipped defaults. Aligned models erase more words in most families, by a small average margin. \citet{garcesarias2026truncation} show that human-chosen tokens often fall outside common truncation sets. We change the object and the question. We test entities the model recognises against near-miss foils, in the answer to a knowledge prompt, under the exact deployed processor chain, with a numeric margin and a check against real samples. We then ask why post-training changes reachability, stage by stage, and what follows for knowledge estimates.

> **Collapse at the answer slot.** Instruction-tuned models give one answer where a distribution is wanted, with a large top-two logit gap at the answer position \citep{jang2026instruction}. Across 105 models, single-answer category prompts such as ``Name a tree.'' converge on one answer, most strongly for heavily post-trained assistants \citep{parikh2026census}. Post-training also multiplies confident errors on long-tail factual questions \citep{huang2026alignment}. Our first-token result connects this concentration to an exact consequence: the recognised entities it pushes out of the kept set have probability zero.

(The bib keys jang2026instruction, parikh2026census and huang2026alignment are proposals; entries are in notes/drafts/collision_2026-11-09.md.)

## 5. Proposed contribution list for the introduction (replaces the current five items)

1. **Recognised but unreachable.** We pair the per-token survival test of \citet{awad2026lost} with a recognition test against near-miss foils. We run each model's deployed processor chain exactly and add a calibrated numeric margin. Real samples never contradict a robust verdict: no violations in 831,000 samples at full scale (Section~\ref{sec:method}).
2. **How much, at deployed settings.** Among entities Qwen2.5-7B-Instruct recognises, \gzQwenInstShipped{} of the pilot entities are unreachable under its shipped sampler, against \gzQwenBasePnine{} for its base model at top-$p$ 0.9 (Section~\ref{sec:howmuch}).
3. **Where it comes from.** Across OLMo-2's post-training stages most of the rise happens at supervised fine-tuning. The cut moves to the first token of the name, and truncation by probability mass, not rank, does the deleting (Section~\ref{sec:where}).
4. **What it does to measurement.** Sample-based knowledge estimators converge to the reachable set, not the recognised set. We compute the gap exactly: Chao1 gives \peQwenShippedChao{} against \peQwenShippedRecog{} recognised (Section~\ref{sec:measure}).
5. **Who loses, and what a fix costs.** Rarer recognised entities are deleted more: odds ratio \ptOrUnreachQwenInst{} per tenfold Wikipedia coverage. Relaxing truncation at the first token recovers only \pfRecoveredQwen{}; we report the frontier against invented answers (Section~\ref{sec:fix}). Update this item if G3b passes.

Item 1 must not say "we introduce an exact test". The 831,000 figure (554,000 Qwen plus 277,000 OLMo, notes/decisions.md G2 entry) comes from results/P3; replace it with a P3 macro before it enters the paper.

## 6. Abstract sentence to change

Current: "We give an exact, sampling-free test of this, one forward pass per answer, with a numeric margin that makes the verdict robust to floating-point differences between implementations."

Proposed: "Building on an exact per-token survival test \citep{awad2026lost}, we add a recognition test against near-miss foils and a numeric margin that makes the verdict robust to floating-point differences, and we check it against hundreds of thousands of real samples."

(An abstract may not allow a citation in the ACL style; if not, name the paper in the introduction's first contribution and drop "Building on" from the abstract only if the introduction credits the test in its first two paragraphs.)
