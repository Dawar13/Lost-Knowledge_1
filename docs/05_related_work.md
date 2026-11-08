# MB5 related work and collision map

**Rule:** this is a lead list assembled during idea selection. Before any entry
goes into the paper, open its arXiv or ACL Anthology page, confirm title,
authors, year and venue, and confirm the claim we attribute to it. Classic
citations marked (memory) were not re-checked at all. Re-run the collision
search (arXiv, OpenReview ICLR 2027 submissions, ACL Anthology, Semantic
Scholar) at P1 start and again in the final 48 hours before submission.

## Closest work: what each does and what it leaves us

| Work | What it does | What it does not do (our gap) |
|---|---|---|
| Lost in Sampling: Word Coverage Score (arXiv 2605.27268) | Same exact token-survival test as ours, for ordinary words in book contexts; reports large shares of words unreachable at default settings; documents shipped defaults of several open models | No knowledge, no recognition test, no post-training analysis, no estimator consequence |
| The Truncation Blind Spot (arXiv 2603.18482) | Shows a share of human-written tokens fall outside common truncation boundaries; content words cut more than function words (machine-text detection setting) | Not about knowledge or entities |
| Empty Shelves or Lost Keys? (arXiv 2602.14080) | Models encode most facts but fail to recall many; recall gap larger for rare facts | Samples at temperature 1 without truncation; no reachability |
| Inside-Out (arXiv 2503.15299) | Models can rank the gold answer first internally yet never generate it in 1,000 samples | No truncation analysis; explains a low-probability gap, not a zero-probability one |
| KnowSum (arXiv 2506.02058) | Estimates unseen knowledge from samples with smoothed Good-Turing | Its estimates measure reachable support (our section 4 correction) |
| Long-tail knowledge survey (arXiv 2602.16201) | Names decoding-time truncation as a possible mechanism | Does not measure it |
| Artificial Hivemind (arXiv 2510.22954, NeurIPS 2025 Best Paper) | Intra- and inter-model homogeneity on open-ended prompts | Similarity at fixed sample size; does not separate decoding from model |
| Multilingual Artificial Hivemind (arXiv 2610.16755) | Homogenisation across 13 languages | No reachability, no knowledge |
| Verbalized Sampling (arXiv 2510.01171) | Typicality bias in preference data; prompting restores diversity | Suggests suppressed mass survives; we show where it becomes exactly zero |
| Where does output diversity collapse in post-training? (arXiv 2604.16027) | Stage-wise diversity loss in Olmo 3 | Similarity metrics; no truncation; no knowledge |
| Tracing Multilingual Factual Knowledge Acquisition (arXiv 2505.14824) | Recall in OLMo driven by fact frequency | Greedy or plain decoding |

## Theory and background

- Kalavasis, Mehrotra, Velegkas, On the Limits of Language Generation: Trade-Offs Between Hallucination and Mode Collapse (arXiv 2411.09642). Breadth versus validity in the limit.
- Holtzman et al., The Curious Case of Neural Text Degeneration, ICLR 2020 (arXiv 1904.09751) (memory). Nucleus sampling.
- Hewitt, Manning, Liang, Truncation Sampling as Language Model Desmoothing, Findings of EMNLP 2022 (arXiv 2210.15191) (memory). Epsilon and eta sampling; the cleanest prior statement of what truncation is for.
- Nguyen et al., min-p sampling (arXiv 2407.01082) (memory).
- Lee et al., Factuality Enhanced Language Models for Open-Ended Text Generation, NeurIPS 2022 (arXiv 2206.04624) (memory). Factual-nucleus sampling: a truncation designed for factuality; directly relevant to our mitigation frontier.
- Kandpal et al., Large Language Models Struggle to Learn Long-Tail Knowledge, ICML 2023 (arXiv 2211.08411) (memory).
- Mallen et al., When Not to Trust Language Models (PopQA), ACL 2023 (arXiv 2212.10511) (memory).
- Kirk et al., Understanding the Effects of RLHF on LLM Generalisation and Diversity, ICLR 2024 (arXiv 2310.06452) (memory).
- ECLeKTic (arXiv 2502.21228): cross-lingual knowledge transfer; context for E8.

## Positioning sentence for the paper

Prior work shows that models know more than they say (recall gaps) and that truncation removes words (coverage), but none measures whether truncation makes **recognised knowledge impossible to say**, under the samplers models ship with, nor what that does to every sample-based estimate of model knowledge.

## Reviewer objections to pre-empt

1. *"Nobody uses top-p for factual questions."* Chat defaults ship with truncation and users never see them; transformers adds top-k=50 when a config sets nothing. We report full curves, not one setting.
2. *"Recognition is easier than recall; this is trivial."* Our quantity is probability exactly zero, not low recall. Low recall can be fixed by sampling more; zero cannot.
3. *"Tokenisation artefact."* Aliases count; non-canonical paths are audited (E4, E7).
4. *"Your fix increases hallucination."* We report the frontier, not a single win; the invented-name rate is hand-audited.
5. *"Yes/no recognition is pattern matching."* Foils come from sibling categories in the same region and language family, the threshold is set on foils, and recognition also requires the 4-way choice.
