# Collision search, 9 November 2026 (draft, fork D)

Scope: papers from May 2026 onwards on truncation sampling and knowledge, long-tail entities, unreachable or zero-probability outputs, diversity collapse after post-training, and unseen-species estimators for LLM knowledge. This extends the 8 November search in docs/05b_reference_verification.md (section 5); items already listed there are marked "known".

## How the search was run

- **Web search** (standard mode), 12 queries: truncation and long-tail entities; zero probability and post-training; unseen-species estimators and decoding; first-token concentration after instruction tuning; sampling defaults and reachability; knowledge collapse and list prompts; generation against recognition; OpenReview ICLR 2027; ACL Anthology 2026; sample-then-verify for recall; KnowSum follow-ups; papers citing Lost in Sampling.
- **arXiv API**, 27 phrase queries sorted by submission date, kept if submitted on or after 1 May 2026. Examples: "top-p" with knowledge; "truncation sampling"; "min-p"; unreachable with sampling; "long-tail" with decoding; "knowledge collapse"; "mode collapse" with "post-training"; "output diversity" with "post-training"; "zero probability"; "Good-Turing"; Chao1; "species richness"; "One-Word Census"; "reachable" with decoding; "homogenization" with knowledge.
- **Semantic Scholar** search returned only unrelated items.
- **DBLP** rate-limited (HTTP 429).
- **OpenReview ICLR 2027:** submissions could not be searched directly. Web search found no matching submission.
- **ACL Anthology:** reached through web search and direct page fetches.
- **Not done:** a direct OpenReview API sweep of ICLR 2027 submissions, and Semantic Scholar with an API key. Both go in the final 48-hour re-run.

**Result.** No paper found measures exact unreachability of knowledge a model demonstrably recognises, or corrects sample-based knowledge estimates for truncation. Lost in Sampling (known) is still the closest on method, and no paper citing it was found (aggregator pages list zero citations). The new risk is on the *mechanism* framing: two 2026 papers already show post-training collapse at the answer slot (rows 1 and 2), so our first-token result must be framed as linking that concentration to exact zero probability of recognised entities across stages.

## Close or adjacent hits (each opened and checked)

| # | item (verified link) | what it does | overlap | how to cite or contrast |
|---|---|---|---|---|
| 1 | Jang, Kim, Lee. Instruction-Tuned Language Models Cannot Sample from Distributions They Can Describe. Findings of EMNLP 2026, pp. 48757 to 48777. https://aclanthology.org/2026.findings-emnlp.2278/ (arXiv 2607.25292) | Instruct models describe a distribution but give one answer in over 94% of calls ("KNOWS/DOES split"); "the top-two logit gap at temperature zero exceeds 14 nats on some targets"; a base-vs-instruct comparison "establishes the alignment-induced direction" | **medium-high on mechanism** (answer-slot concentration after post-training); no truncation, no facts or recognition | cite in related work (collapse at the answer slot) and in Section 5 beside our first-token result. Known (05b section 5) but not yet in references.bib. Author order: the Anthology lists Jang, Kim, Lee; arXiv lists Jang, Lee, Kim. Use the Anthology order. |
| 2 | Parikh. The One-Word Census: Answer-Choice Conformity Across 105 Language Models. arXiv 2607.12796 (v4, July 2026). https://arxiv.org/abs/2607.12796 | 96 single-turn prompts that "each name a category with many valid one-word answers ("Name a tree.")", 8 samples per model, "requested temperature is 1.0"; "heavily post-trained assistants from the major labs are the most conformist" | **medium**: same prompt form as ours, sample level only; no truncation analysis (the text never mentions top-p or truncation), no knowledge test | cite in homogenisation related work. Our contrast: within one model, part of that convergence is exact (zero probability), and we can say which recognised answers can never appear. Follow-up by the same author: arXiv 2607.18476 (JSON output deepens convergence); cite only if needed. |
| 3 | Huang, Li, Wu, Zhou, Yu, Jia, Ma, Zhang. The Alignment Paradox: How Post-Training Amplifies Confident Hallucinations in Language Models. arXiv 2609.32617 (September 2026). https://arxiv.org/abs/2609.32617 | Across five families, instruct models multiply high-confidence errors (p at least 0.95) on long-tail factual queries "by more than an order of magnitude (10x to 35x)" against base models | **adjacent**: post-training concentrates mass on long-tail questions even when wrong; no truncation | cite in the mechanism discussion. It is also a named risk for the G3b self-check fix: confident wrong answers may pass the model's own check. |
| 4 | Wright, Masud, Moore, Yadav, Antoniak, Christensen, Park, Augenstein. What and Whose Knowledge? Measuring Epistemic Diversity in Large Language Models. arXiv 2510.04226 (v7; arXiv comment "Accepted to EMNLP 2026"). https://arxiv.org/abs/2510.04226 | Measures epistemic diversity (diversity of real-world claims) and frames low diversity as a knowledge-collapse risk | **theme** | cite in the introduction's theme paragraph. Its decoding settings were not checked; check them before saying anything about decoding. The title changed from earlier versions; use v7's. |
| 5 | Hodel, West. Epistemic diversity across language models mitigates knowledge collapse. arXiv 2512.15011 (v3). https://arxiv.org/abs/2512.15011 | Simulated recursive training; a diverse ecosystem of models resists knowledge collapse better than a monoculture | **theme** | optional theme citation |
| 6 | La Rosa. Temperature Fragility and the Conditional Benefits of Truncation Sampling. arXiv 2609.15476 (known) | Truncation helps accuracy only where high temperature hurts; deployed defaults sit at T 0.6 to 1.0 | adjacent | optional, in the limitations or fix section (truncation is not needed for accuracy at deployed temperatures) |
| 7 | Yang et al. Beyond Reasoning: Reinforcement Learning Unlocks Parametric Knowledge in LLMs. arXiv 2605.07153 (known) | RL moves correct answers "from the low-probability tail into reliable greedy generations" | adjacent (low, not zero, probability) | optional in "Knowing without saying" |
| 8 | McCabe, Melamed, Hartvigsen, Huang. Estimating Semantic Alphabet Size for LLM Uncertainty Quantification. arXiv 2509.14478 | Good-Turing style coverage correction for semantic entropy from LLM samples | adjacent to the estimator angle (sample coverage), not knowledge | optional; venue not on the arXiv page (one web summary says ICLR 2026, unverified) |
| 9 | Matthews, Hamilton. Language Models are Lazy Poets: Creative Mode Collapse at Scale. arXiv 2610.24223 | 1.1 million poems from 16 open models; large vocabularies "severely underutilized" | low | not needed |
| 10 | Annotations Mitigate Post-Training Mode Collapse (arXiv 2605.09995); Don't Repeat Yourself: Self-Supervised Fine-Tuning for Coverage (arXiv 2609.31688) | training-side fixes for post-training mode collapse and coverage | low | optional, as training-side alternatives to a decoding fix; authors not checked |
| 11 | Variance-Calibrated Modulation (arXiv 2606.22511); Adaptive Nucleus Truncation (arXiv 2606.13982); Min-k Sampling (arXiv 2604.11012, ACL 2026, known) | new truncation samplers | low; Min-k's limitations name long-tail entities (known) | Min-k is optional as evidence the risk is acknowledged but unmeasured |

## Leads found but not verified (do not cite)

- A practitioner blog post on "First Token Cutoff" (antirez.com/news/142), surfaced by web search. Not opened, and not citable.
- SENECA, arXiv 2605.00668 (small-sample entropy estimation with missing mass). Not opened.
- A "sampling-based search with self-verification" paper, cited by a news summary. Not identified or opened. Find and verify it before writing the G3b related work, because generate-then-verify is an established pattern.

## Top three wording changes

1. **Abstract and contribution 1:** stop presenting the exact test as ours; credit Awad et al. for the per-token survival test (see notes/drafts/novelty_contrast.md, sections 5 and 6).
2. **Mechanism (Section 5 and related work):** cite Jang et al. (2026), Parikh (2026) and Karouzos et al. (2026). Frame the first-token result as linking answer-slot concentration to exact zero probability of recognised entities, stage by stage, not as discovering post-training concentration.
3. **Fix (Section 7):** use KnowSum's own call to move "toward extraction, actively surfacing knowledge that the model encodes but rarely generates". Name the Alignment Paradox as the risk for a self-check filter. Make no novelty claim for generate-then-verify itself.

## Proposed bib entries (verified on the pages above, 9 November 2026)

```bibtex
@inproceedings{jang2026instruction,
  title     = {Instruction-Tuned Language Models Cannot Sample from Distributions They Can Describe},
  author    = {Jang, Chaemin and Kim, Jihee and Lee, Dongman},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026},
  address   = {Budapest, Hungary},
  publisher = {Association for Computational Linguistics},
  pages     = {48757--48777},
  doi       = {10.18653/2026.findings-emnlp.2278},
  url       = {https://aclanthology.org/2026.findings-emnlp.2278/}
}

@misc{parikh2026census,
  title         = {The One-Word Census: Answer-Choice Conformity Across 105 Language Models},
  author        = {Parikh, Tapan},
  year          = {2026},
  eprint        = {2607.12796},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}

@misc{huang2026alignment,
  title         = {The Alignment Paradox: How Post-Training Amplifies Confident Hallucinations in Language Models},
  author        = {Huang, Qingjia and Li, Yakai and Wu, Jianguo and Zhou, Qihang and Yu, Aimin and Jia, Xiaoqi and Ma, Luping and Zhang, Weijuan},
  year          = {2026},
  eprint        = {2609.32617},
  archivePrefix = {arXiv}
}
```

Notes on the entries:
- The Anthology DOI is printed as "10.18653/2026.findings-emnlp.2278", without the usual "v1/". Copy it exactly as the Anthology gives it, and re-check it before submission.
- The primaryClass of 2609.32617 was not read; it is left out on purpose.
