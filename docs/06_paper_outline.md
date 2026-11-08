# MB5 paper outline (ACL long paper, 8 pages)

Title options (decide after G1):
- Unreachable Knowledge: Decoding Defaults, Not Forgetting, Hide What Language Models Know
- Known but Unsayable: Exact Reachability of Long-Tail Knowledge under LLM Samplers

## Abstract skeleton (fill numbers only from results/)

Language models are said to collapse onto common answers and lose the long tail. We separate three cases: knowledge the model lacks, knowledge it holds but rarely says, and knowledge it holds but cannot say because the sampler assigns it probability exactly zero. With an exact, sampling-free reachability test (one forward pass per answer), we find that [RBU at shipped defaults]% of entities that [model] verifies with high confidence are unreachable under its shipped decoding settings, rising from [base]% before post-training to [instruct]% after it, with [x]% of failures at the first token of the name. Sample-based estimates of model knowledge therefore measure the sampler, and we give the exact correction. The loss falls on [low-popularity / non-English] entities. Relaxing truncation only at the first answer token recovers [y]% of unreachable knowledge at a cost of [z] points of invented names.

## Sections

1. **Introduction**: the three cases; the "probability exactly zero" point; contributions.
2. **Exact reachability**: keep rules, p\*, k\*, min-p\*; combined-sampler check; validation against generate().
3. **TailCat**: construction, popularity, foils, languages; recognition protocol.
4. **How much recognised knowledge is unreachable?** Headline table across models at shipped defaults; RBU(p) curves.
5. **Where it comes from**: OLMo stages; first-token concentration; failure positions.
6. **What it does to measurement**: coverage ceiling and estimator bias.
7. **Who loses**: popularity and languages.
8. **A targeted fix**: frontier plot.
9. **Related work**, **Limitations** (canonical tokenisation, recognition proxy, Wikidata completeness, open models only), **Ethics**.

## Figures (plan before running P3)

1. RBU(p) curves for base and instruct, shipped setting marked. (The figure people will screenshot.)
2. Histogram of first failing position.
3. RBU across OLMo stages with first-token top-1 mass on a second axis.
4. Distinct valid names against number of samples, with the reachable ceiling and the recognised level as horizontal lines; Chao1 estimate marked.
5. Frontier: distinct valid names against invented-name rate for each sampler; first-token relaxation highlighted.
6. RBU by popularity decile and by prompt language.

Table 1: per model, shipped settings, RBU-mean, RBU-all, CIs, recognised count.
