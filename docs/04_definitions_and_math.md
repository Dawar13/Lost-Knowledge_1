# MB5 definitions and the maths that carries weight

Every quantity here is something the code computes and the paper reports. Nothing is decorative.

## 1. Samplers

Given logits z over vocabulary V at a decoding step, a sampler applies, in transformers' order: repetition penalty, temperature T, top-k, top-p, min-p. Each step sets some logits to minus infinity; sampling then draws from the softmax of what remains. A token is **kept** if its processed logit is finite.

Keep rules for a target token t (proved equal to transformers' warpers by `tests/test_semantics.py`):

- **top-k**: let rank(t) = number of tokens with logit strictly greater than z_t. Kept iff rank(t) < k.
- **top-p**: let G(t) = total probability (at temperature T, after any top-k) of tokens strictly more likely than t. Kept iff G(t) < p. In words: the nucleus is filled from the top, and t gets in if the mass ahead of it has not yet reached p.
- **min-p**: let gap(t) = z_max minus z_t. Kept iff exp(-gap(t)/T) >= min_p, i.e. t's probability is at least min_p times the top token's.

## 2. Reachability of an answer

Write entity e, after prompt x, as tokens t_1 ... t_L (in-context canonical tokenisation). Let h_j be the prompt plus t_1 ... t_{j-1}.

- e is **reachable** under sampler S iff every t_j is kept at h_j. Its probability under S is then positive; otherwise it is exactly zero, and no number of samples, retries or self-consistency votes can produce it (through this tokenisation).
- **Critical nucleus level**: p\*(e) = max_j G_j(e). e is reachable under top-p iff p > p\*(e). One forward pass gives p\*(e) exactly.
- **Critical k**: k\*(e) = max_j rank_j(e) + 1. Reachable under top-k iff k >= k\*(e).
- **Critical min-p**: m\*(e) = min_j exp(-gap_j(e)/T). Reachable under min-p iff min_p <= m\*(e).
- **Bottleneck position**: the j that attains the maximum (or minimum). Hypothesis H3 says it is usually j = 1, the first token of the name.

For combined samplers (shipped defaults) we do not use formulas: we run the actual processor chain at every position and check survival. The formulas draw the curves; the chain decides the headline.

Surface forms: e is reachable for a prompt if any of its forms (label, up to three aliases) is reachable. Non-canonical token paths are audited in E4 rather than enumerated.

**Numeric margin (added in P0, before any G0 data).** Logits are computed in bf16. Two computations of the same logits (our batched teacher-forced pass, and KV-cached `generate()`) differ slightly, and bf16 logits tie often, so a token sitting on a truncation boundary can be kept by one and cut by the other. "Probability exactly zero" is only defensible away from the boundary. Write kept_S(t; m) for whether target t survives sampler S after its raw logit is raised by m relative to every other token. Every rule we model (repetition penalty, temperature, top-k, top-p, min-p) is monotone in m, and any m different from 0 breaks all ties with t. Then:

- e is **robustly unreachable** under S if some position has kept_S(t_j; +ε) false: it stays cut even if numeric jitter favours it by ε.
- e is **robustly reachable** if every position has kept_S(t_j; -ε) true.
- otherwise e is **boundary**: the verdict depends on numerics.

ε is calibrated per model from the measured gap between `generate()` and teacher-forced logits (`mb5.selfcheck`), before the gate run. The headline RBU counts robustly unreachable entities (a lower bound). Exact RBU (ε = 0) and the upper bound (not robustly reachable) are reported beside it. V1 then reads: no robustly unreachable answer is ever sampled.

## 3. Recognition and RBU

- **V(e)** = P(Yes) / (P(Yes) + P(No)) for the question 'Is "e" <category>?', pooling token variants.
- **tau_c** = 95th percentile of V over the foils of category c, so at most 5% of foils pass.
- **Recognised**: V(e) >= tau_c AND the 4-way choice against three foils picks e.
- **RBU(S)** = |recognised and unreachable under S| / |recognised|. RBU-mean averages over prompt wordings; RBU-all counts an entity only if it is unreachable under every wording (the conservative headline). "Unreachable" here means robustly unreachable (section 2); exact and upper-bound RBU are reported alongside.

Why this pairing matters: low RBU means "the model does not know it" explains the missing tail; high RBU means the sampler, not the model, removes knowledge the model can verify.

## 4. Why sample-based knowledge estimates are biased, and by exactly how much

Let q_S(e) be the probability that one sample under S is entity e. After N independent samples, the expected number of distinct entities seen is

E[D_N] = sum over e of [1 - (1 - q_S(e))^N],

which rises to |{e : q_S(e) > 0}|, the reachable set, as N grows. Every unseen-species estimator (Good-Turing, Chao1 = S_obs + f_1^2 / (2 f_2), coverage-based rarefaction) is built to estimate the size of that support from samples. So applied to truncated samples it estimates the reachable set, never the recognised set. The gap between "knowledge estimated from samples" and "knowledge the model recognises" is

|recognised| - |recognised and reachable| = RBU(S) x |recognised|,

which we compute exactly rather than estimate. This turns a vague caveat in KnowSum-style work into a correction term. E10 checks it empirically.

## 5. Why the zeros come from truncation, and what post-training does

The KL-regularised preference optimum is

pi\*(y | x) proportional to pi_ref(y | x) exp(r(x, y) / beta),

where pi_ref is the reference (pre-alignment) model, r the reward and beta the KL weight. exp(.) is never zero, so pi\* has the same support as pi_ref: preference training alone cannot make a recognised answer impossible. What it can do is concentrate mass. If the first-token distribution of answers concentrates on a few head entities, then for every tail entity the mass ahead of its first token, G_1(e), grows, and once G_1(e) >= p the whole entity is deleted at once.

Testable consequences:
- RBU rises from base to aligned models (H2) while untruncated log-probabilities of the same entities stay finite.
- Failures concentrate at position 1 (H3).
- Per category, RBU is predicted by first-token concentration, measured as the top-1 mass of the first answer token, c = max_v pi(v | x). The fit RBU ~ c across categories and stages is the mechanism plot.

(SFT is not exactly a KL-regularised optimum, but finetuning with cross-entropy also leaves logits finite; the same argument holds: only truncation produces exact zeros.)

## 6. Statistics

- **Unit of resampling**: categories. Entities within a category share a prompt and are not independent. All CIs are cluster bootstrap over categories (2,000 resamples).
- **Paired comparisons** (base against instruct, stage against stage): entities recognised by both models; sign test on discordant pairs (normal approximation, two-sided).
- **Regressions** (E8, E9, mechanism): mixed-effects logistic, unreachable ~ predictors + (1 | category); report odds ratios with CIs.
- **Multiple comparisons**: Holm correction across the pre-registered tests in docs/02; everything else is labelled exploratory.
- **Pre-registration**: thresholds in docs/02 are fixed before G0 runs. Changes after seeing data go in notes/decisions.md with a reason and are reported in the paper.
