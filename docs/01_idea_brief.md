# MB5 idea brief: Unreachable Knowledge

Working title: **"Unreachable Knowledge: Decoding Defaults, Not Forgetting, Hide What Language Models Know"**
Venue: ACL 2027, Special Theme "Homogenization and knowledge collapse in LLMs" (fallback track: Language Modeling).
ARR deadline: 4 January 2027, 23:59 AoE.

## 1. The open problem

Language models are said to collapse onto common answers and lose the long tail. The theme call asks for the mechanisms behind this, including decoding.

Three different things can sit behind "the model never says rare fact X":

1. **Not stored.** The model does not know X.
2. **Stored but rare.** The model knows X and gives it a small probability. Ask enough times and X appears.
3. **Stored but impossible.** The model knows X, but the sampler gives X probability **exactly zero**. Ask a million times and X never appears.

Case 3 is easy to miss. Top-p, top-k and min-p sampling delete every token outside a "nucleus" before sampling. If one token of a rare name falls outside the nucleus at any step, the whole name is unreachable. Nearly every deployed chat model ships with such a sampler as its default.

Evidence that the gap is real but unmeasured for knowledge:
- Frontier models encode most facts but fail to recall a large share (Empty Shelves or Lost Keys, arXiv 2602.14080).
- A model can rank the gold answer first internally and still never generate it in 1,000 samples (Inside-Out, arXiv 2503.15299). That study samples without truncation, so it misses case 3.
- Exact token-survival analysis exists only for ordinary words (Lost in Sampling, arXiv 2605.27268) and for human-written tokens (Truncation Blind Spot, arXiv 2603.18482).
- Unseen-knowledge estimators (KnowSum, arXiv 2506.02058) count what appears in samples, so they inherit whatever the sampler deletes.

Nobody has measured how much **recognised** knowledge falls into case 3, under the samplers models actually ship with.

## 2. Candidate causes, to test against each other

- **C1 Decoding defaults alone.** Truncation deletes tail tokens even in base models.
- **C2 Post-training concentration.** Alignment piles probability onto a few head answers at the first answer token, pushing tail names across the truncation threshold. Preference training (KL-regularised) cannot set a probability to zero by itself, so the deletion needs both C2 and truncation.
- **C3 Tokenisation.** Rare names, especially in Indic scripts, split into more tokens, so there are more chances to be cut.
- **C4 Prompt format.** Preambles ("Sure! One example is ...") change the context and may rescue or doom names.

## 3. The method

**Exact reachability.** One teacher-forced forward pass per (prompt, name) gives, at every answer position j, the mass of tokens strictly more likely than the target. Top-p keeps the target iff that mass is below p. So each name has a critical level

p\*(e) = max over positions of (mass strictly above the target token),

and the name is reachable under top-p iff p > p\*(e). The same pass gives the critical k for top-k and the critical ratio for min-p. For the shipped sampler (repetition penalty, temperature, top-k, top-p together) we run transformers' own processor chain on every position and check whether the target survives. No sampling noise; exact.

**Recognition.** Separately, ask the model whether the name belongs to the category (yes/no) and to pick it out of three hard foils from sibling categories (4-way choice). Recognised = passes both, with the yes/no threshold set so at most 5% of foils pass.

**Headline quantity.** RBU, the recognised-but-unreachable rate: the share of recognised entities with probability exactly zero under a given sampler.

**Validation.** Draw real samples and confirm that no entity we call unreachable ever appears. This is a code check; it must come out at exactly zero.

## 4. What the paper shows (if the gates pass)

1. A share of knowledge the model demonstrably holds has probability exactly zero under its own shipped default sampler.
2. Where it comes from: the share rises through post-training (OLMo-2 base, SFT, DPO, Instruct), and the cut happens at the first token of the name.
3. Why the measurement literature is biased: any estimator fitted to samples measures the reachable support, not knowledge. The bias is exactly the RBU mass, which we compute.
4. Who loses: the effect is concentrated in low-popularity entities and (to test) in Indian-language prompts, which is cultural homogenisation by default setting.
5. A fix: relax truncation only at the first content token of an answer, and report the frontier of recovered facts against invented names.

## 5. The delta

- **Conceptual:** splits "knowledge collapse" into forgetting versus a decoding artefact, with an exact test. That is the mechanism-and-mitigation question the theme asks.
- **Methodological:** an exact, sampling-free reachability statistic any lab can run in one forward pass, plus a correction for sample-based knowledge estimators.
- **Practical:** changes how a setting everyone uses (top-p/top-k defaults) is chosen and reported. "Your default sampler silently deletes X% of what your model knows" is the citable line.
- **Resource:** TailCat, a multilingual long-tail category benchmark with popularity and hard foils.

## 6. The experiments that decide the paper

- **G0, go/no-go (about 0.5 GPU-hours).** Qwen2.5-7B-Instruct and its base, 10 long-tail categories plus one head control, three prompt wordings. Is RBU at the shipped default at least 10% (strong) or at least 2% (go), with tight intervals, zero sampling violations and valid recognition? If RBU is under 1% everywhere, the idea dies here.
- **G1, mechanism (about 2 GPU-hours).** OLMo-2 stage chain and two more families. Does RBU rise with post-training, and do most failures happen at the first token? This decides whether the story is "alignment plus truncation" or "truncation alone".
- **G2, generality and who loses (about 6 GPU-hours).** Full TailCat, four languages, popularity strata, robustness to wording, aliases, preambles and tokenisation paths.
- **G3, fix (about 2 GPU-hours).** Does first-token relaxation recover at least half the unreachable recognised entities without raising invented names by more than 2 points?

Decision rules are pre-registered in docs/02_phases_and_gates.md.
