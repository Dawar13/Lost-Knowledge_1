# Unreachable Knowledge

**Language models can recognise a fact and still be unable to say it.**

When a model writes an answer, common sampling methods (top-p, top-k, min-p) keep only the most likely next tokens and delete the rest. A name is several tokens long. If any one of them is deleted at any step, that name has probability exactly zero: no number of samples or retries will ever produce it.

This project measures how much of what a model demonstrably recognises falls into this trap under the sampling settings models ship with, why it happens, and what it means for how we measure what models know.

## A small example

Ask a popular 7B instruct model for a U.S. state capital 1,000 times with its default settings, and you get only four different answers. In a multiple-choice test, the same model picks the right capital more than nine times in ten. Under the same top-p setting, its base model can produce every capital in our list; the instruct model, only a handful.

## How we measure it

- **Recognition.** A model counts as recognising a name only if it confirms the name belongs to the category and picks it out from near-miss look-alikes (for example, Tamil poets against Telugu and Kannada poets).
- **Reachability.** One forward pass per name tells us exactly whether every token of the name survives the sampler's cut. No sampling is needed.
- **Validation.** We checked these verdicts against more than 800,000 real samples. None contradicted them.

The entities come from Wikidata: more than 30,000 long-tail people, places, works and organisations in close to 300 categories, most of them from South Asia, Africa, Southeast Asia and Latin America.

## What we find so far

- **Most recognised long-tail knowledge cannot be said.** Under common settings, more than 90% of the long-tail names an instruct model recognises are impossible to produce, against between a half and two thirds for the base models they come from.
- **Post-training is the cause.** Following one model family through each training stage, most of the rise happens at supervised fine-tuning. The cut also moves to the very first token of the answer: post-training piles probability onto a few popular answers, and everything else is cut before it can start.
- **Buried, then sealed.** Post-training makes most rare names extremely unlikely; truncation then makes them impossible.
- **Sample-based knowledge estimates hit a ceiling.** Counting what a model says and extrapolating gives estimates in the hundreds, while the model recognises more than 15,000 of the names.
- **Rarer entities lose more.** The less known an entity, the more likely the model recognises it but can never say it.
- **Simple fixes are not enough.** Relaxing the cut at the first token only recovers less than a quarter of the lost names, because names are cut again later. A second approach is under test.

Models so far: Qwen2.5 and OLMo-2 at 7B, base and instruct, plus every OLMo-2 post-training stage. A third family is in progress.

## Status

Work in progress. A paper is in preparation; more details will follow with it.

## Repository

- `src/mb5/`: the exact reachability test, the recognition test and the dataset builder
- `tests/`: checks that our sampler rules match the Hugging Face `transformers` implementation
- `configs/`: model and category settings

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-lock.txt
export PYTHONPATH=src
python -m pytest -q tests
```
