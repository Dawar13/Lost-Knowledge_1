# MB5: Unreachable Knowledge

Does the sampler a language model ships with make knowledge the model recognises impossible to say? Exact, sampling-free measurement for ACL 2027 (Special Theme).

- Start here: `docs/01_idea_brief.md`
- Phases and go/no-go rules: `docs/02_phases_and_gates.md`
- Agent instructions: `CLAUDE.md`

## Where things stand

- Gates: `results/gates/` (G0 signed under option B, G1 signed, G2 and G3 decided by the agent under the human's delegation; see the sign-off files).
- Every run, its provenance and what it showed: `results/INDEX.md`.
- Every non-trivial decision, dated, with reasons: `notes/decisions.md`.
- Experiment status: top of `docs/03_experiments.md`.
- Paper draft: `paper/main.tex` (compiles to `paper/main.pdf`). Every number in it is a macro in `paper/numbers.tex`, generated from `results/` by `python -m mb5.paper_numbers`; tables in `paper/tables/` are generated the same way. Verified references: `paper/references.bib`, `docs/05b_reference_verification.md`.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-lock.txt   # exact versions used for all results
export HF_TOKEN=...                       # Llama and Gemma are gated (not needed for Qwen and OLMo)
export PYTHONPATH=src
python -m mb5.check_env                   # shared GPU: check free memory first
python -m pytest -q tests                 # must pass (CPU, seconds)
SMOKE=1 bash scripts/run_g0.sh            # 0.5B smoke test
bash scripts/run_g0.sh                    # gate G0 (Qwen2.5-7B pair, pilot_v2)
bash scripts/run_g1.sh                    # gate G1 (OLMo-2 stages; Llama if HF_TOKEN)
bash scripts/run_p3_robust.sh             # wording and temperature robustness
bash scripts/run_p5.sh                    # gate G3 (mitigation frontier)
bash scripts/run_p3.sh                    # gate G2 (full TailCat) and P4 estimators
bash scripts/run_scale.sh                 # scale ladder (exploratory)
python -m mb5.paper_numbers               # regenerate paper numbers and tables
```

`scripts/mb5env.sh <cmd>` runs a command with the project's venv, an isolated Hugging Face cache and offline model loading (the cached snapshot recorded in each `meta.json` is the one used). On a shared GPU, `MB5_WAIT_GPU_MIN` makes jobs wait for memory instead of failing; the code always keeps 8 GiB free for other users.

The TailCat builder queries the public Wikidata SPARQL endpoint and caches every raw result under `data/tailcat/<split>/raw/` (for the full build, compressed as `data/tailcat/full_v1_raw.tar.gz`).
