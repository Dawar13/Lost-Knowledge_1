# MB5: Unreachable Knowledge

Does the sampler a language model ships with make knowledge the model recognises impossible to say? Exact, sampling-free measurement for ACL 2027 (Special Theme).

- Start here: `docs/01_idea_brief.md`
- Phases and go/no-go rules: `docs/02_phases_and_gates.md`
- Agent instructions: `CLAUDE.md`

## Quick start

```bash
pip install -r requirements.txt
export HF_TOKEN=...                       # Llama and Gemma are gated (needed from P2)
export PYTHONPATH=src
python -m mb5.check_env                   # shared GPU: check free memory first
python -m pytest -q tests                 # must pass (CPU, seconds)
SMOKE=1 bash scripts/run_g0.sh            # builds TailCat pilot, runs a 0.5B smoke test
bash scripts/run_g0.sh                    # the go/no-go experiment (G0)
cat results/gates/G0.md                   # verdict + numbers
```

The TailCat builder queries the public Wikidata SPARQL endpoint. If a category comes back too small, fix its QID in `configs/pilot_categories.yaml` (the builder prints a sanity report).
