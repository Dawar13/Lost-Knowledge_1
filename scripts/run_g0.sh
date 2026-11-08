#!/usr/bin/env bash
# Gate G0: the go/no-go experiment. Run from the repository root.
#   bash scripts/run_g0.sh            full pilot (about 0.5 GPU-hours on one 80 GB card)
#   SMOKE=1 bash scripts/run_g0.sh    5-minute smoke test on Qwen2.5-0.5B-Instruct
set -euo pipefail
export PYTHONPATH=src
OUT=${OUT:-results/G0}
DATA=${DATA:-data/tailcat/pilot}

echo "== environment and GPU (shared card: never kill other jobs)"
python -m mb5.check_env

echo "== unit tests (must pass before any GPU time)"
python -m pytest -q tests

if [ ! -f "$DATA/entities.jsonl" ]; then
  echo "== building TailCat pilot from Wikidata"
  python -m mb5.tailcat --config configs/pilot_categories.yaml --out "$DATA"
  echo "Read the sanity report above. Fix any category marked TOO SMALL, then re-run."
fi

if [ "${SMOKE:-0}" = "1" ]; then
  python -m mb5.run_model --model Qwen/Qwen2.5-0.5B-Instruct --data "$DATA" --out results/smoke \
    --templates 2 --limit-per-category 40 --sample-check --n-samples 200
  python -m mb5.analysis --results results/smoke --gate SMOKE --instruct Qwen2.5-0.5B-Instruct
  echo "Smoke test: a BLOCKED verdict is expected (too few recognised entities for a 0.5B model)."
  echo "What must hold: the run completes and sampling violations are 0."
  exit 0
fi

python -m mb5.run_model --model Qwen/Qwen2.5-7B-Instruct --data "$DATA" --out "$OUT" \
  --templates 3 --sample-check --n-samples 1000
python -m mb5.run_model --model Qwen/Qwen2.5-7B --data "$DATA" --out "$OUT" --templates 3
python -m mb5.analysis --results "$OUT" --gate G0 --instruct Qwen2.5-7B-Instruct --base Qwen2.5-7B
echo "== decision written to results/gates/G0.md"
