#!/usr/bin/env bash
# Gate G0: the go/no-go experiment. Run from the repository root.
#   bash scripts/run_g0.sh            full pilot (about 0.5 GPU-hours on one 80 GB card)
#   SMOKE=1 bash scripts/run_g0.sh    5-minute smoke test on Qwen2.5-0.5B-Instruct
set -euo pipefail
export PYTHONPATH=src
OUT=${OUT:-results/G0}
DATA=${DATA:-data/tailcat/pilot_v2}  # v2: G0 audit fixes (notes/decisions.md); v1 was the A100 attempt

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

INST=Qwen/Qwen2.5-7B-Instruct
BASE=Qwen/Qwen2.5-7B
# Shared card: lower BS if the memory estimate (common.estimate_need_gib) does not fit.
BS=${BS:-32}
# Resumable: a step whose output already exists is skipped (delete it to redo the step).
done_step() { [ -f "$1" ] && echo "== skip (exists): $1"; }

# Numeric margin (docs/04 section 2, rule pre-registered in notes/decisions.md):
# dense self-check on each G0 model with seed 1 (the gate's own sample check uses
# seed 0, so V1 validates epsilon out of sample); epsilon = the larger of the two
# rule values. Set EPS to skip calibration (only to reproduce a logged run).
if [ -z "${EPS:-}" ]; then
  echo "== calibrating the numeric margin (seed 1)"
  done_step "$OUT/calibration/${INST#*/}/selfcheck.json" || \
  python -m mb5.selfcheck --model "$INST" --data "$DATA" --out "$OUT/calibration" --seed 1 \
    --n-samples 500 --gap-samples 50 --specs shipped_hf,p0.90_T1 --batch-size "$BS"
  done_step "$OUT/calibration/${BASE#*/}/selfcheck.json" || \
  python -m mb5.selfcheck --model "$BASE" --data "$DATA" --out "$OUT/calibration" --seed 1 \
    --n-samples 500 --gap-samples 50 --specs p0.90_T1 --batch-size "$BS"
  EPS=$(python - "$OUT/calibration" <<'PY'
import json, sys
from pathlib import Path
vals = [json.loads(p.read_text())["epsilon_rule"] for p in sorted(Path(sys.argv[1]).glob("*/selfcheck.json"))]
if len(vals) != 2 or any(v is None for v in vals):
    sys.exit(f"epsilon calibration failed or exceeded the grid: {vals}")
print(max(vals))
PY
)
fi
echo "== numeric margin epsilon = $EPS raw logits"

done_step "$OUT/${INST#*/}/meta.json" || \
python -m mb5.run_model --model "$INST" --data "$DATA" --out "$OUT" \
  --templates 3 --sample-check --n-samples 1000 --margin "$EPS" --batch-size "$BS"
done_step "$OUT/${BASE#*/}/meta.json" || \
python -m mb5.run_model --model "$BASE" --data "$DATA" --out "$OUT" --templates 3 --margin "$EPS" \
  --batch-size "$BS"
python -m mb5.analysis --results "$OUT" --gate G0 --instruct Qwen2.5-7B-Instruct --base Qwen2.5-7B
echo "== decision written to results/gates/G0.md"
