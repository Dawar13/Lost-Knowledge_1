#!/usr/bin/env bash
# P3 scale ladder (docs/02: "the scale ladder if budget allows"): Qwen2.5-Instruct at 0.5B, 1.5B, 3B on
# pilot_v2; the 7B comes from results/G0_5090. Exploratory (no gate). Plan in notes/decisions.md.
# Each model gets its own epsilon by the G0 rule (self-check, seed 1). Resumable.
set -euo pipefail
export PYTHONPATH=src
OUT=${OUT:-results/P3_scale}
DATA=${DATA:-data/tailcat/pilot_v2}
done_step() { [ -f "$1" ] && echo "== skip (exists): $1"; }
python -m mb5.check_env
python -m pytest -q tests
for m in Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct; do
  tag=${m#*/}
  done_step "$OUT/calibration/$tag/selfcheck.json" || \
  python -m mb5.selfcheck --model "$m" --data "$DATA" --out "$OUT/calibration" --seed 1 \
    --n-samples 500 --gap-samples 50 --specs shipped_hf,p0.90_T1
  eps=$(python -c "import json; print(json.load(open('$OUT/calibration/$tag/selfcheck.json'))['epsilon_rule'])")
  echo "== $tag epsilon $eps"
  done_step "$OUT/$tag/meta.json" || \
  python -m mb5.run_model --model "$m" --data "$DATA" --out "$OUT" --templates 3 --margin "$eps"
  python -m mb5.metrics --results "$OUT" --data "$DATA" --instruct "$tag" --metrics-dir "metrics/$tag"
done
echo "== scale ladder done"
