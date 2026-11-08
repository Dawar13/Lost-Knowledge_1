#!/usr/bin/env bash
# P3 robustness on pilot_v2 (E7): five wordings and the temperature sweep for the Qwen and OLMo pairs.
# Plan fixed in notes/decisions.md (2026-11-08). Resumable.
set -euo pipefail
export PYTHONPATH=src
OUT=${OUT:-results/P3_robust}
DATA=${DATA:-data/tailcat/pilot_v2}
BS=${BS:-32}
done_step() { [ -f "$1" ] && echo "== skip (exists): $1"; }
python -m mb5.check_env
python -m pytest -q tests
for spec in "Qwen/Qwen2.5-7B-Instruct 0.75" "Qwen/Qwen2.5-7B 0.75" "allenai/OLMo-2-1124-7B-Instruct 0.5" "allenai/OLMo-2-1124-7B 0.5"; do
  set -- $spec
  done_step "$OUT/${1#*/}/meta.json" || \
  python -m mb5.run_model --model "$1" --data "$DATA" --out "$OUT" --templates 5 --margin "$2" --batch-size "$BS"
done
python -m mb5.metrics --results "$OUT" --data "$DATA" --instruct Qwen2.5-7B-Instruct --base Qwen2.5-7B --metrics-dir metrics/qwen2.5
python -m mb5.metrics --results "$OUT" --data "$DATA" --instruct OLMo-2-1124-7B-Instruct --base OLMo-2-1124-7B --metrics-dir metrics/olmo2
echo "== P3 robustness done"
