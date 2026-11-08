#!/usr/bin/env bash
# P3 / gate G2 on the full TailCat (capped split), plus the P4 estimator tables. Plan fixed in
# notes/decisions.md (2026-11-08). Resumable: finished steps are skipped.
set -euo pipefail
export PYTHONPATH=src
FULL=${FULL:-data/tailcat/full_v1}
DATA=${DATA:-data/tailcat/full_v1_s150}
OUT=${OUT:-results/P3}
BS=${BS:-32}
done_step() { [ -f "$1" ] && echo "== skip (exists): $1"; }

if [ ! -f "$DATA/entities.jsonl" ]; then
  echo "== capped split from the cached full build (member cap 150, foil cap 100)"
  mkdir -p "$DATA" && cp -r "$FULL/raw" "$DATA/"
  python -m mb5.tailcat --config configs/tailcat_full.yaml --out "$DATA" --cap 150 --foil-cap 100 --drop-small 30 --cache-only
fi
python -m mb5.check_env
python -m pytest -q tests

# model, epsilon (G0/G1, same GPU), sample-check specs (instruct models only)
RUNS=("Qwen/Qwen2.5-7B-Instruct 0.75 shipped_hf,p0.90_T1"
      "allenai/OLMo-2-1124-7B-Instruct 0.5 p0.90_T1"
      "Qwen/Qwen2.5-7B 0.75 -"
      "allenai/OLMo-2-1124-7B 0.5 -")
for r in "${RUNS[@]}"; do
  set -- $r
  sc=(); [ "$3" != "-" ] && sc=(--sample-check --n-samples 1000 --sample-specs "$3")
  done_step "$OUT/${1#*/}/meta.json" || \
  python -m mb5.run_model --model "$1" --data "$DATA" --out "$OUT" --templates 3 --margin "$2" --batch-size "$BS" "${sc[@]}"
done

python -m mb5.metrics --results "$OUT" --data "$DATA" --instruct Qwen2.5-7B-Instruct --base Qwen2.5-7B --metrics-dir metrics/qwen2.5
python -m mb5.metrics --results "$OUT" --data "$DATA" --instruct OLMo-2-1124-7B-Instruct --base OLMo-2-1124-7B --metrics-dir metrics/olmo2
python -m mb5.estimators --results "$OUT" --model Qwen2.5-7B-Instruct
python -m mb5.estimators --results "$OUT" --model OLMo-2-1124-7B-Instruct
python -m mb5.g2 --full "$OUT" --robust results/P3_robust \
  --models Qwen2.5-7B-Instruct,Qwen2.5-7B,OLMo-2-1124-7B-Instruct,OLMo-2-1124-7B
echo "== P3 done; decision written to results/gates/G2.md"
