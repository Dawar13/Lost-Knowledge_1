#!/usr/bin/env bash
# Llama-3.1 addenda (2026-11-09, logged in notes/decisions.md before running), after run_remaining.sh:
#   1. P5 frontier for Llama-3.1-8B-Instruct (reference: shipped_hf, as for Qwen)
#   2. multilingual classes for its samples (CPU) and P6 E15, E13, E14 for it (addendum; G3b's decision
#      stays on the two pre-registered models, so this writes to results/P6_llama with its own report)
#   3. full TailCat (capped split) for the Llama pair, as P3: 3 templates; sample check at scale with the
#      shipped sampler only; metrics and estimator tables (the multi-family G2 addendum is CPU, later)
# Epsilon is the G1_llama rule value. Resumable.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src HF_HOME="$PWD/hf_home" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 MB5_WAIT_GPU_MIN="${MB5_WAIT_GPU_MIN:-30}"
BS=${BS:-32}
INST=meta-llama/Llama-3.1-8B-Instruct; BASE=meta-llama/Llama-3.1-8B
IT=Llama-3.1-8B-Instruct; BT=Llama-3.1-8B
EPS=$(python -c "import json;print(json.load(open('results/G1_llama/$IT/meta.json'))['margin'])")
echo "== epsilon $EPS"
done_step() { [ -f "$1" ] && echo "== skip (exists): $1"; }

# 1. frontier
done_step "results/P5/$IT/frontier.json" || \
python -m mb5.frontier --model "$INST" --data data/tailcat/pilot_v2 --out results/P5 \
  --recog "results/G1_llama/$IT/recog.jsonl.gz" --reference shipped_hf --margin "$EPS" --batch-size "$BS"

# 2. multilingual classes (CPU) and P6 for Llama
done_step "results/analysis_cpu/invented_ml_llama/samples_classes_$IT.jsonl.gz" || \
python -m mb5.invented_ml analyse --data data/tailcat/pilot_v2 --labels data/tailcat/pilot_v2_labels_ml \
  --p5 results/P5 --models "$IT" --out results/analysis_cpu/invented_ml_llama
python -m mb5.p6 --model "$INST" --data data/tailcat/pilot_v2 --cues data/tailcat/pilot_v2_cues/cues.jsonl \
  --open "results/G1_llama/$IT" --p5 "results/P5/$IT" --out results/P6_llama --steps e15,e13,e14 --batch-size "$BS"
python -m mb5.p6_analysis --results results/P6_llama --gates results/P6_llama/gates \
  --stages results/G1_three_families/stages.csv \
  --classes-dir results/analysis_cpu/invented_ml_llama \
  --model "$IT:open=results/G1_llama/$IT,p5=results/P5/$IT,base=$BASE"

# 3. full TailCat for the Llama pair
DATA=data/tailcat/full_v1_s150
done_step "results/P3/$IT/meta.json" || \
python -m mb5.run_model --model "$INST" --data "$DATA" --out results/P3 --templates 3 --margin "$EPS" --batch-size "$BS" \
  --sample-check --n-samples 1000 --sample-specs shipped_hf
done_step "results/P3/$BT/meta.json" || \
python -m mb5.run_model --model "$BASE" --data "$DATA" --out results/P3 --templates 3 --margin "$EPS" --batch-size "$BS"
python -m mb5.metrics --results results/P3 --data "$DATA" --instruct "$IT" --base "$BT" --metrics-dir metrics/llama3.1
python -m mb5.estimators --results results/P3 --model "$IT"
echo "== Llama addenda done"
