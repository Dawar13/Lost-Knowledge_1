#!/usr/bin/env bash
# Gate G1 (P2): where does unreachability come from? Run from the repository root.
#   bash scripts/run_g1.sh     OLMo-2 base, SFT, DPO, Instruct; plus Llama-3.1-8B base and Instruct if HF_TOKEN
#                              is set (gated). The Qwen2.5 pair comes from G0 (results/G0_5090, option B).
# Resumable: a step whose output exists is skipped. Plan fixed in notes/decisions.md before running.
set -euo pipefail
export PYTHONPATH=src
OUT=${OUT:-results/G1}
DATA=${DATA:-data/tailcat/pilot_v2}
G0=${G0:-results/G0_5090}
BS=${BS:-32}
MODELS=(allenai/OLMo-2-1124-7B allenai/OLMo-2-1124-7B-SFT allenai/OLMo-2-1124-7B-DPO allenai/OLMo-2-1124-7B-Instruct)
if [ -n "${HF_TOKEN:-}" ]; then MODELS+=(meta-llama/Llama-3.1-8B meta-llama/Llama-3.1-8B-Instruct); fi
done_step() { [ -f "$1" ] && echo "== skip (exists): $1"; }
is_chat() { case "$1" in *-Instruct|*-SFT|*-DPO) return 0;; *) return 1;; esac; }

echo "== environment and GPU"
python -m mb5.check_env
echo "== unit tests (must pass before any GPU time)"
python -m pytest -q tests

# Numeric margin: the G0 rule per model (self-check, seed 1); G1 uses the largest rule value over its models.
if [ -z "${EPS:-}" ]; then
  echo "== calibrating the numeric margin (seed 1)"
  for m in "${MODELS[@]}"; do
    done_step "$OUT/calibration/${m#*/}/selfcheck.json" || \
    python -m mb5.selfcheck --model "$m" --data "$DATA" --out "$OUT/calibration" --seed 1 \
      --n-samples 500 --gap-samples 50 --specs shipped_hf,p0.90_T1 --batch-size "$BS"
  done
  EPS=$(python - "$OUT/calibration" "${#MODELS[@]}" <<'PY'
import json, sys
from pathlib import Path
vals = [json.loads(p.read_text())["epsilon_rule"] for p in sorted(Path(sys.argv[1]).glob("*/selfcheck.json"))]
if len(vals) < int(sys.argv[2]) or any(v is None for v in vals):
    sys.exit(f"epsilon calibration incomplete or beyond the grid: {vals}")
print(max(vals))
PY
)
fi
echo "== numeric margin epsilon = $EPS raw logits"

for m in "${MODELS[@]}"; do
  sc=(); is_chat "$m" && sc=(--sample-check --n-samples 1000)
  done_step "$OUT/${m#*/}/meta.json" || \
  python -m mb5.run_model --model "$m" --data "$DATA" --out "$OUT" --templates 3 --margin "$EPS" \
    --batch-size "$BS" "${sc[@]}"
done

echo "== metrics per model"
python -m mb5.metrics --results "$OUT" --data "$DATA" --instruct OLMo-2-1124-7B-SFT --metrics-dir metrics/OLMo-2-1124-7B-SFT
python -m mb5.metrics --results "$OUT" --data "$DATA" --instruct OLMo-2-1124-7B-DPO --metrics-dir metrics/OLMo-2-1124-7B-DPO
python -m mb5.metrics --results "$OUT" --data "$DATA" --instruct OLMo-2-1124-7B-Instruct --base OLMo-2-1124-7B \
  --metrics-dir metrics/olmo2_instruct_vs_base
if [ -n "${HF_TOKEN:-}" ]; then
  python -m mb5.metrics --results "$OUT" --data "$DATA" --instruct Llama-3.1-8B-Instruct --base Llama-3.1-8B \
    --metrics-dir metrics/llama3.1_instruct_vs_base
fi

echo "== gate analysis"
ARGS=(--model "qwen2.5:base=$G0/Qwen2.5-7B" --model "qwen2.5:instruct=$G0/Qwen2.5-7B-Instruct"
      --model "olmo2:base=$OUT/OLMo-2-1124-7B" --model "olmo2:sft=$OUT/OLMo-2-1124-7B-SFT"
      --model "olmo2:dpo=$OUT/OLMo-2-1124-7B-DPO" --model "olmo2:instruct=$OUT/OLMo-2-1124-7B-Instruct")
if [ -n "${HF_TOKEN:-}" ]; then
  ARGS+=(--model "llama3.1:base=$OUT/Llama-3.1-8B" --model "llama3.1:instruct=$OUT/Llama-3.1-8B-Instruct")
fi
python -m mb5.analysis_multi --gate G1 --out "$OUT" "${ARGS[@]}"
echo "== decision written to results/gates/G1.md"
