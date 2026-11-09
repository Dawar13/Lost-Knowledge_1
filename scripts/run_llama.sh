#!/usr/bin/env bash
# Third family for G1 (the original plan; notes/decisions.md 2026-11-09): Llama-3.1-8B base and instruct on
# pilot_v2, same protocol as run_g1.sh and run_granite.sh. Gated: needs Meta access and HF_TOKEN; the first
# run must download the weights, so unset HF_HUB_OFFLINE (scripts/mb5env.sh sets it). The Llama 3.1 chat
# template inserts a default system message with dates; prompts are stored verbatim in prompts.json. Resumable.
: "${HF_TOKEN:?set HF_TOKEN (read-only token of an account with Meta Llama 3.1 access)}"
set -euo pipefail
export PYTHONPATH=src
OUT=${OUT:-results/G1_llama}
DATA=${DATA:-data/tailcat/pilot_v2}
MODELS=(meta-llama/Llama-3.1-8B meta-llama/Llama-3.1-8B-Instruct)
done_step() { [ -f "$1" ] && echo "== skip (exists): $1"; }
python -m mb5.check_env
python -m pytest -q tests
for m in "${MODELS[@]}"; do
  done_step "$OUT/calibration/${m#*/}/selfcheck.json" || \
  python -m mb5.selfcheck --model "$m" --data "$DATA" --out "$OUT/calibration" --seed 1 \
    --n-samples 500 --gap-samples 50 --specs shipped_hf,p0.90_T1
done
EPS=$(python - "$OUT/calibration" <<'PY'
import json, sys
from pathlib import Path
vals = [json.loads(p.read_text())["epsilon_rule"] for p in sorted(Path(sys.argv[1]).glob("*/selfcheck.json"))]
if len(vals) < 2 or any(v is None for v in vals):
    sys.exit(f"epsilon calibration incomplete or beyond the grid: {vals}")
print(max(vals))
PY
)
echo "== numeric margin epsilon = $EPS"
done_step "$OUT/Llama-3.1-8B/meta.json" || \
python -m mb5.run_model --model meta-llama/Llama-3.1-8B --data "$DATA" --out "$OUT" --templates 3 --margin "$EPS"
done_step "$OUT/Llama-3.1-8B-Instruct/meta.json" || \
python -m mb5.run_model --model meta-llama/Llama-3.1-8B-Instruct --data "$DATA" --out "$OUT" --templates 3 \
  --margin "$EPS" --sample-check --n-samples 1000
python -m mb5.metrics --results "$OUT" --data "$DATA" --instruct Llama-3.1-8B-Instruct --base Llama-3.1-8B
python -m mb5.analysis_multi --gate G1_three_families --out results/G1_three_families \
  --model qwen2.5:base=results/G0_5090/Qwen2.5-7B --model qwen2.5:instruct=results/G0_5090/Qwen2.5-7B-Instruct \
  --model olmo2:base=results/G1/OLMo-2-1124-7B --model olmo2:sft=results/G1/OLMo-2-1124-7B-SFT \
  --model olmo2:dpo=results/G1/OLMo-2-1124-7B-DPO --model olmo2:instruct=results/G1/OLMo-2-1124-7B-Instruct \
  --model llama3.1:base="$OUT/Llama-3.1-8B" --model llama3.1:instruct="$OUT/Llama-3.1-8B-Instruct"
echo "== Llama done; addendum written to results/gates/G1_three_families.md"
