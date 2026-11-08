#!/usr/bin/env bash
# Third family for G1 (replaces Llama-3.1, access not granted): IBM Granite 3.1 8B base and instruct on
# pilot_v2, same protocol as run_g1.sh. Plan in notes/decisions.md (2026-11-09). Resumable.
set -euo pipefail
export PYTHONPATH=src
OUT=${OUT:-results/G1_granite}
DATA=${DATA:-data/tailcat/pilot_v2}
MODELS=(ibm-granite/granite-3.1-8b-base ibm-granite/granite-3.1-8b-instruct)
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
done_step "$OUT/granite-3.1-8b-base/meta.json" || \
python -m mb5.run_model --model ibm-granite/granite-3.1-8b-base --data "$DATA" --out "$OUT" --templates 3 --margin "$EPS"
done_step "$OUT/granite-3.1-8b-instruct/meta.json" || \
python -m mb5.run_model --model ibm-granite/granite-3.1-8b-instruct --data "$DATA" --out "$OUT" --templates 3 \
  --margin "$EPS" --sample-check --n-samples 1000
python -m mb5.metrics --results "$OUT" --data "$DATA" --instruct granite-3.1-8b-instruct --base granite-3.1-8b-base
python -m mb5.analysis_multi --gate G1_three_families --out results/G1_three_families \
  --model qwen2.5:base=results/G0_5090/Qwen2.5-7B --model qwen2.5:instruct=results/G0_5090/Qwen2.5-7B-Instruct \
  --model olmo2:base=results/G1/OLMo-2-1124-7B --model olmo2:sft=results/G1/OLMo-2-1124-7B-SFT \
  --model olmo2:dpo=results/G1/OLMo-2-1124-7B-DPO --model olmo2:instruct=results/G1/OLMo-2-1124-7B-Instruct \
  --model granite3.1:base="$OUT/granite-3.1-8b-base" --model granite3.1:instruct="$OUT/granite-3.1-8b-instruct"
echo "== Granite done; addendum written to results/gates/G1_three_families.md"
