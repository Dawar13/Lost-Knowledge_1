#!/usr/bin/env bash
# P5 / gate G3: mitigation frontier on pilot_v2 for the two instruct models. Plan fixed in
# notes/decisions.md (2026-11-08). Resumable.
set -euo pipefail
export PYTHONPATH=src
OUT=${OUT:-results/P5}
DATA=${DATA:-data/tailcat/pilot_v2}
done_step() { [ -f "$1" ] && echo "== skip (exists): $1"; }
python -m mb5.check_env
python -m pytest -q tests
done_step "$OUT/Qwen2.5-7B-Instruct/frontier.json" || \
python -m mb5.frontier --model Qwen/Qwen2.5-7B-Instruct --data "$DATA" --out "$OUT" \
  --recog results/G0_5090/Qwen2.5-7B-Instruct/recog.jsonl.gz --reference shipped_hf --margin 0.75
done_step "$OUT/OLMo-2-1124-7B-Instruct/frontier.json" || \
python -m mb5.frontier --model allenai/OLMo-2-1124-7B-Instruct --data "$DATA" --out "$OUT" \
  --recog results/G1/OLMo-2-1124-7B-Instruct/recog.jsonl.gz --reference p0.90_T1 --margin 0.5
python -m mb5.g3 --results "$OUT" --models Qwen2.5-7B-Instruct,OLMo-2-1124-7B-Instruct
echo "== decision written to results/gates/G3.md"
