#!/usr/bin/env bash
# P6: E13 ask-directly control, E14 list control, E15 sample wide then self-check (gate G3b).
# Pre-registered in notes/decisions.md (2026-11-09) before any data. Both instruct models on pilot_v2,
# each with the margin of its open-prompt run (Qwen 0.75 from G0_5090, OLMo 0.5 from G1). Resumable:
# mb5.p6 skips every step whose output exists. E15 reuses the stored P5 samples (no new generation).
set -euo pipefail
export PYTHONPATH=src
OUT=${OUT:-results/P6}
DATA=${DATA:-data/tailcat/pilot_v2}
CUES=${CUES:-data/tailcat/pilot_v2_cues/cues.jsonl}
# Multilingual invented-name classes (mb5.invented_ml, CPU; G3b is decided under them, notes/decisions.md 2026-11-09)
CLASSES=${CLASSES:-results/analysis_cpu/invented_ml}
for t in Qwen2.5-7B-Instruct OLMo-2-1124-7B-Instruct; do
  [ -f "$CLASSES/samples_classes_$t.jsonl.gz" ] || { echo "missing $CLASSES/samples_classes_$t.jsonl.gz (run mb5.invented_ml analyse)"; exit 1; }
done
BS=${BS:-32}
STEPS=${STEPS:-e13,e14,e15}
QWEN=Qwen2.5-7B-Instruct
OLMO=OLMo-2-1124-7B-Instruct
python -m mb5.check_env
python -m pytest -q tests
# The cues are committed data (built on 2026-11-09 from Wikidata); rebuild only if missing (needs network).
[ -f "$CUES" ] || python -m mb5.cues --data "$DATA" --out "$(dirname "$CUES")"
python -m mb5.p6 --model "Qwen/$QWEN" --data "$DATA" --cues "$CUES" --open "results/G0_5090/$QWEN" \
  --p5 "results/P5/$QWEN" --out "$OUT" --margin 0.75 --steps "$STEPS" --batch-size "$BS"
python -m mb5.p6 --model "allenai/$OLMO" --data "$DATA" --cues "$CUES" --open "results/G1/$OLMO" \
  --p5 "results/P5/$OLMO" --out "$OUT" --margin 0.5 --steps "$STEPS" --batch-size "$BS"
python -m mb5.p6_analysis --results "$OUT" \
  --model "$QWEN:open=results/G0_5090/$QWEN,p5=results/P5/$QWEN,base=Qwen/Qwen2.5-7B" \
  --model "$OLMO:open=results/G1/$OLMO,p5=results/P5/$OLMO,base=allenai/OLMo-2-1124-7B"
echo "== E13/E14 in $OUT/summary.md; G3b decision in results/gates/G3b.md"
