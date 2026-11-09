#!/usr/bin/env bash
# One shot (2026-11-09): every remaining GPU experiment, in order of reviewer impact per dollar
# (notes/decisions.md): P6 for both instruct models (E15 self-check scores first, then E13 ask
# directly, then E14 list control), then Llama-3.1-8B base and instruct (third family for G1).
# The Llama addenda run only with MB5_EXTRAS=1.
# Run after scripts/setup_box.sh. Resumable: rerun to continue after any stop.
#   HF_TOKEN=... bash scripts/run_remaining.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export MB5_WAIT_GPU_MIN="${MB5_WAIT_GPU_MIN:-30}"
STEPS=e15,e13,e14 scripts/mb5env.sh bash scripts/run_p6.sh
scripts/mb5env.sh bash scripts/run_llama.sh
# Llama addenda (frontier, P6 addendum, full TailCat; about 2.5 GPU-hours) only on request: the last
# affordable GPU session runs the core only (notes/decisions.md, 2026-11-09 scope cut).
if [ "${MB5_EXTRAS:-0}" = 1 ]; then bash scripts/run_extra_llama.sh; else echo "== Llama addenda skipped (MB5_EXTRAS=1 to run them)"; fi
echo "== all remaining GPU experiments done"
