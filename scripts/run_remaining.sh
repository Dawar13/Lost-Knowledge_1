#!/usr/bin/env bash
# One shot (2026-11-09): every remaining GPU experiment, in order of reviewer impact per dollar
# (notes/decisions.md): P6 for both instruct models (E15 self-check scores first, then E13 ask
# directly, then E14 list control), then Llama-3.1-8B base and instruct (third family for G1).
# Run after scripts/setup_box.sh. Resumable: rerun to continue after any stop.
#   HF_TOKEN=... bash scripts/run_remaining.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export MB5_WAIT_GPU_MIN="${MB5_WAIT_GPU_MIN:-30}"
STEPS=e15,e13,e14 scripts/mb5env.sh bash scripts/run_p6.sh
scripts/mb5env.sh bash scripts/run_llama.sh
bash scripts/run_extra_llama.sh
echo "== all remaining GPU experiments done"
