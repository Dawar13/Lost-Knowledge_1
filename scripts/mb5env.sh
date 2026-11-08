#!/usr/bin/env bash
# Run a command inside the MB5 environment on the shared GPU box.
#   scripts/mb5env.sh <command...>
# Isolated venv and HF cache under the repo (other people's jobs use their own
# caches), offline hub so the cached snapshot (whose sha meta.json records) is
# what loads, and a polite wait for GPU memory instead of failing
# (common.require_gpu_memory).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src
export HF_HOME="$PWD/hf_home"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export MB5_WAIT_GPU_MIN="${MB5_WAIT_GPU_MIN:-720}"
export PYTHONUNBUFFERED=1
exec "$@"
