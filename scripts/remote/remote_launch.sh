#!/usr/bin/env bash
# On the GPU box: setup then every remaining run, detached from the SSH session. Needs HF_TOKEN in env.
set -euo pipefail
: "${HF_TOKEN:?}"
cd /root/mb5
mkdir -p logs
nohup env HF_TOKEN="$HF_TOKEN" bash -c 'bash scripts/setup_box.sh > logs/setup_box.log 2>&1 && echo SETUP_OK >> logs/setup_box.log && bash scripts/run_remaining.sh > logs/run_remaining.log 2>&1; echo "EXIT $?" >> logs/run_remaining.log' > /dev/null 2>&1 &
echo "launched pid $!"
