#!/usr/bin/env bash
# On the GPU box: keep the old checkout aside, make a clean clone (so a git push can update it),
# reuse the old venv and HF cache (pinned snapshots), print state.
set -euo pipefail
cd /root
if [ -d mb5/.git ] && [ ! -d mb5_old ]; then mv mb5 mb5_old; fi
if [ ! -d mb5/.git ]; then
  if [ -d mb5_old/.git ]; then git clone -q /root/mb5_old /root/mb5; else git init -q -b main /root/mb5; fi
fi
cd /root/mb5
git config receive.denyCurrentBranch updateInstead
git config core.autocrlf false
for d in .venv hf_home; do
  if [ ! -e "/root/mb5/$d" ] && [ -e "/root/mb5_old/$d" ]; then mv "/root/mb5_old/$d" "/root/mb5/$d"; fi
done
echo "box head: $(git log --oneline -1 2>/dev/null || echo none)"
ls hf_home/hub 2>/dev/null || echo "no hf cache"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
df -h /root | tail -1
