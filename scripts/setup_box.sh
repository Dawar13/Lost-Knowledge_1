#!/usr/bin/env bash
# Fresh GPU box (2026-11-09): venv with the pinned packages (requirements-lock.txt: torch 2.15.0,
# transformers 5.22.2, as in every earlier run) and the exact model snapshots already recorded in
# meta.json, downloaded into the repo's HF cache. refs/main is pointed at the pinned snapshot so the
# offline runs (HF_HUB_OFFLINE=1, scripts/mb5env.sh) load exactly it. Weights only as safetensors;
# Llama's original/ checkpoint is skipped. Needs HF_TOKEN in the environment for the gated Llama repos;
# the token is never written to disk here.
#   HF_TOKEN=... bash scripts/setup_box.sh
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] || python3 -m venv .venv || { apt-get update -qq && apt-get install -y -qq python3-venv && python3 -m venv .venv; }
source .venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements-lock.txt
export HF_HOME="$PWD/hf_home" HF_HUB_OFFLINE=0 HF_HUB_DISABLE_TELEMETRY=1
MODELS="${MODELS:-Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28 allenai/OLMo-2-1124-7B-Instruct@470b1fba1ae01581f270116362ee4aa1b97f4c84 meta-llama/Llama-3.1-8B@d04e592bb4f6aa9cfee91e2e20afa771667e1d4b meta-llama/Llama-3.1-8B-Instruct@0e9e39f249a16976918f6564b8830bc894c89659}"
python - $MODELS <<'PY'
import os, sys, time
from pathlib import Path
from huggingface_hub import snapshot_download
for spec in sys.argv[1:]:
    repo, rev = spec.split("@")
    t = time.time()
    p = Path(snapshot_download(repo, revision=rev, token=os.environ.get("HF_TOKEN"),
                               ignore_patterns=["original/*", "*.pth", "*.pt", "*.bin", "*.gguf", "*.h5", "*.msgpack"]))
    ref = p.parent.parent / "refs" / "main"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(rev)
    gb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9
    print(f"{repo}@{rev[:7]}: {gb:.1f} GB in {time.time() - t:.0f}s", flush=True)
PY
python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('transformers', transformers.__version__)"
