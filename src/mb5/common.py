"""Shared helpers: I/O, seeding, GPU etiquette on a shared card, model loading."""
from __future__ import annotations

import csv
import gzip
import json
import os
import random
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- I/O
def write_jsonl(path, rows, gz: bool | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gz = path.suffix == ".gz" if gz is None else gz
    opener = gzip.open if gz else open
    with opener(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------- GPU etiquette
def gpu_free_gib() -> tuple[float, float]:
    free, total = torch.cuda.mem_get_info()
    return free / 2**30, total / 2**30


HEADROOM_GIB = 8.0  # CLAUDE.md hard rule 1: always leave this much free for other people's jobs


def _nvsmi_free_gib() -> tuple[float, float] | None:
    """(free, total) GiB from nvidia-smi without creating a CUDA context."""
    dev = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0].strip() or "0"
    try:
        out = subprocess.run(["nvidia-smi", "-i", dev, "--query-gpu=memory.free,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
        f, t = (float(x) for x in out.strip().splitlines()[0].split(","))
        return f / 1024, t / 1024
    except Exception:
        return None


def require_gpu_memory(need_gib: float, headroom_gib: float = HEADROOM_GIB, wait_minutes: float | None = None,
                       poll_seconds: float = 60.0):
    """The GPU is shared with other people's experiments. Never kill their
    processes. Start only if free memory covers this job AND leaves
    `headroom_gib` for others. If MB5_WAIT_GPU_MIN (minutes) is set, poll until
    it does instead of exiting. Once started, cap this process's allocator at
    `need_gib` so that a bad estimate makes our job fail, not theirs."""
    if wait_minutes is None:
        wait_minutes = float(os.environ.get("MB5_WAIT_GPU_MIN", "0"))
    deadline = time.time() + 60.0 * wait_minutes
    # Wait with nvidia-smi first so that a long wait does not hold a CUDA
    # context (about 0.5 GiB) on a card other people are using.
    while wait_minutes > 0:
        nv = _nvsmi_free_gib()
        if nv is None or nv[0] >= need_gib + headroom_gib or time.time() >= deadline:
            break
        print(f"[gpu] waiting: {nv[0]:.1f} GiB free, need {need_gib:.1f} + {headroom_gib:.0f} GiB headroom "
              f"({time.strftime('%H:%M:%S')})", flush=True)
        time.sleep(poll_seconds)
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device visible. Check CUDA_VISIBLE_DEVICES.")
    while True:
        free, total = gpu_free_gib()
        if free >= need_gib + headroom_gib:
            break
        if time.time() >= deadline:
            raise SystemExit(
                f"Only {free:.1f} GiB free of {total:.1f} GiB; need about {need_gib:.1f} GiB plus "
                f"{headroom_gib:.0f} GiB headroom for other jobs. Do not kill them. Wait (MB5_WAIT_GPU_MIN), "
                "lower --batch-size, or use a smaller model."
            )
        print(f"[gpu] waiting: {free:.1f} GiB free, need {need_gib:.1f} + {headroom_gib:.0f} GiB headroom "
              f"({time.strftime('%H:%M:%S')})", flush=True)
        time.sleep(poll_seconds)
    torch.cuda.set_per_process_memory_fraction(min(1.0, need_gib / total))
    print(f"[gpu] {free:.1f} GiB free of {total:.1f} GiB; this job is capped at {need_gib:.1f} GiB, "
          f"leaving at least {free - need_gib:.1f} GiB for others", flush=True)


def estimate_need_gib(hf_id: str, batch_size: int) -> float:
    """Rough: bf16 weights + logits buffer + slack."""
    import re

    m = re.search(r"(\d+(?:\.\d+)?)[bB]\b", hf_id.replace("-", " "))
    params_b = float(m.group(1)) if m else 8.0
    return params_b * 2.1 + 0.05 * batch_size + 4.0


def append_gpu_ledger(script: str, model: str, seconds: float, note: str = ""):
    path = ROOT / "logs" / "gpu_ledger.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["utc_time", "host", "gpu", "script", "model", "gpu_hours", "note"])
        w.writerow([time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), socket.gethostname(),
                    name, script, model, f"{seconds / 3600:.4f}", note])


# ---------------------------------------------------------------- models
def is_chat_model(hf_id: str) -> bool:
    s = hf_id.lower()
    return any(t in s for t in ("instruct", "-it", "chat", "-sft", "-dpo"))


def load_model(hf_id: str, device: str = "cuda", revision: str | None = None):
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hf_id, revision=revision)
    major = int(transformers.__version__.split(".")[0])
    kw = {"revision": revision}
    # transformers >= 5 uses `dtype`; older versions use `torch_dtype`.
    kw["dtype" if major >= 5 else "torch_dtype"] = torch.bfloat16 if device != "cpu" else torch.float32
    if "gemma-2" in hf_id.lower():
        kw["attn_implementation"] = "eager"  # recommended for Gemma-2 soft-capping
    model = AutoModelForCausalLM.from_pretrained(hf_id, **kw).to(device).eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model, tok


def model_commit(model, hf_id: str, revision: str | None = None) -> str | None:
    """Commit hash of the weights actually loaded. transformers >= 5 no longer
    sets config._commit_hash, so fall back to the HF cache snapshot directory
    (.../snapshots/<sha>/config.json), which is offline and exact."""
    c = getattr(model.config, "_commit_hash", None)
    if c:
        return c
    try:
        from huggingface_hub import try_to_load_from_cache

        p = try_to_load_from_cache(hf_id, "config.json", revision=revision)
        if isinstance(p, str):
            return Path(p).parent.name
    except Exception:  # pragma: no cover
        pass
    return None


def code_version() -> dict:
    """Which code produced a result (CLAUDE.md rule 5): git HEAD and dirty flag
    if the tree is a git checkout, plus a content hash of src/ and configs/
    that is valid either way."""
    import hashlib

    h = hashlib.sha256()
    for sub in ("src", "configs", "scripts"):
        for p in sorted((ROOT / sub).rglob("*")):
            if p.is_file() and p.suffix in (".py", ".yaml", ".yml", ".sh"):
                h.update(str(p.relative_to(ROOT)).replace("\\", "/").encode())
                h.update(p.read_bytes())
    out = {"tree_sha256": h.hexdigest()[:16], "git_head": None, "git_dirty": None}
    try:
        run = lambda *a: subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True,  # noqa: E731
                                        timeout=20)
        r = run("rev-parse", "HEAD")
        if r.returncode == 0:
            out["git_head"] = r.stdout.strip()
            out["git_dirty"] = bool(run("status", "--porcelain", "--", "src", "configs", "scripts").stdout.strip())
    except Exception:  # pragma: no cover
        pass
    return out


def run_meta(model, hf_id: str) -> dict:
    import transformers

    return {
        "model": hf_id,
        "commit": model_commit(model, hf_id),
        "code": code_version(),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def model_tag(hf_id: str) -> str:
    return hf_id.split("/")[-1]


def peak_gib() -> float | None:
    """Peak memory our allocator held during this process (for grounding estimate_need_gib)."""
    return torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
