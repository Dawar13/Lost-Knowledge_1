"""Decode a few shipped-config samples for one category on CPU and show how
sample_check.norm() sees them. Debug aid for empty 'valid' counts."""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "src")
from mb5.common import load_model, read_jsonl  # noqa: E402
from mb5.prompts import GEN_TEMPLATES, encode_prompt, render  # noqa: E402
from mb5.sample_check import generate_samples, norm  # noqa: E402

torch.set_grad_enabled(False)
model_id, cid = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 30
data = Path("data/tailcat/pilot")
cats = {c["id"]: c for c in json.loads((data / "categories.json").read_text())}
c = cats[cid]
names = {norm(f) for e in read_jsonl(data / "entities.jsonl") if e["category"] == cid
         for f in [e["label"]] + e.get("aliases", [])}
m, tok = load_model(model_id, device="cpu")
prompt = render(tok, GEN_TEMPLATES[0].format(singular=c["singular"], plural=c["plural"]), True)
print(repr(prompt))
ids = encode_prompt(tok, prompt, True)
outs = generate_samples(m, ids, n, 24, "cpu", tok.pad_token_id, None, batch_size=n, seed=0)
for s in outs:
    raw = tok.decode(s, skip_special_tokens=True)
    t = norm(raw)
    print(f"{t in names!s:5s} {raw!r:60s} -> {t!r}  ids={s[:6]}")
