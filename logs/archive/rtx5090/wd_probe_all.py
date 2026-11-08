"""Time the inner member/foil SPARQL query of every category (COUNT only)."""
import sys
import time

import requests
import yaml

sys.path.insert(0, "src")
from mb5.tailcat import ENDPOINT, HEADERS  # noqa: E402

cfg = yaml.safe_load(open(sys.argv[1]))
only = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
for c in cfg["categories"]:
    if only and c["id"] not in only:
        continue
    for kind in ("where", "foil_where"):
        q = "SELECT (COUNT(DISTINCT ?item) AS ?n) WHERE { " + c[kind] + " }"
        t = time.time()
        try:
            r = requests.get(ENDPOINT, params={"query": q}, headers=HEADERS, timeout=90)
            n = r.json()["results"]["bindings"][0]["n"]["value"] if r.ok else None
            print(f"{c['id']:26s} {kind:10s} HTTP {r.status_code} n={n} {time.time() - t:5.1f}s", flush=True)
        except Exception as e:
            print(f"{c['id']:26s} {kind:10s} FAIL {type(e).__name__} {time.time() - t:5.1f}s", flush=True)
        time.sleep(1)
print("PROBE DONE", flush=True)
