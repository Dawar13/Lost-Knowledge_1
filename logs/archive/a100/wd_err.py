"""Show the WDQS error body for a failing query and try alternative forms."""
import sys
import time

import requests

sys.path.insert(0, "src")
from mb5.tailcat import ENDPOINT, HEADERS  # noqa: E402

forms = {
    "group_runFirst": "{ ?item wdt:P131* wd:Q1186 } hint:Prior hint:runFirst true . ?item wdt:P31/wdt:P279* wd:Q4022 .",
    "subquery_runFirst": "{ SELECT ?item WHERE { ?item wdt:P131* wd:Q1186 } } hint:Prior hint:runFirst true . "
                         "?item wdt:P31/wdt:P279* wd:Q4022 .",
    "subquery_classes": "{ SELECT ?item WHERE { ?item wdt:P131* wd:Q1186 } } hint:Prior hint:runFirst true . "
                        "?item wdt:P31 ?cls . { SELECT ?cls WHERE { ?cls wdt:P279* wd:Q4022 } } ",
}
for name, where in forms.items():
    q = "SELECT (COUNT(DISTINCT ?item) AS ?n) WHERE { " + where + " }"
    t = time.time()
    r = requests.get(ENDPOINT, params={"query": q}, headers=HEADERS, timeout=90)
    if r.ok:
        print(f"{name:20s} HTTP 200 n={r.json()['results']['bindings'][0]['n']['value']} {time.time() - t:.1f}s")
    else:
        lines = [ln for ln in r.text.splitlines() if "Exception" in ln][:2]
        print(f"{name:20s} HTTP {r.status_code} {time.time() - t:.1f}s :: {' | '.join(ln[:300] for ln in lines)}")
    time.sleep(2)
