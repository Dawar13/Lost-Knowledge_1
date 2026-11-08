import time, requests, sys
sys.path.insert(0, "src")
from mb5.tailcat import QUERY, ENDPOINT, HEADERS
where = "?item wdt:P31 wd:Q5 ; wdt:P106 wd:Q49757 . { ?item wdt:P6886 wd:Q5885 } UNION { ?item wdt:P1412 wd:Q5885 }"
variants = {
  "original": QUERY.format(where=where, lang="en", limit=20000),
  "runOnce": QUERY.replace("{{ SELECT DISTINCT ?item WHERE {{ {where} }} LIMIT {limit} }}",
                           "{{ SELECT DISTINCT ?item WHERE {{ {where} }} LIMIT {limit} }} hint:SubQuery hint:runOnce true .").format(where=where, lang="en", limit=20000),
}
# fix: hint must sit inside the subquery group
variants["runOnce"] = QUERY.replace("{{ SELECT DISTINCT ?item WHERE {{ {where} }} LIMIT {limit} }}",
    "{{ SELECT DISTINCT ?item WHERE {{ {where} }} LIMIT {limit} }}").format(where=where, lang="en", limit=20000)
variants["runOnce"] = variants["runOnce"].replace("LIMIT 20000 }", "LIMIT 20000 } hint:Prior hint:runFirst true .", 1)
variants["optimizerNone"] = "PREFIX hint: <http://www.bigdata.com/queryHints#>\n" + QUERY.format(where=where, lang="en", limit=20000).replace("WHERE {\n", "WHERE {\n  hint:Query hint:optimizer \"None\" .\n", 1)
for name, q in variants.items():
    t = time.time()
    try:
        r = requests.get(ENDPOINT, params={"query": q}, headers=HEADERS, timeout=90)
        n = len(r.json()["results"]["bindings"]) if r.ok else None
        print(f"{name:14s} HTTP {r.status_code} rows {n} in {time.time()-t:.1f}s", flush=True)
    except Exception as e:
        print(f"{name:14s} failed {type(e).__name__} {time.time()-t:.1f}s", flush=True)
    time.sleep(2)
