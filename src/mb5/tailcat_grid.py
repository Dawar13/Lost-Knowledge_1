"""Generate the full TailCat category config (P3, E1) as a (type x country) grid
plus Indian-language writer categories. Writes plain YAML (no dependencies).

  python -m mb5.tailcat_grid --out configs/tailcat_full.yaml

Design (docs/03 E1; fixed in notes/decisions.md before the build):
  * 14 entity types x 24 countries; 17 countries are from South Asia, Africa,
    Southeast Asia and Latin America (target: at least 40% of kept categories).
  * Foils: the same type in two neighbouring or easily confused countries
    (FOILS below; foil countries need not be member countries).
  * 12 Indian-language writer categories by writing language (P6886), foils
    from two sibling languages: needed for the multilingual analysis (E8).
  * Fast query shapes only: direct P31 / P106 (no P279* paths), country via
    P17 (places, clubs), P27 (people) or P495 (films).
  * After the build, a category is kept only if it has at least 30 members and
    30 foils with min_sitelinks 2 (the builder's sanity report); others are
    dropped by that rule alone.
QIDs: Wikidata items; the sanity report's example labels are the check.
"""
from __future__ import annotations

import argparse
from pathlib import Path

COUNTRIES = {  # qid, display name, region
    "india": ("Q668", "India", "south_asia"), "pakistan": ("Q843", "Pakistan", "south_asia"),
    "bangladesh": ("Q902", "Bangladesh", "south_asia"), "sri_lanka": ("Q854", "Sri Lanka", "south_asia"),
    "nepal": ("Q837", "Nepal", "south_asia"),
    "nigeria": ("Q1033", "Nigeria", "africa"), "kenya": ("Q114", "Kenya", "africa"),
    "ghana": ("Q117", "Ghana", "africa"), "ethiopia": ("Q115", "Ethiopia", "africa"),
    "south_africa": ("Q258", "South Africa", "africa"),
    "indonesia": ("Q252", "Indonesia", "southeast_asia"), "philippines": ("Q928", "the Philippines", "southeast_asia"),
    "vietnam": ("Q881", "Vietnam", "southeast_asia"),
    "brazil": ("Q155", "Brazil", "latin_america"), "mexico": ("Q96", "Mexico", "latin_america"),
    "argentina": ("Q414", "Argentina", "latin_america"), "colombia": ("Q739", "Colombia", "latin_america"),
    "japan": ("Q17", "Japan", "east_asia"), "iran": ("Q794", "Iran", "west_asia"),
    "turkey": ("Q43", "Turkey", "west_asia"), "poland": ("Q36", "Poland", "europe"),
    "finland": ("Q33", "Finland", "europe"), "france": ("Q142", "France", "europe"),
    "usa": ("Q30", "the United States", "north_america"),
}
FOIL_COUNTRIES = {"south_korea": "Q884", "china": "Q148", "greece": "Q41", "czechia": "Q213", "sweden": "Q34",
                  "estonia": "Q191", "belgium": "Q31", "italy": "Q38", "canada": "Q16", "united_kingdom": "Q145"}
FOILS = {
    "india": ["pakistan", "bangladesh"], "pakistan": ["india", "iran"], "bangladesh": ["india", "pakistan"],
    "sri_lanka": ["india", "bangladesh"], "nepal": ["india", "pakistan"],
    "nigeria": ["ghana", "kenya"], "kenya": ["ethiopia", "nigeria"], "ghana": ["nigeria", "kenya"],
    "ethiopia": ["kenya", "nigeria"], "south_africa": ["kenya", "nigeria"],
    "indonesia": ["philippines", "vietnam"], "philippines": ["indonesia", "vietnam"],
    "vietnam": ["indonesia", "philippines"],
    "brazil": ["argentina", "colombia"], "mexico": ["colombia", "argentina"], "argentina": ["brazil", "mexico"],
    "colombia": ["mexico", "brazil"],
    "japan": ["south_korea", "china"], "iran": ["turkey", "pakistan"], "turkey": ["iran", "greece"],
    "poland": ["czechia", "finland"], "finland": ["sweden", "estonia"], "france": ["belgium", "italy"],
    "usa": ["canada", "united_kingdom"],
}

# type id: (singular template, plural template, member pattern with {C})
TYPES = {
    "poets": ("a poet from {N}", "poets from {N}", "?item wdt:P106 wd:Q49757 ; wdt:P27 wd:{C} ; wdt:P31 wd:Q5 ."),
    "novelists": ("a novelist from {N}", "novelists from {N}",
                  "?item wdt:P106 wd:Q6625963 ; wdt:P27 wd:{C} ; wdt:P31 wd:Q5 ."),
    "painters": ("a painter from {N}", "painters from {N}",
                 "?item wdt:P106 wd:Q1028181 ; wdt:P27 wd:{C} ; wdt:P31 wd:Q5 ."),
    "film_directors": ("a film director from {N}", "film directors from {N}",
                       "?item wdt:P106 wd:Q2526255 ; wdt:P27 wd:{C} ; wdt:P31 wd:Q5 ."),
    "composers": ("a composer from {N}", "composers from {N}",
                  "?item wdt:P106 wd:Q36834 ; wdt:P27 wd:{C} ; wdt:P31 wd:Q5 ."),
    "architects": ("an architect from {N}", "architects from {N}",
                   "?item wdt:P106 wd:Q42973 ; wdt:P27 wd:{C} ; wdt:P31 wd:Q5 ."),
    "rivers": ("a river in {N}", "rivers in {N}", "?item wdt:P31 wd:Q4022 ; wdt:P17 wd:{C} ."),
    "lakes": ("a lake in {N}", "lakes in {N}", "?item wdt:P31 wd:Q23397 ; wdt:P17 wd:{C} ."),
    "mountains": ("a mountain in {N}", "mountains in {N}", "?item wdt:P31 wd:Q8502 ; wdt:P17 wd:{C} ."),
    "islands": ("an island of {N}", "islands of {N}", "?item wdt:P31 wd:Q23442 ; wdt:P17 wd:{C} ."),
    "universities": ("a university in {N}", "universities in {N}", "?item wdt:P31 wd:Q3918 ; wdt:P17 wd:{C} ."),
    "museums": ("a museum in {N}", "museums in {N}", "?item wdt:P31 wd:Q33506 ; wdt:P17 wd:{C} ."),
    "films": ("a film from {N}", "films from {N}", "?item wdt:P31 wd:Q11424 ; wdt:P495 wd:{C} ."),
    "football_clubs": ("a football club from {N}", "football clubs from {N}",
                       "?item wdt:P31 wd:Q476028 ; wdt:P17 wd:{C} ."),
}

INDIAN_LANGUAGES = {  # id: (qid, name, sibling languages for foils)
    "tamil": ("Q5885", "Tamil", ["telugu", "malayalam"]), "telugu": ("Q8097", "Telugu", ["tamil", "kannada"]),
    "kannada": ("Q33673", "Kannada", ["telugu", "tamil"]), "malayalam": ("Q36236", "Malayalam", ["tamil", "kannada"]),
    "bengali": ("Q9610", "Bengali", ["odia", "assamese"]), "hindi": ("Q1568", "Hindi", ["urdu", "marathi"]),
    "marathi": ("Q1571", "Marathi", ["gujarati", "hindi"]), "gujarati": ("Q5137", "Gujarati", ["marathi", "hindi"]),
    "punjabi": ("Q58635", "Punjabi", ["urdu", "hindi"]), "odia": ("Q33810", "Odia", ["bengali", "assamese"]),
    "assamese": ("Q29401", "Assamese", ["bengali", "odia"]), "urdu": ("Q1617", "Urdu", ["hindi", "punjabi"]),
}


def _qid(country: str) -> str:
    return COUNTRIES[country][0] if country in COUNTRIES else FOIL_COUNTRIES[country]


def _union(pattern: str, qids: list[str]) -> str:
    return " UNION ".join("{ " + pattern.replace("{C}", q) + " }" for q in qids)


def categories() -> list[dict]:
    out = []
    for t, (sing, plur, pat) in TYPES.items():
        for c, (q, name, region) in COUNTRIES.items():
            out.append({"id": f"{t}__{c}", "type": t, "country": c, "region": region,
                        "singular": sing.format(N=name), "plural": plur.format(N=name),
                        "where": pat.replace("{C}", q),
                        "foil_where": _union(pat, [_qid(f) for f in FOILS[c]])})
    for lang, (q, name, sibs) in INDIAN_LANGUAGES.items():
        pat = "?item wdt:P6886 wd:{L} ; wdt:P31 wd:Q5 ."
        out.append({"id": f"writers_lang__{lang}", "type": "writers_by_language", "country": "india",
                    "region": "south_asia", "language": lang,
                    "singular": f"a writer who wrote in {name}", "plural": f"writers who wrote in {name}",
                    "where": pat.replace("{L}", q),
                    "foil_where": " UNION ".join("{ " + pat.replace("{L}", INDIAN_LANGUAGES[s][0]) + " }"
                                                 for s in sibs)})
    return out


def _yaml_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cats = categories()
    lines = ["# Full TailCat (P3). GENERATED by `python -m mb5.tailcat_grid`; edit the generator, not this file.",
             "settings:", "  member_cap: 400", "  foil_cap: 200", "  strata: 5", "  max_aliases: 3", "  seed: 0",
             "  min_sitelinks: 2", "", "categories:"]
    for c in cats:
        lines.append(f"  - id: {c['id']}")
        for k in ("type", "country", "region", "language", "singular", "plural", "where", "foil_where"):
            if k in c:
                lines.append(f"    {k}: {_yaml_str(c[k])}")
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    regions = {}
    for c in cats:
        regions[c["region"]] = regions.get(c["region"], 0) + 1
    target = sum(v for k, v in regions.items() if k in ("south_asia", "africa", "southeast_asia", "latin_america"))
    print(f"{len(cats)} categories -> {args.out}; by region {regions}; target regions {target / len(cats):.0%}")


if __name__ == "__main__":
    main()
