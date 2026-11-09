"""Multilingual invented-name metric (mb5.invented_ml) and its use in the G3b analysis."""
import json
from pathlib import Path

import pytest

from mb5.common import read_jsonl
from mb5.frontier import _candidates
from mb5.invented_ml import Matcher, audit_sheet, candidates, ck, reclassify, script
from mb5.p6_analysis import apply_classes, e15_summary, g3b_report
from mb5.sample_check import norm

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "data/tailcat/pilot_v2"
P5 = ROOT / "results/P5"


def test_keys_scripts_and_candidates():
    assert ck(norm("村上 春樹")) == "村上春樹" and ck(norm("Haruki  Murakami")) == "haruki murakami"
    assert script("夏目漱石") == "cjk" and script("Natsume Soseki") == "latin"
    assert script("திருவள்ளுவர்") == "other" and script("1234") == "none"
    # a full-width parenthesis becomes "(" under NFKC, with no space before it
    assert "泰戈尔" in candidates(norm("泰戈尔（注：实际上泰戈尔是用孟加拉语写作的）"))
    t = norm("Natsume Soseki (novelist)")
    assert candidates(t) == _candidates(t) and "natsume soseki" in candidates(t)  # non-CJK: unchanged


def _tiny(tmp_path: Path) -> Path:
    d = tmp_path / "split"
    (d / "raw").mkdir(parents=True)
    ents = [{"qid": "Q1", "label": "Natsume Soseki", "aliases": [], "sitelinks": 90, "category": "nov"},
            {"qid": "Q4", "label": "Kobo Abe", "aliases": [], "sitelinks": 50, "category": "nov"},
            {"qid": "Q5", "label": "Abe Kobo Two", "aliases": [], "sitelinks": 5, "category": "nov"}]
    (d / "entities.jsonl").write_text("\n".join(json.dumps(e) for e in ents), encoding="utf-8")
    (d / "foils.jsonl").write_text(json.dumps({"qid": "Q3", "label": "Leo Tolstoy", "sitelinks": 99,
                                               "category": "nov"}), encoding="utf-8")
    (d / "valid_names.json").write_text(json.dumps({"nov": ["natsume soseki", "kobo abe", "abe kobo two",
                                                            "mori ogai"]}), encoding="utf-8")
    rows = [{"qid": q, "label": "x", "aliases": [], "sitelinks": 1} for q in ("Q1", "Q2", "Q4", "Q5")]
    (d / "raw" / "nov.members.en.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")
    return d


ML = {"Q1": ["夏目 漱石", "Natsume Sōseki", "x"], "Q2": ["森鷗外"], "Q3": ["托尔斯泰"],
      "Q4": ["安部公房"], "Q5": ["安部公房"]}  # Q4 and Q5 share a name: ambiguous, not indexed


def test_matcher_multilingual_rules(tmp_path):
    d = _tiny(tmp_path)
    en, ml = Matcher(d), Matcher(d, ML)
    assert en.classify("夏目漱石", "nov") == ("invented", None)
    assert ml.classify("夏目漱石", "nov") == ("member", "Q1")          # kanji, spaces ignored
    assert ml.classify("Natsume Sōseki", "nov") == ("member", "Q1")    # another-language Latin label
    assert ml.classify("Natsume Soseki", "nov") == ("member", "Q1")    # English index first
    assert ml.classify("森鷗外", "nov") == ("valid", None)               # uncapped member
    assert ml.classify("托尔斯泰（俄国作家）", "nov") == ("foil", None)  # cut at the parenthesis
    assert ml.classify("安部公房", "nov") == ("valid", None)  # ambiguous between members Q4 and Q5: a real name, no QID
    assert ml.classify("x", "nov") == ("invented", None)                 # one-character names are not indexed
    assert ml.classify("泰戈尔", "nov") == ("invented", None)
    assert ml.classify("  ", "nov") == ("empty", None)
    assert ml.ambiguous["nov"] == 1


def test_reclassify_checks_the_english_class(tmp_path):
    d = _tiny(tmp_path)
    s = [{"spec": "reference", "category": "nov", "i": 0, "text": "夏目漱石", "class": "invented", "member_qid": None},
         {"spec": "reference", "category": "nov", "i": 1, "text": "Kobo Abe", "class": "member", "member_qid": "Q4"}]
    rows, mism = reclassify(s, Matcher(d), Matcher(d, ML))
    assert mism == 0 and [r["class_ml"] for r in rows] == ["member", "member"]
    s[1]["class"] = "invented"
    assert reclassify(s, Matcher(d), Matcher(d, ML))[1] == 1


@pytest.mark.skipif(not (P5 / "Qwen2.5-7B-Instruct/samples_frontier.jsonl.gz").exists(), reason="no P5 samples")
def test_english_matcher_reproduces_every_stored_p5_class():
    m = Matcher(PILOT)
    for model in ("Qwen2.5-7B-Instruct", "OLMo-2-1124-7B-Instruct"):
        bad = [r for r in read_jsonl(P5 / model / "samples_frontier.jsonl.gz")
               if m.classify(r["text"], r["category"]) != (r["class"], r["member_qid"])]
        assert not bad, (model, len(bad), bad[:3])


def test_audit_sheet_is_stratified_distinct_and_seeded():
    pool = [{"model": m, "spec": s, "category": "c", "answer": f"{m}{s}{i % 7}"}
            for m in ("A", "B") for s in ("x", "y") for i in range(40)]
    a, b = audit_sheet(pool, 12, 0), audit_sheet(pool, 12, 0)
    assert a == b and len(a) == 12
    assert len({(r["model"], r["spec"], r["answer"]) for r in a}) == 12
    assert {(r["model"], r["spec"]) for r in a} == {("A", "x"), ("A", "y"), ("B", "x"), ("B", "y")}
    assert all(r["human_label"] == "" for r in a)


def test_g3b_under_multilingual_classes():
    # reference: 6 member answers, 4 kanji answers of member m2 counted invented in English
    recog = [{"kind": "foil", "category": "c", "V": i / 100, "qid": f"F{i}"} for i in range(30)]
    recog += [{"kind": "member", "category": "c", "qid": q, "V": v, "mc_correct": 1} for q, v in [("m1", .9), ("m2", .8)]]
    reach = [{"qid": q, "category": "c", "reach__reference": q == "m1", "robust_unreach__reference": q != "m1",
              "reach__untruncated_T1": True, "robust_unreach__untruncated_T1": False,
              "reach__k100_T1": True, "robust_unreach__k100_T1": False} for q in ("m1", "m2")]
    S, C = [], []
    for i in range(10):
        kanji = i >= 6
        for spec in ("reference", "untruncated_T1", "k100_T1"):
            S.append({"spec": spec, "category": "c", "i": i, "text": "漢字" if kanji else "m1",
                      "class": "invented" if kanji else "member", "member_qid": None if kanji else "m1"})
            C.append({"spec": spec, "category": "c", "i": i, "class_en": S[-1]["class"],
                      "member_qid_en": S[-1]["member_qid"], "class_ml": "member", "member_qid_ml": "m2" if kanji else "m1"})
    scores = [{"category": "c", "key": "漢字", "V": 0.0}]
    en = e15_summary(S, recog, scores, reach)
    ml = e15_summary(apply_classes(S, C), recog, scores, reach)
    assert en["reference_unfiltered"]["invented_rate"] == pytest.approx(0.4)
    assert ml["reference_unfiltered"]["invented_rate"] == 0
    # under the multilingual metric the kanji answers are member m2 and use its stored V (kept)
    assert ml["quantiles"]["0.95"]["rows"]["untruncated_T1"]["filtered"]["n_kept"] == 10
    assert en["quantiles"]["0.95"]["rows"]["untruncated_T1"]["filtered"]["n_kept"] == 6
    rep = g3b_report({"A": ml | {"metric": "multilingual"}}, english={"A": en | {"metric": "english"}})
    assert "English-only invented-name metric" in rep and "multilingual class of mb5.invented_ml" in rep
    with pytest.raises(ValueError):
        apply_classes(S, C[:-1])
    C[0] = dict(C[0], class_en="foil")
    with pytest.raises(ValueError):
        apply_classes(S, C)
