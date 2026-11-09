"""mb5.region: percentile ranks, the gap contrast, and that adjusting for composition removes a gap that is
pure composition (synthetic data, CPU)."""
import math

import numpy as np

from mb5.region import REF, REGIONS, SOUTH, WEST, design, fit_gee, gap_contrast, paired, sitelink_percentiles


def test_sitelink_percentiles_mid_rank_with_ties():
    ents = [{"qid": q, "category": "c", "sitelinks": s} for q, s in (("Q1", 1), ("Q2", 5), ("Q3", 5), ("Q4", 9))]
    p = sitelink_percentiles(ents)
    assert math.isclose(p["Q1|c"], 0.5 / 4)
    assert math.isclose(p["Q2|c"], p["Q3|c"]) and math.isclose(p["Q2|c"], (2.5 - 0.5) / 4)
    assert math.isclose(p["Q4|c"], 3.5 / 4)


def test_gap_contrast_weights():
    rows = [{"region": r, "type": "t"} for r in REGIONS]
    _, names = design(rows, ())
    c = gap_contrast(names)
    assert c[names.index("const")] == 0
    for r in SOUTH:
        assert math.isclose(c[names.index(f"region[{r}]")], 1 / len(SOUTH))
    for r in WEST:
        if r != REF:
            assert math.isclose(c[names.index(f"region[{r}]")], -1 / len(WEST))
    assert f"region[{REF}]" not in names


def test_composition_adjustment_removes_a_pure_composition_gap():
    rng = np.random.default_rng(0)
    rows, y, groups = [], [], []
    for i in range(80):
        reg = REGIONS[i % len(REGIONS)]
        n_rec = int(rng.integers(60, 120)) if reg in WEST else int(rng.integers(5, 40))
        for j in range(n_rec):
            p = 1 / (1 + math.exp(-(-3 + 1.2 * math.log(n_rec))))  # depends on composition only
            yy = int(rng.random() < p)
            rows.append({"region": reg, "type": "t", "log_n_recognised": math.log(n_rec)})
            y.append(yy)
            groups.append(f"c{i}")
    X0, n0 = design(rows, ())
    X1, n1 = design(rows, ("log_n_recognised",))
    g0 = fit_gee(y, X0, groups, n0)["gap_logodds"]
    g1 = fit_gee(y, X1, groups, n1)
    assert g0 < -0.5  # south looks less unreachable before adjustment
    assert abs(g1["gap_logodds"]) < abs(g0) / 3  # most of the gap goes once composition is in
    assert g1["terms"]["log_n_recognised"]["coef"] > 0.5


def test_paired_loss_rate():
    cats = {"a": {"region": "europe"}, "b": {"region": "africa"}}
    R = {"I": {"rec": {"e1|a": True, "e2|b": True, "e3|b": True},
               "reach": {("e1|a", 0): {"s": False}, ("e2|b", 0): {"s": False}, ("e3|b", 0): {"s": True}}},
         "B": {"rec": {"e1|a": True, "e2|b": True, "e3|b": False},
               "reach": {("e1|a", 0): {"s": True}, ("e2|b", 0): {"s": True}, ("e3|b", 0): {"s": True}}}}
    out = {r["region"]: r for r in paired("I", "B", R, "s", cats)}
    assert out["ALL"]["rows_recognised_by_both"] == 2  # e3 is not recognised by the base model
    assert out["ALL"]["loss_rate_given_base_reachable"] == 1.0
    assert out["africa"]["rbu_instruct"] == 1.0 and out["africa"]["rbu_base"] == 0.0
