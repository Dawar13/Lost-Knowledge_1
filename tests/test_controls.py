"""mb5.controls: the model fixed-effect statistics separate a between-model gap from a within-model
association, and the stage statistic works within categories (synthetic data, CPU)."""
import math

import numpy as np

from mb5.controls import _boot, _stage_stats, _within_stats


def _tab(within_slope, gap, n=11, seed=0):
    rng = np.random.default_rng(seed)
    cats = [f"c{i}" for i in range(n)]
    tab = {}
    for m, off in (("base", 0.0), ("instruct", gap)):
        x = rng.uniform(0, 1, n)
        y = off + within_slope * x + rng.normal(0, 0.01, n)
        tab[m] = {c: (float(xi + off), float(yi)) for c, xi, yi in zip(cats, x, y)}
    return tab, cats


def test_between_model_gap_alone_gives_pooled_but_not_within_correlation():
    tab, cats = _tab(within_slope=0.0, gap=5.0)
    s = _within_stats(tab, ["base", "instruct"], cats)
    assert s["pooled_spearman"] > 0.5  # the gap alone makes the pooled correlation look strong
    assert abs(s["within_pearson_model_fe"]) < 0.5
    assert abs(s["fe_slope"]) < 0.1


def test_within_model_association_is_recovered():
    tab, cats = _tab(within_slope=1.0, gap=5.0)
    s = _within_stats(tab, ["base", "instruct"], cats)
    assert all(v > 0.9 for v in s["per_model"].values())
    assert s["within_rank_corr"] > 0.9 and math.isclose(s["fe_slope"], 1.0, abs_tol=0.05)
    ci = _boot(_within_stats, tab, ["base", "instruct"], cats, b=50)
    lo, hi = ci["fe_slope"]
    assert lo < 1.0 < hi or math.isclose(lo, 1.0, abs_tol=0.05)


def test_stage_stats_within_category():
    cats = [f"c{i}" for i in range(5)]
    stages = ["s0", "s1", "s2", "s3"]
    tab = {s: {c: (k * 0.1 + i, k * 0.2 + 10 * i) for i, c in enumerate(cats)} for k, s in enumerate(stages)}
    st = _stage_stats(tab, stages, cats)
    assert st["share_categories_positive"] == 1.0 and math.isclose(st["mean_within_category_spearman"], 1.0)
