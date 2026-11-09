"""Maths of mb5.support (zero versus tiny): sight probabilities, prefix-minimal forms, bounds."""
import math

import numpy as np

from mb5.support import expected_distinct, n_star, p_seen, prefix_minimal, q_entity, q_trunc_bounds


def test_p_seen_matches_closed_form_and_is_stable():
    for q, n in [(0.5, 1), (0.1, 10), (0.01, 1000), (0.3, 7)]:
        assert math.isclose(float(p_seen(q, n)), 1 - (1 - q) ** n, rel_tol=1e-12)
    # tiny q: 1 - (1 - q)^n underflows naively; the stable form gives about n q
    assert math.isclose(float(p_seen(1e-15, 1000)), 1e-12, rel_tol=1e-6)
    assert float(p_seen(0.0, 10**9)) == 0.0
    assert float(p_seen(1.0, 1)) == 1.0
    assert np.all(np.diff(p_seen(1e-4, np.array([1, 10, 100, 1000, 10**6]))) >= 0)


def test_expected_distinct_rises_to_support_size():
    q = np.array([0.5, 0.25, 1e-3, 1e-6])
    vals = [expected_distinct(q, n) for n in (1, 10, 10**3, 10**6, 10**9)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    assert math.isclose(vals[0], q.sum(), rel_tol=1e-12)  # one sample: expected distinct = total mass
    assert vals[-1] <= len(q) and vals[-1] > len(q) - 1e-3
    assert expected_distinct(np.array([]), 100) == 0.0


def test_prefix_minimal_drops_extensions_and_duplicates():
    seqs = [(1, 2), (1, 2, 3), (4,), (1, 2), (5, 6), (5,)]
    # (1,2,3) extends (1,2); the second (1,2) is a duplicate; (5,6) extends (5,)
    assert prefix_minimal(seqs) == [0, 2, 5]
    assert prefix_minimal([(7, 8), (7, 9)]) == [0, 1]  # shared first token, neither a prefix: disjoint events


def test_q_entity_union_of_disjoint_forms():
    qs = [0.2, 0.05, 0.1, 0.01]
    seqs = [(1, 2), (1, 2, 3), (4,), (1, 9)]
    q_max, q_any = q_entity(qs, seqs)
    assert q_max == 0.2
    assert math.isclose(q_any, 0.2 + 0.1 + 0.01)  # (1,2,3) is inside the event "begins with (1,2)"
    assert q_entity([], []) == (0.0, 0.0)


def test_truncated_bounds():
    qs = [0.02, 0.01]
    seqs = [(1, 2), (3, 4, 5)]
    lo, hi = q_trunc_bounds(qs, seqs, sure=[True, False], maybe=[True, True], top_p=0.9)
    assert lo == 0.02
    assert math.isclose(hi, 0.02 / 0.9**2 + 0.01 / 0.9**3)
    assert lo <= hi
    lo, hi = q_trunc_bounds(qs, seqs, sure=[False, False], maybe=[False, False], top_p=0.9)
    assert lo == 0.0 and hi == 0.0  # robustly cut everywhere: exactly zero
    lo, hi = q_trunc_bounds(qs, seqs, sure=[True, True], maybe=[True, True], top_p=None)
    assert lo == 0.02 and hi is None  # no bound for chains other than pure top-p


def test_n_star_crossing():
    prompts = [np.array([0.5, 1e-3]), np.array([1e-2])]
    target = 2.0
    n = n_star(prompts, target)
    assert n is not None
    total = lambda m: sum(expected_distinct(q, m) for q in prompts)  # noqa: E731
    assert total(n) >= target - 1e-9 and total(n * 0.99) < target
    assert n_star(prompts, 10.0) is None  # more than the support can never be reached
