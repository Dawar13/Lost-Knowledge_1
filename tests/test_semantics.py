"""Our analytic keep-rules must agree with transformers' own warpers.

Run on CPU, no model download:  PYTHONPATH=src python -m pytest -q tests
"""
import torch

from mb5.reachability import _left_pad_prefixes, chain_kept, row_stats
from mb5.samplers import build_chain

torch.manual_seed(0)
B, V = 48, 400
EPS = 1e-5  # ignore targets sitting exactly on a threshold (float ties)


def _kept_by_chain(spec, rows, ids=None):
    ids = ids if ids is not None else torch.zeros((rows.shape[0], 1), dtype=torch.long)
    out = build_chain(spec)(ids, rows.clone())
    return torch.isfinite(out)


def _all_targets(rows):
    R = rows.shape[0]
    r = rows.repeat_interleave(V, 0)
    t = torch.arange(V).repeat(R)
    return r, t


def test_top_p_rule_matches_hf():
    rows = torch.randn(B, V) * 3
    for T in (1.0, 0.7):
        for p in (0.5, 0.8, 0.9, 0.95):
            kept = _kept_by_chain({"temperature": T, "top_p": p}, rows)
            r, t = _all_targets(rows)
            G = row_stats(r, t, temps=(T,))[f"G_T{T}"].view(B, V)
            clear = (G - p).abs() > EPS
            assert torch.equal(kept[clear], (G < p)[clear]), (T, p)


def test_top_k_rule_matches_hf():
    rows = torch.randn(B, V) * 3
    for k in (1, 5, 20, 50):
        kept = _kept_by_chain({"top_k": k}, rows)
        r, t = _all_targets(rows)
        rank = row_stats(r, t, temps=(1.0,))["rank"].view(B, V)
        assert torch.equal(kept, rank < k), k


def test_min_p_rule_matches_hf():
    rows = torch.randn(B, V) * 3
    for T in (1.0, 0.7):
        for m in (0.02, 0.05, 0.1, 0.3):
            kept = _kept_by_chain({"temperature": T, "min_p": m}, rows)
            r, t = _all_targets(rows)
            gap = row_stats(r, t, temps=(T,))["gap"].view(B, V)
            ratio = torch.exp(-gap / T)
            clear = (ratio - m).abs() > EPS
            assert torch.equal(kept[clear], (ratio >= m)[clear]), (T, m)


def test_top_k_then_top_p_renormalises():
    """With both, top-p acts on the top-k-renormalised distribution."""
    rows = torch.randn(B, V) * 3
    k, p = 20, 0.9
    kept = _kept_by_chain({"top_k": k, "top_p": p}, rows)
    masked = rows.masked_fill(~_kept_by_chain({"top_k": k}, rows), float("-inf"))
    probs = torch.softmax(masked, 1)
    for b in range(B):
        for t in range(V):
            if not torch.isfinite(masked[b, t]):
                assert not kept[b, t]
                continue
            G = probs[b][masked[b] > masked[b, t]].sum().item()
            if abs(G - p) > EPS:
                assert bool(kept[b, t]) == (G < p)


def test_left_padding_is_exact_for_repetition_penalty():
    rows = torch.randn(6, V) * 3
    prefixes = [[5, 9, 9, 17], [3], [4, 4, 4, 4, 4, 8], [1, 2], [7, 7], [0, 399, 12]]
    chain = build_chain({"repetition_penalty": 1.3, "top_p": 0.9})
    padded = _left_pad_prefixes(prefixes, rows.device)
    joint = chain(padded, rows.clone())
    for i, p in enumerate(prefixes):
        single = chain(torch.tensor([p]), rows[i : i + 1].clone())
        assert torch.equal(torch.isfinite(joint[i]), torch.isfinite(single[0]))


def test_chain_kept_helper():
    rows = torch.randn(10, V) * 3
    targets = torch.randint(0, V, (10,))
    prefixes = [[int(x)] for x in torch.randint(0, V, (10,))]
    spec = {"repetition_penalty": 1.05, "temperature": 0.7, "top_k": 20, "top_p": 0.8}
    got = chain_kept(build_chain(spec), rows, targets, prefixes)
    for i in range(10):
        out = build_chain(spec)(torch.tensor([prefixes[i]]), rows[i : i + 1].clone())
        assert bool(got[i]) == bool(torch.isfinite(out[0, targets[i]]))


def test_shift_is_monotone_and_breaks_ties():
    """chain_kept(shift=+m) >= kept(0) >= kept(shift=-m) elementwise, on bf16-like
    logits full of exact ties (rounded to 1/8), for every processor we model."""
    rows = (torch.randn(B, V) * 3 * 8).round() / 8  # many exact ties, as in bf16
    targets = torch.randint(0, V, (B,))
    prefixes = [[int(x) for x in torch.randint(0, V, (3,))] for _ in range(B)]
    for spec in ({"top_p": 0.9}, {"top_k": 20}, {"min_p": 0.05}, {"temperature": 0.7, "top_p": 0.8},
                 {"repetition_penalty": 1.05, "temperature": 0.7, "top_k": 20, "top_p": 0.8}):
        ch = build_chain(spec)
        for t_shift in range(V // 50):  # many targets per row
            t = (targets + t_shift * 50) % V
            k0 = chain_kept(ch, rows, t, prefixes)
            hi = chain_kept(ch, rows, t, prefixes, shift=+0.01)
            lo = chain_kept(ch, rows, t, prefixes, shift=-0.01)
            assert torch.all(hi >= k0) and torch.all(k0 >= lo), spec


def test_shifted_top_p_matches_strict_rule_without_ties():
    """After a shift no logit ties the target, so HF's top-p verdict must equal
    the analytic rule (mass strictly above < p) on the shifted logits."""
    rows = (torch.randn(B, V) * 3 * 8).round() / 8
    for m in (+0.01, -0.01):
        for t in range(0, V, 7):
            tt = torch.full((B,), t)
            x = rows.clone()
            x[:, t] += m
            G = row_stats(x, tt, temps=(1.0,))["G_T1.0"]
            kept = chain_kept(build_chain({"top_p": 0.9}), rows, tt, [[0]] * B, shift=m)
            clear = (G - 0.9).abs() > EPS
            assert torch.equal(kept[clear], (G < 0.9)[clear]), (m, t)


def test_signed_margin_is_the_flip_point():
    """d from signed_margin: the target is kept with shift -d + tol and cut with
    shift -d - tol (tol = bisection resolution), on tie-heavy logits, for
    single and combined samplers. Its sign agrees with the unshifted verdict
    away from the boundary."""
    from mb5.reachability import DSTAR_BOUND, signed_margin

    tol = 2 * DSTAR_BOUND / 2**10 + 1e-4
    rows = (torch.randn(B, V) * 3 * 8).round() / 8
    prefixes = [[int(x) for x in torch.randint(0, V, (3,))] for _ in range(B)]
    for spec in ({"top_p": 0.9}, {"top_k": 20}, {"min_p": 0.05},
                 {"repetition_penalty": 1.05, "temperature": 0.7, "top_k": 20, "top_p": 0.8}):
        ch = build_chain(spec)
        for t0 in range(0, V, 37):
            t = (torch.arange(B) * 13 + t0) % V
            d = signed_margin(ch, rows, t, prefixes)
            inside = d.abs() < DSTAR_BOUND - tol
            up = chain_kept(ch, rows, t, prefixes, shift=-d + tol)
            down = chain_kept(ch, rows, t, prefixes, shift=-d - tol)
            assert torch.all(up[inside]) and not torch.any(down[inside]), spec
            k0 = chain_kept(ch, rows, t, prefixes)
            clear = d.abs() > tol
            assert torch.equal(k0[clear], (d > 0)[clear]), spec
