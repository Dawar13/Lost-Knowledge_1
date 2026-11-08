"""Exact reachability of an answer string under a sampler.

For an answer tokenised as t_1..t_L after prompt x, one teacher-forced forward
pass gives the next-token logits z_j at every answer position. From these we
compute, per position j:

  rank_j   number of tokens with a strictly larger logit than t_j
           (top-k keeps t_j  iff  rank_j < k)
  G_j(T)   probability mass, at temperature T, of tokens strictly more likely
           than t_j  (top-p keeps t_j  iff  G_j(T) < p, the transformers/vLLM rule)
  gap_j    z_max - z_j  (min-p keeps t_j  iff  exp(-gap_j / T) >= min_p)

and per answer:

  p_star(T)    = max_j G_j(T)        reachable under top-p  iff  p > p_star
  k_star       = max_j rank_j + 1    reachable under top-k  iff  k >= k_star
  minp_star(T) = min_j exp(-gap_j/T) reachable under min-p  iff  min_p <= minp_star

These analytic values ignore repetition penalty and assume one truncation at a
time; they draw the curves. The authoritative yes/no for a full spec
(e.g. shipped defaults: rep. penalty + T + top-k + top-p) comes from running
the actual transformers processor chain on every row and checking whether the
target token survives (its processed logit stays finite).
"""
from __future__ import annotations

import math

import torch

from .samplers import build_chain

TEMPS = (1.0, 0.7)


def _left_pad_prefixes(prefixes: list[list[int]], device) -> torch.Tensor:
    """Left-pad each prefix with its own first token. Repetition penalty only
    looks at the *set* of previous tokens, so this padding is exact."""
    L = max(len(p) for p in prefixes)
    out = torch.empty((len(prefixes), L), dtype=torch.long)
    for i, p in enumerate(prefixes):
        out[i] = torch.tensor([p[0]] * (L - len(p)) + p, dtype=torch.long)
    return out.to(device)


def row_stats(rows: torch.Tensor, targets: torch.Tensor, temps=TEMPS) -> dict[str, torch.Tensor]:
    """rows: [R, V] float32 raw logits; targets: [R] token ids."""
    z_t = rows.gather(1, targets[:, None])  # [R,1]
    above = rows > z_t  # strictly more likely than the target, at any T
    out = {
        "rank": above.sum(1),
        "gap": rows.max(1).values - z_t.squeeze(1),
    }
    for T in temps:
        logp = torch.log_softmax(rows / T, dim=1)
        out[f"G_T{T}"] = (logp.exp() * above).sum(1)
        out[f"logp_T{T}"] = logp.gather(1, targets[:, None]).squeeze(1)
    return out


def chain_kept(chain, rows: torch.Tensor, targets: torch.Tensor, prefixes: list[list[int]],
               shift: float = 0.0) -> torch.Tensor:
    """True where the target token survives the processor chain.

    shift moves the target's raw logit by that amount relative to every other
    token before the chain runs. Every processor we model (repetition penalty,
    temperature, top-k, top-p, min-p) is monotone in the target's relative
    logit, so kept(shift=+m) False means the target is cut even if numeric
    jitter between two computations of the logits favours it by m; kept(-m)
    True means it survives jitter of m against it. A non-zero shift also
    breaks every tie with the target, so the result no longer depends on how
    the sort orders tied tokens."""
    if len(chain) == 0:
        return torch.ones(len(targets), dtype=torch.bool, device=rows.device)
    ids = _left_pad_prefixes(prefixes, rows.device)
    x = rows.clone()
    if torch.is_tensor(shift) or shift:  # scalar, or one shift per row
        x[torch.arange(len(targets), device=x.device), targets] += shift
    out = chain(ids, x)
    return torch.isfinite(out.gather(1, targets[:, None]).squeeze(1))


DSTAR_BOUND = 8.0  # raw logits; |d*| is clipped here
DSTAR_ITERS = 10  # bisection steps: resolution 2 * 8 / 2**10 = 0.016 logits


def signed_margin(chain, rows: torch.Tensor, targets: torch.Tensor, prefixes: list[list[int]]) -> torch.Tensor:
    """Per row, the signed logit margin d of the target under the chain:
    d > 0: kept, and still kept with its logit lowered by up to d;
    d < 0: cut, and still cut with its logit raised by up to |d|.
    Found by bisection on the shift (kept is monotone in it), clipped to
    +-DSTAR_BOUND. An answer's margin is the minimum over its positions, so it
    is robustly unreachable at margin eps iff that minimum is below -eps."""
    R = len(targets)
    dev = rows.device
    if len(chain) == 0:
        return torch.full((R,), DSTAR_BOUND, device=dev)
    B = DSTAR_BOUND
    k_top = chain_kept(chain, rows, targets, prefixes, shift=torch.full((R,), B, device=dev))
    k_bot = chain_kept(chain, rows, targets, prefixes, shift=torch.full((R,), -B, device=dev))
    lo = torch.full((R,), -B, device=dev)  # largest shift known to be cut
    hi = torch.full((R,), B, device=dev)  # smallest shift known to be kept
    for _ in range(DSTAR_ITERS):
        mid = (lo + hi) / 2
        k = chain_kept(chain, rows, targets, prefixes, shift=mid)
        hi = torch.where(k, mid, hi)
        lo = torch.where(k, lo, mid)
    d = -(lo + hi) / 2
    d = torch.where(k_bot, torch.full_like(d, B), d)  # kept even at -B
    d = torch.where(~k_top, torch.full_like(d, -B), d)  # cut even at +B
    return d


@torch.no_grad()
def _forward_rows(model, batch, device, pad_id: int):
    """batch: list of (prompt_ids, answer_ids). Returns logits rows predicting
    each answer token, in order, as float32 [sum L, V]."""
    seqs = [p + a[:-1] for p, a in batch]
    S = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), S), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), S), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        att[i, : len(s)] = 1
    logits = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits
    rows = []
    for i, (p, a) in enumerate(batch):
        P = len(p)
        rows.append(logits[i, P - 1 : P - 1 + len(a)].float())
    return torch.cat(rows, 0)


@torch.no_grad()
def score_items(
    model,
    items: list[dict],
    specs: dict[str, dict],
    device: str = "cuda",
    batch_size: int = 32,
    row_chunk: int = 512,
    pad_id: int = 0,
    keep_positions: bool = True,
    progress: bool = True,
    margin: float = 0.0,
    dstar: bool = True,
) -> list[dict]:
    """items: dicts with 'prompt_ids', 'answer_ids' (+ any metadata, copied through).

    Returns one record per item with analytic critical values and, for every
    spec, exact reachability and the first failing answer position (-1 if none).

    margin (raw-logit units, see chain_kept) adds a three-way classification
    that is robust to numeric jitter between this teacher-forced pass and any
    other computation of the same logits (e.g. KV-cached generate()):
      robust_unreach__S  some position is cut even with the target raised by margin
      robust_reach__S    every position survives with the target lowered by margin
      neither            boundary: the verdict depends on numerics
    With margin 0 both equal the exact verdict.

    dstar=True also records dstar__S, the answer's signed logit margin (minimum
    over positions of signed_margin) and dstar_pos__S, so RBU can be computed
    at any margin afterwards: robustly unreachable at eps iff dstar < -eps.
    """
    chains = {name: build_chain(spec) for name, spec in specs.items()}
    # Sort by length to reduce padding; restore order at the end.
    order = sorted(range(len(items)), key=lambda i: len(items[i]["prompt_ids"]) + len(items[i]["answer_ids"]))
    results: list[dict | None] = [None] * len(items)
    n_batches = math.ceil(len(order) / batch_size)
    for b in range(n_batches):
        idx = order[b * batch_size : (b + 1) * batch_size]
        batch = [(items[i]["prompt_ids"], items[i]["answer_ids"]) for i in idx]
        rows = _forward_rows(model, batch, device, pad_id)
        # Row bookkeeping.
        targets, prefixes, owner, pos = [], [], [], []
        for k, (p, a) in enumerate(batch):
            for j, t in enumerate(a):
                targets.append(t)
                prefixes.append(p + a[:j])
                owner.append(k)
                pos.append(j)
        targets_t = torch.tensor(targets, dtype=torch.long, device=rows.device)
        stats: dict[str, list] = {}
        kept: dict[str, list] = {n: [] for n in chains}
        kept_hi: dict[str, list] = {n: [] for n in chains}
        kept_lo: dict[str, list] = {n: [] for n in chains}
        dmarg: dict[str, list] = {n: [] for n in chains}
        for c0 in range(0, rows.shape[0], row_chunk):
            sl = slice(c0, c0 + row_chunk)
            pre = prefixes[c0 : c0 + row_chunk]
            st = row_stats(rows[sl], targets_t[sl])
            for key, v in st.items():
                stats.setdefault(key, []).append(v.cpu())
            for n, ch in chains.items():
                k0 = chain_kept(ch, rows[sl], targets_t[sl], pre).cpu()
                kept[n].append(k0)
                if margin:
                    kept_hi[n].append(chain_kept(ch, rows[sl], targets_t[sl], pre, shift=+margin).cpu())
                    kept_lo[n].append(chain_kept(ch, rows[sl], targets_t[sl], pre, shift=-margin).cpu())
                else:
                    kept_hi[n].append(k0)
                    kept_lo[n].append(k0)
                if dstar:
                    dmarg[n].append(signed_margin(ch, rows[sl], targets_t[sl], pre).cpu())
        dmarg = {n: torch.cat(v).tolist() for n, v in dmarg.items()} if dstar else {}
        stats = {k: torch.cat(v).tolist() for k, v in stats.items()}
        kept = {n: torch.cat(v).tolist() for n, v in kept.items()}
        kept_hi = {n: torch.cat(v).tolist() for n, v in kept_hi.items()}
        kept_lo = {n: torch.cat(v).tolist() for n, v in kept_lo.items()}
        # Aggregate per item.
        per = [dict(rank=[], gap=[], **{f"G_T{T}": [] for T in TEMPS}, **{f"logp_T{T}": [] for T in TEMPS},
                    **{f"kept__{n}": [] for n in chains}, **{f"kepthi__{n}": [] for n in chains},
                    **{f"keptlo__{n}": [] for n in chains}, **{f"d__{n}": [] for n in chains}) for _ in batch]
        for r, k in enumerate(owner):
            d = per[k]
            d["rank"].append(stats["rank"][r])
            d["gap"].append(stats["gap"][r])
            for T in TEMPS:
                d[f"G_T{T}"].append(stats[f"G_T{T}"][r])
                d[f"logp_T{T}"].append(stats[f"logp_T{T}"][r])
            for n in chains:
                d[f"kept__{n}"].append(bool(kept[n][r]))
                d[f"kepthi__{n}"].append(bool(kept_hi[n][r]))
                d[f"keptlo__{n}"].append(bool(kept_lo[n][r]))
                if dstar:
                    d[f"d__{n}"].append(dmarg[n][r])
        for k, i in enumerate(idx):
            d = per[k]
            rec = {key: val for key, val in items[i].items() if key not in ("prompt_ids",)}
            rec["L"] = len(items[i]["answer_ids"])
            for T in TEMPS:
                G = d[f"G_T{T}"]
                rec[f"p_star_T{T}"] = max(G)
                rec[f"p_star_pos_T{T}"] = int(max(range(len(G)), key=G.__getitem__))
                rec[f"logp_sum_T{T}"] = float(sum(d[f"logp_T{T}"]))
                rec[f"minp_star_T{T}"] = float(min(math.exp(-g / T) for g in d["gap"]))
            rec["k_star"] = int(max(d["rank"]) + 1)
            rec["k_star_pos"] = int(max(range(len(d["rank"])), key=d["rank"].__getitem__))
            for n in chains:
                ks = d[f"kept__{n}"]
                rec[f"reach__{n}"] = all(ks)
                rec[f"failpos__{n}"] = next((j for j, v in enumerate(ks) if not v), -1)
                hi = d[f"kepthi__{n}"]
                rec[f"robust_unreach__{n}"] = not all(hi)
                rec[f"robust_reach__{n}"] = all(d[f"keptlo__{n}"])
                rec[f"failpos_robust__{n}"] = next((j for j, v in enumerate(hi) if not v), -1)
                if dstar:
                    dm = d[f"d__{n}"]
                    jm = int(min(range(len(dm)), key=dm.__getitem__))
                    rec[f"dstar__{n}"] = round(float(dm[jm]), 4)
                    rec[f"dstar_pos__{n}"] = jm
            if keep_positions:
                rec["G_T1.0_pos"] = [round(x, 6) for x in d["G_T1.0"]]
                rec["rank_pos"] = d["rank"]
            rec["margin"] = margin
            results[i] = rec
        if progress and (b % 20 == 0 or b == n_batches - 1):
            print(f"[reach] batch {b + 1}/{n_batches}", flush=True)
    return results  # type: ignore[return-value]
