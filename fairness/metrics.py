"""
Fairness metrics for SPA allocations.

Used to *report* equity indicators (not to drive the core objective directly).
Functions exposed:
  - gini(values): float in [0, 1] measuring inequality of satisfaction.
  - ranks_to_satisfaction(ranks, max_rank=6, num_prefs=None): ndarray of scores.
  - leximin_compare(sat_a, sat_b): {-1, 0, 1} lexical min ordering compare.

These are consistent with the dissertation’s fairness discussion (Gini / Leximin).
"""

from __future__ import annotations
from typing import Iterable, Sequence, Optional

import numpy as np

__all__ = ["gini", "ranks_to_satisfaction", "leximin_compare"]


def gini(values: Iterable[float]) -> float:
    """
    Computing the Gini coefficient of a 1‑D array of *non‑negative* values.
    Interpreted here as satisfaction scores (higher = better; 0 allowed).

    Robustness:
      - Empty / all-NaN -> returns 0.0
      - NaNs are ignored (rows dropped) before computation
      - Negative values are shifted so min becomes 0
      - Clamps result to [0, 1] to avoid floating noise

    Parameters

    values : Iterable[float]
        Satisfaction values (non-negative ideally).

    Returns

    float
        Gini coefficient in [0, 1]. 0 = perfectly equal, 1 = maximal inequality.
    """
    x = np.asarray(list(values), dtype=float)

    if x.size == 0:
        return 0.0

    # Droping NaNs for numerical stability
    x = x[~np.isnan(x)]
    if x.size == 0:
        return 0.0

    # Shift if any negative (shouldn't happen for satisfaction)
    mn = np.nanmin(x)
    if mn < 0:
        x = x - mn

    # All zeros -> define Gini = 0 (equal but poor)
    if np.allclose(x, 0.0):
        return 0.0

    # Sort ascending and apply standard formula
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    denom = cumx[-1]
    # Safe guard (shouldn't be 0 because of check above)
    if denom == 0:
        return 0.0

    g = (n + 1 - 2 * (cumx / denom).sum()) / n

    # Clamp to [0, 1] to avoid tiny negatives/overs due to FP error
    return float(max(0.0, min(1.0, g)))


def ranks_to_satisfaction(
    ranks: Sequence[float],
    max_rank: int = 6,
    num_prefs: Optional[int] = None,
) -> np.ndarray:
    """
    Converting preference rank (1 = best .. max_rank = worst) -> satisfaction score.

    Policy:
      score = max_rank - rank + 1
      Out-of-range ranks produce scores <= 0; final scores are clipped at 0.

    Parameters

    ranks : Sequence[float]
        Preference ranks.
    max_rank : int, default 6
        Worst rank (defines the top score). Ignored if num_prefs is provided.
    num_prefs : int, optional
        If provided, overrides max_rank. Kept for compatibility with core/fitness.py.

    Returns

    np.ndarray
        Satisfaction scores (>= 0).
    """
    if num_prefs is not None:
        max_rank = int(num_prefs)

    r = np.asarray(ranks, dtype=float)
    # NaNs get worst score (0 after clipping)
    r = np.where(np.isnan(r), max_rank, r)
    scores = (max_rank - r + 1)
    return np.clip(scores, a_min=0, a_max=None)


def leximin_compare(sat_a: Sequence[float], sat_b: Sequence[float]) -> int:
    """
    Leximin ordering: sort satisfaction ascending and compare the worst-off first.

    Returns
  
    int
        +1 if A is better under leximin,
        -1 if B is better,
         0 if tie (including identical arrays).

    Notes

    - If lengths differ, comparison is made on the common prefix of the
      sorted arrays; if equal over that prefix and lengths differ, the
      longer vector with remaining non-negative values is considered better.
      (This is a pragmatic policy; free to enforce equal lengths upstream.)
    """
    a = np.sort(np.asarray(sat_a, dtype=float))
    b = np.sort(np.asarray(sat_b, dtype=float))

    # Comparing elementwise over the shortest length
    m = min(a.size, b.size)
    for ai, bi in zip(a[:m], b[:m]):
        if ai > bi:
            return 1
        if ai < bi:
            return -1

    # If equal over the common prefix, prefer the longer one with any positive tail
    if a.size > b.size and np.any(a[m:] > 0):
        return 1
    if b.size > a.size and np.any(b[m:] > 0):
        return -1

    return 0
