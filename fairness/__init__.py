"""
Fairness metrics for Student-Project Allocation (SPA).

Exports:
- gini(values): float in [0, 1] measuring inequality of satisfaction.
- ranks_to_satisfaction(ranks, max_rank=6, num_prefs=None): ndarray scores.
- leximin_compare(sat_a, sat_b): {-1, 0, 1} leximin ordering compare.

These metrics are used for reporting (optional) and do not affect the core
objective unless explicitly added as a weighted term.
"""

from .metrics import gini, ranks_to_satisfaction, leximin_compare

__all__ = ["gini", "ranks_to_satisfaction", "leximin_compare"]
