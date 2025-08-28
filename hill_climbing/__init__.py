"""
Hill Climbing (HC) package for Student Project Allocation (SPA).

This package contains:
- operators.py -> neighbourhood move functions (e.g., reassign, swap)
- hc_main.py  -> main Hill Climbing algorithm using evaluate_solution() from core/fitness.py

Goal:
Implement and test a Hill Climbing search to improve allocations from
either random or greedy initialisation, optimising fairness and feasibility.

"""

__all__ = ["operators"]
