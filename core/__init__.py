"""
Core scoring and utilities for the Student-Project Allocation (SPA) system.

Public API:

- evaluate_solution(...) : master objective & constraint scorer
- Weights                : dataclass for config weights
- list_to_alloc_df(...)  : helper to convert allocation lists into ranked DataFrames

Usage:
    from core import evaluate_solution, list_to_alloc_df
"""

from .fitness import evaluate_solution, Weights
from .utils import list_to_alloc_df

__all__ = [
    "evaluate_solution",
    "Weights",
    "list_to_alloc_df",
]

__version__ = "1.0.0"
