# core/utils.py
"""
Utilities for normalising and validating allocation outputs across algorithms.

`list_to_alloc_df(...)` converts a raw allocation (e.g., a list/array of project IDs in
student order) into the **canonical** DataFrame consumed by `core/fitness.evaluate_solution(...)`.
It computes the assigned preference rank, flags out-of-bounds assignments, and enforces
shape/column checks so downstream scoring is reliable and reproducible.
"""

from __future__ import annotations
from typing import Sequence, Any
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Converts any algorithm's output list into the canonical allocation DataFrame
# expected by core/fitness.evaluate_solution() for scoring.
def list_to_alloc_df(
    allocation: Sequence[Any],
    students_df: pd.DataFrame,
    num_prefs: int,
    project_col_name: str = "assigned_project",
) -> pd.DataFrame:
    """
    Building the canonical allocation DataFrame given an allocation list.

    Returns columns:
      - 'Student ID'
      - project_col_name (default 'assigned_project')
      - 'assigned_rank' (1..num_prefs; worst rank if not in list)
      - '_rank_oob_violation' (0/1; 1 if assigned project not in top-N)
    """
    if len(allocation) != len(students_df):
        raise ValueError("allocation length must match students_df length (row-aligned).")

    if "Student ID" not in students_df.columns:
        raise ValueError("students_df must contain 'Student ID'.")

    out = pd.DataFrame({
        "Student ID": students_df["Student ID"].to_numpy(),
        project_col_name: np.asarray(allocation),
    })

    # Preference columns validation
    pref_cols = [f"Preference {i}" for i in range(1, num_prefs + 1)]
    missing = [c for c in pref_cols if c not in students_df.columns]
    if missing:
        raise ValueError(f"students_df missing preference columns: {missing}")

    # Coerce to str to avoid mismatches when Excel preferences are strings and allocation is int
    assigned = out[project_col_name].astype(str).to_numpy().reshape(-1, 1)   # (n,1)
    pref_mat = students_df[pref_cols].astype(str).to_numpy()                 # (n,num_prefs)

    equals = (pref_mat == assigned)                                          # bool matrix
    # First (leftmost) match index +1; 0 means not found
    first_true = np.where(equals, np.arange(1, num_prefs + 1), 0).max(axis=1)
    oob = first_true == 0

    out["assigned_rank"] = np.where(oob, num_prefs, first_true).astype(int)
    out["_rank_oob_violation"] = oob.astype(int)
    return out


def utc_timestamp_filename() -> str:
    """
    Returning a UTC ISO-8601 timestamp with 'Z' suffix, safe for filenames
    (colons replaced with '-'), e.g., '2025-08-10T22-47-20Z'.
    """
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
        .replace(":", "-")
    )


def utc_timestamp() -> str:
    """
    Return an ISO-8601 UTC timestamp with 'Z' suffix (human/CSV friendly),
    e.g., '2025-08-11T01:02:35Z'.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
