# spa_analysis/scripts/utils/logger.py
"""
LEGACY LOGGER (Provenance Only)

This file is retained only for provenance and debugging history.
It is **NOT used in the final pipeline** (replaced by `core/utils.py`).

Kept so that older phase-2 experiments and `phase2_checks.csv`
remain reproducible for personal understanding.

Lightweight CSV logger for experiment summaries.
- Always writes to <project root>/results/phase2_checks.csv
- Adds timestamp + run_id so multiple runs are easy to group/filter.

"""

import warnings
warnings.warn(
    "Deprecated: logger.py is legacy and kept only for provenance. "
    "Not used in the final pipeline.",
    DeprecationWarning,
    stacklevel=2
)

import os, csv, hashlib
from datetime import datetime

# Resolving the project root regardless of where the script is called
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def file_sha256(path: str) -> str:
    """Return short SHA-256 (first 8 chars) of a file for reproducibility."""
    if not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:8]

def log_breakdown(breakdown: dict, tag: str, run_id: str, extras: dict | None = None):
    """
    Appending a row to results/phase2_checks.csv

    Args:
      breakdown: e.g. {"pref_penalty": 287, "capacity_viol": 34, ...}
      tag:       label for the row (e.g., 'phase2_smoke', 'phase2_greedy')
      run_id:    identifier for this script execution (same for all rows in one run)
      extras:    optional dict to add more columns (e.g., dataset_hash, algo_name)
    """
    os.makedirs(os.path.join(BASE, "results"), exist_ok=True)
    path = os.path.join(BASE, "results", "phase2_checks.csv")

    row = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "run_id": run_id,         # <-- NEW: group multiple rows from same run
        "tag": tag,
        **breakdown,
        **(extras or {}),
    }

    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
