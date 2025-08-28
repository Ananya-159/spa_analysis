"""
Smoke test for fairness metrics in SPA.

This version:
- Works from ANY working directory (adds project root to sys.path).
- Saves CSV with JSON-quoted lists so Excel keeps ranks/satisfaction in ONE cell.

What it does:
1) Reads num_preferences from config.json (fallback = 6).
2) Runs two scenarios (equalish vs skewed), converts ranks->satisfaction.
3) Computes Gini and Leximin comparison, prints a table.
4) Saves results to results/smoke/ as CSV + Markdown with a UTC timestamp.
"""

from __future__ import annotations

import json
import sys
import csv
from pathlib import Path
from typing import Dict, Any, List, Tuple


# Making sure project root (spa_analysis) is importable 

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1]               # .../spa_analysis
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Project-local imports
from fairness import gini, ranks_to_satisfaction, leximin_compare

# Timestamp helper (prefer core.utils; fallback if not available)
try:
    from core.utils import utc_timestamp_filename
except Exception:
    from datetime import datetime, timezone
    def utc_timestamp_filename() -> str:
        """UTC timestamp safe for filenames, e.g., 2025-08-13T02-30-00Z"""
        return (datetime.now(timezone.utc)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z")
                .replace(":", "-"))

# Paths
CONFIG_PATH = ROOT / "config.json"
RESULTS_DIR = ROOT / "results" / "smoke"


#  helpers 

def load_num_prefs(default: int = 6) -> int:
    """Read num_preferences from config.json; fallback to `default` on any issue."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        n = int(cfg.get("num_preferences", default))
        return n if n > 0 else int(default)
    except Exception:
        return int(default)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def format_row(cols: List[str], widths: List[int]) -> str:
    return " | ".join(col.ljust(w) for col, w in zip(cols, widths))


def scenario_results(name: str, ranks: List[int], num_prefs: int) -> Dict[str, Any]:
    """
    Convert ranks -> satisfaction, compute Gini, return summary dict.
    """
    scores = ranks_to_satisfaction(ranks, num_prefs=num_prefs)
    g = float(gini(scores))
    return {
        "case": name,
        "ranks": list(ranks),
        "scores": [int(s) for s in scores],  # ints are nicer to read
        "gini": g,
    }


#  main 

def main() -> Tuple[List[Dict[str, Any]], str]:
    print("[SMOKE] Fairness metrics\n")

    num_prefs = load_num_prefs(default=6)
    print(f"[info] num_preferences = {num_prefs}\n")

    # Two simple scenarios
    ranks_equalish = [1, 2, 3, 2, 1, 3]   # relatively fair
    ranks_skewed   = [1, 6, 6, 6, 6, 6]   # more unequal

    res_equal = scenario_results("equalish", ranks_equalish, num_prefs)
    res_skewed = scenario_results("skewed", ranks_skewed, num_prefs)

    # Leximin comparison on satisfaction arrays
    cmp_val = leximin_compare(res_equal["scores"], res_skewed["scores"])
    cmp_text = {1: "equalish > skewed (fairer)", -1: "skewed > equalish", 0: "tie"}[cmp_val]

    # Console table
    rows = [res_equal, res_skewed]
    widths = [10, 22, 28, 8]
    header = ["case", "ranks", "satisfaction", "gini"]
    print(format_row(header, widths))
    print(format_row(["-" * w for w in widths], widths))
    for r in rows:
        print(format_row(
            [r["case"], str(r["ranks"]), str(r["scores"]), f"{r['gini']:.4f}"],
            widths,
        ))
    print("\nLeximin comparison:", cmp_text)

    # Saving the outputs
    ensure_dir(RESULTS_DIR)
    ts = utc_timestamp_filename()

    # Excel‑friendly CSV (lists stored as JSON strings) 
    csv_path = RESULTS_DIR / f"fairness_smoke_{ts}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case", "ranks", "satisfaction", "gini"])
        for r in rows:
            # Dump lists as JSON strings so Excel keeps them in a single cell
            w.writerow([
                r["case"],
                json.dumps(r["ranks"], ensure_ascii=False),
                json.dumps(r["scores"], ensure_ascii=False),
                f"{r['gini']:.6f}",
            ])

    #  Markdown summary 
    md_path = RESULTS_DIR / f"fairness_smoke_{ts}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Fairness Smoke Test\n\n")
        f.write(f"- Timestamp: {ts}\n")
        f.write(f"- num_preferences: {num_prefs}\n\n")
        f.write("| case     | ranks                 | satisfaction               | gini   |\n")
        f.write("|----------|-----------------------|----------------------------|--------|\n")
        for r in rows:
            f.write(f"| {r['case']:<8} | {str(r['ranks']):<21} | {str(r['scores']):<26} | {r['gini']:.4f} |\n")
        f.write(f"\n**Leximin:** {cmp_text}\n")

    print(f"\nSaved:\n  - {csv_path}\n  - {md_path}\n")
    print("[SMOKE] Done.")
    return rows, ts


if __name__ == "__main__":
    main()
