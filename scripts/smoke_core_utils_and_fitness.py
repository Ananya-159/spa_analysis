"""
Master smoke tests (with CSV + Markdown saves) for:
  • core/utils.list_to_alloc_df()
  • core/fitness.evaluate_solution()

This version:
  1) Saves a timestamped Markdown summary in results/smoke/
  2) Also saves a "latest" Markdown copy in results/summary/ for easy viewing
  3) Prints dataset hash for reproducibility
  4) Includes script version/date in summary

Run from anywhere:
    python scripts/smoke_core_utils_and_fitness.py
"""

from __future__ import annotations
import sys
import json
import hashlib
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone


# Script metadata

SCRIPT_VERSION = "1.0"
SCRIPT_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%d")


# Ensure project root is importable regardless of current working directory

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1]  # .../spa_analysis
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.utils import list_to_alloc_df, utc_timestamp, utc_timestamp_filename  # noqa: E402
from core.fitness import evaluate_solution  # noqa: E402



# Dataset helper functions

def make_students_df(num_prefs: int, eligibility_list: list[bool] | None = None) -> pd.DataFrame:
    """Build a Students DataFrame with preferences padded to num_prefs."""
    if eligibility_list is None:
        eligibility_list = [True, True, True]
    data = {
        "Student ID": [1, 2, 3],
        "Average": [70, 65, 80],
        "Client Based Eligibility": eligibility_list,
        "Preference 1": ["P1", "P2", "P3"],
        "Preference 2": ["P2", "P3", "P1"],
        "Preference 3": ["P3", "P1", "P2"],
    }
    for i in range(4, num_prefs + 1):
        data[f"Preference {i}"] = [None, None, None]
    return pd.DataFrame(data)


def make_projects_df() -> pd.DataFrame:
    """Minimal Projects DataFrame (all required columns)."""
    return pd.DataFrame({
        "Project ID": ["P1", "P2", "P3"],
        "Supervisor ID": ["S1", "S1", "S2"],
        "Project Type": ["Standard", "Client based", "Standard"],
        "Min Students": [1, 1, 1],
        "Max Students": [2, 1, 2],
        "Minimum Average Required": [60, 60, 60],
        "Second Round Eligible": [True, True, True],
    })


def make_supervisors_df() -> pd.DataFrame:
    """Minimal Supervisors DataFrame (all required columns)."""
    return pd.DataFrame({
        "Supervisor ID": ["S1", "S2"],
        "Max Student Capacity": [3, 2],
        "Min Student Capacity": [1, 1],
    })


def make_allocation_df() -> pd.DataFrame:
    """Allocation where all students get their 1st choice."""
    return pd.DataFrame({
        "Student ID": [1, 2, 3],
        "assigned_project": ["P1", "P2", "P3"],
        "assigned_rank": [1, 1, 1],
    })



# Utility: dataset hash for reproducibility

def dataset_hash(dataset_path: Path) -> str:
    """Return SHA256 hash of dataset file contents."""
    if not dataset_path.exists():
        return "N/A"
    h = hashlib.sha256()
    with open(dataset_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# UTILS smoke tests

def run_utils_tests(num_prefs: int) -> pd.DataFrame:
    """Run Clean + Out-of-list cases for list_to_alloc_df()."""
    students_df = make_students_df(num_prefs)

    allocation_clean = ["P1", "P2", "P3"]
    df_clean = list_to_alloc_df(allocation_clean, students_df, num_prefs)
    df_clean.insert(0, "case", "utils_clean")

    allocation_oob = ["P1", "P9", "P3"]
    df_oob = list_to_alloc_df(allocation_oob, students_df, num_prefs)
    df_oob.insert(0, "case", "utils_oob")

    return pd.concat([df_clean, df_oob], ignore_index=True)



# FITNESS smoke tests

def run_fitness_tests(num_prefs: int, config: dict) -> pd.DataFrame:
    """Run Clean + Eligibility violation cases for evaluate_solution()."""
    projects_df = make_projects_df()
    supervisors_df = make_supervisors_df()
    alloc_df = make_allocation_df()

    rows = []

    students_clean = make_students_df(num_prefs, [True, True, True])
    _, bd_c = evaluate_solution(alloc_df, students_clean, projects_df, supervisors_df, config, with_fairness=True)
    rows.append({"case": "fitness_clean", **bd_c})

    students_bad = make_students_df(num_prefs, [True, False, True])
    _, bd_v = evaluate_solution(alloc_df, students_bad, projects_df, supervisors_df, config, with_fairness=True)
    rows.append({"case": "fitness_elig_violation", **bd_v})

    return pd.DataFrame(rows)



# Save helpers

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_config_snapshot(cfg: dict, out_dir: Path, ts: str) -> Path:
    snap_path = out_dir / f"config_snapshot_{ts}.json"
    with open(snap_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return snap_path


def save_markdown_summary(utils_df: pd.DataFrame, fitness_df: pd.DataFrame,
                          out_dir: Path, ts: str, ds_hash: str) -> Path:
    md_path = out_dir / f"smoke_summary_{ts}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Smoke Test Summary ({utc_timestamp()})\n")
        f.write(f"**Script version:** {SCRIPT_VERSION} ({SCRIPT_DATE})\n\n")
        f.write(f"**Dataset hash:** {ds_hash}\n\n")
        f.write("## UTILS: list_to_alloc_df()\n")
        f.write(utils_df.to_markdown(index=False))
        f.write("\n\n## FITNESS: evaluate_solution()\n")
        f.write(fitness_df.to_markdown(index=False))
    return md_path



# Main

def main() -> None:
    cfg_path = ROOT / "config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    num_prefs = int(cfg.get("num_preferences", 6))
    ts = utc_timestamp_filename()

    # Calculating the dataset hash 
    ds_path = ROOT / "data" / "SPA_Dataset_With_Min_Max_Capacity.xlsx"
    ds_hash = dataset_hash(ds_path)

    print(f"[smoke] num_preferences = {num_prefs}")
    print(f"[smoke] dataset_hash = {ds_hash}")

    utils_df = run_utils_tests(num_prefs)
    fitness_df = run_fitness_tests(num_prefs, cfg)

    out_dir = ROOT / "results" / "smoke"
    ensure_dir(out_dir)

    utils_path = out_dir / f"utils_smoke_{ts}.csv"
    fitness_path = out_dir / f"fitness_smoke_{ts}.csv"
    utils_df.to_csv(utils_path, index=False)
    fitness_df.to_csv(fitness_path, index=False)

    cfg_snapshot_path = save_config_snapshot(cfg, out_dir, ts)
    md_summary_path = save_markdown_summary(utils_df, fitness_df, out_dir, ts, ds_hash)

    # Saving "latest" Markdown copy to docs/ for GitHub rendering
    latest_dir = ROOT / "results" / "summary"
    ensure_dir(latest_dir)
    latest_md_path = latest_dir / "smoke_summary_latest.md"
    with open(latest_md_path, "w", encoding="utf-8") as f:
     f.write(open(md_summary_path, "r", encoding="utf-8").read())

    print("\nSaved:")
    print(f"  - {utils_path}")
    print(f"  - {fitness_path}")
    print(f"  - {cfg_snapshot_path}")
    print(f"  - {md_summary_path}")
    print(f"  - {latest_md_path} (latest snapshot)")

    print("\nQuick expectations:")
    print("  UTILS:")
    print("    • Clean case → _rank_oob_violation = 0 for all, ranks mostly 1")
    print("    • Out-of-list → one row with _rank_oob_violation = 1, rank = num_prefs")
    print("  FITNESS:")
    print("    • Clean case → pref_penalty = 3; total = 3 × gamma_pref; other violations 0")
    print("    • Eligibility case → elig_viol = 1; total = (3 × gamma_pref) + (1 × beta_eligibility)")


if __name__ == "__main__":
    main()
