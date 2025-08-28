# scripts/verify_alloc_vs_prefs.py
"""
Verifying per-student allocation CSV ranks against the Students sheet preferences.

Usage:
  Running from repo root (same folder as config.json), e.g. in Spyder or:
    python scripts/verify_alloc_vs_prefs.py
"""

from __future__ import annotations
import os, json, glob
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_cfg():
    with open(os.path.join(ROOT, "config.json"), "r", encoding="utf-8") as f:
        return json.load(f)

def latest_alloc_csv(results_dir: str) -> str:
    pats = [
        os.path.join(results_dir, "*_hc_alloc*.csv"),
        os.path.join(results_dir, "*_ga_alloc*.csv"),   # future-proof if GA/HH use same pattern
        os.path.join(results_dir, "*_hh_alloc*.csv"),
    ]
    matches = []
    for p in pats:
        matches.extend(glob.glob(p))
    if not matches:
        raise FileNotFoundError(f"No per-student allocation CSVs found under {results_dir}")
    return max(matches, key=os.path.getmtime)

def expected_rank_for(student_row: pd.Series, assigned_project: str, num_prefs: int) -> int:
    # Build the preference list in order
    prefs = [student_row.get(f"Preference {i}") for i in range(1, num_prefs + 1)]
    try:
        return prefs.index(assigned_project) + 1
    except ValueError:
        return num_prefs  # convention: not in list -> worst rank

def main():
    cfg = load_cfg()
    num_prefs = int(cfg.get("num_preferences", 6))
    data_excel = os.path.join(ROOT, cfg["paths"]["data_excel"])
    results_dir = os.path.join(ROOT, cfg["paths"]["results_dir"])

    # Loading the dataset (Students only needed)
    students = pd.read_excel(data_excel, sheet_name="Students")
    students = students.set_index("Student ID")

    # Load latest allocation CSV
    alloc_path = latest_alloc_csv(results_dir)
    alloc = pd.read_csv(alloc_path)

    # Handle column names (with/without OutOfList)
    # Normalize columns to: Student ID, Assigned Project, Matched Preference Rank
    col_map = {
        "Student ID": "Student ID",
        "Assigned Project": "Assigned Project",
        "Matched Preference Rank": "Matched Preference Rank",
    }
    missing = [c for c in col_map if c not in alloc.columns]
    if missing:
        raise KeyError(f"{alloc_path} missing columns: {missing}")
    alloc = alloc[list(col_map.keys())].rename(columns=col_map)

    # Computing expected rank by looking at Students sheet
    expected = []
    missing_students = 0
    for _, row in alloc.iterrows():
        sid = row["Student ID"]
        pid = row["Assigned Project"]
        try:
            srow = students.loc[sid]
        except KeyError:
            # unknown Student ID in CSV
            expected.append({"Student ID": sid, "Assigned Project": pid, "expected_rank": num_prefs, "reason": "unknown_student"})
            missing_students += 1
            continue
        exp = expected_rank_for(srow, pid, num_prefs)
        expected.append({"Student ID": sid, "Assigned Project": pid, "expected_rank": exp})

    exp_df = pd.DataFrame(expected)
    merged = alloc.merge(exp_df, on=["Student ID", "Assigned Project"], how="left")

    # Flag mismatches
    merged["rank_matches"] = merged["Matched Preference Rank"].astype(int) == merged["expected_rank"].astype(int)
    mismatches = merged[~merged["rank_matches"]].copy()

    # Out-of-list counts (expected == num_prefs but matched might still be num_prefs)
    out_of_list = (merged["expected_rank"] == num_prefs).sum()

    # Summary
    total = len(merged)
    ok = int(merged["rank_matches"].sum())
    bad = total - ok
    print("\n=== Verification report ===")
    print(f"Allocation file: {os.path.relpath(alloc_path, ROOT)}")
    print(f"Students sheet : {os.path.relpath(data_excel, ROOT)}")
    print(f"num_preferences: {num_prefs}")
    print(f"Total rows     : {total}")
    print(f"Matches        : {ok}")
    print(f"Mismatches     : {bad}")
    print(f"Out-of-list    : {out_of_list}")
    if missing_students:
        print(f"Unknown Student IDs in allocation CSV: {missing_students}")

    if bad > 0:
        out_bad = os.path.join(results_dir, "verify_mismatches.csv")
        mismatches.to_csv(out_bad, index=False)
        print(f"\nWrote mismatch details → {os.path.relpath(out_bad, ROOT)}")
        print("Columns include: Student ID, Assigned Project, Matched Preference Rank, expected_rank")

if __name__ == "__main__":
    main()
