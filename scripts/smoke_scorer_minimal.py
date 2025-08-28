# scripts/smoke_scorer_minimal.py
"""
Minimal end-to-end smoke test for core.fitness.evaluate_solution().
Covers:
 - vectorised joins (alloc -> projects, alloc -> students)
 - client eligibility violation
 - min-average violation
 - unknown project ID in allocation (counts as violation, no crash)
 - project & supervisor under-fill penalties
 - rank derivation via core.utils.list_to_alloc_df
 - optional fairness (gini_satisfaction)
"""

import pandas as pd

import os, sys

# Adding the project root to sys.path 
HERE = os.path.abspath(os.path.dirname(__file__))     # .../spa_analysis/scripts
ROOT = os.path.abspath(os.path.join(HERE, ".."))      # .../spa_analysis
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# import from package
from core.fitness import evaluate_solution
from core.utils   import list_to_alloc_df

def main():
   
    # Tiny dataset (4 students)
   
    students_df = pd.DataFrame({
        "Student ID": ["S1", "S2", "S3", "S4"],
        "Average":    [75,   80,   55,   65],
        "Client Based Eligibility": [True, False, True, True],
        "Preference 1": ["P1", "P2", "P2", "P2"],
        "Preference 2": ["P2", "P3", "P1", "P3"],
        "Preference 3": ["P3", "P1", "P3", "P1"],
    })

    projects_df = pd.DataFrame({
        "Project ID": ["P1", "P2"],
        "Project Title": ["Proj 1", "Proj 2"],
        "Supervisor ID": ["A", "B"],
        "Project Type": ["Client based", "Research based"],
        "Min Students": [2, 2],
        "Max Students": [5, 5],
        "Pre-requisites": ["", ""],
        "Minimum Average Required": [70, 60],
        "Second Round Eligible": [False, False],
    })

    supervisors_df = pd.DataFrame({
        "Supervisor ID": ["A", "B"],
        "Supervisor Name": ["Sup A", "Sup B"],
        "No. of Projects": [1, 1],
        "Min Student Capacity": [3, 3],
        "Max Student Capacity": [6, 6],
    })

   
    # Config (inline)
    
    cfg = {
        "num_preferences": 3,
        "weights": {
            "gamma_pref": 1.0,
            "alpha_capacity": 1.0,
            "beta_eligibility": 1.0,
            "delta_under": 20.0,   # under-fill weight
        },
        "paths": {
            "data_excel": "N/A",
            "results_dir": "results",
        }
    }


    # Allocation that triggers everything:
    #  S1 -> P1 (ok, rank 1)
    #  S2 -> P1 (client-based but NOT eligible -> violation)
    #  S3 -> P2 (avg 55 < 60 -> min-average violation)
    #  S4 -> BAD (unknown project -> violation; also out-of-list)
    # Under-fill: P2 gets only 1 (<2); Sup A gets 2 (<3), Sup B gets 1 (<3)
 
    allocation = ["P1", "P1", "P2", "BAD"]

    alloc_df = list_to_alloc_df(
        allocation=allocation,
        students_df=students_df,
        num_prefs=cfg["num_preferences"],
        project_col_name="assigned_project"
    )

    print("\n=== Allocation DF ===")
    print(alloc_df)

  
    # Run scorer (no fairness)
  
    total, br = evaluate_solution(
        alloc_df,
        students_df,
        projects_df,
        supervisors_df,
        cfg,
        with_fairness=False
    )
    print("\n=== SCORER (with_fairness=False) ===")
    print("TOTAL:", total)
    print("BREAKDOWN:", br)


    # Run scorer (with fairness)
  
    total_f, br_f = evaluate_solution(
        alloc_df,
        students_df,
        projects_df,
        supervisors_df,
        cfg,
        with_fairness=True
    )
    print("\n=== SCORER (with_fairness=True) ===")
    print("TOTAL:", total_f)
    print("BREAKDOWN:", br_f)

    # Quick sanity hints (don’t assert, just print expectations):
    # - elig_viol should be >= 3 (client mismatch + min avg + unknown ID)
    # - under_cap should be > 0 (project P2 under-filled, supervisors under-filled)
    # - pref_penalty present; total includes delta_under * under_cap
    # - with_fairness=True adds gini_satisfaction to the breakdown

if __name__ == "__main__":
    main()
