# genetic_algorithm/ga_utils.py

"""
Utilities shared by the Genetic Algorithm (GA):
- Dataset loading (supports Excel or alternate path via config)
- Population initialization (random or greedy)
- Individual repair (soft repair to fix invalid project IDs only)
- Run ID generation (timestamped, safe for filenames)
- Convergence logging to CSV
- Final allocation export (with preference ranks + notes)
"""

from __future__ import annotations

import os
import sys
import csv
import random
from datetime import datetime, timezone
from typing import Dict, Tuple, List

import pandas as pd

# Bootstrap so `core.*` imports work from any subfolder
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Core helpers
from core.utils import list_to_alloc_df
from core.fitness import evaluate_solution  # (not used here, but available for quick checks)

#  Data Loading

def _pick_dataset_path(cfg: Dict) -> str:
    """
    Resolve dataset path from config['paths'].
    Supports either 'dataset' or 'data_excel'.
    """
    paths = cfg.get("paths", {}) or {}
    if "data_excel" in paths:
        rel = paths["data_excel"]
    elif "dataset" in paths:
        rel = paths["dataset"]
    else:
        raise ValueError("config['paths'] must contain 'data_excel' or 'dataset'")
    return os.path.abspath(os.path.join(ROOT, rel))

def load_dataset(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load Excel workbook and normalize all ID columns to strings.
    Sheets: 'Students', 'Projects', 'Supervisors'
    """
    path = _pick_dataset_path(config)

    students_df = pd.read_excel(path, sheet_name="Students")
    projects_df = pd.read_excel(path, sheet_name="Projects")
    supervisors_df = pd.read_excel(path, sheet_name="Supervisors")

    # Normalize IDs to strings (safe for joins)
    for df, col in [(projects_df, "Project ID"), 
                    (projects_df, "Supervisor ID"),
                    (supervisors_df, "Supervisor ID"),
                    (students_df, "Student ID")]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return students_df, projects_df, supervisors_df

# Initialisation & Repair 

def _row_preferences(row: pd.Series, num_prefs: int) -> List[str]:
    """
    Extracts a student's list of preferences.
    Supports column headers: 'Preference 1', 'Preference 2', etc.
    """
    prefs = []
    for i in range(1, num_prefs + 1):
        col = f"Preference {i}"
        if col in row and pd.notna(row[col]):
            prefs.append(str(row[col]).strip())
    return prefs

def initialise_population(init_strategy: str,
                          students_df: pd.DataFrame,
                          projects_df: pd.DataFrame,
                          config: Dict):
    """
    Initialize one DEAP Individual (list of project IDs).
    - 'random': randomly pick one of the student's preferences (or fallback).
    - 'greedy': assign the first preference if possible (or fallback).
    """
    from deap import creator

    num_prefs = int(config.get("num_preferences", 6))
    valid_projects = projects_df["Project ID"].astype(str).str.strip().tolist()
    valid_set = set(valid_projects)

    genes = []
    for _, row in students_df.iterrows():
        prefs = _row_preferences(row, num_prefs)
        prefs = [p for p in prefs if p in valid_set]

        if init_strategy == "greedy" and prefs:
            genes.append(prefs[0])
        elif prefs:
            genes.append(random.choice(prefs))
        else:
            genes.append(random.choice(valid_projects))  # fallback: random valid project

    return creator.Individual(genes)

def repair_individual(individual: List[str],
                      students_df: pd.DataFrame,
                      projects_df: pd.DataFrame,
                      config: Dict) -> List[str]:
    """
    Soft repair:
    Fixes only truly invalid project IDs (i.e., not found in the project list).
    Does NOT fix:
      - Over-capacity
      - Eligibility
    These will be handled by the fitness function.

    Replaces invalid project with:
    - A valid project from student's preferences, if available.
    - Else, a random valid project from the dataset.
    """
    num_prefs = int(config.get("num_preferences", 6))
    valid_projects = set(projects_df["Project ID"].astype(str).str.strip().tolist())

    for i, pid in enumerate(individual):
        p = str(pid).strip()
        if p not in valid_projects:
            # Attempting to replace with one of student's valid preferences
            prefs = [
                str(students_df.iloc[i].get(f"Preference {j+1}")).strip()
                for j in range(num_prefs)
                if pd.notna(students_df.iloc[i].get(f"Preference {j+1}"))
            ]
            prefs = [q for q in prefs if q in valid_projects]
            individual[i] = random.choice(prefs) if prefs else random.choice(list(valid_projects))

    return individual

# Logging & Run ID 

def generate_run_id() -> str:
    """
    Generate a UTC run ID (filename-safe).
    Example: '20250818T231045Z'
    """
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def log_convergence(convergence_list: List[Dict], filename: str) -> None:
    """
    Save per-generation convergence metrics to a CSV.
    Fields expected: generation, total, pref_penalty, capacity_viol, etc.
    """
    if not convergence_list:
        return
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(convergence_list[0].keys()))
        writer.writeheader()
        writer.writerows(convergence_list)

#  Exporting the Allocation 

def save_allocation_csv(individual: List[str],
                        students_df: pd.DataFrame,
                        filename: str,
                        config: Dict) -> None:
    """
    Convert a chromosome into a per-student allocation CSV, including:
    - Student ID
    - Assigned Project
    - Matched Preference Rank
    - OutOfList (1 if not in preferences)
    - Rank Note (e.g., "Worst (in list)")
    """
    num_prefs = int(config.get("num_preferences", 6))
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    alloc_df = list_to_alloc_df(
        individual,
        students_df,
        num_prefs,
        project_col_name="assigned_project",
    )

    pref_cols = [f"Preference {i}" for i in range(1, num_prefs + 1)]
    pref_sets = {
        str(row["Student ID"]).strip(): {
            str(row[c]).strip()
            for c in pref_cols
            if c in students_df.columns and pd.notna(row[c])
        }
        for _, row in students_df.iterrows()
    }

    rows = []
    for _, r in alloc_df.iterrows():
        sid = str(r["Student ID"]).strip()
        pid = str(r["assigned_project"]).strip()
        rank = int(r["assigned_rank"])

        out_of_list = int(pid not in pref_sets.get(sid, set()))

        note = ""
        if rank == num_prefs:
            note = "Worst (not in list)" if out_of_list else "Worst (in list)"

        rows.append({
            "Student ID": sid,
            "Assigned Project": pid,
            "Matched Preference Rank": rank,
            "OutOfList": out_of_list,
            "Rank Note": note,
        })

    pd.DataFrame(rows).to_csv(filename, index=False)

# Public API 

__all__ = [
    "load_dataset",
    "initialise_population",
    "repair_individual",
    "generate_run_id",
    "log_convergence",
    "save_allocation_csv",
]
