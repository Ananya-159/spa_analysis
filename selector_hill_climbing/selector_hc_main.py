# spa_analysis/selector_hill_climbing/selector_hc_main.py

"""
Main runner for the Selector Hill Climbing (Selector-HC) algorithm in Student Project Allocation (SPA).

This module coordinates the adaptive hill climbing search:
- Loads dataset and configuration.
- Applies low-level heuristics (swap, reassign, mutation-micro) from hill_climbing.
- Uses ChoiceFunction (operator selection) and Acceptance (solution acceptance).
- Logs convergence, saves final allocations, and records a summary of the run.

Outputs are written to `results/selector_hill_climbing_runs/` as allocation CSVs,
convergence logs, and summary tables.
"""

#  Force import resolution from the root of spa_analysis 
import sys
import os
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parents[1]  # .../spa_analysis
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

#  Standard imports 
import json
import random
import pandas as pd
import time
import hashlib
from datetime import datetime, timezone
from collections import Counter

# Project-specific modules
from core.utils import list_to_alloc_df
from core.fitness import evaluate_solution
from hill_climbing.operators import op_swap, op_reassign, op_mutation_micro

#  Robust import for controller 
# Tries package import first; if it fails (e.g., Spyder CWD issues), loads from local file.
try:
    from selector_hill_climbing.controller import ChoiceFunction, ChoiceConfig, Acceptance, AcceptConfig
except ModuleNotFoundError:
    import importlib.util

    # Try controller.py
    ctrl_candidates = [
        current_file.parent / "controller.py",
        current_file.parent / "selector_controller.py",
    ]
    loaded = False
    for cand in ctrl_candidates:
        if cand.exists():
            spec = importlib.util.spec_from_file_location("selector_hc_controller_fallback", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            ChoiceFunction = mod.ChoiceFunction
            ChoiceConfig = mod.ChoiceConfig
            Acceptance = mod.Acceptance
            AcceptConfig = mod.AcceptConfig
            loaded = True
            break
    if not loaded:
        raise

#  Helpers 
def tsz():
    """Timestamp-safe string for filenames (with Z for UTC)."""
    ts = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")
    return ts.replace(":", "-") + "Z"

def hash_dataset(path):
    """Dataset hashing (for reproducibility trace)."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:8]

# Main Selector-HC Runner 
def run_selector_hc(
    config_path="config_final_ga.json",
    seed=123,
    iters=1000,
    accept="better",              # "better" or "sa"
    epsilon=0.1,
    softmax_tau=None,
    results_dir="results/selector_hill_climbing_runs/"
):
    start_time = time.perf_counter()
    rng = random.Random(seed)

    # Loading the config
    config_path = project_root / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    cfg = json.load(open(config_path, "r", encoding="utf-8"))

    # Loading the dataset
    excel_path = project_root / cfg["paths"]["data_excel"]
    print("Checking Excel path:", excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    print("Excel file found!")

    dataset_id = hash_dataset(excel_path)
    students = pd.read_excel(excel_path, sheet_name="Students")
    projects = pd.read_excel(excel_path, sheet_name="Projects")
    supervisors = pd.read_excel(excel_path, sheet_name="Supervisors")
    num_prefs = int(cfg["num_preferences"])

    # Extracting the preferences
    prefs = [[row[f"Preference {i}"] for i in range(1, num_prefs + 1)] for _, row in students.iterrows()]
    sol = [rng.choice(p) if p else "" for p in prefs]

    # Fitness evaluator
    def fitness_of(s):
        alloc_df = list_to_alloc_df(s, students, num_prefs)
        expected = {"assigned_project", "assigned_rank", "_rank_oob_violation"}
        missing = expected - set(alloc_df.columns)
        if missing:
            raise ValueError(f"alloc_df is missing required columns: {missing}")
        total, breakdown = evaluate_solution(alloc_df, students, projects, supervisors, cfg, with_fairness=True)
        alloc_df = alloc_df.rename(columns={
            "assigned_project": "Assigned Project",
            "assigned_rank": "Matched Preference Rank",
            "_rank_oob_violation": "OutOfList"
        })
        alloc_df["Rank Note"] = alloc_df["OutOfList"].apply(lambda x: "Worst (in list)" if x else "0")
        return total, breakdown, alloc_df

    # Initial solution evaluation
    fitness_best, breakdown_best, alloc_best = fitness_of(sol)
    sol_best = sol.copy()

    # Low-level heuristics (LLHs)
    ops = {
        "swap": lambda s: op_swap(s, rng),
        "reassign": lambda s: op_reassign(s, prefs, rng),
        "mut_micro": lambda s: op_mutation_micro(s, prefs, rng),
    }

    choice = ChoiceFunction(list(ops.keys()), ChoiceConfig(epsilon=epsilon, window=50, softmax_tau=softmax_tau))
    acceptor = Acceptance(AcceptConfig(scheme=accept, T0=1.0, decay=0.997))

    out_dir = project_root / results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    converg_path = out_dir / f"{tsz()}_selectorhc_convergence_seed{seed}.csv"
    conv_rows = []
    op_counter = Counter()

    # Main Loop
    for t in range(1, iters + 1):
        op_name = choice.choose(rng)
        op_counter[op_name] += 1

        candidate = ops[op_name](sol.copy())
        fitness_old = fitness_of(sol)[0]
        fitness_new, breakdown_new, alloc_new = fitness_of(candidate)
        delta = fitness_new - fitness_old

        choice.update(op_name, -delta)
        if acceptor.accept(delta, rng):
            sol = candidate
            if fitness_new < fitness_best:
                fitness_best, breakdown_best, sol_best, alloc_best = fitness_new, breakdown_new, candidate, alloc_new

        cap_viol = breakdown_best.get("capacity_viol", 0) or 0
        elig_viol = breakdown_best.get("elig_viol", 0) or 0
        total_violations = cap_viol + elig_viol

        conv_rows.append({
            "iter": t,
            "op": op_name,
            "fitness_score": fitness_best,
            "pref_penalty": breakdown_best.get("pref_penalty", None),
            "capacity_viol": cap_viol,
            "elig_viol": elig_viol,
            "under_cap": breakdown_best.get("under_cap", None),
            "total_violations": total_violations,
            "gini": breakdown_best.get("gini_satisfaction", None),
        })

    pd.DataFrame(conv_rows).to_csv(converg_path, index=False)

    run_id = f"selectorhc_{seed}"
    alloc_path = out_dir / f"{tsz()}_selectorhc_alloc_{run_id}.csv"
    alloc_best.to_csv(alloc_path, index=False)

    final_total_viol = breakdown_best.get("capacity_viol", 0) + breakdown_best.get("elig_viol", 0)
    runtime_sec = round(time.perf_counter() - start_time, 4)
    operator_summary = json.dumps(dict(op_counter.most_common()))

    summary_row = {
        "timestamp": tsz(),
        "run_id": run_id,
        "tag": "selector_hc",
        "avg_rank": float(alloc_best["Matched Preference Rank"].mean()),
        "top1_pct": float((alloc_best["Matched Preference Rank"] == 1).mean() * 100),
        "top3_pct": float((alloc_best["Matched Preference Rank"] <= 3).mean() * 100),
        "gini_satisfaction": breakdown_best.get("gini_satisfaction", None),
        "pref_penalty": breakdown_best.get("pref_penalty", None),
        "capacity_viol": breakdown_best.get("capacity_viol", None),
        "elig_viol": breakdown_best.get("elig_viol", None),
        "under_cap": breakdown_best.get("under_cap", None),
        "total_violations": final_total_viol,
        "fitness_score": float(fitness_best),
        "runtime_sec": runtime_sec,
        "iterations_taken": iters,
        "operator_seq": operator_summary,
        "dataset_hash": dataset_id,
    }

    sum_path = out_dir / "selectorhc_summary_log.csv"
    if sum_path.exists():
        existing = pd.read_csv(sum_path)
        pd.concat([existing, pd.DataFrame([summary_row])], ignore_index=True).to_csv(sum_path, index=False)
    else:
        pd.DataFrame([summary_row]).to_csv(sum_path, index=False)

# Running as script
if __name__ == "__main__":
    run_selector_hc()
    print("Selector Hill Climbing run completed successfully and results saved.")
