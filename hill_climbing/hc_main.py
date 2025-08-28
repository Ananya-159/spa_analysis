"""
Hill Climbing for Student-Project Allocation (SPA)

This script:
  -Loads config + dataset
  -Builds RANDOM and GREEDY initial solutions
  -Runs Hill Climbing (first-improvement using operators)
  -Tracks operator usage and repair attempts
  -Calculates satisfaction metrics (Top-1, Top-3 %, avg rank)
  -Logs results to results/hc_summary_log.csv
  -Logs convergence per iteration to results/convergence_hc_<tag>_<runid>.csv
  -Saves per-student allocation CSV after each run


"""

from __future__ import annotations
import os, sys, json, uuid, random, hashlib, csv, time
from typing import List, Dict, Tuple
import pandas as pd


# Path bootstrap

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

#  Imports

from hill_climbing.operators import reassign_student, swap_students, greedy_repair, random_operator
from fairness.metrics import gini, ranks_to_satisfaction
from core.fitness import evaluate_solution
from core.utils import list_to_alloc_df, utc_timestamp_filename, utc_timestamp


# Global Settings

ITERS = 2000
PATIENCE = 400
REPAIR_EVERY = 0

#  Utilities

def load_config(root: str) -> Dict:
    with open(os.path.join(root, "config.json"), "r", encoding="utf-8") as f:
        return json.load(f)

def load_dataset(xlsx: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    students = pd.read_excel(xlsx, sheet_name="Students")
    projects = pd.read_excel(xlsx, sheet_name="Projects")
    supervisors = pd.read_excel(xlsx, sheet_name="Supervisors")
    return students, projects, supervisors

def extract_preferences(students_df: pd.DataFrame, num_prefs: int) -> List[List[str]]:
    cols = [f"Preference {i}" for i in range(1, num_prefs + 1)]
    return students_df[cols].values.tolist()

def _bool_from_excel(series: pd.Series) -> List[bool]:
    return (
        series.astype(str).str.strip().str.lower()
        .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
        .fillna(False).tolist()
    )

def greedy_initial(preferences, students_df, projects_df, supervisors_df, rng) -> List[str]:
    proj_max = projects_df.set_index("Project ID")["Max Students"].to_dict()
    proj_type = projects_df.set_index("Project ID")["Project Type"].to_dict()
    proj_min = projects_df.set_index("Project ID")["Minimum Average Required"].to_dict()
    proj_sup = projects_df.set_index("Project ID")["Supervisor ID"].to_dict()
    sup_max = supervisors_df.set_index("Supervisor ID")["Max Student Capacity"].to_dict()
    proj_load = {pid: 0 for pid in proj_max}
    sup_load = {sid: 0 for sid in sup_max}
    avg = pd.to_numeric(students_df["Average"], errors="coerce").fillna(-1e9).tolist()
    eligible = _bool_from_excel(students_df["Client Based Eligibility"])
    order = list(range(len(preferences)))
    rng.shuffle(order)
    alloc = ["" for _ in preferences]

    for s in order:
        for pid in preferences[s]:
            if str(proj_type[pid]).strip().lower().startswith("client") and not eligible[s]: continue
            if avg[s] < float(proj_min[pid]): continue
            if proj_load[pid] >= int(proj_max[pid]): continue
            sup = proj_sup[pid]
            if sup_load[sup] >= int(sup_max[sup]): continue
            alloc[s] = pid
            proj_load[pid] += 1
            sup_load[sup] += 1
            break
        if not alloc[s] and preferences[s]:
            alloc[s] = preferences[s][0]
    return alloc

def random_initial(preferences: List[List[str]], rng: random.Random) -> List[str]:
    return [rng.choice(p) if p else "" for p in preferences]

def hash_dataset(students_df, projects_df, supervisors_df) -> str:
    blob = (
        students_df.to_csv(index=False) +
        projects_df.to_csv(index=False) +
        supervisors_df.to_csv(index=False)
    ).encode("utf-8")
    return hashlib.md5(blob).hexdigest()[:8]

def save_convergence_csv(path: str, rows: List[Dict]):
    if not rows: return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iter", "best_total", "gini", "total_violations"])
        writer.writeheader()
        writer.writerows(rows)


#  Main Hill climbing function

def hill_climb(
    start_sol, students_df, projects_df, supervisors_df, config,
    num_prefs, iters, patience, seed, repair_every, convergence_path
):
    rng = random.Random(seed)
    prefs_cache = extract_preferences(students_df, num_prefs)

    alloc_df = list_to_alloc_df(start_sol, students_df, num_prefs, project_col_name="assigned_project")
    cur_total, cur_break = evaluate_solution(alloc_df, students_df, projects_df, supervisors_df, config)
    best_total, best_break = cur_total, cur_break
    current = start_sol.copy()
    no_improve = 0
    operator_log = []
    repairs_used = 0
    convergence_data = []

    for t in range(1, iters + 1):
        op = random_operator()
        operator_log.append(op.__name__)
        cand, _ = (op(current, prefs_cache, rng=rng) if op is reassign_student else op(current, rng=rng))

        if repair_every and t % repair_every == 0:
            def fit(sol): return evaluate_solution(list_to_alloc_df(sol, students_df, num_prefs), students_df, projects_df, supervisors_df, config)[0]
            cand, _ = greedy_repair(cand, prefs_cache, fitness_fn=fit, attempts=30, rng=rng)
            repairs_used += 1

        cand_df = list_to_alloc_df(cand, students_df, num_prefs, project_col_name="assigned_project")
        cand_total, cand_break = evaluate_solution(cand_df, students_df, projects_df, supervisors_df, config)

        if cand_total < cur_total:
            current, cur_total, cur_break = cand, cand_total, cand_break
            if cand_total < best_total:
                best_total, best_break = cand_total, cand_break
            no_improve = 0
        else:
            no_improve += 1

        # Tracking the convergence
        sat = ranks_to_satisfaction(cand_df["assigned_rank"], num_prefs)
        g = float(max(0.0, min(1.0, gini(sat))))
        viol = sum(float(cand_break.get(k, 0.0)) for k in ["capacity_viol", "elig_viol", "under_cap"])
        convergence_data.append({"iter": t, "best_total": best_total, "gini": g, "total_violations": viol})

        if no_improve >= patience:
            break

    save_convergence_csv(convergence_path, convergence_data)
    return current, best_break | {"total": float(best_total)}, t, operator_log, repairs_used


# Logging the functions

def append_row(results_dir: str, row: Dict[str, object]):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "hc_summary_log.csv")
    df_new = pd.DataFrame([row])
    if not os.path.exists(path):
        df_new.to_csv(path, index=False)
    else:
        df_old = pd.read_csv(path)
        all_columns = list(set(df_old.columns) | set(df_new.columns))
        df_old = df_old.reindex(columns=all_columns)
        df_new = df_new.reindex(columns=all_columns)
        pd.concat([df_old, df_new], ignore_index=True).to_csv(path, index=False)

def save_per_student_csv(alloc_df, results_dir, run_id="", num_prefs=None) -> str:
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{utc_timestamp_filename()}_hc_alloc{('_' + run_id) if run_id else ''}.csv")
    cols = ["Student ID", "assigned_project", "assigned_rank"]
    rename = {"assigned_project": "Assigned Project", "assigned_rank": "Matched Preference Rank"}
    if "_rank_oob_violation" in alloc_df.columns:
        export_df = alloc_df[cols + ["_rank_oob_violation"]].rename(columns={**rename, "_rank_oob_violation": "OutOfList"})
    else:
        export_df = alloc_df[cols].rename(columns=rename)
        export_df["OutOfList"] = 0
    N = num_prefs if num_prefs else int(export_df["Matched Preference Rank"].max())
    export_df["Rank Note"] = export_df.apply(
        lambda row: "Worst (not in list)" if (row["Matched Preference Rank"] == N and row["OutOfList"] == 1)
        else ("Worst (in list)" if row["Matched Preference Rank"] == N else ""), axis=1
    )
    export_df.to_csv(out_path, index=False)
    return out_path


# Running the experiment

def run_once(tag, init_sol, students_df, projects_df, supervisors_df, config, num_prefs, iters, patience, seed, results_dir):
    start_time = time.time()
    run_id = uuid.uuid4().hex[:8]
    convergence_path = os.path.join(results_dir, f"convergence_hc_{tag}_{run_id}.csv")

    final_sol, br, iterations, ops, repairs = hill_climb(
        init_sol, students_df, projects_df, supervisors_df, config, num_prefs,
        iters, patience, seed, repair_every=REPAIR_EVERY, convergence_path=convergence_path
    )

    elapsed_sec = time.time() - start_time
    alloc_df = list_to_alloc_df(final_sol, students_df, num_prefs, project_col_name="assigned_project")
    assigned_ranks = alloc_df["assigned_rank"].to_numpy()
    sat = ranks_to_satisfaction(assigned_ranks, num_prefs)
    g = float(max(0.0, min(1.0, gini(sat))))
    avg_rank = float(assigned_ranks.mean())
    top1_pct = float((assigned_ranks == 1).sum() / len(assigned_ranks)) * 100
    top3_pct = float(((assigned_ranks >= 1) & (assigned_ranks <= 3)).sum() / len(assigned_ranks)) * 100
    total_violations = sum(float(br.get(k, 0.0)) for k in ["capacity_viol", "elig_viol", "under_cap"])
    d_hash = hash_dataset(students_df, projects_df, supervisors_df)
    out_csv = save_per_student_csv(alloc_df, results_dir, run_id=run_id, num_prefs=num_prefs)

    row = {
        "timestamp": utc_timestamp(),
        "run_id": run_id,
        "tag": tag,
        "pref_penalty": float(br.get("pref_penalty", 0.0)),
        "capacity_viol": float(br.get("capacity_viol", 0.0)),
        "elig_viol": float(br.get("elig_viol", 0.0)),
        "under_cap": float(br.get("under_cap", 0.0)),
        "total": float(br.get("total", 0.0)),
        "gini_satisfaction": g,
        "iterations_taken": iterations,
        "repair_used": repairs,
        "operator_seq": ";".join(ops[:20]) + ("..." if len(ops) > 20 else ""),
        "dataset_hash": d_hash,
        "avg_rank": avg_rank,
        "top1_pct": top1_pct,
        "top3_pct": top3_pct,
        "total_violations": total_violations,
        "runtime_sec": round(elapsed_sec, 2),
    }

    append_row(results_dir, row)
    print(f"\n[{tag}] RunID={run_id} total={row['total']:.1f}  gini={g:.3f}  top1={top1_pct:.1f}%  time={elapsed_sec:.2f}s")
    print(f" -> Saved allocation: {os.path.relpath(out_csv, ROOT)}")
    print(f" -> Saved convergence log: {os.path.relpath(convergence_path, ROOT)}")


#  Main driver

def main():
    cfg = load_config(ROOT)
    data_excel = os.path.join(ROOT, cfg["paths"]["data_excel"])
    results_dir = os.path.join(ROOT, cfg["paths"]["results_dir"])
    num_prefs = int(cfg.get("num_preferences", 6))
    students_df, projects_df, supervisors_df = load_dataset(data_excel)
    preferences = extract_preferences(students_df, num_prefs)

    print("Running 30 Hill Climbing runs (15 random + 15 greedy)")
    for i, seed in enumerate(range(101, 131), start=1):
        rng = random.Random(seed)
        init_random = random_initial(preferences, rng)
        run_once("phase3_hc_random", init_random, students_df, projects_df, supervisors_df, cfg, num_prefs, ITERS, PATIENCE, seed, results_dir)

        init_greedy = greedy_initial(preferences, students_df, projects_df, supervisors_df, rng)
        run_once("phase3_hc_greedy", init_greedy, students_df, projects_df, supervisors_df, cfg, num_prefs, ITERS, PATIENCE, seed, results_dir)

if __name__ == "__main__":
    main()
