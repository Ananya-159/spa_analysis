"""
ga_main.py — Main GA runner

Genetic Algorithm for Student–Project Allocation (SPA).

This module implements the Genetic Algorithm (GA) approach for solving SPA,
including crossover, mutation, and selection operators.
It produces allocation results and summary logs under `results/Genetic_algorithm_runs/`.

Summary and debug outputs are preserved.
"""



#Imports + Setup + Helpers
from __future__ import annotations
import argparse, hashlib, json, os, random, time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from deap import base, creator, tools

# Paths 
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

#  Project imports 
from core.fitness import evaluate_solution
from core.utils import list_to_alloc_df
from genetic_algorithm.crossover import one_point_crossover_with_repair
from genetic_algorithm.mutation import mutate_individual_with_repair
from genetic_algorithm.ga_utils import (
    load_dataset, initialise_population, repair_individual,
    log_convergence, save_allocation_csv, generate_run_id
)


#Helpers + DEAP Setup

def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def utc_ts_filename() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def hash_dataset(students_df, projects_df, supervisors_df) -> str:
    blob = (
        students_df.to_csv(index=False) +
        projects_df.to_csv(index=False) +
        supervisors_df.to_csv(index=False)
    ).encode("utf-8")
    return hashlib.md5(blob).hexdigest()[:8]

def append_row(results_dir: Path, row: Dict[str, object], filename: str = "ga_summary_log.csv") -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / filename
    df_new = pd.DataFrame([row])
    if not path.exists():
        df_new.to_csv(path, index=False)
    else:
        df_old = pd.read_csv(path)
        all_cols = list({*df_old.columns, *df_new.columns})
        df_old = df_old.reindex(columns=all_cols)
        df_new = df_new.reindex(columns=all_cols)
        pd.concat([df_old, df_new], ignore_index=True).to_csv(path, index=False)

def hamming_distance(a: List[str], b: List[str]) -> float:
    if not a or len(a) != len(b): return 0.0
    return sum(1 for x, y in zip(a, b) if x != y) / float(len(a))

def compute_diversity(pop: List[List[str]], best: List[str]) -> Tuple[int, float]:
    if not pop: return 0, 0.0
    uniq = len({tuple(ind) for ind in pop})
    avg_ham = float(np.mean([hamming_distance(ind, best) for ind in pop])) if len(pop) > 1 else 0.0
    return uniq, avg_ham

def log_debug_metrics(path: Path, rows: List[Dict]):
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)

#  DEAP setup (Spyder-safe)
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)

@dataclass
class GAParams:
    pop_size: int
    cx_prob: float
    mut_prob: float
    max_gens: int
    patience: int
    elitism: int

#run_ga_once()- Core GA 
def run_ga_once(tag: str, init_strategy: str,
                students_df, projects_df, supervisors_df,
                config: Dict, params: GAParams, seed: int,
                results_dir: Path, debug_mode: bool = False) -> Dict[str, object]:

    rng = random.Random(seed)
    np.random.seed(seed)
    start_t = time.perf_counter()
    cx_ops = mut_ops = repair_calls = 0
    num_prefs = int(config.get("num_preferences", 6))

    toolbox = base.Toolbox()
    toolbox.register("individual", initialise_population, init_strategy, students_df, projects_df, config)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def _evaluate(individual):
        alloc_df = list_to_alloc_df(individual, students_df, num_prefs, "assigned_project")
        total, breakdown = evaluate_solution(alloc_df, students_df, projects_df, supervisors_df, config, with_fairness=True)
        return total, breakdown

    toolbox.register("evaluate", _evaluate)

    def counted_repair(ind: List[str]) -> List[str]:
        nonlocal repair_calls
        repair_calls += 1
        return repair_individual(ind, students_df, projects_df, config)

    toolbox.register("mate", one_point_crossover_with_repair, repair_fn=counted_repair)
    toolbox.register("mutate", mutate_individual_with_repair, student_df=students_df, repair_fn=counted_repair)
    toolbox.register("select", tools.selRoulette)

    population = toolbox.population(n=params.pop_size)
    for ind in population:
        total, _ = toolbox.evaluate(ind)
        ind.fitness.values = (total,)

    best_ind = tools.selBest(population, 1)[0][:]
    best_total, best_br = toolbox.evaluate(best_ind)

    convergence = []
    debug_rows = []
    no_improve = 0

# GA Loop + Logging with fitness_score & total_violations
    for gen in range(1, params.max_gens + 1):
        cx_ops_gen = mut_ops_gen = 0
        repairs_before = repair_calls

        offspring = toolbox.select(population, len(population) - params.elitism)
        offspring = list(map(toolbox.clone, offspring))

        for i in range(1, len(offspring), 2):
            if rng.random() < params.cx_prob:
                toolbox.mate(offspring[i - 1], offspring[i])
                cx_ops_gen += 1
                for ind in [offspring[i - 1], offspring[i]]:
                    if hasattr(ind.fitness, "values"): del ind.fitness.values

        for ind in offspring:
            if rng.random() < params.mut_prob:
                toolbox.mutate(ind)
                mut_ops_gen += 1
                if hasattr(ind.fitness, "values"): del ind.fitness.values

        for ind in (x for x in offspring if not x.fitness.valid):
            total, _ = toolbox.evaluate(ind)
            ind.fitness.values = (total,)

        elites = tools.selBest(population, params.elitism)
        population = offspring + elites

        gen_best = tools.selBest(population, 1)[0]
        gen_total, gen_br = toolbox.evaluate(gen_best)

        if gen_total < best_total:
            best_ind = gen_best[:]
            best_total = gen_total
            best_br = gen_br
            no_improve = 0
        else:
            no_improve += 1

        avg_total = float(np.mean([ind.fitness.values[0] for ind in population]))
        repairs_gen = repair_calls - repairs_before

        convergence.append({
            "generation": gen,
            "fitness_score": float(gen_br["total"]),  # renamed
            "avg_total": float(avg_total),
            "pref_penalty": int(gen_br["pref_penalty"]),
            "capacity_viol": int(gen_br["capacity_viol"]),
            "elig_viol": int(gen_br["elig_viol"]),
            "under_cap": int(gen_br["under_cap"]),
            "total_violations": int(gen_br["capacity_viol"] + gen_br["elig_viol"] + gen_br["under_cap"]),
            "gini": float(gen_br.get("gini_satisfaction", 0.0)),
            "cx_ops_gen": int(cx_ops_gen),
            "mut_ops_gen": int(mut_ops_gen),
            "repair_calls_gen": int(repairs_gen),
        })

        if debug_mode:
            uniq, ham = compute_diversity(population, gen_best)
            debug_rows.append({"generation": gen, "unique_individuals": uniq, "avg_hamming_to_best": ham})

        cx_ops += cx_ops_gen
        mut_ops += mut_ops_gen
        print(f"-> Gen {gen}: fitness_score={gen_total:.1f}, unique={len(set(map(tuple, population)))}")

        if no_improve >= params.patience:
            break

#Summary Output + Main Block
    run_id = generate_run_id()
    d_hash = hash_dataset(students_df, projects_df, supervisors_df)
    conv_name = f"convergence_ga_{tag}_{run_id}.csv"
    log_convergence(convergence, str(results_dir / conv_name))

    if debug_mode and debug_rows:
        dbg_name = f"ga_debug_{tag}_{run_id}.csv"
        log_debug_metrics(results_dir / dbg_name, debug_rows)

    alloc_file = results_dir / f"{utc_ts_filename()}_ga_alloc_{run_id}.csv"
    save_allocation_csv(best_ind, students_df, str(alloc_file), config)

    alloc_df = list_to_alloc_df(best_ind, students_df, num_prefs, "assigned_project")
    ranks = alloc_df["assigned_rank"].to_numpy()
    avg_rank = float(np.mean(ranks))
    top1_pct = float((ranks == 1).sum() / len(ranks) * 100.0)
    top3_pct = float(((ranks >= 1) & (ranks <= 3)).sum() / len(ranks) * 100.0)
    total_violations = int(best_br["capacity_viol"] + best_br["elig_viol"] + best_br["under_cap"])
    runtime_sec = time.perf_counter() - start_t

    row = {
        "run_id": run_id, "timestamp": utc_timestamp(), "tag": tag, "init_strategy": init_strategy,
        "seed": seed, "dataset_hash": d_hash,
        "pop_size": params.pop_size, "cx_prob": params.cx_prob, "mut_prob": params.mut_prob,
        "elitism": params.elitism, "patience": params.patience,
        "pref_penalty": int(best_br["pref_penalty"]),
        "capacity_viol": int(best_br["capacity_viol"]),
        "elig_viol": int(best_br["elig_viol"]),
        "under_cap": int(best_br["under_cap"]),
        "fitness_score": float(best_br["total"]),
        "gini_satisfaction": float(best_br.get("gini_satisfaction", 0.0)),
        "iterations": len(convergence),
        "avg_rank": avg_rank, "top1_pct": top1_pct, "top3_pct": top3_pct,
        "total_violations": total_violations,
        "alloc_csv": alloc_file.name, "conv_csv": conv_name,
        "cx_ops": cx_ops, "mut_ops": mut_ops, "repair_calls": repair_calls,
        "repair_per_op": float(repair_calls / max(cx_ops + mut_ops, 1)),
        "runtime_sec": runtime_sec,
    }
    return row

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_final_ga.json")
    parser.add_argument("--one_run", action="store_true")
    parser.add_argument("--init", type=str, choices=["random", "greedy"], default="random")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log_diversity_batch", action="store_true")
    args = parser.parse_args()

    with open(ROOT / args.config, "r") as f:
        config = json.load(f)

    results_base = ROOT / config.get("paths", {}).get("results_dir", "results")
    results_base.mkdir(parents=True, exist_ok=True)
    results_dir = results_base / "Genetic_algorithm_runseed_tests" if args.one_run else results_base
    results_dir.mkdir(parents=True, exist_ok=True)

    students_df, projects_df, supervisors_df = load_dataset(config)
    params = GAParams(
        pop_size=int(config.get("ga_population", 120)),
        cx_prob=float(config.get("ga_crossover_prob", 0.8)),
        mut_prob=float(config.get("ga_mutation_prob", 0.3)),
        max_gens=int(config.get("ga_generations", 200)),
        patience=int(config.get("ga_patience", 50)),
        elitism=int(config.get("ga_elitism", 1)),
    )

    if args.one_run:
        row = run_ga_once("debug_test", args.init, students_df, projects_df, supervisors_df,
                          config, params, args.seed, results_dir, debug_mode=True)
        print(f"[TEST] run_id={row['run_id']} fitness={row['fitness_score']:.1f} gini={row['gini_satisfaction']:.3f}")
        return

    for s in range(101, 131):
        row = run_ga_once("ga_random", "random", students_df, projects_df, supervisors_df,
                          config, params, s, results_dir, debug_mode=args.log_diversity_batch)
        append_row(results_dir, row)
        print(f"[{row['tag']}] run_id={row['run_id']} fitness={row['fitness_score']:.1f} gini={row['gini_satisfaction']:.3f}")


# Quick toggle switch
RUN_MODE = "batch"  # or "debug"

if __name__ == "__main__":
    if RUN_MODE == "debug":
        import sys
        sys.argv = [
            "ga_main.py",
            "--config", "config_final_ga.json",
            "--one_run",
            "--init", "random",
            "--seed", "999",
        ]
    main()
