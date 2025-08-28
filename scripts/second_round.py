# spa_analysis/scripts/second_round.py
"""
Second-Round Allocation (HC-only).

Applies a swap-only Hill Climbing reallocation step to improve feasibility 
and fairness when no new project capacity is available. Generates updated 
allocations, logs, and summary statistics.Used in pipeline with second_round_driver.py.

"""

from __future__ import annotations

import sys
import re
from pathlib import Path
import json
from datetime import datetime, timezone
from time import perf_counter
import random
import argparse
import numpy as np
import pandas as pd

# Make local packages importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

# Project imports
from core.utils import list_to_alloc_df
from core.fitness import evaluate_solution
from hill_climbing.hc_main import hash_dataset

# Plotting colors
GA_BLUE = "#1f77b4"
GA_GREEN = "green"
GA_YELLOW = "yellow"


# CLI


def parse_args():
    p = argparse.ArgumentParser(description="Second-round allocation (HC-only).")
    default_alloc = ROOT / "results" / "Hill_climbing_runs" / "2025-08-16T02-18-30Z_hc_alloc_594d10fb.csv"
    
    p.add_argument("--alloc_csv", type=str, default=str(default_alloc),
                   help="Path to the base HC allocation CSV to post-process")

    #  All logs, summary, stats will be saved here
    p.add_argument("--runs_dir", type=str, default=str(ROOT / "results" / "Second_round_hill_climbing"),
                   help="Directory to save allocation + detailed logs")

    #  Plots will be saved here
    p.add_argument("--plots_dir", type=str, default=str(ROOT / "results" / "plots" / "Second_round_hill_climbing"),
                   help="Directory to save plots")

    # Summary + stats also go into Second_round_hill_climbing now
    p.add_argument("--summary_dir", type=str, default=str(ROOT / "results" / "Second_round_hill_climbing"),
                   help="Directory to append HC-compatible summary + stats rows")
    
    return p.parse_args()

# Parse args and prepare directories 
args = parse_args()
original_alloc_path = Path(args.alloc_csv)
runs_dir    = Path(args.runs_dir)
plots_dir   = Path(args.plots_dir)
summary_dir = Path(args.summary_dir)

runs_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)
summary_dir.mkdir(parents=True, exist_ok=True)

# Utilities

def ts_utc_filename() -> str:
    """UTC timestamp string safe for filenames: YYYY-MM-DDTHH-MM-SSZ."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

def prefs_matrix_from_students(students_df: pd.DataFrame, num_prefs: int) -> list[list[str]]:
    cols = [f"Preference {i}" for i in range(1, num_prefs + 1)]
    return students_df[cols].astype(str).values.tolist()

def build_allocation_list_from_export(students_df: pd.DataFrame,
                                      export_alloc_df: pd.DataFrame,
                                      project_col: str) -> list[str]:
    merged = students_df[["Student ID"]].merge(
        export_alloc_df[["Student ID", project_col]], on="Student ID", how="left"
    )
    return merged[project_col].fillna("").astype(str).tolist()

def rank_stats(series_or_array) -> tuple[float, float, float]:
    """Return (avg_rank, top1_pct, top3_pct)."""
    r = pd.to_numeric(pd.Series(series_or_array), errors="coerce").dropna()
    if len(r) == 0:
        return (float("nan"), float("nan"), float("nan"))
    avg = float(r.mean())
    top1 = float((r == 1).mean() * 100.0)
    top3 = float(((r >= 1) & (r <= 3)).mean() * 100.0)
    return avg, top1, top3

def safe_mpl_import():
    try:
        import matplotlib.pyplot as plt  # noqa
        return True
    except Exception:
        print("[WARN] matplotlib not available; skipping plots.")
        return False

def _add_bar_labels(ax, rects):
    """Integer labels on top of bars (used in rank distribution)."""
    for rect in rects:
        h = rect.get_height()
        if np.isnan(h):
            continue
        ax.annotate(f"{int(h)}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

def _infer_source_hc_run_id(alloc_csv_path: Path) -> str:
    """Parse run id from filename like ..._hc_alloc_<runid>.csv."""
    m = re.search(r"hc_alloc_([A-Za-z0-9]+)\.csv$", alloc_csv_path.name)
    return m.group(1) if m else ""

source_hc_run_id = _infer_source_hc_run_id(original_alloc_path)

def matched_or_ci(b, c, eps=0.5):
    """
    Compute matched odds ratio and 95% CI (log scale) from McNemar table.
    Adds continuity correction if b or c is 0.
    """
    if b == 0 or c == 0:
        b += eps
        c += eps
    or_ = b / c
    se = np.sqrt(1.0 / b + 1.0 / c)
    lo = np.exp(np.log(or_) - 1.96 * se)
    hi = np.exp(np.log(or_) + 1.96 * se)
    return float(or_), float(lo), float(hi)



# Loading config + Excel data

config_path = ROOT / "config.json"
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

excel_path = ROOT / config["paths"]["data_excel"]
students_df = pd.read_excel(excel_path, sheet_name="Students")
projects_df = pd.read_excel(excel_path, sheet_name="Projects")
supervisors_df = pd.read_excel(excel_path, sheet_name="Supervisors")
num_prefs = int(config["num_preferences"])

# Base allocation to post-process
alloc_df = pd.read_csv(original_alloc_path)
PROJECT_COL = "Assigned Project"
RANK_COL = "Matched Preference Rank"

# Identifying students for reallocation
needs_mask = alloc_df[PROJECT_COL].isna() | (alloc_df[RANK_COL] > 4)
students_to_reassign = alloc_df.loc[needs_mask].copy()
print(f"[INFO] Students needing reallocation: {len(students_to_reassign)}")

# Second-round-eligible projects with free capacity
project_caps: dict[str, int] = config.get("project_capacities", {})
assigned_counts = alloc_df[PROJECT_COL].value_counts().to_dict()
sr_flag = projects_df.set_index("Project ID")["Second Round Eligible"]
available_projects = {
    proj: int(project_caps[proj]) - int(assigned_counts.get(proj, 0))
    for proj in project_caps
    if (project_caps[proj] - assigned_counts.get(proj, 0) > 0)
    and str(sr_flag.get(proj, "FALSE")).strip().upper() == "TRUE"
}
print(f"[INFO] Second-round eligible projects with free capacity: {len(available_projects)}")



# Mode A: Lightweight HC for subset

def run_lightweight_subset_hc(
    subset_student_ids, students_df, projects_df, supervisors_df,
    config, available_projects, seed=99, iters=100
):
    rng = random.Random(seed)
    subset = students_df[students_df["Student ID"].isin(subset_student_ids)]
    prefs = prefs_matrix_from_students(subset, num_prefs)

    init = [
        (rng.choice([p for p in pref if p in available_projects])
         if any(p in available_projects for p in pref) else "")
        for pref in prefs
    ]
    best = init[:]

    def score(alloc_list: list[str]) -> float:
        df = list_to_alloc_df(alloc_list, subset, num_prefs)
        val, _ = evaluate_solution(df, subset, projects_df, supervisors_df, config, with_fairness=False)
        return float(val)

    best_val = score(best)
    for _ in range(iters):
        cand = best[:]
        if rng.random() < 0.5 and len(cand) >= 2:
            i, j = rng.randrange(len(cand)), rng.randrange(len(cand))
            cand[i], cand[j] = cand[j], cand[i]
        else:
            i = rng.randrange(len(cand))
            allowed = [p for p in prefs[i] if p in available_projects]
            if allowed:
                cand[i] = rng.choice(allowed)
        cand_val = score(cand)
        if cand_val < best_val:
            best, best_val = cand, cand_val

    return list_to_alloc_df(best, subset, num_prefs)


# Mode B: Full-cohort swap-only HC

def run_swap_only_hc_on_full(
    full_alloc_list, focus_indices, students_df, projects_df,
    supervisors_df, config, seed=99, iters=800
):
    rng = random.Random(seed)

    def score_of(alloc_list: list[str]) -> float:
        df = list_to_alloc_df(alloc_list, students_df, num_prefs)
        val, _ = evaluate_solution(df, students_df, projects_df, supervisors_df, config, with_fairness=False)
        return float(val)

    best = full_alloc_list[:]
    best_score = score_of(best)
    N = len(best)

    for _ in range(iters):
        i = rng.choice(focus_indices)
        j = rng.randrange(N)
        if i == j:
            continue
        cand = best[:]
        cand[i], cand[j] = cand[j], cand[i]
        cand_score = score_of(cand)
        if cand_score < best_score:
            best, best_score = cand, cand_score

    return best


# Execute Mode A (if capacity) or Mode B (swap-only)

algo_start = perf_counter()

if len(available_projects) > 0:
    mode_label = "modeA_subset"
    print("[INFO] Running Mode A: lightweight HC with available capacity.")
    subset_ids = students_to_reassign["Student ID"].tolist()
    second_alloc_df = run_lightweight_subset_hc(
        subset_student_ids=subset_ids,
        students_df=students_df,
        projects_df=projects_df,
        supervisors_df=supervisors_df,
        config=config,
        available_projects=available_projects,
        seed=99,
        iters=100,
    )
    final_alloc_df = alloc_df.copy()
    for _, row in second_alloc_df.iterrows():
        sid = row["Student ID"]
        final_alloc_df.loc[final_alloc_df["Student ID"] == sid, PROJECT_COL] = row["assigned_project"]
        final_alloc_df.loc[final_alloc_df["Student ID"] == sid, RANK_COL] = row["assigned_rank"]

else:
    mode_label = "modeB_swap_only"
    print("[INFO] Running Mode B: swap-only HC (no available capacity).")
    full_alloc_list = build_allocation_list_from_export(students_df, alloc_df, PROJECT_COL)
    needs_ids = set(students_to_reassign["Student ID"].tolist())
    focus_indices = [i for i, sid in enumerate(students_df["Student ID"].tolist()) if sid in needs_ids]

    improved_alloc_list = run_swap_only_hc_on_full(
        full_alloc_list=full_alloc_list,
        focus_indices=focus_indices,
        students_df=students_df,
        projects_df=projects_df,
        supervisors_df=supervisors_df,
        config=config,
        seed=99,
        iters=800,
    )

    improved_alloc_df = list_to_alloc_df(improved_alloc_list, students_df, num_prefs)
    final_alloc_df = alloc_df.copy()
    join_df = improved_alloc_df.rename(columns={
        "assigned_project": "assigned_project_tmp",
        "assigned_rank": "assigned_rank_tmp",
    })[["Student ID", "assigned_project_tmp", "assigned_rank_tmp"]]

    final_alloc_df = final_alloc_df.merge(join_df, on="Student ID", how="left")
    final_alloc_df[PROJECT_COL] = final_alloc_df["assigned_project_tmp"].fillna(final_alloc_df[PROJECT_COL])
    final_alloc_df[RANK_COL] = final_alloc_df["assigned_rank_tmp"].fillna(final_alloc_df[RANK_COL]).astype(int)
    final_alloc_df.drop(columns=["assigned_project_tmp", "assigned_rank_tmp"], inplace=True)

runtime_sec = perf_counter() - algo_start

def plot_second_round_effects(
    alloc_before: pd.DataFrame,
    alloc_after: pd.DataFrame,
    needs_mask: pd.Series,
    rank_col: str,
    plots_dir: Path,
    tag: str
):
    """
    Generates 4 updated plots:
    1) Pie: Impact (Improved / Unchanged / Worsened)
    2) Bar: Top-1 / Top-3 Satisfaction After Round 2
    3) Bar: Rank Distribution Before vs After
    4) Bar: Avg Assigned Rank (Cohort vs Affected)
    """
    if not safe_mpl_import():
        return
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)

    b = alloc_before.loc[needs_mask, ["Student ID", rank_col]].rename(columns={rank_col: "rank_before"})
    a = alloc_after.loc[needs_mask, ["Student ID", rank_col]].rename(columns={rank_col: "rank_after"})
    m = b.merge(a, on="Student ID", how="inner")
    n_affected = len(m)

    improved = int((m["rank_after"] < m["rank_before"]).sum())
    worsened = int((m["rank_after"] > m["rank_before"]).sum())
    unchanged = int((m["rank_after"] == m["rank_before"]).sum())

    avg_before_c, top1_before_c, top3_before_c = rank_stats(alloc_before[rank_col])
    avg_after_c,  top1_after_c,  top3_after_c  = rank_stats(alloc_after[rank_col])
    avg_before_a, top1_before_a, top3_before_a = rank_stats(m["rank_before"])
    avg_after_a,  top1_after_a,  top3_after_a  = rank_stats(m["rank_after"])

    # STYLE CLEANER
    def clean_plot(ax):
        for position, spine in ax.spines.items():
            if position in {"top", "right"}:
                spine.set_visible(False)
            else:
                spine.set_linewidth(1)
        ax.tick_params(axis='both', which='both', length=3)
   
    # 1) PIE CHART — IMPACT
    labels = [f"{label} ({count})" for label, count in zip(
      ["Improved", "Unchanged", "Worsened"], [improved, unchanged, worsened])]

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    wedges, texts, autotexts = ax.pie(
      [improved, unchanged, worsened],
      labels=labels,
      autopct="%1.0f%%",
      colors=[GA_GREEN, GA_YELLOW, GA_BLUE],
      startangle=140,
      textprops={"fontsize": 11}
  )

    # Contrast-adjusted text colors for slices
    for i, autotext in enumerate(autotexts):
      autotext.set_color("white" if wedges[i].get_facecolor()[0] < 0.5 else "black")

    # Better-centered sample size
    ax.text(0, -0.1, f"n = {n_affected}", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.set_title("Preference Rank Changes After Second-Round Allocation\n(Hill Climbing)", fontsize=13)

    clean_plot(ax)
    fig.text(0.5, -0.04, "Only includes students affected by second-round reallocation",
           ha='center', va='center', fontsize=9, color='gray')
    plt.tight_layout()
    plt.savefig(plots_dir / f"{tag}_impact_pie.png", dpi=150)
    plt.close(fig)


    # 2) BAR — TOP-1 / TOP-3 Satisfaction
    labels = ["Top-1 Preference", "Top-3 Preferences"]
    after_vals = [top1_after_a, top3_after_a]
    x = np.arange(len(labels))
    width = 0.5

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    bars = ax.bar(x, after_vals, width, color=GA_BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Percentage of Reassigned Students (%)", fontsize=11)
    ax.set_xlabel("Preference Match", fontsize=11)
    ax.set_title("Top 1 and Top 3 Preference Match After Reallocation\n(Reassigned Students — Hill Climbing)", fontsize=13)

    # Add McNemar OR annotation (safe hyphens)
    or1, lo1, hi1 = matched_or_ci(mc1["b_0to1"], mc1["c_1to0"])
    or3, lo3, hi3 = matched_or_ci(mc3["b_0to1"], mc3["c_1to0"])
    text = (
        f"Top-1: OR = {or1:.2f} (95% CI {lo1:.2f}–{hi1:.2f})\n"
        f"Top-3: OR = {or3:.2f} (95% CI {lo3:.2f}–{hi3:.2f})"
    )
    ax.text(0.02, 0.95, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.8))

    # Optional: bar labels (matplotlib >= 3.4)
    try:
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    except Exception:
        pass

    clean_plot(ax)
    fig.text(0.5, -0.04, "Only includes students affected by second-round reallocation",
             ha='center', va='center', fontsize=9, color='gray')
    plt.tight_layout()
    plt.savefig(plots_dir / f"{tag}_top1_top3_after_only.png", dpi=150)
    plt.close(fig)

    # 3) RANK DISTRIBUTION — BEFORE vs AFTER
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    width = 0.45
    max_rank = int(max((m["rank_before"].max(), m["rank_after"].max())))
    ranks = np.arange(1, max_rank + 1)
    counts_before = [(m["rank_before"] == r).sum() for r in ranks]
    counts_after  = [(m["rank_after"]  == r).sum() for r in ranks]

    bars_b = ax.bar(ranks - width / 2, counts_before, width, color=GA_YELLOW, label="Before second round")
    bars_a = ax.bar(ranks + width / 2, counts_after,  width, color=GA_BLUE, label="After second round")

    _add_bar_labels(ax, bars_b)
    _add_bar_labels(ax, bars_a)

    ax.set_xlabel("Assigned Preference Rank (1 = most preferred)", fontsize=11)
    ax.set_ylabel("Number of Reassigned Students", fontsize=11)
    ax.set_title("Assigned Preference Ranks Before vs After Reallocation\n(Affected Students — Hill Climbing)", fontsize=13)
    ax.set_xticks(ranks)
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 1.02), fontsize=11)

    clean_plot(ax)
    fig.text(0.5, -0.04, "Only includes students affected by second-round reallocation",
             ha='center', va='center', fontsize=9, color='gray')
    plt.tight_layout()
    plt.savefig(plots_dir / f"{tag}_rank_dist_affected_before_after.png", dpi=150)
    plt.close(fig)

    # 4) AVERAGE PREFERENCE RANK — Cohort vs Affected
    groups = ["All students (cohort)", "Reassigned students"]
    before_avg = [avg_before_c, avg_before_a]
    after_avg  = [avg_after_c,  avg_after_a]
    x = np.arange(len(groups)) * 2
    width = 0.45

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.bar(x - width / 2, before_avg, width, color=GA_YELLOW, label="Before second round")
    ax.bar(x + width / 2, after_avg,  width, color=GA_BLUE, label="After second round")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylabel("Average Assigned Preference Rank\n(lower is better)", fontsize=11)
    ax.set_xlabel("Student Group", fontsize=11)
    ax.set_title("Change in Average Assigned Preference Rank\nCohort vs Reassigned Students — Hill Climbing", fontsize=13)
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 1.02), fontsize=11)

    # Wilcoxon annotation (safe hyphens)
    # Plot 4: Average Assigned Preference Rank — Cohort vs Affected
    text = (
    f"Wilcoxon: W = {wil['wilcoxon_stat']:.1f}, p = {wil['wilcoxon_p']:.4g}\n"
    f"Δmedian = {wil['median_delta']:+.2f}, r = {wil['effect_size_r']:.3f}"
    )
    ax.text(0.02, 0.80, text, transform=ax.transAxes, ha="left", va="top",
        fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.8))

    clean_plot(ax)
    fig.text(0.5, -0.04, "Only includes students affected by second-round reallocation",
             ha='center', va='center', fontsize=9, color='gray')
    plt.tight_layout()
    plt.savefig(plots_dir / f"{tag}_avg_rank_before_after.png", dpi=150)
    plt.close(fig)

    print("[PLOT] All updated second-round impact plots saved.")



# Evaluation + Metrics

canonical_after = list_to_alloc_df(
    allocation=build_allocation_list_from_export(students_df, final_alloc_df, PROJECT_COL),
    students_df=students_df,
    num_prefs=num_prefs,
)

fitness_total, breakdown = evaluate_solution(
    alloc_df=canonical_after,
    students_df=students_df,
    projects_df=projects_df,
    supervisors_df=supervisors_df,
    config=config,
    with_fairness=True,
)

# Stats: before/after (cohort)
avg_before, top1_before, top3_before = rank_stats(alloc_df[RANK_COL])
avg_after,  top1_after,  top3_after  = rank_stats(final_alloc_df[RANK_COL])

# Stats: affected students (paired)
before_focus = alloc_df.loc[needs_mask, ["Student ID", RANK_COL]].rename(columns={RANK_COL: "rank_before"})
after_focus  = final_alloc_df.loc[needs_mask, ["Student ID", RANK_COL]].rename(columns={RANK_COL: "rank_after"})
compare_focus = before_focus.merge(after_focus, on="Student ID", how="inner")

improved = int((compare_focus["rank_after"] < compare_focus["rank_before"]).sum())
worsened = int((compare_focus["rank_after"] > compare_focus["rank_before"]).sum())
unchanged = int((compare_focus["rank_after"] == compare_focus["rank_before"]).sum())
num_students_reassigned = int(needs_mask.sum())

# IDs / timestamps / breakdowns
ts = ts_utc_filename()  # filename-friendly UTC with Z
ts_iso_z = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")  # ISO with 'Z'
d_hash = hash_dataset(students_df, projects_df, supervisors_df)
iterations_taken = 100 if mode_label == "modeA_subset" else 800
run_id = f"sr_{mode_label}_{ts}"
tag = "second_round"
operator_seq = mode_label
total_violations = float(breakdown.get("capacity_viol", 0.0)) + float(breakdown.get("elig_viol", 0.0)) + float(breakdown.get("under_cap", 0.0))

# Clarify fitness naming
breakdown = dict(breakdown)
breakdown["fitness_total"] = fitness_total


# Saving the Allocation

alloc_out = runs_dir / f"{ts}_second_round_alloc.csv"
final_alloc_df.to_csv(alloc_out, index=False)
print(f"[SAVE] Allocation saved: {alloc_out.name}")


# Detailed Log (no alias fields; Z timestamp)

detail_breakdown = {k: v for k, v in breakdown.items() if k != "total"}
detailed_log_path = runs_dir / f"{ts}_second_round_log.csv"
detailed_row = {
    **detail_breakdown,  # includes fitness_total; NOT 'total'
    "gini_satisfaction": breakdown.get("gini_satisfaction", None),

    "timestamp": ts_iso_z,          # ISO-8601 with Z
    "mode": mode_label,
    "dataset_hash": d_hash,

    "num_students_reassigned": num_students_reassigned,
    "num_available_projects": int(len(available_projects)),

    # Explicit cohort stats
    "avg_rank_before": avg_before,
    "avg_rank_after":  avg_after,
    "top1_before": top1_before,
    "top1_after":  top1_after,
    "top3_before": top3_before,
    "top3_after":  top3_after,

    # Affected deltas
    "improved_count": improved,
    "worsened_count": worsened,
    "unchanged_count": unchanged,

    # Run metadata
    "run_id": run_id,
    "tag": tag,
    "operator_seq": operator_seq,
    "iterations_taken": iterations_taken,
    "runtime_sec": runtime_sec,
    "total_violations": total_violations,
    "repair_used": 0,

    # Traceability for driver joins
    "source_hc_run_id": source_hc_run_id,
}
pd.DataFrame([detailed_row]).to_csv(detailed_log_path, index=False)
print(f"[SAVE] Detailed log saved: {detailed_log_path.name}")


# HC-Compatible Summary Log 

summary_row = {
    "timestamp": ts,  # filename-style UTC with Z
    "run_id": run_id,
    "tag": tag,

    # Final aliases for HC comparability
    "avg_rank": avg_after,
    "top1_pct": top1_after,
    "top3_pct": top3_after,

    "gini_satisfaction": breakdown.get("gini_satisfaction", None),
    "pref_penalty": float(breakdown.get("pref_penalty", 0.0)),
    "capacity_viol": float(breakdown.get("capacity_viol", 0.0)),
    "elig_viol": float(breakdown.get("elig_viol", 0.0)),
    "under_cap": float(breakdown.get("under_cap", 0.0)),

 
    "total_violations": float(total_violations),
    "runtime_sec": runtime_sec,
    "iterations_taken": iterations_taken,
    "repair_used": 0,
    "operator_seq": operator_seq,
    "dataset_hash": d_hash,
    "proj_underfill": float(breakdown.get("proj_underfill", 0.0)),
    "sup_underfill": float(breakdown.get("sup_underfill", 0.0)),
    "fitness_total": float(fitness_total),
    "num_students_reassigned": num_students_reassigned,

    # Source HC run id for across-runs aggregation
    "source_hc_run_id": source_hc_run_id,
}

summary_path = summary_dir / "second_round_summary_log.csv"
if summary_path.exists():
    prev = pd.read_csv(summary_path)
    # Purge legacy 'total' column if it exists to remove redundancy
    if "total" in prev.columns:
        prev = prev.drop(columns=["total"])
    out = pd.concat([prev, pd.DataFrame([summary_row])], ignore_index=True)
else:
    out = pd.DataFrame([summary_row])

out.to_csv(summary_path, index=False)
print(f"[SAVE] HC-compatible summary appended to: {summary_path.name}")



# Paired Stats: Wilcoxon, McNemar, ΔGini (affected students)

from scipy.stats import wilcoxon, norm, chi2
from fairness.metrics import ranks_to_satisfaction, gini

def wilcoxon_rank_change(before_ranks, after_ranks):
    before = np.asarray(before_ranks)
    after = np.asarray(after_ranks)
    mask = ~np.isnan(before) & ~np.isnan(after)
    before = before[mask]
    after = after[mask]
    n = len(before)
    if n == 0:
        return {"wilcoxon_stat": np.nan, "wilcoxon_p": np.nan,
                "median_delta": np.nan, "effect_size_r": np.nan, "n": 0}
    res = wilcoxon(before, after)
    median_delta = float(np.median(after - before))
    z = norm.isf(res.pvalue / 2.0)
    r = abs(z) / np.sqrt(n) if not np.isnan(z) else np.nan
    return {"wilcoxon_stat": float(res.statistic), "wilcoxon_p": float(res.pvalue),
            "median_delta": median_delta, "effect_size_r": r, "n": n}

def mcnemar_from_pairs(before_bin, after_bin):
    b = int(((before_bin == 0) & (after_bin == 1)).sum())
    c = int(((before_bin == 1) & (after_bin == 0)).sum())
    if (b + c) == 0:
        return {"mcnemar_chi2": np.nan, "mcnemar_p": np.nan, "b_0to1": b, "c_1to0": c}
    chi2_cc = ((abs(b - c) - 1.0) ** 2.0) / (b + c)
    p = chi2.sf(chi2_cc, df=1)
    return {"mcnemar_chi2": chi2_cc, "mcnemar_p": p, "b_0to1": b, "c_1to0": c}

def bootstrap_delta_gini(before_ranks, after_ranks, num_prefs, B=2000, seed=99):
    rng = np.random.default_rng(seed)
    before = np.asarray(before_ranks); after = np.asarray(after_ranks)
    mask = ~np.isnan(before) & ~np.isnan(after)
    before = before[mask]; after = after[mask]
    n = len(before)
    if n == 0:
        return {"delta_gini_median": np.nan, "delta_gini_ci_low": np.nan, "delta_gini_ci_high": np.nan, "B": 0}
    deltas = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        sb = ranks_to_satisfaction(before[idx], num_prefs)
        sa = ranks_to_satisfaction(after[idx], num_prefs)
        deltas.append(gini(sa) - gini(sb))
    deltas = np.array(deltas)
    return {
        "delta_gini_median": float(np.median(deltas)),
        "delta_gini_ci_low": float(np.percentile(deltas, 2.5)),
        "delta_gini_ci_high": float(np.percentile(deltas, 97.5)),
        "B": B,
    }

# Computing the stats
rank_b = compare_focus["rank_before"].to_numpy(dtype=float)
rank_a = compare_focus["rank_after"].to_numpy(dtype=float)
wil = wilcoxon_rank_change(rank_b, rank_a)

top1_b = (rank_b == 1).astype(int)
top1_a = (rank_a == 1).astype(int)
top3_b = ((rank_b >= 1) & (rank_b <= 3)).astype(int)
top3_a = ((rank_a >= 1) & (rank_a <= 3)).astype(int)

def _pct(x): return float(np.nanmean(x) * 100.0)
mc1 = mcnemar_from_pairs(top1_b, top1_a)
mc3 = mcnemar_from_pairs(top3_b, top3_a)
top1_delta_pp = _pct(top1_a) - _pct(top1_b)
top3_delta_pp = _pct(top3_a) - _pct(top3_b)
gini_boot = bootstrap_delta_gini(rank_b, rank_a, num_prefs=num_prefs)

# Console summary
print("\n[STATS] Second round (affected students)")
print(f"[STATS] n = {len(rank_b)}")
print(f"[STATS] Wilcoxon: W = {wil['wilcoxon_stat']:.3f}, p = {wil['wilcoxon_p']:.4g}, "
      f"median Δrank = {wil['median_delta']:+.2f}, r = {wil['effect_size_r']:.3f}")
print(f"[STATS] McNemar Top‑1: χ² = {mc1['mcnemar_chi2']:.3f}, p = {mc1['mcnemar_p']:.4g}, Δpp = {top1_delta_pp:+.1f}")
print(f"[STATS] McNemar Top‑3: χ² = {mc3['mcnemar_chi2']:.3f}, p = {mc3['mcnemar_p']:.4g}, Δpp = {top3_delta_pp:+.1f}")
print(f"[STATS] ΔGini: median = {gini_boot['delta_gini_median']:+.3f} "
      f"(95% CI {gini_boot['delta_gini_ci_low']:+.3f} to {gini_boot['delta_gini_ci_high']:+.3f})")

# Save paired stats
stats_out = summary_dir / "second_round_stats.csv"
stats_row = {
    "timestamp": ts,  # filename-style timestamp; works for across-runs
    "run_id": run_id,
    "mode": mode_label,
    "dataset_hash": d_hash,
    "num_students_reassigned": num_students_reassigned,
    "wilcoxon_stat": wil["wilcoxon_stat"],
    "wilcoxon_p": wil["wilcoxon_p"],
    "median_delta_rank": wil["median_delta"],
    "effect_size_r": wil["effect_size_r"],
    "wilcoxon_n": wil["n"],
    "mcnemar_top1_chi2": mc1["mcnemar_chi2"],
    "mcnemar_top1_p": mc1["mcnemar_p"],
    "top1_delta_pp": top1_delta_pp,
    "b_0to1_top1": mc1["b_0to1"],
    "c_1to0_top1": mc1["c_1to0"],
    "mcnemar_top3_chi2": mc3["mcnemar_chi2"],
    "mcnemar_top3_p": mc3["mcnemar_p"],
    "top3_delta_pp": top3_delta_pp,
    "b_0to1_top3": mc3["b_0to1"],
    "c_1to0_top3": mc3["c_1to0"],
    "delta_gini_median": gini_boot["delta_gini_median"],
    "delta_gini_ci_low": gini_boot["delta_gini_ci_low"],
    "delta_gini_ci_high": gini_boot["delta_gini_ci_high"],
    "bootstrap_B": gini_boot["B"],
    "source_hc_run_id": source_hc_run_id,  # join key for driver
}
if stats_out.exists():
    pd.concat([pd.read_csv(stats_out), pd.DataFrame([stats_row])], ignore_index=True).to_csv(stats_out, index=False)
else:
    pd.DataFrame([stats_row]).to_csv(stats_out, index=False)
print(f"[SAVE] Paired stats (Option A) appended to: {stats_out.name}")

# Ensure matched OR + CI are computed for Markdown
or1, lo1, hi1 = matched_or_ci(mc1["b_0to1"], mc1["c_1to0"])
or3, lo3, hi3 = matched_or_ci(mc3["b_0to1"], mc3["c_1to0"])

# Plots

plot_second_round_effects(
    alloc_before=alloc_df,
    alloc_after=final_alloc_df,
    needs_mask=needs_mask,
    rank_col=RANK_COL,
    plots_dir=plots_dir,
    tag=("modeA" if len(available_projects) > 0 else "modeB")
)

markdown_path = summary_dir / f"{ts}_second_round_stats.md"
with open(markdown_path, "w", encoding="utf-8") as f:
    f.write(f"""# Second-Round Statistical Summary

**n = {wil['n']} (reassigned students)**

### Wilcoxon Signed-Rank Test
- W = {wil['wilcoxon_stat']:.1f}, p = {wil['wilcoxon_p']:.4g}
- Median Δrank = {wil['median_delta']:+.2f}, r = {wil['effect_size_r']:.3f}

### McNemar's Test
- **Top‑1:** χ² = {mc1['mcnemar_chi2']:.2f}, p = {mc1['mcnemar_p']:.4g}
  - OR = {or1:.2f} (95% CI {lo1:.2f}, {hi1:.2f})
- **Top‑3:** χ² = {mc3['mcnemar_chi2']:.2f}, p = {mc3['mcnemar_p']:.4g}
  - OR = {or3:.2f} (95% CI {lo3:.2f}, {hi3:.2f})

### ΔGini (Fairness)
- Median = {gini_boot['delta_gini_median']:+.3f}
- 95% CI: ({gini_boot['delta_gini_ci_low']:+.3f}, {gini_boot['delta_gini_ci_high']:+.3f})
""")

print(f"[SAVE] Markdown summary written: {markdown_path.name}")
