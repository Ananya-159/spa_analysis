# spa_analysis/scripts/plot_selector_hc_batch.py

"""
Batch-level plotting for Selector Hill Climbing experiments.
Generates aggregated plots (boxplots, violins, convergence curves, operator usage)
using `selectorhc_summary_log.csv`, with outputs saved in `/results/plots/selector_hill_climbing/`.
"""

import os, glob, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path

# Setup
ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "results" / "selector_hill_climbing_runs" / "selectorhc_summary_log.csv"
PLOT_DIR = ROOT / "results" / "plots" / "selector_hill_climbing"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Styling
sns.set(style="white")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

COLOR_BLUE   = "#1f77b4"
COLOR_GREEN  = "green"
COLOR_YELLOW = "EBC944"  
palette = {"Selector Hill Climbing": COLOR_BLUE}

# Load the Data (limit to 30 most recent runs using the actual timestamp format YYYY-MM-DDTHH-MM-SSZ)
df = pd.read_csv(SUMMARY_PATH)
if "timestamp" in df.columns:
    ts = pd.to_datetime(df["timestamp"], format="%Y-%m-%dT%H-%M-%SZ", errors="coerce")
    if ts.notna().any():
        df = (df.assign(timestamp=ts)
                .dropna(subset=["timestamp"])
                .sort_values("timestamp", ascending=False)
                .head(30)
                .reset_index(drop=True))
    else:
        print("[WARN] Timestamp parse failed; using full dataset.")
else:
    print("[WARN] 'timestamp' column not found; using full dataset.")

df["tag_label"] = "Selector Hill Climbing"  # algorithm label shown on x-axis / legend

# Clean, readable labels
LABELS = {
    "fitness_score":     "Fitness Score",
    "pref_penalty":      "Penalty",
    "avg_rank":          "Avg Pref Rank",
    "top1_pct":          "Top 1 Preference Match Rate (%)",
    "top3_pct":          "Top 3 Preference Match Rate (%)",
    "gini_satisfaction": "Gini Index",
    "total_violations":  "Total Violations",
    "runtime_sec":       "Runtime (s)"
}

# Boxplot with Median Line
def boxplot_with_median(y_col, title, ylabel, fname, integer_y=False, median_decimals=2):
    if y_col not in df.columns or df.empty:
        print(f"[SKIP] Missing column or empty data: {y_col}")
        return

    plt.figure(figsize=(7, 5))
    ax = sns.boxplot(data=df, x="tag_label", y=y_col, hue="tag_label",
                     palette=palette, legend=False)

    for i, tag in enumerate(df["tag_label"].unique()):
        vals = df.loc[df["tag_label"] == tag, y_col].dropna()
        if vals.empty:
            continue
        median = vals.median()
        ax.text(i, median, f"Median = {median:.{median_decimals}f}",
                ha="center", va="center", fontsize=9,
                color="white", fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.7))

    if integer_y:
        ax.yaxis.set_major_locator(mtick.MaxNLocator(integer=True))

    ax.set(title=title, xlabel="Algorithm", ylabel=ylabel)
    ax.grid(False); sns.despine()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{fname}.png", dpi=300)
    plt.close()

# Violin Plot for Fairness
def violin_fairness():
    y = "gini_satisfaction"
    if y not in df.columns or df.empty:
        print(f"[SKIP] Missing column or empty data: {y}")
        return

    plt.figure(figsize=(7, 5))
    ax = sns.violinplot(data=df, x="tag_label", y=y, hue="tag_label",
                        palette=palette, inner=None, legend=False)

    vals = df[y].dropna()
    if vals.empty:
        print("[SKIP] No values to plot for fairness.")
        plt.close(); return

    q1, q2, q3 = vals.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    lower = max(vals.min(), q1 - 1.5 * iqr)
    upper = min(vals.max(), q3 + 1.5 * iqr)

    ax.vlines(0, q1, q3, color="black", linewidth=5)
    ax.vlines(0, lower, upper, color="black", linewidth=1)
    ax.hlines(q2, -0.25, 0.25, color="black", linewidth=2)
    ax.text(0, q2, f"Median = {q2:.3f}", ha="center", va="center", fontsize=9,
            color="white", fontweight="bold",
            bbox=dict(facecolor="black", edgecolor="black", boxstyle="round,pad=0.2"))

    ax.set(title="Selector Hill Climbing — Fairness (Gini Index)",
           xlabel="Algorithm", ylabel=LABELS["gini_satisfaction"])
    ax.grid(False); sns.despine()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "selectorhc_fairness_gini.png", dpi=300)
    plt.close()

# Strip Plot for Top-1 / Top-3
def stripplot_summary(y_col, title, ylabel, fname):
    if y_col not in df.columns or df.empty:
        print(f"[SKIP] Missing column or empty data: {y_col}")
        return

    plt.figure(figsize=(6.5, 5))
    ax = sns.stripplot(data=df, x="tag_label", y=y_col, hue="tag_label",
                       palette=palette, jitter=0.1, size=5, legend=False)

    vals = df[y_col].dropna()
    if not vals.empty:
        mean = vals.mean()
        median = vals.median()
        ax.scatter(0, mean, color="red", marker="D", s=20)
        ax.scatter(0, median, color="black", marker="s", s=20)

        legend = [
            # NEW: clarify the blue dots
            Line2D([0], [0], marker='o', color='w', label='Runs',
                   markerfacecolor=COLOR_BLUE, markeredgecolor=COLOR_BLUE, markersize=6),
            Line2D([0], [0], marker='D', color='w', label=f"Mean = {mean:.2f}",
                   markerfacecolor='red', markersize=6),
            Line2D([0], [0], marker='s', color='w', label=f"Median = {median:.2f}",
                   markerfacecolor='black', markersize=6),
        ]
        ax.legend(handles=legend, loc='upper left', bbox_to_anchor=(1.02, 1),
                  fontsize=8, title="Summary")

    ax.set(title=title, xlabel="Algorithm", ylabel=ylabel)
    ax.grid(False); sns.despine()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{fname}.png", dpi=300)
    plt.close()


# Helper: get iterations series robustly
def _get_iterations_series():
    for cand in ["iterations_taken", "iterations", "iterations_"]:
        if cand in df.columns:
            s = pd.to_numeric(df[cand], errors="coerce").dropna()
            if not s.empty:
                return s
    return None

# Bar: Iterations
def bar_iterations():
    s = _get_iterations_series()
    if s is None:
        print("[SKIP] Missing or empty iterations column")
        return

    mean = s.mean()
    std = s.std()

    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    ax.bar("Selector Hill Climbing", mean, yerr=std, capsize=5, color=COLOR_BLUE)
    ax.text(0, mean + (std if np.isfinite(std) else 0) + 1, f"{int(round(mean))}", ha="center", fontsize=10)

    ax.yaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    ax.set(title="Selector Hill Climbing — Average Iterations to Convergence (± SD)",
           xlabel="Algorithm", ylabel="Iterations")
    ax.grid(False); sns.despine()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "selectorhc_iterations.png", dpi=300)
    plt.close()

# Bar: Constraint Violations (Grouped)
def bar_constraint_violations_grouped():
    cols = ["capacity_viol", "elig_viol", "under_cap"]
    label_map = {
        "capacity_viol": "Capacity Violation",
        "elig_viol": "Eligibility Violation",
        "under_cap": "Under Capacity"
    }

    if not all(col in df.columns for col in cols) or df.empty:
        print("[SKIP] Missing violation columns or empty data")
        return

    summary = df[cols].mean()

    x_label = "Selector Hill Climbing"
    values = summary.values
    labels = [label_map[c] for c in cols]
    colors = ["#4C72B0", "#55A868", "#EFC94C"]

    plt.figure(figsize=(8, 5.5))
    ax = plt.gca()

    x = np.arange(len(values))  # [0,1,2]
    bars = ax.bar(x, values, color=colors, edgecolor="black")

    # value labels on bars
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05, f"{h:.1f}",
                ha="center", va="bottom", fontsize=11)

    # X axis: show one centered tick label and bigger fonts 
    center = float(np.mean(x))  # 1.0
    ax.set_xticks([center])
    ax.set_xticklabels([x_label], fontsize=13)          # bigger tick label
    ax.set_xlabel("Algorithm", fontsize=14, labelpad=8) # x-axis title & size

    # Y axis & title font sizes
    ax.set_ylabel("Average Number of Violations", fontsize=14)
    ax.set_title("Selector Hill Climbing — Average Constraint Violations (Across 30 Most Recent Runs)",
                 fontsize=14)

    # Legend
    legend_patches = [Patch(color=c, label=l) for c, l in zip(colors, labels)]
    ax.legend(handles=legend_patches, title="Violation Type", loc='upper right', fontsize=11, title_fontsize=12)

    # Ticks styling
    ax.tick_params(axis='y', labelsize=12)

    ax.yaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    ax.set_axisbelow(True)
    ax.grid(False, axis='y', linestyle='--', alpha=0.3)
    sns.despine()

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "selectorhc_constraint_violations_grouped.png", dpi=300)
    plt.close()


# Boxplot: Total Violations
def boxplot_total_violations():
    boxplot_with_median(
        "total_violations",
        "Selector Hill Climbing — Total Constraint Violations (Lower = Better)",
        LABELS["total_violations"],
        "selectorhc_total_violations",
        integer_y=True,
        median_decimals=0
    )

# Scatter: Gini vs Violations
def scatter_fairness_vs_violations():
    if not {"gini_satisfaction", "total_violations"}.issubset(df.columns) or df.empty:
        print("[SKIP] Missing columns or empty data for fairness vs violations")
        return
    plt.figure(figsize=(6.5, 5))
    ax = sns.scatterplot(data=df, x="total_violations", y="gini_satisfaction",
                         color=COLOR_BLUE, s=70)
    ax.set(title="Selector Hill Climbing — Fairness vs Total Violations",
           xlabel=LABELS["total_violations"],
           ylabel=LABELS["gini_satisfaction"])
    ax.grid(False); sns.despine()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "selectorhc_fairness_vs_violations.png", dpi=300)
    plt.close()

# Histogram: Matched Preference Ranks
def histogram_assigned_ranks():
    files = sorted(
        glob.glob(str(ROOT / "results" / "selector_hill_climbing_runs" / "*_selectorhc_alloc_*.csv")),
        key=os.path.getmtime, reverse=True
    )
    if not files:
        return
    alloc_df = pd.read_csv(files[0])
    col = "Matched Preference Rank"
    if col not in alloc_df.columns:
        return

    ser = pd.to_numeric(alloc_df[col], errors="coerce").dropna().astype(int)
    if ser.empty:
        return

    plt.figure(figsize=(6.5, 5))
    bins = range(ser.min(), ser.max() + 2)
    ax = sns.histplot(ser, bins=bins, color=COLOR_BLUE, edgecolor="black")
    ax.yaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    ax.set(title="Selector Hill Climbing — Histogram of Assigned Preference Ranks",
           xlabel="Assigned Preference Rank", ylabel="Number of Students")
    ax.grid(False); sns.despine()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "selectorhc_histogram_assigned_ranks.png", dpi=300)
    plt.close()

# Load convergence curves (limit ing to 30 newest files)
def load_convergence_curves():
    files = sorted(
        glob.glob(str(ROOT / "results" / "selector_hill_climbing_runs" / "*_selectorhc_convergence_seed*.csv")),
        key=os.path.getmtime, reverse=True
    )[:30]
    runs = []
    for f in files:
        try:
            d = pd.read_csv(f)
            if {"capacity_viol", "elig_viol"}.issubset(d.columns):
                d["total_violations"] = (
                    d["capacity_viol"].astype(float) + d["elig_viol"].astype(float)
                )
            runs.append(d)
        except Exception:
            continue
    return runs

# Convergence: Mean ± Standard Deviation
def plot_convergence_mean_std(metric, title, ylabel, filename, integer_y=False):
    runs = load_convergence_curves()
    values = [d[metric].to_numpy() for d in runs if metric in d.columns]
    if not values:
        print(f"[SKIP] No data for '{metric}'")
        return

    min_len = min(len(v) for v in values)
    data = np.array([v[:min_len] for v in values])
    mean, std = data.mean(axis=0), data.std(axis=0)
    x = np.arange(min_len)

    plt.figure(figsize=(8, 5.5))
    ax = plt.gca()
    ax.plot(x, mean, color=COLOR_BLUE, lw=2.5, label="Mean")
    ax.fill_between(x, mean - std, mean + std, color=COLOR_BLUE, alpha=0.2, label="± Std Dev")
    if integer_y:
        ax.yaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    ax.set(title=title, xlabel="Iteration", ylabel=ylabel)
    ax.legend(); ax.grid(False); sns.despine()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=300)
    plt.close()

# Convergence: Sampled Runs
def plot_sampled_convergence(metric, title, ylabel, filename, n=3, integer_y=False):
    runs = load_convergence_curves()
    valid = [d for d in runs if metric in d.columns]
    if not valid:
        print(f"[SKIP] No data for '{metric}'")
        return
    sampled = random.sample(valid, min(n, len(valid)))
    styles = ["-", "--", ":"]

    plt.figure(figsize=(8, 5.5))
    ax = plt.gca()
    for i, d in enumerate(sampled):
        ax.plot(d["iter"], d[metric], linestyle=styles[i % 3],
                color=COLOR_BLUE, lw=2.2, label=f"Run {i+1}")
    if integer_y:
        ax.yaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    ax.set(title=title, xlabel="Iteration", ylabel=ylabel)
    ax.legend(title="Sampled Runs"); ax.grid(False); sns.despine()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=300)
    plt.close()

if __name__ == "__main__":
    print(" Generating all Selector Hill Climbing batch plots...")

    # Boxplots
    boxplot_with_median("fitness_score",
                        "Selector Hill Climbing — Fitness Score (Lower = Better)",
                        LABELS["fitness_score"], "selectorhc_total_score",
                        integer_y=False, median_decimals=2)

    boxplot_with_median("pref_penalty",
                        "Selector Hill Climbing — Penalty",
                        LABELS["pref_penalty"], "selectorhc_preference_penalty",
                        integer_y=True, median_decimals=0)

    # Use desired y-axis text directly:
    boxplot_with_median("avg_rank",
                        "Selector Hill Climbing — Average Assigned Preference Rank",
                        "Average Assigned Preference Rank",
                        "selectorhc_avg_assigned_rank",
                        integer_y=True, median_decimals=2)

    boxplot_with_median("runtime_sec",
                        "Selector Hill Climbing — Runtime per Run (s)",
                        LABELS["runtime_sec"], "selectorhc_runtime",
                        integer_y=False, median_decimals=2)

    boxplot_with_median("total_violations",
                        "Selector Hill Climbing — Total Constraint Violations",
                        LABELS["total_violations"], "selectorhc_total_violations",
                        integer_y=True, median_decimals=0)

    # Violin
    violin_fairness()

    # Stripplots
    stripplot_summary("top1_pct",
                      "Selector Hill Climbing — Top 1 Preference Match Rate (%)",
                      LABELS["top1_pct"], "selectorhc_top1_pct")

    stripplot_summary("top3_pct",
                      "Selector Hill Climbing — Top 3 Preference Match Rate (%)",
                      LABELS["top3_pct"], "selectorhc_top3_pct")

    # Bars / Scatter / Histogram
    bar_iterations()
    bar_constraint_violations_grouped()
    scatter_fairness_vs_violations()
    histogram_assigned_ranks()

    # Convergence (Mean ± Standard Deviation)
    plot_convergence_mean_std("fitness_score",
                              "Selector Hill Climbing — Fitness Score Convergence (Mean ± Standard Deviation)",
                              LABELS["fitness_score"], "selectorhc_convergence_score_avg.png")

    plot_convergence_mean_std("gini",
                              "Selector Hill Climbing — Fairness Convergence (Mean ± Standard Deviation)",
                              LABELS["gini_satisfaction"], "selectorhc_convergence_gini_avg.png")

    plot_convergence_mean_std("total_violations",
                              "Selector Hill Climbing — Violations Convergence (Mean ± Standard Deviation)",
                              LABELS["total_violations"], "selectorhc_convergence_violations_avg.png",
                              integer_y=True)

    # Convergence (Sampled)
    plot_sampled_convergence("fitness_score",
                             "Selector Hill Climbing — Fitness Score Convergence (Sampled Runs)",
                             LABELS["fitness_score"], "selectorhc_convergence_score_sampled.png")

    plot_sampled_convergence("gini",
                             "Selector Hill Climbing — Fairness Convergence (Sampled Runs)",
                             LABELS["gini_satisfaction"], "selectorhc_convergence_gini_sampled.png")

    plot_sampled_convergence("total_violations",
                             "Selector Hill Climbing — Violations Convergence (Sampled Runs)",
                             LABELS["total_violations"], "selectorhc_convergence_violations_sampled.png",
                             integer_y=True)

    print(f"\n All plots saved in: {PLOT_DIR}")

