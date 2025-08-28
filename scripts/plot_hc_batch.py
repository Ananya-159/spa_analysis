# spa_analysis/scripts/plot_hc_batch.py
"""
Batch-level plotting for Hill Climbing (HC) experiments.
Generates aggregated plots (boxplots, histograms, convergence curves, fairness metrics)
using `hc_summary_log.csv`, with outputs saved in `/results/plots/Hill_climbing/`.
"""

import os
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Setup
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_CSV = os.path.join(ROOT, "results", "Hill_climbing_runs", "hc_summary_log.csv")
PLOT_DIR = os.path.join(ROOT, "results", "plots", "Hill_climbing")
os.makedirs(PLOT_DIR, exist_ok=True)

# Style
sns.set(style="white")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

# Load Data
df = pd.read_csv(RESULTS_CSV)
df["tag_label"] = df["tag"].str.replace("phase3_hc_", "Hill Climbing ").str.replace("_", " ").str.title()

palette = {
    "Hill Climbing Random": "#1f77b4",
    "Hill Climbing Greedy": "#2ca02c"
}

# Plotting Functions 

def boxplot_with_median(data, y, title, ylabel, fname):
    if y not in data.columns: return
    plt.figure(figsize=(7, 5))
    ax = sns.boxplot(data=data, x="tag_label", y=y, hue="tag_label", palette=palette, legend=False)
    for i, tag in enumerate(data["tag_label"].unique()):
        median = data[data["tag_label"] == tag][y].median()
        ax.text(i, median, f"Median = {median:.2f}", ha="center", va="center", fontsize=9,
                color="white", fontweight="bold", bbox=dict(facecolor="black", alpha=0.7))
    ax.set_title(f"Hill Climbing: {title}", loc="center")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Algorithm")
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{fname}.png"), dpi=300)
    plt.show()

def stripplot_with_summary_legend(y, title, ylabel, fname):
    if y not in df.columns: return
    plt.figure(figsize=(5.8, 5.2))
    ax = sns.stripplot(data=df, x="tag_label", y=y, hue="tag_label", palette=palette, legend=False,
                       jitter=0.1, dodge=False, size=5)

    means = df.groupby("tag_label")[y].mean().round(2)
    medians = df.groupby("tag_label")[y].median().round(2)

    marker_colors = {
        "Hill Climbing Random": {"mean": "red", "median": "red"},
        "Hill Climbing Greedy": {"mean": "black", "median": "black"}
    }

    handles = []
    for i, tag in enumerate(df["tag_label"].unique()):
        m1 = means[tag]
        m2 = medians[tag]
        c = marker_colors[tag]
        ax.scatter(i, m1, color=c["mean"], marker="D", s=15)
        ax.scatter(i, m2, color=c["median"], marker="s", s=15)
        handles.append(Line2D([0], [0], marker='D', color='w', label=f"Mean ({tag}) = {m1:.2f}%",
                              markerfacecolor=c["mean"], markersize=5))
        handles.append(Line2D([0], [0], marker='s', color='w', label=f"Median ({tag}) = {m2:.2f}%",
                              markerfacecolor=c["median"], markersize=5))

    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7,
              title="Summary", title_fontsize=8, frameon=False)

    ax.set_xticklabels(["Hill Climbing\nRandom", "Hill Climbing\nGreedy"])
    ax.set_title(f"Hill Climbing: {title}", loc="center")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Algorithm")
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{fname}.png"), dpi=300, bbox_inches="tight")
    plt.show()

def violin_fairness():
    y = "gini_satisfaction"
    if y not in df.columns: return
    plt.figure(figsize=(7, 5))
    ax = sns.violinplot(data=df, x="tag_label", y=y, palette=palette, inner=None)
    for i, tag in enumerate(df["tag_label"].unique()):
        vals = df[df["tag_label"] == tag][y]
        q1, q2, q3 = vals.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        lower = max(min(vals), q1 - 1.5 * iqr)
        upper = min(max(vals), q3 + 1.5 * iqr)
        ax.vlines(i, q1, q3, color="black", linewidth=5)
        ax.vlines(i, lower, upper, color="black", linewidth=1)
        ax.hlines(q2, i - 0.25, i + 0.25, color="black", linewidth=2)
        ax.text(i, q2 + 0.005, f"Median = {q2:.3f}", ha="center", va="bottom", fontsize=9,
                color="black", fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="black", alpha=0.85, boxstyle="round,pad=0.2"))
    ax.set_title("Hill Climbing: Fairness (Gini Index of Satisfaction)", loc="center")
    ax.set_ylabel("Gini Index (0 = Perfect Fairness, 1 = Max Inequality)")
    ax.set_xlabel("Algorithm")
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "fairness_gini.png"), dpi=300)
    plt.show()

def bar_iterations():
    y = "iterations_taken"
    if y not in df.columns: return
    plt.figure(figsize=(7, 5))
    means = df.groupby("tag_label")[y].mean()
    stds = df.groupby("tag_label")[y].std()
    ax = plt.gca()
    bars = ax.bar(means.index, means.values, yerr=stds, capsize=5,
                  color=[palette[k] for k in means.index])
    for bar, val, err in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, val + err + 20, f"{int(val)}", ha="center", fontsize=10)
    ax.set_title("Hill Climbing: Average Iterations to Convergence (± SD)", loc="center")
    ax.set_ylabel("Number of Iterations")
    ax.set_xlabel("Algorithm")
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "iterations_convergence.png"), dpi=300)
    plt.show()

def bar_constraint_violations():
    cols = ["capacity_viol", "elig_viol", "under_cap"]
    if not all(col in df.columns for col in cols): return
    summary = df.groupby("tag_label")[cols].mean().reset_index()
    melted = summary.melt(id_vars="tag_label", var_name="Violation Type", value_name="Average Count")

    legend_labels = {
        "capacity_viol": "Capacity Violation",
        "elig_viol": "Eligibility Violation",
        "under_cap": "Under-Capacity"
    }

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(data=melted, x="tag_label", y="Average Count", hue="Violation Type",
                     palette={"capacity_viol": "#1f77b4", "elig_viol": "#ffcc00", "under_cap": "#2ca02c"})
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [legend_labels.get(label, label) for label in labels]
    ax.legend(handles, new_labels, loc="upper left", bbox_to_anchor=(1, 1),
              fontsize=9, title="Violation Type", title_fontsize=10)

    ax.set_title("Hill Climbing: Breakdown of Constraint Violations", loc="center")
    ax.set_ylabel("Mean Number of Violations")
    ax.set_xlabel("Algorithm")
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "constraint_violations.png"), dpi=300, bbox_inches="tight")
    plt.show()

def boxplot_total_violations():
    boxplot_with_median(df, "total_violations",
                        "Total Constraint Violations (Lower = Better)",
                        "Total Violations", "total_violations")

def scatter_fairness_vs_violations():
    plt.figure(figsize=(6.5, 5))
    ax = sns.scatterplot(data=df, x="total_violations", y="gini_satisfaction",
                         hue="tag_label", palette=palette, s=70)
    ax.set_title("Hill Climbing: Fairness vs Total Violations", loc="center")
    ax.set_xlabel("Total Violations")
    ax.set_ylabel("Gini Index (0 = Fair)")
    ax.legend(title="Algorithm")
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "fairness_vs_violations.png"), dpi=300)
    plt.show()

def histogram_assigned_ranks():
    latest_allocs = sorted(glob.glob(os.path.join(ROOT, "results", "Hill_climbing_runs", "*_hc_alloc_*.csv")), reverse=True)
    if not latest_allocs: return
    alloc_df = pd.read_csv(latest_allocs[0])
    if "Matched Preference Rank" not in alloc_df.columns: return
    plt.figure(figsize=(6.5, 5))
    ax = sns.histplot(alloc_df["Matched Preference Rank"],
                      bins=range(1, alloc_df["Matched Preference Rank"].max() + 2),
                      color="#1f77b4", edgecolor="black")
    ax.set_title("Hill Climbing: Histogram of Assigned Preference Ranks", loc="center")
    ax.set_xlabel("Assigned Preference Rank")
    ax.set_ylabel("Number of Students")
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "histogram_assigned_ranks.png"), dpi=300)
    plt.show()

#  Convergence 

def load_and_group_convergence_data():
    paths = sorted(glob.glob(os.path.join(ROOT, "results", "Hill_climbing_runs", "convergence_hc_*.csv")))
    grouped = {"Hill Climbing Random": [], "Hill Climbing Greedy": []}
    for path in paths:
        df_curve = pd.read_csv(path)
        tag = os.path.basename(path)
        if "_random_" in tag:
            grouped["Hill Climbing Random"].append(df_curve)
        elif "_greedy_" in tag:
            grouped["Hill Climbing Greedy"].append(df_curve)
    return grouped

def plot_metric_vs_iteration(metric_col, title, ylabel, filename, color_map):
    grouped_data = load_and_group_convergence_data()
    plt.figure(figsize=(8, 5.5))
    line_handles, patch_handles = [], []
    for label, runs in grouped_data.items():
        if not runs: continue
        df_all = pd.concat(runs).groupby("iter")[metric_col].agg(['mean', 'std'])
        x = df_all.index
        y = df_all["mean"]
        y_std = df_all["std"]
        plt.plot(x, y, color=color_map[label], linewidth=2.5)
        line_handles.append(Line2D([0], [0], color=color_map[label], lw=2.5, label=f"{label} (Mean)"))
        plt.fill_between(x, y - y_std, y + y_std, alpha=0.2, color=color_map[label])
        patch_handles.append(Patch(facecolor=color_map[label], alpha=0.2, label=f"{label} (± Std Dev)"))
    plt.title(f"Hill Climbing: {title}", loc="center")
    plt.xlabel("Iterations")
    plt.ylabel(ylabel)
    plt.legend(handles=line_handles + patch_handles, title="Legend", fontsize=10)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300)
    plt.show()

def plot_few_sample_runs(metric_col, title, ylabel, filename, color_map, n_samples=3):
    grouped_data = load_and_group_convergence_data()
    plt.figure(figsize=(8, 5.5))
    handles = []
    for label, runs in grouped_data.items():
        if not runs: continue
        sampled = random.sample(runs, min(n_samples, len(runs)))
        linestyles = ["-", "--", ":"]
        for i, df_run in enumerate(sampled):
            ls = linestyles[i % len(linestyles)]
            label_text = label if i == 0 else None
            plt.plot(df_run["iter"], df_run[metric_col],
                     linewidth=2.2, linestyle=ls,
                     color=color_map[label], label=label_text)
        handles.append(Line2D([0], [0], color=color_map[label], lw=2.2, label=label))
    plt.title(f"Hill Climbing: {title}", loc="center")
    plt.xlabel("Iterations")
    plt.ylabel(ylabel)
    plt.legend(handles=handles, title="Sampled Runs", fontsize=9)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300)
    plt.show()

#  Master Run

if __name__ == "__main__":
    print("Generating all plots...")

    # Evaluation
    boxplot_with_median(df, "total", "Fitness Score (Lower = Better)", "Fitness Score", "fitness_score")
    boxplot_with_median(df, "pref_penalty", "Preference Penalty (Sum of Ranks)", "Total Sum of Preference Ranks", "preference_penalty")
    boxplot_with_median(df, "avg_rank", "Average Assigned Preference Rank (Lower = Better)", "Average Assigned Preference Rank", "avg_assigned_rank")
    boxplot_with_median(df, "runtime_sec", "Runtime per Run (Seconds)", "Runtime (Seconds)", "runtime_sec")

    stripplot_with_summary_legend("top1_pct", "Top-1 Preference Match Rate (Percentage)", "Percentage of Students", "top1_pct")
    stripplot_with_summary_legend("top3_pct", "Top-3 Preference Match Rate (Percentage)", "Percentage of Students", "top3_pct")

    violin_fairness()
    bar_iterations()
    bar_constraint_violations()
    boxplot_total_violations()
    scatter_fairness_vs_violations()
    histogram_assigned_ranks()

    # Convergence
    plot_metric_vs_iteration("best_total", "Convergence of Fitness Score (Mean ± Std Dev)", "Best Fitness Score", "convergence_score_curve_avg.png", palette)
    plot_metric_vs_iteration("gini", "Convergence of Fairness (Mean ± Std Dev)", "Gini Index", "convergence_fairness_curve_avg.png", palette)
    plot_metric_vs_iteration("total_violations", "Convergence of Violations (Mean ± Std Dev)", "Total Constraint Violations", "convergence_violations_curve_avg.png", palette)

    plot_few_sample_runs("best_total", "Sampled Convergence: Fitness Score", "Best Fitness Score", "sampled_convergence_score.png", palette)
    plot_few_sample_runs("gini", "Sampled Convergence: Fairness", "Gini Index", "sampled_convergence_fairness.png", palette)
    plot_few_sample_runs("total_violations", "Sampled Convergence: Violations", "Total Constraint Violations", "sampled_convergence_violations.png", palette)

    print(f"\nAll plots saved in: {PLOT_DIR}")
