"""
plot_ga_batch.py

Generates 18 plots for GA batch results:
- Box, violin, strip, bar (integer y-axis for counts), scatter, histogram (integer y-axis)
- Convergence curves: mean ± std, and sampled
- Grid-free; GA blue (#1f77b4); annotated medians
"""

# Setup, styling, file discovery, summary loader
import os, glob, random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick  # integer y-axis ticks for count metrics
from pathlib import Path
from matplotlib.lines import Line2D

# Paths (support both locations depending on config) 
ROOT = Path(__file__).resolve().parent.parent
GA_BASE_DIRS = [
    ROOT / "results" / "Genetic_algorithm_runs",    # typical GA folder
    ROOT / "results" / "Hill_climbing_runs",        # if GA configured to save here
]

PLOT_DIR = ROOT / "results" / "plots" / "Genetic_algorithm"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Styling 
sns.set(style="white")
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10
})
GA_COLOR  = "#1f77b4"  # HC-style blue
GA_GREEN  = "green"
GA_YELLOW = "yellow"
palette   = {"Genetic Algorithm Random": GA_COLOR}  # single GA flavor

# Helpers 
def _first_existing(*paths: Path) -> Path | None:
    """Return first existing path; else None."""
    for p in paths:
        if p.exists():
            return p
    return None

def find_summary_csv() -> Path:
    """Find ga_summary_log.csv under either GA base directory."""
    candidates = [d / "ga_summary_log.csv" for d in GA_BASE_DIRS]
    path = _first_existing(*candidates)
    if not path:
        raise FileNotFoundError("ga_summary_log.csv not found in expected results folders.")
    return path

def load_summary() -> pd.DataFrame:
    """Load summary CSV; harmonize columns (fitness_score alias for legacy total)."""
    df = pd.read_csv(find_summary_csv())
    df["tag_label"] = "Genetic Algorithm Random"  # single GA label
    if "fitness_score" not in df.columns and "total" in df.columns:
        df = df.rename(columns={"total": "fitness_score"})
    return df

# Loading the summary once for all summary-based plots
df = load_summary()


#Box/Violin/Strip/Scatter (with optional integer y-axis for ranks)
def boxplot_with_median(
    y_col: str, title: str, ylabel: str, fname: str,
    *, integer_y: bool = False, median_decimals: int = 2
):
    """
    Single-group boxplot with annotated median.
    - integer_y: force integer y-ticks (for rank/violation/iteration counts)
    - median_decimals: decimals in median label (0 for counts, 2 for continuous)
    """
    if y_col not in df.columns:
        print(f"[SKIP] Missing column: {y_col}")
        return

    plt.figure(figsize=(7, 5))
    ax = sns.boxplot(data=df, x="tag_label", y=y_col, hue="tag_label",
                     palette=palette, legend=False)

    # median label
    for i, tag in enumerate(df["tag_label"].unique()):
        med = df.loc[df["tag_label"] == tag, y_col].median()
        ax.text(i, med, f"Median = {med:.{median_decimals}f}",
                ha="center", va="center", fontsize=9,
                color="white", fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.7))

    if integer_y:
        ax.yaxis.set_major_locator(mtick.MaxNLocator(integer=True))

    ax.set(title=title, xlabel="Algorithm", ylabel=ylabel)
    ax.grid(False); sns.despine()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{fname}.png", dpi=300)
    plt.show()


def violin_fairness():
    """Violin for Gini with IQR/whiskers and median label."""
    y = "gini_satisfaction"
    if y not in df.columns:
        print(f"[SKIP] Missing column: {y}")
        return
    plt.figure(figsize=(7, 5))
    ax = sns.violinplot(data=df, x="tag_label", y=y, hue="tag_label",
                        palette=palette, inner=None, legend=False)

    for i, tag in enumerate(df["tag_label"].unique()):
        vals = df.loc[df["tag_label"] == tag, y]
        q1, q2, q3 = vals.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        lower = max(vals.min(), q1 - 1.5 * iqr)
        upper = min(vals.max(), q3 + 1.5 * iqr)

        ax.vlines(i, q1, q3, color="black", linewidth=5)     # IQR
        ax.vlines(i, lower, upper, color="black", linewidth=1)  # whiskers
        ax.hlines(q2, i - 0.25, i + 0.25, color="black", linewidth=2)  # median
        ax.text(i, q2, f"Median = {q2:.3f}", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold",
                bbox=dict(facecolor="black", edgecolor="black",
                          boxstyle="round,pad=0.2"))

    ax.set(title="Genetic Algorithm: Fairness (Gini Index of Satisfaction)", xlabel="Algorithm",
           ylabel="Gini Index of Satisfaction (0 = Perfect Fairness)")
    ax.grid(False); sns.despine()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "ga_fairness_gini.png", dpi=300)
    plt.show()


def stripplot_summary(y_col: str, title: str, ylabel: str, fname: str):
    """Stripplot with mean/median markers + legend."""
    if y_col not in df.columns:
        print(f"[SKIP] Missing column: {y_col}")
        return

    plt.figure(figsize=(6.5, 5))
    ax = sns.stripplot(data=df, x="tag_label", y=y_col, palette=palette,
                       jitter=0.1, size=5, legend=False)

    means   = df.groupby("tag_label")[y_col].mean().round(2)
    medians = df.groupby("tag_label")[y_col].median().round(2)
    handles = []
    for i, tag in enumerate(df["tag_label"].unique()):
        ax.scatter(i, means[tag],   color="red",   marker="D", s=20)
        ax.scatter(i, medians[tag], color="black", marker="s", s=20)
        handles += [
            Line2D([0], [0], marker='D', color='w', markerfacecolor='red',
                   label=f"Mean = {means[tag]:.2f}", markersize=6),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='black',
                   label=f"Median = {medians[tag]:.2f}", markersize=6),
        ]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1),
              fontsize=8, title="Summary")
    ax.set(title=title, xlabel="Algorithm", ylabel=ylabel)
    ax.grid(False); sns.despine()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{fname}.png", dpi=300)
    plt.show()


def scatter_fairness_vs_violations():
    """Scatter: gini_satisfaction vs total_violations (per run)."""
    need = {"gini_satisfaction", "total_violations"}
    if not need.issubset(df.columns):
        print("[SKIP] Columns missing for scatter plot.")
        return
    plt.figure(figsize=(6.5, 5))
    ax = sns.scatterplot(data=df, x="total_violations", y="gini_satisfaction",
                         hue="tag_label", palette=palette, s=70)
    ax.set(title="Genetic Algorithm: Fairness vs Total Violations",
           xlabel="Total Number of Constraint Violations",
           ylabel="Gini Index of Satisfaction (0 = Perfect Fairness)")
    ax.grid(False); sns.despine()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "ga_fairness_vs_violations.png", dpi=300)
    plt.show()

#Bars (iterations, violations with integer y-axis), Histogram (integer y-axis)
def bar_iterations():
    """Average iterations to convergence (integer y-axis; SD only if non-zero)."""
    if "iterations" not in df.columns:
        print("[SKIP] 'iterations' column missing.")
        return

    mean = df.groupby("tag_label")["iterations"].mean()
    std  = df.groupby("tag_label")["iterations"].std()
    yerr = std.values if std.values.size and std.values[0] != 0 else None

    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    bars = ax.bar(mean.index, mean.values, yerr=yerr, capsize=5, color=GA_COLOR)
    for bar, val in zip(bars, mean):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.5, f"{int(round(val))}",
                ha="center", fontsize=10)

    ax.yaxis.set_major_locator(mtick.MaxNLocator(integer=True))  # integer ticks
    ax.set(title="Genetic Algorithm: Average Iterations to Convergence", xlabel="Algorithm", ylabel="Iterations")
    ax.grid(False); sns.despine()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "ga_iterations.png", dpi=300)
    plt.show()


def bar_constraint_violations():
    """Grouped barplot of average violations by type (integer y-axis)."""
    cols = ["capacity_viol", "elig_viol", "under_cap"]
    if not all(c in df.columns for c in cols):
        print("[SKIP] Violation columns missing in summary.")
        return

    avg_df = df.groupby("tag_label")[cols].mean().reset_index()
    melted = avg_df.melt(id_vars="tag_label", value_vars=cols,
                         var_name="Violation Type", value_name="Avg Count")

    #  Rename to human-friendly labels
    label_map = {
        "capacity_viol": "Capacity Violation",
        "elig_viol": "Eligibility Violation",
        "under_cap": "Under Capacity"
    }
    melted["Violation Type"] = melted["Violation Type"].map(label_map)

    #  Matching palette
    color_map = {
        "Capacity Violation": GA_COLOR,
        "Eligibility Violation": GA_GREEN,
        "Under Capacity": GA_YELLOW
    }

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(data=melted, x="tag_label", y="Avg Count", hue="Violation Type",
                     palette=color_map)

    ax.yaxis.set_major_locator(mtick.MaxNLocator(integer=True))  # integer ticks

    for container in ax.containers:
        for bar in container:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}",
                        ha="center", fontsize=9)

    ax.set(title="Genetic Algorithm: Average Constraint Violations per Run (30 Runs)",
           xlabel="", ylabel="Average Number of Violations")
    ax.set_xticklabels(melted["tag_label"].unique())
    ax.legend(title="Violation Type")
    ax.grid(False)
    sns.despine()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "ga_constraint_violations.png", dpi=300)
    plt.show()


def histogram_assigned_ranks():
    """Histogram of Assigned Preference Ranks (integer y-axis)."""
    patterns = [d / "**" / "ga_alloc_*.csv" for d in GA_BASE_DIRS]
    files = sorted([f for p in patterns for f in glob.glob(str(p), recursive=True)],
                   key=os.path.getmtime, reverse=True)
    if not files:
        print("[SKIP] No GA allocation files found.")
        return
    alloc_df = pd.read_csv(files[0])

    candidates = [
        "Matched Preference Rank", "Matched Preference",
        "Matched_Preference_Rank", "MatchedPreferenceRank",
    ]
    col = next((c for c in candidates if c in alloc_df.columns), None)
    if not col:
        print(f"[SKIP] Rank column not found in {files[0]}")
        return

    ser = pd.to_numeric(alloc_df[col], errors="coerce").dropna().astype(int)
    if ser.empty:
        print("[SKIP] Rank series empty.")
        return

    plt.figure(figsize=(6.5, 5))
    bins = range(ser.min(), ser.max() + 2)
    ax = sns.histplot(ser, bins=bins, color=GA_COLOR, edgecolor="black")
    ax.yaxis.set_major_locator(mtick.MaxNLocator(integer=True))  # integer ticks
    ax.set(title="Genetic Algorithm: Histogram of Assigned Preference Ranks",
           xlabel="Assigned Preference Rank", ylabel="Number of Students")
    ax.grid(False); sns.despine()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "ga_histogram_assigned_ranks.png", dpi=300)
    plt.show()

#Convergence loaders + mean±std & sampled (with labels & integer y for violations)
def load_convergence_curves():
    """
    Load all convergence files from both base dirs.
    - Harmonize 'total' -> 'fitness_score'
    - Compute 'total_violations' if missing (capacity + elig + under_cap)
    """
    patterns = [d / "**" / "convergence_ga_*.csv" for d in GA_BASE_DIRS]
    files = [f for p in patterns for f in glob.glob(str(p), recursive=True)]
    runs = []
    for f in files:
        try:
            d = pd.read_csv(f)
            if "fitness_score" not in d.columns and "total" in d.columns:
                d = d.rename(columns={"total": "fitness_score"})
            need = {"capacity_viol", "elig_viol", "under_cap"}
            if "total_violations" not in d.columns and need.issubset(d.columns):
                d["total_violations"] = (
                    d["capacity_viol"].astype(float) +
                    d["elig_viol"].astype(float) +
                    d["under_cap"].astype(float)
                )
            runs.append(d)
        except Exception:
            continue
    return runs


def plot_convergence_mean_std(metric: str, title: str, ylabel: str, filename: str, *, integer_y=False):
    """Mean ± SD across runs for a metric; optional integer y ticks."""
    runs = load_convergence_curves()
    values = [d[metric].to_numpy() for d in runs if metric in d.columns]
    if not values:
        print(f"[SKIP] No convergence data for '{metric}'.")
        return

    min_len = min(len(v) for v in values)
    data = np.array([v[:min_len] for v in values])
    mean, std = data.mean(axis=0), data.std(axis=0)
    x = np.arange(min_len)

    plt.figure(figsize=(8, 5.5))
    ax = plt.gca()
    ax.plot(x, mean, color=GA_COLOR, lw=2.5, label="Mean")
    ax.fill_between(x, mean - std, mean + std, color=GA_COLOR, alpha=0.2, label="± Std Dev")
    if integer_y:
        ax.yaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    ax.set(title=title, xlabel="Generation", ylabel=ylabel)
    ax.legend(); ax.grid(False); sns.despine()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=300)
    plt.show()


def plot_sampled_convergence(metric: str, title: str, ylabel: str, filename: str, n: int = 3, *, integer_y=False):
    """Plot a few sampled runs for the metric; optional integer y ticks."""
    runs = load_convergence_curves()
    valid = [d for d in runs if metric in d.columns and "generation" in d.columns]
    if not valid:
        print(f"[SKIP] No convergence data for '{metric}'.")
        return

    sampled = random.sample(valid, min(n, len(valid)))
    styles = ["-", "--", ":"]
    plt.figure(figsize=(8, 5.5))
    ax = plt.gca()
    for i, d in enumerate(sampled):
        ax.plot(d["generation"], d[metric], linestyle=styles[i % 3],
                color=GA_COLOR, lw=2.2, label=f"Run {i+1}")
    if integer_y:
        ax.yaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    ax.set(title=title, xlabel="Generation", ylabel=ylabel)
    ax.legend(title="Sampled Runs"); ax.grid(False); sns.despine()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=300)
    plt.show()

#Master calls (all 18 plots with clearer labels)
if __name__ == "__main__":
    print("[INFO] Generating all 18 GA plots...")

    # Boxplots 
    boxplot_with_median("fitness_score",   "Genetic Algorithm: Total Fitness Score (Lower = Better)",              "Fitness Score (lower = better)", "ga_total_score", integer_y=False, median_decimals=2)
    boxplot_with_median("pref_penalty",    "Genetic Algorithm: Preference Penalty (Sum of Ranks)",         "Sum of Preference Ranks",         "ga_preference_penalty", integer_y=True,  median_decimals=0)
    boxplot_with_median("avg_rank",        "Genetic Algorithm: Average Assigned Preference Rank (Lower = Better)",    "Average Assigned Preference Rank",                    "ga_avg_assigned_rank", integer_y=True,  median_decimals=2)
    boxplot_with_median("runtime_sec",     "Genetic Algorithm: Runtime per Run (Seconds)",                 "Runtime (Seconds)",               "ga_runtime",           integer_y=False, median_decimals=2)
    boxplot_with_median("total_violations","Genetic Algorithm: Total Constraint Violations",               "Violations (count)",              "ga_total_violations",  integer_y=True,  median_decimals=0)

    # Violin + Stripplots 
    violin_fairness()
    stripplot_summary("top1_pct", "Genetic Algorithm: Top-1 Preference Match Rate (%)",  "% of Students Assigned to 1st Choice",     "ga_top1_pct")
    stripplot_summary("top3_pct", "Genetic Algorithm: Top-3 Preference Match Rate (%)",  "% of Students Assigned to Top-3 Choices",  "ga_top3_pct")


    # Bars / Scatter / Histogram 
    bar_iterations()
    bar_constraint_violations()
    scatter_fairness_vs_violations()
    histogram_assigned_ranks()

    #  Convergence (Mean ± SD) 
    plot_convergence_mean_std("fitness_score",
        "Genetic Algorithm: Best Fitness Score Convergence (Mean ± SD)", "Fitness Score (lower = better allocation)",
        "ga_convergence_score_avg.png", integer_y=False)
    plot_convergence_mean_std("gini",
        "Genetic Algorithm: Fairness Convergence (Mean ± SD)", "Gini Index of Satisfaction (0 = Perfect Fairness)",
        "ga_convergence_gini_avg.png", integer_y=False)
    plot_convergence_mean_std("total_violations",
        "Genetic Algorithm: Violations Convergence (Mean ± SD)", "Total Number of Constraint Violations",
        "ga_convergence_violations_avg.png", integer_y=True)

    #  Convergence (Sampled) 
    plot_sampled_convergence("fitness_score",
        "Genetic Algorithm: Best Fitness Score Convergence (Sampled Runs)", "Fitness Score (lower = better allocation)",
        "ga_convergence_score_sampled.png", integer_y=False)
    plot_sampled_convergence("gini",
        "Genetic Algorithm: Fairness Convergence (Sampled Runs)", "Gini Index of Satisfaction (0 = Perfect Fairness)",
        "ga_convergence_gini_sampled.png", integer_y=False)
    plot_sampled_convergence("total_violations",
        "Genetic Algorithm: Violations Convergence (Sampled Runs)", "Total Number of Constraint Violations",
        "ga_convergence_violations_sampled.png", integer_y=True)

    print(f"\n All plots saved to: {PLOT_DIR}")
