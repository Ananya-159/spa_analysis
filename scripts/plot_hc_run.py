# scripts/plot_hc_run.py
"""
Single-run Hill Climbing convergence plot.

Generates a 3-panel figure:
1. Fitness Score (best_total)
2. Gini Index (Fairness)
3. Total Constraint Violations

"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path
import seaborn as sns

# Color settings for consistent theme
HC_BLUE   = "#1f77b4"
HC_GREEN  = "green"
HC_YELLOW = "gold"

# Root project path
ROOT = Path(__file__).resolve().parents[1]

# Load the convergence CSV for a given run
def load_convergence(run_id: str, tag: str) -> pd.DataFrame:
    path = ROOT / "results" / "Hill_climbing_runs" / f"convergence_hc_{tag}_{run_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No convergence file for run_id={run_id}, tag={tag}")
    print(f"[LOAD] {path.name}")
    return pd.read_csv(path)

# Load caption info from summary log
def load_caption(run_id: str) -> str:
    summary = ROOT / "results" / "Hill_climbing_runs" / "hc_summary_log.csv"
    if not summary.exists():
        return f"Hill Climbing run (run_id={run_id})"
    df = pd.read_csv(summary)
    row = df[df["run_id"] == run_id]
    if row.empty:
        return f"Hill Climbing run (run_id={run_id})"
    r = row.iloc[0]
    # Split caption into two lines using \n
    return (
        f"Hill Climbing run (tag={r['tag']}, run_id={run_id})\n"
        f"Fitness score={r['total']:.0f}, Gini={r['gini_satisfaction']:.3f}, "
        f"Top1 Preference Match Rate ={r['top1_pct']:.1f}%, Average Preference Rank={r['avg_rank']:.2f}"
    )

# Main plotting function
def plot_hc_run(
    run_id: str,
    tag: str,
    save_fig: bool = True,
    smooth: bool = False,
    smooth_window: int = 5
):
    # Load data
    df = load_convergence(run_id, tag)
    caption = load_caption(run_id)

    # Set up plot
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    font_title, font_label, font_ticks = 15, 13, 11

    # Panel 1: Fitness Score 
    axs[0].plot(df["iter"], df["best_total"], color=HC_BLUE, linewidth=2, label="Fitness Score")
    axs[0].set_title("Fitness Score Over Iterations", fontsize=font_title)
    axs[0].set_ylabel("Total Score (lower is better)", fontsize=font_label)
    axs[0].tick_params(labelsize=font_ticks)
    axs[0].legend(loc="upper right", fontsize=11)
    axs[0].grid(False)

    # Panel 2: Fairness (Gini Index)
    gini_vals = df["gini"].rolling(smooth_window, center=True).mean() if smooth else df["gini"]
    axs[1].plot(df["iter"], gini_vals, color=HC_GREEN, linewidth=2, label="Gini (Fairness)")
    axs[1].set_title("Fairness Over Iterations", fontsize=font_title)
    axs[1].set_ylabel("Gini Index (0 = fair)", fontsize=font_label)
    axs[1].tick_params(labelsize=font_ticks)
    axs[1].legend(loc="upper right", fontsize=11)
    axs[1].grid(False)

    # Panel 3: Total Constraint Violations
    y_vals = df["total_violations"].rolling(smooth_window, center=True).mean() if smooth else df["total_violations"]
    axs[2].plot(df["iter"], y_vals, color=HC_YELLOW, linewidth=2, label="Total Violations")
    axs[2].set_title("Constraint Violations Over Iterations", fontsize=font_title)
    axs[2].set_ylabel("Violations", fontsize=font_label)
    axs[2].set_xlabel("Iteration", fontsize=font_label)
    axs[2].tick_params(labelsize=font_ticks)
    axs[2].yaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    axs[2].legend(loc="upper right", fontsize=11)
    axs[2].grid(False)

    # Super Title (with split) 
    plt.suptitle(caption, fontsize=font_title + 1)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for title

    # Save figure if requested 
    if save_fig:
        out_dir = ROOT / "results" / "plots" / "Hill_climbing"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"hc_run_convergence_{run_id}.png"
        plt.savefig(out_path, dpi=300)
        print(f"[SAVE] Plot saved to: {out_path.name}")

    plt.show()

# Run directly from script (update run_id/tag as needed)
if __name__ == "__main__":
    plot_hc_run(
        run_id="d6293262",
        tag="phase3_hc_greedy",
        save_fig=True,
        smooth=True
    )
