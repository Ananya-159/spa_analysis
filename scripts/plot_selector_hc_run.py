# spa_analysis/scripts/plot_selector_hc_run.py

"""
Single-run plotting for Selector Hill Climbing (Selector-HC) experiments.

Generates a 3-panel visualization of one Selector-HC run, showing:
1) Fitness score
2) Fairness (Gini)
3) Operator usage (with optional smoothing)

Outputs saved under /results/plots/selector_hill_climbing/
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path
import seaborn as sns
from collections import Counter

# Consistent colors
COLOR_BLUE   = "#1f77b4"
COLOR_GREEN  = "green"
COLOR_YELLOW = "yellow"

def plot_selector_run(
    seed: str,
    save_fig: bool = True,
    smooth_ops: bool = True,
    smooth_window: int = 5
):
    """
    Plot 3-panel Selector-HC run:
    1) Fitness score
    2) Fairness (Gini)
    3) Operator usage with optional smoothing

    Includes caption and title with: fitness, gini, top‑1, avg_rank

    Parameters:
    - seed (str): Selector-HC seed identifier
    - save_fig (bool): whether to save the figure as PNG
    - smooth_ops (bool): enable operator smoothing overlay
    - smooth_window (int): smoothing window size
    """
    ROOT = Path(__file__).resolve().parents[1]
    run_file = ROOT / "results" / "selector_hill_climbing_runs" / f"*selectorhc_convergence_seed{seed}.csv"

    # Find convergence CSV
    matches = list(run_file.parent.glob(run_file.name))
    if not matches:
        raise FileNotFoundError(f"No convergence file found for seed={seed}")
    df = pd.read_csv(matches[0])
    print(f"[LOAD] Loaded: {matches[0].name}")

    # Building the title from summary
    caption = f"Selector Hill Climbing run (seed={seed})"
    sum_path = ROOT / "results" / "selector_hill_climbing_runs" / "selectorhc_summary_log.csv"
    if sum_path.exists():
     try:
        sum_df = pd.read_csv(sum_path)
        row = sum_df[sum_df["run_id"] == f"selectorhc_{seed}"].iloc[0]
        caption = (
            f"Selector Hill Climbing run (seed={seed})\n"
            f"Fitness Score={row['fitness_score']:.0f}, "
            f"Gini={row['gini_satisfaction']:.3f}, "
            f"Top-1 Preference Match Rate={row['top1_pct']:.1f}%, "
            f"Average Preference Rank={row['avg_rank']:.2f}"
        )
     except Exception:
        pass


    # Operator usage counts
    op_counts = Counter(df["op"].dropna())
    all_ops = ["swap", "reassign", "mut_micro"]
    op_freq = [op_counts.get(op, 0) for op in all_ops]

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    font_title, font_label, font_ticks = 15, 13, 11

    # Panel 1: Fitness
    axs[0].plot(df["iter"], df["fitness_score"], color=COLOR_BLUE, linewidth=2, label="Fitness Score")
    axs[0].set_title("Fitness Score Over Iterations", fontsize=font_title)
    axs[0].set_ylabel("Fitness Score (lower is better)", fontsize=font_label)
    axs[0].tick_params(labelsize=font_ticks)
    axs[0].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=11)
    axs[0].grid(False)

    # Panel 2: Gini (Fairness)
    gini_vals = df["gini"].rolling(smooth_window, center=True).mean() if smooth_ops else df["gini"]
    axs[1].plot(df["iter"], gini_vals, COLOR_GREEN , linewidth=2, label="Gini (Fairness)")
    axs[1].set_title("Fairness Over Iterations", fontsize=font_title)
    axs[1].set_ylabel("Gini Index (0 = fair)", fontsize=font_label)
    axs[1].tick_params(labelsize=font_ticks)
    axs[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=11)
    axs[1].grid(False)

    # Panel 3: Operator usage
    op_series = df["op"]
    iter_series = df["iter"]
    for op, color in zip(all_ops, [COLOR_BLUE, COLOR_GREEN, COLOR_YELLOW]):
        y = (op_series == op).astype(int)
        axs[2].plot(iter_series, y, label=op, alpha=0.4, color=color)

        # Smoothing overlay
        if smooth_ops:
            smooth_y = y.rolling(smooth_window, min_periods=1, center=True).mean()
            axs[2].plot(iter_series, smooth_y, color=color, linewidth=2)

    axs[2].set_title("Operator Usage Over Iterations", fontsize=font_title)
    axs[2].set_ylabel("Operator Applied (1 = Yes)", fontsize=font_label)
    axs[2].set_xlabel("Iteration", fontsize=font_label)
    axs[2].tick_params(labelsize=font_ticks)
    axs[2].yaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    axs[2].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=11)
    axs[2].grid(False)

    # Main title above all panels
    fig.suptitle(caption, fontsize=font_title + 1, y=0.98)
    plt.tight_layout(rect=[0, 0, 0.88, 0.96])  # shrinks plot width to leave legend space

    # Save
    if save_fig:
        out_dir = ROOT / "results" / "plots" / "selector_hill_climbing"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"selectorhc_run_convergence_{seed}.png"
        plt.savefig(out_path, dpi=300)
        print(f"[SAVE] Plot saved to: {out_path.name}")

    plt.show()


# Entry point
if __name__ == "__main__":
    plot_selector_run(
        seed="123",          # Actual seed of first run - randomly chosen
        save_fig=True,
        smooth_ops=True,     # Keeping it enabled for clear smoothed overlays
        smooth_window=5
    )
