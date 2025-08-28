# spa_analysis/scripts/plot_ga_run.py

"""
Single-run plotting for Genetic Algorithm (GA) experiments.

Generates 3-panel visualizations for a given GA run, including fitness convergence,
fairness (Gini), and violation trends. Reads data from `/results/Genetic_algorithm_runs/`
and saves plots into `/results/plots/Genetic_algorithm/`.

"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from typing import Optional

# Colors
GA_BLUE   = "#1f77b4"
GA_GREEN  = "green"
GA_YELLOW = "yellow"

# Loaders
def read_data_file(path_stem: Path) -> pd.DataFrame:
    csv_path = path_stem.with_suffix(".csv")
    xlsx_path = path_stem.with_suffix(".xlsx")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if xlsx_path.exists():
        return pd.read_excel(xlsx_path)
    raise FileNotFoundError(f" File not found for stem: {path_stem}")

def find_debug_dir(root: Path) -> Path:
    candidates = [
        root / "results" / "Genetic_algorithm_runs" / "Genetic_algorithm_runseed_tests",
        root / "results" / "Hill_climbing_runs" / "Genetic_algorithm_runseed_tests",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]

def find_ga_summary_csv(root: Path) -> Optional[Path]:
    for p in [
        root / "results" / "Genetic_algorithm_runs" / "ga_summary_log.csv",
        root / "results" / "Hill_climbing_runs" / "ga_summary_log.csv",
    ]:
        if p.exists():
            return p
    return None

def load_ga_summary(root: Path) -> Optional[pd.DataFrame]:
    path = find_ga_summary_csv(root)
    if path is None:
        return None
    df = pd.read_csv(path)
    if "fitness_score" not in df.columns and "total" in df.columns:
        df = df.rename(columns={"total": "fitness_score"})
    parts = {"capacity_viol", "elig_viol", "under_cap"}
    if "total_violations" not in df.columns and parts.issubset(df.columns):
        df["total_violations"] = (
            pd.to_numeric(df["capacity_viol"], errors="coerce") +
            pd.to_numeric(df["elig_viol"], errors="coerce") +
            pd.to_numeric(df["under_cap"],  errors="coerce")
        )
    for c in ["fitness_score", "top1_pct", "gini_satisfaction", "total_violations", "runtime_sec"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def build_caption(root: Path, run_id: str) -> str:
    df = load_ga_summary(root)
    if df is None:
        return f"Genetic Algorithm debug run (run_id={run_id})."
    row = df[df["run_id"].astype(str) == str(run_id)]
    if row.empty:
        return f"Genetic Algorithm debug run (run_id={run_id})."
    r = row.iloc[0]
    parts = [f"Genetic Algorithm run (run_id={run_id}"]
    if "seed" in row.columns and pd.notna(r.get("seed")):
        parts.append(f"seed={int(r['seed'])}")
    if pd.notna(r.get("fitness_score")):
        parts.append(f"fitness={r['fitness_score']:.0f}")
    if pd.notna(r.get("gini_satisfaction")):
        parts.append(f"gini={r['gini_satisfaction']:.3f}")
    if pd.notna(r.get("top1_pct")):
        parts.append(f"top1={r['top1_pct']:.1f}%")
    if pd.notna(r.get("total_violations")):
        parts.append(f"viol={int(r['total_violations'])}")
    parts.append(")")
    return ", ".join(parts)

# Main Plot
def plot_convergence_and_diversity(run_id: str, save_fig=True, smooth_ops=False, smooth_window=3):
    # Paths
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent
    results_path = find_debug_dir(root)

    # Stems
    conv_stem = results_path / f"convergence_ga_debug_test_{run_id}"
    debug_stem = results_path / f"ga_debug_debug_test_{run_id}"
    conv_df = read_data_file(conv_stem)
    debug_df = read_data_file(debug_stem)

    if "fitness_score" not in conv_df.columns and "total" in conv_df.columns:
        conv_df = conv_df.rename(columns={"total": "fitness_score"})

    # Fonts
    font_title, font_label, font_ticks, legend_font = 16, 14, 12, 12
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    caption = build_caption(root, run_id)

    # Panel 1: Fitness
    axs[0].plot(conv_df["generation"], conv_df["fitness_score"], label="Best Fitness Score", color=GA_BLUE, linewidth=2)
    axs[0].plot(conv_df["generation"], conv_df["avg_total"], label="Avg Population Fitness", color=GA_GREEN, linestyle="--")
    axs[0].set_ylabel("Fitness Score\n(lower = better allocation)", fontsize=font_label)
    axs[0].set_title(f"Convergence Over Generations — Run {run_id}", fontsize=font_title)
    axs[0].legend(fontsize=legend_font)
    axs[0].tick_params(axis="both", labelsize=font_ticks)
    axs[0].grid(False)

    # Panel 2: Diversity
    axs[1].plot(debug_df["generation"], debug_df["unique_individuals"], label="Unique Individuals", color=GA_BLUE)
    axs[1].plot(debug_df["generation"], debug_df["avg_hamming_to_best"], label="Avg Hamming to Best", color=GA_GREEN)
    axs[1].set_ylabel("Diversity", fontsize=font_label)
    axs[1].set_title("Diversity Over Generations", fontsize=font_title)
    axs[1].legend(fontsize=legend_font)
    axs[1].tick_params(axis="both", labelsize=font_ticks)
    axs[1].grid(False)

    # Panel 3: Operator Usage
    avg_cx = conv_df["cx_ops_gen"].mean()
    avg_mut = conv_df["mut_ops_gen"].mean()
    avg_repair = conv_df["repair_calls_gen"].mean()

    axs[2].plot(conv_df["generation"], conv_df["cx_ops_gen"], label=f"Crossover (avg ≈ {avg_cx:.1f})", color=GA_BLUE)
    axs[2].plot(conv_df["generation"], conv_df["mut_ops_gen"], label=f"Mutation (avg ≈ {avg_mut:.1f})", color=GA_GREEN)
    axs[2].plot(conv_df["generation"], conv_df["repair_calls_gen"], label=f"Repair (avg ≈ {avg_repair:.1f})", color=GA_YELLOW)

    if smooth_ops and len(conv_df) >= smooth_window:
        axs[2].plot(conv_df["generation"], conv_df["cx_ops_gen"].rolling(smooth_window, center=True, min_periods=1).mean(),
                    color=GA_BLUE, linewidth=2.5, alpha=0.4)
        axs[2].plot(conv_df["generation"], conv_df["mut_ops_gen"].rolling(smooth_window, center=True, min_periods=1).mean(),
                    color=GA_GREEN, linewidth=2.5, alpha=0.4)
        axs[2].plot(conv_df["generation"], conv_df["repair_calls_gen"].rolling(smooth_window, center=True, min_periods=1).mean(),
                    color=GA_YELLOW, linewidth=2.5, alpha=0.4)

    axs[2].set_xlabel("Generation", fontsize=font_label)
    axs[2].set_ylabel("Operation Count", fontsize=font_label)
    axs[2].set_title("Operator Usage Per Generation", fontsize=font_title)
    axs[2].legend(fontsize=legend_font)
    axs[2].tick_params(axis="both", labelsize=font_ticks)
    axs[2].yaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    axs[2].grid(False)

    # Caption and Layout
    plt.suptitle(caption, fontsize=font_title + 1)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(left=0.12)

    # Save
    if save_fig:
        save_folder = root / "results" / "plots" / "Genetic_algorithm"
        save_folder.mkdir(parents=True, exist_ok=True)
        out_path = save_folder / f"ga_run_convergence_{run_id}.png"
        plt.savefig(out_path, dpi=300)
        print(f"Plot saved to: {out_path.resolve()}")

    # Console caption
    print("\nSuggested caption:")
    print(caption)
    plt.show()

# Entry point
if __name__ == "__main__":
    plot_convergence_and_diversity(
        run_id="20250819T230648Z",  # Replace as needed
        save_fig=True,
        smooth_ops=False,
        smooth_window=3
    )
