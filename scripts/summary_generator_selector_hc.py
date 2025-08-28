"""
Generates a markdown summary: selectorhc_summary.md
Includes:
- Mean ± Standard Deviation
- Median ± Interquartile Range
- Top 5 runs by Fitness Score, Top-1 Preference Match Rate , Top-3 Preference Match Rate , Gini and Violations
- Best/worst by metric
- Operator usage
- Dataset hash traceability
(Aggregates only 30 most recent runs)

"""

import pandas as pd
import json
import locale
import warnings
from pathlib import Path
from scipy.stats import iqr
from tabulate import tabulate

# Locale and warnings
locale.setlocale(locale.LC_ALL, 'C')
warnings.filterwarnings("ignore", message="Could not infer format")

# Paths
ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = ROOT / "results/selector_hill_climbing_runs/selectorhc_summary_log.csv"
OUTPUT_MD = ROOT / "results" / "summary" / "selectorhc_summary.md"

# Label mappings
COLUMN_LABELS = {
    "fitness_score": "Fitness Score",
    "pref_penalty": "Penalty",
    "avg_rank": "Average Preference Rank",
    "top1_pct": "Top-1 Preference Match Rate (%)",
    "top3_pct": "Top-3 Preference Match Rate (%)",
    "gini_satisfaction": "Gini Index",
    "total_violations": "Total Violations",
    "runtime_sec": "Runtime (s)"
}

# Friendly operator label mapping
OPERATOR_LABELS = {
    "swap": "Swap",
    "mut_micro": "Micro Mutation",
    "reassign": "Reassign"
}

def generate_selectorhc_summary():
    if not SUMMARY_CSV.exists():
        print(" selectorhc_summary_log.csv not found.")
        return

    df = pd.read_csv(SUMMARY_CSV)
    if df.empty:
        print(" selectorhc_summary_log.csv is empty.")
        return

    # Use only latest 30 by timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%dT%H-%M-%SZ", errors="coerce")
        df = (
            df.dropna(subset=["timestamp"])
              .sort_values("timestamp", ascending=False)
              .head(30)
              .reset_index(drop=True)
        )
    else:
        print("'timestamp' column not found. Using full dataset.")

    metrics = list(COLUMN_LABELS.keys())
    md = []
    md.append("# Selector-HC Summary Results")
    md.append("")
    md.append(f"This summary aggregates **{len(df)}** recent Selector-HC runs.")
    md.append("")

    # Mean ± Standard Deviation
    md.append("## Mean ± Standard Deviation by Metric\n")
    mean_std_data = []
    for col in metrics:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            mean_std_data.append([COLUMN_LABELS[col], f"{mean:.2f}", f"{std:.2f}"])
    md.append(tabulate(mean_std_data, headers=["Metric", "Mean", "Standard Deviation"], tablefmt="github", showindex=False))
    md.append("")

    # Median + Interquartile Range
    md.append("## Median (Interquartile Range) by Metric\n")
    median_iqr_data = []
    for col in metrics:
        if col in df.columns:
            median = df[col].median()
            iqr_val = iqr(df[col])
            median_iqr_data.append([COLUMN_LABELS[col], f"{median:.2f}", f"{iqr_val:.2f}"])
    md.append(tabulate(median_iqr_data, headers=["Metric", "Median", "Interquartile Range"], tablefmt="github", showindex=False))
    md.append("")

    # Top 5 by Fitness Score
    md.append("## Top 5 Runs by Fitness Score (lower = better)\n")
    cols = ["run_id", "fitness_score", "top1_pct", "gini_satisfaction", "total_violations", "runtime_sec"]
    if all(c in df.columns for c in cols):
        top_fitness = df.nsmallest(5, "fitness_score")[cols].round(3)
        top_fitness = top_fitness.rename(columns=COLUMN_LABELS)
        md.append(tabulate(top_fitness, headers="keys", tablefmt="github", showindex=False))
        md.append("")

    # Top 5 by Top-1 Preference Match Rate %
    md.append("## Top 5 Runs by Top-1 Preference Match Rate (%)\n")
    cols = ["run_id", "top1_pct", "fitness_score", "gini_satisfaction", "total_violations", "runtime_sec"]
    if all(c in df.columns for c in cols):
        top_top1 = df.nlargest(5, "top1_pct")[cols].round(3)
        top_top1 = top_top1.rename(columns=COLUMN_LABELS)
        md.append(tabulate(top_top1, headers="keys", tablefmt="github", showindex=False))
        md.append("")

    # NEW: Top 5 by Top-3 Preference Match Rate %
    md.append("## Top 5 Runs by Top-3 Preference Match Rate (%)\n")
    cols = ["run_id", "top3_pct", "fitness_score", "gini_satisfaction", "total_violations", "runtime_sec"]
    if all(c in df.columns for c in cols):
        top_top3 = df.nlargest(5, "top3_pct")[cols].round(3)
        top_top3 = top_top3.rename(columns=COLUMN_LABELS)
        md.append(tabulate(top_top3, headers="keys", tablefmt="github", showindex=False))
        md.append("")

    # Top 5 by Fairness (lowest Gini)
    md.append("## Top 5 Runs by Fairness (Gini Index; lower = fairer)\n")
    cols = ["run_id", "gini_satisfaction", "fitness_score", "top1_pct", "total_violations", "runtime_sec"]
    if all(c in df.columns for c in cols):
        top_fair = df.nsmallest(5, "gini_satisfaction")[cols].round(3)
        top_fair = top_fair.rename(columns=COLUMN_LABELS)
        md.append(tabulate(top_fair, headers="keys", tablefmt="github", showindex=False))
        md.append("")

    # Best/Worst by Metric
    md.append("## Best / Worst Runs by Metric\n")
    best_worst_data = []
    for col in metrics:
        if col in df.columns:
            valid = df[col].dropna()
            if not valid.empty:
                best = df.loc[valid.idxmin()]
                worst = df.loc[valid.idxmax()]
                best_worst_data.append([COLUMN_LABELS[col], "Best", best["run_id"], f"{best[col]:.2f}"])
                best_worst_data.append([COLUMN_LABELS[col], "Worst", worst["run_id"], f"{worst[col]:.2f}"])
    md.append(tabulate(best_worst_data, headers=["Metric", "Type", "run_id", "value"], tablefmt="github", showindex=False))
    md.append("")

    # Operator Usage
    md.append("## Operator Usage (Aggregated)\n")
    op_counts = {}
    if "operator_seq" in df.columns:
        for val in df["operator_seq"].dropna():
            try:
                parsed = json.loads(val)
                for op, count in parsed.items():
                    op_counts[op] = op_counts.get(op, 0) + count
            except:
                continue

    total_ops = sum(op_counts.values())
    if total_ops > 0:
        op_data = []
        for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / total_ops
            op_name = OPERATOR_LABELS.get(op, op)  # fallback to original name
            op_data.append([op_name, count, f"{pct:.1f}%"])
        md.append(tabulate(op_data, headers=["Operator", "Total Used", "% Usage"], tablefmt="github", showindex=False))
    else:
        md.append("_No operator usage recorded._")
    md.append("")

    # Dataset Hashes
    md.append("## Dataset Hashes Used\n")
    if "dataset_hash" in df.columns:
        hashes = df["dataset_hash"].dropna().unique()
        if hashes.any():
            for h in hashes:
                md.append(f"- `{h}`")
        else:
            md.append("_No dataset hashes recorded._")
    else:
        md.append("_No dataset hashes recorded._")

    # Save Markdown
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f" Markdown summary saved to: {OUTPUT_MD}")

if __name__ == "__main__":
    generate_selectorhc_summary()
