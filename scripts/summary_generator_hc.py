"""
summary_generator_hc.py - Generates hc_summary.md

Outputs:
- Mean ± Standard Deviation
- Median ± Interquartile Range
- Top 5 runs by: Fitness, Top-1 Preference Match Rate,Top-3 Preference Match Rate, Gini, Violations
- Best/worst by metric

"""

import os
import pandas as pd
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "Hill_climbing_runs" / "hc_summary_log.csv"
OUT_DIR = ROOT / "results" / "summary"
OUT_MD = OUT_DIR / "hc_summary.md"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Column labels
COLUMN_LABELS = {
    "fitness_score":     "Fitness Score",
    "pref_penalty":      "Penalty",
    "avg_rank":          "Avg Pref Rank",
    "top1_pct":          "Top-1 Match (%)",
    "top3_pct":          "Top-3 Match (%)",
    "gini_satisfaction": "Gini Index",
    "total_violations":  "Total Violations",
    "runtime_sec":       "Runtime (s)"
}

# Metric settings
METRICS = {
    "fitness_score":     {"label": "Fitness Score",    "higher_better": False},
    "pref_penalty":      {"label": "Penalty",          "higher_better": False},
    "avg_rank":          {"label": "Avg Pref Rank",    "higher_better": False},
    "top1_pct":          {"label": "Top-1 Match (%)",  "higher_better": True},
    "top3_pct":          {"label": "Top-3 Match (%)",  "higher_better": True},
    "gini_satisfaction": {"label": "Gini Index",        "higher_better": False},
    "total_violations":  {"label": "Total Violations", "higher_better": False},
    "runtime_sec":       {"label": "Runtime (s)",       "higher_better": False},
}

# Load and preprocess data
df = pd.read_csv(CSV_PATH)

df["tag_label"] = (
    df["tag"]
    .astype(str)
    .str.replace("phase3_hc_", "HC ", regex=False)
    .str.replace("_", " ")
    .apply(lambda s: "HC Random" if "random" in s.lower()
           else "HC Greedy" if "greedy" in s.lower()
           else s.strip().title())
)

df["fitness_score"] = pd.to_numeric(df["total"], errors="coerce")

# Formatting functions
def fmt_mean_std(s: pd.Series) -> str:
    return f"{s.mean():.2f}\n±{s.std():.2f}" if not s.empty else "-"

def fmt_median_iqr(s: pd.Series, col: str) -> str:
    if s.empty:
        return "-"
    q1, q2, q3 = s.quantile([0.25, 0.5, 0.75])
    if col in {"fitness_score", "pref_penalty", "total_violations", "runtime_sec"}:
        return f"{int(round(q2))}\n[{int(round(q1))}–{int(round(q3))}]"
    elif col in {"gini_satisfaction", "avg_rank"}:
        return f"{q2:.3f}\n[{q1:.3f}–{q3:.3f}]"
    elif col in {"top1_pct", "top3_pct"}:
        return f"{q2:.1f}\n[{q1:.1f}–{q3:.1f}]"
    else:
        return f"{q2:.2f}\n[{q1:.2f}–{q3:.2f}]"

# Summary tables
def mean_std_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tag, group in df.groupby("tag_label"):
        row = {"Algorithm": tag}
        for col, meta in METRICS.items():
            if col in group.columns:
                row[meta["label"]] = fmt_mean_std(group[col])
        rows.append(row)
    return pd.DataFrame(rows)

def median_iqr_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tag, group in df.groupby("tag_label"):
        row = {"Algorithm": tag}
        for col, meta in METRICS.items():
            if col in group.columns:
                row[meta["label"]] = fmt_median_iqr(group[col], col)
        rows.append(row)
    return pd.DataFrame(rows)

# Top N
def top_n(df: pd.DataFrame, col: str, n: int = 5, higher=True):
    if col not in df:
        return pd.DataFrame()
    return df.sort_values(col, ascending=not higher).head(n)

# Best/worst
def best_worst(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, meta in METRICS.items():
        if col not in df:
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        best = df.loc[s.idxmax()] if meta["higher_better"] else df.loc[s.idxmin()]
        worst = df.loc[s.idxmin()] if meta["higher_better"] else df.loc[s.idxmax()]
        for typ, r in zip(["Best", "Worst"], [best, worst]):
            rows.append({
                "Metric": meta["label"],
                "Type": typ,
                "run_id": r.get("run_id", ""),
                "Algorithm": r.get("tag_label", ""),
                "value": round(r[col], 3)
            })
    return pd.DataFrame(rows)

# Rename columns for output
def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=COLUMN_LABELS)

# Markdown output
def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={**COLUMN_LABELS, "tag_label": "Algorithm"})

# Markdown output helper
def df_to_md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False) if not df.empty else "_No data available._"

# Write markdown
with open(OUT_MD, "w", encoding="utf-8") as f:
    # Count algorithms
    count_per_algo = df["tag_label"].value_counts()
    random_count = count_per_algo.get("HC Random", 0)
    greedy_count = count_per_algo.get("HC Greedy", 0)

    f.write("# Hill Climbing Summary Results\n\n")
    f.write(f"This summary aggregates **{random_count} HC Random** and **{greedy_count} HC Greedy** runs.\n\n")

    f.write("## Mean ± Standard Deviation by Algorithm\n\n")
    f.write(df_to_md(mean_std_table(df)))
    f.write("\n\n")

    f.write("## Median (Interquartile Range) by Algorithm\n\n")
    f.write(df_to_md(median_iqr_table(df)))
    f.write("\n\n")

    top_cols = ["run_id", "tag_label", "fitness_score", "top1_pct",
                "gini_satisfaction", "total_violations", "runtime_sec"]

    f.write("### Top 5 Runs by Fitness (lower = better)\n\n")
    f.write(df_to_md(rename_columns(
        top_n(df, "fitness_score", 5, higher=False)[top_cols])))
    f.write("\n\n")

    f.write("### Top 5 Runs by Top-1 Preference Match Rate (%)\n\n")
    f.write(df_to_md(rename_columns(
        top_n(df, "top1_pct", 5, higher=True)[top_cols])))
    f.write("\n\n")

    # Top-3 section (mirrors Top-1 section)
    f.write("### Top 5 Runs by Top-3 Preference Match Rate (%)\n\n")
    top_cols_top3 = ["run_id", "tag_label", "fitness_score", "top3_pct",
                     "gini_satisfaction", "total_violations", "runtime_sec"]
    f.write(df_to_md(rename_columns(
        top_n(df, "top3_pct", 5, higher=True)[top_cols_top3])))
    f.write("\n\n")

    f.write("### Top 5 Runs by Fairness (Gini; lower = better)\n\n")
    f.write(df_to_md(rename_columns(
        top_n(df, "gini_satisfaction", 5, higher=False)[top_cols])))
    f.write("\n\n")

    f.write("### Top 5 Runs by Violations (lower = better)\n\n")
    f.write(df_to_md(rename_columns(
        top_n(df, "total_violations", 5, higher=False)[top_cols])))
    f.write("\n\n")

    f.write("## Best/Worst by Metric\n\n")
    f.write(df_to_md(best_worst(df)))
    f.write("\n\n")

    f.write("_Auto-generated from `hc_summary_log.csv`_\n")

print(f" Markdown summary saved to: {OUT_MD}")
