# spa_analysis/scripts/summary_generator_ga.py

"""
Generates a clean Genetic Algorithm summary (ga_summary.md)
Includes:
- Mean ± Standard Deviation
- Median ± Interquartile Range
- Top 5 by: Fitness, Top-1 Preference Match Rate, Top-3 Preference Match Rate, Gini, Violations
- Best/Worst by metric
- Operator statistics

"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from typing import Dict, Optional

TOP_N = 5
ROOT = Path(__file__).resolve().parents[1]

# Label Mapping for All Tables
COLUMN_LABELS = { 
    "fitness_score": "Fitness Score", 
    "pref_penalty": "Penalty", 
    "avg_rank": "Avg Pref Rank", 
    "top1_pct": "Top-1 Match (%)", 
    "top3_pct": "Top-3 Match (%)", 
    "gini_satisfaction": "Gini Index", 
    "total_violations": "Total Violations", 
    "runtime_sec": "Runtime (s)" 
}

METRICS = {
    k: {"label": v, "higher_better": (k in {"top1_pct", "top3_pct"})}
    for k, v in COLUMN_LABELS.items()
}

OP_LABELS = {
    "cx_ops": "Crossover Operations",
    "mut_ops": "Mutation Operations",
    "repair_calls": "Repair Calls",
    "repair_per_op": "Repairs per Operation"
}

def find_summary_csv() -> Path:
    for p in [
        ROOT / "results/Genetic_algorithm_runs/ga_summary_log.csv",
    ]:
        if p.exists(): return p
    raise FileNotFoundError("ga_summary_log.csv not found.")

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "fitness_score" not in df and "total" in df:
        df = df.rename(columns={"total": "fitness_score"})
    if "total_violations" not in df and {"capacity_viol", "elig_viol", "under_cap"}.issubset(df.columns):
        df["total_violations"] = df["capacity_viol"] + df["elig_viol"] + df["under_cap"]
    if "tag_label" not in df.columns:
        df["tag_label"] = (
            df.get("tag", "GA_Random")
            .astype(str)
            .str.replace("GA_", "GA ", regex=False)
            .str.replace("_", " ")
            .apply(lambda s: "GA Random" if s.strip().lower() in {"ga", "ga random"} else s)
        )

    for col in list(COLUMN_LABELS.keys()) + list(OP_LABELS.keys()):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def fmt_mean_std(s: pd.Series) -> str:
    return f"{s.mean():.2f} ± {s.std():.2f}" if not s.empty else "-"

def fmt_median_iqr(s: pd.Series, col_name: Optional[str] = None) -> str:
    if s.empty:
        return "-"
    q1, q2, q3 = s.quantile([0.25, 0.5, 0.75])
    if col_name in {"fitness_score", "pref_penalty", "total_violations", "runtime_sec"}:
        return f"{int(round(q2))}\n[{int(round(q1))}–{int(round(q3))}]"
    elif col_name in {"gini_satisfaction", "avg_rank"}:
        return f"{q2:.3f}\n[{q1:.3f}–{q3:.3f}]"
    elif col_name in {"top1_pct", "top3_pct"}:
        return f"{q2:.1f}\n[{q1:.1f}–{q3:.1f}]"
    else:
        return f"{q2:.2f}\n[{q1:.2f}–{q3:.2f}]"

def mean_std_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tag, group in df.groupby("tag_label"):
        row = {"Algorithm": tag}
        for col, meta in METRICS.items():
            if col in group:
                row[meta["label"]] = fmt_mean_std(group[col])
        rows.append(row)
    return pd.DataFrame(rows)

def median_iqr_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tag, group in df.groupby("tag_label"):
        row = {"Algorithm": tag}
        for col, meta in METRICS.items():
            if col in group:
                row[meta["label"]] = fmt_median_iqr(group[col], col_name=col)
        rows.append(row)
    return pd.DataFrame(rows)

def top_n_tables(df: pd.DataFrame, n=TOP_N) -> Dict[str, pd.DataFrame]:
    out = {}
    if "fitness_score" in df:     out["fitness"]    = df.nsmallest(n, "fitness_score")
    if "top1_pct" in df:          out["top1"]       = df.nlargest(n, "top1_pct")
    if "top3_pct" in df:          out["top3"]       = df.nlargest(n, "top3_pct")
    if "gini_satisfaction" in df: out["fairness"]   = df.nsmallest(n, "gini_satisfaction")
    if "total_violations" in df:  out["violations"] = df.nsmallest(n, "total_violations")
    return out

def best_worst_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, meta in METRICS.items():
        if col not in df: continue
        col_data = df[col].dropna()
        if col_data.empty: continue
        best = df.loc[col_data.idxmax() if meta["higher_better"] else col_data.idxmin()]
        worst = df.loc[col_data.idxmin() if meta["higher_better"] else col_data.idxmax()]
        for typ, r in zip(["Best", "Worst"], [best, worst]):
            rows.append({
                "Metric": meta["label"],
                "Type": typ,
                "run_id": r.get("run_id", ""),
                "seed": int(r["seed"]) if "seed" in r else "",
                "value": round(r[col], 3)
            })
    return pd.DataFrame(rows)

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=COLUMN_LABELS)

def df_to_md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False) if not df.empty else "_No data available._"

def main(top_n=TOP_N):
    summary_path = find_summary_csv()
    df = pd.read_csv(summary_path)
    df = ensure_columns(df)
    n_runs = len(df)

    summary_dir = ROOT / "results/summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    md_path = summary_dir / "ga_summary.md"

    tops = top_n_tables(df, top_n)
    mean_tbl = mean_std_table(df)
    iqr_tbl  = median_iqr_table(df)
    bw_tbl   = best_worst_rows(df)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Genetic Algorithm Summary Results\n\n")
        f.write(f"This summary aggregates **{n_runs}** Genetic Algorithm runs.\n\n")

        f.write("## Mean ± Standard Deviation by Algorithm\n\n")
        f.write(df_to_md(mean_tbl)); f.write("\n\n")

        f.write("## Median (Interquartile Range) by Algorithm\n\n")
        f.write(df_to_md(iqr_tbl)); f.write("\n\n")

        # Top-N sections with renamed columns
        if "fitness" in tops:
            f.write(f"### Top {top_n} Runs by Fitness (lower = better)\n\n")
            cols = ["run_id", "seed", "fitness_score", "top1_pct", "gini_satisfaction", "total_violations", "runtime_sec"]
            f.write(df_to_md(rename_columns(tops["fitness"][cols]))); f.write("\n\n")

        if "top1" in tops:
            f.write(f"### Top {top_n} Runs by Top-1 Preference Match Rate (%)\n\n")
            cols = ["run_id", "seed", "top1_pct", "fitness_score", "gini_satisfaction", "total_violations", "runtime_sec"]
            f.write(df_to_md(rename_columns(tops["top1"][cols]))); f.write("\n\n")

        if "top3" in tops:
            f.write(f"### Top {top_n} Runs by Top-3 Preference Match Rate (%)\n\n")
            cols = ["run_id", "seed", "top3_pct", "fitness_score", "gini_satisfaction", "total_violations", "runtime_sec"]
            f.write(df_to_md(rename_columns(tops["top3"][cols]))); f.write("\n\n")

        if "fairness" in tops:
            f.write(f"### Top {top_n} Runs by Fairness (Gini; lower = better)\n\n")
            cols = ["run_id", "seed", "gini_satisfaction", "fitness_score", "top1_pct", "total_violations", "runtime_sec"]
            f.write(df_to_md(rename_columns(tops["fairness"][cols]))); f.write("\n\n")

        if "violations" in tops:
            f.write(f"### Top {top_n} Runs by Violations (lower = better)\n\n")
            cols = ["run_id", "seed", "total_violations", "fitness_score", "top1_pct", "gini_satisfaction", "runtime_sec"]
            f.write(df_to_md(rename_columns(tops["violations"][cols]))); f.write("\n\n")

        if not bw_tbl.empty:
            f.write("## Best/Worst by Metric\n\n")
            f.write(df_to_md(bw_tbl)); f.write("\n\n")

        op_cols = [c for c in OP_LABELS if c in df.columns]
        if op_cols:
            f.write("## Operator Statistics (Mean ± Standard Deviation)\n\n")
            f.write("| Metric                  | Mean | Standard Deviation |\n")
            f.write("|-------------------------|------|-------------------|\n")
            for col in op_cols:
                mean = df[col].mean()
                std = df[col].std()
                f.write(f"| {OP_LABELS[col]} | {mean:.2f} | {std:.2f} |\n")
            f.write("\n")

        f.write("_Auto-generated from `ga_summary_log.csv`_\n")

    print(f"Markdown saved to: {md_path.resolve()}")

if __name__ == "__main__":
    main()
