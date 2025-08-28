 # scripts/analyze_hc_stats.py

"""
Statistical analysis for Hill Climbing (HC) batch results.

Core kept:
- Enhanced stability tests: Mann–Whitney U (BH/FDR), Cliff’s δ (+ magnitude, optional CI),
  group medians + bootstrap 95% CI, Hodges–Lehmann Δ + CI for TOST equivalence,
  practical thresholds flag
- Original stability tests (compact table): Welch’s t, MW p, Cohen’s d, rank-biserial r
- Descriptives (overall + by tag) with compact, human-friendly tables
- Feasibility rate + histogram (no outlines)
- Spearman heatmap + 3 key scatterplots 
- Top-5 tables
- Tee console to Markdown: results/summary/hc_analysis_output.md
- Save & show plots (Spyder Plots pane)
- Enhanced results saved BOTH as UTF-8-BOM CSV (Excel-safe) and ASCII-header XLSX 

"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import sys
from datetime import datetime
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # proxy legend handles

#  Config 
ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "results" / "Hill_climbing_runs"
CSV_PATH = RUNS_DIR / "hc_summary_log.csv"
PLOTS_DIR = ROOT / "results" / "plots" / "Hill_climbing"
SUMMARY_DIR = ROOT / "results" / "summary"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

# Show plots in Spyder’s Plots pane (save + show, don't close immediately)
SHOW_IN_SPYDER = True

# Plot style (publication-friendly)
sns.set(style="white")
sns.set_context("talk")
matplotlib.rcParams.update({
    "axes.titlesize": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})
COLOR = "#1f77b4"

# Metrics to analyze (keys must exist or be derivable)
# NOTE: Labels reflect current convention (lower better for fitness & gini here).
METRICS = [
    ("fitness_score", "Fitness (lower=better)"),
    ("avg_rank", "Average Assigned Preference Rank (lower=better)"),
    ("top1_pct", "Top-1 Match (%)"),
    ("top3_pct", "Top-3 Match (%)"),
    ("gini_satisfaction", "Gini (0=fair)"),
    ("total_violations", "Total Violations (lower=better)"),
    ("runtime_sec", "Runtime (sec)")
]

# Practical thresholds (Δ) for "meaningfulness"
# Absolute unless noted as relative.
PRACTICAL_DELTA = {
    "avg_rank": 0.15,
    "top1_pct": 2.0,                # percentage points
    "top3_pct": 3.0,                # percentage points
    "gini_satisfaction": 0.01,
    "fitness_score": ("rel", 0.01), # 1% relative improvement (lower is better)
    "total_violations": 1.0,
    "runtime_sec": ("rel", 0.05),   # 5% relative improvement
}

#  Tee logger 
class Tee:
    """Tee/stdout: write to console and to Markdown file (UTF-8-safe for Windows/Markdown)."""
    def __init__(self, out_path: Path):
        self.out_path = out_path
        self._file = None
        self._stdout = sys.stdout

    def __enter__(self):
        # Save file with BOM to support Windows Markdown viewers
        self._file = open(self.out_path, "w", encoding="utf-8-sig")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._file.write(f"# Hill Climbing Analysis Output\n\n**Timestamp:** {ts}\n\n**Source CSV:** `{CSV_PATH}`\n\n")
        sys.stdout = self
        return self

    def write(self, data):
        # Replace non-ASCII/special characters with safe ASCII equivalents
        safe_data = (data
                     .replace("±", "+/-")
                     .replace("→", "->")
                     .replace("–", "-")
                     .replace("“", '"')
                     .replace("”", '"')
                     .replace("’", "'")
                     .replace("•", "-"))
        self._stdout.write(safe_data)
        self._file.write(safe_data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._stdout
        if self._file:
            self._file.close()

# Plot helpers 
def compute_spearman(x, y):
    """Return (rho, p) for Spearman correlation on aligned, non-NaN pairs."""
    x = pd.Series(x).dropna()
    y = pd.Series(y).dropna()
    idx = x.index.intersection(y.index)
    if len(idx) >= 3:
        rho, p = stats.spearmanr(x.loc[idx], y.loc[idx])
        return float(rho), float(p)
    return (np.nan, np.nan)

def add_top_right_legend_and_stats(ax, rho, p, handles, gap=0.02, fontsize=11):
    """
    Put the legend in the upper-right, then place the Spearman box just below it
    (also upper-right aligned). Safe fallback if renderer fails.
    """
    leg = ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.9)
    y_text = 0.86  # fallback
    try:
        fig = ax.figure
        fig.canvas.draw()
        bbox = leg.get_window_extent(fig.canvas.get_renderer())
        inv = ax.transAxes.inverted()
        (_, y0), (_, y1) = inv.transform([[bbox.x0, bbox.y0], [bbox.x1, bbox.y1]])
        height = y1 - y0
        y_text = 0.98 - height - gap
    except Exception:
        pass
    if pd.notna(rho) and pd.notna(p):
        ax.text(
            0.98, y_text, f"Spearman ρ={rho:.2f}, p={p:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.85),
            fontsize=fontsize
        )

# Stats helpers 
def cohen_d_ind(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return np.nan
    sa2, sb2 = a.var(ddof=1), b.var(ddof=1)
    denom = (na + nb - 2)
    sp = np.sqrt(((na - 1) * sa2 + (nb - 1) * sb2) / denom) if denom > 0 else np.nan
    return (a.mean() - b.mean()) / sp if sp and sp > 0 else np.nan

def rank_biserial_from_U(U, n1, n2) -> float:
    return 1 - 2 * U / (n1 * n2)

def cliffs_delta(a, b) -> float:
    a = np.asarray(a); b = np.asarray(b)
    if len(a) == 0 or len(b) == 0: return np.nan
    gt = 0; lt = 0
    for x in a:
        gt += np.sum(x > b); lt += np.sum(x < b)
    n = len(a) * len(b)
    return np.nan if n == 0 else (gt - lt) / n

def cliffs_magnitude(delta: float) -> str:
    if pd.isna(delta): return "na"
    ad = abs(delta)
    return "negligible" if ad < 0.147 else "small" if ad < 0.33 else "medium" if ad < 0.474 else "large"

def bootstrap_ci_median(x, n_boot=10000, alpha=0.05, seed=42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    if len(x) == 0: return (np.nan, np.nan)
    boot = [np.median(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [100*alpha/2, 100*(1-alpha/2)])
    return float(lo), float(hi)

def bootstrap_cliffs_ci(a, b, n_boot=5000, alpha=0.05, seed=42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    if len(a) == 0 or len(b) == 0: return (np.nan, np.nan)
    a = np.asarray(a); b = np.asarray(b)
    vals = [cliffs_delta(rng.choice(a, len(a), True), rng.choice(b, len(b), True)) for _ in range(n_boot)]
    lo, hi = np.percentile(vals, [100*alpha/2, 100*(1-alpha/2)])
    return float(lo), float(hi)

def hodges_lehmann_diff(a, b) -> float:
    a = np.asarray(a); b = np.asarray(b)
    if len(a) == 0 or len(b) == 0: return np.nan
    diffs = []
    for x in a: diffs.extend(x - b)
    return float(np.median(diffs))

def bootstrap_ci_hl(a, b, n_boot=5000, alpha=0.05, seed=42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    a = np.asarray(a); b = np.asarray(b)
    if len(a) == 0 or len(b) == 0: return (np.nan, np.nan)
    vals = [hodges_lehmann_diff(rng.choice(a, len(a), True), rng.choice(b, len(b), True)) for _ in range(n_boot)]
    lo, hi = np.percentile(vals, [100*alpha/2, 100*(1-alpha/2)])
    return float(lo), float(hi)

def benjamini_hochberg(pvals: List[float]) -> List[float]:
    m = len(pvals)
    ranked = sorted(enumerate(pvals), key=lambda x: x[1])
    qvals = [0.0] * m; prev_q = 1.0
    for rank, (idx, p) in enumerate(ranked, start=1):
        q = min(prev_q, (p * m) / rank)
        qvals[idx] = q; prev_q = q
    return qvals

# Load & clean 
def clean_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    if "fitness_score" not in df.columns and "total" in df.columns:
        df = df.rename(columns={"total": "fitness_score"})
    if "total_violations" not in df.columns:
        parts = [c for c in ["capacity_viol", "elig_viol", "under_cap"] if c in df.columns]
        if parts:
            df["total_violations"] = df[parts].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    for c in ["fitness_score","avg_rank","top1_pct","top3_pct","gini_satisfaction","total_violations","runtime_sec"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    def _label(tag: str) -> str:
        s = str(tag or "")
        if "greedy" in s: return "HC Greedy"
        if "random" in s: return "HC Random"
        return s or "HC"
    df["tag_label"] = df["tag"].map(_label) if "tag" in df.columns else "HC"
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df

# Original stability (compact) 
def test_block(dfA, dfB, metric_key, metric_label, nameA, nameB):
    a = dfA[metric_key].dropna() if metric_key in dfA.columns else None
    b = dfB[metric_key].dropna() if metric_key in dfB.columns else None
    if a is None or b is None or len(a) < 2 or len(b) < 2: return None
    t_stat, t_p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    U, mw_p = stats.mannwhitneyu(a, b, alternative="two-sided")
    d = cohen_d_ind(a, b)
    r = rank_biserial_from_U(U, len(a), len(b))
    return {
        "Metric": metric_label,
        "Group A": nameA, "nA": len(a), "Mean A": np.mean(a),
        "Group B": nameB, "nB": len(b), "Mean B": np.mean(b),
        "Welch t": t_stat, "t p": t_p, "MW p": mw_p, "Cohen d": d, "Rank-biserial r": r
    }

def print_stability_compact(df: pd.DataFrame, title: str):
    """Readable, narrow output: one bullet per metric (no wide markdown table)."""
    print("\n" + title)
    print("-" * len(title))
    if df.empty:
        print("(no rows)")
        return

    def _short(lbl: str) -> str:
        if not isinstance(lbl, str):
            return str(lbl)
        return (lbl.replace("Top 10 (best fitness)", "Top-10")
                   .replace("Bottom 10 (worst fitness)", "Bottom-10")
                   .replace("First 15", "First-15")
                   .replace("Last 15", "Last-15")
                   .replace("Odd runs", "Odd")
                   .replace("Even runs", "Even"))

    for _, r in df.iterrows():
        metric = r["Metric"]
        a = _short(r["Group A"]); b = _short(r["Group B"])
        nA = int(r["nA"]); nB = int(r["nB"])
        mwp = float(r["MW p"])
        tval = float(r["Welch t"]); tp = float(r["t p"])
        d = float(r["Cohen d"]); rb = float(r["Rank-biserial r"])
        print(f"- {metric}: {a} (n={nA}) vs {b} (n={nB}) → "
              f"MW p={mwp:.3f}, t={tval:.3f} (p={tp:.3f}), d={d:.3f}, r={rb:.3f}")


def run_stability_tests(df: pd.DataFrame):
    # 1) First 15 vs Last 15
    if len(df) >= 30:
        first = df.iloc[:15]; last = df.iloc[-15:]
        rows = [r for (k,l) in METRICS if (r:=test_block(first,last,k,l,"First 15","Last 15"))]
        print_stability_compact(pd.DataFrame(rows), "Stability: First 15 vs Last 15")
    # 2) Odd vs Even
    odd = df.iloc[::2]; even = df.iloc[1::2]
    rows = [r for (k,l) in METRICS if (r:=test_block(odd,even,k,l,"Odd runs","Even runs"))]
    print_stability_compact(pd.DataFrame(rows), "Stability: Odd vs Even")
    # 3) Top 10 vs Bottom 10 by fitness
    if "fitness_score" in df.columns and len(df) >= 20:
        by_fit = df.sort_values("fitness_score")
        top10 = by_fit.iloc[:10]; bot10 = by_fit.iloc[-10:]
        rows = [r for (k,l) in METRICS if (r:=test_block(top10,bot10,k,l,"Top 10 (best fitness)","Bottom 10 (worst fitness)"))]
        print_stability_compact(pd.DataFrame(rows), "Contrast: Top 10 vs Bottom 10 (by fitness)")

# Enhanced stability tests with BH, δ, bootstrap medians, TOST 
@dataclass
class SplitSpec:
    nameA: str
    nameB: str
    dfA: pd.DataFrame
    dfB: pd.DataFrame

def oriented_value(metric_key: str, values: np.ndarray) -> np.ndarray:
    lower_better = {"fitness_score","avg_rank","gini_satisfaction","total_violations","runtime_sec"}
    return -np.asarray(values) if metric_key in lower_better else np.asarray(values)

def practical_threshold(metric_key: str, medA: float, medB: float) -> Tuple[float, bool, str]:
    delta = PRACTICAL_DELTA.get(metric_key, None)
    if delta is None or (pd.isna(medA) or pd.isna(medB)): return (np.nan, False, "Δ=na")
    lower_better = {"fitness_score","avg_rank","gini_satisfaction","total_violations","runtime_sec"}
    raw_change = (medA - medB) if metric_key in lower_better else (medB - medA)  # + if B better
    if isinstance(delta, tuple) and delta[0] == "rel":
        base = medA if medA != 0 else np.nan
        rel = (raw_change / base) if pd.notna(base) else np.nan
        thresh = delta[1]
        return (rel, bool(pd.notna(rel) and abs(rel) >= thresh), f"+/-{thresh:.2%} rel")
    else:
        thresh = float(delta)
        return (raw_change, abs(raw_change) >= thresh, f"+/-{thresh:g}")

def enhanced_split_results(splits: List[SplitSpec]) -> pd.DataFrame:
    rows = []
    for split in splits:
        for key, label in METRICS:
            a = split.dfA[key].dropna() if key in split.dfA.columns else None
            b = split.dfB[key].dropna() if key in split.dfB.columns else None
            if a is None or b is None or len(a) < 2 or len(b) < 2: continue
            # Test
            _, p_mw = stats.mannwhitneyu(a, b, alternative="two-sided")
            # Effect
            ao = oriented_value(key, a); bo = oriented_value(key, b)
            delta = cliffs_delta(ao, bo); delta_lo, delta_hi = bootstrap_cliffs_ci(ao, bo, n_boot=3000)
            # Medians + CI
            medA = float(np.median(a)); medB = float(np.median(b))
            medA_lo, medA_hi = bootstrap_ci_median(a, n_boot=5000); medB_lo, medB_hi = bootstrap_ci_median(b, n_boot=5000)
            # HL diff + CI (TOST)
            hl = hodges_lehmann_diff(a, b); hl_lo, hl_hi = bootstrap_ci_hl(a, b, n_boot=3000)
            # Practicality + Equivalence
            eff_val, is_prac, delta_note = practical_threshold(key, medA, medB)
            if isinstance(PRACTICAL_DELTA.get(key, None), tuple):
                base = medA if medA != 0 else np.nan
                if pd.isna(base): equiv = "Inconclusive"
                else:
                    rel_lo = hl_lo / base; rel_hi = hl_hi / base
                    bound = PRACTICAL_DELTA[key][1]
                    equiv = "Equivalent" if (rel_lo >= -bound and rel_hi <= bound) else "Not equivalent"
            else:
                bound = PRACTICAL_DELTA.get(key, np.nan)
                equiv = "Equivalent" if (pd.notna(hl_lo) and pd.notna(hl_hi) and (hl_lo >= -bound) and (hl_hi <= bound)) else "Not equivalent"
            favored = split.nameB if np.median(bo) > np.median(ao) else split.nameA
            rows.append({
                "Split": f"{split.nameA} vs {split.nameB}",
                "Metric": key, "Metric Label": label,
                "nA": len(a), "nB": len(b),
                "Median A": medA, "Median A CI": f"[{medA_lo:.3f}, {medA_hi:.3f}]",
                "Median B": medB, "Median B CI": f"[{medB_lo:.3f}, {medB_hi:.3f}]",
                "MW p": p_mw,
                "Cliff δ": delta, "δ CI": f"[{delta_lo:.3f}, {delta_hi:.3f}]", "δ Magnitude": cliffs_magnitude(delta),
                "HL Δ(A-B)": hl, "HL CI": f"[{hl_lo:.3f}, {hl_hi:.3f}]",
                "Equivalence (±Δ)": equiv, "Δ note": delta_note,
                "Practical Effect": eff_val, "Practically Meaningful?": "True" if is_prac else "False",
                "Favored Group": favored
            })
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["q (BH)"] = benjamini_hochberg(df["MW p"].tolist())
    df["Significant? (q<0.05)"] = df["q (BH)"].apply(lambda q: "True" if q < 0.05 else "False")
    return df

# Correlations & plots (Spearman only) 
def spearman_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rho = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for i in cols:
        for j in cols:
            x, y = df[i].dropna(), df[j].dropna()
            idx = x.index.intersection(y.index)
            r = stats.spearmanr(x.loc[idx], y.loc[idx])[0] if len(idx) >= 3 else np.nan
            rho.loc[i, j] = r
    return rho

def plot_heatmap(mat: pd.DataFrame, title: str, fname: str, vmin=-1, vmax=1):
    # Human-readable label mapping
    label_map = {
        "fitness_score": "Fitness Score",
        "runtime_sec": "Runtime (seconds)",
        "gini_satisfaction": "Gini Index",
        "total_violations": "Total Violations",
        "top1_pct": "Top-1 Preference Match Rate",
        "top3_pct": "Top-3 Preference Match Rate",
        "avg_rank": "Average Assigned Preference Rank"
    }

    # Apply mapping to row/col names
    mat = mat.rename(index=label_map, columns=label_map)

    plt.figure(figsize=(7.6, 6.0))
    ax = sns.heatmap(
    mat,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 9},  # <-- Smaller font size for annotations
    cmap="coolwarm_r",  
    vmin=vmin,
    vmax=vmax,
    center=0,
    cbar_kws={"shrink": 0.85}
)

    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    sns.despine()
    plt.tight_layout()
    out = PLOTS_DIR / fname
    plt.savefig(out, dpi=300)
    if SHOW_IN_SPYDER:
        plt.show(block=False)
    print(f"[SAVE] {out.name}")




def reg_scatter(df, xcol, ycol, title, fname):
    if xcol not in df.columns or ycol not in df.columns:
        return
    data = df[[xcol, ycol]].dropna()
    if len(data) < 3:
        return

    #  Human-readable axis labels
    label_map = {
    "fitness_score": "Fitness Score",
    "runtime_sec": "Runtime (seconds)",
    "gini_satisfaction": "Gini Index",
    "total_violations": "Total Violations",
    "top1_pct": "Top-1 Preference Match Rate",
    "top3_pct": "Top-3 Preference Match Rate",
    "avg_rank": "Average Assigned Preference Rank"
}

    x_label = label_map.get(xcol, xcol)
    y_label = label_map.get(ycol, ycol)

    plt.figure(figsize=(7.2, 5.6))
    ax = sns.regplot(
        data=data, x=xcol, y=ycol,
        scatter_kws={"s": 50, "alpha": 0.7, "color": COLOR, "edgecolors": "none", "linewidths": 0},
        line_kws={"color": "green", "lw": 2}, ci=None
    )
    ax.set_title(title, loc="center")  #  Title centered
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    rho, p = compute_spearman(data[xcol], data[ycol])
    scatter_proxy = Line2D([0], [0], marker='o', color='w', label='Runs',
                           markerfacecolor=COLOR, markersize=8, alpha=0.7)
    line_proxy = Line2D([0], [0], color='green', lw=2, label='Linear fit')
    add_top_right_legend_and_stats(ax, rho, p, handles=[scatter_proxy, line_proxy], gap=0.02, fontsize=11)

    sns.despine()
    plt.tight_layout()
    out = PLOTS_DIR / fname
    plt.savefig(out, dpi=300)
    if SHOW_IN_SPYDER:
        plt.show(block=False)
    print(f"[SAVE] {out.name}")



def run_correlations_and_plots(df: pd.DataFrame):
    cols = [c for c in ["fitness_score","gini_satisfaction","total_violations","runtime_sec","avg_rank","top1_pct","top3_pct"] if c in df.columns]
    if len(cols) < 2: return
    spear = spearman_matrix(df, cols)
    print("\nCorrelation (Spearman ρ) — Overall")
    print("-" * 32)
    print(tabulate(spear.round(3), headers="keys", tablefmt="github"))
    plot_heatmap(spear, "Hill Climbing Spearman Correlations (Overall, 30 runs)", "hc_corr_heatmap_overall.png")
    # Only 3 key scatters for lean output
    pairs = [
    (
        "fitness_score", "total_violations",
        "Fitness Score vs Total Violations — Hill Climbing",
        "scatter_fitness_vs_violations.png"
    ),
    (
        "fitness_score", "runtime_sec",
        "Fitness Score vs Runtime (seconds) — Hill Climbing",
        "scatter_fitness_vs_runtime.png"
    ),
    (
        "top1_pct", "gini_satisfaction",
        "Top-1 Preference Match Rate vs Fairness (Gini Index) — Hill Climbing",
        "scatter_top1_vs_gini.png"
    )
]


    for xcol, ycol, title, fname in pairs:
        if xcol in cols and ycol in cols:
            reg_scatter(df, xcol, ycol, title, fname)

# Convenience tables 
def print_header(title: str):
    print("\n" + title)
    print("-" * len(title))


def print_median_iqr(df, cols):
    out = []
    for c in cols:
        if c not in df.columns: continue
        x = df[c].dropna()
        if len(x) == 0: continue
        med = x.median(); q1, q3 = x.quantile([0.25, 0.75])
        out.append({"Metric": c, "Median": med, "IQR": f"[{q1:.3f}–{q3:.3f}]"})
    print_header("Hill Climbing — Median (IQR)")
    print(tabulate(pd.DataFrame(out), headers="keys", tablefmt="github", floatfmt=".3f") if out else "(no metrics)")

def format_mean_sd(x: pd.Series) -> str:
    return f"{x.mean():.3f} ± {x.std(ddof=1):.3f}"

def format_median_iqr(x: pd.Series) -> str:
    q1, q3 = x.quantile([0.25, 0.75])
    return f"{x.median():.3f} [{q1:.3f}–{q3:.3f}]"

def print_by_tag_compact(df: pd.DataFrame, metrics: List[str]):
    """Long-form, compact summary by tag to avoid ugly MultiIndex headers."""
    if "tag_label" not in df.columns:
        return
    rows = []
    for tag, sub in df.groupby("tag_label"):
        for m in metrics:
            if m not in sub.columns: continue
            x = sub[m].dropna()
            if len(x) == 0: continue
            rows.append({
                "tag_label": tag,
                "metric": m,
                "n": len(x),
                "mean ± sd": format_mean_sd(x),
                "median [IQR]": format_median_iqr(x),
                "min": float(np.min(x)),
                "max": float(np.max(x)),
            })
    if not rows:
        return
    tidy = pd.DataFrame(rows)
    print_header("HC — Descriptive Summary by Tag (HC Greedy vs HC Random)")
    print(tabulate(tidy, headers="keys", tablefmt="github", floatfmt=".3f"))

def print_top5_tables(df):
    def top5(by, asc=True, title=""):
        cols = [c for c in ["run_id","fitness_score","top1_pct","gini_satisfaction","total_violations","runtime_sec"] if c in df.columns]
        tbl = df.sort_values(by, ascending=asc).loc[:, cols].head(5)
        print_header(title); print(tabulate(tbl.round(3), headers="keys", tablefmt="github"))
    if "fitness_score" in df.columns: top5("fitness_score", True, "Top 5 Runs by Fitness (lower=better)")
    if "top1_pct" in df.columns: top5("top1_pct", False, "Top 5 Runs by Top 1 Preference (%)")
    if "gini_satisfaction" in df.columns: top5("gini_satisfaction", True, "Top 5 Runs by Fairness (Gini; lower=better)")
    if "total_violations" in df.columns: top5("total_violations", True, "Top 5 Runs by Violations (lower=better)")

#  Feasibility 
def feasibility_rate(df: pd.DataFrame):
    if "total_violations" not in df.columns:
        print_header("Feasibility"); print("total_violations not found; skipping."); return
    x = df["total_violations"].dropna()
    feasible = int((x == 0).sum()); total = int(len(x))
    rate = (feasible / total * 100) if total else 0.0
    print_header("Feasibility"); print(f"Feasible runs (zero violations): {feasible}/{total} ({rate:.1f}%)")
    plt.figure(figsize=(7.2, 5.2))
    ax = sns.histplot(x, bins=min(20, max(5, int(x.max()+1))), color=COLOR, edgecolor=None, linewidth=0)
    ax.set_title("Distribution of Total Violations — Hill Climbing"); ax.set_xlabel("Total Violations per run"); ax.set_ylabel("Count of runs")
    sns.despine(); plt.tight_layout()
    out = PLOTS_DIR / "hist_total_violations.png"
    plt.savefig(out, dpi=300)
    if SHOW_IN_SPYDER: plt.show(block=False)
    print(f"[SAVE] {out.name}")

# Save helpers (Excel/CSV, Excel-safe headers) 
def save_enhanced_tables(enh: pd.DataFrame):
    """
    Save enhanced results as:
    - CSV with UTF-8 BOM (Excel will detect encoding; no strange characters)
    - XLSX with ASCII headers and tick marks replaced (for guaranteed Windows Excel compatibility)
    """
    # CSV with BOM for Excel
    full_csv = RUNS_DIR / "enhanced_stability_results.csv"
    enh.to_csv(full_csv, index=False, encoding="utf-8-sig")
    print(f"(Full enhanced results saved: `{full_csv.name}`)")

    # Build an Excel-friendly copy (ASCII headers, no special symbols)
    enh_excel = enh.copy()
    tick_map = {"True": "Yes", "False": "No"}
    for col in ["Practically Meaningful?", "Significant? (q<0.05)"]:
        if col in enh_excel.columns:
            enh_excel[col] = enh_excel[col].map(tick_map).fillna(enh_excel[col])

    enh_excel = enh_excel.rename(columns={
        "Cliff δ": "Cliff_delta",
        "δ CI": "delta_CI",
        "HL Δ(A-B)": "HL_delta_A_minus_B",
        "HL CI": "HL_delta_CI",
        "Equivalence (±Δ)": "Equivalence_pm_Delta",
        "Δ note": "Delta_note",
    })

    # Replace ± in any object columns
    for col in enh_excel.columns:
        if enh_excel[col].dtype == object:
            enh_excel[col] = enh_excel[col].astype(str).replace("±", "+/-", regex=False)

    xlsx_out = RUNS_DIR / "enhanced_stability_results.xlsx"
    try:
        with pd.ExcelWriter(xlsx_out, engine="xlsxwriter") as writer:
            enh_excel.to_excel(writer, index=False)
    except Exception:
        # Fallback to openpyxl if xlsxwriter is not available
        with pd.ExcelWriter(xlsx_out, engine="openpyxl") as writer:
            enh_excel.to_excel(writer, index=False)
    print(f"(Excel-friendly copy saved: `{xlsx_out.name}`)")

#  Main 
def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"HC summary not found: {CSV_PATH}")

    # Ensure no stray blank figure from previous sessions
    plt.close('all')

    df = pd.read_csv(CSV_PATH)
    df = clean_and_prepare(df)

    with Tee(SUMMARY_DIR / "hc_analysis_output.md"):
        print("\n# Hill Climbing (HC) — Analysis Summary")

        # Descriptives
        desc_cols = [k for k,_ in METRICS if k in df.columns]
        desc = df[desc_cols].describe().T
        print("\nHC — Descriptive Summary (All Runs)")
        print("-" * 33)
        print(tabulate(desc.round(3), headers="keys", tablefmt="github"))

        # By tag (compact, long-form)
        print_by_tag_compact(df, desc_cols)

        # Median (IQR) + Top5
        print_median_iqr(df, desc_cols)
        print_top5_tables(df)

        # Original stability (now compact)
        run_stability_tests(df)

        # Enhanced stability tests
        print("\n## Enhanced Stability Tests (BH, δ, CIs, TOST, Practicality)")
        splits: List[SplitSpec] = []
        if len(df) >= 30:
            splits.append(SplitSpec("First 15", "Last 15", df.iloc[:15], df.iloc[-15:]))
        splits.append(SplitSpec("Odd runs", "Even runs", df.iloc[::2], df.iloc[1::2]))
        if "fitness_score" in df.columns and len(df) >= 20:
            by_fit = df.sort_values("fitness_score")
            splits.append(SplitSpec("Top 10 (best fitness)", "Bottom 10 (worst fitness)", by_fit.iloc[:10], by_fit.iloc[-10:]))

        enh = enhanced_split_results(splits)
        if enh.empty:
            print("- No enhanced results computed (insufficient data).")
        else:
            save_enhanced_tables(enh)
            for sname in enh["Split"].unique():
                sub = enh[enh["Split"] == sname].copy()
                sig = sub[sub["q (BH)"] < 0.05].sort_values("q (BH)")
                nsig = sub[sub["q (BH)"] >= 0.05]
                print(f"\n### {sname}")
                if len(sig):
                    bullets = []
                    for _, r in sig.iterrows():
                        bullets.append(
                            f"- **{r['Metric']}**: favors **{r['Favored Group']}**, "
                            f"δ={r['Cliff δ']:.2f} ({r['δ Magnitude']}), q={r['q (BH)']:.3f}, "
                            f"{r['Practically Meaningful?']}, {r['Equivalence (±Δ)']}"
                        )
                    print("\n".join(bullets))
                else:
                    print("- No BH-significant differences (q≥0.05).")
                eq_count = (sub["Equivalence (±Δ)"] == "Equivalent").sum()
                print(f"- Equivalence within practical bounds: **{eq_count}/{len(sub)}** metrics.")
                if len(nsig):
                    print(f"- Non-significant (q≥0.05): {', '.join(nsig['Metric'].tolist())}")

        # Correlations & plots
        run_correlations_and_plots(df)

        # Feasibility
        feasibility_rate(df)

        # Best/Worst by fitness
        if "fitness_score" in df.columns:
            best = df.nsmallest(1, "fitness_score").iloc[0]
            worst = df.nlargest(1, "fitness_score").iloc[0]
            print("\nBest/Worst Runs by Fitness")
            print("-" * 26)
            print(tabulate(
                [{
                    "Type":"Best", "run_id":best.get("run_id",""), "tag":best.get("tag_label",""),
                    "fitness":best["fitness_score"], "gini":best.get("gini_satisfaction",np.nan),
                    "viol":best.get("total_violations",np.nan), "runtime":best.get("runtime_sec",np.nan)
                },
                {
                    "Type":"Worst", "run_id":worst.get("run_id",""), "tag":worst.get("tag_label",""),
                    "fitness":worst["fitness_score"], "gini":worst.get("gini_satisfaction",np.nan),
                    "viol":worst.get("total_violations",np.nan), "runtime":worst.get("runtime_sec",np.nan)
                }],
                headers="keys", tablefmt="github", floatfmt=".3f"
            ))

if __name__ == "__main__":
    main()
