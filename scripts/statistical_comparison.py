# scripts/statistical_comparison.py
"""
Cross-Algorithm Statistical Comparison (Hill Climbing vs Genetic Algorithm vs Selector Hill Climbing)

Purpose:
  Compare Hill Climbing, Genetic Algorithm, Selector Hill Climbing across all core metrics,
  run robust non-parametric tests, and produce clean, publication-ready plots + tables.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
from itertools import combinations
from datetime import datetime
import sys, glob
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import levene, kruskal
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Names 
NAME_HC  = "Hill Climbing"
NAME_GA  = "Genetic Algorithm"
NAME_SHC = "Selector Hill Climbing"

STACKED_LABEL = {
    NAME_HC:  "Hill\nClimbing",
    NAME_GA:  "Genetic\nAlgorithm",
    NAME_SHC: "Selector Hill\nClimbing",
}
def stacked(name: str) -> str:
    return STACKED_LABEL.get(name, name.replace(" ", "\n", 1))

#  Paths 
ROOT = Path(__file__).resolve().parents[1]
CSV_HC  = ROOT / "results" / "Hill_climbing_runs" / "hc_summary_log.csv"
CSV_GA  = ROOT / "results" / "Genetic_algorithm_runs" / "ga_summary_log.csv"
CSV_SHC = ROOT / "results" / "selector_hill_climbing_runs" / "selectorhc_summary_log.csv"

ALLOC_HC  = ROOT / "results" / "Hill_climbing_runs"
ALLOC_GA  = ROOT / "results" / "Genetic_algorithm_runs"
ALLOC_SHC = ROOT / "results" / "selector_hill_climbing_runs"

SUMMARY_DIR   = ROOT / "results" / "summary"
PLOTS_DIR     = ROOT / "results" / "plots" / "statistical_comparison"
COMPARISON_CSV= ROOT / "results" / "comparison.csv"
MD_LOG        = SUMMARY_DIR / "statistical_comparison_tests.md"

SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
SHOW_IN_SPYDER = True

# Styling 
sns.set_context("talk")
matplotlib.rcParams.update({
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})
TITLE_FONTSIZE = 13
LABEL_FONTSIZE = 11
TICK_FONTSIZE  = 9
LEGEND_FONTSIZE= 9

BLUE  = "#1f77b4"
GREEN = "#2ca02c"
YELLOW = "#ffcc00"
RED    = "#d62728"
ALGO_ORDER = [NAME_HC, NAME_GA, NAME_SHC]

#  Palette & Legend 
def algo_palette(algos: List[str]) -> Dict[str, str]:
    present = [a for a in ALGO_ORDER if a in set(algos)]
    if len(present) == 1:  return {present[0]: BLUE}
    if len(present) == 2:  return {present[0]: BLUE, present[1]: GREEN}
    return {present[0]: BLUE, present[1]: GREEN, present[2]: YELLOW}

def add_algo_legend(ax: plt.Axes, algos: List[str], palette: Dict[str, str]):
    handles = [Patch(facecolor=palette[a], edgecolor="none", label=a) for a in algos]
    leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1),
                    borderaxespad=0.0, frameon=True, framealpha=0.9)
    if leg:
        leg.set_title(None)
        for txt in leg.get_texts(): txt.set_fontsize(LEGEND_FONTSIZE)

#  Helpers 
def _format_median_number(val: float) -> str:
    if val is None or np.isnan(val): return "na"
    s = f"{val:.2f}".rstrip('0').rstrip('.')
    return s

def annotate_box_medians(ax: plt.Axes, data: pd.DataFrame, group_col: str, y_col: str,
                         pad_frac=0.015):
    if group_col not in data.columns or y_col not in data.columns: return
    groups = [g for g in pd.unique(data[group_col]) if pd.notna(g)]
    if not groups: return
    y_min, y_max = ax.get_ylim()
    pad = (y_max - y_min) * pad_frac if y_max > y_min else 0.05
    meds = data.groupby(group_col)[y_col].median()
    for i, g in enumerate(groups):
        if g not in meds or pd.isna(meds[g]): continue
        y = float(meds[g])
        ax.text(i, y + pad, f"median={_format_median_number(y)}",
                ha="center", va="bottom", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="black", alpha=0.85, pad=1.0))

def _save_tight(out: Path, rect=(0, 0, 1, 0.95)):
    sns.despine()
    plt.tight_layout(rect=rect)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    if SHOW_IN_SPYDER: plt.show(block=False)
    print(f"[PLOT] {out.name}")

#  Core metrics & stats 
# Metrics to analyze (lower is better for fitness/gini/violations; higher better for top1/top3)
METRICS = [
    ("fitness_score",     "Fitness Score (lower = better)",      "lower"),
    ("total_violations",  "Total Violations (lower = better)",   "lower"),
    ("avg_rank",          "Average Assigned Preference Rank (lower = better)", "lower"),
    ("gini_satisfaction", "Fairness — Gini Satisfaction (lower = fairer)",  "lower"),
    ("top1_pct",          "Top-1 Preference Match (%) (higher = better)", "higher"),
    ("top3_pct",          "Top-3 Preference Match (%) (higher = better)", "higher"),
    ("runtime_sec",       "Runtime (seconds, lower = better)",   "lower"),
]
PRACTICAL_DELTA = {
    "avg_rank": 0.15, "top1_pct": 2.0, "top3_pct": 3.0, "gini_satisfaction": 0.01,
    "fitness_score": ("rel", 0.01), "total_violations": 1.0, "runtime_sec": ("rel", 0.05),
}

class Tee:
    def __init__(self, out_path: Path):
        self.out_path = out_path
        self._file = None
        self._stdout = sys.stdout
    def __enter__(self):
        self._file = open(self.out_path, "w", encoding="utf-8")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = (f"# Statistical Comparison — {NAME_HC} vs {NAME_GA} vs {NAME_SHC}\n\n"
                  f"**Timestamp:** {ts}\n\n"
                  f"**Sources:**\n"
                  f"- `{CSV_HC}`\n- `{CSV_GA}`\n- `{CSV_SHC}`\n\n"
                  f"**Multiple Testing:** BH/FDR on 7 KW p-values; BH/FDR on 21 pairwise p-values\n\n")
        self._file.write(header)
        sys.stdout = self
        return self
    def write(self, data):
        self._stdout.write(data); self._file.write(data)
    def flush(self):
        self._stdout.flush(); self._file.flush()
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._stdout
        if self._file: self._file.close()

def benjamini_hochberg(pvals: List[float]) -> List[float]:
    m = len(pvals)
    ranked = sorted(enumerate(pvals), key=lambda x: (np.inf if pd.isna(x[1]) else x[1]))
    qvals = [np.nan] * m; prev_q = 1.0; rank = 0
    for idx, p in ranked:
        rank += 1
        q = np.nan if pd.isna(p) else min(prev_q, (p * m) / rank)
        prev_q = q if not pd.isna(q) else prev_q
        qvals[idx] = q
    return qvals

def epsilon_squared_kw(H: float, n_total: int, k_groups: int) -> float:
    if n_total <= k_groups or not np.isfinite(H): return np.nan
    return (H - k_groups + 1) / (n_total - k_groups)

def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a); b = np.asarray(b)
    if len(a) == 0 or len(b) == 0: return np.nan
    gt = 0; lt = 0
    for x in a: gt += np.sum(x > b); lt += np.sum(x < b)
    n = len(a) * len(b)
    return np.nan if n == 0 else (gt - lt) / n

def delta_magnitude(delta: float) -> str:
    if pd.isna(delta): return "na"
    ad = abs(delta)
    return "negligible" if ad < 0.147 else "small" if ad < 0.33 else "medium" if ad < 0.474 else "large"

def bootstrap_ci_median(x, n_boot=5000, alpha=0.05, seed=42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    if len(x) == 0: return (np.nan, np.nan)
    boot = [np.median(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [100*alpha/2, 100*(1-alpha/2)])
    return float(lo), float(hi)

def to_percent_if_needed(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce"); finite = s[np.isfinite(s)]
    return s * 100.0 if len(finite) and finite.max() <= 1.0 else s

#  Summary loading for core metrics 
def load_algo_df(path: Path, algo: str) -> pd.DataFrame:
    if not path.exists(): raise FileNotFoundError(f"Missing summary CSV: {path}")
    df = pd.read_csv(path)
    rename_map = {}
    if "fitness_score" not in df.columns and "total" in df.columns:
        rename_map["total"] = "fitness_score"
    if rename_map: df = df.rename(columns=rename_map)
    if "total_violations" not in df.columns:
        parts = [c for c in ["capacity_viol", "elig_viol", "under_cap"] if c in df.columns]
        if parts: df["total_violations"] = df[parts].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    for key, _, _ in METRICS:
        if key in df.columns: df[key] = pd.to_numeric(df[key], errors="coerce")
    if "top1_pct" in df.columns: df["top1_pct"] = to_percent_if_needed(df["top1_pct"])
    if "top3_pct" in df.columns: df["top3_pct"] = to_percent_if_needed(df["top3_pct"])
    df["algorithm"] = algo
    return df

def build_combined_df() -> pd.DataFrame:
    dfs = []
    if CSV_HC.exists():  dfs.append(load_algo_df(CSV_HC, NAME_HC))
    if CSV_GA.exists():  dfs.append(load_algo_df(CSV_GA, NAME_GA))
    if CSV_SHC.exists(): dfs.append(load_algo_df(CSV_SHC, NAME_SHC))
    if not dfs: raise RuntimeError("No input CSVs were found for Hill Climbing / Genetic Algorithm / Selector Hill Climbing.")
    return pd.concat(dfs, ignore_index=True)

def filter_to_n_per_algorithm(df: pd.DataFrame, n: int = 30, time_col: str = "timestamp") -> pd.DataFrame:
    if "algorithm" not in df.columns: return df
    df_sorted = df.sort_values(time_col) if time_col in df.columns else df.copy()
    return df_sorted.groupby("algorithm", group_keys=False).head(n).reset_index(drop=True)

#  Supervisor tests (variance & mean) 
def test_supervisor_load_variance(loads_run: pd.DataFrame):
    print("\n## Supervisor Load Variance Test (Levene’s)")
    groups = []
    for algo in ALGO_ORDER:
        loads = (loads_run[loads_run["algorithm"] == algo]
                 .groupby("run_id")["count"]
                 .apply(np.std, ddof=1)
                 .dropna().values)
        if len(loads) > 0:
            print(f"- {algo}: std across runs (n={len(loads)}): {np.mean(loads):.2f} ± {np.std(loads):.2f}")
            groups.append(loads)
    if len(groups) >= 2 and all(len(g) > 1 for g in groups):
        stat, p = levene(*groups)
        print(f"Levene’s Test: stat={stat:.3f}, p={p:.4g}")
    else:
        print("Insufficient data for Levene’s test.")

def test_supervisor_load_means(loads_run: pd.DataFrame):
    print("\n## Supervisor Load Comparison (KW on supervisor means)")
    groups = []
    for algo in ALGO_ORDER:
        means = (loads_run[loads_run["algorithm"] == algo]
                 .groupby("run_id")["count"]
                 .mean().dropna().values)
        print(f"- {algo}: mean per supervisor per run = {np.mean(means):.2f} ± {np.std(means):.2f} (n={len(means)})")
        groups.append(means)
    if len(groups) >= 2 and all(len(g) > 1 for g in groups):
        H, p = kruskal(*groups)
        print(f"KW Test: H={H:.3f}, p={p:.4g}")
    else:
        print("Insufficient data for KW test.")

#  Allocation loading (deterministic via summaries) 
def _summary_df(path: Path) -> pd.DataFrame:
    if not path.exists(): return pd.DataFrame()
    try:
        s = pd.read_csv(path)
        if "timestamp" in s.columns:
            s = s.sort_values("timestamp")
        if "run_id" not in s.columns and "runid" in s.columns:
            s = s.rename(columns={"runid": "run_id"})
        return s
    except Exception:
        return pd.DataFrame()

def _dominant_dataset_hash(s: pd.DataFrame) -> str | None:
    if "dataset_hash" not in s.columns or s.empty: return None
    return s["dataset_hash"].dropna().astype(str).value_counts().idxmax()

def _find_alloc_by_runid(algo_name: str, run_id: str) -> list[Path]:
    if algo_name == NAME_HC:
        pattern = f"*_*hc_alloc*{run_id}*.csv"; folder = ALLOC_HC
    elif algo_name == NAME_GA:
        pattern = f"*_*ga_alloc*{run_id}*.csv"; folder = ALLOC_GA
    else:
        pattern = f"*_*selectorhc_alloc*{run_id}*.csv"; folder = ALLOC_SHC
    return [Path(p) for p in glob.glob(str(folder / pattern))]

def _read_alloc_csv(path: Path, algo: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    cols = {c.lower(): c for c in df.columns}
    def get_col(*names):
        for n in names:
            if n in cols: return cols[n]
        return None
    col_sid  = get_col("student id", "student_id")
    col_proj = get_col("assigned project", "assigned_project")
    col_rank = get_col("matched preference rank", "assigned_rank")
    col_avg  = get_col("average")
    if col_proj is None or col_rank is None: return pd.DataFrame()
    out = pd.DataFrame({
        "Student ID": df[col_sid] if col_sid else np.nan,
        "Assigned Project": df[col_proj].astype(str).str.strip(),
        "Matched Preference Rank": pd.to_numeric(df[col_rank], errors="coerce"),
    })
    if col_avg: out["Average"] = pd.to_numeric(df[col_avg], errors="coerce")
    out["algorithm"] = algo
    out["run_id"] = path.stem
    return out

def load_all_allocations(n_per_algo: int = 30) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    # HC
    s = _summary_df(CSV_HC)
    if not s.empty:
        s = s.tail(max(n_per_algo, 1))
        dh = _dominant_dataset_hash(s)
        if dh is not None: s = s[s.get("dataset_hash", "").astype(str) == dh]
        if "alloc_csv" in s.columns:
            for fname in s["alloc_csv"].dropna().astype(str):
                p = ALLOC_HC / fname
                if p.exists(): frames.append(_read_alloc_csv(p, NAME_HC))
        else:
            for rid in s["run_id"].dropna().astype(str):
                for p in _find_alloc_by_runid(NAME_HC, rid):
                    frames.append(_read_alloc_csv(p, NAME_HC))
    else:
        for p in glob.glob(str(ALLOC_HC / "*_hc_alloc_*.csv")):
            frames.append(_read_alloc_csv(Path(p), NAME_HC))

    # GA
    s = _summary_df(CSV_GA)
    if not s.empty:
        s = s.tail(max(n_per_algo, 1))
        dh = _dominant_dataset_hash(s)
        if dh is not None: s = s[s.get("dataset_hash", "").astype(str) == dh]
        if "alloc_csv" in s.columns:
            for fname in s["alloc_csv"].dropna().astype(str):
                p = ALLOC_GA / fname
                if p.exists(): frames.append(_read_alloc_csv(p, NAME_GA))
        else:
            for rid in s["run_id"].dropna().astype(str):
                for p in _find_alloc_by_runid(NAME_GA, rid):
                    frames.append(_read_alloc_csv(p, NAME_GA))
    else:
        for p in glob.glob(str(ALLOC_GA / "*_ga_alloc_*.csv")):
            frames.append(_read_alloc_csv(Path(p), NAME_GA))

    # Selector-HC
    s = _summary_df(CSV_SHC)
    if not s.empty:
        s = s.tail(max(n_per_algo, 1))
        dh = _dominant_dataset_hash(s)
        if dh is not None: s = s[s.get("dataset_hash", "").astype(str) == dh]
        if "alloc_csv" in s.columns:
            for fname in s["alloc_csv"].dropna().astype(str):
                p = ALLOC_SHC / fname
                if p.exists(): frames.append(_read_alloc_csv(p, NAME_SHC))
        else:
            for rid in s["run_id"].dropna().astype(str):
                for p in _find_alloc_by_runid(NAME_SHC, rid):
                    frames.append(_read_alloc_csv(p, NAME_SHC))
    else:
        for p in glob.glob(str(ALLOC_SHC / "*_selectorhc_alloc_*.csv")):
            frames.append(_read_alloc_csv(Path(p), NAME_SHC))

    if not frames:
        return pd.DataFrame()

    out = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    if "Assigned Project" in out.columns:
        out["Assigned Project"] = out["Assigned Project"].astype(str).str.strip()
    if "Matched Preference Rank" in out.columns:
        out = out.dropna(subset=["Matched Preference Rank"])
    return out

#  Preference Match Success (+ Chi-square) 
def plot_pref_match_distribution(alloc: pd.DataFrame):
    if alloc.empty or "Matched Preference Rank" not in alloc.columns:
        print("[SKIP] Preference Match plot: no allocation data or rank column.")
        return

    data = alloc.copy()
    max_rank = int(np.nanmax(data["Matched Preference Rank"])) if len(data) else 0
    max_rank = min(max_rank if max_rank > 0 else 6, 10)
    ranks = list(range(1, max_rank + 1))

    dist = (data.groupby(["algorithm", "Matched Preference Rank"])
                 .size().reset_index(name="count"))
    tot = dist.groupby("algorithm")["count"].sum().rename("total")
    dist = dist.merge(tot, on="algorithm", how="left")
    dist["pct"] = dist["count"] / dist["total"] * 100.0

    all_idx = pd.MultiIndex.from_product([ALGO_ORDER, ranks], names=["algorithm", "Matched Preference Rank"])
    dist = dist.set_index(["algorithm", "Matched Preference Rank"]).reindex(all_idx, fill_value=0).reset_index()

    pal = algo_palette(ALGO_ORDER)
    plt.figure(figsize=(9.2, 5.8))
    ax = sns.barplot(data=dist, x="Matched Preference Rank", y="pct",
                     hue="algorithm", palette=[pal[a] for a in ALGO_ORDER])
    ax.set_title("Preference Match Success Rate by Algorithm", fontsize=TITLE_FONTSIZE, pad=10)
    ax.set_xlabel("Matched Preference Rank", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Percentage of Students (%)", fontsize=LABEL_FONTSIZE)
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True, framealpha=0.9, title=None)
    if leg:
        for txt in leg.get_texts(): txt.set_fontsize(LEGEND_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    _save_tight(PLOTS_DIR / "pref_match_distribution.png", rect=(0, 0, 0.82, 0.95))

def chi_square_pref_match(alloc: pd.DataFrame):
    if alloc.empty or "Matched Preference Rank" not in alloc.columns:
        print("[SKIP] Preference Match chi-square: no allocation data.")
        return
    print("\n## Preference Match Distribution — Chi-square Goodness-of-Fit (per algorithm)")
    for algo in ALGO_ORDER:
        sub = alloc[alloc["algorithm"] == algo]["Matched Preference Rank"].dropna().astype(int)
        if len(sub) < 5:
            print(f"- {algo}: insufficient data (n={len(sub)}).")
            continue
        vals, counts = np.unique(sub, return_counts=True)
        expected = np.ones_like(counts) * counts.mean()
        try:
            chi2, p = stats.chisquare(counts, f_exp=expected)
            print(f"- {algo}: χ²={chi2:.2f}, p={p:.4g} (n={len(sub)}, ranks={len(vals)})")
        except Exception as e:
            print(f"- {algo}: chi-square failed ({e}).")

#  Average vs Project (with Students fallback) 
def _students_average_map() -> pd.Series | None:
    for xlsx in ROOT.rglob("*.xlsx"):
        try:
            stu = pd.read_excel(xlsx, sheet_name="Students")
        except Exception:
            continue
        if {"Student ID", "Average"}.issubset(set(stu.columns)):
            s = stu.set_index("Student ID")["Average"]
            return pd.to_numeric(s, errors="coerce")
    return None

def plot_avg_vs_project(alloc: pd.DataFrame, top_k: int = 10):
    data = alloc.copy()
    if "Average" not in data.columns or data["Average"].isna().all():
        if "Student ID" in data.columns:
            avg_map = _students_average_map()
            if avg_map is not None:
                data["Average"] = pd.to_numeric(data["Student ID"], errors="coerce").map(avg_map)

    if "Average" not in data.columns or data["Average"].isna().all():
        print("[SKIP] Average vs Project: could not find/attach 'Average' (need Students sheet or column).")
        return

    top_projects = data["Assigned Project"].value_counts().head(top_k).index.tolist()
    subset = data[data["Assigned Project"].isin(top_projects)].copy()

    pal = algo_palette(ALGO_ORDER)
    # Wider figure based on number of projects; hide outliers for clarity
    fig_w = max(11.0, len(top_projects) * 1.2)
    plt.figure(figsize=(fig_w, 6.2))
    ax = sns.boxplot(data=subset, x="Assigned Project", y="Average",
                     hue="algorithm", palette=[pal[a] for a in ALGO_ORDER], showfliers=False)
    ax.set_title(f"Student Average vs Allocated Project (Top {len(top_projects)} projects) — by Algorithm", fontsize=TITLE_FONTSIZE, pad=10)
    ax.set_xlabel("Assigned Project", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Student Average", fontsize=LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True, framealpha=0.9, title=None)
    if leg:
        for txt in leg.get_texts(): txt.set_fontsize(LEGEND_FONTSIZE-1)
    plt.xticks(rotation=18, ha="right")
    _save_tight(PLOTS_DIR / "avg_vs_project.png", rect=(0, 0, 0.82, 0.95))


def kw_avg_across_projects(alloc: pd.DataFrame):
    data = alloc.copy()
    if "Average" not in data.columns or data["Average"].isna().all():
        if "Student ID" in data.columns:
            avg_map = _students_average_map()
            if avg_map is not None:
                data["Average"] = pd.to_numeric(data["Student ID"], errors="coerce").map(avg_map)

    if "Average" not in data.columns or data["Average"].isna().all():
        print("[SKIP] KW (Average across projects): could not find/attach 'Average'.")
        return

    print("\n## Average vs Project — Kruskal–Wallis across projects (per algorithm)")
    for algo in ALGO_ORDER:
        sub = data[data["algorithm"] == algo].copy()
        if sub.empty:
            print(f"- {algo}: no rows.")
            continue
        groups = [g["Average"].dropna().values for _, g in sub.groupby("Assigned Project")]
        groups = [arr for arr in groups if len(arr) >= 3]
        if len(groups) < 2:
            print(f"- {algo}: insufficient project groups for KW.")
            continue
        H, p = stats.kruskal(*groups)
        print(f"- {algo}: H={H:.2f}, p={p:.4g} (projects={len(groups)})")

#  Supervisor fairness (per-run mean±SD, Gini, KW, Levene/KW tests) 
def _discover_excel_with_projects() -> Dict[str, str] | None:
    for xlsx in ROOT.rglob("*.xlsx"):
        try:
            proj = pd.read_excel(xlsx, sheet_name="Projects")
            sup  = pd.read_excel(xlsx, sheet_name="Supervisors")
        except Exception:
            continue
        need_proj = {"Project ID", "Supervisor ID"}
        need_sup  = {"Supervisor ID"}
        if need_proj.issubset(set(proj.columns)) and need_sup.issubset(set(sup.columns)):
            sup_name_col = None
            for c in ["Supervisor Name", "Name", "Full Name"]:
                if c in sup.columns: sup_name_col = c; break
            if sup_name_col is None:
                sup_name_col = "Supervisor ID"
            proj = proj[["Project ID", "Supervisor ID"]].copy()
            sup  = sup[["Supervisor ID", sup_name_col]].copy()
            return {"proj": proj.to_json(orient="records"),
                    "sup": sup.to_json(orient="records"),
                    "name_col": sup_name_col}
    return None

def _build_supervisor_map() -> pd.DataFrame | None:
    meta = _discover_excel_with_projects()
    if meta is None: return None
    proj = pd.read_json(meta["proj"])
    sup  = pd.read_json(meta["sup"])
    name_col = meta["name_col"]
    m = proj.merge(sup, on="Supervisor ID", how="left")
    m = m.rename(columns={"Project ID": "Assigned Project", name_col: "Supervisor"})
    m["Assigned Project"] = m["Assigned Project"].astype(str).str.strip()
    return m[["Assigned Project", "Supervisor"]]

def gini(arr: np.ndarray) -> float:
    x = np.sort(np.asarray(arr, dtype=float))
    n = len(x)
    if n == 0: return np.nan
    if np.allclose(x, 0): return 0.0
    cumx = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

def supervisor_fairness_plots_and_stats(alloc: pd.DataFrame):
    sup_map = _build_supervisor_map()
    if alloc.empty or sup_map is None:
        print("[SKIP] Supervisor fairness: allocation data or project→supervisor map not available.")
        return

    df = alloc.merge(sup_map, on="Assigned Project", how="left")
    loads_run = (df.groupby(["algorithm", "run_id", "Supervisor"])
                   .size().reset_index(name="count"))

    loads_summary = (loads_run.groupby(["algorithm", "Supervisor"])["count"]
                              .agg(["mean", "std", "count"])
                              .reset_index())
    top_sup = (loads_summary.groupby("Supervisor")["mean"].sum()
                             .nlargest(12).index.tolist())

    pal = algo_palette(ALGO_ORDER)
    fig, ax = plt.subplots(figsize=(10, 6.2))
    width = 0.75 / max(1, len(ALGO_ORDER))
    x = np.arange(len(top_sup))

    for i, algo in enumerate(ALGO_ORDER):
        sub = loads_summary[(loads_summary["algorithm"] == algo) &
                            (loads_summary["Supervisor"].isin(top_sup))].copy()
        sub = sub.set_index("Supervisor").reindex(top_sup).reset_index()
        means = sub["mean"].fillna(0).values
        sds   = sub["std"].fillna(0).values
        ax.bar(x + i*width - (width*(len(ALGO_ORDER)-1)/2.0),
               means, width=width, color=pal[algo], edgecolor="none", label=algo)
        ax.errorbar(x + i*width - (width*(len(ALGO_ORDER)-1)/2.0),
                    means, yerr=sds, fmt="none", ecolor="black", elinewidth=1.0, capsize=2)

    ax.set_xticks(x)
    ax.set_xticklabels(top_sup, rotation=20, ha="right", fontsize=TICK_FONTSIZE)
    ax.set_xlabel("Supervisor", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Students Allocated (mean per run)", fontsize=LABEL_FONTSIZE)
    ax.set_title(f"Supervisor Workload — Top {len(top_sup)} Supervisors (mean per run ± SD)",
                 fontsize=TITLE_FONTSIZE, pad=10)

    # Force legend to use exact palette colors
    handles = [Patch(facecolor=pal[a], edgecolor="black", label=a) for a in ALGO_ORDER]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1),
              frameon=True, framealpha=0.9, fontsize=LEGEND_FONTSIZE, title=None)

    _save_tight(PLOTS_DIR / "supervisor_loads_top12.png", rect=(0, 0, 0.82, 0.95))


    # Gini(load) per algorithm: mean ± SD across runs
    print("\n## Supervisor Fairness — Gini(load) per algorithm (mean ± SD across runs)")
    for algo in ALGO_ORDER:
        glist = []
        for _, grp in loads_run[loads_run["algorithm"] == algo].groupby("run_id"):
            glist.append(gini(grp["count"].values))
        if glist:
            print(f"- {algo}: {np.mean(glist):.3f} ± {np.std(glist, ddof=1):.3f}  (runs={len(glist)})")
        else:
            print(f"- {algo}: no runs found.")

    # KW on rank distributions across supervisors (pooled)
    if "Matched Preference Rank" in df.columns:
        print("\n## Supervisor Satisfaction Variation — Kruskal–Wallis across supervisors (per algorithm)")
        for algo in ALGO_ORDER:
            sub = df[df["algorithm"] == algo]
            groups = [g["Matched Preference Rank"].dropna().values
                      for _, g in sub.groupby("Supervisor")]
            groups = [arr for arr in groups if len(arr) >= 5]
            if len(groups) < 2:
                print(f"- {algo}: insufficient supervisor groups for KW.")
            else:
                H, p = stats.kruskal(*groups)
                print(f"- {algo}: H={H:.2f}, p={p:.4g} (supervisors={len(groups)})")
    else:
        print("[SKIP] Supervisor satisfaction KW: no 'Matched Preference Rank' in allocations.")

    # Additional variance & mean tests across algorithms
    test_supervisor_load_variance(loads_run)
    test_supervisor_load_means(loads_run)

#  Plotting helpers with smaller fonts 
def _apply_stacked_xticklabels(ax: plt.Axes, algos: List[str]):
    ax.set_xticklabels([stacked(a) for a in algos], fontsize=TICK_FONTSIZE)

def plot_box(df: pd.DataFrame, y_key: str, title: str, y_label: str, out_name: str):
    data = df[["algorithm", y_key]].dropna()
    if data.empty:
        return

    algos = [a for a in ALGO_ORDER if a in data["algorithm"].unique()]
    pal = algo_palette(algos)

    plt.figure(figsize=(7.2, 5.4))
    ax = sns.boxplot(
        data=data, x="algorithm", y=y_key, order=algos,
        palette=[pal[a] for a in algos], showfliers=True
    )

    #  Annotate medians 
    annotate_box_medians(ax, data, "algorithm", y_key)

    # Force legend to use the correct colors (no mismatch) 
    handles = [Patch(facecolor=pal[a], edgecolor="black", label=a) for a in algos]
    leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1),
                    frameon=True, framealpha=0.9, fontsize=LEGEND_FONTSIZE)
    if leg:
        leg.set_title(None)

    # Cosmetics 
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=8)
    ax.set_xlabel("Algorithm", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.set_xticklabels([stacked(a) for a in algos], fontsize=TICK_FONTSIZE)

    _save_tight(PLOTS_DIR / out_name, rect=(0, 0, 0.82, 0.95))


def plot_strip_with_stats(df: pd.DataFrame, y_key: str, title: str, y_label: str, out_name: str):
    data = df[["algorithm", y_key]].dropna()
    if data.empty:
        return

    algos = [a for a in ALGO_ORDER if a in data["algorithm"].unique()]
    pal = algo_palette(algos)

    plt.figure(figsize=(8.0, 5.8))
    ax = sns.stripplot(
        data=data, x="algorithm", y=y_key, order=algos,
        palette=[pal[a] for a in algos], alpha=0.85, jitter=0.22, edgecolor="none"
    )

    # Overlay means and medians + build legend with actual values
    handles = []
    for i, a in enumerate(algos):
        vals = data.loc[data["algorithm"] == a, y_key].dropna().values
        if len(vals) == 0:
            continue

        m = float(np.mean(vals))
        md = float(np.median(vals))

        # Plot markers
        ax.scatter(i, m,  marker="D", s=70, color=pal[a], edgecolor="black", linewidths=0.6, zorder=5)
        ax.scatter(i, md, marker="s", s=70, color=pal[a], edgecolor="black", linewidths=0.6, zorder=5)

        # Add legend entries
        h_mean = Line2D([0],[0], marker='D', color='w',
                        label=f"{a} — mean={m:.2f}",
                        markerfacecolor=pal[a], markeredgecolor='black', markersize=7)
        h_median = Line2D([0],[0], marker='s', color='w',
                          label=f"{a} — median={md:.2f}",
                          markerfacecolor=pal[a], markeredgecolor='black', markersize=7)
        handles.extend([h_mean, h_median])

    # Manual legend with exact colors
    leg = ax.legend(handles=handles, title="Summary markers",
                    loc="upper left", bbox_to_anchor=(1.02, 1),
                    frameon=True, framealpha=0.95,
                    fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_FONTSIZE)
    if leg:
        for txt in leg.get_texts():
            txt.set_fontsize(LEGEND_FONTSIZE)

    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=10)
    ax.set_xlabel("Algorithm", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.set_xticklabels([stacked(a) for a in algos], fontsize=TICK_FONTSIZE)

    _save_tight(PLOTS_DIR / out_name, rect=(0, 0, 0.82, 0.95))



def plot_bar_runtime(df: pd.DataFrame, y_key: str, title: str, y_label: str, out_name: str):
    data = df[["algorithm", y_key]].dropna()
    if data.empty:
        return

    algos = [a for a in ALGO_ORDER if a in data["algorithm"].unique()]
    pal = algo_palette(algos)

    rows = []
    for a in algos:
        vals = data.loc[data["algorithm"] == a, y_key].dropna().values
        if len(vals) == 0:
            continue
        rows.append({"algorithm": a,
                     "mean": float(np.mean(vals)),
                     "sd": float(np.std(vals, ddof=1)),
                     "n": len(vals)})
    summ = pd.DataFrame(rows)
    if summ.empty:
        return

    #  plot 
    fig, ax = plt.subplots(figsize=(6.8, 5.2))  # slightly more compact
    x = np.arange(len(algos))
    width = 0.6

    for i, a in enumerate(algos):
        row = summ.loc[summ["algorithm"] == a]
        if row.empty:
            continue
        m = float(row["mean"].iloc[0])
        sd = float(row["sd"].iloc[0])
        n = int(row["n"].iloc[0])

        ax.bar(i, m, width=width, color=pal[a], edgecolor="none")
        ax.errorbar(i, m, yerr=sd, fmt="none", ecolor="black", elinewidth=1, capsize=3)

        ax.text(i, m, f"{m:.2f}", ha="center", va="bottom", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="black", alpha=0.85, pad=1.2))

    #  cosmetics 
    ax.set_xticks(x)
    ax.set_xticklabels([stacked(a) for a in algos], fontsize=TICK_FONTSIZE)
    ax.set_title(f"{title} (30 runs per algorithm)", fontsize=TITLE_FONTSIZE, pad=10)
    ax.set_xlabel("Algorithm", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=LABEL_FONTSIZE)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # reduce top white space

    out = PLOTS_DIR / out_name
    plt.savefig(out, dpi=300, bbox_inches="tight")
    if SHOW_IN_SPYDER:
        plt.show(block=False)
    print(f"[PLOT] {out.name}")


def plot_scatter_tradeoff(df: pd.DataFrame, x_key: str, y_key: str, title: str, xlab: str, ylab: str, out_name: str):
    data = df[["algorithm", x_key, y_key]].dropna()
    if data.empty: return
    algos = [a for a in ALGO_ORDER if a in data["algorithm"].unique()]
    pal = algo_palette(algos)
    plt.figure(figsize=(7.2, 5.4)); ax = plt.gca()
    for a in algos:
        sub = data[data["algorithm"] == a]
        ax.scatter(sub[x_key], sub[y_key], s=40, alpha=0.8, color=pal[a], label=a, edgecolors="none")
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=10)
    ax.set_xlabel(xlab, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylab, fontsize=LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    leg = ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    if leg:
        for txt in leg.get_texts(): txt.set_fontsize(LEGEND_FONTSIZE)
    _save_tight(PLOTS_DIR / out_name)


def plot_traffic_light_heatmap(comp_df: pd.DataFrame, out_name="heatmap_traffic_light.png"):
    if comp_df.empty:
        return

    def parse_med(cell: str) -> float:
        if cell == "na" or pd.isna(cell):
            return np.nan
        try:
            return float(str(cell).split(" ")[0])
        except Exception:
            return np.nan

    # Build matrix (rows=metrics, cols=algorithms)
    mat = []
    metric_names = []
    for _, row in comp_df.iterrows():
        metric_names.append(row["metric"])
        hc  = parse_med(row.get(NAME_HC,  "na"))
        ga  = parse_med(row.get(NAME_GA,  "na"))
        shc = parse_med(row.get(NAME_SHC, "na"))
        mat.append([hc, ga, shc])
    M = np.array(mat, dtype=float)

    # Rank each row: 0=best, 1=mid, 2=worst
    ranks = np.zeros_like(M)
    for i, (_, _, direction) in enumerate(METRICS):
        vals = M[i, :]
        if np.all(np.isnan(vals)):
            ranks[i, :] = np.nan
            continue
        order = np.argsort(vals) if direction == "lower" else np.argsort(-vals)
        rk = np.empty_like(order); rk[order] = np.arange(len(order))
        ranks[i, :] = rk

    # Colors for rank levels (use global constants)
    color_map = {0: GREEN, 1: YELLOW, 2: RED}
    colors = np.empty(ranks.shape, dtype=object)
    for i in range(colors.shape[0]):
        for j in range(colors.shape[1]):
            r = ranks[i, j]
            colors[i, j] = color_map.get(int(r), "#cccccc")

    #  Plot 
    plt.figure(figsize=(9.2, 6.2))   # a touch wider to leave legend space
    ax = plt.gca()

    for i in range(colors.shape[0]):
        for j in range(colors.shape[1]):
            rect = plt.Rectangle((j, i), 1, 1, facecolor=colors[i, j], edgecolor="white")
            ax.add_patch(rect)
            val = M[i, j]
            txt = _format_median_number(val) if not np.isnan(val) else "na"
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=10)

    # Axes cosmetics
    ax.set_xlim(0, 3)
    ax.set_ylim(0, len(metric_names))
    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_xticklabels([stacked(NAME_HC), stacked(NAME_GA), stacked(NAME_SHC)], fontsize=TICK_FONTSIZE)
    ax.set_yticks(np.arange(len(metric_names)) + 0.5)
    ax.set_yticklabels(metric_names, fontsize=TICK_FONTSIZE)
    ax.set_xlabel("Algorithm", fontsize=LABEL_FONTSIZE)
    ax.invert_yaxis()
    ax.set_title("Traffic Light Summary of Medians across Algorithms", fontsize=TITLE_FONTSIZE, pad=10)

    # Compact legend outside (right) with matching colors and smaller text
    legend_patches = [
        Patch(facecolor=GREEN,  edgecolor="black", label="Best"),
        Patch(facecolor=YELLOW, edgecolor="black", label="Mid"),
        Patch(facecolor=RED,    edgecolor="black", label="Worst"),
    ]
    leg = ax.legend(
        handles=legend_patches, title="Ranking",
        loc="center left", bbox_to_anchor=(1.03, 0.5),
        frameon=True, framealpha=0.9,
        fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_FONTSIZE,
        borderpad=0.3, labelspacing=0.25, handlelength=1.2, handletextpad=0.5
    )

    # Leave space on right for legend
    plt.tight_layout(rect=(0, 0, 0.88, 0.94))
    plt.savefig(PLOTS_DIR / out_name, dpi=300, bbox_inches="tight")
    if SHOW_IN_SPYDER:
        plt.show(block=False)
    print(f"[PLOT] {out_name}")



#  Core metric statistics engine 
@dataclass
class MetricResult:
    key: str; label: str; direction: str; groups: List[str]
    n_by_group: Dict[str, int]; median_by_group: Dict[str, float]
    iqr_by_group: Dict[str, Tuple[float, float]]; ci_by_group: Dict[str, Tuple[float, float]]
    kw_H: float; kw_p: float; kw_q: float; eps2: float
    pair_rows: List[Dict]; best: str; stat_sig: str

def run_statistics(df: pd.DataFrame) -> Tuple[List[MetricResult], pd.DataFrame]:
    algos_present = [a for a in ALGO_ORDER if a in df["algorithm"].unique()]
    if len(algos_present) < 2:
        raise RuntimeError(f"Need at least 2 algorithms to compare; got {algos_present}")

    kw_pvals = []
    per_metric_data = {}

    for key, label, direction in METRICS:
        sub = df[["algorithm", key]].dropna()
        if key not in sub.columns or sub[key].dropna().empty:
            groups = {a: 0 for a in algos_present}
            res = MetricResult(key, label, direction, algos_present, groups, {}, {}, {}, np.nan, np.nan, np.nan, np.nan, [], "", "No")
            per_metric_data[key] = {"sub": sub, "result": res, "groups_arrays": {}}
            kw_pvals.append(np.nan)
            continue

        groups_arrays = {a: sub.loc[sub["algorithm"] == a, key].dropna().values for a in algos_present}
        n_by_group = {a: len(groups_arrays[a]) for a in algos_present}

        sufficient = [a for a in algos_present if n_by_group[a] >= 5]
        all_arrays = [groups_arrays[a] for a in algos_present if len(groups_arrays[a]) > 0]
        has_variance = any(np.std(arr) > 0 for arr in all_arrays)

        if len(sufficient) < 2 or not has_variance:
            kw_H, kw_p = np.nan, np.nan
        else:
            kw_H, kw_p = stats.kruskal(*[groups_arrays[a] for a in algos_present if len(groups_arrays[a]) > 0])

        kw_pvals.append(kw_p)

        median_by_group = {}
        iqr_by_group = {}
        ci_by_group = {}
        for a in algos_present:
            vals = groups_arrays[a]
            if len(vals) > 0:
                median_by_group[a] = float(np.median(vals))
                q1, q3 = np.quantile(vals, [0.25, 0.75])
                iqr_by_group[a] = (float(q1), float(q3))
                ci_by_group[a] = bootstrap_ci_median(vals, n_boot=5000)
            else:
                median_by_group[a] = np.nan
                iqr_by_group[a] = (np.nan, np.nan)
                ci_by_group[a] = (np.nan, np.nan)

        if direction == "lower":
            best = min(median_by_group, key=lambda a: (np.inf if np.isnan(median_by_group[a]) else median_by_group[a]))
        else:
            best = max(median_by_group, key=lambda a: (-np.inf if np.isnan(median_by_group[a]) else median_by_group[a]))

        eps2 = epsilon_squared_kw(kw_H, int(np.sum([len(arr) for arr in all_arrays])), len(algos_present))

        res = MetricResult(key, label, direction, algos_present, n_by_group,
                           median_by_group, iqr_by_group, ci_by_group,
                           kw_H, kw_p, np.nan, eps2, [], best, "No")
        per_metric_data[key] = {"sub": sub, "result": res, "groups_arrays": groups_arrays}

    # BH adjust on KW p-values
    kw_qvals = benjamini_hochberg(kw_pvals)
    for (key, _, _), q in zip(METRICS, kw_qvals):
        per_metric_data[key]["result"].kw_q = q

    # Pairwise p-values across all metrics
    all_pairs = []; pair_index = []
    for key, label, direction in METRICS:
        groups_arrays = per_metric_data[key]["groups_arrays"]
        if not groups_arrays: continue
        res = per_metric_data[key]["result"]
        if res.kw_q is not None and not np.isnan(res.kw_q) and res.kw_q < 0.05:
            for a, b in combinations(res.groups, 2):
                A = groups_arrays[a]; B = groups_arrays[b]
                if len(A) < 2 or len(B) < 2: p = np.nan
                else: _, p = stats.mannwhitneyu(A, B, alternative="two-sided")
                all_pairs.append(p); pair_index.append((key, a, b))
        else:
            for a, b in combinations(res.groups, 2):
                all_pairs.append(np.nan); pair_index.append((key, a, b))

    q_pairs = benjamini_hochberg(all_pairs)

    # Fill pairwise rows with Cliff's delta (+ favors group B)
    for (key, a, b), p, q in zip(pair_index, all_pairs, q_pairs):
        res = per_metric_data[key]["result"]; arrs = per_metric_data[key]["groups_arrays"]
        if not arrs: continue
        A = arrs.get(a, np.array([])); B = arrs.get(b, np.array([]))
        if res.direction == "lower":
            Ao = -A; Bo = -B
        else:
            Ao = A; Bo = B
        dlt = np.nan if (len(Ao) == 0 or len(Bo) == 0) else cliffs_delta(Ao, Bo)
        favored = b if (not np.isnan(dlt) and dlt > 0) else a if (not np.isnan(dlt) and dlt < 0) else "tie"
        res.pair_rows.append({
            "a": a, "b": b, "U_p": p, "q_pair": q,
            "delta": dlt, "magnitude": delta_magnitude(dlt),
            "favored": favored
        })

    # Build comparison CSV & stat_sig
    rows_csv = []
    for key, label, direction in METRICS:
        res = per_metric_data[key]["result"]

        if res.kw_q is not None and not np.isnan(res.kw_q) and res.kw_q < 0.05:
            rivals = [a for a in res.groups if a != res.best]
            wins = 0; pair_summary_for_csv = []
            for pair in res.pair_rows:
                a, b = pair["a"], pair["b"]
                if res.best in (a, b):
                    opp = b if a == res.best else a
                    qpair = pair["q_pair"]; fav = pair["favored"]
                    dlt = pair["delta"]; mag = pair["magnitude"]
                    if (qpair is not None and not np.isnan(qpair) and qpair < 0.05 and fav == res.best):
                        wins += 1; pair_summary_for_csv.append(f"{res.best}>{opp} (q={qpair:.3f}, δ={dlt:.2f} {mag})")
                    else:
                        if qpair is not None and not np.isnan(qpair):
                            pair_summary_for_csv.append(f"{res.best} vs {opp} (q={qpair:.3f}, δ={dlt:.2f} {mag}, no win)")
                        else:
                            pair_summary_for_csv.append(f"{res.best} vs {opp} (q=na)")
            if wins >= len(rivals) and len(rivals) > 0: res.stat_sig = "Yes"
            elif wins > 0: res.stat_sig = "Partial"
            else: res.stat_sig = "No"
        else:
            pair_summary_for_csv = ["KW ns"]

        def fmt_med(a):
            med = res.median_by_group.get(a, np.nan)
            q1, q3 = res.iqr_by_group.get(a, (np.nan, np.nan))
            return f"{_format_median_number(med)} [{_format_median_number(q1)}–{_format_median_number(q3)}]" if not np.isnan(med) else "na"

        rows_csv.append({
            "metric": label, "metric_key": key, "direction": direction,
            NAME_HC:  fmt_med(NAME_HC),
            NAME_GA:  fmt_med(NAME_GA),
            NAME_SHC: fmt_med(NAME_SHC),
            "Best": res.best if res.best in res.groups else "na",
            "KW_q": f"{res.kw_q:.4f}" if not np.isnan(res.kw_q) else "na",
            "Stat_Sig": res.stat_sig,
        })

    comp_df = pd.DataFrame(rows_csv)
    return [per_metric_data[k]["result"] for k, _, _ in METRICS], comp_df

#  Main 
def main():
    plt.close('all')
    combo = build_combined_df()
    combo = filter_to_n_per_algorithm(combo, n=30)

    with Tee(MD_LOG):
        results, comp_df = run_statistics(combo)
        for res in results:
            print(f"\n## {res.label}")
            print(", ".join([f"{a}: n={res.n_by_group.get(a,0)}" for a in res.groups]))
            for a in res.groups:
                med = res.median_by_group.get(a, np.nan)
                q1,q3 = res.iqr_by_group.get(a,(np.nan,np.nan))
                ci_lo,ci_hi = res.ci_by_group.get(a,(np.nan,np.nan))
                print(f"- {a}: median={_format_median_number(med)}  IQR=[{_format_median_number(q1)}–{_format_median_number(q3)}]  95% CI=[{_format_median_number(ci_lo)}–{_format_median_number(ci_hi)}]")
            print(f"KW: H={res.kw_H:.3f}, p={res.kw_p:.4g}, q={res.kw_q:.4g}, ε²={res.eps2:.3f}")

        comp_df.to_csv(COMPARISON_CSV, index=False, encoding="utf-8-sig")
        print(f"\n[CSV] Saved: {COMPARISON_CSV}")

        # Allocation-based comparative plots + tests
        alloc = load_all_allocations()
        if not alloc.empty:
            print("\n[ALLOC] Loaded allocation records:", len(alloc))
            plot_pref_match_distribution(alloc); chi_square_pref_match(alloc)
            plot_avg_vs_project(alloc); kw_avg_across_projects(alloc)
            supervisor_fairness_plots_and_stats(alloc)
        else:
            print("\n[SKIP] No allocation CSVs found for comparative plots.")

    # Metric plots
    plot_box(combo,"fitness_score","Fitness Score by Algorithm (lower=better)","Fitness Score","box_fitness.png")
    plot_box(combo,"total_violations","Total Violations by Algorithm (lower=better)","Total Violations","box_violations.png")
    plot_box(combo,"avg_rank","Average Assigned Preference Rank by Algorithm (lower=better)","Average Preference Rank", "box_avg_rank.png")
    plot_box(combo,"gini_satisfaction","Fairness by Algorithm — Gini Index (lower=fairer)","Gini Index", "box_gini.png")
    plot_strip_with_stats(combo, "top1_pct", "Top-1 Preference Match Rate (%) by Algorithm",
                      "Top-1 Preference Match Rate (%)", "strip_top1.png")

    plot_strip_with_stats(combo, "top3_pct", "Top-3 Preference Match Rate (%) by Algorithm",
                      "Top-3 Preference Match Rate (%)", "strip_top3.png")

    plot_bar_runtime(combo,"runtime_sec","Runtime by Algorithm (lower=better)","Runtime (s)","bar_runtime.png")
    plot_scatter_tradeoff(combo,"gini_satisfaction","total_violations",
                      "Fairness vs Feasibility across Algorithms",
                      "Gini Index","Total Violations","scatter_gini_vs_violations.png")
    plot_traffic_light_heatmap(comp_df)

    print(f"\n[MD] Log saved to: {MD_LOG}")
    print(f"[DIR] Plots saved to: {PLOTS_DIR}")

if __name__ == "__main__":
    main()
