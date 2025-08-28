"""SPA EDA (RAW DATA ONLY — no allocations)

- Loads Excel workbook (Students/Projects/Supervisors)
- Computes input-space summaries only (no *_alloc_*.csv usage)
- Plots : GPA hist, eligibility count (Blue=Not Eligible, Green=Eligible),
- Course box (mean ◆ + console stats), prefs-per-student (annotates if constant),
- Preference mentions (sorted), capacity dist, top-20 pref heatmap,
- Supervisor capacity, project type distribution (blue/green/yellow),
- Top-10 most preferred projects, GPA density (eligibility + project type),
- GPA by project type (mean ±95% CI with *capped* error bars + n in labels).
- Correlation matrix: auto-drops constants and annotates if “Prefs Listed” was dropped.
- Statistical tests (console + markdown):
   -Shapiro + Levene; Welch t-test (CI + Cohen’s d);
   -Mann–Whitney (rank-biserial r); ANOVA (η²) + optional Tukey;
   -Kruskal–Wallis + optional Dunn; χ² on pref-buckets (skips if df=0).
 - Outputs:
   plots   → eda_analysis/plots/
   report  → eda_analysis/summary/eda_summary.md"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas.core.groupby")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats  # statistical tests

# Optional post-hoc deps (guarded)
try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

try:
    import scikit_posthocs as sp
    HAS_SCPH = True
except Exception:
    HAS_SCPH = False



# Paths & Config


ROOT = Path(__file__).resolve().parents[1]
DATA_XLSX   = ROOT / "data" / "SPA_Dataset_With_Min_Max_Capacity.xlsx"
PLOTS_DIR   = ROOT / "eda_analysis" / "plots"
SUMMARY_DIR = ROOT / "eda_analysis" / "summary"

# Global SPA config (root). We only read "num_preferences" here.
GLOBAL_CONFIG = ROOT / "config.json"

# Local EDA-only config (optional). Allows tweaking of colors, formats, bins, etc.
EDA_CONFIG = ROOT / "eda_analysis" / "eda_config.json"


def load_global_num_prefs(config_path: Path) -> int:
    """Read num_preferences from the *root* config.json (fallback=6)."""
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        cfg = {}
    return int(cfg.get("num_preferences", 6))


def load_local_eda_cfg(config_path: Path) -> dict:
    """Load optional eda_analysis/eda_config.json. Returns defaults if missing."""
    default = {
        "random_seed": 42,
        "num_bins_average_hist": 10,
        "allocated_preference_simulation": {"distribution": [0.4, 0.25, 0.15, 0.10, 0.06, 0.04]},  # not used here
        "plot_colors": {
            "primary":   "#1f77b4",  # blue
            "secondary": "#2ca02c",  # green
            "tertiary":  "#ffcc00",  # yellow
        },
        "heatmap_cmap": "Blues",
        "output_formats": ["png", "pdf", "svg"],
    }
    try:
        user = json.loads(config_path.read_text(encoding="utf-8"))
        # shallow merge is fine for the keys
        default.update({k: user.get(k, default[k]) for k in default.keys()})
        if "plot_colors" in user:
            default["plot_colors"].update(user["plot_colors"])
        return default
    except Exception:
        return default



# Style (will be adjusted by local EDA config)


sns.set_theme(style="white")
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "legend.fontsize": 10,
    "axes.grid": False,
})

# These globals will be overwritten by EDA config in main()
COLOR_PRIMARY   = "#1f77b4"  # blue
COLOR_SECONDARY = "#2ca02c"  # green
COLOR_TERTIARY  = "#ffcc00"  # yellow
COLOR_RED       = "#d62728"
COLOR_GREY      = "#6c757d"
HEATMAP_CMAP    = "Blues"
OUTPUT_FORMATS  = ["png", "pdf", "svg"]
NUM_BINS_AVG    = 10

# Fixed mapping: blue = Not Eligible, green = Eligible
ELIGIBILITY_PALETTE = {
    "Not Eligible": COLOR_PRIMARY,
    "Not Eligible (False)": COLOR_PRIMARY,
    "Eligible": COLOR_SECONDARY,
    "Eligible (True)": COLOR_SECONDARY,
    "Unknown": COLOR_GREY,
}

# SHOW behavior (Spyder plots pane auto-captures on inline/Qt)
SHOW = os.environ.get("EDA_SHOW", "0") == "1"

def _is_inline_backend() -> bool:
    try:
        b = matplotlib.get_backend().lower()
        return ("inline" in b) or ("qt" in b)
    except Exception:
        return False

def save_fig(path_no_ext: Path, show: Optional[bool] = None):
    """Save with all formats requested in local EDA config."""
    if show is None:
        show = SHOW
    plt.tight_layout()
    for ext in OUTPUT_FORMATS:
        plt.savefig(f"{path_no_ext}.{ext}", bbox_inches="tight")
    if show or _is_inline_backend():
        plt.show()
    else:
        plt.close()


# Data loading


def load_workbook(xlsx_path: Path):
    xls = pd.ExcelFile(xlsx_path)
    students    = pd.read_excel(xls, "Students")
    projects    = pd.read_excel(xls, "Projects")
    supervisors = pd.read_excel(xls, "Supervisors")
    return students, projects, supervisors



# RAW-only helpers (summaries)


def preference_overlap_top3(students: pd.DataFrame, projects: pd.DataFrame) -> Dict:
    cols = ["Preference 1", "Preference 2", "Preference 3"]
    top3 = students[cols].values.flatten()
    uniq = {str(x) for x in top3 if pd.notna(x)}
    total = projects["Project ID"].astype(str).nunique()
    share = len(uniq) / total if total > 0 else np.nan
    return {"unique_top3_count": len(uniq), "total_projects": total, "coverage_share": share}

def dataset_summary_block(students, projects, supervisors, num_prefs) -> str:
    elig_pct = 100.0 * (students["Client Based Eligibility"] == True).mean()
    top3 = preference_overlap_top3(students, projects)
    proj_min = projects["Min Students"].min()
    proj_max = projects["Max Students"].max()
    sup_min  = supervisors["Min Student Capacity"].min() if "Min Student Capacity" in supervisors.columns else np.nan
    sup_max  = supervisors["Max Student Capacity"].max()
    return (
        "```\n"
        "Dataset Summary:\n\n"
        f"- Total Students: {len(students)}\n"
        f"- Total Projects: {len(projects)}\n"
        f"- Total Supervisors: {len(supervisors)}\n"
        f"- Mean GPA: {students['Average'].mean():.2f}\n"
        f"- Client Eligible Students: {elig_pct:.1f}%\n"
        f"- Preference overlap (top 3): {top3['coverage_share']*100:.1f}% of projects\n"
        f"- Project Min Capacity: {proj_min}\n"
        f"- Project Max Capacity: {proj_max}\n"
        f"- Supervisor Max Student Capacity: {sup_max}\n"
        f"- Supervisor Min Student Capacity: {sup_min}\n"
        "```\n"
    )

def top3_demand_table(students: pd.DataFrame, projects: pd.DataFrame) -> pd.DataFrame:
    cols = ["Preference 1", "Preference 2", "Preference 3"]
    counts = (
        pd.Series(students[cols].values.flatten())
        .dropna().astype(str).value_counts()
        .rename_axis("Project ID").reset_index(name="Top-3 Preference Count")
        .sort_values("Top-3 Preference Count", ascending=False)
    )
    out = counts.head(5).copy()
    if "Project Title" in projects.columns:
        title_map = projects.set_index("Project ID")["Project Title"].astype(str).to_dict()
        out["Project Title"] = out["Project ID"].map(title_map)
    return out



# Small helpers (ticks, printing)


def _int_y_ticks(ax): ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
def _int_x_ticks(ax): ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

def _print_and_collect(header: str, lines: List[str]) -> List[str]:
    print("\n[TEST] " + header)
    for ln in lines:
        for sub in str(ln).split("\n"):
            print("  " + sub)
    return [f"### {header}", *[f"- {ln}" for ln in lines]]


# Plots


def plot_average_hist(students: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(students["Average"], bins=NUM_BINS_AVG, color=COLOR_PRIMARY, edgecolor="white")
    ax.set_title("Distribution of Student Averages")
    ax.set_xlabel("Average (GPA)"); ax.set_ylabel("Number of Students")
    _int_y_ticks(ax)
    save_fig(PLOTS_DIR / "average_distribution")

def plot_client_eligibility(students: pd.DataFrame):
    """Blue=Not Eligible, Green=Eligible (fixed mapping)."""
    lab = students["Client Based Eligibility"].map({False: "Not Eligible", True: "Eligible"}).fillna("Unknown")
    df = pd.DataFrame({"Eligibility": lab})
    order = ["Not Eligible", "Eligible"] + (["Unknown"] if "Unknown" in df["Eligibility"].unique() else [])
    plt.figure(figsize=(10, 5))
    ax = sns.countplot(
        x="Eligibility", data=df, order=order,
        hue="Eligibility", palette=ELIGIBILITY_PALETTE, dodge=False, legend=False, width=0.6
    )
    ax.set_title("Client-Based Project Eligibility (Student Response)")
    ax.set_xlabel("Client-Based Eligibility (from Students sheet)"); ax.set_ylabel("Number of Students")
    _int_y_ticks(ax)
    save_fig(PLOTS_DIR / "client_eligibility_count")

def plot_course_box(students: pd.DataFrame):
    """Box per course + mean as ◆; console summary printed."""
    df = pd.melt(students[["Course 1", "Course 2", "Course 3"]], var_name="Course", value_name="Score")
    stats_tbl = df.groupby("Course")["Score"].agg(mean="mean", median="median", sd=lambda s: s.std(ddof=1), n="count").round(2)
    print("\n[CourseBox] Per-course summary (mean/median/sd/n):\n" + stats_tbl.to_string())
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(
        x="Course", y="Score", data=df, hue="Course", dodge=False, legend=False,
        palette={"Course 1": COLOR_PRIMARY, "Course 2": COLOR_SECONDARY, "Course 3": COLOR_TERTIARY}
    )
    means = stats_tbl["mean"].reindex(["Course 1", "Course 2", "Course 3"])
    for i, mu in enumerate(means.values): ax.scatter(i, mu, color="black", s=30, marker="D", zorder=3)
    ax.set_title("Score Distribution per Course"); ax.set_xlabel("Course"); ax.set_ylabel("Score")
    median_proxy = Line2D([0],[0], color="black", lw=1.5, label="Median (box line)")
    mean_proxy   = ax.scatter([],[], s=60, marker="D", c="black", label="Mean")
    ax.legend(handles=[median_proxy, mean_proxy], loc="upper right", frameon=True, scatterpoints=1)
    save_fig(PLOTS_DIR / "course_score_distribution")

def plot_preferences_per_student(students: pd.DataFrame, num_prefs: int, when_constant: str = "annotate"):
    st = students.copy()
    pref_cols = [f"Preference {i}" for i in range(1, num_prefs + 1)]
    st["Prefs Listed"] = st[pref_cols].notna().sum(axis=1)
    dist = st["Prefs Listed"].value_counts().sort_index()
    print("\n[PrefsPerStudent] Distribution of number of preferences listed:\n" + dist.to_string())
    uniq = st["Prefs Listed"].dropna().unique()
    only_val = int(uniq[0]) if len(uniq) == 1 else None
    is_constant = only_val is not None
    if is_constant and when_constant == "skip":
        print(f"[PrefsPerStudent] Skipped plot: all students listed {only_val} preferences.")
        return
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(st["Prefs Listed"], bins=np.arange(1, num_prefs + 2) - 0.5, color=COLOR_PRIMARY, edgecolor="white")
    ax.set_title("Preferences Listed per Student"); ax.set_xlabel("Number of Preferences"); ax.set_ylabel("Student Count")
    _int_y_ticks(ax)
    if is_constant:
        msg = "All students listed maximum preferences" if only_val == num_prefs else f"All students listed {only_val} preferences"
        ax.text(0.02, 0.95, msg, transform=ax.transAxes, ha="left", va="top", fontsize=10, color=COLOR_GREY)
    save_fig(PLOTS_DIR / "preferences_per_student")

def plot_preference_mentions(students: pd.DataFrame, pref_cols: List[str]):
    """Counts how many times each project is listed across all preference ranks (sorted desc)."""
    prefs_flat = students[pref_cols].values.flatten()
    counts = (pd.Series(prefs_flat).dropna().astype(str).value_counts()
              .rename_axis("Project ID").reset_index(name="Times Listed"))
    counts = counts.sort_values("Times Listed", ascending=False)

    plt.figure(figsize=(16, 6))
    ax = sns.barplot(data=counts, x="Project ID", y="Times Listed", color=COLOR_PRIMARY)
    ax.set_title("Total Number of Times Each Project Was Listed as a Preference", fontsize=16)
    ax.set_xlabel("Project ID", fontsize=16)
    ax.set_ylabel("Times Listed in Student Preferences", fontsize=16)
    ax.tick_params(axis="x", labelrotation=90, labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    _int_y_ticks(ax)
    save_fig(PLOTS_DIR / "preferences_by_project")

def plot_top10_preferred_projects(students: pd.DataFrame, pref_cols: List[str]):
    prefs_flat = students[pref_cols].values.flatten()
    counts = (pd.Series(prefs_flat).dropna().astype(str).value_counts()
              .rename_axis("Project ID").reset_index(name="Total Preference Mentions")
              .sort_values("Total Preference Mentions", ascending=False).head(10))
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=counts, x="Project ID", y="Total Preference Mentions", color=COLOR_PRIMARY)
    ax.set_title("Top 10 Most Preferred Projects (All Ranks Combined)")
    ax.set_xlabel("Project ID"); ax.set_ylabel("Total Preference Mentions")
    _int_y_ticks(ax)
    save_fig(PLOTS_DIR / "top10_preferred_projects")

def plot_project_capacity_distribution(projects: pd.DataFrame):
    min_counts = projects["Min Students"].value_counts().sort_index()
    max_counts = projects["Max Students"].value_counts().sort_index()
    x_vals = sorted(set(min_counts.index).union(set(max_counts.index)))
    width = 0.4
    plt.figure(figsize=(10, 6)); ax = plt.gca()
    ax.bar([x - width/2 for x in x_vals], [min_counts.get(x, 0) for x in x_vals], width=width, label="Min Capacity", color=COLOR_PRIMARY)
    ax.bar([x + width/2 for x in x_vals], [max_counts.get(x, 0) for x in x_vals], width=width, label="Max Capacity", color=COLOR_SECONDARY)
    ax.set_title("Project Capacity Distribution (Min vs Max)")
    ax.set_xlabel("Capacity (Number of Students)"); ax.set_ylabel("Number of Projects")
    ax.set_xticks(x_vals); ax.legend()
    _int_y_ticks(ax)
    save_fig(PLOTS_DIR / "project_capacity_distribution")

def plot_preference_heatmap(students: pd.DataFrame, projects: pd.DataFrame, num_prefs: int):
    heat = pd.DataFrame(0, index=projects["Project ID"].astype(str), columns=[f"Pref {i}" for i in range(1, num_prefs + 1)])
    for i in range(1, num_prefs + 1):
        counts = students[f"Preference {i}"].dropna().astype(str).value_counts()
        common = heat.index.intersection(counts.index)
        heat.loc[common, f"Pref {i}"] = counts.loc[common].astype(int)
    top3 = heat[["Pref 1", "Pref 2", "Pref 3"]].sum(axis=1)
    top20_ids = top3.sort_values(ascending=False).head(20).index
    heat20 = heat.loc[top20_ids]
    plt.figure(figsize=(14, 6))
    ax = sns.heatmap(heat20, cmap=HEATMAP_CMAP, linewidths=0.2, linecolor="white", cbar_kws={"label": "Count"})
    ax.set_title("Project Preference Heatmap (Counts by Rank) — Top-20 by Top-3 demand")
    ax.set_xlabel("Preference Rank"); ax.set_ylabel("Project ID")
    save_fig(PLOTS_DIR / "preference_heatmap")

def plot_supervisor_capacity(supervisors: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x="Supervisor Name", y="Max Student Capacity", data=supervisors, color=COLOR_PRIMARY)
    ax.set_title("Supervisor Maximum Student Capacity")
    ax.set_xlabel("Supervisor"); ax.set_ylabel("Max Students")
    ax.tick_params(axis="x", labelrotation=45)
    _int_y_ticks(ax)
    save_fig(PLOTS_DIR / "supervisor_capacity")

def plot_project_type_distribution(projects: pd.DataFrame):
    if "Project Type" not in projects.columns: return
    counts = projects["Project Type"].astype(str).value_counts()
    order = list(counts.index)
    palette = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TERTIARY][:len(order)]
    df = pd.DataFrame({"Project Type": order, "count": [counts[o] for o in order]})
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df, x="Project Type", y="count", hue="Project Type", palette=palette, dodge=False, legend=False)
    ax.set_title("Project Type Distribution"); ax.set_xlabel("Project Type"); ax.set_ylabel("Number of Projects")
    _int_y_ticks(ax)
    save_fig(PLOTS_DIR / "project_type_distribution")

def plot_correlation_matrix(students: pd.DataFrame, num_prefs: int):
    """Correlation heatmap. Drops constant columns; title annotates if 'Prefs Listed' was constant."""
    st = students.copy()
    st["Prefs Listed"] = st[[f"Preference {i}" for i in range(1, num_prefs + 1)]].notna().sum(axis=1)
    cols = ["Average", "Course 1", "Course 2", "Course 3", "Prefs Listed"]
    usable, dropped = [], []
    for c in cols:
        s = pd.to_numeric(st[c], errors="coerce")
        if s.nunique(dropna=True) > 1: usable.append(c)
        else: dropped.append(c)
    if len(usable) < 2:
        print("[Correlation] Skipped: fewer than 2 non-constant variables."); return
    df = st[usable].dropna()
    if df.empty:
        print("[Correlation] Skipped: no rows after dropping NaNs."); return
    corr = df.corr(method="pearson")
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", linewidths=0.4, linecolor="white",
                     cbar_kws={"label": "Correlation"})
    title = "Correlation Matrix (GPA & Courses)"
    if "Prefs Listed" in dropped:
        title += " — note: 'Prefs Listed' constant, dropped"
    ax.set_title(title)
    save_fig(PLOTS_DIR / "correlation_matrix")

def plot_gpa_density_by_eligibility(students: pd.DataFrame):
    df = students.copy()
    df["Eligibility"] = df["Client Based Eligibility"].map({True: "Eligible", False: "Not Eligible"})
    df = df.dropna(subset=["Average", "Eligibility"])
    if df.empty: return
    plt.figure(figsize=(12, 6))
    for lab in ["Eligible", "Not Eligible"]:
        sub = df[df["Eligibility"] == lab]["Average"]
        if len(sub) > 1:
            sns.kdeplot(sub, label=lab, fill=True, color=ELIGIBILITY_PALETTE.get(lab, COLOR_GREY), alpha=0.35)
    plt.title("GPA Density by Client Eligibility"); plt.xlabel("Average (GPA)"); plt.ylabel("Density")
    plt.legend(frameon=True, title=None)
    save_fig(PLOTS_DIR / "gpa_density_by_eligibility")

def plot_gpa_density_by_projecttype(students: pd.DataFrame, projects: pd.DataFrame):
    if "Project Type" not in projects.columns: return
    prefs = students[["Student ID", "Preference 1"]].dropna().rename(columns={"Preference 1": "Project ID"})
    merged = prefs.merge(projects[["Project ID", "Project Type"]], on="Project ID", how="left") \
                  .merge(students[["Student ID", "Average"]], on="Student ID", how="left") \
                  .dropna(subset=["Average", "Project Type"])
    if merged.empty or merged["Project Type"].nunique() < 2: return
    order = [x for x in ["Client based", "Research based", "Student sourced"] if x in merged["Project Type"].unique()]
    color_map = {"Client based": COLOR_PRIMARY, "Research based": COLOR_SECONDARY, "Student sourced": COLOR_TERTIARY}
    plt.figure(figsize=(14, 6))
    for ptype in order:
        sub = merged.loc[merged["Project Type"] == ptype, "Average"]
        if len(sub) > 1:
            sns.kdeplot(sub, label=ptype, fill=True, alpha=0.35, color=color_map[ptype])
    plt.title("GPA Density by First Preference Project Type"); plt.xlabel("Average (GPA)"); plt.ylabel("Density")
    plt.legend(frameon=True)
    save_fig(PLOTS_DIR / "gpa_density_by_projecttype")

def plot_gpa_by_projecttype_meanci(students: pd.DataFrame, projects: pd.DataFrame):
    """Mean ±95% CI by project type with *capped* error bars and (n=...) x‑labels."""
    if "Project Type" not in projects.columns: return
    prefs = students[["Student ID", "Preference 1"]].dropna().rename(columns={"Preference 1": "Project ID"})
    merged = prefs.merge(projects[["Project ID", "Project Type"]], on="Project ID", how="left") \
                  .merge(students[["Student ID", "Average"]], on="Student ID", how="left") \
                  .dropna(subset=["Average", "Project Type"])
    if merged.empty or merged["Project Type"].nunique() < 2: return

    g = merged.groupby("Project Type")["Average"]
    mean = g.mean(); std = g.std(ddof=1); n = g.count()
    se = std / np.sqrt(n); tcrit = stats.t.ppf(0.975, n - 1); h = se * tcrit
    lo = mean - h; hi = mean + h

    order  = [x for x in ["Client based", "Student sourced", "Research based"] if x in mean.index]
    labels = [f"{pt} (n={int(n[pt])})" for pt in order]
    colors = {"Client based": COLOR_PRIMARY, "Student sourced": COLOR_SECONDARY, "Research based": COLOR_TERTIARY}

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    xs = np.arange(len(order))
    for i, pt in enumerate(order):
        ax.errorbar(
            xs[i], mean[pt],
            yerr=[[mean[pt] - lo[pt]], [hi[pt] - mean[pt]]],
            fmt="o", color=colors[pt], ecolor=colors[pt],
            elinewidth=2, capsize=6, capthick=2, markersize=6,
        )
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_title("Mean GPA by First Preference Project Type (±95% CI)")
    ax.set_xlabel("Project Type (with sample size)"); ax.set_ylabel("Average (GPA)")
    save_fig(PLOTS_DIR / "gpa_by_projecttype_meanci")

def plot_gpa_by_client_eligibility(students: pd.DataFrame):
    """Average GPA by Client-Based Eligibility, clean bars (no CI)."""
    df = students.copy()
    df["Client Eligibility"] = df["Client Based Eligibility"].map({True: "Eligible", False: "Not Eligible"})
    df = df.dropna(subset=["Average", "Client Eligibility"])
    if df.empty: return
    palette = {"Eligible": COLOR_SECONDARY, "Not Eligible": COLOR_PRIMARY}
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(
        x="Client Eligibility", y="Average", data=df,
        hue="Client Eligibility", palette=palette, dodge=False,
        errorbar=None, legend=False
    )
    ax.set_title("Average GPA by Client-Based Project Eligibility")
    ax.set_xlabel("Client-Based Project Eligibility"); ax.set_ylabel("Average (GPA)")
    _int_y_ticks(ax)
    save_fig(PLOTS_DIR / "gpa_by_client_eligibility")



# Statistical helpers & tests


def _mean_ci(series: pd.Series, alpha: float = 0.05) -> Tuple[float, float, float]:
    x = series.dropna().astype(float).values
    n = len(x); mean = float(np.mean(x)) if n > 0 else np.nan
    if n < 2: return mean, np.nan, np.nan
    se = stats.sem(x); h = se * stats.t.ppf(1 - alpha/2, n - 1)
    return mean, mean - h, mean + h

def _cohen_d(x: pd.Series, y: pd.Series) -> float:
    x = x.dropna().astype(float).values; y = y.dropna().astype(float).values
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2: return np.nan
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    sp = np.sqrt(((nx - 1)*sx**2 + (ny - 1)*sy**2) / (nx + ny - 2))
    return 0.0 if sp == 0 else (x.mean() - y.mean()) / sp

def _eta_squared_anova(groups: List[np.ndarray]) -> float:
    all_vals = np.concatenate(groups); gm = np.mean(all_vals)
    ss_between = sum(len(g)*(np.mean(g) - gm)**2 for g in groups)
    ss_total = sum(((all_vals - gm)**2))
    return float(ss_between / ss_total) if ss_total > 0 else np.nan

def run_normality_and_levene(students: pd.DataFrame) -> List[str]:
    e1 = students[students["Client Based Eligibility"] == True]["Average"].dropna()
    e0 = students[students["Client Based Eligibility"] == False]["Average"].dropna()
    lines: List[str] = []
    if len(e1) >= 3:
        W1, p1 = stats.shapiro(e1); lines.append(f"Shapiro–Wilk (Eligible): W={W1:.3f}, p={p1:.4f}")
    else: lines.append("Shapiro–Wilk (Eligible): Not enough data.")
    if len(e0) >= 3:
        W0, p0 = stats.shapiro(e0); lines.append(f"Shapiro–Wilk (Not Eligible): W={W0:.3f}, p={p0:.4f}")
    else: lines.append("Shapiro–Wilk (Not Eligible): Not enough data.")
    if len(e1) >= 2 and len(e0) >= 2:
        L, pL = stats.levene(e1, e0, center="median")
        lines.append(f"Levene’s test (equal variances): W={L:.3f}, p={pL:.4f} → {'reject equal variances' if pL<0.05 else 'fail to reject'} at α=0.05")
    else:
        lines.append("Levene’s test: Not enough data.")
    return _print_and_collect("Normality (Shapiro) & Homogeneity (Levene)", lines)

def run_t_test_on_gpa_by_eligibility(students: pd.DataFrame) -> List[str]:
    e1 = students[students["Client Based Eligibility"] == True]["Average"].dropna()
    e0 = students[students["Client Based Eligibility"] == False]["Average"].dropna()
    if len(e1) < 2 or len(e0) < 2:
        return _print_and_collect("Welch’s t-test (Average GPA by Client Eligibility)", ["Not enough data."])
    s1, s0 = e1.var(ddof=1), e0.var(ddof=1); n1, n0 = len(e1), len(e0)
    t_stat, p = stats.ttest_ind(e1, e0, equal_var=False)
    df_num = (s1/n1 + s0/n0) ** 2
    df_den = (s1**2)/((n1**2)*(n1-1)) + (s0**2)/((n0**2)*(n0-1))
    df_welch = df_num / df_den if df_den > 0 else np.nan
    d = _cohen_d(e1, e0)
    m1, l1, u1 = _mean_ci(e1); m0, l0, u0 = _mean_ci(e0)
    interp = "SIGNIFICANT" if p < 0.05 else "not significant"
    return _print_and_collect(
        "Welch’s t-test (Average GPA by Client Eligibility)",
        [
            f"Group 1 (Eligible): n={n1}, mean={m1:.2f} (95% CI {l1:.2f}–{u1:.2f}), sd={e1.std(ddof=1):.2f}",
            f"Group 2 (Not Eligible): n={n0}, mean={m0:.2f} (95% CI {l0:.2f}–{u0:.2f}), sd={e0.std(ddof=1):.2f}",
            f"t={t_stat:.3f}, df≈{df_welch:.1f}, p={p:.4f}, Cohen’s d={d:.3f}",
            f"Interpretation: Difference is {interp} at α=0.05",
        ]
    )

def run_mannwhitney_on_gpa_by_eligibility(students: pd.DataFrame) -> List[str]:
    e1 = students[students["Client Based Eligibility"] == True]["Average"].dropna()
    e0 = students[students["Client Based Eligibility"] == False]["Average"].dropna()
    if len(e1) < 1 or len(e0) < 1:
        return _print_and_collect("Mann–Whitney U (Average GPA by Client Eligibility)", ["Not enough data."])
    u, p = stats.mannwhitneyu(e1, e0, alternative="two-sided")
    r = 1 - 2 * (u / (len(e1) * len(e0)))  # rank-biserial
    interp = "SIGNIFICANT" if p < 0.05 else "not significant"
    return _print_and_collect(
        "Mann–Whitney U (Average GPA by Client Eligibility)",
        [
            f"Group sizes: Eligible n={len(e1)}, Not Eligible n={len(e0)}",
            f"U={u:.3f}, p={p:.4f}, rank‑biserial r={r:.3f}",
            f"Interpretation: Difference is {interp} at α=0.05",
        ]
    )

def run_anova_on_gpa_by_first_pref_type(students: pd.DataFrame, projects: pd.DataFrame) -> List[str]:
    if "Project Type" not in projects.columns:
        return _print_and_collect("ANOVA (Average GPA by First Preference Project Type)", ["Skipped: 'Project Type' column not present."])
    prefs = students[["Student ID", "Preference 1"]].dropna().rename(columns={"Preference 1": "Project ID"})
    merged = prefs.merge(projects[["Project ID", "Project Type"]], on="Project ID", how="left") \
                  .merge(students[["Student ID", "Average"]], on="Student ID", how="left") \
                  .dropna(subset=["Average", "Project Type"])
    if merged["Project Type"].nunique() < 2:
        return _print_and_collect("ANOVA (Average GPA by First Preference Project Type)", ["Not enough groups."])
    groups = [g["Average"].values for _, g in merged.groupby("Project Type")]
    f_stat, p = stats.f_oneway(*groups)
    eta2 = _eta_squared_anova(groups)
    interp = "SIGNIFICANT" if p < 0.05 else "not significant"
    lines = [
        f"Groups: {', '.join(map(str, merged['Project Type'].unique()))}",
        f"F={f_stat:.3f}, p={p:.4f}, η²={eta2:.3f}",
        f"Interpretation: Difference is {interp} at α=0.05",
    ]
    if p < 0.05 and HAS_STATSMODELS:
        try:
            tuk = pairwise_tukeyhsd(endog=merged["Average"], groups=merged["Project Type"], alpha=0.05)
            lines.append("Tukey HSD:\n" + str(tuk.summary()))
        except Exception as e:
            lines.append(f"Tukey HSD: error: {e}")
    elif p < 0.05 and not HAS_STATSMODELS:
        lines.append("Tukey HSD: skipped (statsmodels not installed)")
    return _print_and_collect("ANOVA (Average GPA by First Preference Project Type)", lines)

def run_kruskal_and_dunn(students: pd.DataFrame, projects: pd.DataFrame) -> List[str]:
    if "Project Type" not in projects.columns:
        return _print_and_collect("Kruskal–Wallis (GPA by First Preference Project Type)", ["Skipped: 'Project Type' column not present."])
    prefs = students[["Student ID", "Preference 1"]].dropna().rename(columns={"Preference 1": "Project ID"})
    merged = prefs.merge(projects[["Project ID", "Project Type"]], on="Project ID", how="left") \
                  .merge(students[["Student ID", "Average"]], on="Student ID", how="left") \
                  .dropna(subset=["Average", "Project Type"])
    if merged["Project Type"].nunique() < 2:
        return _print_and_collect("Kruskal–Wallis (GPA by First Preference Project Type)", ["Not enough groups."])
    groups = [g["Average"].values for _, g in merged.groupby("Project Type")]
    H, p = stats.kruskal(*groups)
    lines = [
        f"Groups: {', '.join(map(str, merged['Project Type'].unique()))}",
        f"H={H:.3f}, p={p:.4f}",
        f"Interpretation: {'SIGNIFICANT' if p < 0.05 else 'not significant'} at α=0.05",
    ]
    if p < 0.05 and HAS_SCPH:
        try:
            dunn = sp.posthoc_dunn(merged, val_col="Average", group_col="Project Type", p_adjust="bonferroni")
            lines.append("Dunn’s post-hoc (Bonferroni):\n" + dunn.round(4).to_string())
        except Exception as e:
            lines.append(f"Dunn’s: error: {e}")
    elif p < 0.05 and not HAS_SCPH:
        lines.append("Dunn’s: skipped (scikit-posthocs not installed)")
    return _print_and_collect("Kruskal–Wallis (Average GPA by First Preference Project Type)", lines)

def run_chi2_prefcount_by_eligibility(students: pd.DataFrame, num_prefs: int) -> List[str]:
    st = students.copy()
    st["Prefs Listed"] = st[[f"Preference {i}" for i in range(1, num_prefs + 1)]].notna().sum(axis=1)
    st["Pref Bucket"] = pd.cut(st["Prefs Listed"], bins=[0, 3, 6], labels=["1–3", "4–6"], include_lowest=True)
    tab = pd.crosstab(st["Pref Bucket"], st["Client Based Eligibility"])
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        msg = "Skipped: contingency table has only one row/column (df=0) — all students listed the same pref count."
        return _print_and_collect("Chi-squared (Pref-count bucket × Client Eligibility)", [msg, f"Observed table:\n{tab.to_string()}"])
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    interp = "ASSOCIATED" if p < 0.05 else "not associated"
    return _print_and_collect(
        "Chi-squared (Pref-count bucket × Client Eligibility)",
        [f"Observed table:\n{tab.to_string()}", f"χ²={chi2:.3f}, df={dof}, p={p:.4f}", f"Interpretation: Variables are {interp} at α=0.05"]
    )


# Variable overview & optional extras


def variable_overview_tables(students: pd.DataFrame) -> str:
    df = students.copy(); rows = []
    for col in df.columns:
        s = df[col]; vtype = str(s.dtype); missing = int(s.isna().sum()); unique = int(s.nunique(dropna=True))
        row = {"Variable": col, "Type": vtype, "Missing": missing, "Uniques": unique}
        if pd.api.types.is_numeric_dtype(s):
            row.update({"Min": float(np.nanmin(s)), "Max": float(np.nanmax(s)),
                        "Mean": float(np.nanmean(s)), "SD": float(np.nanstd(s, ddof=1))})
        rows.append(row)
    tab = pd.DataFrame(rows)
    base_cols = ["Variable", "Type", "Missing", "Uniques", "Min", "Max", "Mean", "SD"]
    return tab.reindex(columns=base_cols).round(3).to_markdown(index=False)

def outlier_summary(students: pd.DataFrame) -> str:
    cols = [c for c in ["Average", "Course 1", "Course 2", "Course 3"] if c in students.columns]
    lines = ["```\nOutlier Summary (|z| > 3):"]
    for c in cols:
        s = pd.to_numeric(students[c], errors="coerce"); z = (s - s.mean()) / s.std(ddof=1)
        lines.append(f"- {c}: {int((z.abs() > 3).sum())}")
    lines.append("```"); return "\n".join(lines)

def correlation_pvalues_table(students: pd.DataFrame, num_prefs: int) -> Optional[pd.DataFrame]:
    st = students.copy()
    st["Prefs Listed"] = st[[f"Preference {i}" for i in range(1, num_prefs + 1)]].notna().sum(axis=1)
    cols = ["Average", "Course 1", "Course 2", "Course 3", "Prefs Listed"]
    usable = []
    for c in cols:
        s = pd.to_numeric(st[c], errors="coerce")
        if s.nunique(dropna=True) > 1: usable.append(c)
    if len(usable) < 2: return None
    df = st[usable].dropna()
    if df.empty: return None
    cols = list(df.columns)
    pmat = pd.DataFrame(np.nan, index=cols, columns=cols)
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            _, p = stats.pearsonr(df[cols[i]], df[cols[j]])
            pmat.iloc[i, j] = p
    return pmat



# Main


def main():
    print(f"[EDA] Matplotlib backend: {matplotlib.get_backend().lower()} | grid: OFF")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    # Loading the configs
    num_prefs = load_global_num_prefs(GLOBAL_CONFIG)
    eda_cfg   = load_local_eda_cfg(EDA_CONFIG)

    # Apply local EDA config overrides
    global COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TERTIARY, HEATMAP_CMAP, OUTPUT_FORMATS, NUM_BINS_AVG, ELIGIBILITY_PALETTE
    np.random.seed(int(eda_cfg.get("random_seed", 42)))
    NUM_BINS_AVG   = int(eda_cfg.get("num_bins_average_hist", 10))
    HEATMAP_CMAP   = eda_cfg.get("heatmap_cmap", "Blues")
    OUTPUT_FORMATS = list(eda_cfg.get("output_formats", ["png", "pdf", "svg"]))

    # Colors (and keep fixed eligibility convention)
    COLOR_PRIMARY   = eda_cfg["plot_colors"].get("primary", COLOR_PRIMARY)
    COLOR_SECONDARY = eda_cfg["plot_colors"].get("secondary", COLOR_SECONDARY)
    COLOR_TERTIARY  = eda_cfg["plot_colors"].get("tertiary", COLOR_TERTIARY)
    ELIGIBILITY_PALETTE = {
        "Not Eligible": COLOR_PRIMARY,
        "Not Eligible (False)": COLOR_PRIMARY,
        "Eligible": COLOR_SECONDARY,
        "Eligible (True)": COLOR_SECONDARY,
        "Unknown": COLOR_GREY,
    }

    # Loading the data
    students_df, projects_df, supervisors_df = load_workbook(DATA_XLSX)
    pref_cols = [f"Preference {i}" for i in range(1, num_prefs + 1)]

    # Markdown sections
    lines: List[str] = [
        "# EDA Summary (SPA Dataset — Raw Only)",
        dataset_summary_block(students_df, projects_df, supervisors_df, num_prefs),
        "## Descriptive Statistics (GPA & Courses)",
        students_df[["Average", "Course 1", "Course 2", "Course 3"]].describe(percentiles=[.25, .5, .75]).round(2).to_markdown(),
    ]

    top5 = top3_demand_table(students_df, projects_df)
    lines += ["", "## Top 5 Most In‑Demand Projects by Top‑3 Preferences", top5.to_markdown(index=False)]
    lines += ["", "## Variable Overview (Students)", variable_overview_tables(students_df)]
    lines += ["", "## Outlier Summary", outlier_summary(students_df)]

    # Plots
    plot_average_hist(students_df)
    plot_client_eligibility(students_df)
    plot_course_box(students_df)
    plot_preferences_per_student(students_df, num_prefs)
    plot_preference_mentions(students_df, pref_cols)
    plot_top10_preferred_projects(students_df, pref_cols)
    plot_project_capacity_distribution(projects_df)
    plot_preference_heatmap(students_df, projects_df, num_prefs)
    plot_supervisor_capacity(supervisors_df)
    plot_project_type_distribution(projects_df)
    plot_correlation_matrix(students_df, num_prefs)
    plot_gpa_by_client_eligibility(students_df)
    plot_gpa_density_by_eligibility(students_df)
    plot_gpa_by_projecttype_meanci(students_df, projects_df)
    plot_gpa_density_by_projecttype(students_df, projects_df)

    # Tests
    lines += ["", "## Statistical Tests (Raw)"]
    lines += run_normality_and_levene(students_df)
    lines += run_t_test_on_gpa_by_eligibility(students_df)
    lines += run_mannwhitney_on_gpa_by_eligibility(students_df)
    lines += run_anova_on_gpa_by_first_pref_type(students_df, projects_df)
    lines += run_kruskal_and_dunn(students_df, projects_df)
    lines += run_chi2_prefcount_by_eligibility(students_df, num_prefs)

    # Optional: correlation p-values
    ptab = correlation_pvalues_table(students_df, num_prefs)
    if ptab is not None:
        lines += ["", "## Correlation (Pearson) p‑values (upper triangle)", ptab.round(4).to_markdown()]

    SUMMARY_MD = SUMMARY_DIR / "eda_summary.md"
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[EDA] Plots saved to   : {PLOTS_DIR}")
    print(f"[EDA] Summary written to: {SUMMARY_MD}")


if __name__ == "__main__":
    main()
