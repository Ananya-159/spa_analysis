"""SECOND-ROUND DRIVER (Automation over selected HC runs)

- Selects HC runs (best + median by a metric)
- Locates *_hc_alloc_<runid>.csv files
- Calls second_round.py with explicit output folders
- Aggregates across-runs (Option B) stats using per-run deltas

"""
from __future__ import annotations

import sys
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

# Project root & canonical locations
ROOT = Path(__file__).resolve().parents[1]
HC_DIR = ROOT / "results" / "Hill_climbing_runs"
SUMMARY_DIR = ROOT / "results" / "Second_round_allocations"
RUNS_DIR_OVERRIDE = ROOT / "results" / "Second_round_allocations"               # per-run allocation + detailed log
PLOTS_DIR_OVERRIDE = ROOT / "results" / "plots" / "Second_round_hill_climbing"  # plots (can be shared)
STATS_CSV = SUMMARY_DIR / "second_round_stats.csv"                               # appended by second_round.py
HC_LOG = HC_DIR / "hc_summary_log.csv"                                          # input for picking runs
SECOND_ROUND_SCRIPT = ROOT / "scripts" / "second_round.py"

# Selection policy
# Prefer avg_rank (satisfaction-facing); driver falls back to 'total' if missing.
SELECT_BY = "avg_rank"
TAGS_TO_INCLUDE = None   # e.g., {"phase3_hc_random","phase3_hc_greedy"} or None for all

PYTHON = sys.executable

def find_allocation_csv(run_id: str) -> Path | None:
    """Find per-student allocation CSV for a given HC run_id. Pattern: *hc_alloc*_<runid>.csv"""
    pattern = f"*hc_alloc*_{run_id}.csv"
    matches = sorted(HC_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None

def pick_runs(hc_df: pd.DataFrame, prefer_by: str = "avg_rank") -> list[dict]:
    """
    Select [median, best] rows (dicts) by prefer_by. Falls back to 'total' if prefer_by unavailable.
    De-duplicates by run_id. Optional tag filtering via TAGS_TO_INCLUDE.
    """
    by = prefer_by if prefer_by in hc_df.columns else "total"
    df2 = hc_df.copy()
    if TAGS_TO_INCLUDE:
        df2 = df2[df2["tag"].isin(TAGS_TO_INCLUDE)]
    if by not in df2.columns:
        raise ValueError(f"Column '{by}' not found in {HC_LOG.name}")
    df2 = df2.sort_values(by=by, ascending=True).reset_index(drop=True)
    if df2.empty:
        raise RuntimeError("No HC rows available after filtering.")
    best = df2.iloc[0].to_dict()
    median = df2.iloc[len(df2)//2].to_dict()

    selected, seen = [], set()
    for row in (median, best):
        rid = str(row.get("run_id", "")).strip()
        if rid and rid not in seen:
            selected.append(row); seen.add(rid)
    return selected

def run_second_round_on(csv_path: Path):
    """Invoke second_round.py with explicit output dirs."""
    cmd = [
        PYTHON, str(SECOND_ROUND_SCRIPT),
        "--alloc_csv", str(csv_path),
        "--runs_dir", str(RUNS_DIR_OVERRIDE),
        "--plots_dir", str(PLOTS_DIR_OVERRIDE),
        "--summary_dir", str(SUMMARY_DIR),
    ]
    print(f"[RUN] {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if res.returncode != 0:
        raise RuntimeError(f"second_round.py failed for {csv_path}")

def wilcoxon_across_runs(series: pd.Series) -> dict:
    """
    Wilcoxon signed-rank across runs on a delta series (median vs 0).
    Gracefully handle n<2 or all-zero deltas (p=1, W=0).
    """
    try:
        from scipy.stats import wilcoxon
        s = pd.to_numeric(series, errors="coerce").dropna()
        n = len(s)
        if n < 2:
            return {"wilcoxon_stat": np.nan, "wilcoxon_p": np.nan, "n_runs": n, "median": float(s.median() if n else np.nan)}
        if np.allclose(s.values, 0.0, atol=1e-12):
            return {"wilcoxon_stat": 0.0, "wilcoxon_p": 1.0, "n_runs": n, "median": 0.0}
        res = wilcoxon(s)  # test median of differences vs 0
        return {"wilcoxon_stat": float(res.statistic), "wilcoxon_p": float(res.pvalue), "n_runs": n, "median": float(s.median())}
    except Exception as e:
        print(f"[WARN] SciPy Wilcoxon unavailable/failed: {e}")
        s = pd.to_numeric(series, errors="coerce").dropna()
        return {"wilcoxon_stat": np.nan, "wilcoxon_p": np.nan, "n_runs": int(len(s)), "median": float(s.median() if len(s) else np.nan)}

def main():
    # 1) Loading the HC summary
    if not HC_LOG.exists():
        raise FileNotFoundError(f"HC summary log not found: {HC_LOG}")
    hc = pd.read_csv(HC_LOG)

    # 2) Pick Best + Median (by SELECT_BY, fallback handled inside)
    selected = pick_runs(hc, prefer_by=SELECT_BY)
    print(f"[INFO] Selected {len(selected)} runs (by '{SELECT_BY}'):")
    for row in selected:
        print(f"   - run_id={row.get('run_id','')}  tag={row.get('tag','')}  total={row.get('total','')}  avg_rank={row.get('avg_rank','')}")

    # 3) For each selected run, find allocation CSV and run second_round.py
    for row in selected:
        rid = str(row.get("run_id", "")).strip()
        csv_path = find_allocation_csv(rid)
        if not csv_path:
            raise FileNotFoundError(f"No allocation CSV found for run_id={rid} in {HC_DIR}")
        print(f"     allocation: {csv_path.relative_to(ROOT)}")
        run_second_round_on(csv_path)

    # 4) After runs, computing the across-runs stats from second_round_stats.csv
    if not STATS_CSV.exists():
        print(f"[WARN] No {STATS_CSV.name} produced yet; skipping Option B.")
        print(" Driver finished.")
        return

    stats = pd.read_csv(STATS_CSV)

    # Matching via source_hc_run_id (written by second_round.py)
    wanted_ids = {str(r.get("run_id","")).strip() for r in selected}
    if "source_hc_run_id" in stats.columns:
        sub = stats[stats["source_hc_run_id"].astype(str).isin(wanted_ids)].copy()
    else:
        # Fallback for older second_round.py that didn’t write source_hc_run_id
        sub = stats[stats["run_id"].astype(str).isin(wanted_ids)].copy()

    if len(sub) < 2:
        print(f"[WARN] Only {len(sub)} stats rows found for selected runs; Option B needs >= 2.")
        print(" Driver finished.")
        return

    # Two deltas to test across runs (median vs 0): Top-3 percentage points and median rank change
    w_top3 = wilcoxon_across_runs(sub["top3_delta_pp"])
    w_medr = wilcoxon_across_runs(sub["median_delta_rank"])

    print("\n[OPTION B] Across selected runs (paired deltas)")
    print(f"[Across] Top‑3 Δpp:  n={w_top3['n_runs']}, median={w_top3['median']:+.2f} pp, "
          f"Wilcoxon: W={w_top3['wilcoxon_stat']}, p={w_top3['wilcoxon_p']}")
    print(f"[Across] Median Δrank: n={w_medr['n_runs']}, median={w_medr['median']:+.2f}, "
          f"Wilcoxon: W={w_medr['wilcoxon_stat']}, p={w_medr['wilcoxon_p']}")

   

    # Save concise outputs
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    #  Save the across-run summary in results/summary
    out_csv = ROOT / "results" / "summary" / "second_round_across_runs_summary.csv"
    row_out = {
        "selected_by": SELECT_BY,
        "n_runs": w_top3["n_runs"],
        "top3_delta_median_pp": w_top3["median"],
        "top3_wilcoxon_stat": w_top3["wilcoxon_stat"],
        "top3_wilcoxon_p": w_top3["wilcoxon_p"],
        "median_delta_rank_median": w_medr["median"],
        "rank_wilcoxon_stat": w_medr["wilcoxon_stat"],
        "rank_wilcoxon_p": w_medr["wilcoxon_p"],
        "selected_run_ids": ",".join(sorted(wanted_ids)),
    }
    pd.DataFrame([row_out]).to_csv(out_csv, index=False)

    md = ROOT / "results" / "summary" / "second_round_across_runs_summary.md"
    md.write_text(
        "# Second‑Round: Across Selected Runs (Option B)\n\n"
        f"- Selected by: **{SELECT_BY}**\n"
        f"- n runs: **{w_top3['n_runs']}**\n"
        f"- Top‑3 Δpp: median **{w_top3['median']:+.2f}** pp; Wilcoxon W={w_top3['wilcoxon_stat']}, p={w_top3['wilcoxon_p']}\n"
        f"- Median Δrank: median **{w_medr['median']:+.2f}**; Wilcoxon W={w_medr['wilcoxon_stat']}, p={w_medr['wilcoxon_p']}\n"
        f"- Runs: `{','.join(sorted(wanted_ids))}`\n",
        encoding="utf-8"
    )
    print(f"[SAVE] Across‑runs summary saved: {out_csv.name}, {md.name}")
    print("\n Second‑round driver complete.")




if __name__ == "__main__":
    main()
