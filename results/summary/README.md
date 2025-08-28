### Summary

This folder contains **auto-generated summaries** of GA, HC, Selector-based Hill Climbing (Selector-HC), and Second-Round Hill Climbing results.

### Files

- `hc_summary.md` → Markdown report with random and greedy initialisation across 30 HC runs each.
- `ga_summary.md` → Markdown report aggregating 30 GA runs — mean ± std, median (IQR), and Top-N runs.
- `smoke_summary_latest` → Markdown report for quick/debug runs.
- `second_round_across_runs_summary.csv` → Concise across-run Wilcoxon results (Option B: Across multiple runs, aggregate per-run deltas).
- `second_round_across_runs_summary.md` → Markdown version of across-run results.
- `selectorhc_summary.md` → Markdown summary of Selector-HC performance — mean ± std, medians, top-ranked runs, operator stats.
- `_analysis_output.md` → Detailed internal outputs and statistical analysis for GA, HC, and Selector-HC.
- `statistical_comparison_tests.md` → Markdown summary of statistical comparison for HC vs GA vs Selector-HC.
- `README.md` → this file.

### Usage

- Can use `second_round_across_runs_summary.csv` for **across-run (Option B: Across multiple runs, aggregate per-run deltas)** robustness checks.
- Compare `selectorhc_summary.md` with `ga_summary.md` and `hc_summary.md` for final internal understanding and evaluation of algorithm performance.
