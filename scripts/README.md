### Scripts

This folder contains entry points and utilities for running experiments, plotting results, and generating summaries in the Student-Project Allocation (SPA) framework.

### Files

**`fairness_smoke_test.py`**  
Quick test script to validate fairness metrics (used only in early debugging).

**`plot_ga_batch.py`**  
Batch-level plots for Genetic Algorithm (GA) runs.

**`plot_ga_run.py`**  
Single-run plots for GA allocations.

**`plot_hc_batch.py`**  
Batch-level plots for Hill Climbing (HC) runs.

**`plot_hc_run.py`**  
Single-run plots for Hill Climbing allocations.

**`plot_selector_hc_batch.py`**  
Batch-level plots for Selector-based Hill Climbing (Selector-HC) runs: boxplots, violins, convergence curves, operator usage.

**`plot_selector_hc_run.py`**  
Single-run 3-panel plots (fitness, Gini, operator usage) for Selector-HC experiments.

**`statistical_comparison.py`**  
Statistical comparison of GA vs HC vs Selector-HC results (e.g., Wilcoxon/Welch tests).

**`second_round.py`**  
Applies the second-round allocation (HC-based reallocation of flagged students).  
Produces updated allocations, logs, plots, and statistics.

**`second_round_driver.py`**  
Automates second-round experiments across selected HC runs (best/median).  

**`smoke_core_utils_and_fitness.py`**  
Core helpers for smoke tests (allocation evaluation, fitness scoring).

**`smoke_scorer_minimal.py`**  
Minimal scorer for quick checks.

**`summary_generator_ga.py`**  
Builds GA summary logs across runs.

**`summary_generator_hc.py`**  
Builds HC summary logs across runs.

**`summary_generator_selector_hc.py`**  
Builds Selector-HC summary logs across multiple runs.

**`selector_hc_batch_runner.py`**  
Executes a batch of Selector-HC runs using different seeds (e.g., 100–130).

**`test_dataset_load.py`**  
Validates dataset loading (Excel/JSON/metadata).

**`test_fitness_one.py`**  
Simple unit test of the fitness function on a small dataset.

**`verify_alloc_vs_prefs.py`**  
Verifies that allocations respect submitted preferences.

**`utils/`**  
Legacy helper modules (e.g., `logger.py`) for Phase-2 smoke tests.  
*Not used in the final pipeline, kept for provenance.*

**`__init__.py`**  
Marks this folder as a Python package.

**README.md** - this file

### Usage / Notes

**Batch experiments**  
Run GA or HC via their respective main scripts in `/genetic_algorithm/` or `/hill_climbing/`.  
Summaries and plots are written to `/results/`.

**Selector-based Hill Climbing experiments**  
Use `selector_hc_batch_runner.py` for batch runs.  
Visualize results with `plot_selector_hc_run.py` and `plot_selector_hc_batch.py`.  
Generate aggregated summaries via `summary_generator_selector_hc.py`.

**Second-round policy experiments**  
Run `second_round.py` directly for one allocation,  
or `second_round_driver.py` to automate across multiple HC runs.

**Plots**  
Use `plot_ga_batch.py`, `plot_ga_run.py`, `plot_hc_batch.py`, `plot_hc_run.py`,  
or Selector-HC plotting scripts to regenerate visualizations in `/results/plots/`.

**Summaries**  
`summary_generator_ga.py`, `summary_generator_hc.py`, and `summary_generator_selector_hc.py`  
append new rows to `/results/summary/`.

**Statistical comparison**  
Run `statistical_comparison.py` to evaluate differences across GA, HC, and Selector-HC.

**Quick checks / debugging**  
`fairness_smoke_test.py`, `test_dataset_load.py`, and `test_fitness_one.py` can be run independently.  
Smoke utilities are provided in `/scripts/utils/` (kept for provenance only).

**Legacy note** – `scripts/utils` is provenance-only.
