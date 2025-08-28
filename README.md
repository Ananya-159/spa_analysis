### Student Project Allocation (SPA)

Note:** This repository is provided **for Master’s dissertation submission only** at **The University of Manchester**.
> All rights reserved — no reuse, redistribution, or modification is permitted without the author’s explicit permission.  
> **License & Usage** – Refer to `LICENSE.md` and `CITATION.cff` for details.

This repository implements the **objective function + constraint model** and **metaheuristic approaches** (e.g., **Hill Climbing** baseline, **Genetic Algorithm**, **Selector Hill Climbing**) for SPA.

It also extends the framework with a **Second Round reallocation step** (policy layer on top of HC results).

The aim is to quantify trade-offs between:
- **Student preference** (preference rank satisfaction)
- **Capacity feasibility** (respecting project/supervisor limits)
- **Eligibility constraints**
- Optional reporting of **fairness metrics** (e.g., Gini of satisfaction)


### Repository Structure
spa_analysis/
├── core/ # Fitness function + penalties (preference, capacity, eligibility)
├── fairness/ # Fairness metrics (e.g., Gini, leximin helpers)
├── hill_climbing/ # HC operators and search driver (hc_main.py)
├── selector_hill_climbing/ # Adaptive HC using operator Choice Function (Swap, Reassign, Mut-Micro)
├── genetic_algorithm/ # GA operators and main driver
├── scripts/ # Experiment runners, plotting, summary generation
├── results/ # Logs, plots, summaries (auto-generated)
├── data/ # Synthetic SPA dataset (safe to publish)
├── data_generators/ # Script to generate synthetic data
├── eda_analysis/ # Exploratory data analysis
├── .gitignore
├── CITATION.cff # Citation metadata (restricted use)
├── LICENSE.md # License (All rights reserved – dissertation only)
├── README.md # Main documentation (this file)
├── requirements.txt # Python dependencies
├── environment.yml # Conda environment definition
├── config.json # Default project configuration
├── config_final_ga # Final GA configuration file
├── CONTRIBUTING.md # Policy on contributions (none allowed)
├── SECURITY.md # Security policy (academic use only)
└── init.py # Marks package root


### Optimisation Objective (minimise)

```python
total = gamma * sum(rank) + alpha * capacity_viol + beta * elig_viol

-Large alpha / beta prioritize feasibility (hard-ish constraints)
- gamma keeps preference as a soft cost
- See the dissertation for full definitions and references

** How to Reproduce Hill Climbing Experiments** 

1. Run Hill Climbing (30 runs: 15 random + 15 greedy)

bash
python -m hill_climbing.hc_main

Outputs:
- results/hc_summary_log.csv
- Per-run allocations: results/*_hc_alloc_<runid>.csv

2.Generate Analysis Plots

bash
python -m scripts.plot_convergence

Generates:
- Score / Fairness / Constraint violation plots
- Convergence curves over iterations

3. Export Markdown Summary

bash
python -m scripts.summary_generator_hc

Generates:
- results/summary/hc_summary.md (tables)

** Second Round Allocation (Policy Layer)**
The second round applies to selected HC allocations.
It reassigns only flagged students (those unsatisfied or eligible for swaps) using a swap-only hill climbing step.

Evaluated at two levels:

Option A – within a single run
Option B – across multiple runs

Outputs:
- results/Second_round_allocations/ → per-student allocations (*_second_round_alloc.csv)
- results/Second_round_allocations/ → detailed run logs (*_second_round_log.csv)
- results/plots/Second_round_hill_climbing/ → four plots per run:
  - Pie: Improved / Unchanged / Worsened
  - Top-1 / Top-3 satisfaction
  - Preference rank distribution (before vs after)
  - Average preference rank (cohort vs affected)
- results/summary/second_round_summary_log.csv → HC-style summary row
- results/summary/second_round_stats.csv → per-run paired tests (Wilcoxon, McNemar, ΔGini)

Statistical Evidence:

- Option A – paired tests per student (Wilcoxon, McNemar, ΔGini bootstrap CI)
- Option B – Wilcoxon on per-run deltas (Top-3%, median Δrank)

** How to Reproduce Second Round Results**

Option A: Single run (paired, per-student tests)

bash
python -m scripts.second_round --alloc_csv results/Hill_climbing_runs/<alloc_file>.csv

- Outputs plots, logs, stats to results/
- Console prints Wilcoxon, McNemar, ΔGini

Option B: Multiple runs (paired across runs)

bash
python -m scripts.second_round_driver

- Automatically selects best + median HC runs (by avg_rank, fallback total)
- Runs second_round.py on both
- Aggregates deltas and applies Wilcoxon

Outputs:
- results/summary/second_round_across_runs_summary.csv
- results/summary/second_round_across_runs_summary.md

** How to Reproduce Genetic Algorithm Experiments**

1. Run Genetic Algorithm (30 runs with different seeds)

bash
python -m genetic_algorithm.ga_main

Outputs:

- results/ga_summary_log.csv
- Per-run allocations: results/*_ga_alloc_<runid>.csv

2. Generate Analysis Plots

bash
python -m scripts.plot_ga_batch
python -m scripts.plot_ga_run

- Boxplots, histograms, convergence curves
- Fairness and satisfaction plots

3. Export Markdown Summary

bash
python -m scripts.summary_generator_ga
results/summary/ga_summary.md

** How to Reproduce Selector-based Hill Climbing Experiments**

1. Run Selector-HC (30 runs using different seeds)

bash
python -m selector_hc_main

Outputs:

- results/selector_hill_climbing_runs/selectorhc_summary_log.csv
- Per-run allocations: <timestamp>_selectorhc_alloc_<runid>.csv
- Per-run convergence logs: <timestamp>_selectorhc_convergence_<runid>.csv

2. Generate Analysis Plots

bash
python -m scripts.plot_selector_hc_batch
python -m scripts.plot_selector_hc_run

- Boxplots, violin plots, operator usage
- Mean ± SD convergence (fitness, fairness, violations)
- Histogram of preference ranks, Top-1/Top-3 plots
Saved to: results/plots/selector_hill_climbing/

3. Export Markdown Summary

bash
python -m scripts.summary_generator_selector_hc

- results/summary/selectorhc_summary.md
- Contains: mean ± std, medians, Top-N runs, operator stats

** Reproducibility / Environment Setup** 
To install dependencies, use one of:

Option A – Conda (recommended)
bash
conda env create -f environment.yml
conda activate spa_env
Full environment spec including Python version + pinned dependencies

Best for dissertation reproducibility

Option B – pip
bash
pip install -r requirements.txt
Lightweight alternative (core dependencies only)

See: results/summary/REPRODUCIBILITY.md for config, seeds, and figure reproducibility info.

**Notes**

- environment.yml → full Conda environment (source of truth)
- requirements.txt → fallback pip setup
- All versions are pinned to ensure reproducibility
- Optional libraries (statsmodels, scikit-posthocs) are for EDA only
- Prefer Option A unless otherwise required

**Optional Post-Analysis Step** 
For additional metric breakdowns or convergence trend plots, run:

# For Genetic Algorithm
python -m scripts.analyze_ga_stats

# For Hill Climbing
python -m scripts.analyze_hc_stats

# For Selector-based Hill Climbing
python -m scripts.analyze_selector_hc_stats

These scripts generate secondary plots and CSVs (e.g., operator usage trends, convergence statistics, violation breakdowns), stored in results/ and results/plots/.

# Statistical Comparison Across Algorithms** 
To generate cross-method plots and statistical test results comparing GA, HC, and Selector-HC:

python -m scripts.statistical_comparison

This will produce:
- Pairwise Wilcoxon/McNemar/ΔGini test results
- Boxplots, strip plots, and traffic-light heatmaps

Saved to:
- results/summary/statistical_comparison_tests.md
- results/plots/statistical_comparison/

Can use this for final insight into which algorithm performs best across key metrics.