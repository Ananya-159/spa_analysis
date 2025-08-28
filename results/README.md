### Results

This folder is **auto-generated** – all outputs from optimization runs, analysis scripts, second-round allocations, and plotting tools are written here.

### Subfolders and Files

`comparison.csv` - includes statistical comparison results across different algorithms

**`Genetic_algorithm_runs/`**  
Contains CSV logs and allocation files for GA batch runs (typically 30 seeds).  
Includes:
- `ga_summary_log.csv` → summary log across runs
- `convergence_*.csv` → per-generation logs
- `*_ga_alloc_*.csv` → allocation outputs

**`Hill_climbing_runs/`**  
Contains CSV logs and allocation files for Hill Climbing runs (typically 30 seeds).  
Includes analogous outputs to GA (summary log, convergence logs, allocations).

**`selector_hill_climbing_runs/`**  
Contains per-run outputs for the Selector-based Hill Climbing experiments (typically 30+ seeds).  
Includes:
- `*_selectorhc_convergence_*.csv` → iteration-wise fitness, Gini, violations
- `*_selectorhc_alloc_*.csv` → final student-project allocations
- `selectorhc_summary_log.csv` → aggregated summary across all Selector-HC runs  
Used by: `plot_selector_hc_run.py`, `plot_selector_hc_batch.py`, `analyze_selector_hc_stats.py`  
Plots saved in: `results/plots/selector_hill_climbing/`

**`Second_round_allocations/`**  
Contains per-student allocation CSVs and detailed logs after applying the **second round** procedure (HC swap-only mode if no free capacity).  
- `second_round_log.csv` → detailed run log with constraint violations, fitness, runtime, Gini satisfaction  
Complementary plots in: `results/plots/Second_round_hill_climbing/`  
Aggregated summaries in:
- `results/summary/second_round_summary_log.csv`
- `results/summary/second_round_stats.csv`

**`Second_round_hill_climbing/`**  
Driver outputs from running second round on selected HC runs (e.g., best + median by `avg_rank` or `total`).  
- Uses `second_round_driver.py` to automate multiple runs.  
- Across-run paired stats (Option B, aggregate per-run deltas) saved to:  
  - `results/summary/second_round_across_runs_summary.csv`  
  - `results/summary/second_round_across_runs_summary.md`

**`plots/`**  
Processed figures generated from batch analysis and second-round scripts.  
Organized into subfolders:
- `Genetic_algorithm/` → all GA plots  
- `Hill_climbing/` → all HC plots  
- `Second_round_hill_climbing/` → impact plots (Improved/Unchanged/Worsened, Top-1/Top-3 satisfaction, Assigned Rank Distribution, Average Preference Rank)  
- `selector_hill_climbing/` → all Selector-HC plots: fitness/Gini convergence, operator usage, satisfaction boxplots  
- `statistical_comparison/` → comparative plots across GA, HC, Selector-HC

**`summary/`**  
Markdown and CSV summaries aggregating performance across GA, HC, Selector-HC, and second-round.  
Includes:
- `ga_summary.md`, `ga_summary_log.csv`  
- `hc_summary.md`, `hc_summary_log.csv`  
- `selectorhc_summary_log.csv` → Selector-HC summary across runs  
- `second_round_summary_log.csv`, `second_round_stats.csv` → per-run, Option A stats (per-student paired tests)  
- `second_round_across_runs_summary.csv`, `.md` → aggregate per-run deltas  
- `statistical_comparison_tests.md` → across GA, HC, Selector-HC

**`smoke/`**  
Initial folder for quick smoke tests (short/debug runs).  
Safe to ignore in dissertation and final interpretation.


### Usage

Use this folder as the central location for **all experiment outputs**.  
For details, refer to the individual README files inside each subfolder.
