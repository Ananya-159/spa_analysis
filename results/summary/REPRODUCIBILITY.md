### Reproducibility Notes

These notes capture the exact snapshot and configuration used to generate the dissertation figures and tables.



### Repository Snapshot

- **Commit SHA:** <paste from `git rev-parse HEAD`>
- **Python:** `Python 3.12.4`
- **Environment:** `requirements.txt` (pinned) / `environment.yml`
- **Dataset:** `data/SPA_Dataset_With_Min_Max_Capacity.xlsx`



### Hill Climbing (HC)

- **Configs used:** `config.json`
- **Runs:** 30 runs (15 random + 15 greedy)  
  - Random seeds: 101–115  
  - Greedy seeds: 116–130
- **Scripts used:**
  - `hill_climbing/hc_main.py` (runs)
  - `scripts/plot_hc_batch.py` (batch plots)
  - `scripts/plot_hc_run.py` (single run plot)
  - `scripts/summary_generator_hc.py` (Markdown summary) 
  - `scripts/analyze_hc_stats.py` (internal statistical tests)
- **Outputs:**
  - Logs: `results/Hill_climbing_runs/`
  - Plots: `results/plots/Hill_climbing/`
  - Summary: `results/summary/hc_summary.md`
  - Analysis: `results/summary/hc_analysis_output.md`



### Genetic Algorithm (GA)

- **Configs used:** `config_final_ga.json`
- **Seeds / runs:** 30 runs across seeds 101–130.
- **Single‑run plot:** seed=999, `run_id=<auto: YYYYMMDDTHHMMSSZ>`
- **Scripts used:**
  - `genetic_algorithm/ga_main.py` (runs)
  - `scripts/plot_ga_batch.py` (batch plots)
  - `scripts/plot_ga_run.py` (single-run plot)
  - `scripts/summary_generator_ga.py` (Markdown + Top‑N)
  - `scripts/analyze_ga_stats.py` (internal statistical tests)
- **Outputs:**
  - Logs: `results/Genetic_algorithm_runs/`
  - Plots: `results/plots/Genetic_algorithm/`
  - Summary: `results/summary/ga_summary.md`
  - Analysis: `results/summary/ga_analysis_output.md`


### Selector-based Hill Climbing (Selector-HC)

- **Configs used:** `config_final_ga.json` 
- **Seeds / runs:** 30 runs across seeds 101–130.
- **Scripts used:**
  - `selector_hill_climbing/selector_hc_main.py` (runs)
  - `scripts/plot_selector_hc_batch.py` (batch plots)
  - `scripts/plot_selector_hc_run.py` (single run plot)
  - `scripts/summary_generator_selector_hc.py` (Markdown summary)
  - `scripts/analyze_selector_hc_stats.py` (internal statistical tests)
- **Outputs:**
  - Logs: `results/selector_hill_climbing_runs/`
  - Plots: `results/plots/selector_hill_climbing/`
  - Summary: `results/summary/selectorhc_summary.md`
  - Analysis: `results/summary/selectorhc_analysis_output.md`


### Second Round (Policy Layer on HC)

- **Mode:** swap‑only HC reallocation for flagged students (no new capacity)
- **Options evaluated:**
  - **Option A** (within one run): per‑student paired tests
  - **Option B** (across runs): per‑run deltas aggregated over selected HC runs (best + median)
- **Scripts used:**
  - `scripts/second_round.py` (Option A)
  - `scripts/second_round_driver.py` (Option B automation)
- **Outputs:**
  - Per‑student allocations: `results/Second_round_allocations/*_second_round_alloc.csv`
  - Per‑run logs: `results/Second_round_allocations/*_second_round_log.csv`
  - Impact plots: `results/plots/Second_round_hill_climbing/`
  - Summaries:
    - HC‑style: `results/summary/second_round_summary_log.csv`
    - Option A: `results/summary/second_round_stats.csv`
    - Option B: `results/summary/second_round_across_runs_summary.csv/.md`


### Statistical Comparison (GA vs HC vs Selector-HC)

- **Inputs used:**
  - `results/Genetic_algorithm_runs/ga_summary_log.csv`
  - `results/Hill_climbing_runs/hc_summary_log.csv`
  - `results/selector_hill_climbing_runs/selectorhc_summary_log.csv`
- **Scripts used:**
  - `scripts/statistical_comparison.py`
- **Outputs:**
  - Comparison table: `results/comparison.csv`
  - Markdown summary: `results/summary/statistical_comparison_tests.md`
  - Plots (from `results/plots/statistical_comparison/`):
    - `box_fitness.png`, `box_violations.png`, `box_avg_rank.png`, `box_gini.png`
    - `strip_top1.png`, `strip_top3.png`
    - `bar_runtime.png`, `scatter_gini_vs_violations.png`
    - `heatmap_traffic_light.png`, `pref_match_distribution.png`, `supervisor_loads_top12`


### Notes

- **run_id:** auto‑generated as `YYYYMMDDTHHMMSSZ`, appears in filenames of single‑run outputs  
- **Re‑create environment:** see root `README.md` for Conda/pip setup  
