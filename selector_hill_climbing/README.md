### Selector Hill Climbing (Selector-HC)

This folder contains the implementation of the Selector Hill Climbing (Selector-HC) algorithm for the Student Project Allocation (SPA) problem. It uses a Choice Function to adaptively select low-level hill climbing operators.

### Files

- `selector_hc_main.py` — main runner for a single Selector-HC experiment.
- `selector_controller.py` — Choice Function logic, operator updates, acceptance strategies.
- `README.md` — this file

### Usage

- Run an individual Selector-HC experiment:

```bash
python selector_hc_main.py

Output files are written to:

- results/selector_hill_climbing_runs/ → per-run allocations, logs, and convergence data.
- results/selector_hill_climbing_runs/selectorhc_summary_log.csv → aggregated run summary.

Visualize results using files in scripts/ folder:

- plot_selector_run.py → 3-panel convergence plot for one run.
- plot_selector_batch.py → full convergence, boxplots, violin plots

### Notes

- Uses 3 operators: swap, reassign, and mut_micro.
- The Choice Function updates operator weights based on recent performance.
- Designed to be modular for future hybridization or replacement of components.
- config_final_ga.json — configuration file shared with GA runs.