### Hill Climbing (HC)

This folder contains the implementation of the Hill Climbing algorithm for the Student–Project Allocation (SPA) problem.


### Files
- `hc_main.py` — entry point for running Hill Climbing (batch or single runs).
- `operators.py` — neighbourhood operators (reassign, swap, greedy repair) used to generate candidate solutions during local search.

- `__init__.py` — marks this folder as a package.
- `README.md` - this file


### Usage
- Run experiments via `hc_main.py` to generate allocations.
- Outputs are written to:
  - `results/Hill_climbing_runs/` → per-run allocation and logs.
  - `results/summary/hc_summary_log.csv` → aggregated run summary.
- Use the companion plotting scripts in `results/plots/Hill_climbing/` for visual analysis.


### Notes
- The file `hc_summary_log.csv` may appear with an Excel icon on Windows, but it is a standard CSV file.
- HC is typically run with multiple seeds (e.g., 30 runs) for performance comparison against GA.
