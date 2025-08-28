### Genetic Algorithm (GA)

This folder contains the implementation of the Genetic Algorithm for the Student–Project Allocation (SPA) problem.


### Files
- `ga_main.py` — entry point for running GA (batch or single runs).
- `crossover.py` — crossover operator with repair.
- `mutation.py` — mutation operators with repair.
- `selection.py` — selection operators (tournament, etc.).
- `ga_utils.py` — dataset loading, population initialization, repair helpers, logging.
- `README.md` — this file


### Usage
- Run experiments via `ga_main.py` to generate allocations.
- Outputs are written to:
  - `results/Genetic_algorithm_runs/` → per-run allocation and logs.
  - `results/summary/ga_summary_log.csv` → aggregated run summary.
- Use the companion plotting scripts in `results/plots/Genetic_algorithm/` for visual analysis.



### Notes
- The file `ga_summary_Log.csv` may appear with an Excel icon on Windows, but it is a standard CSV file.
- All code here is modularized to allow reuse in hybrid methods or comparisons.
- The genetic_algorithm/ folder is used as a collection of scripts; no __init__.py is required since imports are executed file-by-file.