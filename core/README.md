### core/

This folder contains the core logic for evaluating Student–Project Allocation (SPA) solutions.
It defines how an allocation is scored, how constraint violations are calculated, and provides utility functions to standardize data formats across algorithms.

### Files

## `fitness.py`

Implements the core evaluation logic used to assign a fitness score to student–project allocations.

Includes:
- `preference_penalty(alloc, prefs)`: Sum of assigned ranks (**lower is better**)
- `capacity_violations(...)`: Counts over-capacity issues for projects/supervisors
- `eligibility_violations(...)`: Checks violations like GPA cutoffs (e.g., GPA min/required).
- `evaluate_solution(...)`: Combines weighted penalties to produce a single score


**The objective score is minimised.**

## `utils.py`

Contains general-purpose helper functions to support the analysis workflow.

Key function:
- `list_to_alloc_df(...)`: Converts a raw allocation list and student DataFrame into a canonical format used for scoring. Computes:
  - `assigned_project`
  - `assigned_rank` (1 to N, or inf if unmatched)
  - `_rank_oob_violation` (0/1: if project not in top-N preferences)

Also validates preference columns, ensures alignment, and catches common input errors.

## `README.md` –   This file

## `__init__.py` – Exposes the public API (`evaluate_solution`, `Weights`, `list_to_alloc_df`)

### **Used by**

These core modules are **imported by all major scripts** in the analysis:

- `hill_climbing/` — move evaluation and best-neighbour scoring
- `genetic_algorithm/` — fitness evaluation after crossover/mutation
- `hyper_heuristic/` — operator acceptance/selection based on evaluated score
- `scripts/second_round.py` — reallocation scoring (Mode A/B)
-  batch/summary generators that create `*_summary_log.csv` (when a fresh score is required)
- `scripts/analyze_hh_stats.py`
- `scripts/statistical_comparison.py`
 

Can import them like this in any script:

```python
from core.fitness import evaluate_solution
from core.utils import list_to_alloc_df


> Note: the analysis scripts read the precomputed logs; the modules above are where
> objective values are typically **computed**.

```
### **Configuration**
Penalty weights (`gamma_pref`, `alpha_capacity`, `beta_eligibility`, `delta_under`)
and the number of preferences (`num_preferences`) are read from a root-level configuration (e.g., `config.json`).
These values are used by `evaluate_solution(...)` to calculate the objective score.

