### Fairness metrics for Student-Project Allocation (SPA)

This module provides helper functions to **measure** and **compare** fairness in SPA allocations.  
These metrics are used for **reporting purposes** only and do **not** affect the core optimisation objective unless explicitly added as a weighted term.

### Files
- `metrics.py` – Implementation of fairness metrics
- `__init__.py` – Exposes public API (`gini`, `ranks_to_satisfaction`, `leximin_compare`)
- `README.md` –   This file

### Functions

- **`gini(values)`**  
  Calculates the Gini coefficient *(0 = perfectly fair, 1 = maximally unequal)*.  
  Interprets the input as **satisfaction scores** (higher = better).  
  Returns a float in \[0, 1].

- **`ranks_to_satisfaction(ranks, max_rank=6, num_prefs=None)`**  
  Converts project preference ranks *(1 = best, higher = worse)* to satisfaction scores.  
  If `num_prefs` is provided, it overrides `max_rank`.

- **`leximin_compare(sat_a, sat_b)`**  
  Compares two satisfaction distributions using **leximin ordering**.  
  Returns:
  - `1` if `sat_a` is better  
  - `-1` if `sat_b` is better  
  - `0` if tie

### Quick usage

```python
from fairness import gini, ranks_to_satisfaction

# Example: ranks -> satisfaction -> gini
ranks = [1, 2, 3]
scores = ranks_to_satisfaction(ranks, num_prefs=6)
print("Scores:", scores)
print("Gini:", gini(scores))
