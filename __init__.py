"""
SPA Analysis — Dissertation project for optimising Student–Project Allocation.

This package integrates evaluation logic, fairness metrics, and metaheuristic
search algorithms, with optional policy-layer reallocation and analysis tools.

Contents:
- core/               → evaluation logic (fitness function, penalties) (Phase 1)
- fairness/           → fairness metrics (e.g., Gini, leximin helpers) (Phase 1)
- eda_analysis/       → exploratory data analysis (Phase 2)
- hill_climbing/      → Phase 3 algorithm (HC operators and driver)
- genetic_algorithm/  → Phase 4 algorithm (GA operators and driver)
- hyper_heuristic/    → adaptive Choice Function + operator selection
- scripts/            → experiment runners, plotting, and summary generators
- results/            → auto-generated logs, allocations, summaries, plots
- data/               → synthetic dataset (safe to publish)
- data_generators/    → synthetic dataset generator script 

"""
