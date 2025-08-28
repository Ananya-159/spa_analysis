# Selector Hill Climbing — Analysis Output

**Timestamp:** 2025-08-27 22:47:14

**Source CSV:** `C:\Users\anany\Desktop\spa_analysis\results\selector_hill_climbing_runs\selectorhc_summary_log.csv`


# Selector Hill Climbing — Analysis Summary

Selector Hill Climbing — Descriptive Summary (Most Recent Runs)
----------------------------------------------------------
|                                  |   count |     mean |     std |      min |      25% |      50% |      75% |      max |
|----------------------------------|---------|----------|---------|----------|----------|----------|----------|----------|
| Fitness Score                    |      30 | 7143.63  | 925.177 | 4939     | 6683.25  | 7296.5   | 7667     | 9428     |
| Average Assigned Preference Rank |      30 |    4.003 |   0.158 |    3.745 |    3.897 |    4.01  |    4.088 |    4.49  |
| Top 1 Preference Match Rate (%)  |      30 |   15.752 |   3.444 |    9.804 |   12.745 |   15.196 |   19.363 |   21.569 |
| Top 3 Preference Match Rate (%)  |      30 |   41.275 |   3.908 |   26.471 |   39.216 |   41.667 |   44.118 |   46.078 |
| Gini Index                       |      30 |    0.358 |   0.011 |    0.342 |    0.348 |    0.358 |    0.363 |    0.387 |
| Total Violations                 |      30 |   22.367 |   3.079 |   15     |   21     |   23     |   24     |   30     |
| Runtime (seconds)                |      30 |   55.263 |  17.446 |   26.252 |   46.332 |   47.792 |   65.192 |   92.42  |

Selector Hill Climbing — Descriptive Summary by Tag
---------------------------------------------------
|    | tag_label              | metric                           |   n | mean ± sd          | median [IQR]                 |      min |      max |
|----|------------------------|----------------------------------|-----|--------------------|------------------------------|----------|----------|
|  0 | Selector Hill Climbing | Fitness Score                    |  30 | 7143.633 ± 925.177 | 7296.500 [6683.250–7667.000] | 4939.000 | 9428.000 |
|  1 | Selector Hill Climbing | Average Assigned Preference Rank |  30 | 4.003 ± 0.158      | 4.010 [3.897–4.088]          |    3.745 |    4.490 |
|  2 | Selector Hill Climbing | Top 1 Preference Match Rate (%)  |  30 | 15.752 ± 3.444     | 15.196 [12.745–19.363]       |    9.804 |   21.569 |
|  3 | Selector Hill Climbing | Top 3 Preference Match Rate (%)  |  30 | 41.275 ± 3.908     | 41.667 [39.216–44.118]       |   26.471 |   46.078 |
|  4 | Selector Hill Climbing | Gini Index                       |  30 | 0.358 ± 0.011      | 0.358 [0.348–0.363]          |    0.342 |    0.387 |
|  5 | Selector Hill Climbing | Total Violations                 |  30 | 22.367 ± 3.079     | 23.000 [21.000–24.000]       |   15.000 |   30.000 |
|  6 | Selector Hill Climbing | Runtime (seconds)                |  30 | 55.263 ± 17.446    | 47.792 [46.332–65.192]       |   26.252 |   92.420 |

Selector Hill Climbing — Median (IQR)
-------------------------------------
|    | Metric                           |   Median | IQR                 |
|----|----------------------------------|----------|---------------------|
|  0 | Fitness Score                    | 7296.500 | [6683.250–7667.000] |
|  1 | Average Assigned Preference Rank |    4.010 | [3.897–4.088]       |
|  2 | Top 1 Preference Match Rate (%)  |   15.196 | [12.745–19.363]     |
|  3 | Top 3 Preference Match Rate (%)  |   41.667 | [39.216–44.118]     |
|  4 | Gini Index                       |    0.358 | [0.348–0.363]       |
|  5 | Total Violations                 |   23.000 | [21.000–24.000]     |
|  6 | Runtime (seconds)                |   47.792 | [46.332–65.192]     |

Top 5 Runs by Fitness (lower=better)
------------------------------------
|    | run_id         |   Fitness Score |   Top 1 Preference Match Rate (%) |   Gini Index |   Total Violations |   Runtime (seconds) |
|----|----------------|-----------------|-----------------------------------|--------------|--------------------|---------------------|
|  1 | selectorhc_129 |            4939 |                            12.745 |        0.343 |                 15 |              45.856 |
|  3 | selectorhc_127 |            5538 |                            14.706 |        0.374 |                 17 |              56.133 |
| 13 | selectorhc_117 |            5853 |                            14.706 |        0.357 |                 18 |              72.074 |
| 12 | selectorhc_118 |            6136 |                            14.706 |        0.368 |                 19 |              63.449 |
|  6 | selectorhc_124 |            6160 |                            19.608 |        0.358 |                 19 |              47.671 |

Top 5 Runs by Top 1 Preference (%)
----------------------------------
|    | run_id         |   Fitness Score |   Top 1 Preference Match Rate (%) |   Gini Index |   Total Violations |   Runtime (seconds) |
|----|----------------|-----------------|-----------------------------------|--------------|--------------------|---------------------|
| 21 | selectorhc_109 |            6682 |                            21.569 |        0.347 |                 21 |              59.744 |
|  0 | selectorhc_130 |            6687 |                            21.569 |        0.351 |                 21 |              47.378 |
| 16 | selectorhc_114 |            7308 |                            20.588 |        0.35  |                 23 |              65.47  |
| 26 | selectorhc_104 |            7008 |                            20.588 |        0.345 |                 22 |              43.407 |
| 27 | selectorhc_103 |            6989 |                            19.608 |        0.347 |                 22 |              36.354 |

Top 5 Runs by Fairness (Gini; lower=better)
-------------------------------------------
|    | run_id         |   Fitness Score |   Top 1 Preference Match Rate (%) |   Gini Index |   Total Violations |   Runtime (seconds) |
|----|----------------|-----------------|-----------------------------------|--------------|--------------------|---------------------|
|  7 | selectorhc_123 |            7316 |                            14.706 |        0.342 |                 23 |              47.371 |
|  1 | selectorhc_129 |            4939 |                            12.745 |        0.343 |                 15 |              45.856 |
| 26 | selectorhc_104 |            7008 |                            20.588 |        0.345 |                 22 |              43.407 |
|  5 | selectorhc_125 |            8209 |                            11.765 |        0.345 |                 26 |              46.584 |
| 10 | selectorhc_120 |            7597 |                            15.686 |        0.346 |                 24 |              92.42  |

Top 5 Runs by Violations (lower=better)
---------------------------------------
|    | run_id         |   Fitness Score |   Top 1 Preference Match Rate (%) |   Gini Index |   Total Violations |   Runtime (seconds) |
|----|----------------|-----------------|-----------------------------------|--------------|--------------------|---------------------|
|  1 | selectorhc_129 |            4939 |                            12.745 |        0.343 |                 15 |              45.856 |
|  3 | selectorhc_127 |            5538 |                            14.706 |        0.374 |                 17 |              56.133 |
| 13 | selectorhc_117 |            5853 |                            14.706 |        0.357 |                 18 |              72.074 |
|  6 | selectorhc_124 |            6160 |                            19.608 |        0.358 |                 19 |              47.671 |
| 12 | selectorhc_118 |            6136 |                            14.706 |        0.368 |                 19 |              63.449 |

Stability: First 15 vs Last 15
------------------------------
- Fitness Score (lower=better): First-15 (n=15) vs Last-15 (n=15) → MW p=0.481, t=-1.206 (p=0.240), d=-0.440, r=0.156
- Average Assigned Preference Rank (lower=better): First-15 (n=15) vs Last-15 (n=15) → MW p=0.406, t=1.343 (p=0.190), d=0.490, r=-0.182
- Top-1 Preference Match Rate (%): First-15 (n=15) vs Last-15 (n=15) → MW p=0.030, t=-2.346 (p=0.026), d=-0.857, r=0.467
- Top-3 Preference Match Rate (%): First-15 (n=15) vs Last-15 (n=15) → MW p=0.661, t=-0.773 (p=0.447), d=-0.282, r=0.098
- Gini (0=fair): First-15 (n=15) vs Last-15 (n=15) → MW p=0.836, t=0.181 (p=0.858), d=0.066, r=0.049
- Total Violations (lower=better): First-15 (n=15) vs Last-15 (n=15) → MW p=0.427, t=-1.258 (p=0.221), d=-0.459, r=0.173
- Runtime (sec): First-15 (n=15) vs Last-15 (n=15) → MW p=0.184, t=1.416 (p=0.168), d=0.517, r=-0.289

Stability: Odd vs Even
----------------------
- Fitness Score (lower=better): Odd (n=15) vs Even (n=15) → MW p=0.455, t=-0.061 (p=0.951), d=-0.022, r=0.164
- Average Assigned Preference Rank (lower=better): Odd (n=15) vs Even (n=15) → MW p=1.000, t=-0.189 (p=0.851), d=-0.069, r=0.000
- Top-1 Preference Match Rate (%): Odd (n=15) vs Even (n=15) → MW p=0.819, t=-0.102 (p=0.919), d=-0.037, r=0.053
- Top-3 Preference Match Rate (%): Odd (n=15) vs Even (n=15) → MW p=0.416, t=0.961 (p=0.346), d=0.351, r=-0.178
- Gini (0=fair): Odd (n=15) vs Even (n=15) → MW p=0.803, t=-0.139 (p=0.891), d=-0.051, r=-0.058
- Total Violations (lower=better): Odd (n=15) vs Even (n=15) → MW p=0.477, t=-0.058 (p=0.954), d=-0.021, r=0.156
- Runtime (sec): Odd (n=15) vs Even (n=15) → MW p=0.678, t=-0.643 (p=0.526), d=-0.235, r=0.093

Contrast: Top 10 vs Bottom 10 (by fitness)
------------------------------------------
- Fitness Score (lower=better): Top-10 (n=10) vs Bottom-10 (n=10) → MW p=0.000, t=-7.253 (p=0.000), d=-3.244, r=1.000
- Average Assigned Preference Rank (lower=better): Top-10 (n=10) vs Bottom-10 (n=10) → MW p=0.198, t=-1.801 (p=0.090), d=-0.805, r=0.350
- Top-1 Preference Match Rate (%): Top-10 (n=10) vs Bottom-10 (n=10) → MW p=0.128, t=1.382 (p=0.184), d=0.618, r=-0.410
- Top-3 Preference Match Rate (%): Top-10 (n=10) vs Bottom-10 (n=10) → MW p=0.287, t=1.292 (p=0.216), d=0.578, r=-0.290
- Gini (0=fair): Top-10 (n=10) vs Bottom-10 (n=10) → MW p=0.678, t=-0.721 (p=0.481), d=-0.322, r=0.120
- Total Violations (lower=better): Top-10 (n=10) vs Bottom-10 (n=10) → MW p=0.000, t=-6.981 (p=0.000), d=-3.122, r=1.000
- Runtime (sec): Top-10 (n=10) vs Bottom-10 (n=10) → MW p=0.345, t=-1.520 (p=0.150), d=-0.680, r=0.260

## Enhanced Stability Tests (BH, δ, CIs, TOST, Practicality)
(Full enhanced results saved: `enhanced_stability_results.csv`)
(Excel-friendly copy saved: `enhanced_stability_results.xlsx`)

### First 15 vs Last 15
- **fitness_score**: favors **First 15**, δ=0.16 (small), q=0.002, Yes, Not equivalent
- **avg_rank**: favors **Last 15**, δ=-0.18 (small), q=0.002, No, Not equivalent
- **top1_pct**: favors **Last 15**, δ=-0.47 (medium), q=0.002, Yes, Not equivalent
- **top3_pct**: favors **Last 15**, δ=-0.10 (negligible), q=0.002, No, Equivalent
- **gini_satisfaction**: favors **First 15**, δ=0.05 (negligible), q=0.002, No, Equivalent
- **total_violations**: favors **First 15**, δ=0.17 (small), q=0.002, Yes, Not equivalent
- **runtime_sec**: favors **Last 15**, δ=-0.29 (small), q=0.002, No, Not equivalent
- Equivalence within practical bounds: **2/7** metrics.

### Odd runs vs Even runs
- **fitness_score**: favors **Odd runs**, δ=0.16 (small), q=0.002, Yes, Not equivalent
- **avg_rank**: favors **Odd runs**, δ=0.00 (negligible), q=0.002, No, Equivalent
- **top1_pct**: favors **Even runs**, δ=-0.05 (negligible), q=0.002, No, Not equivalent
- **top3_pct**: favors **Odd runs**, δ=0.18 (small), q=0.002, No, Equivalent
- **gini_satisfaction**: favors **Even runs**, δ=-0.06 (negligible), q=0.002, No, Not equivalent
- **total_violations**: favors **Odd runs**, δ=0.16 (small), q=0.002, Yes, Not equivalent
- **runtime_sec**: favors **Odd runs**, δ=0.09 (negligible), q=0.002, Yes, Not equivalent
- Equivalence within practical bounds: **2/7** metrics.

### Top 10 (best fitness) vs Bottom 10 (worst fitness)
- **fitness_score**: favors **Top 10 (best fitness)**, δ=1.00 (large), q=0.002, Yes, Not equivalent
- **avg_rank**: favors **Top 10 (best fitness)**, δ=0.35 (medium), q=0.002, No, Not equivalent
- **top1_pct**: favors **Top 10 (best fitness)**, δ=0.41 (medium), q=0.002, Yes, Not equivalent
- **top3_pct**: favors **Top 10 (best fitness)**, δ=0.29 (small), q=0.002, No, Not equivalent
- **gini_satisfaction**: favors **Top 10 (best fitness)**, δ=0.12 (negligible), q=0.002, No, Not equivalent
- **runtime_sec**: favors **Top 10 (best fitness)**, δ=0.26 (small), q=0.002, Yes, Not equivalent
- **total_violations**: favors **Top 10 (best fitness)**, δ=1.00 (large), q=0.003, Yes, Not equivalent
- Equivalence within practical bounds: **0/7** metrics.

Correlation (Spearman ρ) — Overall
--------------------------------
|                   |   fitness_score |   gini_satisfaction |   total_violations |   runtime_sec |   avg_rank |   top1_pct |   top3_pct |
|-------------------|-----------------|---------------------|--------------------|---------------|------------|------------|------------|
| fitness_score     |           1     |               0.1   |              0.992 |         0.101 |      0.285 |     -0.306 |     -0.239 |
| gini_satisfaction |           0.1   |               1     |              0.039 |         0.254 |      0.757 |     -0.216 |     -0.705 |
| total_violations  |           0.992 |               0.039 |              1     |         0.096 |      0.22  |     -0.258 |     -0.191 |
| runtime_sec       |           0.101 |               0.254 |              0.096 |         1     |      0.096 |      0.043 |     -0.048 |
| avg_rank          |           0.285 |               0.757 |              0.22  |         0.096 |      1     |     -0.744 |     -0.862 |
| top1_pct          |          -0.306 |              -0.216 |             -0.258 |         0.043 |     -0.744 |      1     |      0.479 |
| top3_pct          |          -0.239 |              -0.705 |             -0.191 |        -0.048 |     -0.862 |      0.479 |      1     |
[SAVE] selectorhc_corr_heatmap_overall.png
[SAVE] selectorhc_scatter_fitness_vs_violations.png
[SAVE] selectorhc_scatter_fitness_vs_runtime.png
[SAVE] selectorhc_scatter_top1_vs_gini.png

Selector Hill Climbing — Operator Usage (sum across runs)
---------------------------------------------------------
|    | operator   |   used_count |
|----|------------|--------------|
|  0 | swap       |    25680.000 |
|  1 | mut_micro  |     2462.000 |
|  2 | reassign   |     1858.000 |
[SAVE] selectorhc_operator_usage.png

Feasibility
-----------
Feasible runs (zero violations): 0/30 (0.0%)
[SAVE] selectorhc_hist_total_violations.png

Best/Worst Runs by Fitness
--------------------------
| Type   | run_id         | tag                    |   Fitness Score |   Gini Index |   Total Violations |   Runtime (seconds) |
|--------|----------------|------------------------|-----------------|--------------|--------------------|---------------------|
| Best   | selectorhc_129 | Selector Hill Climbing |        4939.000 |        0.343 |             15.000 |              45.855 |
| Worst  | selectorhc_106 | Selector Hill Climbing |        9428.000 |        0.361 |             30.000 |              31.835 |
