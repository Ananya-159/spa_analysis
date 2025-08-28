# Genetic Algorithm Analysis Output

**Timestamp:** 2025-08-27 22:27:46

**Source CSV:** `C:\Users\anany\Desktop\spa_analysis\results\Genetic_algorithm_runs\ga_summary_log.csv`


# Genetic Algorithm (GA) — Analysis Summary

GA — Descriptive Summary (All Runs)
---------------------------------
|                   |   count |      mean |     std |       min |       25% |       50% |       75% |       max |
|-------------------|---------|-----------|---------|-----------|-----------|-----------|-----------|-----------|
| fitness_score     |      30 | 21124.3   | 603.521 | 20000     | 20661.5   | 21167     | 21472     | 22623     |
| avg_rank          |      30 |     3.559 |   0.146 |     3.333 |     3.429 |     3.564 |     3.662 |     3.824 |
| top1_pct          |      30 |    16.209 |   3.602 |    10.784 |    12.745 |    15.686 |    18.627 |    23.529 |
| top3_pct          |      30 |    47.843 |   4.995 |    34.314 |    44.118 |    48.529 |    50.98  |    54.902 |
| gini_satisfaction |      30 |     0.281 |   0.014 |     0.248 |     0.273 |     0.279 |     0.293 |     0.312 |
| total_violations  |      30 |    74.867 |   2.623 |    70     |    73     |    75     |    76     |    81     |
| runtime_sec       |      30 |    84.689 |  11.971 |    74.565 |    78.973 |    79.607 |    85.889 |   129.982 |

GA — Descriptive Summary by Tag
-------------------------------
|    | tag_label   | metric            |   n | mean ± sd           | median [IQR]                    |       min |       max |
|----|-------------|-------------------|-----|---------------------|---------------------------------|-----------|-----------|
|  0 | GA Random   | fitness_score     |  30 | 21124.333 ± 603.521 | 21167.000 [20661.500–21472.000] | 20000.000 | 22623.000 |
|  1 | GA Random   | avg_rank          |  30 | 3.559 ± 0.146       | 3.564 [3.429–3.662]             |     3.333 |     3.824 |
|  2 | GA Random   | top1_pct          |  30 | 16.209 ± 3.602      | 15.686 [12.745–18.627]          |    10.784 |    23.529 |
|  3 | GA Random   | top3_pct          |  30 | 47.843 ± 4.995      | 48.529 [44.118–50.980]          |    34.314 |    54.902 |
|  4 | GA Random   | gini_satisfaction |  30 | 0.281 ± 0.014       | 0.279 [0.273–0.293]             |     0.248 |     0.312 |
|  5 | GA Random   | total_violations  |  30 | 74.867 ± 2.623      | 75.000 [73.000–76.000]          |    70.000 |    81.000 |
|  6 | GA Random   | runtime_sec       |  30 | 84.689 ± 11.971     | 79.607 [78.973–85.889]          |    74.565 |   129.982 |

GA — Median (IQR)
-----------------
|    | Metric            |    Median | IQR                   |
|----|-------------------|-----------|-----------------------|
|  0 | fitness_score     | 21167.000 | [20661.500–21472.000] |
|  1 | avg_rank          |     3.564 | [3.429–3.662]         |
|  2 | top1_pct          |    15.686 | [12.745–18.627]       |
|  3 | top3_pct          |    48.529 | [44.118–50.980]       |
|  4 | gini_satisfaction |     0.279 | [0.273–0.293]         |
|  5 | total_violations  |    75.000 | [73.000–76.000]       |
|  6 | runtime_sec       |    79.607 | [78.973–85.889]       |

Top 5 Runs by Fitness (lower=better)
------------------------------------
|    | run_id           |   fitness_score |   top1_pct |   gini_satisfaction |   total_violations |   runtime_sec |
|----|------------------|-----------------|------------|---------------------|--------------------|---------------|
| 27 | 20250819T235913Z |           20000 |     23.529 |               0.28  |                 73 |        86.549 |
| 11 | 20250819T233722Z |           20227 |     20.588 |               0.274 |                 70 |        78.759 |
| 10 | 20250819T233603Z |           20245 |     15.686 |               0.278 |                 70 |        78.972 |
|  7 | 20250819T233205Z |           20577 |     15.686 |               0.285 |                 73 |        78.872 |
| 12 | 20250819T233841Z |           20580 |     18.627 |               0.264 |                 74 |        79.266 |

Top 5 Runs by Top 1 Preference (%)
----------------------------------
|    | run_id           |   fitness_score |   top1_pct |   gini_satisfaction |   total_violations |   runtime_sec |
|----|------------------|-----------------|------------|---------------------|--------------------|---------------|
| 27 | 20250819T235913Z |           20000 |     23.529 |               0.28  |                 73 |        86.549 |
| 20 | 20250819T234912Z |           21429 |     22.549 |               0.296 |                 74 |        79.679 |
|  2 | 20250819T232438Z |           21466 |     21.569 |               0.294 |                 75 |        74.565 |
| 26 | 20250819T235747Z |           20845 |     20.588 |               0.276 |                 73 |        87.548 |
| 11 | 20250819T233722Z |           20227 |     20.588 |               0.274 |                 70 |        78.759 |

Top 5 Runs by Fairness (Gini; lower=better)
-------------------------------------------
|    | run_id           |   fitness_score |   top1_pct |   gini_satisfaction |   total_violations |   runtime_sec |
|----|------------------|-----------------|------------|---------------------|--------------------|---------------|
|  1 | 20250819T232324Z |           21784 |     17.647 |               0.248 |                 78 |       111.65  |
|  6 | 20250819T233046Z |           22623 |     18.627 |               0.263 |                 78 |        79.295 |
| 12 | 20250819T233841Z |           20580 |     18.627 |               0.264 |                 74 |        79.266 |
| 23 | 20250819T235326Z |           20652 |     14.706 |               0.265 |                 77 |        85.496 |
|  5 | 20250819T232927Z |           21149 |     12.745 |               0.265 |                 73 |        78.99  |

Top 5 Runs by Violations (lower=better)
---------------------------------------
|    | run_id           |   fitness_score |   top1_pct |   gini_satisfaction |   total_violations |   runtime_sec |
|----|------------------|-----------------|------------|---------------------|--------------------|---------------|
| 11 | 20250819T233722Z |           20227 |     20.588 |               0.274 |                 70 |        78.759 |
| 10 | 20250819T233603Z |           20245 |     15.686 |               0.278 |                 70 |        78.972 |
| 28 | 20250820T000038Z |           21406 |     10.784 |               0.282 |                 72 |        84.952 |
| 25 | 20250819T235619Z |           20859 |     10.784 |               0.278 |                 72 |        86.231 |
| 14 | 20250819T234118Z |           20874 |     15.686 |               0.292 |                 73 |        77.84  |

Stability: First 15 vs Last 15
------------------------------
- Fitness (lower=better): First-15 (n=15) vs Last-15 (n=15) → MW p=0.901, t=0.434 (p=0.668), d=0.158, r=-0.031
- Average Assigned Preference Rank (lower=better): First-15 (n=15) vs Last-15 (n=15) → MW p=0.191, t=-1.318 (p=0.198), d=-0.481, r=0.284
- Top-1 Match (%): First-15 (n=15) vs Last-15 (n=15) → MW p=0.381, t=0.689 (p=0.497), d=0.252, r=-0.191
- Top-3 Match (%): First-15 (n=15) vs Last-15 (n=15) → MW p=0.884, t=0.141 (p=0.889), d=0.051, r=0.036
- Gini (0=fair): First-15 (n=15) vs Last-15 (n=15) → MW p=0.018, t=-2.478 (p=0.019), d=-0.905, r=0.511
- Total Violations (lower=better): First-15 (n=15) vs Last-15 (n=15) → MW p=0.834, t=-0.412 (p=0.684), d=-0.150, r=0.049
- Runtime (sec): First-15 (n=15) vs Last-15 (n=15) → MW p=0.184, t=0.804 (p=0.434), d=0.293, r=0.289

Stability: Odd vs Even
----------------------
- Fitness (lower=better): Odd (n=15) vs Even (n=15) → MW p=0.213, t=1.427 (p=0.165), d=0.521, r=-0.271
- Average Assigned Preference Rank (lower=better): Odd (n=15) vs Even (n=15) → MW p=1.000, t=-0.072 (p=0.943), d=-0.026, r=0.004
- Top-1 Match (%): Odd (n=15) vs Even (n=15) → MW p=0.478, t=0.689 (p=0.496), d=0.252, r=-0.156
- Top-3 Match (%): Odd (n=15) vs Even (n=15) → MW p=0.480, t=-1.003 (p=0.325), d=-0.366, r=0.156
- Gini (0=fair): Odd (n=15) vs Even (n=15) → MW p=0.803, t=0.350 (p=0.729), d=0.128, r=-0.058
- Total Violations (lower=better): Odd (n=15) vs Even (n=15) → MW p=0.557, t=0.550 (p=0.587), d=0.201, r=-0.129
- Runtime (sec): Odd (n=15) vs Even (n=15) → MW p=0.934, t=-0.822 (p=0.420), d=-0.300, r=0.022

Contrast: Top 10 vs Bottom 10 (by fitness)
------------------------------------------
- Fitness (lower=better): Top-10 (n=10) vs Bottom-10 (n=10) → MW p=0.000, t=-8.448 (p=0.000), d=-3.778, r=1.000
- Average Assigned Preference Rank (lower=better): Top-10 (n=10) vs Bottom-10 (n=10) → MW p=0.427, t=-0.850 (p=0.407), d=-0.380, r=0.220
- Top-1 Match (%): Top-10 (n=10) vs Bottom-10 (n=10) → MW p=1.000, t=0.059 (p=0.954), d=0.026, r=0.000
- Top-3 Match (%): Top-10 (n=10) vs Bottom-10 (n=10) → MW p=0.129, t=1.069 (p=0.302), d=0.478, r=-0.410
- Gini (0=fair): Top-10 (n=10) vs Bottom-10 (n=10) → MW p=0.385, t=-0.728 (p=0.478), d=-0.326, r=0.240
- Total Violations (lower=better): Top-10 (n=10) vs Bottom-10 (n=10) → MW p=0.008, t=-3.098 (p=0.006), d=-1.386, r=0.700
- Runtime (sec): Top-10 (n=10) vs Bottom-10 (n=10) → MW p=0.307, t=-1.126 (p=0.287), d=-0.504, r=0.280

## Enhanced Stability Tests (BH, δ, CIs, TOST, Practicality)
(Full enhanced results saved: `enhanced_stability_results.csv`)
(Excel-friendly copy saved: `enhanced_stability_results.xlsx`)

### First 15 vs Last 15
- **fitness_score**: favors **Last 15**, δ=-0.03 (negligible), q=0.004, Yes, Not equivalent
- **avg_rank**: favors **First 15**, δ=0.28 (small), q=0.004, No, Not equivalent
- **top1_pct**: favors **First 15**, δ=0.19 (small), q=0.004, Yes, Not equivalent
- **top3_pct**: favors **Last 15**, δ=-0.04 (negligible), q=0.004, No, Not equivalent
- **gini_satisfaction**: favors **First 15**, δ=0.51 (large), q=0.004, Yes, Not equivalent
- **total_violations**: favors **First 15**, δ=0.05 (negligible), q=0.004, No, Not equivalent
- **runtime_sec**: favors **First 15**, δ=0.29 (small), q=0.004, Yes, Not equivalent
- Equivalence within practical bounds: **0/7** metrics.

### Odd runs vs Even runs
- **fitness_score**: favors **Even runs**, δ=-0.27 (small), q=0.004, Yes, Not equivalent
- **avg_rank**: favors **Even runs**, δ=0.00 (negligible), q=0.004, No, Equivalent
- **top1_pct**: favors **Odd runs**, δ=0.16 (small), q=0.004, No, Not equivalent
- **top3_pct**: favors **Even runs**, δ=-0.16 (small), q=0.004, Yes, Not equivalent
- **gini_satisfaction**: favors **Even runs**, δ=-0.06 (negligible), q=0.004, No, Not equivalent
- **total_violations**: favors **Even runs**, δ=-0.13 (negligible), q=0.004, Yes, Not equivalent
- **runtime_sec**: favors **Even runs**, δ=0.02 (negligible), q=0.004, No, Not equivalent
- Equivalence within practical bounds: **1/7** metrics.

### Top 10 (best fitness) vs Bottom 10 (worst fitness)
- **fitness_score**: favors **Top 10 (best fitness)**, δ=1.00 (large), q=0.004, Yes, Not equivalent
- **avg_rank**: favors **Top 10 (best fitness)**, δ=0.22 (small), q=0.004, No, Not equivalent
- **top1_pct**: favors **Bottom 10 (worst fitness)**, δ=0.00 (negligible), q=0.004, No, Not equivalent
- **top3_pct**: favors **Top 10 (best fitness)**, δ=0.41 (medium), q=0.004, Yes, Not equivalent
- **gini_satisfaction**: favors **Top 10 (best fitness)**, δ=0.24 (small), q=0.004, No, Not equivalent
- **total_violations**: favors **Top 10 (best fitness)**, δ=0.70 (large), q=0.004, Yes, Not equivalent
- **runtime_sec**: favors **Top 10 (best fitness)**, δ=0.28 (small), q=0.004, No, Not equivalent
- Equivalence within practical bounds: **0/7** metrics.

Correlation (Spearman ρ) — Overall
--------------------------------
|                   |   fitness_score |   gini_satisfaction |   total_violations |   runtime_sec |   avg_rank |   top1_pct |   top3_pct |
|-------------------|-----------------|---------------------|--------------------|---------------|------------|------------|------------|
| fitness_score     |           1     |               0.135 |              0.654 |         0.265 |      0.157 |     -0.093 |     -0.266 |
| gini_satisfaction |           0.135 |               1     |              0.032 |        -0.318 |      0.64  |     -0.104 |     -0.516 |
| total_violations  |           0.654 |               0.032 |              1     |         0.331 |      0.07  |     -0.087 |     -0.169 |
| runtime_sec       |           0.265 |              -0.318 |              0.331 |         1     |     -0.294 |     -0.054 |      0.312 |
| avg_rank          |           0.157 |               0.64  |              0.07  |        -0.294 |      1     |     -0.671 |     -0.822 |
| top1_pct          |          -0.093 |              -0.104 |             -0.087 |        -0.054 |     -0.671 |      1     |      0.435 |
| top3_pct          |          -0.266 |              -0.516 |             -0.169 |         0.312 |     -0.822 |      0.435 |      1     |
[SAVE] ga_corr_heatmap_overall.png
[SAVE] scatter_fitness_vs_violations.png
[SAVE] scatter_fitness_vs_runtime.png
[SAVE] scatter_top1_vs_gini.png

GA — Spearman Correlations: Params vs Metrics
---------------------------------------------
|    | param    | metric   | rho      | p   |   n |
|----|----------|----------|----------|-----|-----|
|  0 | pop_size | —        | constant | —   |  30 |
|  1 | cx_prob  | —        | constant | —   |  30 |
|  2 | mut_prob | —        | constant | —   |  30 |
|  3 | elitism  | —        | constant | —   |  30 |

GA — Spearman Correlations: Ops vs Metrics
------------------------------------------
|    | op            | metric            |    rho |     p |   n |
|----|---------------|-------------------|--------|-------|-----|
|  0 | cx_ops        | avg_rank          | -0.085 | 0.656 |  30 |
|  1 | cx_ops        | fitness_score     |  0.079 | 0.676 |  30 |
|  2 | cx_ops        | gini_satisfaction | -0.279 | 0.136 |  30 |
|  3 | cx_ops        | runtime_sec       |  0.243 | 0.196 |  30 |
|  4 | cx_ops        | top1_pct          | -0.057 | 0.766 |  30 |
|  5 | cx_ops        | top3_pct          |  0.094 | 0.62  |  30 |
|  6 | cx_ops        | total_violations  | -0.112 | 0.555 |  30 |
|  7 | mut_ops       | avg_rank          |  0.019 | 0.922 |  30 |
|  8 | mut_ops       | fitness_score     |  0.393 | 0.032 |  30 |
|  9 | mut_ops       | gini_satisfaction | -0.096 | 0.613 |  30 |
| 10 | mut_ops       | runtime_sec       |  0.253 | 0.178 |  30 |
| 11 | mut_ops       | top1_pct          | -0.133 | 0.482 |  30 |
| 12 | mut_ops       | top3_pct          |  0.028 | 0.885 |  30 |
| 13 | mut_ops       | total_violations  |  0.213 | 0.259 |  30 |
| 14 | repair_calls  | avg_rank          | -0.081 | 0.672 |  30 |
| 15 | repair_calls  | fitness_score     |  0.309 | 0.096 |  30 |
| 16 | repair_calls  | gini_satisfaction | -0.241 | 0.2   |  30 |
| 17 | repair_calls  | runtime_sec       |  0.37  | 0.044 |  30 |
| 18 | repair_calls  | top1_pct          | -0.104 | 0.586 |  30 |
| 19 | repair_calls  | top3_pct          |  0.118 | 0.535 |  30 |
| 20 | repair_calls  | total_violations  |  0.071 | 0.711 |  30 |
| 21 | repair_per_op | avg_rank          | -0.07  | 0.712 |  30 |
| 22 | repair_per_op | fitness_score     | -0.306 | 0.101 |  30 |
| 23 | repair_per_op | gini_satisfaction | -0.048 | 0.8   |  30 |
| 24 | repair_per_op | runtime_sec       | -0.038 | 0.842 |  30 |
| 25 | repair_per_op | top1_pct          |  0.078 | 0.681 |  30 |
| 26 | repair_per_op | top3_pct          |  0.023 | 0.905 |  30 |
| 27 | repair_per_op | total_violations  | -0.18  | 0.342 |  30 |

Feasibility
-----------
Feasible runs (zero violations): 0/30 (0.0%)
[SAVE] ga_hist_total_violations.png

Best/Worst Runs by Fitness
--------------------------
| Type   | run_id           | tag       |   fitness |   gini |   viol |   runtime |
|--------|------------------|-----------|-----------|--------|--------|-----------|
| Best   | 20250819T235913Z | GA Random | 20000.000 |  0.280 | 73.000 |    86.549 |
| Worst  | 20250819T233046Z | GA Random | 22623.000 |  0.263 | 78.000 |    79.295 |
