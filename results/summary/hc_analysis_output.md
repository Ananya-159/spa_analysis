# Hill Climbing Analysis Output

**Timestamp:** 2025-08-27 22:15:41

**Source CSV:** `C:\Users\anany\Desktop\spa_analysis\results\Hill_climbing_runs\hc_summary_log.csv`


# Hill Climbing (HC) — Analysis Summary

HC — Descriptive Summary (All Runs)
---------------------------------
|                   |   count |      mean |      std |      min |       25% |       50% |       75% |       max |
|-------------------|---------|-----------|----------|----------|-----------|-----------|-----------|-----------|
| fitness_score     |      60 | 13575.3   | 1981.19  | 9377     | 12377.8   | 13382     | 14380.2   | 18316     |
| avg_rank          |      60 |     3.392 |    0.281 |    2.912 |     3.113 |     3.436 |     3.61  |     3.863 |
| top1_pct          |      60 |    24.788 |    5.869 |   12.745 |    19.608 |    23.529 |    29.657 |    36.275 |
| top3_pct          |      60 |    54.428 |    5.999 |   43.137 |    50     |    54.412 |    59.804 |    65.686 |
| gini_satisfaction |      60 |     0.304 |    0.028 |    0.242 |     0.279 |     0.308 |     0.323 |     0.359 |
| total_violations  |      60 |    13.85  |    1.867 |   10     |    13     |    14     |    15     |    18     |
| runtime_sec       |      60 |    56.897 |   37.378 |   23.32  |    34.698 |    43.225 |    54.395 |   148.24  |

HC — Descriptive Summary by Tag (HC Greedy vs HC Random)
--------------------------------------------------------
|    | tag_label   | metric            |   n | mean +/- sd            | median [IQR]                    |       min |       max |
|----|-------------|-------------------|-----|----------------------|---------------------------------|-----------|-----------|
|  0 | HC Greedy   | fitness_score     |  30 | 13042.433 +/- 2132.435 | 13326.000 [11373.750-14316.000] |  9377.000 | 18304.000 |
|  1 | HC Greedy   | avg_rank          |  30 | 3.174 +/- 0.182        | 3.137 [3.020-3.319]             |     2.912 |     3.559 |
|  2 | HC Greedy   | top1_pct          |  30 | 29.216 +/- 3.626       | 29.412 [26.716-31.373]          |    22.549 |    36.275 |
|  3 | HC Greedy   | top3_pct          |  30 | 58.562 +/- 4.391       | 59.314 [54.902-61.765]          |    49.020 |    65.686 |
|  4 | HC Greedy   | gini_satisfaction |  30 | 0.284 +/- 0.022        | 0.281 [0.264-0.304]             |     0.242 |     0.324 |
|  5 | HC Greedy   | total_violations  |  30 | 13.633 +/- 2.008       | 14.000 [12.000-15.000]          |    10.000 |    18.000 |
|  6 | HC Greedy   | runtime_sec       |  30 | 57.666 +/- 39.856      | 43.790 [31.570-53.090]          |    23.320 |   148.240 |
|  7 | HC Random   | fitness_score     |  30 | 14108.200 +/- 1688.418 | 14365.500 [13363.750-15370.250] | 10364.000 | 18316.000 |
|  8 | HC Random   | avg_rank          |  30 | 3.610 +/- 0.170        | 3.613 [3.495-3.738]             |     3.098 |     3.863 |
|  9 | HC Random   | top1_pct          |  30 | 20.359 +/- 4.044       | 19.608 [18.627-22.549]          |    12.745 |    32.353 |
| 10 | HC Random   | top3_pct          |  30 | 50.294 +/- 4.309       | 50.490 [46.324-53.676]          |    43.137 |    60.784 |
| 11 | HC Random   | gini_satisfaction |  30 | 0.323 +/- 0.019        | 0.322 [0.311-0.335]             |     0.275 |     0.359 |
| 12 | HC Random   | total_violations  |  30 | 14.067 +/- 1.721       | 14.000 [13.000-15.000]          |    10.000 |    18.000 |
| 13 | HC Random   | runtime_sec       |  30 | 56.127 +/- 35.393      | 43.225 [35.528-56.523]          |    25.220 |   146.740 |

Hill Climbing — Median (IQR)
----------------------------
|    | Metric            |    Median | IQR                   |
|----|-------------------|-----------|-----------------------|
|  0 | fitness_score     | 13382.000 | [12377.750-14380.250] |
|  1 | avg_rank          |     3.436 | [3.113-3.610]         |
|  2 | top1_pct          |    23.529 | [19.608-29.657]       |
|  3 | top3_pct          |    54.412 | [50.000-59.804]       |
|  4 | gini_satisfaction |     0.308 | [0.279-0.323]         |
|  5 | total_violations  |    14.000 | [13.000-15.000]       |
|  6 | runtime_sec       |    43.225 | [34.698-54.395]       |

Top 5 Runs by Fitness (lower=better)
------------------------------------
|    | run_id   |   fitness_score |   top1_pct |   gini_satisfaction |   total_violations |   runtime_sec |
|----|----------|-----------------|------------|---------------------|--------------------|---------------|
| 43 | 67ab7573 |            9377 |     26.471 |               0.295 |                 11 |         34.03 |
| 45 | 2ca57605 |           10339 |     28.431 |               0.31  |                 10 |         25.35 |
| 13 | 2316ec80 |           10363 |     22.549 |               0.324 |                 10 |         42.88 |
| 10 | 9d2bbbda |           10364 |     20.588 |               0.318 |                 10 |         53.38 |
| 39 | 3ace27aa |           10372 |     24.51  |               0.315 |                 11 |         29.61 |

Top 5 Runs by Top 1 Preference (%)
----------------------------------
|    | run_id   |   fitness_score |   top1_pct |   gini_satisfaction |   total_violations |   runtime_sec |
|----|----------|-----------------|------------|---------------------|--------------------|---------------|
|  1 | 46e917a4 |           16331 |     36.275 |               0.283 |                 17 |         29.24 |
| 37 | 2896a468 |           14313 |     34.314 |               0.28  |                 14 |         25.59 |
| 55 | 78d0667a |           18304 |     33.333 |               0.269 |                 18 |        130.91 |
| 25 | fa342de1 |           16304 |     33.333 |               0.26  |                 16 |         30.09 |
| 49 | 054940d4 |           13349 |     32.353 |               0.268 |                 15 |         30.75 |

Top 5 Runs by Fairness (Gini; lower=better)
-------------------------------------------
|    | run_id   |   fitness_score |   top1_pct |   gini_satisfaction |   total_violations |   runtime_sec |
|----|----------|-----------------|------------|---------------------|--------------------|---------------|
| 29 | 975ab178 |           14317 |     29.412 |               0.242 |                 15 |         49.58 |
| 11 | 4133498b |           13328 |     28.431 |               0.258 |                 14 |         57.44 |
| 25 | fa342de1 |           16304 |     33.333 |               0.26  |                 16 |         30.09 |
| 33 | 297369d9 |           14321 |     32.353 |               0.261 |                 15 |         36.17 |
|  5 | 89bb2cca |           13325 |     30.392 |               0.261 |                 14 |         95.53 |

Top 5 Runs by Violations (lower=better)
---------------------------------------
|    | run_id   |   fitness_score |   top1_pct |   gini_satisfaction |   total_violations |   runtime_sec |
|----|----------|-----------------|------------|---------------------|--------------------|---------------|
|  2 | 9268f16e |           10386 |     16.667 |               0.345 |                 10 |         77.46 |
| 10 | 9d2bbbda |           10364 |     20.588 |               0.318 |                 10 |         53.38 |
| 13 | 2316ec80 |           10363 |     22.549 |               0.324 |                 10 |         42.88 |
| 45 | 2ca57605 |           10339 |     28.431 |               0.31  |                 10 |         25.35 |
| 39 | 3ace27aa |           10372 |     24.51  |               0.315 |                 11 |         29.61 |

Stability: First 15 vs Last 15
------------------------------
- Fitness (lower=better): First-15 (n=15) vs Last-15 (n=15) -> MW p=0.093, t=-1.735 (p=0.095), d=-0.633, r=0.364
- Average Assigned Preference Rank (lower=better): First-15 (n=15) vs Last-15 (n=15) -> MW p=0.177, t=1.345 (p=0.189), d=0.491, r=-0.293
- Top-1 Match (%): First-15 (n=15) vs Last-15 (n=15) -> MW p=0.288, t=-0.917 (p=0.367), d=-0.335, r=0.231
- Top-3 Match (%): First-15 (n=15) vs Last-15 (n=15) -> MW p=0.253, t=-1.175 (p=0.250), d=-0.429, r=0.249
- Gini (0=fair): First-15 (n=15) vs Last-15 (n=15) -> MW p=0.213, t=1.495 (p=0.146), d=0.546, r=-0.271
- Total Violations (lower=better): First-15 (n=15) vs Last-15 (n=15) -> MW p=0.076, t=-1.811 (p=0.081), d=-0.661, r=0.378
- Runtime (sec): First-15 (n=15) vs Last-15 (n=15) -> MW p=0.803, t=-0.391 (p=0.699), d=-0.143, r=-0.058

Stability: Odd vs Even
----------------------
- Fitness (lower=better): Odd (n=30) vs Even (n=30) -> MW p=0.003, t=2.146 (p=0.036), d=0.554, r=-0.447
- Average Assigned Preference Rank (lower=better): Odd (n=30) vs Even (n=30) -> MW p=0.000, t=9.571 (p=0.000), d=2.471, r=-0.913
- Top-1 Match (%): Odd (n=30) vs Even (n=30) -> MW p=0.000, t=-8.930 (p=0.000), d=-2.306, r=0.873
- Top-3 Match (%): Odd (n=30) vs Even (n=30) -> MW p=0.000, t=-7.360 (p=0.000), d=-1.900, r=0.804
- Gini (0=fair): Odd (n=30) vs Even (n=30) -> MW p=0.000, t=7.342 (p=0.000), d=1.896, r=-0.811
- Total Violations (lower=better): Odd (n=30) vs Even (n=30) -> MW p=0.282, t=0.897 (p=0.373), d=0.232, r=-0.160
- Runtime (sec): Odd (n=30) vs Even (n=30) -> MW p=0.900, t=-0.158 (p=0.875), d=-0.041, r=-0.020

Contrast: Top 10 vs Bottom 10 (by fitness)
------------------------------------------
- Fitness (lower=better): Top-10 (n=10) vs Bottom-10 (n=10) -> MW p=0.000, t=-15.562 (p=0.000), d=-6.960, r=1.000
- Average Assigned Preference Rank (lower=better): Top-10 (n=10) vs Bottom-10 (n=10) -> MW p=0.256, t=1.272 (p=0.224), d=0.569, r=-0.310
- Top-1 Match (%): Top-10 (n=10) vs Bottom-10 (n=10) -> MW p=0.130, t=-1.492 (p=0.160), d=-0.667, r=0.410
- Top-3 Match (%): Top-10 (n=10) vs Bottom-10 (n=10) -> MW p=0.172, t=-1.429 (p=0.178), d=-0.639, r=0.370
- Gini (0=fair): Top-10 (n=10) vs Bottom-10 (n=10) -> MW p=0.345, t=1.381 (p=0.186), d=0.617, r=-0.260
- Total Violations (lower=better): Top-10 (n=10) vs Bottom-10 (n=10) -> MW p=0.000, t=-11.943 (p=0.000), d=-5.341, r=1.000
- Runtime (sec): Top-10 (n=10) vs Bottom-10 (n=10) -> MW p=0.307, t=0.612 (p=0.548), d=0.274, r=-0.280

## Enhanced Stability Tests (BH, δ, CIs, TOST, Practicality)
(Full enhanced results saved: `enhanced_stability_results.csv`)
(Excel-friendly copy saved: `enhanced_stability_results.xlsx`)

### First 15 vs Last 15
- **fitness_score**: favors **First 15**, δ=0.36 (medium), q=0.000, True, Not equivalent
- **avg_rank**: favors **Last 15**, δ=-0.29 (small), q=0.000, True, Not equivalent
- **top1_pct**: favors **Last 15**, δ=-0.23 (small), q=0.000, True, Not equivalent
- **top3_pct**: favors **Last 15**, δ=-0.25 (small), q=0.000, False, Not equivalent
- **gini_satisfaction**: favors **Last 15**, δ=-0.27 (small), q=0.000, True, Not equivalent
- **total_violations**: favors **First 15**, δ=0.38 (medium), q=0.000, True, Not equivalent
- **runtime_sec**: favors **Last 15**, δ=-0.06 (negligible), q=0.000, True, Not equivalent
- Equivalence within practical bounds: **0/7** metrics.

### Odd runs vs Even runs
- **fitness_score**: favors **Even runs**, δ=-0.45 (medium), q=0.000, True, Not equivalent
- **avg_rank**: favors **Even runs**, δ=-0.91 (large), q=0.000, True, Not equivalent
- **top1_pct**: favors **Even runs**, δ=-0.87 (large), q=0.000, True, Not equivalent
- **top3_pct**: favors **Even runs**, δ=-0.80 (large), q=0.000, True, Not equivalent
- **gini_satisfaction**: favors **Even runs**, δ=-0.81 (large), q=0.000, True, Not equivalent
- **total_violations**: favors **Odd runs**, δ=-0.16 (small), q=0.000, False, Equivalent
- **runtime_sec**: favors **Odd runs**, δ=-0.02 (negligible), q=0.000, False, Not equivalent
- Equivalence within practical bounds: **1/7** metrics.

### Top 10 (best fitness) vs Bottom 10 (worst fitness)
- **fitness_score**: favors **Top 10 (best fitness)**, δ=1.00 (large), q=0.000, True, Not equivalent
- **avg_rank**: favors **Bottom 10 (worst fitness)**, δ=-0.31 (small), q=0.000, True, Not equivalent
- **top1_pct**: favors **Bottom 10 (worst fitness)**, δ=-0.41 (medium), q=0.000, True, Not equivalent
- **top3_pct**: favors **Bottom 10 (worst fitness)**, δ=-0.37 (medium), q=0.000, True, Not equivalent
- **gini_satisfaction**: favors **Bottom 10 (worst fitness)**, δ=-0.26 (small), q=0.000, True, Not equivalent
- **total_violations**: favors **Top 10 (best fitness)**, δ=1.00 (large), q=0.000, True, Not equivalent
- **runtime_sec**: favors **Bottom 10 (worst fitness)**, δ=-0.28 (small), q=0.000, True, Not equivalent
- Equivalence within practical bounds: **0/7** metrics.

Correlation (Spearman ρ) — Overall
--------------------------------
|                   |   fitness_score |   gini_satisfaction |   total_violations |   runtime_sec |   avg_rank |   top1_pct |   top3_pct |
|-------------------|-----------------|---------------------|--------------------|---------------|------------|------------|------------|
| fitness_score     |           1     |              -0.08  |              0.907 |        -0.116 |     -0.045 |      0.023 |      0.104 |
| gini_satisfaction |          -0.08  |               1     |             -0.332 |         0.049 |      0.937 |     -0.711 |     -0.921 |
| total_violations  |           0.907 |              -0.332 |              1     |        -0.068 |     -0.318 |      0.274 |      0.381 |
| runtime_sec       |          -0.116 |               0.049 |             -0.068 |         1     |      0.051 |     -0.075 |     -0.031 |
| avg_rank          |          -0.045 |               0.937 |             -0.318 |         0.051 |      1     |     -0.886 |     -0.961 |
| top1_pct          |           0.023 |              -0.711 |              0.274 |        -0.075 |     -0.886 |      1     |      0.827 |
| top3_pct          |           0.104 |              -0.921 |              0.381 |        -0.031 |     -0.961 |      0.827 |      1     |
[SAVE] hc_corr_heatmap_overall.png
[SAVE] scatter_fitness_vs_violations.png
[SAVE] scatter_fitness_vs_runtime.png
[SAVE] scatter_top1_vs_gini.png

Feasibility
-----------
Feasible runs (zero violations): 0/60 (0.0%)
[SAVE] hist_total_violations.png

Best/Worst Runs by Fitness
--------------------------
| Type   | run_id   | tag       |   fitness |   gini |   viol |   runtime |
|--------|----------|-----------|-----------|--------|--------|-----------|
| Best   | 67ab7573 | HC Greedy |  9377.000 |  0.295 | 11.000 |    34.030 |
| Worst  | 6e414435 | HC Random | 18316.000 |  0.275 | 18.000 |    52.890 |
