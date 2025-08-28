# Selector-HC Summary Results

This summary aggregates **30** recent Selector-HC runs.

## Mean ± Standard Deviation by Metric

| Metric                          |    Mean |   Standard Deviation |
|---------------------------------|---------|----------------------|
| Fitness Score                   | 7143.63 |               925.18 |
| Penalty                         |  408.3  |                16.12 |
| Average Preference Rank         |    4    |                 0.16 |
| Top-1 Preference Match Rate (%) |   15.75 |                 3.44 |
| Top-3 Preference Match Rate (%) |   41.27 |                 3.91 |
| Gini Index                      |    0.36 |                 0.01 |
| Total Violations                |   22.37 |                 3.08 |
| Runtime (s)                     |   55.26 |                17.45 |

## Median (Interquartile Range) by Metric

| Metric                          |   Median |   Interquartile Range |
|---------------------------------|----------|-----------------------|
| Fitness Score                   |  7296.5  |                983.75 |
| Penalty                         |   409    |                 19.5  |
| Average Preference Rank         |     4.01 |                  0.19 |
| Top-1 Preference Match Rate (%) |    15.2  |                  6.62 |
| Top-3 Preference Match Rate (%) |    41.67 |                  4.9  |
| Gini Index                      |     0.36 |                  0.02 |
| Total Violations                |    23    |                  3    |
| Runtime (s)                     |    47.79 |                 18.86 |

## Top 5 Runs by Fitness Score (lower = better)

| run_id         |   Fitness Score |   Top-1 Preference Match Rate (%) |   Gini Index |   Total Violations |   Runtime (s) |
|----------------|-----------------|-----------------------------------|--------------|--------------------|---------------|
| selectorhc_129 |            4939 |                            12.745 |        0.343 |                 15 |        45.856 |
| selectorhc_127 |            5538 |                            14.706 |        0.374 |                 17 |        56.133 |
| selectorhc_117 |            5853 |                            14.706 |        0.357 |                 18 |        72.074 |
| selectorhc_118 |            6136 |                            14.706 |        0.368 |                 19 |        63.449 |
| selectorhc_124 |            6160 |                            19.608 |        0.358 |                 19 |        47.671 |

## Top 5 Runs by Top-1 Preference Match Rate (%)

| run_id         |   Top-1 Preference Match Rate (%) |   Fitness Score |   Gini Index |   Total Violations |   Runtime (s) |
|----------------|-----------------------------------|-----------------|--------------|--------------------|---------------|
| selectorhc_109 |                            21.569 |            6682 |        0.347 |                 21 |        59.744 |
| selectorhc_130 |                            21.569 |            6687 |        0.351 |                 21 |        47.378 |
| selectorhc_114 |                            20.588 |            7308 |        0.35  |                 23 |        65.47  |
| selectorhc_104 |                            20.588 |            7008 |        0.345 |                 22 |        43.407 |
| selectorhc_124 |                            19.608 |            6160 |        0.358 |                 19 |        47.671 |

## Top 5 Runs by Top-3 Preference Match Rate (%)

| run_id         |   Top-3 Preference Match Rate (%) |   Fitness Score |   Gini Index |   Total Violations |   Runtime (s) |
|----------------|-----------------------------------|-----------------|--------------|--------------------|---------------|
| selectorhc_129 |                            46.078 |            4939 |        0.343 |                 15 |        45.856 |
| selectorhc_114 |                            46.078 |            7308 |        0.35  |                 23 |        65.47  |
| selectorhc_120 |                            45.098 |            7597 |        0.346 |                 24 |        92.42  |
| selectorhc_110 |                            45.098 |            7293 |        0.355 |                 23 |        55.06  |
| selectorhc_108 |                            45.098 |            7066 |        0.357 |                 22 |        72.528 |

## Top 5 Runs by Fairness (Gini Index; lower = fairer)

| run_id         |   Gini Index |   Fitness Score |   Top-1 Preference Match Rate (%) |   Total Violations |   Runtime (s) |
|----------------|--------------|-----------------|-----------------------------------|--------------------|---------------|
| selectorhc_123 |        0.342 |            7316 |                            14.706 |                 23 |        47.371 |
| selectorhc_129 |        0.343 |            4939 |                            12.745 |                 15 |        45.856 |
| selectorhc_104 |        0.345 |            7008 |                            20.588 |                 22 |        43.407 |
| selectorhc_125 |        0.345 |            8209 |                            11.765 |                 26 |        46.584 |
| selectorhc_120 |        0.346 |            7597 |                            15.686 |                 24 |        92.42  |

## Best / Worst Runs by Metric

| Metric                          | Type   | run_id         |   value |
|---------------------------------|--------|----------------|---------|
| Fitness Score                   | Best   | selectorhc_129 | 4939    |
| Fitness Score                   | Worst  | selectorhc_106 | 9428    |
| Penalty                         | Best   | selectorhc_109 |  382    |
| Penalty                         | Worst  | selectorhc_119 |  458    |
| Average Preference Rank         | Best   | selectorhc_109 |    3.75 |
| Average Preference Rank         | Worst  | selectorhc_119 |    4.49 |
| Top-1 Preference Match Rate (%) | Best   | selectorhc_121 |    9.8  |
| Top-1 Preference Match Rate (%) | Worst  | selectorhc_109 |   21.57 |
| Top-3 Preference Match Rate (%) | Best   | selectorhc_119 |   26.47 |
| Top-3 Preference Match Rate (%) | Worst  | selectorhc_129 |   46.08 |
| Gini Index                      | Best   | selectorhc_123 |    0.34 |
| Gini Index                      | Worst  | selectorhc_119 |    0.39 |
| Total Violations                | Best   | selectorhc_129 |   15    |
| Total Violations                | Worst  | selectorhc_106 |   30    |
| Runtime (s)                     | Best   | selectorhc_101 |   26.25 |
| Runtime (s)                     | Worst  | selectorhc_120 |   92.42 |

## Operator Usage (Aggregated)

| Operator       |   Total Used | % Usage   |
|----------------|--------------|-----------|
| Swap           |        25680 | 85.6%     |
| Micro Mutation |         2462 | 8.2%      |
| Reassign       |         1858 | 6.2%      |

## Dataset Hashes Used

- `4226630e`