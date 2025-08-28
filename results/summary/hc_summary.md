# Hill Climbing Summary Results

This summary aggregates **30 HC Random** and **30 HC Greedy** runs.

## Mean ± Standard Deviation by Algorithm

| Algorithm   | Fitness Score   | Penalty   | Avg Pref Rank   | Top-1 Match (%)   | Top-3 Match (%)   | Gini Index   | Total Violations   | Runtime (s)   |
|:------------|:----------------|:----------|:----------------|:------------------|:------------------|:-------------|:-------------------|:--------------|
| HC Greedy   | 13042.43        | 323.77    | 3.17            | 29.22             | 58.56             | 0.28         | 13.63              | 57.67         |
|             | ±2132.43        | ±18.60    | ±0.18           | ±3.63             | ±4.39             | ±0.02        | ±2.01              | ±39.86        |
| HC Random   | 14108.20        | 368.20    | 3.61            | 20.36             | 50.29             | 0.32         | 14.07              | 56.13         |
|             | ±1688.42        | ±17.34    | ±0.17           | ±4.04             | ±4.31             | ±0.02        | ±1.72              | ±35.39        |

## Median (Interquartile Range) by Algorithm

| Algorithm   | Fitness Score   | Penalty   | Avg Pref Rank   | Top-1 Match (%)   | Top-3 Match (%)   | Gini Index    | Total Violations   | Runtime (s)   |
|:------------|:----------------|:----------|:----------------|:------------------|:------------------|:--------------|:-------------------|:--------------|
| HC Greedy   | 13326           | 320       | 3.137           | 29.4              | 59.3              | 0.281         | 14                 | 44            |
|             | [11374–14316]   | [308–338] | [3.020–3.319]   | [26.7–31.4]       | [54.9–61.8]       | [0.264–0.304] | [12–15]            | [32–53]       |
| HC Random   | 14366           | 368       | 3.613           | 19.6              | 50.5              | 0.322         | 14                 | 43            |
|             | [13364–15370]   | [356–381] | [3.495–3.738]   | [18.6–22.5]       | [46.3–53.7]       | [0.311–0.335] | [13–15]            | [36–57]       |

### Top 5 Runs by Fitness (lower = better)

| run_id   | Algorithm   |   Fitness Score |   Top-1 Match (%) |   Gini Index |   Total Violations |   Runtime (s) |
|:---------|:------------|----------------:|------------------:|-------------:|-------------------:|--------------:|
| 67ab7573 | HC Greedy   |            9377 |           26.4706 |     0.29534  |                 11 |         34.03 |
| 2ca57605 | HC Greedy   |           10339 |           28.4314 |     0.309621 |                 10 |         25.35 |
| 2316ec80 | HC Greedy   |           10363 |           22.549  |     0.324423 |                 10 |         42.88 |
| 9d2bbbda | HC Random   |           10364 |           20.5882 |     0.317983 |                 10 |         53.38 |
| 3ace27aa | HC Greedy   |           10372 |           24.5098 |     0.315405 |                 11 |         29.61 |

### Top 5 Runs by Top-1 Preference Match Rate (%)

| run_id   | Algorithm   |   Fitness Score |   Top-1 Match (%) |   Gini Index |   Total Violations |   Runtime (s) |
|:---------|:------------|----------------:|------------------:|-------------:|-------------------:|--------------:|
| 46e917a4 | HC Greedy   |           16331 |           36.2745 |     0.282951 |                 17 |         29.24 |
| 2896a468 | HC Greedy   |           14313 |           34.3137 |     0.280206 |                 14 |         25.59 |
| 78d0667a | HC Greedy   |           18304 |           33.3333 |     0.268771 |                 18 |        130.91 |
| fa342de1 | HC Greedy   |           16304 |           33.3333 |     0.26021  |                 16 |         30.09 |
| 054940d4 | HC Greedy   |           13349 |           32.3529 |     0.268289 |                 15 |         30.75 |

### Top 5 Runs by Top-3 Preference Match Rate (%)

| run_id   | Algorithm   |   Fitness Score |   Top-3 Match (%) |   Gini Index |   Total Violations |   Runtime (s) |
|:---------|:------------|----------------:|------------------:|-------------:|-------------------:|--------------:|
| 78d0667a | HC Greedy   |           18304 |           65.6863 |     0.268771 |                 18 |        130.91 |
| 297369d9 | HC Greedy   |           14321 |           64.7059 |     0.260908 |                 15 |         36.17 |
| 89bb2cca | HC Greedy   |           13325 |           64.7059 |     0.261158 |                 14 |         95.53 |
| 3f6c6e09 | HC Greedy   |           13348 |           63.7255 |     0.264609 |                 15 |         47.54 |
| 975ab178 | HC Greedy   |           14317 |           63.7255 |     0.242324 |                 15 |         49.58 |

### Top 5 Runs by Fairness (Gini; lower = better)

| run_id   | Algorithm   |   Fitness Score |   Top-1 Match (%) |   Gini Index |   Total Violations |   Runtime (s) |
|:---------|:------------|----------------:|------------------:|-------------:|-------------------:|--------------:|
| 975ab178 | HC Greedy   |           14317 |           29.4118 |     0.242324 |                 15 |         49.58 |
| 4133498b | HC Greedy   |           13328 |           28.4314 |     0.258428 |                 14 |         57.44 |
| fa342de1 | HC Greedy   |           16304 |           33.3333 |     0.26021  |                 16 |         30.09 |
| 297369d9 | HC Greedy   |           14321 |           32.3529 |     0.260908 |                 15 |         36.17 |
| 89bb2cca | HC Greedy   |           13325 |           30.3922 |     0.261158 |                 14 |         95.53 |

### Top 5 Runs by Violations (lower = better)

| run_id   | Algorithm   |   Fitness Score |   Top-1 Match (%) |   Gini Index |   Total Violations |   Runtime (s) |
|:---------|:------------|----------------:|------------------:|-------------:|-------------------:|--------------:|
| 9268f16e | HC Random   |           10386 |           16.6667 |     0.34499  |                 10 |         77.46 |
| 9d2bbbda | HC Random   |           10364 |           20.5882 |     0.317983 |                 10 |         53.38 |
| 2316ec80 | HC Greedy   |           10363 |           22.549  |     0.324423 |                 10 |         42.88 |
| 2ca57605 | HC Greedy   |           10339 |           28.4314 |     0.309621 |                 10 |         25.35 |
| 3ace27aa | HC Greedy   |           10372 |           24.5098 |     0.315405 |                 11 |         29.61 |

## Best/Worst by Metric

| Metric           | Type   | run_id   | Algorithm   |     value |
|:-----------------|:-------|:---------|:------------|----------:|
| Fitness Score    | Best   | 67ab7573 | HC Greedy   |  9377     |
| Fitness Score    | Worst  | 6e414435 | HC Random   | 18316     |
| Penalty          | Best   | 975ab178 | HC Greedy   |   297     |
| Penalty          | Worst  | 1f1ddc07 | HC Random   |   394     |
| Avg Pref Rank    | Best   | 975ab178 | HC Greedy   |     2.912 |
| Avg Pref Rank    | Worst  | 1f1ddc07 | HC Random   |     3.863 |
| Top-1 Match (%)  | Best   | 46e917a4 | HC Greedy   |    36.275 |
| Top-1 Match (%)  | Worst  | 6ef7cb2c | HC Random   |    12.745 |
| Top-3 Match (%)  | Best   | 78d0667a | HC Greedy   |    65.686 |
| Top-3 Match (%)  | Worst  | 1f1ddc07 | HC Random   |    43.137 |
| Gini Index       | Best   | 975ab178 | HC Greedy   |     0.242 |
| Gini Index       | Worst  | 426cc1c2 | HC Random   |     0.359 |
| Total Violations | Best   | 9268f16e | HC Random   |    10     |
| Total Violations | Worst  | 6e414435 | HC Random   |    18     |
| Runtime (s)      | Best   | 1eb4d10f | HC Greedy   |    23.32  |
| Runtime (s)      | Worst  | 1acc53a2 | HC Greedy   |   148.24  |

_Auto-generated from `hc_summary_log.csv`_
