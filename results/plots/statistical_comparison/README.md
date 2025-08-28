### Statistical Comparison Plots

This folder contains **visualizations** comparing the performance of the 3 algorithms —  
**Genetic Algorithm (GA)**, **Hill Climbing (HC)**, and **Selector-based Hill Climbing (Selector-HC)**.

### Plot Types and Files

- **bar_runtime.png** → Bar chart of average runtime (GA vs HC vs Selector-HC).
- **box_avg_rank.png** → Boxplot of average assigned rank across algorithms.
- **box_fitness.png** → Boxplot of overall fitness scores.
- **box_gini.png** → Boxplot of fairness (Gini index).
- **box_violations.png** → Boxplot of constraint violations.
- **heatmap_traffic_light.png** → "Traffic-light" style statistical significance heatmap (pairwise comparisons).
- **scatter_gini_vs_violations.png** → Scatterplot showing trade-off between fairness and violations.
- **strip_top1.png** → Strip plot of % Top-1 satisfaction (GA vs HC vs Selector-HC).
- **strip_top3.png** → Strip plot of % Top-3 satisfaction (GA vs HC vs Selector-HC).
- **pref_match_distribution.png** → Bar plot of Preference Match Succcess Rate by Algorithm
- **supervisor_loads_top12.png** → Bar plot of Supervisor Workload
- **avg_vs_project** → Box plot of Student Average vs Allocated Project by Algorithm 

### Usage

- Use these plots to **compare algorithms** across multiple metrics (fitness, fairness, runtime, satisfaction, violations).
- Complementary to statistical test reports in:  
  - `results/summary/statistical_comparison_tests.md`

### Notes

- Figures are auto-generated; do not edit manually.
- Used directly in the dissertation for **cross-method comparisons**.
