### Second Round Hill Climbing Plots

This folder contains **visualizations** that summarize the effect of the **second round of Hill Climbing reallocation**.

### Plot Types and Files

1. **Impact Pie (`modeB_impact_pie.png`)**  
   - Shows Improved / Unchanged / Worsened counts for affected students.  
   - Colors: Green (Improved), Yellow (Unchanged), Blue (Worsened).  

2. **Top-1 / Top-3 Satisfaction (`modeB_top1_top3_after_only.png`)**  
   - Bar chart comparing before vs after the second round.  
   - Metrics: % of affected students matched to Top-1 or within Top-3 preferences.  

3. **Assigned Preference Rank Distribution (`modeB_rank_dist_affected_before_after.png`)**  
   - Side-by-side bars for each preference rank (1 = best, etc.).  
   - Shows counts of affected students before vs after the second round.  

4. **Average Preference Rank (`modeB_avg_rank_before_after.png`)**  
   - Compares cohort vs affected students before vs after the second round.  
   - Y-axis: average preference rank (lower is better).  

### Usage

- These plots help to **visualize the impact** of the second round.  
- Each plot filename encodes the mode and metric (e.g., `modeB_*`).  
- Plots are complementary to:  
  - Allocations → `results/Second_round_allocations/`  
  - Logs → `results/Second_round_hill_climbing/`  
  - Summaries → `results/summary/`  

### Notes

- Figures are auto-generated; do not edit manually.  
- Safe to delete and regenerate using the plotting scripts.  
