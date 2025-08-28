### Hill Climbing Plots

This folder contains all **figures generated for the Hill Climbing (HC)** experiments.  
Plots are produced by `plot_hc_batch.py`.

### Files

- Boxplots → fitness, rank, satisfaction, runtime, violations  
- Histograms → assigned ranks, total violations  
- Convergence curves → fitness, fairness, violations (per-generation, average, sampled)  
- Scatterplots → satisfaction vs fairness, fitness vs runtime, fitness vs violations  
- Correlation heatmaps  
- Individual run plots (tagged with run IDs)  
- `README.md` → this file  

### Usage

- Plots are used directly in the dissertation for HC results analysis.  
- Each plot file name encodes the metric and run (e.g., `convergence_score_avg.png`, `top1_pct.png`).  
- Complementary data for these plots is stored in:  
  - Logs → `results/Hill_climbing_runs/`  
  - Summaries → `results/summary/hc_summary.md`  

### Notes

- Figures are auto-generated; do not edit them manually.  
- Safe to delete and regenerate using the plotting script (`plot_hc_batch.py`).  
