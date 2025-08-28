### Genetic Algorithm Plots

This folder contains all **figures generated for the Genetic Algorithm (GA)** experiments.  
Plots are produced by `plot_ga_batch.py` and `plot_ga_run.py`.

### Files

- Boxplots → fitness, rank, satisfaction, runtime, violations  
- Histograms → assigned ranks, total violations  
- Convergence curves → fitness, fairness, violations (per-generation)  
- Scatterplots → satisfaction vs fairness, fitness vs violations, etc.  
- Correlation heatmaps  
- Individual run plots (tagged with run IDs)  
- `README.md` → this file  

### Usage

- Plots are used directly in the dissertation for GA results analysis.  
- Each plot file name encodes the metric and run (e.g., `ga_convergence_score.png`, `ga_top1_pct.png`).  
- Companion data for these plots is stored in:  
  - Logs → `results/Genetic_algorithm_runs/`  
  - Summaries → `results/summary/ga_summary.md`  

### Notes

- Figures are auto-generated; do not edit them manually.  
- Safe to delete and regenerate using the plotting scripts (`plot_ga_batch.py`, `plot_ga_run.py`).  
