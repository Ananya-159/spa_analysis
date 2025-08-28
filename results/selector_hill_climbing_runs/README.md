### Selector Hill Climbing Runs

This folder contains the **output logs and summaries** from the *Selector-based Hill Climbing* experiments.

### Files

- `*_selectorhc_alloc_*.csv/xlsx`  
  Per-run allocation outputs (student → project mapping).

- `*_selectorhc_convergence_*.csv/xlsx`  
  Per-run convergence logs (fitness, violations, satisfaction metrics over iterations).

- `selectorhc_summary_log.csv`  
  Aggregated summary log across all Selector HC runs.

- `enhanced_stability_results.csv`  
- `enhanced_stability_results.xlsx`  
  Post-hoc stability analysis and robustness evaluation.

### Notes

- File names are timestamped (UTC) for reproducibility.  
- Allocation files and convergence logs are paired by timestamp.  
- The summary file (`selectorhc_summary_log.csv`) is the primary source for downstream statistical comparison and plotting.  
- Detailed stability results provide robustness insights beyond standard fitness/violation metrics.

