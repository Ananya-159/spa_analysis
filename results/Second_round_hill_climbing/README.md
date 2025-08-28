### Second Round Hill Climbing Runs

This folder stores **logs and supporting outputs** of second-round Hill Climbing reallocation experiments.

### Files and structure

- `*_second_round_log.csv`- 
   Per-run logs () with:  
  - Preference penalty  
  - Constraint violation counts  
  - Fitness total  
  - Gini satisfaction  
  - Average preference rank (before/after, affected vs cohort)  
  - Top-1 / Top-3 satisfaction (before/after)  
  - Operator sequences (if relevant)  
  - Runtime and iteration count  

 - Metadata columns link each run back to its **source HC run** (`source_hc_run_id`).  
 
 - `README.md`- this file.

### Usage/Notes

- Use this folder for **auditing / replicating statistics**.  
- Summaries across multiple runs (Option B, aggregate per-run deltas) are generated in:  
  - `results/summary/second_round_across_runs_summary.csv` → CSV summary across runs  
  - `results/summary/second_round_across_runs_summary.md` → Markdown summary across runs  

- Allocation CSVs are present in `results/Second_round_allocations/`.  
- Visualizations are present in `results/plots/Second_round_hill_climbing/`.  
