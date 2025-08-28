# spa_analysis/scripts/selector_hc_batch_runner.py

"""
Batch runner for Selector Hill Climbing (Selector-HC) experiments.
Runs 30 seeds (101–130) using selector_hc_main.run_selector_hc()
Saves:
 - Allocation & convergence files to: results/selector_hill_climbing_runs/
 - Summary rows to: results/selector_hill_climbing_runs/selectorhc_summary_log.csv

Usage:
 Run from project root (spa_analysis):
 >>> python scripts/selector_hc_batch_runner.py
 
"""

import sys
from pathlib import Path

# Adding the project root to path
root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))

from selector_hill_climbing.selector_hc_main import run_selector_hc

# Batch runner
if __name__ == "__main__":
    for seed in range(101, 131):  # 30 runs: seeds 101 to 130
        print(f"\n Running Selector-HC seed {seed}...")
        run_selector_hc(seed=seed, results_dir="results/selector_hill_climbing_runs/")
    print("\n ALL 30 Selector-HC runs completed successfully!")
