"""
Quick unit test for the fitness function.

Loads a dataset, assigns random projects to students, 
and evaluates the resulting allocation to print total score and breakdown.
"""

import json
from pathlib import Path
import random

import pandas as pd
from core.fitness import evaluate_solution
from core.utils import list_to_alloc_df
from genetic_algorithm.ga_utils import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
with open(PROJECT_ROOT / "config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

students, projects, supervisors = load_dataset(config)

# Random assignment (one valid project per student)
valid_projects = projects["Project ID"].tolist()
chromosome = [random.choice(valid_projects) for _ in range(len(students))]

alloc_df = list_to_alloc_df(chromosome, students, config["num_preferences"])
total, breakdown = evaluate_solution(
    alloc_df, students, projects, supervisors, config, with_fairness=True
)

print("Total:", total)
print("Breakdown:", breakdown)
