"""
Quick smoke test: confirm dataset loads correctly.
"""

import json
from pathlib import Path

# Import load_dataset from GA utils
from genetic_algorithm.ga_utils import load_dataset

# Project root = parent of this scripts folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load config.json
with open(PROJECT_ROOT / "config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

print(" Using dataset path from config.json:", config["paths"]["dataset"])

# Try to load dataset
students, projects, supervisors = load_dataset(config)

print(" Students shape:", students.shape)
print(" Projects shape:", projects.shape)
print(" Supervisors shape:", supervisors.shape)

# Show the first few rows to double check
print("\n--- Students head ---")
print(students.head())

print("\n--- Projects head ---")
print(projects.head())

print("\n--- Supervisors head ---")
print(supervisors.head())
