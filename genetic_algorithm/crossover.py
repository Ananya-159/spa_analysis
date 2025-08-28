# genetic_algorithm/crossover.py
"""
crossover.py — GA crossover operator

Implements one-point crossover with repair for the Student–Project Allocation (SPA) problem.  
Combines two parent solutions into two offspring while ensuring feasibility via a repair function.  
Used by the Genetic Algorithm main runner (`ga_main.py`).
"""

from random import randint
from copy import deepcopy

def one_point_crossover_with_repair(parent1, parent2, repair_fn):
    """
    Performs a one-point crossover between two parents (list of project IDs).
    Ensures that offspring are valid via repair.

    Steps:
    - If parents are identical, skip crossover and return repaired clones.
    - Otherwise, perform one-point crossover at a random position (not at ends).
    - Swap tails to create two children.
    - Apply repair function to both children to ensure feasibility.

    Parameters:
        parent1 (list): First parent chromosome (student → project).
        parent2 (list): Second parent chromosome.
        repair_fn (function): Function to repair invalid genes in offspring.

    Returns:
        tuple: (child1, child2), both repaired.
    """

    # If parents are identical → skip crossover to avoid cloning
    if parent1 == parent2:
        return repair_fn(deepcopy(parent1)), repair_fn(deepcopy(parent2))

    # Defensive check: must be at least 3 genes to allow proper one-point crossover
    if len(parent1) < 3:
        return repair_fn(deepcopy(parent1)), repair_fn(deepcopy(parent2))

    # Choosing a crossover point (not at the ends)
    point = randint(1, len(parent1) - 2)

    # Generating offspring by swapping tails after the crossover point
    child1 = deepcopy(parent1[:point] + parent2[point:])
    child2 = deepcopy(parent2[:point] + parent1[point:])

    # Applying repair to ensure valid offspring
    return repair_fn(child1), repair_fn(child2)
