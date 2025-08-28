# genetic_algorithm/mutation.py
"""
mutation.py — GA mutation operator

Implementing the mutation operators for the Genetic Algorithm in the Student-Project Allocation (SPA) problem.  
Supports per-gene preference changes, occasional resets to random projects, and gene swaps to maintain diversity.  
After mutation, a repair function is applied to ensure feasibility of the chromosome.
"""
import random
from copy import deepcopy
import pandas as pd


def _collect_valid_projects_from_students(student_df):
    """
    Build a set of all project IDs that appear in any preference column.
    Handles both 'Preference 1' style and 'Preference_1' style columns.
    """
    pref_cols = [c for c in student_df.columns if c.lower().startswith("preference")]
    valid = set()
    for c in pref_cols:
        for v in student_df[c].dropna().astype(str).str.strip():
            if v:
                valid.add(v)
    return list(valid)


def _get_student_prefs(student_df, i):
    """
    Return this student's preference list (strings, cleaned), supporting both
    'Preference X' and 'Preference_X' columns.
    """
    row = student_df.iloc[i]
    # Collecting the ordered preference columns
    pref_cols_space = [c for c in student_df.columns if c.startswith("Preference ")]
    pref_cols_underscore = [c for c in student_df.columns if c.startswith("Preference_")]

    prefs = []
    cols = pref_cols_space if pref_cols_space else pref_cols_underscore
    # Sorting by the trailing number to preserve order
    def _order_key(c):
        # working for both 'Preference 1' and 'Preference_1'
        n = "".join(ch for ch in c if ch.isdigit())
        return int(n) if n.isdigit() else 9999

    for c in sorted(cols, key=_order_key):
        v = row.get(c)
        if pd.notna(v):
            s = str(v).strip()
            if s:
                prefs.append(s)
    return prefs


def mutate_individual_with_repair(
    individual,
    student_df,
    repair_fn,
    mutation_rate=0.3,
    swap_rate=0.05,
    reset_rate=0.02,
):
    """
    Mutate an individual (list of project IDs) while maintaining diversity.

    Mutations applied:
      1) With probability `mutation_rate` per gene, change assignment to another
         preference (if available), otherwise a random valid project.
      2) With small probability `reset_rate`, reset a gene to any random valid project.
      3) With small probability `swap_rate`, swap two students' assignments.

    Finally, a soft repair is applied via `repair_fn`.

    Parameters
  
    individual : list[str]
        Chromosome (project ID per student, aligned with student_df rows).
    student_df : pd.DataFrame
        Students sheet (used to read preference columns).
    repair_fn : callable
        Callable that fixes only invalid project IDs (keeps diversity).
    mutation_rate : float
        Per-gene probability of doing a preference change.
    swap_rate : float
        Probability of swapping two genes in the chromosome.
    reset_rate : float
        Per-gene probability of resetting to a random valid project.

    Returns

    list[str]
        Mutated + repaired chromosome.
    """
    mutant = deepcopy(individual)
    num_students = len(mutant)

    # Building a robust universe of valid projects from all preference columns.
    # This avoids 'empty sequence' errors and doesn't rely on the current chromosome.
    valid_projects = _collect_valid_projects_from_students(student_df)

    # If still empty (very unusual), fall back to the current genes that are non-empty.
    if not valid_projects:
        valid_projects = [str(p).strip() for p in mutant if pd.notna(p) and str(p).strip()]
        valid_projects = list(sorted(set(valid_projects)))

    # If it's *still* empty, bail out safely (no mutation) and run repair.
    if not valid_projects:
        return repair_fn(mutant)

    # Gene-wise mutations 
    for i in range(num_students):

        # 1) Preference change (most common)
        if random.random() < mutation_rate:
            prefs = _get_student_prefs(student_df, i)
            prefs = [p for p in prefs if p]  # clean empties
            current = str(mutant[i]).strip()

            # Choosing an alternative preference different from current
            alternatives = [p for p in prefs if p != current]
            if alternatives:
                mutant[i] = random.choice(alternatives)
            else:
                # Fallback: any valid project different from current
                fallback = [p for p in valid_projects if p != current]
                if fallback:  # guard against empty
                    mutant[i] = random.choice(fallback)
                # else: leave as-is

        # 2) Occasional hard reset to *any* valid project (diversity kick)
        if random.random() < reset_rate:
            mutant[i] = random.choice(valid_projects)

    # 3) Swap two students' assignments (small prob, adds structural variety)
    if num_students >= 2 and random.random() < swap_rate:
        a, b = random.sample(range(num_students), 2)
        mutant[a], mutant[b] = mutant[b], mutant[a]

    # Final safety: soft repair only replaces truly invalid IDs
    return repair_fn(mutant)
