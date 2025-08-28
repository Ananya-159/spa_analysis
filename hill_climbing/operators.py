"""
Neighbourhood operators (move functions) for Hill Climbing.

Design notes

• The "solution" is represented as a list of project IDs, index-aligned to the
  Students table/order. Example: solution[i] = project_id assigned to student i.
• `preferences` is a list-of-lists: preferences[i] = [Pxxx, Pyyy, ...] (ranked).
• Operators are *pure* functions: they DO NOT mutate the input solution; they
  return a new candidate solution (deep-copied) + a small move metadata dict.

These operators do not perform feasibility checks themselves; they simply
generate neighbours. The fitness function in `core/fitness.py` decides whether
a neighbour is better (lower total score).

Provided operators

1) reassign_student(...)  → change one student's project to a different choice
2) swap_students(...)     → swap the projects between two students
3) greedy_repair(...)     → optional: try a few quick reassignments that reduce
                            violations, guided by your fitness() callback

"""

from __future__ import annotations
import random
import copy
from typing import List, Sequence, Dict, Tuple, Callable, Optional


# Type hints

Solution = List[str]                # e.g., ["P006", "P020", ...] length == num_students
Preferences = List[Sequence[str]]   # e.g., [["P006","P012",...], ["P020","P001",...], ...]
MoveInfo = Dict[str, object]
FitnessFn = Callable[[Solution], Tuple[float, Dict[str, float]]]
# Expected: fitness_fn(solution) -> (total_score, breakdown_dict)
# where breakdown_dict includes keys like: 'pref_penalty', 'capacity_viol', 'elig_viol'
# (Matches your Phase 2 evaluate/CSV schema.)



# Helper: safe deep-copy wrapper

def _clone(sol: Solution) -> Solution:
    """Return a deep copy of the allocation list (defensive)."""
    return copy.deepcopy(sol)



# Operator 1: Reassigning one student to a different project in their list

def reassign_student(
    solution: Solution,
    preferences: Preferences,
    rng: Optional[random.Random] = None,
) -> Tuple[Solution, MoveInfo]:
    """
    Picking a random student and reassigning them to a *different* project from their
    preference list (if any alternative exists).

    Parameters
 
    solution : list[str]
        Current allocation: index = student index, value = project ID.
    preferences : list[list[str]]
        Ranked preferences per student (same indexing as solution).
    rng : random.Random | None
        Optional RNG for reproducibility (e.g., rng = random.Random(seed)).

    Returns

    (new_solution, move_info)
        new_solution : list[str] (deep-copied candidate)
        move_info    : dict with keys: op, student, from, to
    """
    rng = rng or random
    new_sol = _clone(solution)

    # Choosing a random student
    s = rng.randrange(len(new_sol))

    current_proj = new_sol[s]
    # Candidate projects = all preferences except the current project
    alts = [p for p in preferences[s] if p != current_proj]

    # If no alternative is available, return an unchanged copy (no-op neighbour)
    if not alts:
        return new_sol, {"op": "reassign", "student": s, "from": current_proj, "to": current_proj, "note": "no_alt"}

    # Reassigning to a different project from their list
    new_proj = rng.choice(alts)
    new_sol[s] = new_proj

    return new_sol, {"op": "reassign", "student": s, "from": current_proj, "to": new_proj}


# Operator 2: Swapping the the projects of two randomly chosen students

def swap_students(
    solution: Solution,
    rng: Optional[random.Random] = None,
) -> Tuple[Solution, MoveInfo]:
    """
    Swapping the assignments between two distinct students.

    Parameters
  
    solution : list[str]
        Current allocation (length == num_students).
    rng : random.Random | None
        Optional RNG for reproducibility.

    Returns
 
    (new_solution, move_info)
        move_info keys: op, student_a, student_b, proj_a, proj_b
    """
    rng = rng or random
    n = len(solution)

    # If <2 students, just return a copy (no-op)
    if n < 2:
        return _clone(solution), {"op": "swap", "note": "insufficient_students"}

    i, j = rng.sample(range(n), 2)  # two distinct indices
    new_sol = _clone(solution)
    new_sol[i], new_sol[j] = new_sol[j], new_sol[i]

    return new_sol, {
        "op": "swap",
        "student_a": i,
        "student_b": j,
        "proj_a": solution[i],
        "proj_b": solution[j],
    }



# Optional Operator 3: GreedyRepair — quick local fix attempts to reduce the *violations*

def greedy_repair(
    solution: Solution,
    preferences: Preferences,
    fitness_fn: FitnessFn,
    attempts: int = 50,
    rng: Optional[random.Random] = None,
) -> Tuple[Solution, MoveInfo]:
    """
    Trying up to `attempts` random reassignments that *strictly reduce*
    (capacity_viol + elig_viol). This is a pragmatic helper that can be called when:
      • a candidate is promising on preference but still infeasible, or
      • periodically during HC to nudge the search toward feasibility.

    This does NOT guarantee feasibility; it just seeks improvement.

    Parameters
  
    solution : list[str]
        Current allocation.
    preferences : list[list[str]]
        Ranked preferences per student.
    fitness_fn : callable
        Function that returns (total_score, breakdown_dict) for a given solution.
        The breakdown MUST include 'capacity_viol' and 'elig_viol'.
    attempts : int
        Max number of random reassign tries to find a strictly better (violations) state.
    rng : random.Random | None
        Optional RNG for reproducibility.

    Returns
  
    (best_solution, move_info)
        move_info contains: op='greedy_repair', tried, improved, before, after
    """
    rng = rng or random

    # Evaluating the baseline
    _, base = fitness_fn(solution)
    base_viol = float(base.get("capacity_viol", 0)) + float(base.get("elig_viol", 0))

    best_sol = _clone(solution)
    best_viol = base_viol
    improved = False

    for _ in range(max(0, attempts)):
        # Proposing a simple reassign move (prefer the worst-off students would be better,
        # but keeping it simple & fast here). Still require an actual alternative.
        s = rng.randrange(len(best_sol))
        current_proj = best_sol[s]
        alts = [p for p in preferences[s] if p != current_proj]
        if not alts:
            continue

        cand = _clone(best_sol)
        cand[s] = rng.choice(alts)

        # Checking if the violations improved
        _, br = fitness_fn(cand)
        cand_viol = float(br.get("capacity_viol", 0)) + float(br.get("elig_viol", 0))
        if cand_viol < best_viol:
            best_sol = cand
            best_viol = cand_viol
            improved = True

            # Optional early exit: if reached zero violations, stop immediately.
            if best_viol <= 0:
                break

    move_info = {
        "op": "greedy_repair",
        "tried": attempts,
        "improved": improved,
        "before": {"viol_sum": base_viol},
        "after": {"viol_sum": best_viol},
    }
    return best_sol, move_info



# Utility: Randomly picking one of the available operator funcs

def random_operator() -> Callable:
    """
    Returning one of the primitive operators (reassign or swap) uniformly at random.
    Useful inside your HC main loop:

        op = random_operator()
        if op is reassign_student:
            cand, info = op(current, preferences, rng=rng)
        else:
            cand, info = op(current, rng=rng)

    """
    return random.choice([reassign_student, swap_students])



# Some HH-compatible wrappers - optional


def op_swap(sol, rng):
    """Wrapper for HH: calls swap_students and ignores move_info."""
    new_sol, _ = swap_students(sol, rng)
    return new_sol


def op_reassign(sol, prefs, rng):
    """Wrapper for HH: calls reassign_student and ignores move_info."""
    new_sol, _ = reassign_student(sol, prefs, rng)
    return new_sol


def op_mutation_micro(sol, prefs, rng):
    """Wrapper for HH: alias to reassign (for GA-like mutation)."""
    return op_reassign(sol, prefs, rng)
