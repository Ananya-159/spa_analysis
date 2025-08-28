"""
Objective Function + Constraint Model (single-objective; scoring)

Central "scoring brain" used by ALL algorithms (Naive/Greedy/Random, HC, GA, HH).

Implements:
  - preference_penalty():      SOFT  -> sum of assigned ranks (student satisfaction cost)
  - capacity_violations():     HARD* -> overfill at project/supervisor level (penalised via α)
  - eligibility_violations():  HARD* -> client-eligibility & min GPA & unknown IDs & rank OOB (via β)
  - underfill_penalties():     SOFT  -> shortfall vs min capacities for projects/supervisors (via δ)
  - evaluate_solution():       combines components with WEIGHTS FROM CONFIG, validates schema/ranks,
                               and (optionally) reports fairness (Gini) WITHOUT affecting the total.

* "HARD" here means strongly discouraged via large penalties. The plan
  models a single weighted objective; α, β should be large to operationally enforce feasibility.

Allocation DataFrame expected (produced by the algorithms):
  ['Student ID', 'assigned_project', 'assigned_rank']

Dataset column names used (as in the workbook):
  Students:    'Student ID','Average','Client Based Eligibility','Preference 1'..'Preference N'
  Projects:    'Project ID','Supervisor ID','Project Type','Min Students','Max Students',
               'Minimum Average Required','Second Round Eligible'
  Supervisors: 'Supervisor ID','Max Student Capacity','Min Student Capacity'

Objective to MINIMISE (weights from config.json):
    total = γ·Σ(rank) + α·(over-capacity) + β·(eligibility violations) + δ·(under-fill)

Design notes:
- Unknown Student/Project IDs do not crash; they are counted as eligibility violations (policy).
- Rank bounds are enforced to [1..num_preferences]; out-of-bounds are clipped to worst rank and
  counted as an eligibility violation (policy keeps pipeline robust and consistent).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Union, Tuple

import json
import numpy as np
import pandas as pd

# Fairness reporting is OPTIONAL (only if with_fairness=True); does NOT affect 'total'
try:
    from fairness.metrics import ranks_to_satisfaction, gini
except Exception:  # pragma: no cover
    ranks_to_satisfaction = None
    gini = None


# Weights model + config IO


@dataclass(frozen=True)
class Weights:
    """Penalty weights loaded from config.json (γ, α, β, δ)."""
    gamma_pref: float       # γ  : weight on preference sum (soft satisfaction)
    alpha_capacity: float   # α  : penalty for over-capacity (project/supervisor)  -> HARD via weight
    beta_eligibility: float # β  : penalty for eligibility & join/rank-OOB issues -> HARD via weight
    delta_under: float      # δ  : penalty for under-fill (soft encouragement)


def _load_config(config: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Accept a loaded dict or a path to JSON."""
    if isinstance(config, dict):
        return config
    if isinstance(config, str):
        with open(config, "r", encoding="utf-8") as f:
            return json.load(f)
    raise TypeError("config must be a dict or a path to a JSON file")


def _get_weights(cfg: Dict[str, Any]) -> Weights:
    """Extract weights from cfg['weights'] (γ, α, β, δ)."""
    w = cfg.get("weights", {})
    try:
        return Weights(
            gamma_pref=float(w["gamma_pref"]),
            alpha_capacity=float(w["alpha_capacity"]),
            beta_eligibility=float(w["beta_eligibility"]),
            delta_under=float(w["delta_under"]),
        )
    except KeyError as e:
        missing = {"gamma_pref", "alpha_capacity", "beta_eligibility", "delta_under"} - set(w.keys())
        raise ValueError(f"Missing weight(s) in config['weights']: {sorted(missing)}") from e


def _get_num_prefs(cfg: Dict[str, Any]) -> int:
    """Number of preference columns expected in Students sheet."""
    n = cfg.get("num_preferences", None)
    if not isinstance(n, int) or n <= 0:
        raise ValueError("config['num_preferences'] must be a positive integer")
    return n



# Validation helpers


def _validate_schema(
    alloc_df: pd.DataFrame,
    students_df: pd.DataFrame,
    projects_df: pd.DataFrame,
    supervisors_df: pd.DataFrame,
    num_prefs: int,
) -> None:
    """
    Validate required columns for all input frames; raise clear errors if missing.

    This protects downstream components and ensures the scoring policy is applied to
    a consistent schema. It implements the "schema guard" described in the plan.
    """
    # Allocation
    req_alloc = {"Student ID", "assigned_project", "assigned_rank"}
    missing = req_alloc - set(alloc_df.columns)
    if missing:
        raise ValueError(f"alloc_df missing columns: {sorted(missing)}")

    # Students (ensure Preference 1..N exist according to config['num_preferences'])
    base_stu = {"Student ID", "Average", "Client Based Eligibility"}
    pref_cols = {f"Preference {i}" for i in range(1, num_prefs + 1)}
    req_students = base_stu | pref_cols
    missing = req_students - set(students_df.columns)
    if missing:
        raise ValueError(f"students_df missing columns: {sorted(missing)}")

    # Projects
    req_projects = {
        "Project ID",
        "Supervisor ID",
        "Project Type",
        "Min Students",
        "Max Students",
        "Minimum Average Required",
    }
    missing = req_projects - set(projects_df.columns)
    if missing:
        raise ValueError(f"projects_df missing columns: {sorted(missing)}")

    # Supervisors
    req_sup = {"Supervisor ID", "Max Student Capacity", "Min Student Capacity"}
    missing = req_sup - set(supervisors_df.columns)
    if missing:
        raise ValueError(f"supervisors_df missing columns: {sorted(missing)}")


def _enforce_rank_bounds(alloc_df: pd.DataFrame, num_prefs: int) -> Tuple[pd.DataFrame, int]:
    """
    Ensure assigned_rank ∈ [1..num_prefs].

    POLICY (ties to plan's data hygiene):
    - Out-of-bounds or NaN ranks are CLIPPED to 'num_prefs' (worst rank).
    - Each out-of-bounds incident is reported back so we can count it as an
      ELIGIBILITY VIOLATION (+1) in the β term. This keeps policy consistent:
      bad/unknown ranks degrade feasibility score rather than crashing.
    """
    out = alloc_df.copy()
    rk = pd.to_numeric(out["assigned_rank"], errors="coerce")
    oob_mask = (rk < 1) | (rk > num_prefs) | rk.isna()
    rk = rk.clip(lower=1, upper=num_prefs).fillna(num_prefs)
    out["assigned_rank"] = rk.astype(int)
    return out, int(oob_mask.sum())



# Components


def preference_penalty(alloc_df: pd.DataFrame) -> int:
    """
    SOFT component: Σ(assigned_rank), lower is better (γ multiplies this).
    Implements the "student satisfaction" cost.
    """
    if "assigned_rank" not in alloc_df.columns:
        raise ValueError("alloc_df must include 'assigned_rank' column (1=best rank).")
    return int(pd.to_numeric(alloc_df["assigned_rank"], errors="coerce").fillna(9999).sum())


def capacity_violations(
    alloc_df: pd.DataFrame,
    projects_df: pd.DataFrame,
    supervisors_df: pd.DataFrame,
) -> int:
    """
    HARD (via α): Counts *excess* students beyond capacity limits:
      - Project overfill: assigned > 'Max Students'
      - Supervisor overfill: assigned > 'Max Student Capacity' (via project→supervisor)
    """
    # Project overfill (project-level capacity)
    proj_counts = alloc_df.groupby("assigned_project").size()
    proj_caps = projects_df.set_index("Project ID")["Max Students"]
    proj_over = (proj_counts - proj_caps).clip(lower=0).sum()

    # Supervisor overfill (via mapping Project ID -> Supervisor ID)
    sup_by_proj = projects_df.set_index("Project ID")["Supervisor ID"]
    sup_ids = alloc_df["assigned_project"].map(sup_by_proj)
    sup_counts = sup_ids.value_counts()
    sup_caps = supervisors_df.set_index("Supervisor ID")["Max Student Capacity"]
    sup_over = (sup_counts - sup_caps).clip(lower=0).sum()

    # Return combined overfill units (projects + supervisors)
    return int((proj_over.fillna(0) if hasattr(proj_over, "fillna") else proj_over)
               + (sup_over.fillna(0) if hasattr(sup_over, "fillna") else sup_over))


def eligibility_violations_vectorised(
    alloc_df: pd.DataFrame,
    projects_df: pd.DataFrame,
    students_df: pd.DataFrame,
    extra_rank_oob: int = 0,
) -> int:
    """
    HARD (via β): Vectorised eligibility checks with safe joins.

    Violations counted per allocation row:
      1) Project is 'Client based' but student not eligible.     (Project Type vs Client Based Eligibility)
      2) Student 'Average' < project's 'Minimum Average Required' (GPA minimum per-project)
      3) Unknown Student ID or Project ID (join miss).            (robustness: counts as violation)
      4) Rank out-of-bounds incidents (extra_rank_oob).           (policy from _enforce_rank_bounds)
    """
    # Join alloc → students (bring 'Average', 'Client Based Eligibility')
    stu_cols = ["Student ID", "Average", "Client Based Eligibility"]
    a_stu = alloc_df.merge(students_df[stu_cols], on="Student ID", how="left", indicator="stu_merge")

    # Join alloc → projects (bring 'Project Type', 'Minimum Average Required')
    proj_cols = ["Project ID", "Project Type", "Minimum Average Required"]
    merged = a_stu.merge(
        projects_df[proj_cols],
        left_on="assigned_project",
        right_on="Project ID",
        how="left",
        indicator="proj_merge",
    )

    # Unknown IDs → violation (robustness over crash)
    unknown_mask = (merged["stu_merge"] == "left_only") | (merged["proj_merge"] == "left_only")
    unknown_viol = int(unknown_mask.sum())

    # Client-based mismatch (case-insensitive; workbook uses "Client based")
    proj_type = merged["Project Type"].fillna("").astype(str).str.strip().str.lower()
    is_client_based = proj_type.eq("client based")

    # Robust string→bool conversion for Excel 'TRUE'/'FALSE' (and 1/0, yes/no)
    raw_elig = merged["Client Based Eligibility"]
    student_eligible = (
        raw_elig.astype(str).str.strip().str.lower()
        .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
        .fillna(False)
        .astype(bool)
    )
    client_viol = (is_client_based & ~student_eligible).astype(int)  # +1 per offending row

    # Student Average vs project's Minimum Average Required
    avg = pd.to_numeric(merged["Average"], errors="coerce")
    min_req = pd.to_numeric(merged["Minimum Average Required"], errors="coerce")
    gpa_viol = (avg < min_req).fillna(True).astype(int)  # missing values -> treat as violation

    # Sum all eligibility-related violations + any OOB rank incidents
    return int(client_viol.sum() + gpa_viol.sum() + unknown_viol + int(extra_rank_oob))


def underfill_penalties(
    alloc_df: pd.DataFrame,
    projects_df: pd.DataFrame,
    supervisors_df: pd.DataFrame,
) -> Tuple[int, int, int]:
    """
    SOFT (via δ): Under-fill penalties (encouragement toward min capacities):
      - proj_underfill = Σ max(0, Min Students − assigned_to_project)
      - sup_underfill  = Σ max(0, Min Student Capacity − assigned_to_supervisor)
      - under_cap      = proj_underfill + sup_underfill
    """
    # Project assignments vs project min
    proj_counts = alloc_df["assigned_project"].value_counts()
    proj_df = projects_df.set_index("Project ID")[["Min Students"]].copy()
    proj_df["assigned"] = proj_counts.reindex(proj_df.index).fillna(0).astype(int)
    proj_under = (proj_df["Min Students"].astype(float) - proj_df["assigned"]).clip(lower=0).sum()

    # Supervisor assignments via project → supervisor mapping vs supervisor min
    sup_by_proj = projects_df.set_index("Project ID")["Supervisor ID"]
    sup_ids = alloc_df["assigned_project"].map(sup_by_proj)
    sup_counts = sup_ids.value_counts()
    sup_df = supervisors_df.set_index("Supervisor ID")[["Min Student Capacity"]].copy()
    sup_df["assigned"] = sup_counts.reindex(sup_df.index).fillna(0).astype(int)
    sup_under = (sup_df["Min Student Capacity"].astype(float) - sup_df["assigned"]).clip(lower=0).sum()

    proj_underfill = int(proj_under)
    sup_underfill = int(sup_under)
    under_cap = int(proj_underfill + sup_underfill)
    return proj_underfill, sup_underfill, under_cap



# Master fitness (MINIMISE ME)


def evaluate_solution(
    alloc_df: pd.DataFrame,
    students_df: pd.DataFrame,
    projects_df: pd.DataFrame,
    supervisors_df: pd.DataFrame,
    config: Union[str, Dict[str, Any]],
    with_fairness: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """
    Combine components with weights loaded from config.json.
    Returns: (total, breakdown_dict).

    The returned 'total' implements the single-objective model:
      total = γ·pref + α·capv + β·eligv + δ·under_cap
    Fairness (Gini) is OPTIONAL, reported in breakdown when requested; it does NOT affect 'total'.
    """
    # Load config + weights + expected preference count
    cfg = _load_config(config)
    weights = _get_weights(cfg)
    num_prefs = _get_num_prefs(cfg)

    # Validate schema, then enforce rank bounds (OOB becomes β-eligible)
    _validate_schema(alloc_df, students_df, projects_df, supervisors_df, num_prefs)
    alloc_df, rank_oob_count = _enforce_rank_bounds(alloc_df, num_prefs)

    #Compute components 
    # SOFT: student preference sum
    pref = preference_penalty(alloc_df)

    # HARD via α: over-capacity (projects + supervisors)
    capv = capacity_violations(alloc_df, projects_df, supervisors_df)

    # HARD via β: eligibility (client flag, GPA, unknown IDs, plus rank OOB incidents)
    eligv = eligibility_violations_vectorised(
        alloc_df, projects_df, students_df, extra_rank_oob=rank_oob_count
    )

    # SOFT via δ: under-fill (projects + supervisors)
    proj_underfill, sup_underfill, under_cap = underfill_penalties(
        alloc_df, projects_df, supervisors_df
    )

    #  Weighted total (single objective) 
    total = (
        weights.gamma_pref * pref
        + weights.alpha_capacity * capv
        + weights.beta_eligibility * eligv
        + weights.delta_under * under_cap
    )

    # Breakdown (pipeline expects these keys)
    breakdown: Dict[str, Any] = {
        "pref_penalty": int(pref),
        "capacity_viol": int(capv),
        "elig_viol": int(eligv),
        "proj_underfill": int(proj_underfill),
        "sup_underfill": int(sup_underfill),
        "under_cap": int(under_cap),
        "total": float(total),
    }

    #  Optional fairness reporting (does NOT affect total)
    if with_fairness and (ranks_to_satisfaction is not None) and (gini is not None):
        ranks_arr = alloc_df["assigned_rank"].to_numpy()
        try:
            # Prefer signature that accepts num_prefs (the metrics handles this)
            sat = ranks_to_satisfaction(ranks_arr, num_prefs=num_prefs)
        except TypeError:
            # Fallback if metrics.ranks_to_satisfaction signature differs
            sat = ranks_to_satisfaction(ranks_arr)

        g_val = gini(sat)
        if g_val is not None:
            g_val = float(max(0.0, min(1.0, g_val)))  # clamp for robustness
        breakdown["gini_satisfaction"] = g_val

    return float(total), breakdown
