# spa_analysis/selector_hill_climbing/controller.py

"""
Controller for the Selector Hill Climbing (Selector-HC) framework.

This module implements two key components:
- ChoiceFunction: selects among low-level hill climbing operators using ε-greedy or softmax selection based on recent performance.
- Acceptance: decides whether to accept a candidate solution based on a greedy or simulated annealing strategy.

Used by `selector_hc_main.py` to coordinate operator selection and solution acceptance
during the adaptive hill climbing search process.

This module now clearly represents a selection-based metaheuristic specific to hill climbing variants.
"""


from dataclasses import dataclass
import math
import random


# Configuration for the choice function (how to select operators)
@dataclass
class ChoiceConfig:
    epsilon: float = 0.1       # ε for ε-greedy (probability of random operator)
    window: int = 50           # Number of past iterations to track performance
    softmax_tau: float = None  # Temperature for softmax (None = disable softmax)


class ChoiceFunction:
    """
    Chooses among low-level heuristics (LLHs) based on recent performance.
    Supports ε-greedy and optional softmax selection.
    """

    def __init__(self, op_names, cfg: ChoiceConfig):
        self.cfg = cfg
        self.ops = op_names
        self.hist = {op: [] for op in op_names}  # Tracks Δfitness for each operator

    def update(self, op, delta_fitness):
        """
        Stores recent performance of operator (negative delta = improvement).
        """
        self.hist[op].append(delta_fitness)
        if len(self.hist[op]) > self.cfg.window:
            self.hist[op].pop(0)

    def _score(self, op):
        """
        Computes the average reward (recent Δfitness) for the operator.
        """
        history = self.hist[op]
        return sum(history) / len(history) if history else 0.0

    def choose(self, rng: random.Random):
        """
        Chooses the the next operator to apply using ε-greedy or softmax.
        """
        if self.cfg.softmax_tau:
            # Softmax selection
            scores = [self._score(op) for op in self.ops]
            max_score = max(scores) if scores else 0.0
            exps = [math.exp((s - max_score) / self.cfg.softmax_tau) for s in scores]
            total = sum(exps) or 1.0

            r = rng.random()
            acc = 0.0
            for op, weight in zip(self.ops, exps):
                acc += weight / total
                if r <= acc:
                    return op
            return self.ops[-1]  # fallback

        # ε-greedy selection
        if rng.random() < self.cfg.epsilon:
            return rng.choice(self.ops)
        return max(self.ops, key=self._score)


# Configuration for acceptance strategy
@dataclass
class AcceptConfig:
    scheme: str = "better"  # "better" or "sa"
    T0: float = 1.0         # Starting temperature (for simulated annealing)
    decay: float = 0.995    # Temperature decay rate


class Acceptance:
    """
    Determines whether to accept a candidate solution.
    Supports:
        - "better": greedy (only accept if better)
        - "sa": simulated annealing (accept worse with some probability)
    """

    def __init__(self, cfg: AcceptConfig):
        self.cfg = cfg
        self.T = cfg.T0  # Current temperature (if using SA)

    def accept(self, delta_fitness, rng: random.Random) -> bool:
        """
        Decide whether to accept the candidate solution.

        delta_fitness: f_new - f_old
        If delta < 0 → improvement (we minimize)
        """
        if self.cfg.scheme == "better":
            return delta_fitness < 0

        # Simulated Annealing (SA-lite)
        if delta_fitness < 0:
            return True  # Always accept improvements

        # Accept worse solution with probability ~ exp(-Δf / T)
        prob = math.exp(-delta_fitness / max(self.T, 1e-9))
        self.T *= self.cfg.decay  # Cool down
        return rng.random() < prob
