"""Experiments on the final configuration and two control experiments.

    final       baseline and final configurations on the fixed 70/30 split
    final_cv    the same configurations under repeated stratified 10-fold
                cross-validation (the split changes as well as PSO's seed)
    landscape   PSO on two simple test functions, used to explain why the
                swarm behaves as it does on the ANN problem
"""

from multiprocessing import Pool
from typing import Dict, List, Tuple

import numpy as np

from experiments.common import make_config, run_once
from experiments.suites import STANDARD, TUNED
from src.data.dataset import Standardiser, load_banknote, stratified_k_fold
from src.pso.fitness_function import FitnessFunction
from src.pso.population import Population

# The two configurations compared at the end of the study.
#   baseline : literature defaults, the starting point of every experiment
#   final    : the same network and swarm with the tuned coefficients. The
#              other experiments found no setting that was significantly
#              better than this one, so nothing else is changed.
FINAL_CONFIGS: Dict[str, Dict] = {
    "baseline": make_config(**STANDARD),
    "final": make_config(**TUNED),
}
BUDGET_MULTIPLIERS = (1, 5)   # 1000 and 5000 fitness evaluations


def _scaled(config: Dict, multiplier: int) -> Dict:
    """Same swarm, `multiplier` times as many iterations."""
    return dict(config, iterations=config["iterations"] * multiplier)


def _run(args: Tuple[str, Dict, int, Dict]) -> Dict:
    label, config, seed, factors = args
    return {"label": label, **factors, **run_once(config, seed)}


def final(runs: int, jobs: int) -> List[Dict]:
    """Baseline, tuned and final configurations on the fixed 70/30 split."""
    tasks = []
    for name, config in FINAL_CONFIGS.items():
        for multiplier in BUDGET_MULTIPLIERS:
            scaled = _scaled(config, multiplier)
            budget = scaled["swarm_size"] * scaled["iterations"]
            label = f"{name} | {budget} evals"
            factors = {"configuration": name, "budget": budget}
            tasks += [(label, scaled, seed, factors) for seed in range(runs)]
    with Pool(jobs) as pool:
        return pool.map(_run, tasks)


def _cv_run(args: Tuple[str, Dict, int, int, int, Dict]) -> Dict:
    label, config, repeat, fold, k, factors = args
    X, y = load_banknote()
    train_idx, val_idx = list(stratified_k_fold(y, k=k, seed=repeat))[fold]
    scaler = Standardiser().fit(X[train_idx])   # fitted on the training folds only
    data = (scaler.transform(X[train_idx]), scaler.transform(X[val_idx]), y[train_idx], y[val_idx])
    result = run_once(config, seed=1000 * repeat + fold, data=data)
    return {"label": label, **factors, "repeat": repeat, "fold": fold, **result}


def final_cv(runs: int, jobs: int, k: int = 10, repeats: int = 3) -> List[Dict]:
    """Repeated stratified 10-fold cross-validation of the same configurations."""
    tasks = []
    for name, config in FINAL_CONFIGS.items():
        for multiplier in BUDGET_MULTIPLIERS:
            scaled = _scaled(config, multiplier)
            budget = scaled["swarm_size"] * scaled["iterations"]
            label = f"{name} | {budget} evals"
            factors = {"configuration": name, "budget": budget}
            tasks += [(label, scaled, repeat, fold, k, factors)
                      for repeat in range(repeats) for fold in range(k)]
    with Pool(jobs) as pool:
        return pool.map(_cv_run, tasks)


# ----------------------------------------------------------------------
# Control experiment: the same PSO on functions we understand
# ----------------------------------------------------------------------
class Sphere(FitnessFunction):
    """-|x|^2: one optimum at the origin, so there is a "right" scale."""

    def evaluate(self, position: np.ndarray) -> float:
        return -float(np.sum(position ** 2))


class DirectionOnly(FitnessFunction):
    """Cosine of the angle between x and a fixed target direction. Scaling x
    by any positive number leaves the fitness unchanged, which is also true
    (approximately) of the accuracy of a classifier whose output is
    thresholded."""

    def __init__(self, dimension: int):
        self.target = np.ones(dimension) / np.sqrt(dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return float(position @ self.target / (np.linalg.norm(position) + 1e-12))


def _landscape_run(args: Tuple[str, str, Dict, int]) -> Dict:
    function_name, pso_name, pso, seed = args
    dimension = 49  # same as the 4-8-1 network
    function = Sphere() if function_name == "sphere" else DirectionOnly(dimension)
    population = Population(
        function, dimension, num_particles=25, num_iterations=40,
        inertia_weight=pso["alpha"], cognitive_coefficient=pso["beta"],
        social_coefficient=pso["gamma"], num_informants=3,
        track_diversity=True, seed=seed,
    )
    position, fitness = population.optimise()
    return {
        "label": f"{function_name} | {pso_name}",
        "function": function_name, "pso": pso_name, "seed": seed,
        "best_fitness": fitness,
        "initial_diversity": population.diversity_history[0],
        "final_diversity": population.diversity_history[-1],
        "weight_norm": float(np.sqrt(np.mean(position ** 2))),
        "fitness_history": list(population.fitness_history),
        "diversity_history": list(population.diversity_history),
    }


def landscape(runs: int, jobs: int) -> List[Dict]:
    """PSO on a sphere and on a scale-free function (control experiment)."""
    tasks = [(function_name, pso_name, pso, seed)
             for function_name in ("sphere", "direction")
             for pso_name, pso in (("standard", STANDARD), ("tuned", TUNED))
             for seed in range(runs)]
    with Pool(jobs) as pool:
        return pool.map(_landscape_run, tasks)


EXPERIMENTS = {"final": final, "final_cv": final_cv, "landscape": landscape}
