"""Shared code for the experiments: the baseline configuration and the
function that trains and tests one network with one random seed."""

import time
from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy as np

from src.adapter.activation_search_adapter import ActivationSearchAdapter
from src.adapter.mlp_fitness_adapter import MLPFitnessAdapter
from src.data.dataset import Standardiser, load_banknote, stratified_train_test_split
from src.evaluation.metrics import classification_report
from src.mlp.builder import MLPBuilder
from src.pso.boundary import make_boundary_handler
from src.pso.informant_strategy import make_informant_strategy
from src.pso.population import Population

TEST_FRACTION = 0.3   # 70% training / 30% test
SPLIT_SEED = 42       # the split is the same in every run; only PSO's seed changes

# Baseline setting. Every experiment changes one or two of these and leaves
# the rest alone. The reasons for the values are given in the report.
BASELINE: Dict = {
    # --- ANN ---
    "hidden_layers": [8],        # one hidden layer of 8 neurons (4-8-1, D = 49)
    "activations": ["tanh"],     # activation of each hidden layer
    # --- PSO (Algorithm 39 symbols) ---
    "swarm_size": 25,            # about 10 + 2*sqrt(D), the SPSO rule of thumb
    "iterations": 40,            # 25 * 40 = 1000 fitness evaluations
    "alpha": 0.72,               # velocity retained (inertia)
    "beta": 1.5,                 # pull towards the personal best (cognitive)
    "gamma": 1.5,                # pull towards the informants' best (social)
    "delta": 0.0,                # pull towards the global best
    "epsilon": 1.0,              # jump size
    "informants": 3,             # informants per particle
    "informant_strategy": "random",   # random | nearest | ring
    # --- search space ---
    "init_range": 1.0,           # particles start uniformly in [-1, 1]^D
    "boundary": "none",          # none | clamp | reflect | wrap | random
    "bound": 5.0,                # walls at +/- bound when boundary != "none"
    "velocity_init": "half_diff",
    "max_velocity": None,
    "final_alpha": None,         # set to make alpha decrease linearly
    # --- coupling ---
    "fitness": "accuracy",       # accuracy | loss
    "loss": "bce",               # loss used when fitness == "loss": bce | mse
    "search_activations": False, # True: PSO also chooses the hidden activations
}


@lru_cache(maxsize=None)
def get_data(split_seed: int = SPLIT_SEED) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the data, make the stratified 70/30 split and standardise the
    features with statistics from the training part only."""
    X, y = load_banknote()
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X, y, test_fraction=TEST_FRACTION, seed=split_seed
    )
    scaler = Standardiser().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test


def make_config(**overrides) -> Dict:
    """Baseline configuration with some entries replaced."""
    unknown = set(overrides) - set(BASELINE)
    if unknown:
        raise KeyError(f"Unknown configuration keys: {sorted(unknown)}")
    return {**BASELINE, **overrides}


def run_once(
    config: Dict,
    seed: int,
    data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
) -> Dict:
    """Train one network with PSO and test it.

    Args:
        config: a dictionary with the keys of BASELINE.
        seed:   seed for PSO's random numbers.
        data:   (X_train, X_test, y_train, y_test), already standardised.
                Defaults to the fixed 70/30 split.

    Returns:
        A dictionary of results for this run, including the convergence curve.
    """
    X_train, X_test, y_train, y_test = data if data is not None else get_data()
    y_train_column = y_train.reshape(-1, 1).astype(float)

    # 1. Build the network from the configuration.
    builder = MLPBuilder(rng=np.random.default_rng(seed))
    builder.set_dataset(X_train, y_train_column)
    builder.set_configuration({
        "hidden_layers": config["hidden_layers"],
        "activations": config["activations"],
        "task": "binary_classification",
        "loss": config["loss"],
    })
    network, loss = builder.build()

    # 2. Wrap it so that PSO can evaluate it.
    if config["search_activations"]:
        fitness = ActivationSearchAdapter(
            network, loss, X_train, y_train_column, fitness=config["fitness"],
            gene_range=(-config["init_range"], config["init_range"]),
        )
    else:
        fitness = MLPFitnessAdapter(network, loss, X_train, y_train_column, fitness=config["fitness"])

    # 3. Run PSO.
    swarm_size = config["swarm_size"]
    population = Population(
        fitness_function=fitness,
        dimension=fitness.dimension,
        num_particles=swarm_size,
        num_iterations=config["iterations"],
        inertia_weight=config["alpha"],
        cognitive_coefficient=config["beta"],
        social_coefficient=config["gamma"],
        global_coefficient=config["delta"],
        step_size=config["epsilon"],
        num_informants=min(config["informants"], swarm_size - 1),
        informant_strategy=make_informant_strategy(config["informant_strategy"]),
        init_bounds=(-config["init_range"], config["init_range"]),
        boundary_handler=make_boundary_handler(config["boundary"], -config["bound"], config["bound"]),
        velocity_init=config["velocity_init"],
        max_velocity=config["max_velocity"],
        final_inertia_weight=config["final_alpha"],
        track_diversity=True,
        seed=seed,
    )
    started = time.perf_counter()
    best_position, best_fitness = population.optimise()
    elapsed = time.perf_counter() - started

    # 4. Put the best particle into the network and measure it.
    fitness.install(best_position)
    train = classification_report(y_train, fitness.to_labels(network.predict(X_train)))
    test = classification_report(y_test, fitness.to_labels(network.predict(X_test)))

    return {
        "seed": seed,
        "dimension": fitness.dimension,
        "evaluations": population.evaluations,
        "best_fitness": best_fitness,
        "train_accuracy": train["accuracy"],
        "test_accuracy": test["accuracy"],
        "test_precision": test["precision"],
        "test_recall": test["recall"],
        "test_f1": test["f1"],
        "test_tp": test["tp"], "test_tn": test["tn"],
        "test_fp": test["fp"], "test_fn": test["fn"],
        "weight_norm": float(np.sqrt(np.mean(np.square(network.get_parameters())))),
        "final_diversity": population.diversity_history[-1],
        "chosen_activations": "/".join(
            layer.activation.name for layer in network.layers[:-1]
        ),
        "seconds": elapsed,
        "fitness_history": list(population.fitness_history),
        "diversity_history": list(population.diversity_history),
    }
