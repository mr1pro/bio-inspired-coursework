"""Definitions of the experiments.

Each function returns a list of settings. A setting is a dictionary with

    label    a short unique name, used in tables and plots
    factors  the values of the things being varied (for plotting)
    config   the full configuration passed to experiments.common.run_once

Every setting is run with the same list of seeds (0 .. runs-1) and on the same
train/test split, so the only thing that differs between two settings is the
configuration itself.

Two PSO parameter sets appear throughout:

    STANDARD  alpha = 0.72, beta = gamma = 1.5. The usual "constricted" values
              from the literature (Clerc and Kennedy, 2002).
    TUNED     the best cell of the alpha / (beta + gamma) grid in the
              `coefficients` experiment, chosen by TRAINING accuracy.

Running an experiment under both shows whether a conclusion depends on how
well the swarm itself is set up.
"""

from typing import Callable, Dict, List

from experiments.common import make_config

Setting = Dict

STANDARD = {"alpha": 0.72, "beta": 1.5, "gamma": 1.5}
TUNED = {"alpha": 0.4, "beta": 1.0, "gamma": 1.0}
PSO_SETS = {"standard": STANDARD, "tuned": TUNED}


def _setting(label: str, factors: Dict, **overrides) -> Setting:
    return {"label": label, "factors": factors, "config": make_config(**overrides)}


def _arch_name(hidden: List[int]) -> str:
    """[8, 8] -> '4-8-8-1' (4 inputs and 1 output are fixed by the data)."""
    return "-".join(str(n) for n in [4, *hidden, 1])


# ----------------------------------------------------------------------
# Question 1: ANN architecture
# ----------------------------------------------------------------------
def architecture() -> List[Setting]:
    """Number of hidden layers and neurons per layer (tanh hidden units)."""
    shapes = [[], [2], [4], [8], [16], [32], [4, 4], [8, 8], [16, 16], [8, 8, 8]]
    settings = []
    for pso_name, pso in PSO_SETS.items():
        for hidden in shapes:
            settings.append(_setting(
                f"{pso_name} | {_arch_name(hidden)}",
                {"pso": pso_name, "architecture": _arch_name(hidden),
                 "hidden_layers": len(hidden), "hidden_neurons": sum(hidden)},
                hidden_layers=hidden, activations=["tanh"] * len(hidden), **pso,
            ))
    return settings


def activation() -> List[Setting]:
    """Hidden activation function, for one and for two hidden layers of 8."""
    names = ["logistic", "tanh", "relu", "leaky_relu", "elu", "linear"]
    settings = []
    for pso_name, pso in PSO_SETS.items():
        for hidden in ([8], [8, 8]):
            for name in names:
                settings.append(_setting(
                    f"{pso_name} | {_arch_name(hidden)} | {name}",
                    {"pso": pso_name, "architecture": _arch_name(hidden), "activation": name},
                    hidden_layers=hidden, activations=[name] * len(hidden), **pso,
                ))
    return settings


# ----------------------------------------------------------------------
# Question 2: how to spend a fixed number of fitness evaluations
# ----------------------------------------------------------------------
def budget() -> List[Setting]:
    """Swarm size against number of iterations for a fixed product."""
    settings = []
    for pso_name, pso in PSO_SETS.items():
        for total in (500, 1000, 2000):
            for swarm_size in (5, 10, 25, 50, 100, 250):
                iterations = total // swarm_size
                settings.append(_setting(
                    f"{pso_name} | budget {total} | {swarm_size}x{iterations}",
                    {"pso": pso_name, "budget": total,
                     "swarm_size": swarm_size, "iterations": iterations},
                    swarm_size=swarm_size, iterations=iterations, **pso,
                ))
    return settings


# ----------------------------------------------------------------------
# Question 3: coefficients of the velocity update
# ----------------------------------------------------------------------
def coefficients() -> List[Setting]:
    """Grid over alpha (inertia) and phi = beta + gamma with beta = gamma."""
    settings = []
    for alpha in (0.0, 0.2, 0.4, 0.6, 0.72, 0.9, 1.0):
        for phi in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
            settings.append(_setting(
                f"alpha {alpha} | phi {phi}",
                {"alpha": alpha, "phi": phi},
                alpha=alpha, beta=phi / 2, gamma=phi / 2,
            ))
    return settings


def balance() -> List[Setting]:
    """Cognitive (beta) against social (gamma) pull with their sum fixed."""
    settings = []
    for pso_name, pso in PSO_SETS.items():
        phi = pso["beta"] + pso["gamma"]
        for share in (0.0, 1 / 6, 1 / 3, 1 / 2, 2 / 3, 5 / 6, 1.0):
            beta = round(phi * share, 4)
            gamma = round(phi - beta, 4)
            settings.append(_setting(
                f"{pso_name} | beta {beta:g} gamma {gamma:g}",
                {"pso": pso_name, "phi": phi, "cognitive_share": round(share, 4),
                 "beta": beta, "gamma": gamma},
                alpha=pso["alpha"], beta=beta, gamma=gamma,
            ))
    return settings


def global_best() -> List[Setting]:
    """Moving social pull from the informants' best (gamma) to the global
    best (delta), keeping gamma + delta fixed."""
    settings = []
    for pso_name, pso in PSO_SETS.items():
        social = pso["gamma"]
        for share in (0.0, 0.25, 0.5, 0.75, 1.0):
            delta = round(social * share, 4)
            gamma = round(social - delta, 4)
            settings.append(_setting(
                f"{pso_name} | gamma {gamma:g} delta {delta:g}",
                {"pso": pso_name, "global_share": share, "gamma": gamma, "delta": delta},
                alpha=pso["alpha"], beta=pso["beta"], gamma=gamma, delta=delta,
            ))
    return settings


def jump_size() -> List[Setting]:
    """Epsilon, the jump size of line 26."""
    settings = []
    for pso_name, pso in PSO_SETS.items():
        for epsilon in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
            settings.append(_setting(
                f"{pso_name} | epsilon {epsilon}",
                {"pso": pso_name, "epsilon": epsilon},
                epsilon=epsilon, **pso,
            ))
    return settings


# ----------------------------------------------------------------------
# Further experiments
# ----------------------------------------------------------------------
def informants() -> List[Setting]:
    """Number of informants and how they are chosen."""
    settings = []
    for pso_name, pso in PSO_SETS.items():
        for strategy in ("random", "nearest", "ring"):
            for k in (1, 2, 3, 5, 8, 12, 24):
                settings.append(_setting(
                    f"{pso_name} | {strategy} k={k}",
                    {"pso": pso_name, "strategy": strategy, "informants": k},
                    informant_strategy=strategy, informants=k, **pso,
                ))
    return settings


def boundary() -> List[Setting]:
    """Boundary handling technique and position of the walls."""
    settings = []
    for pso_name, pso in PSO_SETS.items():
        settings.append(_setting(
            f"{pso_name} | none", {"pso": pso_name, "boundary": "none", "bound": float("inf")}, **pso,
        ))
        for handler in ("clamp", "reflect", "wrap", "random"):
            for bound in (1.0, 2.0, 5.0, 10.0):
                settings.append(_setting(
                    f"{pso_name} | {handler} +/-{bound:g}",
                    {"pso": pso_name, "boundary": handler, "bound": bound},
                    boundary=handler, bound=bound, **pso,
                ))
    return settings


def velocity() -> List[Setting]:
    """Velocity limit and the way the initial velocity is chosen."""
    settings = []
    for pso_name, pso in PSO_SETS.items():
        for limit in (None, 0.1, 0.25, 0.5, 1.0, 2.0):
            settings.append(_setting(
                f"{pso_name} | vmax {limit}",
                {"pso": pso_name, "max_velocity": "none" if limit is None else limit,
                 "velocity_init": "half_diff"},
                max_velocity=limit, **pso,
            ))
        settings.append(_setting(
            f"{pso_name} | zero initial velocity",
            {"pso": pso_name, "max_velocity": "none", "velocity_init": "zero"},
            velocity_init="zero", **pso,
        ))
    return settings


def inertia_schedule() -> List[Setting]:
    """Constant inertia against a linearly decreasing one."""
    schedules = [
        ("constant 0.72", 0.72, None),
        ("constant 0.4", 0.4, None),
        ("0.9 -> 0.4", 0.9, 0.4),
        ("0.9 -> 0.2", 0.9, 0.2),
        ("0.72 -> 0.2", 0.72, 0.2),
    ]
    settings = []
    for phi in (2.0, 3.0):
        for name, start, end in schedules:
            settings.append(_setting(
                f"phi {phi:g} | alpha {name}",
                {"phi": phi, "schedule": name},
                alpha=start, final_alpha=end, beta=phi / 2, gamma=phi / 2,
            ))
    return settings


def fitness_function() -> List[Setting]:
    """Accuracy as the fitness (the coursework's choice) against a loss."""
    measures = [("accuracy", "accuracy", "bce"), ("cross-entropy", "loss", "bce"), ("MSE", "loss", "mse")]
    settings = []
    for pso_name, pso in PSO_SETS.items():
        for iterations in (20, 40, 120):
            for name, fitness, loss in measures:
                settings.append(_setting(
                    f"{pso_name} | {25 * iterations} evals | {name}",
                    {"pso": pso_name, "budget": 25 * iterations, "fitness": name},
                    fitness=fitness, loss=loss, iterations=iterations, **pso,
                ))
    return settings


def activation_search() -> List[Setting]:
    """PSO chooses the hidden activations itself, against fixed tanh."""
    settings = []
    for pso_name, pso in PSO_SETS.items():
        for iterations in (40, 120):
            for hidden in ([8], [8, 8]):
                for search in (False, True):
                    mode = "PSO-chosen" if search else "fixed tanh"
                    settings.append(_setting(
                        f"{pso_name} | {25 * iterations} evals | {_arch_name(hidden)} | {mode}",
                        {"pso": pso_name, "budget": 25 * iterations,
                         "architecture": _arch_name(hidden), "activations": mode},
                        hidden_layers=hidden, activations=["tanh"] * len(hidden),
                        search_activations=search, iterations=iterations, **pso,
                    ))
    return settings


EXPERIMENTS: Dict[str, Callable[[], List[Setting]]] = {
    "coefficients": coefficients,
    "balance": balance,
    "global_best": global_best,
    "jump_size": jump_size,
    "architecture": architecture,
    "activation": activation,
    "budget": budget,
    "informants": informants,
    "boundary": boundary,
    "velocity": velocity,
    "inertia_schedule": inertia_schedule,
    "fitness_function": fitness_function,
    "activation_search": activation_search,
}
