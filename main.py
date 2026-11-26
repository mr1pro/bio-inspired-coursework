"""Main application: wires MLP + PSO together (section 6 of CW-design.pdf)
and demonstrates the comparisons required by section 7.4: hyperparameter
and informant-strategy comparisons, with convergence plots saved to results/.

TODO (once your team's final dataset is chosen):
  - Replace load_dataset() with real data loading for your chosen dataset.
    Everything else below is dataset-agnostic and does not need to change.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.adapter.mlp_fitness_adapter import MLPFitnessAdapter
from src.mlp.builder import MLPBuilder
from src.pso.informant_strategy import NearestNeighbourInformantSelection, RandomInformantSelection
from src.pso.population import Population

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

TASK = "multiclass_classification"
HIDDEN_LAYERS = [8, 5]
ACTIVATIONS = ["tanh", "sigmoid"]


def load_dataset():
    """Placeholder: sklearn's Iris dataset, used so this script is runnable
    end-to-end today. Swap this for the team's real coursework dataset --
    nothing else in this file needs to change.
    """
    data = load_iris()
    return data.data, data.target


def decode_predictions(predictions: np.ndarray, task: str) -> np.ndarray:
    """Convert network output probabilities into class labels.

    Binary classification: threshold the single sigmoid output at 0.5.
    Multiclass classification: take the argmax over the softmax outputs.
    """
    if task == "binary_classification":
        return (predictions.ravel() >= 0.5).astype(int)
    return np.argmax(predictions, axis=1)


def decode_true_labels(y: np.ndarray) -> np.ndarray:
    """y may be one-hot (2D) or raw integer labels (1D) -- normalise to 1D."""
    return np.argmax(y, axis=1) if y.ndim > 1 else y


def prepare_data():
    """Steps 1-2: load, split, and preprocess (scaler fit on TRAIN only)."""
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    if TASK == "multiclass_classification":
        encoder = OneHotEncoder(sparse_output=False)
        y_train_model = encoder.fit_transform(y_train.reshape(-1, 1))
    else:
        y_train_model = y_train.reshape(-1, 1).astype(float)

    return X_train, X_test, y_train_model, y_train, y_test


def build_network(X_train, y_train_model):
    """Step 3: configure and build the MLP via the Builder pattern."""
    builder = MLPBuilder()
    builder.set_dataset(X_train, y_train_model)
    builder.set_configuration({
        "hidden_layers": HIDDEN_LAYERS,
        "activations": ACTIVATIONS,
        "task": TASK,
    })
    return builder.build()


def evaluate_on_test(network, X_test, y_test):
    predictions = network.predict(X_test)
    predicted_labels = decode_predictions(predictions, TASK)
    true_labels = decode_true_labels(y_test)
    return {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision": precision_score(true_labels, predicted_labels, average="macro", zero_division=0),
        "recall": recall_score(true_labels, predicted_labels, average="macro", zero_division=0),
        "f1": f1_score(true_labels, predicted_labels, average="macro", zero_division=0),
    }


def run_pso(X_train, y_train_model, X_test, y_test, informant_strategy, seed=1, **pso_overrides):
    """Steps 3-8 for one configuration; returns (metrics, best_fitness, fitness_history)."""
    network, loss = build_network(X_train, y_train_model)
    fitness = MLPFitnessAdapter(network=network, loss=loss, X_train=X_train, y_train=y_train_model)

    pso_kwargs = dict(
        num_particles=30,
        num_informants=3,
        num_iterations=100,
        inertia_weight=0.7,
        cognitive_coefficient=1.5,
        social_coefficient=1.5,
    )
    pso_kwargs.update(pso_overrides)

    population = Population(
        fitness_function=fitness,
        informant_strategy=informant_strategy,
        dimension=network.count_parameters(),
        seed=seed,
        **pso_kwargs,
    )
    best_position, best_fitness = population.optimise()
    network.set_parameters(best_position)
    metrics = evaluate_on_test(network, X_test, y_test)
    return metrics, best_fitness, population.fitness_history


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    X_train, X_test, y_train_model, y_train, y_test = prepare_data()

    # ==========================================
    # Baseline run (steps 3-8 of section 6)
    # ==========================================
    metrics, best_fitness, history = run_pso(
        X_train, y_train_model, X_test, y_test,
        informant_strategy=NearestNeighbourInformantSelection(),
    )
    print("=== Baseline (Nearest-Neighbour informants) ===")
    print("Best training loss:", best_fitness)
    print("Test accuracy:", metrics["accuracy"])
    print("Test precision:", metrics["precision"])
    print("Test recall:", metrics["recall"])
    print("Test F1-score:", metrics["f1"])

    plt.figure()
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Global best training loss")
    plt.title("PSO convergence (baseline)")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "convergence_baseline.png"), dpi=120)
    plt.close()

    # ==========================================
    # Comparison 1 (required by section 7.4): informant-selection strategy
    # ==========================================
    print("\n=== Informant-strategy comparison ===")
    plt.figure()
    for name, strategy in [("random", RandomInformantSelection()),
                            ("nearest_neighbour", NearestNeighbourInformantSelection())]:
        m, bf, hist = run_pso(X_train, y_train_model, X_test, y_test, strategy, seed=1)
        print(f"{name:18s} best_loss={bf:.4f}  test_accuracy={m['accuracy']:.3f}")
        plt.plot(hist, label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Global best training loss")
    plt.title("Informant-strategy comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "convergence_informant_strategies.png"), dpi=120)
    plt.close()

    # ==========================================
    # Comparison 2 (required by section 7.4): a PSO hyperparameter (swarm size)
    # ==========================================
    print("\n=== Swarm-size comparison ===")
    plt.figure()
    for n in (10, 30, 60):
        k = min(3, n - 1)
        m, bf, hist = run_pso(
            X_train, y_train_model, X_test, y_test,
            informant_strategy=NearestNeighbourInformantSelection(),
            seed=1, num_particles=n, num_informants=k,
        )
        print(f"num_particles={n:3d}  best_loss={bf:.4f}  test_accuracy={m['accuracy']:.3f}")
        plt.plot(hist, label=f"N={n}")
    plt.xlabel("Iteration")
    plt.ylabel("Global best training loss")
    plt.title("Swarm-size comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "convergence_swarm_size.png"), dpi=120)
    plt.close()

    print(f"\nConvergence plots saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
