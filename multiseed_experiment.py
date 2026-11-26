"""One-off multi-seed robustness check: Random vs Nearest-Neighbour informant
selection, averaged over 5 seeds, to give a statistically defensible claim
for the report's Discussion section (a single run per strategy is weak
evidence given PSO's stochasticity).
"""
import json
import os

import numpy as np

from main import RESULTS_DIR, prepare_data, run_pso
from src.pso.informant_strategy import NearestNeighbourInformantSelection, RandomInformantSelection

SEEDS = [0, 1, 2, 3, 4]

def main():
    X_train, X_test, y_train_model, y_train, y_test = prepare_data()

    results = {}
    for name, strategy_cls in [("random", RandomInformantSelection), ("nearest_neighbour", NearestNeighbourInformantSelection)]:
        losses, accuracies = [], []
        for seed in SEEDS:
            m, bf, _ = run_pso(X_train, y_train_model, X_test, y_test, strategy_cls(), seed=seed)
            losses.append(bf)
            accuracies.append(m["accuracy"])
        results[name] = {
            "losses": losses,
            "accuracies": accuracies,
            "loss_mean": float(np.mean(losses)),
            "loss_std": float(np.std(losses)),
            "accuracy_mean": float(np.mean(accuracies)),
            "accuracy_std": float(np.std(accuracies)),
        }
        print(f"{name:18s} loss={np.mean(losses):.4f}+/-{np.std(losses):.4f}  "
              f"accuracy={np.mean(accuracies):.3f}+/-{np.std(accuracies):.3f}  "
              f"(seeds={SEEDS})")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "multiseed_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    names = list(results.keys())
    means = [results[n]["accuracy_mean"] for n in names]
    stds = [results[n]["accuracy_std"] for n in names]
    ax.bar(names, means, yerr=stds, capsize=6, color=["#dc2626", "#2563eb"])
    ax.set_ylabel("Test accuracy (mean +/- std over 5 seeds)")
    ax.set_title("Informant-strategy robustness across 5 seeds")
    ax.set_ylim(0, 1.05)
    plt.savefig(os.path.join(RESULTS_DIR, "multiseed_comparison.png"), dpi=120)
    print("Saved results/multiseed_comparison.png and results/multiseed_comparison.json")


if __name__ == "__main__":
    main()
