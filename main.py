"""Train one ANN on the banknote authentication data with PSO and test it.

Examples:
    python main.py                                   # final configuration from the report
    python main.py --hidden 16 --activations relu    # a different network
    python main.py --hidden 8 8 --activations tanh logistic --iterations 120
    python main.py --alpha 0.72 --beta 1.5 --gamma 1.5       # the baseline coefficients
    python main.py --informant-strategy nearest --informants 5
    python main.py --boundary clamp --bound 2
    python main.py --search-activations              # PSO also picks the activations
    python main.py --random-network                  # no training: score a random network
    python main.py --plot                            # save the convergence curve

Run `python main.py --help` for every option. The steps below follow the
order: load data, split, build the ANN, wrap it for PSO, run PSO, put the
best particle back into the ANN, test.
"""

import argparse
import os

import numpy as np

from src.adapter.activation_search_adapter import ActivationSearchAdapter
from src.adapter.mlp_fitness_adapter import MLPFitnessAdapter
from src.data.dataset import Standardiser, load_banknote, stratified_train_test_split
from src.evaluation.metrics import classification_report
from src.mlp.activations import ACTIVATIONS
from src.mlp.builder import MLPBuilder
from src.pso.boundary import BOUNDARY_HANDLERS, make_boundary_handler
from src.pso.informant_strategy import INFORMANT_STRATEGIES, make_informant_strategy
from src.pso.population import Population


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a feedforward ANN with particle swarm optimisation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ann = parser.add_argument_group("ANN")
    ann.add_argument("--hidden", type=int, nargs="*", default=[8],
                     help="neurons in each hidden layer, e.g. --hidden 8 4 (none = no hidden layer)")
    ann.add_argument("--activations", nargs="*", default=None,
                     choices=[a for a in ACTIVATIONS if a != "softmax"],
                     help="activation of each hidden layer; one name is reused for all layers")

    pso = parser.add_argument_group("PSO (symbols as in Algorithm 39)")
    pso.add_argument("--swarm-size", type=int, default=25, help="number of particles")
    pso.add_argument("--iterations", type=int, default=40, help="iterations of the main loop")
    pso.add_argument("--alpha", type=float, default=0.4, help="proportion of velocity retained")
    pso.add_argument("--beta", type=float, default=1.0, help="pull towards the personal best")
    pso.add_argument("--gamma", type=float, default=1.0, help="pull towards the informants' best")
    pso.add_argument("--delta", type=float, default=0.0, help="pull towards the global best")
    pso.add_argument("--epsilon", type=float, default=1.0, help="jump size")
    pso.add_argument("--informants", type=int, default=3, help="informants per particle")
    pso.add_argument("--informant-strategy", choices=list(INFORMANT_STRATEGIES), default="random")
    pso.add_argument("--final-alpha", type=float, default=None,
                     help="if set, alpha decreases linearly to this value")
    pso.add_argument("--max-velocity", type=float, default=None, help="limit on each velocity component")
    pso.add_argument("--init-range", type=float, default=1.0,
                     help="particles start uniformly in [-r, r] in every dimension")
    pso.add_argument("--boundary", choices=list(BOUNDARY_HANDLERS), default="none",
                     help="boundary handling technique")
    pso.add_argument("--bound", type=float, default=5.0, help="walls at +/- this value (if --boundary is used)")

    run = parser.add_argument_group("run")
    run.add_argument("--fitness", choices=["accuracy", "loss"], default="accuracy",
                     help="what PSO maximises: training accuracy, or minus the cross-entropy loss")
    run.add_argument("--search-activations", action="store_true",
                     help="let PSO choose the hidden activations (extra genes in the particle)")
    run.add_argument("--test-fraction", type=float, default=0.3, help="share of the data held out for testing")
    run.add_argument("--split-seed", type=int, default=42, help="seed of the train/test split")
    run.add_argument("--seed", type=int, default=0, help="seed of the PSO run")
    run.add_argument("--random-network", action="store_true",
                     help="skip PSO and evaluate a network with random weights")
    run.add_argument("--plot", action="store_true", help="save results/convergence.png")
    return parser.parse_args()


def print_report(title: str, report: dict) -> None:
    print(f"{title}: accuracy {report['accuracy']:.4f}  precision {report['precision']:.4f}  "
          f"recall {report['recall']:.4f}  F1 {report['f1']:.4f}   "
          f"(TP {report['tp']}, TN {report['tn']}, FP {report['fp']}, FN {report['fn']})")


def main() -> None:
    args = parse_arguments()
    if args.activations is None:
        activations = ["tanh"] * len(args.hidden)
    elif len(args.activations) == 1:
        activations = args.activations * len(args.hidden)
    else:
        activations = args.activations

    # 1. Load the data and make a stratified train/test split.
    X, y = load_banknote()
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X, y, test_fraction=args.test_fraction, seed=args.split_seed
    )
    print(f"Data: {len(y)} samples -> {len(y_train)} training, {len(y_test)} test "
          f"(class 1: {y_train.mean():.3f} of training, {y_test.mean():.3f} of test)")

    # 2. Standardise the inputs with statistics from the training set only.
    scaler = Standardiser().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
    y_train_column = y_train.reshape(-1, 1).astype(float)

    # 3. Build the ANN from the configuration.
    builder = MLPBuilder(rng=np.random.default_rng(args.seed))
    builder.set_dataset(X_train, y_train_column)
    builder.set_configuration({
        "hidden_layers": args.hidden,
        "activations": activations,
        "task": "binary_classification",
    })
    network, loss = builder.build()

    # 4. Wrap the ANN so that PSO can evaluate a particle (this is the coupling).
    if args.search_activations:
        fitness = ActivationSearchAdapter(network, loss, X_train, y_train_column, fitness=args.fitness,
                                          gene_range=(-args.init_range, args.init_range))
    else:
        fitness = MLPFitnessAdapter(network, loss, X_train, y_train_column, fitness=args.fitness)
    print(f"Network: {network.describe()}   parameters: {network.count_parameters()}   "
          f"particle length: {fitness.dimension}")

    if args.random_network:
        # Useful for checking the evaluation code without any optimisation.
        print_report("Random weights, training set", classification_report(
            y_train, fitness.to_labels(network.predict(X_train))))
        print_report("Random weights, test set    ", classification_report(
            y_test, fitness.to_labels(network.predict(X_test))))
        return

    # 5. Configure PSO.
    population = Population(
        fitness_function=fitness,
        dimension=fitness.dimension,
        num_particles=args.swarm_size,
        num_iterations=args.iterations,
        inertia_weight=args.alpha,
        cognitive_coefficient=args.beta,
        social_coefficient=args.gamma,
        global_coefficient=args.delta,
        step_size=args.epsilon,
        num_informants=min(args.informants, args.swarm_size - 1),
        informant_strategy=make_informant_strategy(args.informant_strategy),
        init_bounds=(-args.init_range, args.init_range),
        boundary_handler=make_boundary_handler(args.boundary, -args.bound, args.bound),
        max_velocity=args.max_velocity,
        final_inertia_weight=args.final_alpha,
        track_diversity=True,
        seed=args.seed,
    )

    # 6. Run PSO. Only the training data is used here.
    best_position, best_fitness = population.optimise()
    print(f"PSO: {population.evaluations} fitness evaluations, best fitness {best_fitness:.4f}")

    # 7. Put the best particle into the network.
    fitness.install(best_position)
    print(f"Trained network: {network.describe()}")

    # 8. Measure it on the training set and on the unseen test set.
    print_report("Training set", classification_report(y_train, fitness.to_labels(network.predict(X_train))))
    print_report("Test set    ", classification_report(y_test, fitness.to_labels(network.predict(X_test))))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs("results", exist_ok=True)
        figure, axis = plt.subplots(figsize=(5, 3))
        axis.plot(range(1, len(population.fitness_history) + 1), population.fitness_history)
        axis.set_xlabel("Iteration")
        axis.set_ylabel("Best fitness so far")
        axis.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(os.path.join("results", "convergence.png"), dpi=150)
        print("Saved results/convergence.png")


if __name__ == "__main__":
    main()
