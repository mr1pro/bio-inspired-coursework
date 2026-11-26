"""Sanity test: wire everything together on sklearn's Iris dataset and
confirm the full MLP+PSO+Adapter pipeline actually runs end-to-end.
Not the coursework's final dataset/report — just a scaffold smoke test.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.adapter.mlp_fitness_adapter import MLPFitnessAdapter
from src.mlp.builder import MLPBuilder
from src.pso.informant_strategy import NearestNeighbourInformantSelection, RandomInformantSelection
from src.pso.population import Population


def test_end_to_end_pipeline_runs_and_beats_random_guessing():
    data = load_iris()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    encoder = OneHotEncoder(sparse_output=False)
    y_train_oh = encoder.fit_transform(y_train.reshape(-1, 1))

    builder = MLPBuilder()
    builder.set_dataset(X_train, y_train_oh)
    builder.set_configuration({
        "hidden_layers": [6],
        "activations": ["tanh"],
        "task": "multiclass_classification",
    })
    network, loss = builder.build()

    assert network.count_parameters() > 0

    fitness = MLPFitnessAdapter(network=network, loss=loss, X_train=X_train, y_train=y_train_oh)

    for strategy in (RandomInformantSelection(), NearestNeighbourInformantSelection()):
        population = Population(
            fitness_function=fitness,
            informant_strategy=strategy,
            num_particles=20,
            num_informants=3,
            num_iterations=40,
            dimension=network.count_parameters(),
            seed=0,
        )
        best_position, best_fitness = population.optimise()
        assert best_position.shape[0] == network.count_parameters()
        assert np.isfinite(best_fitness)

        network.set_parameters(best_position)
        predictions = network.predict(X_test)
        predicted_labels = np.argmax(predictions, axis=1)
        accuracy = np.mean(predicted_labels == y_test)

        # Weak sanity bound: PSO-trained net should clearly beat random (1/3) guessing.
        assert accuracy > 0.5, f"accuracy too low: {accuracy}"


if __name__ == "__main__":
    test_end_to_end_pipeline_runs_and_beats_random_guessing()
    print("Smoke test passed.")
