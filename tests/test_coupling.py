"""Tests for the data handling, the metrics and the PSO-ANN coupling."""

import numpy as np
import pytest

from experiments.common import make_config, run_once
from src.adapter.activation_search_adapter import ActivationSearchAdapter
from src.adapter.mlp_fitness_adapter import MLPFitnessAdapter
from src.data.dataset import Standardiser, load_banknote, stratified_k_fold, stratified_train_test_split
from src.evaluation.metrics import classification_report
from src.mlp.builder import MLPBuilder


# -------------------------------------------------------------------- data
def test_dataset_has_the_documented_shape():
    X, y = load_banknote()
    assert X.shape == (1372, 4)
    assert sorted(np.bincount(y).tolist()) == [610, 762]


def test_split_is_70_30_stratified_and_disjoint():
    X, y = load_banknote()
    X_train, X_test, y_train, y_test = stratified_train_test_split(X, y, 0.3, seed=42)
    assert len(y_train) == 960 and len(y_test) == 412
    assert abs(y_train.mean() - y.mean()) < 0.005 and abs(y_test.mean() - y.mean()) < 0.005
    # Row indices must not overlap: compare by index, not by value, because
    # the file contains some repeated rows.
    indices = np.arange(len(y)).reshape(-1, 1).astype(float)
    i_train, i_test, _, _ = stratified_train_test_split(indices, y, 0.3, seed=42)
    assert set(i_train.ravel()).isdisjoint(i_test.ravel())
    assert len(set(i_train.ravel()) | set(i_test.ravel())) == len(y)


def test_standardiser_uses_training_statistics_only():
    train = np.array([[0.0, 10.0], [2.0, 30.0]])
    scaler = Standardiser().fit(train)
    assert np.allclose(scaler.transform(train).mean(axis=0), 0.0)
    assert np.allclose(scaler.transform(train).std(axis=0), 1.0)
    assert np.allclose(scaler.transform(np.array([[1.0, 20.0]])), 0.0)


def test_k_fold_uses_every_sample_exactly_once_for_validation():
    _, y = load_banknote()
    seen = np.concatenate([validation for _, validation in stratified_k_fold(y, k=10, seed=1)])
    assert sorted(seen.tolist()) == list(range(len(y)))
    for train, validation in stratified_k_fold(y, k=10, seed=1):
        assert set(train).isdisjoint(validation)
        assert abs(y[validation].mean() - y.mean()) < 0.02


# ----------------------------------------------------------------- metrics
def test_metrics_on_a_hand_worked_example():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    y_pred = np.array([1, 1, 0, 0, 0, 1, 0, 1])
    report = classification_report(y_true, y_pred)
    assert (report["tp"], report["tn"], report["fp"], report["fn"]) == (3, 3, 1, 1)
    assert np.isclose(report["accuracy"], 6 / 8)
    assert np.isclose(report["precision"], 3 / 4) and np.isclose(report["recall"], 3 / 4)
    assert np.isclose(report["f1"], 0.75)


def test_metrics_do_not_divide_by_zero():
    report = classification_report(np.array([0, 0]), np.array([0, 0]))
    assert report["precision"] == 0.0 and report["recall"] == 0.0 and report["f1"] == 0.0


# ---------------------------------------------------------------- coupling
@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    y = (X[:, 0] > 0).astype(float).reshape(-1, 1)
    builder = MLPBuilder(rng=rng).set_dataset(X, y).set_configuration({
        "hidden_layers": [3, 2], "activations": ["tanh", "tanh"], "task": "binary_classification",
    })
    network, loss = builder.build()
    return network, loss, X, y


def test_fitness_is_training_accuracy(toy):
    network, loss, X, y = toy
    adapter = MLPFitnessAdapter(network, loss, X, y, fitness="accuracy")
    position = np.random.default_rng(1).normal(size=adapter.dimension)
    fitness = adapter.evaluate(position)
    # The particle must now be inside the network.
    assert np.allclose(network.get_parameters(), position)
    manual = np.mean((network.forward(X).ravel() >= 0.5) == y.ravel())
    assert np.isclose(fitness, manual) and adapter.evaluations == 1


def test_loss_fitness_is_the_negative_loss(toy):
    network, loss, X, y = toy
    adapter = MLPFitnessAdapter(network, loss, X, y, fitness="loss")
    position = np.zeros(adapter.dimension)
    assert np.isclose(adapter.evaluate(position), -np.log(2))  # every output is 0.5


def test_activation_genes_are_decoded_into_equal_bins(toy):
    network, loss, X, y = toy
    adapter = ActivationSearchAdapter(network, loss, X, y, choices=["logistic", "tanh", "relu"],
                                      gene_range=(-1.0, 1.0))
    assert adapter.dimension == network.count_parameters() + 2  # two hidden layers
    position = np.zeros(adapter.dimension)
    for genes, expected in [((-0.9, 0.0), ["logistic", "tanh"]), ((0.9, -5.0), ["relu", "logistic"]),
                            ((7.0, 0.34), ["relu", "relu"])]:
        position[-2:] = genes
        assert adapter.decode_activations(position) == expected
        adapter.evaluate(position)
        assert [layer.activation.name for layer in network.layers[:-1]] == expected


def test_pso_trained_network_solves_the_banknote_problem():
    result = run_once(make_config(alpha=0.4, beta=1.0, gamma=1.0), seed=0)
    assert result["evaluations"] == 25 * 40
    assert result["train_accuracy"] > 0.98 and result["test_accuracy"] > 0.97
    assert np.isclose(result["best_fitness"], result["train_accuracy"])


def test_activation_search_run_reports_what_it_chose():
    result = run_once(make_config(search_activations=True, iterations=10), seed=0)
    assert result["dimension"] == 50
    assert result["chosen_activations"] in {"logistic", "tanh", "relu", "leaky_relu", "elu"}
