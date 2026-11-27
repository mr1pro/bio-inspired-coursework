"""Tests for the ANN: activations, layers, network and builder."""

import numpy as np
import pytest

from src.mlp.activations import ACTIVATIONS, make_activation
from src.mlp.builder import MLPBuilder
from src.mlp.layer import Layer
from src.mlp.losses import make_loss
from src.mlp.network import Network


def build(hidden, activations, n_features=4, n_samples=10):
    X = np.zeros((n_samples, n_features))
    y = np.zeros((n_samples, 1))
    builder = MLPBuilder(rng=np.random.default_rng(0))
    builder.set_dataset(X, y)
    builder.set_configuration({
        "hidden_layers": hidden, "activations": activations, "task": "binary_classification",
    })
    return builder.build()


def test_required_activations_have_the_right_values():
    z = np.array([[-2.0, 0.0, 3.0]])
    assert np.allclose(make_activation("logistic").forward(z), 1 / (1 + np.exp(-z)))
    assert np.allclose(make_activation("relu").forward(z), [[0.0, 0.0, 3.0]])
    assert np.allclose(make_activation("tanh").forward(z), np.tanh(z))


def test_logistic_does_not_overflow_for_huge_inputs():
    out = make_activation("logistic").forward(np.array([[-1e6, 1e6]]))
    assert np.all(np.isfinite(out)) and out[0, 0] < 1e-12 and out[0, 1] > 1 - 1e-12


@pytest.mark.parametrize("name", sorted(ACTIVATIONS))
def test_every_activation_keeps_the_shape(name):
    z = np.random.default_rng(0).normal(size=(5, 3))
    assert make_activation(name).forward(z).shape == z.shape


def test_softmax_rows_sum_to_one():
    out = make_activation("softmax").forward(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(out.sum(axis=1), 1.0)


def test_layer_forward_matches_manual_calculation():
    layer = Layer(2, 1, make_activation("linear"))
    layer.set_parameters(np.array([2.0, -1.0, 0.5]))  # weights 2, -1 and bias 0.5
    assert np.allclose(layer.forward(np.array([[1.0, 3.0]])), [[2.0 - 3.0 + 0.5]])


def test_parameter_count_follows_the_formula():
    network, _ = build([8, 5], ["tanh", "relu"])
    # 4->8, 8->5, 5->1 : (4*8+8) + (8*5+5) + (5*1+1)
    assert network.count_parameters() == 40 + 45 + 6


def test_parameters_survive_a_round_trip():
    network, _ = build([6, 3], ["tanh", "logistic"])
    vector = np.random.default_rng(1).normal(size=network.count_parameters())
    network.set_parameters(vector)
    assert np.allclose(network.get_parameters(), vector)


def test_setting_parameters_changes_the_output():
    network, _ = build([4], ["tanh"])
    X = np.random.default_rng(2).normal(size=(7, 4))
    network.set_parameters(np.zeros(network.count_parameters()))
    assert np.allclose(network.forward(X), 0.5)  # logistic(0) for every sample
    network.set_parameters(np.ones(network.count_parameters()))
    assert not np.allclose(network.forward(X), 0.5)


def test_wrong_vector_length_is_rejected():
    network, _ = build([4], ["tanh"])
    with pytest.raises(ValueError):
        network.set_parameters(np.zeros(network.count_parameters() + 1))


def test_network_without_hidden_layer_is_allowed():
    network, _ = build([], [])
    assert network.count_parameters() == 5 and len(network.layers) == 1


def test_builder_uses_one_logistic_output_and_bce_for_binary_tasks():
    network, loss = build([3], ["relu"])
    assert network.layers[-1].output_size == 1
    assert network.layers[-1].activation.name == "logistic"
    assert loss.name == "bce"


def test_builder_handles_multiclass_tasks():
    X = np.zeros((6, 4))
    y = np.eye(3)[[0, 1, 2, 0, 1, 2]]
    network, loss = MLPBuilder().set_dataset(X, y).set_configuration({
        "hidden_layers": [5], "activations": ["tanh"], "task": "multiclass_classification",
    }).build()
    assert network.layers[-1].output_size == 3 and loss.name == "cce"
    assert np.allclose(network.forward(X).sum(axis=1), 1.0)


@pytest.mark.parametrize("config", [
    {"hidden_layers": [4, 4], "activations": ["tanh"], "task": "binary_classification"},
    {"hidden_layers": [0], "activations": ["tanh"], "task": "binary_classification"},
    {"hidden_layers": [4], "activations": ["swish"], "task": "binary_classification"},
    {"hidden_layers": [4], "activations": ["tanh"], "task": "regression"},
])
def test_builder_rejects_bad_configurations(config):
    with pytest.raises(ValueError):
        MLPBuilder().set_dataset(np.zeros((2, 4)), np.zeros((2, 1))).set_configuration(config)


def test_add_layer_checks_sizes():
    network = Network()
    network.add_layer(Layer(4, 3, make_activation("tanh")))
    with pytest.raises(ValueError):
        network.add_layer(Layer(5, 1, make_activation("logistic")))


def test_losses_on_known_values():
    y = np.array([[1.0], [0.0]])
    p = np.array([[0.8], [0.4]])
    assert np.isclose(make_loss("mse").compute(y, p), (0.04 + 0.16) / 2)
    assert np.isclose(make_loss("bce").compute(y, p), -(np.log(0.8) + np.log(0.6)) / 2)
    assert np.isfinite(make_loss("bce").compute(y, np.array([[0.0], [1.0]])))  # no log(0)
