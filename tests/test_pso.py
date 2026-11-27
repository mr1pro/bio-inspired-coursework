"""Tests for PSO on problems where the right answer is known."""

import numpy as np
import pytest

from src.pso.boundary import BOUNDARY_HANDLERS, make_boundary_handler
from src.pso.fitness_function import FitnessFunction
from src.pso.informant_strategy import INFORMANT_STRATEGIES, make_informant_strategy
from src.pso.particle import Particle
from src.pso.population import Population


class Sphere(FitnessFunction):
    """-|x|^2: the best possible fitness is 0, at the origin."""

    def __init__(self):
        self.calls = 0

    def evaluate(self, position):
        self.calls += 1
        return -float(np.sum(position ** 2))


def make_population(**overrides):
    settings = dict(dimension=5, num_particles=20, num_iterations=60, inertia_weight=0.4,
                    cognitive_coefficient=1.0, social_coefficient=1.0, seed=0)
    settings.update(overrides)
    function = Sphere()
    return Population(function, **settings), function


def test_pso_finds_the_optimum_of_the_sphere():
    population, _ = make_population()
    position, fitness = population.optimise()
    assert fitness > -1e-3
    assert np.linalg.norm(position) < 0.05


def test_one_evaluation_per_particle_per_iteration():
    population, function = make_population(num_particles=7, num_iterations=13)
    population.optimise()
    assert function.calls == 7 * 13 == population.evaluations


def test_best_fitness_never_gets_worse():
    population, _ = make_population()
    population.optimise()
    history = np.array(population.fitness_history)
    assert len(history) == 60 and np.all(np.diff(history) >= 0)


def test_same_seed_gives_the_same_run():
    first, _ = make_population(seed=3)
    second, _ = make_population(seed=3)
    other, _ = make_population(seed=4)
    assert np.array_equal(first.optimise()[0], second.optimise()[0])
    assert not np.array_equal(first.global_best_position, other.optimise()[0])


def test_returned_fitness_belongs_to_the_returned_position():
    population, function = make_population()
    position, fitness = population.optimise()
    assert np.isclose(function.evaluate(position), fitness)


def test_target_fitness_stops_the_run_early():
    population, function = make_population(num_iterations=500, target_fitness=-0.5)
    population.optimise()
    assert population.global_best_fitness >= -0.5
    assert function.calls < 20 * 500


def test_inertia_schedule_is_linear():
    population, _ = make_population(num_iterations=11, inertia_weight=0.9, final_inertia_weight=0.4)
    assert np.isclose(population.current_inertia(0), 0.9)
    assert np.isclose(population.current_inertia(5), 0.65)
    assert np.isclose(population.current_inertia(10), 0.4)


def test_velocity_limit_is_respected():
    population, _ = make_population(max_velocity=0.05, inertia_weight=0.9,
                                    cognitive_coefficient=2.0, social_coefficient=2.0)
    population.optimise()
    assert all(np.all(np.abs(p.velocity) <= 0.05 + 1e-12) for p in population.particles)


@pytest.mark.parametrize("bad", [dict(num_informants=0), dict(num_informants=20),
                                 dict(num_particles=1), dict(num_iterations=0),
                                 dict(velocity_init="fast")])
def test_invalid_settings_are_rejected(bad):
    with pytest.raises(ValueError):
        make_population(**bad)


# ---------------------------------------------------------------- particle
def test_velocity_update_matches_algorithm_39_line_24():
    """With beta = gamma = delta = 0 only the inertia term is left."""
    particle = Particle(np.array([1.0, 2.0]), np.array([0.5, -0.5]))
    particle.update_velocity(np.zeros(2), np.zeros(2), alpha=0.5, beta=0.0, gamma=0.0, delta=0.0,
                             rng=np.random.default_rng(0))
    assert np.allclose(particle.velocity, [0.25, -0.25])
    particle.update_position(epsilon=2.0)
    assert np.allclose(particle.position, [1.5, 1.5])


def test_velocity_pull_stays_inside_the_expected_range():
    """b is drawn from [0, beta], so the pull towards x* is between 0 and
    beta * (x* - x) in every dimension."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        particle = Particle(np.zeros(3), np.zeros(3))
        particle.best_position = np.ones(3)
        particle.update_velocity(np.zeros(3), np.zeros(3), alpha=0.0, beta=1.5, gamma=0.0, delta=0.0, rng=rng)
        assert np.all(particle.velocity >= 0.0) and np.all(particle.velocity <= 1.5)


def test_informants_best_includes_the_particle_itself():
    me, worse, better = (Particle(np.full(2, v), np.zeros(2)) for v in (0.0, 1.0, 2.0))
    me.best_fitness, worse.best_fitness, better.best_fitness = 5.0, 1.0, 9.0
    me.informants = [worse]
    assert me.get_informants_best()[1] == 5.0          # own best beats the informant
    me.informants = [worse, better]
    position, fitness = me.get_informants_best()
    assert fitness == 9.0 and np.array_equal(position, better.best_position)


# -------------------------------------------------------------- informants
@pytest.mark.parametrize("name", sorted(INFORMANT_STRATEGIES))
@pytest.mark.parametrize("k", [1, 3, 9])
def test_strategies_return_k_distinct_other_particles(name, k):
    rng = np.random.default_rng(0)
    swarm = [Particle(rng.normal(size=4), np.zeros(4)) for _ in range(10)]
    strategy = make_informant_strategy(name)
    strategy.bind(rng)
    strategy.begin_iteration(swarm)
    for particle in swarm:
        chosen = strategy.select_informants(particle, swarm, k)
        assert len(chosen) == k
        assert all(c is not particle for c in chosen)
        assert len({id(c) for c in chosen}) == k


def test_nearest_neighbour_picks_the_closest_particles():
    swarm = [Particle(np.array([float(x)]), np.zeros(1)) for x in (0, 1, 2, 10, 11)]
    strategy = make_informant_strategy("nearest")
    strategy.begin_iteration(swarm)
    chosen = strategy.select_informants(swarm[0], swarm, 2)
    assert {c.position[0] for c in chosen} == {1.0, 2.0}


def test_ring_picks_index_neighbours():
    swarm = [Particle(np.array([float(i)]), np.zeros(1)) for i in range(6)]
    chosen = make_informant_strategy("ring").select_informants(swarm[0], swarm, 2)
    assert {c.position[0] for c in chosen} == {1.0, 5.0}


# ---------------------------------------------------------------- boundary
@pytest.mark.parametrize("name", [n for n in sorted(BOUNDARY_HANDLERS) if n != "none"])
def test_handlers_bring_particles_back_inside(name):
    handler = make_boundary_handler(name, -1.0, 1.0)
    handler.bind(np.random.default_rng(0))
    position = np.array([-7.3, -1.2, 0.3, 1.4, 25.0])
    new_position, _ = handler.apply(position, np.ones(5))
    assert np.all(new_position >= -1.0) and np.all(new_position <= 1.0)
    assert new_position[2] == 0.3  # a coordinate that was inside is left alone


def test_no_boundary_changes_nothing():
    position, velocity = np.array([50.0]), np.array([3.0])
    new_position, new_velocity = make_boundary_handler("none").apply(position, velocity)
    assert new_position[0] == 50.0 and new_velocity[0] == 3.0


def test_clamp_reflect_and_wrap_do_what_their_names_say():
    position, velocity = np.array([1.3]), np.array([0.5])
    clamped = make_boundary_handler("clamp", -1, 1).apply(position, velocity)
    reflected = make_boundary_handler("reflect", -1, 1).apply(position, velocity)
    wrapped = make_boundary_handler("wrap", -1, 1).apply(position, velocity)
    assert np.isclose(clamped[0][0], 1.0) and clamped[1][0] == 0.0
    assert np.isclose(reflected[0][0], 0.7) and reflected[1][0] == -0.5
    assert np.isclose(wrapped[0][0], -0.7) and wrapped[1][0] == 0.5


def test_bounded_run_never_leaves_the_search_space():
    population, _ = make_population(boundary_handler=make_boundary_handler("reflect", -1, 1),
                                    inertia_weight=0.9, cognitive_coefficient=2.0, social_coefficient=2.0)
    population.optimise()
    assert all(np.all(np.abs(p.position) <= 1.0) for p in population.particles)
