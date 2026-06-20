import random
from types import SimpleNamespace

import numpy as np
import pytest
from deap import base, creator, tools

from ..algorithms import eaMuCommaLambda, eaMuPlusLambda, eaSimple
from ..callbacks.base import BaseCallback
from ..evaluation import create_fit_stats
from ..optimizer_control import shared_fitness
from ..schedules.schedulers import ConstantAdapter


class StopOnFirstStep(BaseCallback):
    def __init__(self):
        self.started = False
        self.ended = False

    def on_start(self, estimator=None):
        self.started = True

    def on_step(self, record=None, logbook=None, estimator=None):
        return True

    def on_end(self, logbook=None, estimator=None):
        self.ended = True


@pytest.fixture
def simple_toolbox():
    if not hasattr(creator, "FitnessAlgorithmTest"):
        creator.create("FitnessAlgorithmTest", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "IndividualAlgorithmTest"):
        creator.create("IndividualAlgorithmTest", list, fitness=creator.FitnessAlgorithmTest)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.IndividualAlgorithmTest,
        toolbox.attr_bool,
        4,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda individual: (sum(individual),))
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.5)
    toolbox.register("select", tools.selBest)
    toolbox.register("map", map)
    return toolbox


@pytest.mark.parametrize(
    "algorithm, kwargs",
    [
        (eaSimple, {}),
        (eaMuPlusLambda, {"mu": 4, "lambda_": 4}),
        (eaMuCommaLambda, {"mu": 4, "lambda_": 4}),
    ],
)
def test_algorithm_stops_when_callback_requests_stop(simple_toolbox, algorithm, kwargs):
    random_state = random.getstate()
    numpy_state = np.random.get_state()

    try:
        random.seed(0)
        np.random.seed(0)
        population = simple_toolbox.population(n=4)
        callback = StopOnFirstStep()
        stats = tools.Statistics(lambda individual: individual.fitness.values)
        stats.register("fitness", max)

        _, logbook, n_gen = algorithm(
            population=population,
            toolbox=simple_toolbox,
            cxpb=ConstantAdapter(0.5, 0.5, 0),
            mutpb=ConstantAdapter(0.5, 0.5, 0),
            ngen=3,
            stats=stats,
            halloffame=tools.HallOfFame(1),
            callbacks=[callback],
            verbose=False,
            estimator=SimpleNamespace(elitism=False),
            **kwargs,
        )
    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)

    assert callback.started
    assert callback.ended
    assert n_gen == 0
    assert len(logbook) == 1


def test_verbose_output_uses_compact_generation_summary(simple_toolbox, capsys):
    population = simple_toolbox.population(n=4)
    stats = tools.Statistics(lambda individual: individual.fitness.values)
    stats.register("fitness", np.mean)
    stats.register("fitness_max", np.max)

    eaSimple(
        population=population,
        toolbox=simple_toolbox,
        cxpb=ConstantAdapter(0.5, 0.5, 0),
        mutpb=ConstantAdapter(0.5, 0.5, 0),
        ngen=1,
        stats=stats,
        halloffame=tools.HallOfFame(1),
        callbacks=[],
        verbose=True,
        estimator=SimpleNamespace(elitism=False),
    )

    output = capsys.readouterr().out

    assert " gen evals" in output
    assert "best" in output
    assert "unique" in output
    assert "events" in output
    assert "unique_individual_ratio" not in output
    assert "fitness_sharing_applied" not in output

    table_lines = [line for line in output.splitlines() if line.strip()]
    assert len({len(line) for line in table_lines}) == 1


def test_algorithm_records_monotonic_best_fitness(simple_toolbox):
    population = [
        creator.IndividualAlgorithmTest([1, 0, 0, 0]),
        creator.IndividualAlgorithmTest([1, 1, 0, 0]),
        creator.IndividualAlgorithmTest([0, 0, 0, 0]),
        creator.IndividualAlgorithmTest([0, 1, 0, 0]),
    ]
    stats = tools.Statistics(lambda individual: individual.fitness.values)
    stats.register("fitness", np.mean)
    stats.register("fitness_max", np.max)
    stats.register("fitness_min", np.min)

    _, logbook, _ = eaSimple(
        population=population,
        toolbox=simple_toolbox,
        cxpb=ConstantAdapter(1.0, 1.0, 0),
        mutpb=ConstantAdapter(1.0, 1.0, 0),
        ngen=4,
        stats=stats,
        halloffame=tools.HallOfFame(1),
        callbacks=[],
        verbose=False,
        estimator=SimpleNamespace(elitism=False),
    )

    best_history = logbook.select("fitness_best")

    assert all(current >= previous for previous, current in zip(best_history, best_history[1:]))


def test_diversity_control_boosts_mutation_and_adds_immigrants(simple_toolbox):
    population = [creator.IndividualAlgorithmTest([0, 0, 0, 0]) for _ in range(4)]
    estimator = SimpleNamespace(
        elitism=False,
        diversity_control=True,
        diversity_threshold=0.5,
        diversity_stagnation_generations=5,
        diversity_mutation_boost=3.0,
        random_immigrants_fraction=0.5,
        local_search=False,
        fit_stats_=create_fit_stats(),
    )

    _, logbook, _ = eaSimple(
        population=population,
        toolbox=simple_toolbox,
        cxpb=ConstantAdapter(0.1, 0.1, 0),
        mutpb=ConstantAdapter(0.2, 0.2, 0),
        ngen=1,
        stats=None,
        halloffame=tools.HallOfFame(1),
        callbacks=[],
        verbose=False,
        estimator=estimator,
    )

    assert logbook[1]["diversity_control_triggered"] is True
    assert logbook[1]["mutation_probability"] == pytest.approx(0.6)
    assert logbook[1]["random_immigrants"] == 2
    assert estimator.fit_stats_["random_immigrants"] == 2


def test_local_search_refines_hall_of_fame_candidates(simple_toolbox):
    population = simple_toolbox.population(n=4)
    estimator = SimpleNamespace(
        elitism=False,
        local_search=True,
        local_search_top_k=1,
        local_search_steps=2,
        local_search_radius=0.25,
        max_features=None,
        fit_stats_=create_fit_stats(),
    )

    _, logbook, _ = eaSimple(
        population=population,
        toolbox=simple_toolbox,
        cxpb=ConstantAdapter(0.0, 0.0, 0),
        mutpb=ConstantAdapter(0.0, 0.0, 0),
        ngen=1,
        stats=None,
        halloffame=tools.HallOfFame(1),
        callbacks=[],
        verbose=False,
        estimator=estimator,
    )

    assert logbook[-1]["local_refinements"] == 2
    assert estimator.fit_stats_["local_refinement_candidates"] == 2


def test_shared_fitness_temporarily_penalizes_crowded_niches(simple_toolbox):
    crowded_a = creator.IndividualAlgorithmTest([1, 1, 1, 1])
    crowded_b = creator.IndividualAlgorithmTest([1, 1, 1, 1])
    isolated = creator.IndividualAlgorithmTest([0, 0, 0, 0])

    for individual in [crowded_a, crowded_b, isolated]:
        individual.fitness.values = (10,)

    population = [crowded_a, crowded_b, isolated]
    estimator = SimpleNamespace(
        fitness_sharing=True,
        sharing_radius=0.25,
        sharing_alpha=1.0,
    )

    with shared_fitness(population, estimator) as sharing_record:
        selected = tools.selBest(population, 1)
        assert selected == [isolated]
        assert crowded_a.fitness.values[0] < isolated.fitness.values[0]
        assert sharing_record["fitness_sharing_applied"] is True
        assert sharing_record["max_niche_count"] == pytest.approx(2.0)

    assert crowded_a.fitness.values == (10,)
    assert crowded_b.fitness.values == (10,)
    assert isolated.fitness.values == (10,)


def test_algorithm_records_fitness_sharing_telemetry(simple_toolbox):
    population = [creator.IndividualAlgorithmTest([1, 1, 1, 1]) for _ in range(4)]
    estimator = SimpleNamespace(
        elitism=False,
        local_search=False,
        fitness_sharing=True,
        sharing_radius=0.25,
        sharing_alpha=1.0,
    )

    _, logbook, _ = eaSimple(
        population=population,
        toolbox=simple_toolbox,
        cxpb=ConstantAdapter(0.0, 0.0, 0),
        mutpb=ConstantAdapter(0.0, 0.0, 0),
        ngen=1,
        stats=None,
        halloffame=tools.HallOfFame(1),
        callbacks=[],
        verbose=False,
        estimator=estimator,
    )

    assert logbook[1]["fitness_sharing_applied"] is True
    assert logbook[1]["mean_niche_count"] >= 1.0
    assert logbook[1]["max_niche_count"] >= 1.0
