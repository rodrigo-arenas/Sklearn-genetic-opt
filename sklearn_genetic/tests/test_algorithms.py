import random
from types import SimpleNamespace

import numpy as np
import pytest
from deap import base, creator, tools

from ..algorithms import eaMuCommaLambda, eaMuPlusLambda, eaSimple
from ..callbacks.base import BaseCallback
from ..evaluation import create_fit_stats
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
