from types import SimpleNamespace

import pytest
from deap import tools
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeClassifier

from .._base import GeneticEstimatorMixin, history_record, reset_adapters
from ..evaluation import create_fit_stats, evaluate_population, record_fit_stats
from ..population import (
    initialize_feature_population,
    random_search_population,
    smart_search_population,
    validate_population_initializer,
)
from ..space import Categorical, Integer, Space


def test_smart_search_population_seeds_warm_start_defaults_and_unique_candidates():
    space = Space(
        {
            "max_depth": Integer(1, 4),
            "criterion": Categorical(["gini", "entropy"]),
        }
    )
    estimator = SimpleNamespace(
        estimator=DecisionTreeClassifier(max_depth=2, criterion="entropy"),
        population_size=5,
        space=space,
        warm_start_configs=[{"max_depth": 4, "criterion": "gini"}],
    )
    toolbox = SimpleNamespace(individual=lambda: [1, "gini"])

    population = smart_search_population(estimator, toolbox, list)

    assert len(population) == estimator.population_size
    assert [4, "gini"] in population
    assert [2, "entropy"] in population
    assert len({tuple(individual) for individual in population}) >= 3


def test_random_search_population_preserves_warm_starts_before_random_candidates():
    space = Space({"max_depth": Integer(1, 4)})
    estimator = SimpleNamespace(
        population_size=3,
        space=space,
        warm_start_configs=[{"max_depth": 4}],
    )
    toolbox = SimpleNamespace(population=lambda n: [[1] for _ in range(n)])

    population = random_search_population(estimator, toolbox, list)

    assert population == [[4], [1], [1]]


def test_feature_population_initializer_delegates_random_and_smart_modes():
    smart_estimator = SimpleNamespace(
        population_initializer="smart",
        population_size=4,
        n_features=5,
        max_features=3,
    )
    random_estimator = SimpleNamespace(
        population_initializer="random",
        population_size=4,
    )
    toolbox = SimpleNamespace(
        individual=lambda: [1, 0, 0, 0, 0],
        population=lambda n: [["random"] for _ in range(n)],
    )

    smart_population = initialize_feature_population(smart_estimator, toolbox, list)
    random_population = initialize_feature_population(random_estimator, toolbox, list)

    assert len(smart_population) == smart_estimator.population_size
    assert all(
        1 <= sum(individual) <= smart_estimator.max_features for individual in smart_population
    )
    assert random_population == [["random"]] * random_estimator.population_size


def test_validate_population_initializer_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="population_initializer must be one of"):
        validate_population_initializer("lhs")


def test_evaluate_population_deduplicates_updates_cache_and_fit_stats():
    calls = []

    def evaluate_individual(individual, n_jobs=None):
        calls.append((individual, n_jobs))
        value = individual[0]
        return [value], {"value": value}, True, False

    estimator = SimpleNamespace(
        use_cache=True,
        fitness_cache={},
        logbook=tools.Logbook(),
        fit_stats_=create_fit_stats(),
        n_jobs=1,
        parallel_backend="auto",
        log_config=None,
        _individual_key=lambda individual: tuple(individual),
        _evaluate_individual=evaluate_individual,
    )

    fitnesses = evaluate_population(
        estimator,
        individuals=[[1], [1], [2]],
        cache_record_key="current_generation_params",
    )

    assert fitnesses == [[1], [1], [2]]
    assert calls == [([1], 1), ([2], 1)]
    assert len(estimator.logbook.chapters["parameters"]) == 3
    assert estimator.fit_stats_["evaluated_candidates"] == 3
    assert estimator.fit_stats_["unique_candidates"] == 2
    assert estimator.fit_stats_["duplicate_candidates"] == 1
    assert estimator.fit_stats_["cross_validate_calls"] == 2


def test_record_fit_stats_accumulates_counts():
    estimator = SimpleNamespace(fit_stats_=create_fit_stats())

    record_fit_stats(
        estimator,
        evaluated=2,
        unique=1,
        cv_calls=1,
        cache_hits=1,
        duplicates=1,
        skipped=0,
    )
    record_fit_stats(estimator, evaluated=1, skipped=1)

    assert estimator.fit_stats_["evaluated_candidates"] == 3
    assert estimator.fit_stats_["unique_candidates"] == 1
    assert estimator.fit_stats_["cache_hits"] == 1
    assert estimator.fit_stats_["skipped_invalid_candidates"] == 1


def test_genetic_estimator_mixin_history_iteration_and_fitted_guard():
    class DummySearch(GeneticEstimatorMixin):
        pass

    search = DummySearch()
    search.estimator = DecisionTreeClassifier().fit([[0], [1]], [0, 1])
    search.refit = True
    search._n_iterations = 1
    search.history = {"gen": [0, 1], "fitness": [0.5, 0.8]}

    assert history_record(search.history, 1) == {"gen": 1, "fitness": 0.8}
    assert search[0] == {"gen": 0, "fitness": 0.5}
    assert list(search) == [
        {"gen": 0, "fitness": 0.5},
        {"gen": 1, "fitness": 0.8},
    ]
    assert len(search) == 1

    search.refit = False
    with pytest.raises(NotFittedError, match="DummySearch instance is not fitted yet"):
        search[0]


def test_reset_adapters_calls_both_adapter_resets():
    calls = []
    estimator = SimpleNamespace(
        crossover_adapter=SimpleNamespace(reset=lambda: calls.append("crossover")),
        mutation_adapter=SimpleNamespace(reset=lambda: calls.append("mutation")),
    )

    reset_adapters(estimator)

    assert calls == ["crossover", "mutation"]
