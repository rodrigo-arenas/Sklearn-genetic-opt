import random
from types import SimpleNamespace

import numpy as np
import pytest
from deap import base, creator, tools
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeClassifier

from .._base import GeneticEstimatorMixin, history_record, reset_adapters
from ..evaluation import create_fit_stats, evaluate_population, record_fit_stats
from ..genetic_search import GAFeatureSelectionCV, GASearchCV
from ..optimizer_control import (
    _bounded_continuous_neighbor,
    _shared_primary_fitness,
    individual_distance,
    inject_random_immigrants,
    local_neighbor,
    mutation_probability,
    niche_counts,
    random_immigrant_count,
    replace_duplicate_candidates,
    sharing_value,
    validate_optimizer_control,
)
from ..population import (
    _default_estimator_params,
    _is_dimension_value_valid,
    _sample_dimension_from_unit,
    _stratified_categorical_value,
    initialize_feature_population,
    initialize_search_population,
    random_search_population,
    smart_feature_population,
    smart_search_population,
    validate_population_initializer,
)
from ..schedules.schedulers import ConstantAdapter
from ..space import Categorical, Continuous, Integer, Space
from ..utils.cv_scores import _rank_scores
from ..utils.random import weighted_bool_individual


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


def test_smart_search_population_is_reproducible_with_seeded_random_state():
    space = Space(
        {
            "max_depth": Integer(1, 10),
            "min_samples_leaf": Integer(1, 5),
            "criterion": Categorical(["gini", "entropy"]),
        }
    )
    estimator = SimpleNamespace(
        estimator=DecisionTreeClassifier(max_depth=2, min_samples_leaf=1, criterion="gini"),
        population_size=6,
        space=space,
        warm_start_configs=[],
    )
    toolbox = SimpleNamespace(individual=lambda: [1, 1, "gini"])

    random.seed(42)
    first_population = smart_search_population(estimator, toolbox, list)
    random.seed(42)
    second_population = smart_search_population(estimator, toolbox, list)

    assert first_population == second_population


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


def test_random_search_population_rejects_invalid_warm_start_config():
    """An invalid warm-start config surfaces a clear error at init time (#220)."""
    space = Space({"max_depth": Integer(1, 4)})
    estimator = SimpleNamespace(
        population_size=3,
        space=space,
        warm_start_configs=[{"max_depths": 4}],  # typo
    )
    toolbox = SimpleNamespace(population=lambda n: [[1] for _ in range(n)])

    with pytest.raises(ValueError, match="not in the search space"):
        random_search_population(estimator, toolbox, list)


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


@pytest.mark.parametrize(
    ("parameter", "value", "message"),
    [
        ("local_search_top_k", 0, "local_search_top_k"),
        ("local_search_steps", 0, "local_search_steps"),
        ("local_search_radius", 0, "local_search_radius"),
        ("diversity_threshold", -0.1, "diversity_threshold"),
        ("diversity_stagnation_generations", 0, "diversity_stagnation_generations"),
        ("diversity_mutation_boost", 0.9, "diversity_mutation_boost"),
        ("random_immigrants_fraction", 1.1, "random_immigrants_fraction"),
        ("sharing_radius", 0, "sharing_radius"),
        ("sharing_alpha", 0, "sharing_alpha"),
        ("selection_pressure_min", 0, "selection_pressure_min"),
        ("selection_pressure_max", 1, "selection_pressure_max"),
        ("offspring_diversity_retries", -1, "offspring_diversity_retries"),
    ],
)
def test_validate_optimizer_control_rejects_invalid_values(parameter, value, message):
    kwargs = {
        "local_search_top_k": 1,
        "local_search_steps": 1,
        "local_search_radius": 0.1,
        "diversity_threshold": 0.1,
        "diversity_stagnation_generations": 1,
        "diversity_mutation_boost": 1.0,
        "random_immigrants_fraction": 0.1,
        "sharing_radius": 0.1,
        "sharing_alpha": 1.0,
        "selection_pressure_min": 2,
        "selection_pressure_max": 4,
        "offspring_diversity_retries": 0,
    }
    kwargs[parameter] = value

    with pytest.raises(ValueError, match=message):
        validate_optimizer_control(**kwargs)


def test_diversity_control_mutation_and_immigrant_edge_cases():
    record = {
        "unique_individual_ratio": 0.0,
        "genotype_diversity": 1.0,
        "stagnation_generations": 0,
    }
    estimator = SimpleNamespace(
        diversity_control=True,
        diversity_threshold=0.5,
        diversity_stagnation_generations=3,
        diversity_mutation_boost=10.0,
        random_immigrants_fraction=0.0,
    )

    probability, triggered = mutation_probability(ConstantAdapter(0.2, 0.2, 0), estimator, record)

    assert probability == 1.0
    assert triggered is True
    assert random_immigrant_count(estimator, 10) == 0
    assert random_immigrant_count(estimator, 0) == 0
    assert (
        inject_random_immigrants(
            [[1], [2]], SimpleNamespace(individual=lambda: [9]), estimator, record
        )
        == 0
    )


def test_random_immigrants_and_duplicate_replacement_update_population():
    record = {
        "unique_individual_ratio": 0.0,
        "genotype_diversity": 0.0,
        "stagnation_generations": 0,
    }
    estimator = SimpleNamespace(
        diversity_control=True,
        diversity_threshold=0.5,
        diversity_stagnation_generations=3,
        random_immigrants_fraction=0.5,
    )
    toolbox = SimpleNamespace(individual=lambda: [9])
    offspring = [[1], [2], [3], [4]]

    assert inject_random_immigrants(offspring, toolbox, estimator, record) == 2
    assert offspring[-2:] == [[9], [9]]

    population = [[1], [1], [2]]
    assert replace_duplicate_candidates(population, toolbox, estimator) == 1
    assert population == [[1], [9], [2]]
    estimator.diversity_control = False
    assert replace_duplicate_candidates(population, toolbox, estimator) == 0


@pytest.mark.parametrize("forced_index", [0, 1, 2])
def test_local_neighbor_for_search_space_dimensions(monkeypatch, forced_index):
    space = Space(
        {
            "depth": Integer(1, 5),
            "rate": Continuous(0.1, 1.0),
            "criterion": Categorical(["gini", "entropy"]),
        }
    )
    estimator = SimpleNamespace(space=space, local_search_radius=0.25)
    parent = [3, 0.5, "gini"]
    monkeypatch.setattr(
        "sklearn_genetic.optimizer_control.random.randrange", lambda *args: forced_index
    )

    neighbor = local_neighbor(estimator, parent, list)

    assert len(neighbor) == len(parent)
    assert neighbor != parent
    assert 1 <= neighbor[0] <= 5
    assert 0.1 <= neighbor[1] <= 1.0
    assert neighbor[2] in {"gini", "entropy"}


def test_local_neighbor_respects_feature_limits_and_non_empty_masks(monkeypatch):
    estimator = SimpleNamespace(local_search_radius=0.1, max_features=1)
    monkeypatch.setattr(
        "sklearn_genetic.optimizer_control.random.sample", lambda values, k: list(values)[:k]
    )
    monkeypatch.setattr("sklearn_genetic.optimizer_control.random.shuffle", lambda values: None)

    neighbor = local_neighbor(estimator, [0, 1, 1], list)

    assert 1 <= sum(neighbor) <= estimator.max_features

    estimator.local_search_radius = 1.0
    estimator.max_features = None
    monkeypatch.setattr("sklearn_genetic.optimizer_control.random.randrange", lambda *args: 0)
    assert local_neighbor(estimator, [1], list) == [1]


def test_log_uniform_continuous_neighbor_stays_within_bounds():
    neighbor = _bounded_continuous_neighbor(
        Continuous(0.001, 1.0, distribution="log-uniform"),
        value=0.1,
        radius=0.25,
    )

    assert 0.001 <= neighbor <= 1.0


def test_fitness_sharing_distance_helpers_cover_search_space_and_empty_masks():
    estimator = SimpleNamespace(
        space=Space(
            {
                "depth": Integer(1, 5),
                "rate": Continuous(0.1, 1.0),
                "criterion": Categorical(["gini", "entropy"]),
            }
        ),
        sharing_radius=0.8,
        sharing_alpha=1.0,
    )

    assert individual_distance(estimator, [1, 0.1, "gini"], [5, 1.0, "entropy"]) == 1.0
    assert individual_distance(SimpleNamespace(), [], []) == 0.0
    assert sharing_value(distance=1.0, radius=0.5, alpha=1.0) == 0.0
    assert niche_counts(
        [[1, 0, 0], [1, 1, 0]], SimpleNamespace(sharing_radius=0.8, sharing_alpha=1.0)
    ) == [
        pytest.approx(1.5833333333333333),
        pytest.approx(1.5833333333333333),
    ]
    assert _shared_primary_fitness([], []) == []


def test_shared_primary_fitness_preserves_minimization_direction():
    if not hasattr(creator, "FitnessCoverageMin"):
        creator.create("FitnessCoverageMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "IndividualCoverageMin"):
        creator.create("IndividualCoverageMin", list, fitness=creator.FitnessCoverageMin)

    first = creator.IndividualCoverageMin([1])
    second = creator.IndividualCoverageMin([2])
    first.fitness.values = (0.2,)
    second.fitness.values = (0.4,)

    assert _shared_primary_fitness([first, second], [1.0, 2.0]) == [
        (pytest.approx(0.2),),
        (pytest.approx(0.4),),
    ]


def test_population_helper_edge_cases():
    assert _is_dimension_value_valid(object(), "x") is False
    assert (
        _default_estimator_params(DecisionTreeClassifier(), Space({"unknown": Integer(1, 2)}))
        is None
    )
    assert (
        _default_estimator_params(
            DecisionTreeClassifier(max_depth=2), Space({"max_depth": Integer(10, 20)})
        )
        is None
    )

    with pytest.raises(TypeError, match="Latin hypercube"):
        _sample_dimension_from_unit(Categorical(["a"]), 0.5)

    assert _stratified_categorical_value(Categorical(["a", "b"], priors=[0.25, 0.75]), 0, 2) == "b"


def test_initialize_search_population_random_and_smart_fallbacks():
    random_estimator = SimpleNamespace(
        population_initializer="random",
        population_size=2,
        warm_start_configs=[],
    )
    toolbox = SimpleNamespace(
        population=lambda n: [["random"] for _ in range(n)], individual=lambda: ["x"]
    )

    assert initialize_search_population(random_estimator, toolbox, list) == [["random"], ["random"]]

    smart_estimator = SimpleNamespace(
        estimator=DecisionTreeClassifier(criterion="gini"),
        population_size=2,
        space=Space({"criterion": Categorical(["gini"])}),
        warm_start_configs=[],
    )

    assert smart_search_population(smart_estimator, toolbox, list) == [["gini"], ["x"]]


def test_smart_population_warm_start_early_return_and_feature_fallback():
    search_estimator = SimpleNamespace(
        estimator=DecisionTreeClassifier(),
        population_size=1,
        space=Space({"max_depth": Integer(1, 5)}),
        warm_start_configs=[{"max_depth": 4}, {"max_depth": 5}],
    )
    toolbox = SimpleNamespace(individual=lambda: [1])

    assert smart_search_population(search_estimator, toolbox, list) == [[4]]

    feature_estimator = SimpleNamespace(population_size=2, n_features=1, max_features=1)
    assert smart_feature_population(feature_estimator, toolbox, list) == [[1], [1]]


def test_weighted_bool_individual_uses_weighted_and_unweighted_sampling(monkeypatch):
    calls = []

    def fake_choices(values, weights=None, k=None):
        calls.append((values, weights, k))
        return [0] * k

    monkeypatch.setattr("sklearn_genetic.utils.random.random.choices", fake_choices)
    monkeypatch.setattr("sklearn_genetic.utils.tools.random.choice", lambda values: 0)

    assert weighted_bool_individual(list, weight=0.75, size=3) == [1, 0, 0]
    assert weighted_bool_individual(list, weight=None, size=2) == [1, 0]
    assert calls == [
        ([0, 1], [0.25, 0.75], 3),
        ([0, 1], None, 2),
    ]


def test_ga_search_evaluate_records_cache_misses_and_hits():
    search = GASearchCV.__new__(GASearchCV)
    search.use_cache = True
    search.fitness_cache = {}
    search.logbook = tools.Logbook()
    search.fit_stats_ = create_fit_stats()
    search.n_jobs = 2
    search.parallel_backend = "cv"
    search._individual_key = lambda individual: tuple(individual)
    calls = []

    def evaluate_individual(individual, n_jobs=None):
        calls.append((list(individual), n_jobs))
        return [0.9], {"depth": individual[0]}, True, False

    search._evaluate_individual = evaluate_individual

    assert GASearchCV.evaluate(search, [3]) == [0.9]
    assert GASearchCV.evaluate(search, [3]) == [0.9]
    assert calls == [([3], 2)]
    assert search.fit_stats_["evaluated_candidates"] == 2
    assert search.fit_stats_["unique_candidates"] == 1
    assert search.fit_stats_["cache_hits"] == 1
    assert len(search.logbook.chapters["parameters"]) == 2


def test_ga_feature_selection_evaluate_records_cache_misses_and_hits():
    search = GAFeatureSelectionCV.__new__(GAFeatureSelectionCV)
    search.use_cache = True
    search.fitness_cache = {}
    search.logbook = tools.Logbook()
    search.fit_stats_ = create_fit_stats()
    search.n_jobs = 1
    search.parallel_backend = "auto"
    search._individual_key = lambda individual: tuple(individual)
    calls = []

    def evaluate_individual(individual, n_jobs=None):
        calls.append((list(individual), n_jobs))
        return [0.8, 2], {"features": list(individual)}, True, False

    search._evaluate_individual = evaluate_individual

    assert GAFeatureSelectionCV.evaluate(search, [1, 0, 1]) == [0.8, 2]
    assert GAFeatureSelectionCV.evaluate(search, [1, 0, 1]) == [0.8, 2]
    assert calls == [([1, 0, 1], 1)]
    assert search.fit_stats_["evaluated_candidates"] == 2
    assert search.fit_stats_["unique_candidates"] == 1
    assert search.fit_stats_["cache_hits"] == 1
    assert len(search.logbook.chapters["parameters"]) == 2


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


def test_rank_scores_orders_finite_scores_descending_for_maximization():
    ranks = _rank_scores([0.7, 0.9, 0.5])

    assert list(ranks) == [2, 1, 3]


def test_rank_scores_breaks_ties_with_deterministic_min_method():
    ranks = _rank_scores([0.9, 0.9, 0.5])

    assert list(ranks) == [1, 1, 3]


def test_rank_scores_supports_minimization_metrics():
    ranks = _rank_scores([0.7, 0.9, 0.5], greater_is_better=False)

    assert list(ranks) == [2, 3, 1]


def test_rank_scores_places_nan_candidates_last():
    ranks = _rank_scores([0.7, np.nan, 0.9])

    assert list(ranks) == [2, 3, 1]
    # NaN never outranks a finite candidate.
    assert ranks[1] == max(ranks)


def test_rank_scores_with_all_nan_does_not_crash():
    ranks = _rank_scores([np.nan, np.nan, np.nan])

    assert list(ranks) == [1, 1, 1]


def test_reset_adapters_calls_both_adapter_resets():
    calls = []
    estimator = SimpleNamespace(
        crossover_adapter=SimpleNamespace(reset=lambda: calls.append("crossover")),
        mutation_adapter=SimpleNamespace(reset=lambda: calls.append("mutation")),
    )

    reset_adapters(estimator)

    assert calls == ["crossover", "mutation"]
