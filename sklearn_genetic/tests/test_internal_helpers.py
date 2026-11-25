import pickle
import random
from statistics import mean, pstdev
from types import SimpleNamespace

import numpy as np
import pytest
from deap import base, creator, tools
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeClassifier

from .._base import history_record, reset_adapters
from .._telemetry import (
    _population_diversity,
    feature_population_diversity,
    population_fitness_stats,
)
from ..callbacks import CVBudgetStopping, ModelCheckpoint, TimerStopping
from ..evaluation import create_fit_stats, evaluate_population, evaluation_context, record_fit_stats
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
    initialize_feature_population,
    initialize_search_population,
    random_search_population,
    smart_feature_population,
    smart_search_population,
    validate_population_initializer,
)
from ..schedules.schedulers import ConstantAdapter
from ..space import Categorical, Continuous, Integer, Space
from ..utils.cv_scores import _rank_scores, _row_statistics
from ..utils.random import weighted_bool_individual


def test_fold_reuse_keeps_shared_positions_and_matches_reordered_folds_by_value():
    model = GASearchCV(
        DecisionTreeClassifier(),
        param_grid={"max_depth": Integer(2, 8), "min_samples_leaf": Integer(1, 4)},
    )
    model.X_ = np.zeros((12, 1))
    train, test = np.arange(8), np.arange(8, 12)
    model._cv_splits = [
        (train, test),
        (train.copy(), test.copy()),
        (train[::-1], test),
        (train.astype(float), test),
        (train, np.arange(12) == 10),
    ]

    assert model._matching_search_folds(model._cv_splits) == {0: 0, 1: 1, 2: 2, 4: 4}
    first, duplicate, reordered, *remaining = model._cv_splits
    assert model._matching_search_folds([reordered, duplicate, first, *remaining]) == {
        0: 2,
        1: 0,
        2: 1,
        4: 4,
    }


@pytest.mark.parametrize("backend", ["deap", "scipy"])
@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize("saved_count", [-1, 1.5, True, "mismatch"])
def test_invalid_checkpoint_cv_count_fails_before_search(
    backend, task, saved_count, tmp_path, monkeypatch
):
    from .. import genetic_search

    if backend == "scipy":
        pytest.importorskip("scipy", minversion="1.15")
    X = np.random.default_rng(4691).normal(size=(48, 8))
    y = X[:, 0] > 0
    settings = dict(
        estimator=DecisionTreeClassifier(random_state=4691),
        cv=2,
        generations=2,
        population_size=6,
        random_state=4691,
        optimizer_backend=backend,
        use_cache=False,
        verbose=False,
    )
    if task == "parameters":
        model = GASearchCV(
            param_grid={"max_depth": Integer(1, 8), "min_samples_leaf": Integer(1, 4)},
            **settings,
        )
    else:
        model = GAFeatureSelectionCV(max_features=3, **settings)

    checkpoint = ModelCheckpoint(tmp_path / "invalid-count.pkl")
    clone(model).fit(X, y, callbacks=[CVBudgetStopping(6), checkpoint])
    snapshot = checkpoint.load()
    stats = snapshot["runtime_state"]["fit_stats_"] if backend == "deap" else snapshot["fit_stats"]
    assert stats["cross_validate_calls"] == snapshot["logbook"][-1]["cross_validate_calls"] == 6
    stats["cross_validate_calls"] = 5 if saved_count == "mismatch" else saved_count
    contents = pickle.dumps(snapshot)
    checkpoint.checkpoint_path.write_bytes(contents)
    monkeypatch.setattr(
        genetic_search, "cross_validate", lambda *args, **kwargs: pytest.fail("unexpected CV")
    )

    detail = "does not match" if saved_count == "mismatch" else "nonnegative integer"
    with pytest.raises(ValueError, match=rf"invalid-count\.pkl.*cross_validate_calls.*{detail}"):
        model.fit(X, y, callbacks=[CVBudgetStopping(3), checkpoint])

    assert checkpoint.checkpoint_path.read_bytes() == contents


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize(
    "missing", ["runtime_state", "fit_stats_", "cross_validate_calls", "population_serial_batches"]
)
def test_deap_checkpoint_recovers_recorded_cv_count_with_missing_fit_stats(
    task, missing, tmp_path, monkeypatch
):
    from .. import genetic_search

    X = np.random.default_rng(5039).normal(size=(48, 6))
    y = X[:, 0] > 0
    settings = dict(
        estimator=DecisionTreeClassifier(random_state=5039),
        cv=2,
        generations=2,
        population_size=6,
        random_state=5039,
        use_cache=False,
        n_jobs=1,
        verbose=False,
    )
    if task == "parameters":
        model = GASearchCV(
            param_grid={"max_depth": Integer(1, 8), "min_samples_leaf": Integer(1, 4)},
            **settings,
        )
    else:
        model = GAFeatureSelectionCV(max_features=3, **settings)

    checkpoint = ModelCheckpoint(tmp_path / "incomplete-stats.pkl")
    first = clone(model).fit(X, y, callbacks=[CVBudgetStopping(6), checkpoint])
    assert first.fit_counts_["search_cv_calls"] == 6
    saved = checkpoint.load()
    if missing == "runtime_state":
        del saved["runtime_state"]
    elif missing == "fit_stats_":
        del saved["runtime_state"]["fit_stats_"]
    else:
        del saved["runtime_state"]["fit_stats_"][missing]
    checkpoint.checkpoint_path.write_bytes(pickle.dumps(saved))

    original = genetic_search.cross_validate
    fresh_fold_fits = []

    def count_cv(*args, **kwargs):
        result = original(*args, **kwargs)
        fresh_fold_fits.append(len(result["fit_time"]))
        return result

    monkeypatch.setattr(genetic_search, "cross_validate", count_cv)
    resumed = clone(model).fit(X, y, callbacks=[CVBudgetStopping(3), checkpoint])

    assert fresh_fold_fits == [2, 2, 2]
    assert resumed.fit_counts_["search_cv_calls"] == checkpoint.resumable_cv_calls_ == 3
    assert resumed.history["cross_validate_calls"] == [6, 9]
    assert resumed.fit_stats_["cross_validate_calls"] == 9
    assert checkpoint.load()["runtime_state"]["fit_stats_"]["cross_validate_calls"] == 9


@pytest.mark.parametrize(
    "runtime, logbook, detail",
    [
        ([], [], r"runtime_state must be a mapping"),
        ({"fit_stats_": []}, [], r"runtime_state.fit_stats_ must be a mapping"),
        ({}, [{"cross_validate_calls": None}], r"logbook\[-1\].cross_validate_calls"),
    ],
)
def test_legacy_checkpoint_rejects_unusable_cv_accounting(runtime, logbook, detail, tmp_path):
    from ..callbacks.model_checkpoint import load_search_checkpoint

    path = tmp_path / "unusable-stats.pkl"
    data = {"estimator_state": {}, "runtime_state": runtime, "logbook": logbook}
    contents = pickle.dumps(data)
    path.write_bytes(contents)

    with pytest.raises(ValueError, match=rf"unusable-stats\.pkl.*{detail}"):
        load_search_checkpoint([ModelCheckpoint(path)], optimizer=None)

    assert path.read_bytes() == contents


@pytest.mark.parametrize("custom_evaluator", [False, True])
def test_validated_feature_batches_preserve_optimizer_and_estimator_hooks(
    monkeypatch, custom_evaluator
):
    from .. import _backend_runtime

    X = np.random.default_rng(4513).normal(size=(40, 32))
    y = X[:, 0] > 0
    canonical = [1, 0] + [0] * 30
    mixed = [False, np.int64(1)] + [False] * 30
    initial_masks = []

    def optimizer(problem, evaluate, report):
        initial_masks.extend(problem.initial_population)
        for batch in (problem.initial_population, [canonical], [mixed]):
            report([handle for _, handle in evaluate(batch)])

    fast_batches, custom_batches = [], []
    evaluate = _backend_runtime.evaluate_population
    evaluate_with_repair = GAFeatureSelectionCV.evaluate_population

    def record_fast(estimator, individuals, cache_record_key, **kwargs):
        if not fast_batches:
            assert all(row is mask for row, mask in zip(individuals, initial_masks, strict=True))
        fast_batches.append(len(individuals))
        return evaluate(estimator, individuals, cache_record_key, **kwargs)

    def record_custom(estimator, individuals):
        assert all(type(row) is list for row in individuals)
        custom_batches.append(tuple(type(bit) for bit in individuals[0][:2]))
        return evaluate_with_repair(estimator, individuals)

    monkeypatch.setattr(_backend_runtime, "evaluate_population", record_fast)
    if custom_evaluator:
        monkeypatch.setattr(GAFeatureSelectionCV, "evaluate_population", record_custom)
    selector = GAFeatureSelectionCV(
        DecisionTreeClassifier(random_state=4513),
        optimizer_backend=optimizer,
        max_features=2,
        cv=2,
        population_size=6,
        generations=2,
        random_state=4513,
        use_cache=False,
        verbose=False,
    ).fit(X, y)

    assert selector.fit_counts_["search_cv_calls"] == 8
    np.testing.assert_array_equal(selector.cv_results_["features"][-2:], [canonical, mixed])
    assert fast_batches == ([] if custom_evaluator else [6, 1])
    assert custom_batches == (
        [(int, int), (int, int), (bool, np.int64)] if custom_evaluator else []
    )


def test_shared_initial_masks_are_copied_for_repair_if_on_start_changes_the_cap():
    from ..callbacks import BaseCallback

    class LowerLimit(BaseCallback):
        def on_start(self, estimator=None):
            estimator.max_features = 1

    def optimizer(problem, evaluate, report):
        assert problem.max_features == 3
        assert any(sum(mask) > 1 for mask in problem.initial_population)
        report([handle for _, handle in evaluate(problem.initial_population)])

    X = np.random.default_rng(45).normal(size=(40, 8))
    model = GAFeatureSelectionCV(
        DecisionTreeClassifier(random_state=45),
        optimizer_backend=optimizer,
        cv=2,
        max_features=3,
        population_size=6,
        generations=0,
        random_state=45,
        use_cache=False,
        verbose=False,
    ).fit(X, X[:, 0] > 0, callbacks=LowerLimit())

    assert model.fit_counts_["search_cv_calls"] == 6
    assert model.cv_results_["n_features"] == [1] * 6


@pytest.mark.parametrize("initializer", ["smart", "random", "spread"])
@pytest.mark.parametrize("engine", ["deap", "deap-budget", "plugin"])
def test_native_initial_masks_are_not_repaired_again(initializer, engine):
    from cProfile import Profile

    X = np.random.default_rng(23).normal(size=(40, 16))
    y = X[:, 0] > 0
    size = 6

    def optimizer(problem, evaluate, report):
        for mask in problem.initial_population:
            assert set(mask) <= {0, 1}
            assert 1 <= sum(mask) <= 3
        report([handle for _, handle in evaluate(problem.initial_population)])

    model = GAFeatureSelectionCV(
        DecisionTreeClassifier(random_state=23),
        optimizer_backend=optimizer if engine == "plugin" else "deap",
        population_initializer=initializer,
        random_state=23,
        cv=2,
        max_features=3,
        population_size=size,
        generations=0,
        use_cache=False,
        verbose=False,
    )
    callbacks = CVBudgetStopping(size) if engine == "deap-budget" else None
    with Profile() as profile:
        model.fit(X, y, callbacks=callbacks)
    repair = GAFeatureSelectionCV._repair_individual.__code__
    calls = sum(entry.callcount for entry in profile.getstats() if entry.code is repair)
    max_repairs = size * int(initializer == "random")
    assert calls <= max_repairs
    assert model.fit_counts_["search_cv_calls"] == size
    assert all(1 <= mask.sum() <= 3 for mask in model.cv_results_["features"])


def test_deap_repairs_masks_from_an_overridden_random_sampler():
    from ..callbacks import BaseCallback

    X = np.random.default_rng(23).normal(size=(40, 16))
    y = X[:, 0] > 0
    observed = []

    class RecordInitialMasks(BaseCallback):
        def on_start(self, estimator=None):
            observed.extend(tuple(mask) for mask in estimator._pop)

    model = GAFeatureSelectionCV(
        DecisionTreeClassifier(random_state=23),
        population_initializer="random",
        random_state=23,
        cv=2,
        max_features=2,
        population_size=6,
        generations=0,
        verbose=False,
    )
    model._new_feature_individual = lambda: model.toolbox._individual_type([2] * X.shape[1])
    model.fit(X, y, callbacks=RecordInitialMasks())

    assert len(observed) == model.population_size
    assert all(set(mask) == {0, 1} and sum(mask) == 2 for mask in observed)


@pytest.mark.parametrize("override_class", [False, True])
def test_deap_repairs_initial_masks_after_custom_on_start(override_class, monkeypatch):
    start_budget = CVBudgetStopping.on_start

    def edit_initial_masks(callback, estimator=None):
        start_budget(callback, estimator)
        estimator._pop[0][:] = [2] * estimator.n_features
        estimator._pop[1][:] = [0] * estimator.n_features

    class EditInitialMasks(CVBudgetStopping):
        on_start = edit_initial_masks

    if override_class:
        monkeypatch.setattr(CVBudgetStopping, "on_start", edit_initial_masks)
        callback = CVBudgetStopping(6)
    else:
        callback = EditInitialMasks(6)

    X = np.random.default_rng(49).normal(size=(40, 8))
    selector = GAFeatureSelectionCV(
        DecisionTreeClassifier(random_state=49),
        cv=2,
        max_features=2,
        population_size=6,
        generations=0,
        random_state=49,
        use_cache=False,
        verbose=False,
    ).fit(X, X[:, 0] > 0, callbacks=callback)

    assert selector.fit_counts_["search_cv_calls"] == 6
    np.testing.assert_array_equal(selector.cv_results_["n_features"][:2], [2, 1])


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


@pytest.mark.parametrize(
    "lower, upper, value, distribution",
    [(-8e307, 8e307, 0.0, "uniform"), (1e-300, 1e300, 1.0, "log-uniform")],
)
@pytest.mark.parametrize("radius", [1.0, np.float64(1.0)], ids=["float", "numpy"])
def test_wide_continuous_neighbor_can_sample_interior_and_bounds(
    lower, upper, value, distribution, radius, monkeypatch
):
    dimension = Continuous(lower, upper, distribution=distribution)
    quantiles = iter((0.1, 0.5, 0.9))
    monkeypatch.setattr(
        random, "uniform", lambda left, right: left + (right - left) * next(quantiles)
    )
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        assert [_bounded_continuous_neighbor(dimension, value, radius) for _ in range(3)] == [
            lower,
            value,
            upper,
        ]


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


@pytest.mark.parametrize("category_kind", ["array", "ordered", "plain"])
def test_dictionary_category_neighbors_and_distances(category_kind):
    from collections import OrderedDict
    from copy import deepcopy
    from ..space.base import BaseDimension

    if category_kind != "array":
        mapping_type = OrderedDict if category_kind == "ordered" else dict
        current = mapping_type([("shift", 1.0), ("power", 2)])
        other = mapping_type(reversed(list(current.items())))
    else:
        current = {"offset": np.zeros(4)}
        other = {"offset": np.ones(4)}
    copied = deepcopy(current)
    estimator = SimpleNamespace(
        space=Space({"transform__kw_args": Categorical([current, other])}),
        sharing_radius=1.0,
        sharing_alpha=1.0,
    )

    assert local_neighbor(estimator, [copied], list)[0] is other
    assert individual_distance(estimator, [current], [copied]) == 0.0
    assert individual_distance(estimator, [current], [other]) == 1.0
    assert niche_counts([[current], [copied], [other]], estimator) == [2.0, 2.0, 1.0]

    class DictionaryDimension(BaseDimension):
        def sample(self):
            return current

    estimator.space = Space({"transform__kw_args": DictionaryDimension()})
    assert individual_distance(estimator, [current], [copied]) == 0.0
    assert individual_distance(estimator, [current], [other]) == 1.0
    assert niche_counts([[current], [copied], [other]], estimator) == [2.0, 2.0, 1.0]


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


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    [
        (-4, 3, [-4, -3, -2, 3, 3]),
        (2**60, 2**60 + 7, [2**60, 2**60 + 1, 2**60 + 2, 2**60 + 7, 2**60 + 7]),
        (2**80, 2**80, [2**80] * 5),
    ],
)
def test_latin_hypercube_integer_decoding_preserves_bounds_and_types(lower, upper, expected):
    units = (0.0, np.nextafter(0.25, 0.0), 0.25, np.nextafter(1.0, 0.0), 1.0)
    decoded = [_sample_dimension_from_unit(Integer(lower, upper), value) for value in units]
    assert decoded == expected
    assert all(type(value) is int for value in decoded)


@pytest.mark.parametrize(
    "dimension",
    [
        Continuous(1e-7, 1e2, distribution="log-uniform"),
        Continuous(1e10, float(np.nextafter(1e10, np.inf)), distribution="log-uniform"),
        Continuous(2**53 + 1, 2**53 + 20),
        Continuous(-0.0, 0.0),
    ],
)
def test_prepared_continuous_coordinates_preserve_scalar_values(dimension):
    from .._mixed_space import decode_real, encode_real, real_coordinate_interval

    lower, upper = dimension._sampling_bounds()
    interval = real_coordinate_interval(dimension)
    for unit in (0.0, np.nextafter(0.0, 1.0), 0.1, 0.5, np.nextafter(1.0, 0.0), 1.0):
        if lower == upper:
            expected = lower
        else:
            raw = lower + unit * (upper - lower)
            if dimension.distribution == "log-uniform":
                log_lower, log_upper = np.log(lower), np.log(upper)
                if log_lower != log_upper:
                    raw = np.exp(log_lower + unit * (log_upper - log_lower))
            expected = dimension._clip_scalar(raw)
        prepared = decode_real(dimension, unit, interval=interval)
        assert prepared.hex() == expected.hex() == decode_real(dimension, unit).hex()
        assert dimension.lower <= prepared <= dimension.upper

        if lower == upper:
            encoded = 0.5
        elif dimension.distribution == "log-uniform" and np.log(lower) != np.log(upper):
            encoded = (np.log(expected) - np.log(lower)) / (np.log(upper) - np.log(lower))
        else:
            encoded = (expected - lower) / (upper - lower)
        assert encode_real(dimension, expected, interval=interval).hex() == encoded.hex()
        assert encode_real(dimension, expected).hex() == encoded.hex()


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


def test_smart_population_warm_start_early_return_and_feature_fallback(monkeypatch):
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

    # The first three exhaust the grid. Later duplicates remain independent chromosomes.
    feature_estimator = SimpleNamespace(
        population_size=10, n_features=2, max_features=None, use_cache=True
    )
    positions = iter([1, 0, 1, 0, 1])
    draws = SimpleNamespace(
        sample=lambda indices, k: [next(positions)] if k else [],
        randint=lambda low, high: high,
    )
    monkeypatch.setattr("sklearn_genetic.population.random", draws)
    masks = smart_feature_population(feature_estimator, toolbox, list)
    assert masks == [[0, 1], [1, 0], [1, 1]] + [[0, 1], [1, 0], [0, 1]] + [[1, 1]] * 4
    masks[0][0] = 1
    assert masks[3] == [0, 1]


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


@pytest.mark.parametrize("custom_dimension", [False, True])
def test_parameter_batches_dispatch_prepared_rows_for_native_dimensions(
    monkeypatch, custom_dimension
):
    from .. import genetic_search

    class CustomInteger(Integer):
        pass

    dimension = (CustomInteger if custom_dimension else Integer)(2, 5)
    search = GASearchCV(DecisionTreeClassifier(), param_grid={"max_depth": dimension})
    rows = [[1], [np.int64(5)], [7]]

    def dispatch(estimator, individuals, record_key, **kwargs):
        assert individuals == [[2], [5], [5]]
        assert record_key == "current_generation_params"
        evaluate = kwargs.get("evaluate")
        if custom_dimension:
            assert evaluate is None
        else:
            assert evaluate is not None
            assert evaluate == estimator._evaluate_prepared_individual
        return []

    monkeypatch.setattr(genetic_search, "_evaluate_population_batch", dispatch)
    assert search.evaluate_population(rows) == []


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


@pytest.mark.parametrize("record_key", ["current_generation_params", "current_generation_features"])
def test_evaluate_population_deduplicates_updates_cache_and_fit_stats(record_key):
    calls = []
    keys = []
    recorded_keys = []

    class RecordingCache(dict):
        def get(self, key, default=None):
            recorded_keys.append(key)
            return super().get(key, default)

    def individual_key(individual):
        keys.append(tuple(individual))
        return keys[-1]

    def evaluate_individual(individual, n_jobs=None):
        calls.append((individual, n_jobs))
        value = individual[0]
        return [value], {"value": value}, True, False

    estimator = SimpleNamespace(
        use_cache=True,
        fitness_cache=RecordingCache(),
        logbook=tools.Logbook(),
        fit_stats_=create_fit_stats(),
        n_jobs=1,
        parallel_backend="auto",
        log_config=None,
        _individual_key=individual_key,
        _evaluate_individual=evaluate_individual,
    )

    fitnesses = evaluate_population(
        estimator,
        individuals=[[1], [1], [2]],
        cache_record_key=record_key,
    )

    assert fitnesses == [[1], [1], [2]]
    assert calls == [([1], 1), ([2], 1)]
    assert len(estimator.logbook.chapters["parameters"]) == 3
    assert estimator.fit_stats_["evaluated_candidates"] == 3
    assert estimator.fit_stats_["unique_candidates"] == 2
    assert estimator.fit_stats_["duplicate_candidates"] == 1
    assert estimator.fit_stats_["cross_validate_calls"] == 2
    assert keys == [(1,), (1,), (2,)]
    assert recorded_keys == [(1,), (2,)]

    fitnesses = evaluate_population(
        estimator,
        individuals=[[2], [3], [3], [1], [2]],
        cache_record_key=record_key,
    )
    assert fitnesses == [[2], [3], [3], [1], [2]]
    assert recorded_keys == [(1,), (2,), (2,), (3,), (1,)]
    assert estimator.fit_stats_["evaluated_candidates"] == 8
    assert estimator.fit_stats_["cache_hits"] == 3
    assert estimator.fit_stats_["duplicate_candidates"] == 2
    assert estimator.fit_stats_["cross_validate_calls"] == 3


@pytest.mark.parametrize("use_cache", [False, True])
def test_serial_population_copies_each_candidate_at_evaluation(use_cache):
    copied = []
    evaluated = []

    class Candidate(list):
        def __iter__(self):
            copied.append(self[0])
            return super().__iter__()

    def evaluate_individual(candidate, n_jobs=None):
        value = candidate[0]
        evaluated.append(value)
        assert copied == evaluated
        candidate[0] = -1
        return [value], {"value": value}, True, False

    rows = [Candidate([value]) for value in range(6)]
    estimator = SimpleNamespace(
        use_cache=use_cache,
        fitness_cache={},
        logbook=tools.Logbook(),
        fit_stats_=create_fit_stats(),
        n_jobs=1,
        parallel_backend="auto",
        log_config=None,
        _individual_key=lambda row: row[0],
        _evaluate_individual=evaluate_individual,
    )

    assert evaluate_population(estimator, rows, "current_generation_params") == [
        [value] for value in range(6)
    ]
    assert [row[0] for row in rows] == list(range(6))


@pytest.mark.parametrize("use_cache", [False, True])
@pytest.mark.parametrize("limit", ["calls", "timer", "both_calls", "both_timer"])
@pytest.mark.parametrize("record_key", ["current_generation_params", "current_generation_features"])
def test_evaluate_population_copies_only_dispatched_candidates(use_cache, limit, record_key):
    copied = []
    evaluated = []
    accessed = []
    cache_keys = []

    class Population(list):
        def __iter__(self):
            for candidate in super().__iter__():
                accessed.append(candidate[0])
                yield candidate

        def __getitem__(self, index):
            candidate = super().__getitem__(index)
            accessed.append(candidate[0])
            return candidate

    class Candidate(list):
        def __iter__(self):
            copied.append(self[0])
            return super().__iter__()

    class StopAfterDispatches(TimerStopping):
        def _time_exceeded(self):
            return len(evaluated) == 3

    def evaluate_individual(candidate, n_jobs=None):
        assert type(candidate) is list
        value = candidate[0]
        candidate[0] = -1  # Evaluation cannot change the optimizer's candidate.
        evaluated.append(value)
        return [value], {"value": value}, value != 1, value == 1

    def individual_key(candidate):
        cache_keys.append(candidate[0])
        return candidate[0]

    rows = [0, 1, 1, 2, 3, 2, 4, 0, 5, 0, 6] if use_cache else [1, 2, 3, 4, 5]
    candidates = Population(Candidate([value]) for value in rows)
    callbacks = []
    if limit != "timer":
        callbacks.append(CVBudgetStopping(5 if limit == "both_timer" else 2))
    if limit != "calls":
        callbacks.append(StopAfterDispatches(1))
    cache = {0: {"fitness": [0], record_key: {"value": 0}}} if use_cache else {}
    estimator = SimpleNamespace(
        callbacks=callbacks,
        use_cache=use_cache,
        fitness_cache=cache,
        logbook=tools.Logbook(),
        fit_stats_=create_fit_stats(),
        n_jobs=1,
        parallel_backend="auto",
        log_config=None,
        _individual_key=individual_key,
        _evaluate_individual=evaluate_individual,
    )

    with evaluation_context(estimator):
        fitnesses = evaluate_population(estimator, candidates, record_key)
        assert estimator._evaluation_budget_exhausted == (limit in {"timer", "both_timer"})

    accepted = rows[:6] if use_cache else rows[:3]
    assert evaluated == copied == [1, 2, 3]
    if use_cache:
        # Include the first fresh proposal beyond the limit to find the end
        # of the reusable prefix without inspecting the rest of the batch.
        # The timer can win after keys for the next dispatch are prepared.
        checked = 9 if limit in {"timer", "both_timer"} else 7
        assert cache_keys == rows[:checked]
    else:
        assert accessed == evaluated
        assert cache_keys == []
    assert [row[0] for row in candidates] == accepted
    assert fitnesses == [[value] for value in accepted]
    assert [row["value"] for row in estimator.logbook.chapters["parameters"]] == accepted
    assert estimator.fit_stats_["cross_validate_calls"] == 2
    assert estimator.fit_stats_["skipped_invalid_candidates"] == 1
    assert estimator.fit_stats_["cache_hits"] == int(use_cache)
    assert estimator.fit_stats_["duplicate_candidates"] == 2 * int(use_cache)


def test_spent_cv_budget_with_timer_prepares_only_the_leading_cached_rows():
    keys = []

    def record_key(row):
        keys.append(row[0])
        return row[0]

    estimator = SimpleNamespace(
        callbacks=[TimerStopping(60), CVBudgetStopping(1)],
        use_cache=True,
        fitness_cache={0: {"fitness": [0], "current_generation_params": {"value": 0}}},
        logbook=tools.Logbook(),
        fit_stats_=create_fit_stats(),
        _individual_key=record_key,
    )
    rows = [[0], [0], [1], [0], [2]]
    with evaluation_context(estimator):
        estimator.fit_stats_["cross_validate_calls"] = 1
        fitnesses = evaluate_population(estimator, rows, "current_generation_params")

    assert keys == [0, 0, 1]
    assert rows == fitnesses == [[0], [0]]
    assert estimator.fit_stats_["cache_hits"] == 2
    assert estimator.fit_stats_["cross_validate_calls"] == 1


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
    search = GASearchCV(
        DecisionTreeClassifier(),
        param_grid={"max_depth": Integer(1, 4), "criterion": Categorical(["gini", "entropy"])},
    )
    search.best_estimator_ = clone(search.estimator).fit([[0], [1]], [0, 1])
    search.refit = True
    search._n_iterations = 1
    search.history = {"gen": [0, 1], "fitness": [0.5, 0.8]}

    assert history_record(search.history, 1) == {"gen": 1, "fitness": 0.8}
    assert search[0] == {"gen": 0, "fitness": 0.5}
    assert list(search) == [
        {"gen": 0, "fitness": 0.5},
        {"gen": 1, "fitness": 0.8},
    ]
    assert len(search) == 2

    search.refit = False
    assert search[0] == {"gen": 0, "fitness": 0.5}
    del search.history
    with pytest.raises(NotFittedError, match="GASearchCV instance is not fitted yet"):
        search[0]


def test_rank_scores_orders_finite_scores_descending_for_maximization():
    ranks = _rank_scores([0.7, 0.9, 0.5])

    assert list(ranks) == [2, 1, 3]


@pytest.mark.parametrize(
    "scores, expected",
    [
        ([0.9, 0.9, 0.5], [1, 1, 3]),
        ([0.5] * 64 + [0.9] + [0.5] * 64, [2] * 64 + [1] + [2] * 64),
    ],
)
def test_rank_scores_breaks_ties_with_deterministic_min_method(scores, expected):
    np.testing.assert_array_equal(_rank_scores(scores), expected)


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


@pytest.mark.parametrize("maximize", [False, True])
@pytest.mark.parametrize(
    "scores, counts",
    [
        (
            [0.5, np.nan, -1e308, 0.5, np.inf, 1e308, -np.inf, -0.0, 0.0],
            [7, 2, 1, 8, 5, 3, 9, 2, 4],
        ),
        ([0.5, -1e308, 0.5, 1e308, -0.0, 0.0], [7, 1, 8, 3, 2, 4]),
        ([np.nan, np.inf, -np.inf], [4, 1, 7]),
    ],
)
def test_rank_scores_counts_reused_rows_for_ties_and_nonfinite_scores(maximize, scores, counts):
    counts = np.asarray(counts)
    expanded = _rank_scores(np.repeat(scores, counts), maximize)
    first = np.cumsum(counts) - counts
    actual = _rank_scores(scores, maximize, counts=counts)
    np.testing.assert_array_equal(actual, expanded[first])


@pytest.mark.parametrize("maximize", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_fully_tied_finite_scores_keep_nonfinite_rows_last_without_sorting(
    maximize, weighted, monkeypatch
):
    from sklearn_genetic.utils import cv_scores

    scores = np.resize([0.0, np.nan, -0.0, np.inf, -np.inf], 200)
    finite = np.isfinite(scores)
    counts = np.arange(len(scores), dtype=int) % 4 + 1 if weighted else None
    finite_count = counts[finite].sum() if weighted else finite.sum()

    def unexpected_sort(*args, **kwargs):
        pytest.fail("a single finite score does not need sorting")

    with monkeypatch.context() as replacements:
        replacements.setattr(cv_scores, "rankdata", unexpected_sort)
        replacements.setattr(np, "argsort", unexpected_sort)
        ranks = _rank_scores(scores, maximize, counts=counts)

    np.testing.assert_array_equal(ranks, np.where(finite, 1, finite_count + 1))


def test_ranked_cv_columns_match_for_reused_training_scores_and_failed_rows():
    from sklearn_genetic.utils.cv_scores import _add_cv_statistics

    failed = np.full(2, np.nan)
    timing = np.zeros(2)
    train = np.array([0.81, 0.95], dtype=np.float32)
    rows = []
    for scores in (failed, np.array([0.7, 0.9]), np.array([0.9, 0.7]), np.array([0.6, 0.7])):
        rows.append(
            dict(
                cv_scores=scores,
                test_score=scores,
                train_score=train,
                fit_time=None if scores is failed else timing,
                score_time=timing,
            )
        )
    records = [rows[index] for index in [1, 2, 0, 3, 3, 3, 0] * 12]
    baseline, reused = {}, {}
    _add_cv_statistics(baseline, records, True, ["score"], True)
    _add_cv_statistics(reused, records, True, ["score"], True, unique_candidates=len(rows))
    np.testing.assert_equal(reused, baseline)


@pytest.mark.parametrize("names", [("score",), ("payload", "score", "index")])
def test_parameter_cv_results_reuse_grid_order_without_sharing_row_dicts(names):
    from sklearn_genetic.evaluation import preserve_candidate_parameters
    from sklearn_genetic.utils.cv_scores import create_gasearch_cv_results_

    class CountingSpace(Space):
        reads = 0

        @property
        def parameters(self):
            self.reads += 1
            return super().parameters

    choices = {"payload": {"mode": "fast"}, "score": ["automatic"], "index": np.int64(4)}
    params = {name: choices[name] for name in names}
    space = CountingSpace({name: Categorical([value]) for name, value in params.items()})
    space.reads = 0
    record = preserve_candidate_parameters(params, {"index", "score"})
    scores = np.array([0.5, 0.7])
    record.update(index=0, score=0.6, cv_scores=scores, test_score=scores)
    record.update(fit_time=scores, score_time=scores)
    result = create_gasearch_cv_results_([record, record], space, False, ["score"])

    assert list(result)[: len(names)] == [f"param_{name}" for name in names]
    assert list(result)[-1] == "params"
    assert result["params"][0] is not result["params"][1]
    for row in result["params"]:
        assert list(row) == list(names)
        assert all(row[name] is params[name] for name in names)
    for name in names:
        assert all(value is params[name] for value in result[f"param_{name}"])
    assert space.reads == 1


def test_feature_cv_results_preserve_mask_counts_ranks_and_scalar_types():
    from sklearn_genetic.utils.cv_scores import create_feature_selection_cv_results_

    class LegacyMask(np.ndarray):
        def sum(self, *args, **kwargs):
            return super().sum(*args, **kwargs) + 4

    ordinary = np.array([True, False, True, False, False])
    masks = [
        ordinary,
        ordinary[::2],
        np.array([1, -2, 2], dtype=np.int16),
        [1, 0, 1],
        ordinary.view(LegacyMask),
    ]
    scores = np.array([0.5, 0.8])
    records = [
        dict(features=mask, cv_scores=scores, test_score=scores, fit_time=scores, score_time=scores)
        for mask in masks
    ]
    results = create_feature_selection_cv_results_(records, False, ["score"])
    expected = [np.sum(mask) for mask in masks]

    assert results["n_features"] == expected
    assert list(map(type, results["n_features"])) == list(map(type, expected))
    np.testing.assert_array_equal(results["rank_n_features"], [2, 2, 1, 2, 5])
    assert all(actual is original for actual, original in zip(results["features"], masks))


def test_legacy_invalid_feature_mask_scores_are_hidden_when_formatting_results():
    from sklearn_genetic.utils.cv_scores import create_feature_selection_cv_results_

    masks = np.array([[True, True], [True, False]])
    scores = np.array([[-100000.0, -100000.0], [-1000000.0, -1000000.0]])
    records = [
        dict(
            features=mask,
            cv_scores=row,
            test_score=row,
            train_score=row,
            fit_time=np.zeros(2),
            score_time=np.zeros(2),
        )
        for mask, row in zip(masks, scores, strict=True)
    ]
    results = create_feature_selection_cv_results_(records, True, ["score"], mask_limit=1)
    np.testing.assert_array_equal(results["skipped_invalid_mask"], [True, False])
    np.testing.assert_array_equal(results["all_fits_failed"], [False, False])
    for subset in ("train", "test"):
        np.testing.assert_array_equal(results[f"mean_{subset}_score"], [np.nan, -1000000])
        np.testing.assert_array_equal(results[f"rank_{subset}_score"], [2, 1])
    np.testing.assert_array_equal(records[0]["test_score"], [-100000, -100000])


def test_repeated_feature_cv_rows_count_each_shared_boolean_mask_once(monkeypatch):
    from sklearn_genetic.utils.cv_scores import create_feature_selection_cv_results_

    original = np.array([True, False, False, True, True, False])
    boolean_masks = (original, original.copy(), original[::2])
    legacy = np.array([1, -2, 0, 1], dtype=np.int16)
    masks = [boolean_masks[2], original, legacy, boolean_masks[1], original] * 16
    scores = np.array([0.5, 0.8])
    records = [
        dict(features=mask, cv_scores=scores, test_score=scores, fit_time=scores, score_time=scores)
        for mask in masks
    ]
    reference = create_feature_selection_cv_results_(records, False, ["score"])
    mask_ids = {id(mask) for mask in boolean_masks}
    counted_masks = []
    count_nonzero = np.count_nonzero

    def track_count(array, *args, **kwargs):
        if id(array) in mask_ids:
            counted_masks.append(id(array))
        return count_nonzero(array, *args, **kwargs)

    monkeypatch.setattr(np, "count_nonzero", track_count)
    result = create_feature_selection_cv_results_(records, False, ["score"], unique_candidates=4)

    np.testing.assert_equal(result, reference)
    assert list(map(type, result["n_features"])) == list(map(type, reference["n_features"]))
    assert all(actual is original for actual, original in zip(result["features"], masks))
    assert counted_masks == [id(boolean_masks[2]), id(original), id(boolean_masks[1])]


@pytest.mark.parametrize("sizes", [(0, 1, 7, 1), (0, 1, 512, 1), (-2, 0, 7, 7), (0, 0.5, 2, 2)])
def test_feature_count_ranks_preserve_minimum_ties_in_long_histories(sizes, monkeypatch):
    from scipy.stats import rankdata
    from sklearn_genetic.utils import cv_scores

    masks = [
        np.ones(size, dtype=bool) if type(size) is int and size >= 0 else [size] for size in sizes
    ] * 64
    scores = np.array([0.5, 0.8])
    records = [
        dict(
            features=mask,
            cv_scores=scores,
            test_score=scores,
            fit_time=scores,
            score_time=scores,
        )
        for mask in masks
    ]
    counted = []
    bincount = np.bincount

    def track_counts(values):
        counted.append(values)
        return bincount(values)

    monkeypatch.setattr(np, "bincount", track_counts)
    result = cv_scores.create_feature_selection_cv_results_(records, False, ["score"])
    expected = list(sizes) * 64
    assert result["n_features"] == expected
    np.testing.assert_array_equal(result["rank_n_features"], rankdata(expected, method="min"))
    eligible = all(type(size) is int and 0 <= size <= len(records) for size in sizes)
    assert len(counted) == int(eligible)


def test_feature_shortlist_keeps_the_denser_distinct_tie_in_a_large_history(monkeypatch):
    estimator = GAFeatureSelectionCV(DecisionTreeClassifier(), final_selection_top_k=3)
    estimator.refit_metric = "score"
    masks = [
        [1, 1, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ] + [[1, 1, 0, 0]] * 2042
    estimator.cv_results_ = {
        "rank_test_score": [1] * 6 + [7] * 6 + [13] * 2036,
        "features": masks,
        "n_features": [sum(mask) for mask in masks],
    }
    sort = np.lexsort
    sizes = []

    def record_sort(keys):
        sizes.append(len(keys[-1]))
        return sort(keys)

    monkeypatch.setattr(np, "lexsort", record_sort)
    assert estimator._top_candidate_indices() == [1, 2, 4]
    assert sizes == [12]


def test_top_candidate_indices_widens_past_a_prefix_of_repeated_parameters():
    grid = {"max_depth": Integer(1, 5), "criterion": Categorical(["gini"])}
    estimator = GASearchCV(DecisionTreeClassifier(), param_grid=grid, final_selection_top_k=3)
    estimator.space = Space(grid)
    estimator.refit_metric = "score"
    estimator.cv_results_ = {
        "rank_test_score": [1] * 12 + [13, 14] + [15] * 2034,
        "params": [
            {"max_depth": depth, "criterion": "gini"} for depth in [2] * 12 + [3, 4] + [5] * 2034
        ],
    }
    assert estimator._top_candidate_indices() == [0, 12, 13]


def test_fully_tied_parameters_keep_row_order_without_sorting(monkeypatch):
    grid = {"max_depth": Integer(1, 5), "criterion": Categorical(["gini"])}
    estimator = GASearchCV(DecisionTreeClassifier(), param_grid=grid, final_selection_top_k=3)
    estimator.space = Space(grid)
    estimator.refit_metric = "score"
    depths = [3, 3, 2, 4] + [5] * 2044
    estimator.cv_results_ = {
        "rank_test_score": np.ones(len(depths), dtype=int),
        "params": [{"max_depth": value, "criterion": "gini"} for value in depths],
    }

    def unexpected_sort(*args, **kwargs):
        pytest.fail("equal ranks have a stable row order already")

    monkeypatch.setattr(np, "argsort", unexpected_sort)
    assert estimator._top_candidate_indices() == [0, 2, 3]


@pytest.mark.parametrize(("width", "batch"), [(20, True), (4096, True), (32768, False)])
@pytest.mark.parametrize("cache_rows", [False, True])
def test_fully_tied_masks_pack_denser_candidates_and_keep_first_occurrences(
    width, batch, cache_rows, monkeypatch
):
    estimator = GAFeatureSelectionCV(DecisionTreeClassifier(), final_selection_top_k=3)
    estimator.refit_metric = "score"
    sparse = np.eye(3, width, dtype=bool)
    dense = np.zeros(width, dtype=bool)
    dense[: width // 2] = True
    shifted = np.roll(dense, 1)
    masks = [*sparse, dense, shifted, dense.copy()] + [dense] * 2042
    estimator.cv_results_ = {
        "rank_test_score": np.ones(len(masks), dtype=int),
        "features": masks,
        "n_features": [int(np.count_nonzero(mask)) for mask in masks],
    }
    if cache_rows:
        estimator.fit_stats_ = {"unique_candidates": 6}
    packbits = np.packbits
    packed = []

    def observe(values, axis=None):
        packed.append((axis, 1 if axis is None else len(values)))
        return packbits(values, axis=axis)

    monkeypatch.setattr(np, "packbits", observe)
    assert estimator._top_candidate_indices() == [0, 1, 4]
    assert sum(count for _, count in packed) == (6 if cache_rows else len(masks))
    assert sum(count for axis, count in packed if axis == 1) == (
        len(masks) - 3 if batch and not cache_rows else 0
    )


@pytest.mark.parametrize("width", [512, 4096, 32768])
@pytest.mark.parametrize("custom_key", [False, True])
def test_feature_shortlist_packs_shared_masks_only_at_first_occurrence(
    width, custom_key, monkeypatch
):
    estimator = GAFeatureSelectionCV(DecisionTreeClassifier(), final_selection_top_k=3)
    estimator.refit_metric = "score"
    first, second, third = np.eye(3, width, dtype=bool)
    masks = [first] * 1024 + [first.copy(), second, third]
    estimator.fit_stats_ = {"unique_candidates": 4}
    estimator.cv_results_ = {
        "rank_test_score": np.ones(len(masks), dtype=int),
        "features": masks,
        "n_features": np.ones(len(masks), dtype=int),
    }
    if custom_key:
        original = estimator._finalist_key
        estimator._finalist_key = lambda index: (index, original(index))
    packbits = np.packbits
    packed = []

    def observe(mask, *args, **kwargs):
        packed.append(id(mask))
        return packbits(mask, *args, **kwargs)

    monkeypatch.setattr(np, "packbits", observe)
    if custom_key:
        assert estimator._top_candidate_indices() == [0, 1, 2]
        assert packed == [id(first)] * 3
    else:
        assert estimator._top_candidate_indices() == [0, 1025, 1026]
        assert packed == [id(mask) for mask in masks[1023:]]


@pytest.mark.parametrize("width", [128, 4096, 32768])
@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("custom_key", [False, True])
def test_large_tied_feature_cutoff_packs_only_the_densest_native_masks(
    width, flat, custom_key, monkeypatch
):
    estimator = GAFeatureSelectionCV(DecisionTreeClassifier(), final_selection_top_k=3)
    estimator.refit_metric = "score"
    middle = np.arange(width) < width // 4
    dense = np.arange(width) < width // 2
    shifted = np.roll(dense, 1)
    masks = [*np.eye(3, width, dtype=bool), *([middle] * 2041)]
    first_dense = len(masks)
    masks.extend([dense, shifted, dense.copy(), shifted.copy()])
    ranks = np.ones(len(masks), dtype=int)
    if not flat:
        # A denser mask with a worse score cannot enter the tied group.
        masks.append(np.ones(width, dtype=bool))
        ranks = np.append(ranks, len(masks))
    estimator.cv_results_ = {
        "rank_test_score": ranks,
        "features": masks,
        "n_features": [int(np.count_nonzero(mask)) for mask in masks],
    }
    packbits = np.packbits
    packed = []

    def observe(values, axis=None):
        packed.append(1 if axis is None else len(values))
        return packbits(values, axis=axis)

    if custom_key:
        # A custom key can identify masks of different densities as duplicates.
        original = estimator._finalist_key
        estimator._finalist_key = lambda index: original(index) if index < 3 else "alias"
        assert estimator._top_candidate_indices() == [0, 1, 3]
    else:
        monkeypatch.setattr(np, "packbits", observe)
        assert estimator._top_candidate_indices() == [0, 1, first_dense + 1]
        assert sum(packed) == 7


@pytest.mark.parametrize("prior_denser_finalist", [False, True])
def test_tied_feature_cutoff_keeps_distinct_masks_with_duplicate_largest_keys(
    prior_denser_finalist,
):
    estimator = GAFeatureSelectionCV(DecisionTreeClassifier(), final_selection_top_k=3)
    estimator.refit_metric = "score"
    width = 64
    sparse = np.eye(3, width, dtype=bool)
    dense = np.arange(width) < width // 2
    if prior_denser_finalist:
        middle = np.arange(width) < width // 4
        initial = [dense, *sparse[:2]]
        ranks = [1] + [2] * 1026
    else:
        # Legacy non-Boolean rows can share a key with a different recorded size.
        middle = dense.astype(float) * 0.5
        initial = list(sparse)
        ranks = [1] * 1027
    masks = [*initial, *([middle] * 1023), dense.copy()]
    estimator.cv_results_ = {
        "rank_test_score": ranks,
        "features": masks,
        "n_features": [mask.sum() for mask in masks],
    }
    assert estimator._top_candidate_indices() == [0, 1, 3]


@pytest.mark.parametrize("gene", [-1, 2, 255, 256, 2**100, True, np.int64(1), 0.5])
def test_wide_feature_diversity_uses_generic_values_for_non_native_genes(gene):
    population = [[0] * 32, [gene] + [0] * 31, [1] + [0] * 31]

    assert feature_population_diversity(population) == _population_diversity(population)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_population_stats_preserve_spreads_at_float_limits(dtype):
    limits = np.finfo(dtype)
    upper = dtype(limits.max)
    previous = np.nextafter(upper, dtype(0))
    smallest = dtype(limits.smallest_subnormal)
    with np.errstate(over="raise", under="raise", invalid="raise"):
        constant = population_fitness_stats([(upper,)] * 5)
        opposite = population_fitness_stats([(upper,), (-upper,)])
        neighboring = population_fitness_stats([(upper,), (previous,)])
        subnormal = population_fitness_stats([(smallest,), (-smallest,)])

    assert constant["fitness"] == upper
    assert constant["fitness_std"] == 0
    assert opposite["fitness"] == 0
    assert opposite["fitness_std"] == upper
    assert neighboring["fitness_std"] == (upper - previous) / 2
    assert subnormal["fitness"] == 0
    assert subnormal["fitness_std"] == smallest


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_row_statistics_match_individual_reductions(dtype, legacy, missing):
    rows = [
        np.array([0.25, np.nan if missing else -0.0, 0.75], dtype=dtype),
        np.array([0.1, 0.2, 0.3], dtype=dtype),
        # The deviation stays positive when the folds differ by one ULP.
        np.array([1024, np.nextafter(dtype(1024), dtype(1025)), 1024], dtype=dtype),
    ]
    if legacy:
        rows.append([0.25, 0.5, 0.75])
    for mean, std in [(np.mean, np.std), (np.nanmean, np.nanstd)]:
        split_scores = [[], [], []]
        means, deviations = _row_statistics(rows, 3, mean, std, split_scores=split_scores)
        for split, column in enumerate(split_scores):
            np.testing.assert_equal(column, [row[split] for row in rows])
            assert [type(value) for value in column] == [type(row[split]) for row in rows]
        for row, average, deviation in zip(rows, means, deviations, strict=True):
            expected_mean, expected_std = mean(row), std(row)
            assert type(average) is type(expected_mean)
            assert type(deviation) is type(expected_std)
            np.testing.assert_equal(average, expected_mean)
            np.testing.assert_equal(deviation, expected_std)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.longdouble])
def test_short_score_mean_preserves_dtype_and_extreme_value_handling(dtype):
    from sklearn_genetic.utils._score_statistics import score_mean

    largest = np.finfo(dtype).max
    with np.errstate(over="raise", invalid="raise"):
        ordinary = score_mean(np.array([0.25, 0.75], dtype=dtype))
        assert type(ordinary) is dtype and ordinary == 0.5
        assert score_mean(np.array([largest, largest], dtype=dtype)) == largest
        assert np.isnan(score_mean(np.array([largest, np.nan, -largest], dtype=dtype)))
        with pytest.raises(FloatingPointError):
            score_mean(np.array([np.inf, -np.inf], dtype=dtype))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("legacy", [False, True])
def test_row_statistics_keep_extreme_scores_and_failed_rows_separate(dtype, legacy):
    largest = np.finfo(dtype).max
    smallest = np.finfo(dtype).tiny
    cancellation = np.array([largest, largest, smallest, -largest, -largest], dtype=dtype)
    successful = np.array([largest, largest, largest, largest, largest], dtype=dtype)
    ordinary = np.array([0.25, 0.0, 0.75, 0.5, 0.0], dtype=dtype)
    failed = np.full(5, np.nan, dtype=dtype)
    rows = [cancellation, successful, ordinary.tolist() if legacy else ordinary, failed]
    with np.errstate(over="raise", under="raise", invalid="raise"):
        averages, deviations = _row_statistics(rows, 5, np.mean, np.std)

    for row, average, deviation in zip(rows[:3], averages, deviations, strict=False):
        scalar = type(np.mean(row[:1]))
        expected = list(map(float, row))
        assert type(average) is type(deviation) is scalar
        assert average == scalar(mean(expected))
        assert deviation == pytest.approx(scalar(pstdev(expected)), rel=1e-7, abs=0)
    assert np.isnan(averages[3]) and np.isnan(deviations[3])


@pytest.mark.parametrize(
    "size, unique, eligible", [(63, 1, False), (64, 16, True), (67, 17, False)]
)
def test_reused_score_rows_groups_by_identity_at_the_cutoff(size, unique, eligible):
    from sklearn_genetic.utils.cv_scores import _reused_score_rows

    distinct = [np.ones(2) for _ in range(unique)]
    positions = tuple(index % unique for index in range(size))
    rows = [distinct[position] for position in positions]
    representatives, index = _reused_score_rows(rows, 2)
    if eligible:
        assert index == positions
        assert all(before is after for before, after in zip(distinct, representatives, strict=True))
    else:
        assert representatives is rows
        assert index is None


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_row_statistics_reuse_cached_arrays_across_blocks_in_row_order(dtype, monkeypatch):
    from sklearn_genetic.utils import cv_scores

    folds = 50
    base = np.arange(folds, dtype=dtype)
    distinct = [base, base.copy(), -base, np.full(folds, np.nan, dtype=dtype)]
    rows = [distinct[index] for index in [2, 1, 3, 0] * 400]
    original_columns = [[] for _ in range(folds)]
    original = _row_statistics(rows, folds, np.mean, np.std, split_scores=original_columns)
    reduced = []
    moments = cv_scores._fold_score_moments

    def count(matrix):
        reduced.append(len(matrix))
        return moments(matrix)

    monkeypatch.setattr(cv_scores, "_fold_score_moments", count)
    reused_columns = [[] for _ in range(folds)]
    reused = _row_statistics(
        rows, folds, np.mean, np.std, split_scores=reused_columns, reuse_rows=True
    )

    assert reduced == [len(distinct)]
    np.testing.assert_array_equal(reused, original)
    np.testing.assert_array_equal(reused_columns, original_columns)
    assert all(type(value) is dtype for column in reused_columns for value in column)


def test_reset_adapters_calls_both_adapter_resets():
    calls = []
    estimator = SimpleNamespace(
        crossover_adapter=SimpleNamespace(reset=lambda: calls.append("crossover")),
        mutation_adapter=SimpleNamespace(reset=lambda: calls.append("mutation")),
    )

    reset_adapters(estimator)

    assert calls == ["crossover", "mutation"]


@pytest.mark.parametrize("selector", [False, True])
def test_fresh_fit_recomputes_scores_on_new_targets(selector):
    from sklearn.dummy import DummyRegressor

    model = DummyRegressor(strategy="constant", constant=0)
    kwargs = dict(
        cv=2,
        population_size=4,
        generations=0,
        verbose=False,
        scoring="neg_mean_absolute_error",
        random_state=13,
    )
    search = (
        GAFeatureSelectionCV(model, **kwargs)
        if selector
        else GASearchCV(model, param_grid={"constant": Categorical([0])}, **kwargs)
    )
    X = np.arange(12).reshape(-1, 1)
    search.fit(X, np.zeros(12))
    search.fit(X, np.ones(12))
    assert search.fit_stats_["cross_validate_calls"] == 1
    assert np.allclose(search.cv_results_["mean_test_score"], -1)


@pytest.mark.parametrize("search_class", [GASearchCV, GAFeatureSelectionCV])
@pytest.mark.parametrize("criteria,scores", [("max", [-3.0, -2.0, -1.0]), ("min", [3.0, 2.0, 1.0])])
def test_non_elitist_selection_respects_fitness_direction(search_class, criteria, scores):
    class Fitness(base.Fitness):
        weights = (1.0 if criteria == "max" else -1.0,)

    population = []
    for score in scores:
        individual = SimpleNamespace(fitness=Fitness((score,)))
        population.append(individual)
    kwargs = {"param_grid": {"max_depth": Integer(1, 3)}} if search_class is GASearchCV else {}
    search = search_class(DecisionTreeClassifier(), elitism=False, criteria=criteria, **kwargs)
    random.seed(41)
    selected = search.select(population, 300)
    assert len(selected) == 300
    assert sum(ind is population[-1] for ind in selected) > sum(
        ind is population[0] for ind in selected
    )


@pytest.mark.parametrize("max_features", [0, -1, 1.5, True, 4])
def test_feature_selection_rejects_invalid_feature_bounds(max_features):
    search = GAFeatureSelectionCV(
        DecisionTreeClassifier(),
        max_features=max_features,
        population_size=4,
        generations=0,
        verbose=False,
    )
    with pytest.raises(ValueError, match="max_features"):
        search.fit(np.ones((12, 3)), np.tile([0, 1], 6))


@pytest.mark.parametrize("algorithm", ["eaSimple", "eaMuPlusLambda", "eaMuCommaLambda"])
def test_feature_selection_evolves_only_nonempty_bounded_masks(algorithm):
    search = GAFeatureSelectionCV(
        DecisionTreeClassifier(random_state=0),
        max_features=1,
        population_size=6,
        generations=3,
        verbose=False,
        algorithm=algorithm,
        random_state=17,
        mutation_probability=0.5,
        crossover_probability=0.5,
    )
    search.fit(np.arange(36).reshape(12, 3), np.tile([0, 1], 6))
    assert search.get_support().sum() == 1
    assert all(np.sum(mask) == 1 for mask in search.cv_results_["features"])


@pytest.mark.parametrize("final_selection", [False, True])
def test_fit_minimization_selects_lowest_scoring_candidate(final_selection):
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    search = GASearchCV(
        DecisionTreeClassifier(),
        param_grid={"max_depth": Integer(1, 3)},
        warm_start_configs=[{"max_depth": 1}, {"max_depth": 3}],
        population_size=2,
        generations=0,
        criteria="min",
        scoring=lambda estimator, X, y: estimator.max_depth,
        final_selection=final_selection,
        final_selection_top_k=1,
        return_train_score=True,
        verbose=False,
        random_state=0,
    )
    search.fit(X, y)
    assert search.best_params_["max_depth"] == 1
    assert search.best_score_ == 1
    assert search.cv_results_["rank_test_score"].tolist() == [1, 2]
    assert search.cv_results_["rank_train_score"].tolist() == [1, 2]


def test_feature_minimization_ranks_train_and_test_scores():
    search = GAFeatureSelectionCV(
        DecisionTreeClassifier(),
        criteria="min",
        cv=2,
        population_size=4,
        generations=0,
        return_train_score=True,
        scoring=lambda estimator, X, y: X.shape[1],
        verbose=False,
    )
    search.fit(np.arange(48).reshape(12, 4), np.tile([0, 1], 6))
    assert search.get_support().sum() == 1
    for split in ["train", "test"]:
        scores = np.asarray(search.cv_results_[f"mean_{split}_score"])
        ranks = search.cv_results_[f"rank_{split}_score"]
        assert np.all(ranks[scores == 1] == 1)
        assert np.all(ranks[scores > 1] > 1)


@pytest.mark.parametrize("value", [0, 500000, 1000000])
def test_integer_local_search_has_bounded_memory(value):
    import tracemalloc

    estimator = SimpleNamespace(space=Space({"size": Integer(0, 1000000)}), local_search_radius=0.1)
    tracemalloc.start()
    try:
        neighbors = [local_neighbor(estimator, [value], list)[0] for _ in range(3)]
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert all(
        max(0, value - 100000) <= x <= min(1000000, value + 100000) and x != value
        for x in neighbors
    )
    assert peak < 1000000


@pytest.mark.parametrize("lower,upper,value", [(1, 5, 1), (1, 5, 3), (1, 5, 5), (7, 7, 7)])
def test_integer_local_search_reaches_all_neighbors(lower, upper, value):
    estimator = SimpleNamespace(
        space=Space({"size": Integer(lower, upper)}), local_search_radius=1.0
    )
    random.seed(23)
    from collections import Counter

    observed = Counter(local_neighbor(estimator, [value], list)[0] for _ in range(10000))
    expected = set(range(lower, upper + 1)) - {value} or {value}
    assert set(observed) == expected
    assert all(abs(count - 10000 / len(expected)) < 250 for count in observed.values())


@pytest.mark.parametrize("cap, size, peak_arrays", [(3, 2, 8), (49997, 2, 8), (12, 16, 23)])
def test_wide_smart_feature_initialization_limits_temporary_memory(
    cap, size, peak_arrays, monkeypatch
):
    import struct
    import tracemalloc

    width = 50000
    estimator = SimpleNamespace(
        population_initializer="smart", population_size=size, n_features=width, max_features=cap
    )
    monkeypatch.setattr("sklearn_genetic.population.random", random.Random(3101))
    tracemalloc.start()
    try:
        population = initialize_feature_population(estimator, None, list)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert [sum(mask) for mask in population] == np.rint(np.linspace(1, cap, size)).tolist()
    assert peak < peak_arrays * width * struct.calcsize("P")
