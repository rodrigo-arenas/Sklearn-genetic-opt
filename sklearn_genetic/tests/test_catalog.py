from functools import partial
import importlib
from math import ldexp
import pickle
from pathlib import Path
from statistics import mean, pstdev
import subprocess
from textwrap import dedent
from threading import Event, get_ident
import sys
import warnings

import numpy as np
import pytest
from sklearn.base import BaseEstimator, OutlierMixin, clone
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import FitFailedWarning, NotFittedError
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

from sklearn_genetic import (
    EvolutionConfig,
    GAFeatureSelectionCV,
    GASearchCV,
    OptimizationConfig,
    PopulationConfig,
    RuntimeConfig,
)
from sklearn_genetic.callbacks import (
    BaseCallback,
    CVBudgetStopping,
    DeltaThreshold,
    ModelCheckpoint,
    ThresholdStopping,
    TimerStopping,
)
from sklearn_genetic.space import Categorical, Continuous, Integer
from sklearn_genetic.tests.test_optimizer_backends import requires_scipy_backend, search
from sklearn_genetic.utils.cv_scores import subsample_cv

BACKENDS = ["nevergrad", "cmaes", "pymoo"]


class _TreeWithObjectArrayOptions(DecisionTreeClassifier):
    def __init__(self, options):
        super().__init__(random_state=51)
        self.options = options

    def fit(self, X, y, sample_weight=None, check_input=True):
        mode = self.options[0]["mode"]
        self.max_depth = 1 if type(mode) is bool else 3
        return super().fit(X, y, sample_weight=sample_weight, check_input=check_input)


@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
@pytest.mark.parametrize("nested_arrays", [False, True])
def test_cloned_estimator_choices_resolve_nested_object_array_options(engine, nested_arrays):
    from sklearn.model_selection import cross_val_score

    if engine != "deap":
        pytest.importorskip(engine)

    def choice(mode):
        options = np.empty(1, dtype=object)
        options[0] = {"mode": mode}
        if nested_arrays:
            options[0]["weights"] = np.array([0.5, np.nan])
        return _TreeWithObjectArrayOptions(options)

    alternate, wanted = choice(True), choice(1)
    X, y = load_iris(return_X_y=True)
    train, test, y_train, _ = train_test_split(X, y, test_size=0.25, random_state=51, stratify=y)
    cv = StratifiedKFold(2, shuffle=True, random_state=51)
    search = GASearchCV(
        Pipeline([("tree", alternate)]),
        param_grid={"tree": Categorical([alternate, wanted])},
        warm_start_configs=[{"tree": clone(wanted)}],
        optimizer_backend=engine,
        population_size=6,
        generations=0,
        cv=cv,
        scoring="accuracy",
        random_state=51,
        error_score="raise",
        verbose=False,
    ).fit(train, y_train, callbacks=CVBudgetStopping(1))

    assert search.fit_counts_["search_cv_calls"] == 1
    assert type(search.cv_results_["param_tree"][0].options[0]["mode"]) is int
    expected = cross_val_score(wanted, train, y_train, cv=cv, scoring="accuracy")
    actual = [search.cv_results_[f"split{fold}_test_score"][0] for fold in range(2)]
    np.testing.assert_array_equal(actual, expected)
    reference = clone(wanted).fit(train, y_train)
    np.testing.assert_array_equal(search.predict(test), reference.predict(test))


@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
def test_feature_search_can_keep_dataframe_for_fold_specific_column_selectors(engine, tmp_path):
    import pandas as pd
    from sklearn.compose import make_column_selector, make_column_transformer
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import OneHotEncoder

    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["petal width (cm)"] = pd.Categorical(X["petal width (cm)"] > 1.3)
    X.index = 3 * np.arange(len(X))
    train, test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=31, stratify=y
    )
    preprocessor = make_column_transformer(
        (
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            make_column_selector(dtype_include="category"),
        ),
        (StandardScaler(), make_column_selector(dtype_include="number")),
    )
    estimator = Pipeline(
        [("encode", preprocessor), ("tree", DecisionTreeClassifier(random_state=31))]
    )
    cv = StratifiedKFold(2, shuffle=True, random_state=31)
    finalist_cv = StratifiedKFold(3, shuffle=True, random_state=32)
    model = GAFeatureSelectionCV(
        estimator,
        cv=cv,
        scoring="accuracy",
        optimizer_backend=engine,
        population_size=6,
        generations=0,
        max_features=3,
        final_selection=True,
        final_selection_cv=finalist_cv,
        final_selection_top_k=2,
        error_score="raise",
        random_state=31,
        verbose=False,
    ).set_params(preserve_dataframe=True)
    fitted = (
        clone(model)
        .set_output(transform="pandas")
        .fit(train, y_train, callbacks=CVBudgetStopping(4))
    )

    assert fitted.fit_counts_["search_cv_calls"] == 4
    assert fitted.fit_counts_["finalist_cv_calls"] == 2
    for index, mask in enumerate(fitted.cv_results_["features"]):
        expected = cross_val_score(estimator, train.iloc[:, mask], y_train, cv=cv)
        actual = [fitted.cv_results_[f"split{fold}_test_score"][index] for fold in range(2)]
        np.testing.assert_array_equal(actual, expected)
    for row in fitted.final_selection_results_["candidates"]:
        mask = np.asarray(row["params"]["features"], dtype=bool)
        expected = cross_val_score(estimator, train.iloc[:, mask], y_train, cv=finalist_cv)
        np.testing.assert_array_equal(row["cv_scores"], expected)
    selected = fitted.get_support()
    reference = clone(estimator).fit(train.iloc[:, selected], y_train)
    np.testing.assert_array_equal(fitted.predict(test), reference.predict(test.iloc[:, selected]))
    assert fitted.score(test, y_test) == reference.score(test.iloc[:, selected], y_test)
    pd.testing.assert_frame_equal(fitted.transform(test), test.iloc[:, selected])
    with pytest.raises(TypeError, match="pandas DataFrame"):
        fitted.predict(test.to_numpy())
    with pytest.raises(ValueError, match="feature names should match"):
        fitted.predict(test.iloc[:, ::-1])
    path = tmp_path / "fitted-selector.pkl"
    fitted.save(path)
    restored = clone(model)
    restored.load(path)
    np.testing.assert_array_equal(restored.predict(test), fitted.predict(test))


@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
def test_feature_dataframe_checkpoint_retains_input_mode(engine, tmp_path):
    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True, as_frame=True)
    model = search("features", optimizer_backend=engine, use_cache=False, preserve_dataframe=True)
    checkpoint = ModelCheckpoint(tmp_path / "frame-search.pkl")
    first = clone(model).fit(X, y, callbacks=[CVBudgetStopping(6), checkpoint])
    resumed = clone(model).fit(X.copy(), y, callbacks=[CVBudgetStopping(3), checkpoint])

    assert resumed.fit_counts_["search_cv_calls"] == 3
    np.testing.assert_array_equal(
        resumed.cv_results_["mean_test_score"][:6], first.cv_results_["mean_test_score"]
    )
    np.testing.assert_array_equal(
        resumed.best_estimator_.feature_names_in_, X.columns[resumed.get_support()]
    )
    with pytest.raises(ValueError, match="Checkpoint .*data"):
        clone(model).set_params(preserve_dataframe=False).fit(X, y, callbacks=checkpoint)
    for changed in (X.astype("Float64"), X.iloc[:, ::-1]):
        with pytest.raises(ValueError, match="Checkpoint .*data"):
            clone(model).fit(changed, y, callbacks=checkpoint)


def test_feature_dataframe_mode_is_available_for_unlabeled_outliers():
    X, _ = load_iris(return_X_y=True, as_frame=True)
    model = GAFeatureSelectionCV(
        IsolationForest(n_estimators=5, random_state=26),
        cv=2,
        population_size=4,
        generations=0,
        preserve_dataframe=True,
        error_score="raise",
        random_state=26,
        verbose=False,
    ).fit(X, callbacks=CVBudgetStopping(3))

    assert model.y_ is None
    selected = X.iloc[:, model.get_support()]
    np.testing.assert_array_equal(model.best_estimator_.feature_names_in_, selected.columns)
    np.testing.assert_array_equal(
        model.score_samples(X), model.best_estimator_.score_samples(selected)
    )


@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
@pytest.mark.parametrize("initializer", ["random", "smart", "spread"])
def test_feature_warm_starts_keep_first_distinct_masks(engine, initializer):
    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    X, X_test, y, _ = train_test_split(X, y, test_size=0.25, random_state=201, stratify=y)
    seeds = np.array([[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 1]], dtype=bool)
    expected = seeds[[0, 2]].copy()
    model = GAFeatureSelectionCV(
        DecisionTreeClassifier(random_state=201),
        cv=StratifiedKFold(2, shuffle=True, random_state=201),
        population_size=6,
        max_features=2,
        optimizer_backend=engine,
        error_score="raise",
        verbose=False,
        random_state=201,
    ).set_params(population_config=PopulationConfig(initializer, warm_start_masks=seeds))

    fitted = clone(model).fit(X, y, callbacks=CVBudgetStopping(2))

    np.testing.assert_array_equal(fitted.cv_results_["features"], expected)
    np.testing.assert_array_equal(seeds, expected[[0, 0, 1]])
    assert fitted.fit_counts_["search_cv_calls"] == 2
    selected = fitted.get_support()
    reference = clone(model.estimator).fit(X[:, selected], y)
    np.testing.assert_array_equal(fitted.predict(X_test), reference.predict(X_test[:, selected]))


@pytest.mark.parametrize(
    ("seeds", "message"),
    [
        ([1, 0, 0, 0], "mask 0.*length 4"),
        ([[1, 0, 0]], "mask 0.*length 4"),
        ([[1, 2, 0, 0]], "mask 0.*Boolean or integer"),
        ([[1.0, 0.0, 0.0, 0.0]], "mask 0.*Boolean or integer"),
        ([[0, 0, 0, 0]], "mask 0.*at least one"),
        ([[1, 1, 1, 0]], "mask 0.*max_features"),
        ("1000", "list, tuple"),
    ],
)
def test_feature_warm_starts_validate_before_candidate_fit(seeds, message, monkeypatch):
    from sklearn_genetic import genetic_search

    X, y = load_iris(return_X_y=True)
    model = GAFeatureSelectionCV(
        DecisionTreeClassifier(),
        max_features=2,
        population_size=6,
        runtime_config=RuntimeConfig(verbose=False, error_score="raise"),
        population_config=PopulationConfig(warm_start_masks=seeds),
    )
    monkeypatch.setattr(
        genetic_search, "cross_validate", lambda *args, **kwargs: pytest.fail("unexpected CV")
    )
    with pytest.raises(ValueError, match=message):
        model.fit(X, y)


def test_deap_feature_warm_start_stays_first_with_timer():
    X, y = load_iris(return_X_y=True)
    seed = [0, 0, 0, 1]
    model = search(
        "features",
        population_config=PopulationConfig(warm_start_masks=[seed]),
    ).fit(X, y, callbacks=[TimerStopping(60), CVBudgetStopping(1)])
    np.testing.assert_array_equal(model.cv_results_["features"], [seed])


@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
def test_feature_warm_start_checkpoint_keeps_scores(engine, tmp_path):
    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    seed = [0, 0, 0, 1]
    model = search(
        "features",
        optimizer_backend=engine,
        population_config=PopulationConfig(warm_start_masks=[seed]),
        use_cache=False,
    )
    checkpoint = ModelCheckpoint(tmp_path / "warm-mask.pkl")
    first = clone(model).fit(X, y, callbacks=[CVBudgetStopping(6), checkpoint])
    resumed = clone(model).fit(X, y, callbacks=[CVBudgetStopping(3), checkpoint])

    assert checkpoint.load()["logbook"][0]["cross_validate_calls"] == 6
    assert resumed.history["cross_validate_calls"][-1] == 9
    assert resumed.fit_counts_["search_cv_calls"] == 3
    np.testing.assert_array_equal(first.cv_results_["features"][0], seed)
    np.testing.assert_array_equal(
        resumed.cv_results_["features"][:6], first.cv_results_["features"]
    )
    np.testing.assert_array_equal(
        resumed.cv_results_["mean_test_score"][:6], first.cv_results_["mean_test_score"]
    )


@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
@pytest.mark.parametrize("grouped", [False, True])
def test_smart_population_keeps_later_distinct_warm_starts(engine, grouped):
    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    X, X_test, y, _ = train_test_split(X, y, test_size=0.25, random_state=201, stratify=y)
    distinct = [{"max_depth": depth, "criterion": "entropy"} for depth in (1, 5, 3, 4, 6, 7)]
    model = GASearchCV(
        DecisionTreeClassifier(max_depth=2, random_state=201),
        cv=StratifiedKFold(2, shuffle=True, random_state=201),
        param_grid={"max_depth": Integer(1, 8), "criterion": Categorical(["gini", "entropy"])},
        population_size=6,
        use_cache=True,
        optimizer_backend=engine,
        error_score="raise",
        verbose=False,
        random_state=201,
    )

    def fit(warm_starts):
        search = clone(model)
        if grouped:
            search.set_params(population_config=PopulationConfig(warm_start_configs=warm_starts))
        else:
            search.set_params(warm_start_configs=warm_starts)
        return search.fit(X, y, callbacks=CVBudgetStopping(2))

    reference = fit(distinct)
    repeated = [distinct[0]] * 6 + distinct[1:] + [{"unused": 1}]
    fitted = fit(repeated)

    assert fitted.cv_results_["params"] == reference.cv_results_["params"] == distinct[:2]
    assert fitted.fit_counts_ == reference.fit_counts_
    assert fitted.fit_counts_["search_cv_calls"] == 2
    assert fitted.best_params_ == reference.best_params_
    np.testing.assert_array_equal(fitted.predict(X_test), reference.predict(X_test))


@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
@pytest.mark.parametrize("kind", ["continuous", "estimator_category"])
@pytest.mark.parametrize("use_cache", [False, True])
def test_smart_population_deduplicates_the_values_that_cv_will_evaluate(engine, kind, use_cache):
    from sklearn.pipeline import Pipeline

    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    X, X_test, y, _ = train_test_split(X, y, test_size=0.25, random_state=207, stratify=y)
    tree = DecisionTreeClassifier(max_depth=2, random_state=207)
    if kind == "continuous":
        origin = 2**56
        estimator = tree
        space = {"ccp_alpha": Continuous(origin, origin + 200), "max_depth": Integer(1, 8)}
        configs = [{"ccp_alpha": float(origin + 16 * i), "max_depth": 2} for i in range(6)]
        aliases = [{"ccp_alpha": origin + i, "max_depth": 2} for i in range(6)]
        differentiator = "ccp_alpha"
    else:
        estimator = Pipeline([("model", DecisionTreeClassifier(random_state=207))])
        space = {"model": Categorical([tree]), "model__min_samples_leaf": Integer(2, 8)}
        configs = [{"model": tree, "model__min_samples_leaf": i} for i in range(2, 8)]
        aliases = [{"model": clone(tree), "model__min_samples_leaf": 2} for _ in range(6)]
        differentiator = "model__min_samples_leaf"

    def fit(warm_starts):
        model = GASearchCV(
            estimator,
            cv=StratifiedKFold(2, shuffle=True, random_state=207),
            param_grid=space,
            warm_start_configs=warm_starts,
            population_size=6,
            generations=0,
            optimizer_backend=engine,
            use_cache=use_cache,
            error_score="raise",
            verbose=False,
            random_state=207,
        )
        return model.fit(X, y, callbacks=CVBudgetStopping(2))

    reference = fit(configs)
    fitted = fit(aliases + configs[1:])
    assert fitted.cv_results_[f"param_{differentiator}"] == [
        config[differentiator] for config in configs[:2]
    ]
    assert fitted.fit_counts_ == reference.fit_counts_
    assert fitted.fit_counts_["search_cv_calls"] == 2
    np.testing.assert_array_equal(
        fitted.cv_results_["mean_test_score"], reference.cv_results_["mean_test_score"]
    )
    np.testing.assert_array_equal(fitted.predict(X_test), reference.predict(X_test))


def test_smart_population_skips_an_estimator_default_that_matches_a_warm_start():
    from sklearn.pipeline import Pipeline

    X, y = load_iris(return_X_y=True)
    tree = DecisionTreeClassifier(max_depth=2, random_state=207)
    fitted = GASearchCV(
        Pipeline([("model", clone(tree))]),
        cv=2,
        param_grid={"model": Categorical([tree]), "model__min_samples_leaf": Integer(1, 8)},
        warm_start_configs=[{"model": clone(tree), "model__min_samples_leaf": 1}],
        population_size=6,
        generations=0,
        use_cache=False,
        verbose=False,
        random_state=207,
    ).fit(X, y)
    candidates = fitted.cv_results_["param_model__min_samples_leaf"]
    assert candidates[0] == 1
    assert len(set(candidates)) == len(candidates) == 6


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
@pytest.mark.parametrize(
    "parameter, value",
    [
        ("population_size", 0),
        ("population_size", "6"),
        ("keep_top_k", 0),
        ("keep_top_k", True),
        ("generations", -1),
        ("generations", 1.5),
        ("generations", False),
    ],
)
def test_invalid_search_counts_are_rejected_before_cv(task, engine, parameter, value, monkeypatch):
    from sklearn_genetic import genetic_search

    if engine != "deap":
        pytest.importorskip(engine)

    def unexpected_cv(*args, **kwargs):
        pytest.fail("invalid search controls reached cross-validation")

    monkeypatch.setattr(genetic_search, "cross_validate", unexpected_cv)
    model = search(task, optimizer_backend=engine).set_params(**{parameter: value})
    with pytest.raises(ValueError, match=parameter):
        model.fit(*load_iris(return_X_y=True))


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
@pytest.mark.parametrize("initializer", ["smart", "random"])
def test_numpy_unsigned_search_counts_match_python_integers(task, engine, initializer):
    if engine != "deap":
        pytest.importorskip(engine)

    X, y = load_iris(return_X_y=True)
    counts = {"population_size": 6, "generations": 1, "keep_top_k": 2}
    if task == "features":
        counts["max_features"] = 2
    model = search(task, optimizer_backend=engine, population_initializer=initializer)
    reference = clone(model).set_params(**counts).fit(X, y)
    unsigned = {name: np.uint64(value) for name, value in counts.items()}
    fitted = clone(model).set_params(**unsigned).fit(X, y)

    assert fitted.fit_counts_ == reference.fit_counts_
    assert fitted.history == reference.history
    assert fitted.best_score_ == reference.best_score_
    values = "params" if task == "parameters" else "features"
    np.testing.assert_equal(fitted.cv_results_[values], reference.cv_results_[values])
    np.testing.assert_array_equal(fitted.predict(X), reference.predict(X))
    for name, value in unsigned.items():
        assert fitted.get_params()[name] is value


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize("saved_generations", [0, -1])
def test_deap_validates_the_generation_count_restored_from_checkpoint(
    task, saved_generations, tmp_path, monkeypatch
):
    from sklearn_genetic import genetic_search

    checkpoint = ModelCheckpoint(tmp_path / "counts.pkl")
    checkpoint.checkpoint_path.write_bytes(
        pickle.dumps({"estimator_state": {"generations": saved_generations}, "logbook": None})
    )
    X, y = load_iris(return_X_y=True)
    model = search(task).set_params(generations="value replaced by checkpoint")

    if saved_generations == 0:
        model.fit(X, y, callbacks=checkpoint)
        assert model.history["gen"] == [0]
        assert model.fit_counts_["search_cv_calls"] > 0
    else:

        def unexpected_cv(*args, **kwargs):
            pytest.fail("invalid restored generations reached cross-validation")

        monkeypatch.setattr(genetic_search, "cross_validate", unexpected_cv)
        with pytest.raises(ValueError, match="generations"):
            model.fit(X, y, callbacks=checkpoint)


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
@pytest.mark.parametrize("name", ["consecutive", "delta"])
def test_unsigned_callback_windows_stop_at_the_same_generation(task, engine, name):
    from sklearn_genetic.backends import scipy_optimizer
    from sklearn_genetic.callbacks import ConsecutiveStopping, DeltaThreshold

    if engine != "deap":
        pytest.importorskip(engine)

    generation = 0

    def score(estimator, X, y):
        return float(min(generation, 2))

    class Advance(BaseCallback):
        def on_step(self, record, logbook, estimator):
            nonlocal generation
            generation = logbook[-1]["gen"] + 1

    X, y = load_iris(return_X_y=True)
    optimizer = scipy_optimizer(stop_on_convergence=False) if engine == "scipy" else engine
    model = search(task, optimizer_backend=optimizer, generations=6, scoring=score, use_cache=False)

    def run(window):
        nonlocal generation
        generation = 0
        callback = (
            ConsecutiveStopping(window, metric="fitness_best")
            if name == "consecutive"
            else DeltaThreshold(0.0, generations=window, metric="fitness_best")
        )
        return clone(model).fit(X, y, callbacks=[callback, Advance()])

    with np.errstate(over="raise", invalid="raise"):
        reference = run(2)
        fitted = run(np.uint64(2))

    assert fitted.history == reference.history
    assert fitted.fit_counts_ == reference.fit_counts_
    assert fitted.best_score_ == reference.best_score_
    values = "params" if task == "parameters" else "features"
    np.testing.assert_equal(fitted.cv_results_[values], reference.cv_results_[values])
    np.testing.assert_array_equal(fitted.predict(X), reference.predict(X))


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
@pytest.mark.parametrize("metric", ["fitness_best", "fitness_best_observed"])
def test_minimizing_search_reaches_threshold_after_unscored_generation(task, engine, metric):
    if engine != "deap":
        pytest.importorskip(engine)
    loss_by_generation = (np.nan, 2.0, 0.5, 0.25)
    generation = calls = 0

    class AdvanceLoss(BaseCallback):
        def on_step(self, record, logbook, estimator):
            nonlocal generation
            generation += 1

    def loss(estimator, X, y):
        nonlocal calls
        calls += 1
        return loss_by_generation[generation] + calls * 1e-6

    model = search(
        task,
        optimizer_backend=engine,
        criteria="min",
        use_cache=False,
        generations=3,
        scoring=loss,
        error_score="raise",
    ).fit(
        *load_iris(return_X_y=True),
        callbacks=[ThresholdStopping(1.0, metric=metric), AdvanceLoss()],
    )
    assert model.history["gen"] == [0, 1, 2]
    assert model.history["finite_fitness_count"][0] == 0
    assert model.history["fitness_best"][1] > 1.0
    assert model.best_score_ <= 1.0


@pytest.mark.parametrize("task", ["parameters", "features"])
def test_observed_threshold_counts_better_repeat_even_when_archive_keeps_first_score(task):
    X, y = load_iris(return_X_y=True)
    if task == "features":
        X = X[:, :2]
        rows = [[1, 0], [0, 1]]
        model = GAFeatureSelectionCV(DecisionTreeClassifier(random_state=7), max_features=1)
    else:
        rows = [[2], [3]]
        model = GASearchCV(
            DecisionTreeClassifier(random_state=7), param_grid={"max_depth": Integer(2, 3)}
        )

    def optimize(problem, evaluate, report):
        _, first = evaluate([rows[0]])[0]
        report([first])
        evaluate([rows[0]])
        report([first])
        _, last = evaluate([rows[1]])[0]
        report([last])

    model.set_params(
        optimizer_backend=optimize,
        population_size=2,
        generations=2,
        cv=2,
        n_jobs=1,
        use_cache=False,
        random_state=7,
        verbose=False,
    )

    def run(metric):
        scores = iter(value for value in (0.4, 0.95, 0.97) for _ in range(2))

        def scoring(estimator, X, y):
            return next(scores)

        return (
            clone(model)
            .set_params(scoring=scoring)
            .fit(X, y, callbacks=ThresholdStopping(0.9, metric=metric))
        )

    observed = run("fitness_best_observed")
    archived = run("fitness_best")
    assert observed.fit_counts_["search_cv_calls"] == 2
    assert archived.fit_counts_["search_cv_calls"] == 3
    assert observed.history["fitness_best_observed"] == [0.4, 0.95]
    assert observed.history["fitness_best"] == [0.4, 0.4]
    np.testing.assert_array_equal(observed.cv_results_["mean_test_score"], [0.4, 0.95])


class _RejectInvalidCandidate(DecisionTreeClassifier):
    minimum_training_rows = 0

    def fit(self, X, y, sample_weight=None, check_input=True):
        if X.shape[0] >= self.minimum_training_rows and (self.max_depth == 1 or X.shape[1] == 1):
            raise ValueError("fixture estimator rejects this candidate")
        return super().fit(X, y, sample_weight=sample_weight, check_input=check_input)


class _RejectSomeFinalists(_RejectInvalidCandidate):
    minimum_training_rows = 50


class _TreeWithDiagnosticParameterNames(DecisionTreeClassifier):
    def __init__(self, fit_time=2, test_accuracy=1, index="left", _candidate_params="original"):
        super().__init__(random_state=19)
        self.fit_time = fit_time
        self.test_accuracy = test_accuracy
        self.index = index
        self._candidate_params = _candidate_params

    def fit(self, X, y, sample_weight=None, check_input=True):
        assert type(self.fit_time) is int and type(self.test_accuracy) is int
        self.max_depth = self.fit_time
        self.min_samples_leaf = self.test_accuracy
        return super().fit(X, y, sample_weight=sample_weight, check_input=check_input)


@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
def test_candidate_parameter_names_are_independent_of_diagnostics(engine, tmp_path, monkeypatch):
    from sklearn_genetic import genetic_search

    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    search = GASearchCV(
        _TreeWithDiagnosticParameterNames(),
        cv=2,
        refit="accuracy",
        scoring={"accuracy": "accuracy"},
        param_grid={
            "fit_time": Integer(1, 3),
            "test_accuracy": Integer(1, 4),
            "index": Categorical(["left", "right"]),
            "_candidate_params": Categorical(["original", "other"]),
        },
        generations=1,
        population_size=6,
        use_cache=True,
        optimizer_backend=engine,
        final_selection=True,
        final_selection_top_k=2,
        random_state=19,
        n_jobs=1,
        verbose=False,
        error_score="raise",
    )
    checkpoint = ModelCheckpoint(tmp_path / "parameters.pkl")
    first = clone(search).fit(X, y, callbacks=[CVBudgetStopping(6), checkpoint])
    resumed = clone(search).fit(X, y, callbacks=[CVBudgetStopping(6), checkpoint])
    result = resumed.cv_results_
    assert result["params"][:6] == first.cv_results_["params"]
    assert resumed.fit_counts_["search_cv_calls"] > 0
    assert resumed.fit_counts_["finalist_cv_calls"] == 2
    assert len(result["mean_fit_time"]) == len(result["params"])
    for name in search.param_grid:
        assert result[f"param_{name}"] == [params[name] for params in result["params"]]
        assert getattr(resumed.best_estimator_, name) == resumed.best_params_[name]
    assert set(result["param_fit_time"]) <= {1, 2, 3}
    assert set(result["param_test_accuracy"]) <= {1, 2, 3, 4}
    assert set(result["param_index"]) <= {"left", "right"}
    assert set(result["param__candidate_params"]) <= {"original", "other"}
    records = resumed.logbook.chapters["parameters"]
    assert all(isinstance(row["index"], int) for row in records)
    assert all(len(row["test_accuracy"]) == 2 for row in records)

    snapshot = checkpoint.load()
    stored = (
        snapshot["runtime_state"]["candidate_logbook"].chapters["parameters"]
        if engine == "deap"
        else snapshot["candidate_records"]
    )
    for row in stored:
        row.pop("_candidate_params")
    checkpoint.checkpoint_path.write_bytes(pickle.dumps(snapshot))

    def unexpected_cv(*args, **kwargs):
        pytest.fail("unrecoverable checkpoint reached cross-validation")

    monkeypatch.setattr(genetic_search, "cross_validate", unexpected_cv)
    with pytest.raises(ValueError, match="Checkpoint did not preserve candidate parameters"):
        clone(search).fit(X, y, callbacks=[checkpoint])


def _cv_failure_search(task, engine, *, all_fail=False, error_score=np.nan):
    options = dict(
        estimator=_RejectInvalidCandidate(max_depth=2, random_state=19),
        cv=2,
        scoring={"accuracy": "accuracy", "f1": "f1_macro"},
        refit="accuracy",
        return_train_score=True,
        optimizer_backend=engine,
        population_size=6,
        generations=0,
        use_cache=False,
        n_jobs=1,
        verbose=False,
        random_state=19,
        error_score=error_score,
    )
    if task == "features":
        return GAFeatureSelectionCV(max_features=1 if all_fail else 2, **options)
    return GASearchCV(param_grid={"max_depth": Integer(1, 1 if all_fail else 2)}, **options)


def _undefined_score(estimator, X, y):
    return np.nan


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
def test_full_candidate_cv_failure_leaves_a_valid_winner(task, engine):
    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    model = _cv_failure_search(task, engine, error_score=100.0)
    with pytest.warns(FitFailedWarning, match="fixture estimator rejects this candidate"):
        model.fit(X, y)
    results = model.cv_results_
    failed = np.isnan(results["mean_test_accuracy"])
    assert failed.any() and (~failed).any()
    assert results["all_fits_failed"].dtype == bool
    np.testing.assert_array_equal(results["all_fits_failed"], failed)
    report = model.cv_results_dataframe(columns=["all_fits_failed"], preserve_index=True)
    np.testing.assert_array_equal(report["all_fits_failed"], failed[report.index])
    finite = np.asarray(results["mean_test_accuracy"])[~failed]
    assert model.history["finite_fitness_count"] == [len(finite)]
    assert model.history["fitness"] == pytest.approx([np.mean(finite)])
    assert model.history["fitness_std"] == pytest.approx([np.std(finite)])
    assert model.history["fitness_min"] == [min(finite)]
    assert model.history["fitness_max"] == [max(finite)]
    for name in ("split0_test_accuracy", "split1_train_f1", "mean_fit_time"):
        assert np.isnan(np.asarray(results[name])[failed]).all()
    assert np.all(results["rank_test_accuracy"][failed] > 1)
    assert not failed[model.best_index_]
    assert model.best_score_ == results["mean_test_accuracy"][model.best_index_]
    assert model.fit_counts_["search_cv_calls"] == 6
    assert model.fit_counts_["search_fold_fits"] == 12
    assert model.fit_counts_["refit_fits"] == 1
    assert model.predict(X).shape == y.shape
    if task == "parameters":
        assert model.best_estimator_.max_depth == 2
    else:
        np.testing.assert_array_equal(model.get_support(), results["features"][model.best_index_])
        assert model.best_estimator_.n_features_in_ == model.n_features_ == 2


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
@pytest.mark.parametrize("scale", [1e-170, 1e155])
def test_finite_score_dispersion_controls_stopping_at_extreme_scales(task, engine, scale):
    if engine != "deap":
        pytest.importorskip(engine)

    def scorer(estimator, X, y):
        size = estimator.max_depth if task == "parameters" else estimator.n_features_in_
        return scale * size

    model = search(task, optimizer_backend=engine).set_params(
        scoring=scorer, population_size=6, generations=1, use_cache=False
    )
    with np.errstate(over="raise", invalid="raise"):
        model.fit(
            *load_iris(return_X_y=True),
            callbacks=ThresholdStopping(0.05 * scale, metric="fitness_std"),
        )
    scores = model.cv_results_["mean_test_score"]
    expected = pstdev(map(float, scores))
    assert expected > 0
    assert model.history["fitness_std"][0] == pytest.approx(expected, rel=1e-12, abs=0)
    assert model.history["fitness_best_observed"] == [max(scores)]
    assert model.best_score_ == max(scores)
    assert model.fit_counts_["search_cv_calls"] == 6


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
@pytest.mark.parametrize("scale", [1e-170, 1e156, 1e308])
@pytest.mark.parametrize("criteria", ["min", "max"])
def test_extreme_fold_scores_remain_usable_through_final_selection(task, engine, scale, criteria):
    if engine != "deap":
        pytest.importorskip(engine)

    def scorer(estimator, X, y):
        size = estimator.max_depth if task == "parameters" else estimator.n_features_in_
        return (1.0 + 0.1 * size + 0.1 * (y[0] > 0)) * scale

    model = search(task, optimizer_backend=engine).set_params(
        cv=KFold(2),
        scoring={"extreme": scorer, "accuracy": "accuracy"},
        refit="extreme",
        criteria=criteria,
        generations=0,
        use_cache=False,
        return_train_score=True,
        final_selection=True,
        final_selection_top_k=2,
    )
    with np.errstate(over="raise", under="raise", invalid="raise"):
        model.fit(*load_iris(return_X_y=True))

    results = model.cv_results_
    for subset in ("test", "train"):
        for i, folds in enumerate(
            zip(
                results[f"split0_{subset}_extreme"],
                results[f"split1_{subset}_extreme"],
                strict=True,
            )
        ):
            scores = list(map(float, folds))
            assert results[f"mean_{subset}_extreme"][i] == pytest.approx(
                mean(scores), rel=1e-14, abs=0
            )
            assert results[f"std_{subset}_extreme"][i] == pytest.approx(
                pstdev(scores), rel=1e-14, abs=0
            )
    preferred = min if criteria == "min" else max
    winner_score = preferred(results["mean_test_extreme"])
    assert model.history["fitness_best_observed"] == [winner_score]
    assert model.best_score_ == winner_score
    assert results["rank_test_extreme"][model.best_index_] == 1
    assert model.fit_counts_["search_cv_calls"] == 6
    assert model.fit_counts_["finalist_cv_calls"] == 2


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
@pytest.mark.parametrize("criteria", ["min", "max"])
def test_first_population_ranks_extreme_scores_of_both_signs(task, engine, criteria):
    if engine != "deap":
        pytest.importorskip(engine)

    def scorer(estimator, X, y):
        size = estimator.max_depth if task == "parameters" else estimator.n_features_in_
        return 1e308 if size > 1 else -1e308

    model = search(task, optimizer_backend=engine).set_params(
        scoring=scorer, criteria=criteria, generations=0, use_cache=False
    )
    with np.errstate(over="raise", invalid="raise"):
        model.fit(*load_iris(return_X_y=True))
    winner = -1e308 if criteria == "min" else 1e308
    assert set(model.cv_results_["mean_test_score"]) == {-1e308, 1e308}
    assert model.history["fitness_best_observed"] == [winner]
    assert model.best_score_ == winner
    assert model.fit_counts_["search_cv_calls"] == 6


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize("criteria", ["min", "max"])
@pytest.mark.parametrize(
    "mode", ["fresh", "fresh_range", "reuse", "partial", "refit", "report", "callable"]
)
def test_search_and_finalist_results_keep_extended_score_precision(task, criteria, mode):
    if np.finfo(np.longdouble).nmant <= np.finfo(float).nmant:
        pytest.skip("requires extended precision")
    magnitude = np.longdouble(1)
    if mode == "fresh_range":
        if np.finfo(np.longdouble).maxexp <= 1026:
            pytest.skip("requires extended exponent range")
        magnitude = np.ldexp(magnitude, 1025)
    toward = magnitude * 2
    lower = np.nextafter(magnitude, toward)
    upper = np.nextafter(lower, toward)
    initial, winner = (lower, upper) if criteria == "max" else (upper, lower)
    assert float(initial) == float(winner)
    fresh = mode in {"fresh", "fresh_range"}

    X, y = load_iris(return_X_y=True)
    candidates = [(1,), (2,)] if task == "parameters" else [(1, 0, 0, 0), (1, 1, 0, 0)]
    options = {"param_grid": {"max_depth": Integer(1, 2)}} if task == "parameters" else {}

    def optimize(problem, evaluate, report):
        report([handle for _, handle in evaluate(candidates)])

    def score(estimator, X, y):
        if fresh and len(X) == 75:
            return 1.0
        size = estimator.max_depth if task == "parameters" else estimator.n_features_in_
        return winner if size == 2 else initial

    model = search(
        task,
        **options,
        optimizer_backend=optimize,
        population_size=2,
        generations=0,
        scoring=score,
        criteria=criteria,
        error_score="raise",
    )
    if fresh:
        model.set_params(cv=KFold(2), final_selection=True, final_selection_cv=KFold(3))
    elif mode == "reuse":
        model.set_params(final_selection="reuse_folds")
    elif mode == "partial":
        folds = list(KFold(3).split(X, y))
        model.set_params(
            cv=folds[:2],
            final_selection="reuse_folds",
            final_selection_cv=[folds[0], folds[2]],
        )
    elif mode == "report":
        model.set_params(refit=False)
    elif mode == "callable":
        model.set_params(refit=lambda _: 1)

    model.fit(X, y)
    assert model.best_index_ == 1
    if not fresh:
        np.testing.assert_array_equal(model.cv_results_["rank_test_score"], [2, 1])
    if mode != "callable":
        assert type(model.best_score_) is np.longdouble
        assert model.best_score_ == winner
    if (
        fresh
        or mode in {"reuse", "partial", "callable"}
        or (mode == "refit" and task == "parameters")
    ):
        selected = model.final_selection_results_
        assert selected["selected_score"] == winner
        if fresh:
            assert type(selected["original_best_score"]) is float
        if mode in ("reuse", "partial"):
            reused = 4 if mode == "reuse" else 2
            assert model.fit_counts_["finalist_reused_folds"] == reused
            assert (
                next(row["score"] for row in selected["candidates"] if row["index"] == 1) == winner
            )


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize("criteria", ["min", "max"])
def test_fitness_sharing_handles_extreme_scores_in_public_search(task, criteria):
    def scorer(estimator, X, y):
        size = estimator.max_depth if task == "parameters" else estimator.n_features_in_
        return 1e308 if size > 1 else -1e308

    model = search(task).set_params(
        scoring=scorer,
        criteria=criteria,
        generations=2,
        use_cache=False,
        fitness_sharing=True,
        algorithm="eaSimple",
        error_score="raise",
    )
    with np.errstate(over="raise", invalid="raise"):
        model.fit(*load_iris(return_X_y=True))
    assert model.history["fitness_sharing_applied"] == [False, True, True]
    assert model.best_score_ == (-1e308 if criteria == "min" else 1e308)
    assert set(model.cv_results_["mean_test_score"]) == {-1e308, 1e308}


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize("engine", ["deap", "custom"])
@pytest.mark.parametrize("criteria", ["min", "max"])
@pytest.mark.parametrize("extended", [False, True], ids=["double", "extended"])
def test_extreme_score_improvements_keep_generation_and_stop_history(
    task, engine, criteria, extended
):
    if extended and np.finfo(np.longdouble).maxexp <= 1027:
        pytest.skip("requires extended exponent range")
    size = np.ldexp(np.longdouble(1), 1026) if extended else 1e308
    winner = -size if criteria == "min" else size
    improvement = 2 * size if extended else np.inf
    generation = 0

    class Advance(BaseCallback):
        def on_step(self, record, logbook, estimator):
            nonlocal generation
            generation += 1

    def score(estimator, X, y):
        return -winner if generation == 0 else winner

    def optimize(problem, evaluate, report):
        if task == "features":
            rows = [[int(column == i) for column in range(4)] for i in range(4)]
            rows.append([1, 1, 0, 0])
        else:
            rows = [[depth, "gini"] for depth in range(1, 6)]
        for row in rows:
            report([evaluate([row])[0][1]])

    X, y = load_iris(return_X_y=True)
    options = dict(
        scoring=score,
        cv=2,
        criteria=criteria,
        generations=4,
        population_size=6,
        use_cache=False,
        random_state=33,
        verbose=False,
        n_jobs=1,
        optimizer_backend=optimize if engine == "custom" else engine,
    )
    if engine == "deap":
        options.update(crossover_probability=0.0, mutation_probability=1.0)
    estimator = DecisionTreeClassifier(random_state=33)
    if task == "features":
        model = GAFeatureSelectionCV(estimator, max_features=2, **options)
    else:
        model = GASearchCV(
            estimator,
            param_grid={"max_depth": Integer(1, 5), "criterion": Categorical(["gini"])},
            **options,
        )

    with np.errstate(over="raise", invalid="raise"):
        model.fit(X, y, callbacks=[Advance(), DeltaThreshold(0.0, metric="fitness_best_observed")])

    assert model.history["gen"] == [0, 1, 2]
    assert model.history["fitness_best_observed"] == [-winner, winner, winner]
    assert model.history["fitness_improvement"] == [0.0, improvement, 0.0]
    assert model.history["fitness_improved"] == [True, True, False]
    assert model.best_score_ == winner


@requires_scipy_backend
@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize(
    "profile, stop_on_convergence, reports",
    [("signed", False, 4), ("plateau", True, 2), ("plateau", False, 4), ("tiny", True, 4)],
)
def test_scipy_continues_or_converges_with_extreme_losses(
    task, profile, stop_on_convergence, reports
):
    from sklearn_genetic.backends import scipy_optimizer

    calls = 0

    def scorer(estimator, X, y):
        nonlocal calls
        assert np.geterr()["over"] == np.geterr()["under"] == "raise"
        calls += 1
        if profile == "plateau":
            return 1e308
        if profile == "tiny":
            return calls * 1e-170
        size = estimator.max_depth if task == "parameters" else estimator.n_features_in_
        return 1e308 if size > 1 else -1e308

    class CheckPolicy(BaseCallback):
        def on_step(self, record, logbook, estimator):
            assert np.geterr()["over"] == np.geterr()["under"] == "raise"

    model = search(
        task,
        optimizer_backend=scipy_optimizer(stop_on_convergence=stop_on_convergence),
        generations=3,
        scoring=scorer,
        criteria="min",
        use_cache=False,
    )
    X, y = load_iris(return_X_y=True)
    with np.errstate(over="raise", under="raise", invalid="raise"):
        model.fit(X, y, callbacks=CheckPolicy())
        assert np.geterr()["over"] == np.geterr()["under"] == "raise"
    assert model.history["gen"] == list(range(reports))
    assert model.fit_counts_["search_cv_calls"] == 6 * reports
    assert model.best_score_ == min(model.cv_results_["mean_test_score"])


@requires_scipy_backend
@pytest.mark.parametrize("task", ["parameters", "features"])
def test_scipy_invalid_candidate_losses_do_not_overflow_solver(task):
    X, y = load_iris(return_X_y=True)
    model = _cv_failure_search(task, "scipy").set_params(
        generations=2, random_state=1 if task == "features" else 19
    )
    with np.errstate(over="raise", invalid="raise"), pytest.warns(FitFailedWarning):
        model.fit(X, y)
    results = model.cv_results_
    assert np.isnan(results["mean_fit_time"]).any()
    assert np.isfinite(model.best_score_)
    assert len(model.predict(X)) == len(y)


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize("strategy", ["OnePlusOne", "TwoPointsDE"])
def test_nevergrad_orders_large_losses_ahead_of_failed_candidates(task, strategy, monkeypatch):
    pytest.importorskip("nevergrad")
    from nevergrad.common.errors import LossTooLargeWarning
    from nevergrad.optimization.base import Optimizer
    import warnings

    from sklearn_genetic import genetic_search
    from sklearn_genetic._nevergrad_backend import optimize

    calls = 0
    fail_candidate = False
    received = []
    original_tell = Optimizer.tell
    original_cv = genetic_search.cross_validate

    class FailSelectedCandidates(DecisionTreeClassifier):
        def fit(self, X, y, sample_weight=None, check_input=True):
            if fail_candidate:
                raise ValueError("fixture rejects this candidate")
            return super().fit(X, y, sample_weight=sample_weight, check_input=check_input)

    def tracked_cv(*args, **kwargs):
        nonlocal calls, fail_candidate
        calls += 1
        fail_candidate = calls > 6 and calls % 3 == 0
        try:
            return original_cv(*args, **kwargs)
        finally:
            fail_candidate = False

    def track_loss(solver, trial, loss, *args, **kwargs):
        received.append(float(loss))
        return original_tell(solver, trial, loss, *args, **kwargs)

    monkeypatch.setattr(Optimizer, "tell", track_loss)
    monkeypatch.setattr(genetic_search, "cross_validate", tracked_cv)

    def score(model, X, y):
        return float(1e22 * (model.ccp_alpha + np.mean(X)))

    model = search(
        task,
        criteria="min",
        generations=2,
        scoring=score,
        use_cache=False,
        error_score=np.nan,
        optimizer_backend=partial(optimize, strategy=strategy),
    ).set_params(estimator=FailSelectedCandidates(random_state=19))
    X, y = load_iris(return_X_y=True)
    with warnings.catch_warnings(), pytest.warns(FitFailedWarning):
        warnings.simplefilter("error", LossTooLargeWarning)
        model.fit(X, y)

    scores = np.asarray(model.cv_results_["mean_test_score"])
    assert model.fit_counts_["search_cv_calls"] == model.population_size * 3
    if task == "features":
        assert all(1 <= sum(mask) <= model.max_features for mask in model.cv_results_["features"])
    assert model.best_score_ == np.nanmin(scores)
    evolved = scores[model.population_size :]
    failed = np.isnan(evolved)
    assert failed.any() and (~failed).any()
    assert len(received) == len(evolved)
    assert np.isfinite(received).all()
    losses = np.asarray(received)
    assert np.min(losses[failed]) > np.max(losses[~failed])
    finite_scores = evolved[~failed]
    assert np.all(np.diff(losses[~failed][np.argsort(finite_scores)]) >= 0)
    assert np.unique(losses[~failed]).size == np.unique(finite_scores).size
    assert np.min(finite_scores) > 5e20
    assert len(model.predict(X)) == len(y)


@pytest.mark.parametrize("failed_generation", [1, 2])
def test_cmaes_continues_after_a_batch_of_failed_candidates(failed_generation):
    pytest.importorskip("cmaes")
    current_generation = 0

    class AdvanceGeneration(BaseCallback):
        def on_step(self, record, logbook, estimator):
            nonlocal current_generation
            current_generation = logbook[-1]["gen"] + 1

    class FailInOneGeneration(DecisionTreeClassifier):
        def fit(self, X, y, sample_weight=None, check_input=True):
            if current_generation == failed_generation:
                raise ValueError("fixture failure in a single generation")
            return super().fit(X, y, sample_weight=sample_weight, check_input=check_input)

    def score(model, X, y):
        return model.ccp_alpha

    X, y = load_iris(return_X_y=True)
    model = search(
        "parameters",
        optimizer_backend="cmaes",
        param_grid={"ccp_alpha": Continuous(0.0, 0.25), "max_depth": Integer(2, 6)},
        scoring=score,
        error_score=np.nan,
        generations=3,
        use_cache=False,
    ).set_params(estimator=FailInOneGeneration(random_state=19))
    with (
        np.errstate(invalid="raise", over="raise"),
        pytest.warns(FitFailedWarning, match="fixture failure in a single generation"),
    ):
        model.fit(X, y, callbacks=AdvanceGeneration())

    calls = model.population_size * 4
    assert model.fit_counts_["search_cv_calls"] == calls
    expected = [model.population_size] * 4
    expected[failed_generation] = 0
    assert model.history["finite_fitness_count"] == expected
    assert np.isnan(model.cv_results_["mean_test_score"]).sum() == model.population_size
    assert np.isfinite(model.best_score_)
    assert len(model.predict(X)) == len(y)


def _scaled_numeric_score(estimator, X, y, *, exponent, integer_only=False):
    unit = (estimator.max_depth - 4) / 2 if integer_only else 2 * estimator.ccp_alpha - 1
    return np.ldexp(np.longdouble(unit), exponent) if exponent >= 1024 else ldexp(unit, exponent)


@pytest.mark.parametrize("criteria", ["min", "max"])
@pytest.mark.parametrize("integer_only", [False, True], ids=["continuous", "integer"])
@pytest.mark.parametrize("exponent", [1023, 2000], ids=["double", "extended"])
def test_cmaes_learns_from_extreme_numeric_losses_at_equal_cv_budget(
    criteria, integer_only, exponent
):
    pytest.importorskip("cmaes")
    if np.finfo(np.longdouble).maxexp <= exponent:
        pytest.skip("requires extended exponent range")
    X, y = load_iris(return_X_y=True)
    space = {"max_depth": Integer(2, 6)}
    if not integer_only:
        space = {"ccp_alpha": Continuous(0.0, 1.0), **space}

    def fit(exponent):
        return search(
            "parameters",
            optimizer_backend="cmaes",
            param_grid=space,
            scoring=partial(_scaled_numeric_score, exponent=exponent, integer_only=integer_only),
            criteria=criteria,
            generations=4,
            use_cache=False,
        ).fit(X, y, callbacks=CVBudgetStopping(30))

    with np.errstate(over="raise", invalid="raise"):
        reference = fit(0)
        extreme = fit(exponent)

    assert reference.fit_counts_ == extreme.fit_counts_
    assert extreme.fit_counts_["search_cv_calls"] == 30
    assert extreme.cv_results_["params"] == reference.cv_results_["params"]
    np.testing.assert_array_equal(
        extreme.cv_results_["rank_test_score"], reference.cv_results_["rank_test_score"]
    )
    assert extreme.best_score_ == np.ldexp(np.longdouble(reference.best_score_), exponent)
    np.testing.assert_array_equal(extreme.predict(X), reference.predict(X))


@pytest.mark.parametrize("exponent", [1023, 2000], ids=["double", "extended"])
def test_cmaes_rescales_earlier_penalties_when_extreme_losses_appear_late(monkeypatch, exponent):
    cmaes = pytest.importorskip("cmaes")
    if np.finfo(np.longdouble).maxexp <= exponent:
        pytest.skip("requires extended exponent range")
    tell = cmaes.CatCMAwM.tell
    traces = []
    generation = 0

    class Advance(BaseCallback):
        def on_step(self, record, logbook, estimator):
            nonlocal generation
            generation = logbook[-1]["gen"] + 1

    def scorer(estimator, X, y):
        return _scaled_numeric_score(estimator, X, y, exponent=exponent if generation == 2 else 0)

    def tracked(solver, solutions):
        before = solver._iqhist_values.copy()
        losses = [loss for _, loss in solutions]
        tell(solver, solutions)
        traces.append((before, losses, solver._iqhist_values.copy()))

    monkeypatch.setattr(cmaes.CatCMAwM, "tell", tracked)
    model = search(
        "parameters",
        optimizer_backend="cmaes",
        param_grid={"ccp_alpha": Continuous(0.0, 1.0), "max_depth": Integer(2, 6)},
        scoring=scorer,
        criteria="min",
        generations=3,
        use_cache=False,
    )
    with np.errstate(over="raise", invalid="raise"):
        model.fit(*load_iris(return_X_y=True), callbacks=Advance())

    assert len(traces) == 3
    previous = traces[0][2]
    assert len(previous) == 1 and previous[0] > 0
    rebased, solver_losses, _ = traces[1]
    raw_losses = model.cv_results_["mean_test_score"][12:18]
    applied_exponent = int(
        np.frexp(max(map(abs, raw_losses)))[1] - np.frexp(max(map(abs, solver_losses)))[1]
    )
    assert applied_exponent > 0
    np.testing.assert_array_equal(solver_losses, np.ldexp(raw_losses, -applied_exponent))
    if exponent >= 1024:
        previous = np.asarray(previous, dtype=np.longdouble)
        last_losses = np.asarray(model.cv_results_["mean_test_score"][18:24], dtype=np.longdouble)
        assert np.any(np.asarray(traces[2][1]) != 0)
        np.testing.assert_array_equal(traces[2][1], np.ldexp(last_losses, -applied_exponent))
    np.testing.assert_array_equal(rebased, np.ldexp(previous, -applied_exponent))


@pytest.mark.parametrize("exponent", [1023, 2000], ids=["double", "extended"])
def test_cmaes_resumes_a_checkpoint_with_scaled_continuous_losses(tmp_path, exponent):
    pytest.importorskip("cmaes")
    if np.finfo(np.longdouble).maxexp <= exponent:
        pytest.skip("requires extended exponent range")
    X, y = load_iris(return_X_y=True)
    model = search(
        "parameters",
        optimizer_backend="cmaes",
        param_grid={"ccp_alpha": Continuous(0.0, 1.0), "max_depth": Integer(2, 6)},
        scoring=partial(_scaled_numeric_score, exponent=exponent),
        generations=4,
        use_cache=False,
    )
    checkpoint = ModelCheckpoint(tmp_path / "scaled.pkl")
    with np.errstate(over="raise", invalid="raise"):
        reference = clone(model).fit(X, y, callbacks=CVBudgetStopping(30))
        clone(model).fit(X, y, callbacks=[CVBudgetStopping(12), checkpoint])
        resumed = clone(model).fit(X, y, callbacks=[CVBudgetStopping(18), checkpoint])

    assert resumed.fit_counts_["search_cv_calls"] == 18
    assert resumed.history == reference.history
    assert resumed.cv_results_["params"] == reference.cv_results_["params"]
    assert resumed.best_score_ == reference.best_score_
    np.testing.assert_array_equal(resumed.predict(X), reference.predict(X))


@pytest.mark.parametrize("task", ["parameters", "features"])
def test_full_candidate_cv_failure_ranks_below_fitted_candidates_with_undefined_scores(task):
    model = _cv_failure_search(task, "deap").set_params(scoring={"accuracy": _undefined_score})
    with pytest.warns(FitFailedWarning):
        model.fit(*load_iris(return_X_y=True))
    assert np.isnan(model.best_score_)
    assert model.history["finite_fitness_count"] == [0]
    for field in ("fitness", "fitness_std", "fitness_min", "fitness_max"):
        assert np.isnan(model.history[field]).all()
    results = model.cv_results_
    failed = np.isnan(results["mean_fit_time"])
    assert failed.any() and (~failed).any()
    assert np.isnan(results["mean_test_accuracy"]).all()
    np.testing.assert_array_equal(results["all_fits_failed"], failed)
    assert not failed[model.best_index_]
    assert results["rank_test_accuracy"][failed].min() > 1


@pytest.mark.parametrize("task", ["parameters", "features"])
def test_partial_candidate_cv_failure_has_no_full_failure_flag(task):
    X, y = load_iris(return_X_y=True)
    rows = np.arange(len(y))
    cv = [(rows[40:], rows[:40]), (rows[:40], rows[40:])]
    model = _cv_failure_search(task, "deap", error_score=-1.0).set_params(
        estimator=_RejectSomeFinalists(max_depth=2, random_state=19), cv=cv
    )
    with pytest.warns(FitFailedWarning, match="fixture estimator rejects this candidate"):
        model.fit(X, y)
    results = model.cv_results_
    assert np.any(np.asarray(results["split0_test_accuracy"]) == -1.0)
    assert np.all(np.asarray(results["split1_test_accuracy"]) >= 0)
    assert not results["all_fits_failed"].any()


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
@pytest.mark.parametrize("refit", [False, "unavailable", "callable"])
def test_all_nonfinite_objective_warns_without_changing_refit(task, engine, refit):
    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)

    def refit_by_accuracy(results):
        return int(np.argmax(results["mean_test_accuracy"]))

    refit_setting = refit_by_accuracy if refit == "callable" else refit
    model = search(
        task,
        optimizer_backend=engine,
        scoring={"unavailable": _undefined_score, "accuracy": "accuracy"},
        refit=refit_setting,
        objective_metric="unavailable",
        use_cache=False,
        error_score="raise",
    )
    with pytest.warns(
        UserWarning, match="No candidate has a finite mean CV score.*'unavailable'"
    ) as emitted:
        model.fit(X, y)
    assert sum("No candidate has a finite" in str(w.message) for w in emitted) == 1
    if refit == "callable":
        assert model.best_index_ == refit_by_accuracy(model.cv_results_)
        assert not hasattr(model, "best_score_")
    else:
        assert np.isnan(model.best_score_)
    assert model.fit_counts_["refit_fits"] == int(bool(refit))
    assert np.isfinite(model.cv_results_["mean_test_accuracy"]).all()


@pytest.mark.parametrize("task", ["parameters", "features"])
def test_nonfinite_objective_warning_names_a_single_builtin_scorer(task):
    X, y = load_iris(return_X_y=True)
    rows = np.arange(len(y))
    model = search(task, scoring="r2", generations=0, refit=False).set_params(
        cv=[(rows[1:], rows[:1]), (rows[:-1], rows[-1:])]
    )
    with pytest.warns(UserWarning, match="No candidate has a finite mean CV score.*'r2'"):
        model.fit(X, y)
    assert np.isnan(model.best_score_)


@pytest.mark.parametrize("task", ["parameters", "features"])
def test_nonfinite_secondary_metric_does_not_warn_when_objective_is_finite(task):
    import warnings

    model = search(
        task,
        scoring={"unavailable": _undefined_score, "accuracy": "accuracy"},
        refit="accuracy",
        error_score="raise",
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="No candidate has a finite mean CV score")
        model.fit(*load_iris(return_X_y=True))
    assert np.isfinite(model.best_score_)
    assert np.isnan(model.cv_results_["mean_test_unavailable"]).all()


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
@pytest.mark.parametrize("refit", [False, "accuracy"])
def test_all_candidate_fits_fail_with_diagnostics_and_no_winner(task, engine, refit):
    if engine != "deap":
        pytest.importorskip(engine)
    model = _cv_failure_search(task, engine, all_fail=True, error_score=100.0).set_params(
        refit=refit, objective_metric="accuracy", final_selection=bool(refit)
    )
    with pytest.warns(FitFailedWarning), pytest.raises(ValueError, match="All search candidates"):
        model.fit(*load_iris(return_X_y=True))
    results = model.cv_results_
    assert results["all_fits_failed"].tolist() == [True] * 6
    assert np.isnan(results["split0_test_accuracy"]).all()
    assert np.isnan(results["split1_train_f1"]).all()
    assert np.isnan(results["mean_fit_time"]).all()
    np.testing.assert_array_equal(results["rank_test_accuracy"], np.ones(6, dtype=int))
    report = model.cv_results_dataframe(columns=["params", "all_fits_failed"], preserve_index=True)
    assert report.index.tolist() == list(range(6))
    assert report["all_fits_failed"].all()
    if task == "parameters":
        assert report["params"].tolist() == results["params"]
    else:
        assert [row["features"] for row in report["params"]] == [
            mask.tolist() for mask in results["features"]
        ]
    assert model.history["cross_validate_calls"] == [6]
    assert model.history["finite_fitness_count"] == [0]
    assert model.search_time_ >= 0
    assert model.fit_counts_["search_cv_calls"] == 6
    assert model.fit_counts_["search_fold_fits"] == 12
    assert model.fit_counts_["finalist_cv_calls"] == 0
    assert model.fit_counts_["refit_fits"] == 0
    assert not hasattr(model, "best_index_")
    with pytest.raises(NotFittedError):
        check_is_fitted(model)


@pytest.mark.parametrize("task", ["parameters", "features"])
def test_full_finalist_cv_failure_leaves_a_valid_candidate(task):
    X, y = load_iris(return_X_y=True)
    full = list(StratifiedKFold(2, shuffle=True, random_state=19).split(X, y))
    pilot = subsample_cv(full, 0.4, stratify=y, random_state=19)
    model = _cv_failure_search(task, "deap", error_score=100.0).set_params(
        estimator=_RejectSomeFinalists(max_depth=2, random_state=19),
        cv=pilot,
        final_selection=True,
        final_selection_cv=full,
        final_selection_top_k=6,
    )
    with pytest.warns(FitFailedWarning, match="fixture estimator rejects this candidate"):
        model.fit(X, y)
    finalists = model.final_selection_results_["candidates"]
    assert all(np.isfinite(row["original_score"]) for row in finalists)
    assert any(np.isnan(row["score"]) for row in finalists)
    assert np.isfinite(model.best_score_)
    assert model.fit_counts_["finalist_cv_calls"] == len(finalists)
    assert model.fit_counts_["finalist_fold_fits"] == 2 * len(finalists)
    assert model.fit_counts_["refit_fits"] == 1
    assert model.predict(X).shape == y.shape


@pytest.mark.parametrize("use_cache", [False, True])
def test_tied_pipeline_finalists_keep_first_distinct_estimator_choices(use_cache):
    choices = [DecisionTreeClassifier(max_depth=depth, random_state=19) for depth in (2, 4, 6)]

    def optimizer(problem, evaluate, report):
        proposals = [(choices[1],), (clone(choices[1]),), (choices[0],), (choices[2],)]
        report([handle for _, handle in evaluate(proposals)])

    model = GASearchCV(
        Pipeline([("model", choices[0])]),
        param_grid={"model": Categorical(choices)},
        optimizer_backend=optimizer,
        scoring=lambda estimator, X, y: 0.5,
        cv=2,
        final_selection=True,
        final_selection_top_k=2,
        population_size=4,
        generations=0,
        use_cache=use_cache,
        random_state=19,
        verbose=False,
        error_score="raise",
    ).fit(*load_iris(return_X_y=True))

    assert model.cv_results_["rank_test_score"].tolist() == [1] * 4
    finalists = model.final_selection_results_["candidates"]
    assert [row["index"] for row in finalists] == [0, 2]
    assert [row["params"]["model"].max_depth for row in finalists] == [4, 2]
    assert model.best_index_ == 0
    assert model.fit_counts_["search_cv_calls"] == (3 if use_cache else 4)
    assert model.fit_counts_["finalist_cv_calls"] == 2


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
def test_all_finalist_fit_failures_keep_cv_diagnostics(task, engine):
    if engine != "deap":
        pytest.importorskip(engine)

    class RejectFullFolds(DecisionTreeClassifier):
        def fit(self, X, y, sample_weight=None, check_input=True):
            if len(X) >= 50:
                raise ValueError("fixture rejects full training folds")
            return super().fit(X, y, sample_weight=sample_weight, check_input=check_input)

    X, y = load_iris(return_X_y=True)
    full = list(StratifiedKFold(2, shuffle=True, random_state=19).split(X, y))
    pilot = subsample_cv(full, 0.4, stratify=y, random_state=19)
    model = search(
        task,
        optimizer_backend=engine,
        final_selection=True,
        final_selection_cv=full,
        final_selection_top_k=2,
        error_score=100.0,
    ).set_params(estimator=RejectFullFolds(random_state=19), cv=pilot)
    with (
        pytest.warns(FitFailedWarning, match="fixture rejects full training folds"),
        pytest.raises(
            ValueError, match="All final-selection candidates produced non-finite scores"
        ),
    ):
        model.fit(X, y)

    results = model.final_selection_results_
    finalists = results["candidates"]
    assert len(finalists) == results["top_k"] == 2
    assert (
        results["selected_index"] is results["selected_score"] is results["selected_params"] is None
    )
    assert model.search_time_ >= 0
    assert results["time_seconds"] >= 0
    assert not results["changed"]
    assert not model.cv_results_["all_fits_failed"].any()
    assert all(np.isfinite(row["original_score"]) for row in finalists)
    assert all(np.isnan(row["score"]) and np.isnan(row["cv_scores"]).all() for row in finalists)
    assert all((row["cv_calls"], row["fold_fits"]) == (1, 2) for row in finalists)
    assert model.fit_counts_["finalist_cv_calls"] == 2
    assert model.fit_counts_["finalist_fold_fits"] == 4
    assert model.fit_counts_["refit_fits"] == 0
    assert not hasattr(model, "best_estimator_")


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
@pytest.mark.parametrize("failure_type", [ValueError, KeyboardInterrupt])
def test_interrupted_finalist_counts_the_cv_call_and_known_fold_fits(task, engine, failure_type):
    if engine != "deap":
        pytest.importorskip(engine)

    attempted = []
    failure = failure_type("fixture interrupts the second finalist")

    class InterruptFinalist(DecisionTreeClassifier):
        def fit(self, X, y, sample_weight=None, check_input=True):
            if len(X) >= 50:
                attempted.append(len(X))
                if len(attempted) == 4:
                    raise failure
            return super().fit(X, y, sample_weight=sample_weight, check_input=check_input)

    X, y = load_iris(return_X_y=True)
    full = list(StratifiedKFold(2, shuffle=True, random_state=19).split(X, y))
    pilot = subsample_cv(full, 0.4, stratify=y, random_state=19)
    model = search(
        task,
        optimizer_backend=engine,
        final_selection=True,
        final_selection_cv=full,
        final_selection_top_k=2,
        generations=0,
        error_score="raise",
    ).set_params(estimator=InterruptFinalist(random_state=19), cv=pilot)

    with pytest.raises(failure_type, match="fixture interrupts the second finalist") as error:
        model.fit(X, y)

    assert error.value is failure
    assert attempted == [75, 75, 75, 75]
    assert model.fit_counts_["finalist_cv_calls"] == 2
    assert model.fit_counts_["finalist_fold_fits"] == 2
    assert model.fit_counts_["refit_fits"] == 0
    results = model.final_selection_results_
    assert results["selected_index"] is None
    assert results["time_seconds"] >= 0
    (completed,) = results["candidates"]
    assert (completed["cv_calls"], completed["fold_fits"]) == (1, 2)
    interrupted = results["failed_candidate"]
    index = interrupted["index"]
    assert index != completed["index"]
    assert (interrupted["cv_calls"], interrupted["fold_fits"]) == (1, 0)
    if task == "parameters":
        assert interrupted["params"] == model.cv_results_["params"][index]
    else:
        assert interrupted["params"]["features"] == tuple(model.cv_results_["features"][index])


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize("undefined_fold", ["reused", "fresh"])
def test_finalist_fold_reuse_keeps_undefined_scores_without_retrying(task, undefined_fold):
    X, y = load_iris(return_X_y=True)
    X[:, 0] = np.arange(len(X))

    def fold(start):
        validation = np.arange(start, start + 4)
        training = np.setdiff1d(np.arange(len(X)), validation)
        return training, validation

    final_splits = [fold(0), fold(12)]
    undefined_start = 0 if undefined_fold == "reused" else 12

    def score(estimator, X, y):
        complexity = estimator.max_depth if task == "parameters" else X.shape[1]
        if complexity == 1 and X[0, 0] == undefined_start:
            return np.nan
        return 1.0 if complexity == 1 else 0.5

    def optimize(problem, evaluate, report):
        candidates = (
            [(1, "gini"), (2, "gini")] if task == "parameters" else [(1, 0, 0, 0), (1, 1, 0, 0)]
        )
        report([handle for _, handle in evaluate(candidates)])

    options = dict(
        final_selection=True,
        final_selection_cv=final_splits,
        final_selection_top_k=2,
        generations=0,
        optimizer_backend=optimize,
        scoring=score,
        use_cache=False,
        error_score="raise",
    )
    if task == "parameters":
        options["param_grid"] = {"max_depth": Integer(1, 2), "criterion": Categorical(["gini"])}
    template = search(task, **options).set_params(cv=[fold(0), fold(4), fold(8)])
    fresh = clone(template).fit(X, y)
    reused = template.set_params(final_selection="reuse_folds").fit(X, y)

    assert fresh.best_index_ == reused.best_index_ == 1
    assert fresh.best_score_ == reused.best_score_ == 0.5
    np.testing.assert_array_equal(fresh.predict(X), reused.predict(X))
    expected = fresh.final_selection_results_["candidates"]
    actual = reused.final_selection_results_["candidates"]
    assert any(np.isnan(row["score"]) for row in actual)
    for before, after in zip(expected, actual, strict=True):
        assert before["index"] == after["index"]
        np.testing.assert_equal(before["cv_scores"], after["cv_scores"])
        assert (before["cv_calls"], before["fold_fits"]) == (1, 2)
        assert (after["cv_calls"], after["fold_fits"], after["reused_folds"]) == (1, 1, 1)
    assert reused.fit_counts_["search_cv_calls"] == fresh.fit_counts_["search_cv_calls"] == 2
    assert reused.fit_counts_["finalist_cv_calls"] == 2
    assert reused.fit_counts_["finalist_fold_fits"] == 2
    assert reused.fit_counts_["finalist_reused_folds"] == 2


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
def test_fit_phase_durations_use_elapsed_clock(task, engine, monkeypatch):
    from types import SimpleNamespace

    from sklearn_genetic import _base, genetic_search

    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    search_elapsed = iter((2.0, 9.5))
    monkeypatch.setattr(_base, "perf_counter", search_elapsed.__next__)
    wall = iter((100.0, 90.0, 170.0, 150.0))
    elapsed = iter((10.0, 13.75, 14.0, 14.125))
    monkeypatch.setattr(
        genetic_search, "time", SimpleNamespace(time=wall.__next__, perf_counter=elapsed.__next__)
    )
    result = search(
        task,
        optimizer_backend=engine,
        generations=0,
        final_selection=True,
        final_selection_cv=3,
        final_selection_top_k=2,
    ).fit(X, y)

    assert result.search_time_ == 7.5
    assert result.final_selection_results_["time_seconds"] == 3.75
    assert result.refit_time_ == 0.125
    assert result.fit_counts_["finalist_fold_fits"] == 6
    assert result.fit_counts_["refit_fits"] == 1
    assert len(result.predict(X)) == len(y)

    with pytest.raises(ValueError, match="requires y"):
        result.fit(X)
    assert not hasattr(result, "search_time_")


@pytest.mark.parametrize("selector_type", [np.array, list, tuple])
def test_finalist_fold_comparison_converts_each_shared_selector_once(selector_type, monkeypatch):
    from types import SimpleNamespace

    from sklearn_genetic import genetic_search

    train = selector_type([0, 1, 2])
    test = selector_type([False, False, False, True])
    calls = []

    def asarray(selector):
        calls.append(selector)
        return np.asarray(selector)

    monkeypatch.setattr(
        genetic_search,
        "np",
        SimpleNamespace(asarray=asarray, ndarray=np.ndarray, array_equal=np.array_equal),
    )
    fold = (train, test)
    assert genetic_search._FinalSelectionMixin._same_fold(fold, fold, 4)
    assert len(calls) == 2 and calls[0] is train and calls[1] is test


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize("score_kind", ["integer", "float", "mixed"])
def test_finalist_reuses_original_score_array_when_all_folds_match(task, score_kind):
    X, y = load_iris(return_X_y=True)
    rows = np.arange(len(X))
    folds = [(rows[:-size], rows[-size:]) for size in (12, 14, 16)]

    def score(estimator, X, y):
        complexity = estimator.max_depth if task == "parameters" else X.shape[1]
        value = 2**53 + 2 * complexity + 1
        if score_kind == "mixed" and len(X) == 12:
            return 0.5
        return float(value) if score_kind == "float" else value

    template = search(
        task,
        scoring=score,
        generations=0,
        final_selection=True,
        final_selection_cv=folds if task == "features" else None,
        final_selection_top_k=2,
        return_train_score=score_kind == "mixed",
        error_score="raise",
    ).set_params(cv=folds)
    fresh = clone(template).fit(X, y)
    reused = template.set_params(final_selection="reuse_folds").fit(X, y)

    assert reused.best_index_ == fresh.best_index_
    assert reused.best_score_ == fresh.best_score_
    np.testing.assert_array_equal(reused.predict(X), fresh.predict(X))
    expected = fresh.final_selection_results_["candidates"]
    actual = reused.final_selection_results_["candidates"]
    assert expected and len(expected) == len(actual)
    for before, after in zip(expected, actual, strict=True):
        assert before["index"] == after["index"]
        assert before["cv_scores"] == after["cv_scores"]
        assert all(type(a) is type(b) for a, b in zip(before["cv_scores"], after["cv_scores"]))
        assert (before["cv_calls"], before["fold_fits"]) == (1, 3)
        assert (after["cv_calls"], after["fold_fits"], after["reused_folds"]) == (0, 0, 3)
    assert reused.fit_counts_["finalist_cv_calls"] == 0
    assert reused.fit_counts_["finalist_fold_fits"] == 0
    assert reused.fit_counts_["finalist_reused_folds"] == 3 * len(expected)


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize("partition", ["training", "validation"])
@pytest.mark.parametrize("container", [list, tuple])
def test_finalist_reuse_respects_sklearn_mixed_selector_behavior(task, partition, container):
    X, y = load_iris(return_X_y=True)
    rows = np.arange(len(X))
    rest = rows[3:]
    mixed = container([False, 1, np.bool_(True), 2])
    positions = np.array([0, 1, 1, 2])
    if partition == "training":
        # Keep representatives of all iris classes in the small training fold.
        rows = np.ravel(np.column_stack((rows[:50], rows[50:100], rows[100:])))
        X, y = X[rows], y[rows]
        search_fold, final_fold = (positions, rest), (mixed, rest)
    else:
        search_fold, final_fold = (rest, positions), (rest, mixed)
    template = search(
        task, final_selection=True, final_selection_top_k=1, error_score="raise"
    ).set_params(cv=[search_fold], final_selection_cv=[final_fold])
    reused = clone(template).set_params(final_selection="reuse_folds")

    try:
        template.fit(X, y, callbacks=CVBudgetStopping(1))
    except ValueError as error:
        assert template.fit_counts_["search_cv_calls"] == 1
        with pytest.raises(ValueError) as actual:
            reused.fit(X, y, callbacks=CVBudgetStopping(1))
        assert str(actual.value) == str(error)
    else:
        reused.fit(X, y, callbacks=CVBudgetStopping(1))
        assert reused.fit_counts_ == template.fit_counts_
        assert reused.fit_counts_["finalist_fold_fits"] == 1
        assert reused.fit_counts_["finalist_reused_folds"] == 0
        assert reused.best_score_ == template.best_score_
        np.testing.assert_array_equal(reused.predict(X), template.predict(X))


class _TupleWithArrayView(tuple):
    def __array__(self, dtype=None, copy=None):
        return np.array([0, 1, 2], dtype=dtype)


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize("custom_stage", ["search", "finalists"])
@pytest.mark.parametrize("extra_search_fold", [False, True])
def test_finalist_fold_does_not_reuse_a_custom_selector_by_its_array_view(
    task, custom_stage, extra_search_fold
):
    rows = np.arange(35)
    X, y = np.column_stack((rows, 8 - rows)), rows % 2
    train = rows[10:]
    plain, custom = rows[:3], _TupleWithArrayView((6, 7, 8))
    search_folds = [(train, custom if custom_stage == "search" else plain)]
    if extra_search_fold:
        # Exercise the fold lookup as well as the equal-length comparison.
        search_folds.insert(0, (train, rows[3:6]))
    final_folds = [(train, custom if custom_stage == "finalists" else plain)]
    _assert_finalist_recomputes_fold(task, X, y, search_folds, final_folds)


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("extra_search_fold", [False, True])
def test_finalist_fold_recomputes_a_mixed_search_selector(task, container, extra_search_fold):
    import pandas as pd

    rows = np.arange(35)
    labels = rows.copy()
    # In some sklearn versions, the mixed sequence indexes the dataframe
    # by label. The integer array always indexes by position.
    labels[[0, 1, 6, 7]] = labels[[6, 7, 0, 1]]
    X = pd.DataFrame({"left": rows, "right": 8 - rows}, index=labels)
    y = rows % 2
    train = rows[10:]
    search_folds = [(train, container([False, True, 0, 1]))]
    if extra_search_fold:
        search_folds.insert(0, (train, rows[3:6]))
    final_folds = [(train, np.array([0, 1, 0, 1]))]
    _assert_finalist_recomputes_fold(task, X, y, search_folds, final_folds, preserve_dataframe=True)


def _assert_finalist_recomputes_fold(
    task, X, y, search_folds, final_folds, *, preserve_dataframe=False
):
    def optimize(problem, evaluate, report):
        proposals = [(1,), (2,)] if task == "parameters" else [(1, 0), (0, 1)]
        report([handle for _, handle in evaluate(proposals)])

    def score(estimator, X, y):
        col = estimator.max_depth - 1 if task == "parameters" else 0
        return float(np.asarray(X)[:, col].mean()) + 0.25

    options = dict(
        cv=search_folds,
        final_selection=True,
        final_selection_cv=final_folds,
        final_selection_top_k=2,
        scoring=score,
        error_score="raise",
        optimizer_backend=optimize,
        n_jobs=1,
        random_state=99,
        population_size=2,
        generations=0,
        verbose=False,
    )
    estimator = DecisionTreeClassifier(random_state=99)
    if task == "parameters":
        model = GASearchCV(estimator, param_grid={"max_depth": Integer(1, 2)}, **options)
    else:
        model = GAFeatureSelectionCV(
            estimator, max_features=1, preserve_dataframe=preserve_dataframe, **options
        )
    fresh = clone(model).fit(X, y)
    reused = model.set_params(final_selection="reuse_folds").fit(X, y)

    assert reused.best_index_ == fresh.best_index_
    assert reused.best_score_ == fresh.best_score_
    assert reused.fit_counts_ == fresh.fit_counts_
    assert reused.fit_counts_["finalist_fold_fits"] == 2
    assert reused.fit_counts_["finalist_reused_folds"] == 0
    np.testing.assert_equal(
        reused.final_selection_results_["candidates"], fresh.final_selection_results_["candidates"]
    )
    np.testing.assert_array_equal(reused.predict(X), fresh.predict(X))


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize("final_cv", ["shared", "positions", "masks", "reordered", "extra"])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
def test_finalist_reuse_preserves_each_occurrence_of_repeated_folds(task, final_cv, engine):
    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    X = X[:, :2].copy()
    X[:, 1] += 100
    X, X_test, y, _ = train_test_split(X, y, test_size=0.25, random_state=22, stratify=y)
    first, second = StratifiedKFold(2).split(X, y)
    folds = [first, second, first]
    visits = 0

    def score(estimator, X, y):
        nonlocal visits
        preferred = estimator.max_depth == 2 if task == "parameters" else X[0, 0] < 100
        if not preferred:
            return 0.5
        value = (0.25, 0.5, 1.0)[visits % 3]
        visits += 1
        return value

    def row_mask(positions):
        mask = np.zeros(len(y), dtype=bool)
        mask[positions] = True
        return mask

    finalist_folds = {
        "shared": None,
        "positions": [(train.copy(), test.copy()) for train, test in folds],
        "masks": [(row_mask(train), row_mask(test)) for train, test in folds],
        "reordered": [second, first, first],
        "extra": [second, first, first, first, first],
    }[final_cv]
    expected_scores = {
        "reordered": [0.5, 0.25, 1.0],
        "extra": [0.5, 0.25, 1.0, 0.25, 1.0],
    }.get(final_cv, [0.25, 0.5, 1.0])
    options = dict(
        optimizer_backend=engine,
        generations=0,
        scoring=score,
        error_score="raise",
        final_selection="reuse_folds",
        final_selection_cv=finalist_folds,
        final_selection_top_k=2,
    )
    if task == "parameters":
        options["param_grid"] = {"max_depth": Integer(2, 3), "criterion": Categorical(["gini"])}
    model = search(task, **options).set_params(cv=folds)
    if task == "features":
        model.max_features = 1
    model.fit(X, y)

    finalists = model.final_selection_results_
    assert finalists["selected_index"] == finalists["original_best_index"] == model.best_index_
    assert model.best_score_ == pytest.approx(np.mean(expected_scores))
    assert finalists["candidates"][0]["cv_scores"] == expected_scores
    assert finalists["candidates"][1]["cv_scores"] == [0.5] * len(expected_scores)
    assert visits == 3
    assert model.fit_counts_["search_cv_calls"] == 2
    assert model.fit_counts_["finalist_cv_calls"] == 0
    assert model.fit_counts_["finalist_fold_fits"] == 0
    assert model.fit_counts_["finalist_reused_folds"] == 2 * len(expected_scores)
    if task == "parameters":
        assert model.best_params_["max_depth"] == 2
        columns, depth = [0, 1], 2
    else:
        np.testing.assert_array_equal(model.get_support(indices=True), [0])
        columns, depth = [0], None
    expected = clone(model.estimator).set_params(max_depth=depth).fit(X[:, columns], y)
    np.testing.assert_array_equal(model.predict(X_test), expected.predict(X_test[:, columns]))


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize("engine", ["deap", "pymoo"])
@pytest.mark.parametrize("score_kind", ["fractional", "integers", "large_float", "mixed_integers"])
@pytest.mark.parametrize("repeat_fold", [False, True])
def test_finalist_reuses_scores_when_all_folds_are_reordered(task, engine, score_kind, repeat_fold):
    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    rows = np.arange(len(X))
    folds = [(rows[:-size], rows[-size:]) for size in (12, 14, 16)]
    final_folds = [folds[2], folds[0], folds[1]]
    if repeat_fold:
        folds.append(folds[0])
        final_folds.append(folds[0])

    def score(estimator, X, y):
        complexity = estimator.max_depth if task == "parameters" else X.shape[1]
        if score_kind == "fractional":
            return 0.01 * (len(X) + complexity)
        if score_kind == "integers":
            return len(X) * complexity
        if score_kind == "large_float":
            return (0.35 + 0.01 * (len(X) + complexity)) * 1e308
        if len(X) == 12:
            return 2**63 + 2 * complexity + 1
        return -(2**54 + 2 * complexity + 1) if len(X) == 14 else 2**54 + 2 * complexity + 1

    template = search(
        task,
        optimizer_backend=engine,
        scoring=score,
        generations=0,
        final_selection=True,
        final_selection_cv=final_folds,
        final_selection_top_k=2,
        error_score="raise",
    ).set_params(cv=folds)
    fresh = clone(template).fit(X, y, callbacks=CVBudgetStopping(6))
    reused = template.set_params(final_selection="reuse_folds").fit(
        X, y, callbacks=CVBudgetStopping(6)
    )

    assert reused.best_index_ == fresh.best_index_
    assert reused.best_score_ == fresh.best_score_
    np.testing.assert_array_equal(reused.predict(X), fresh.predict(X))
    expected = fresh.final_selection_results_["candidates"]
    actual = reused.final_selection_results_["candidates"]
    assert expected and len(expected) == len(actual)
    for before, after in zip(expected, actual, strict=True):
        assert before["index"] == after["index"]
        assert before["cv_scores"] == after["cv_scores"]
        assert all(type(a) is type(b) for a, b in zip(before["cv_scores"], after["cv_scores"]))
        assert after["reused_folds"] == len(final_folds)
    assert reused.fit_counts_["search_cv_calls"] == fresh.fit_counts_["search_cv_calls"] == 6
    assert reused.fit_counts_["finalist_cv_calls"] == 0
    assert reused.fit_counts_["finalist_fold_fits"] == 0
    assert reused.fit_counts_["finalist_reused_folds"] == fresh.fit_counts_["finalist_fold_fits"]


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize("integer_range", ["signed", "unsigned", "mixed"])
def test_finalist_reuses_exact_large_integer_scores_in_a_subset(task, integer_range):
    X, y = load_iris(return_X_y=True)
    rows = np.arange(len(X))
    folds = [(rows[:-size], rows[-size:]) for size in (12, 14, 16)]
    final_folds = [folds[2], folds[1]]

    def score(estimator, X, y):
        complexity = estimator.max_depth if task == "parameters" else X.shape[1]
        if integer_range == "unsigned" or (integer_range == "mixed" and len(X) == 12):
            return 2**63 + 2 * complexity + 1
        value = 2**53 + 2 * complexity + 1
        return -value if len(X) == 14 else value

    template = search(
        task,
        scoring=score,
        generations=0,
        final_selection=True,
        final_selection_cv=final_folds,
        final_selection_top_k=2,
        error_score="raise",
    ).set_params(cv=folds)
    fresh = clone(template).fit(X, y)
    reused = template.set_params(final_selection="reuse_folds").fit(X, y)

    assert reused.best_index_ == fresh.best_index_
    assert reused.best_score_ == fresh.best_score_
    np.testing.assert_array_equal(reused.predict(X), fresh.predict(X))
    expected = fresh.final_selection_results_["candidates"]
    actual = reused.final_selection_results_["candidates"]
    assert expected and len(expected) == len(actual)
    for before, after in zip(expected, actual, strict=True):
        assert before["index"] == after["index"]
        assert before["cv_scores"] == after["cv_scores"]
        assert all(type(value) is int and value % 2 for value in after["cv_scores"])

    matched = 0 if integer_range == "mixed" else len(final_folds)
    assert all(row["reused_folds"] == matched for row in actual)
    assert reused.fit_counts_["finalist_fold_fits"] == (len(final_folds) - matched) * len(actual)


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize(
    "score_kind",
    ["double", "extended", "extended-range", "extended-only-in-search", "integer-at-ceiling"],
)
def test_finalist_reuses_large_scores_when_the_subset_retains_the_float_dtype(task, score_kind):
    if score_kind.startswith("extended") and (
        np.finfo(np.longdouble).nmant <= np.finfo(float).nmant
    ):
        pytest.skip("requires extended precision")
    if score_kind == "extended-range" and np.finfo(np.longdouble).maxexp <= 1030:
        pytest.skip("requires extended exponent range")

    X, y = load_iris(return_X_y=True)
    rows = np.arange(len(X))
    folds = [(rows[:-size], rows[-size:]) for size in (12, 14, 16)]
    final_folds = [folds[2], folds[1]]

    def score(estimator, X, y):
        complexity = estimator.max_depth if task == "parameters" else X.shape[1]
        if score_kind == "integer-at-ceiling":
            return 0.25 if len(X) == 12 else 2**64 - 2 * complexity - len(X) - 1
        scale = complexity + len(X) / 128
        scale = -scale if len(X) == 14 else scale
        if score_kind == "extended-range":
            return np.ldexp(np.longdouble(scale), 1026)
        value = scale * 2.0**90
        if score_kind == "extended" or (score_kind == "extended-only-in-search" and len(X) == 12):
            return np.nextafter(np.longdouble(value), np.longdouble(np.inf))
        return value

    template = search(
        task,
        scoring=score,
        generations=0,
        final_selection=True,
        final_selection_cv=final_folds,
        final_selection_top_k=2,
        error_score="raise",
    ).set_params(cv=folds)
    fresh = clone(template).fit(X, y)
    reused = template.set_params(final_selection="reuse_folds").fit(X, y)

    assert reused.best_index_ == fresh.best_index_
    assert reused.best_score_ == fresh.best_score_
    np.testing.assert_array_equal(reused.predict(X), fresh.predict(X))
    expected = fresh.final_selection_results_["candidates"]
    actual = reused.final_selection_results_["candidates"]
    assert expected and len(expected) == len(actual)
    matched = len(final_folds) if score_kind in {"double", "extended", "extended-range"} else 0
    for before, after in zip(expected, actual, strict=True):
        assert before["index"] == after["index"]
        assert before["score"] == after["score"]
        assert before["cv_scores"] == after["cv_scores"]
        assert all(type(a) is type(b) for a, b in zip(before["cv_scores"], after["cv_scores"]))
        assert after["reused_folds"] == matched
    assert reused.fit_counts_["finalist_fold_fits"] == (len(final_folds) - matched) * len(actual)


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize("large_value", [-(2**54 + 1), 2**63 + 1])
@pytest.mark.parametrize(
    "large_fold_sizes, new_score",
    [
        pytest.param((18,), "integer", id="large-new-score"),
        pytest.param((12, 14, 16), "integer", id="large-saved-with-new-integer"),
        pytest.param((12, 14, 16), "float", id="large-saved-with-new-float"),
        pytest.param((14, 16), "object", id="large-saved-with-object-score"),
    ],
)
def test_finalist_combines_exact_integer_scores_from_search_and_new_folds(
    task, large_value, large_fold_sizes, new_score
):
    from fractions import Fraction

    X, y = load_iris(return_X_y=True)
    rows = np.arange(len(X))
    folds = [(rows[:-size], rows[-size:]) for size in (12, 14, 16)]
    final_folds = [folds[2], (rows[:-18], rows[-18:]), folds[0]]

    def score(estimator, X, y):
        complexity = estimator.max_depth if task == "parameters" else X.shape[1]
        if len(X) in large_fold_sizes:
            return large_value + 2 * complexity + 2 * len(X)
        value = -len(X) * complexity
        if new_score == "object":
            # An integer search array cannot recover this original Boolean type.
            return True if len(X) == 12 else Fraction(4 * value + 1, 4)
        return value + 0.25 if new_score == "float" else value

    template = search(
        task,
        scoring=score,
        generations=0,
        final_selection=True,
        final_selection_cv=final_folds,
        final_selection_top_k=2,
        error_score="raise",
    ).set_params(cv=folds)
    fresh = clone(template).fit(X, y)
    reused = template.set_params(final_selection="reuse_folds").fit(X, y)

    assert reused.best_index_ == fresh.best_index_
    assert reused.best_score_ == fresh.best_score_
    np.testing.assert_array_equal(reused.predict(X), fresh.predict(X))
    expected = fresh.final_selection_results_["candidates"]
    actual = reused.final_selection_results_["candidates"]
    assert expected and len(expected) == len(actual)
    for before, after in zip(expected, actual, strict=True):
        assert before["index"] == after["index"]
        assert before["cv_scores"] == after["cv_scores"]
        assert all(type(a) is type(b) for a, b in zip(before["cv_scores"], after["cv_scores"]))

    counts = reused.fit_counts_
    calls, fits, matched = (2, 4, 0) if new_score == "object" else (1, 1, 2)
    assert counts["finalist_cv_calls"] == calls * len(actual)
    assert counts["finalist_fold_fits"] == fits * len(actual)
    assert counts["finalist_reused_folds"] == matched * len(actual)


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
@pytest.mark.parametrize(
    "mode",
    [
        "shared",
        "copied",
        "partial",
        "mixed",
        "mixed-reverse",
        "mixed-reordered",
        "negative-after-mask",
        "negative-before-indices",
    ],
)
def test_finalists_reuse_matching_row_selectors(task, engine, mode):
    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    X, X_test, y, _ = train_test_split(X, y, test_size=0.25, stratify=y, random_state=41)
    splits = list(StratifiedKFold(3, shuffle=True, random_state=41).split(X, y))

    def mask(indices):
        result = np.zeros(len(X), dtype=bool)
        result[indices] = True
        return result

    cv = [(mask(train), mask(test)) for train, test in splits]
    final_cv = None if mode == "shared" else [(train.copy(), test.copy()) for train, test in cv]
    if mode == "partial":
        train, test = final_cv[1]
        removed = np.flatnonzero(test[1:-1])[0] + 1
        added = np.flatnonzero(train[1:-1])[0] + 1
        # Keep the coarse key and change both masks within it.
        train[removed], train[added] = True, False
        test[removed], test[added] = False, True
    elif mode == "mixed":
        final_cv = splits
    elif mode == "mixed-reverse":
        cv, final_cv = splits, cv
    elif mode == "mixed-reordered":
        final_cv = splits.copy()
        train, test = final_cv[1]
        test = test.copy()
        test[1:-1] = test[1:-1][::-1]
        final_cv[1] = train, test
    elif mode.startswith("negative"):

        def signed(indices):
            return np.where(indices % 2, indices - len(X), indices)

        negative_folds = [(signed(train), signed(test)) for train, test in splits]
        if mode == "negative-after-mask":
            final_cv = negative_folds
        else:
            cv, final_cv = negative_folds, splits

    def order_sensitive(estimator, X, y):
        return float(X[1, 0])

    scoring = order_sensitive if mode == "mixed-reordered" else "accuracy"
    template = search(
        task, optimizer_backend=engine, scoring=scoring, error_score="raise", random_state=41
    ).set_params(cv=cv, final_selection=True, final_selection_cv=final_cv, final_selection_top_k=2)
    fresh = clone(template).fit(X, y, callbacks=CVBudgetStopping(4))
    reused = template.set_params(final_selection="reuse_folds").fit(
        X, y, callbacks=CVBudgetStopping(4)
    )

    assert reused.best_index_ == fresh.best_index_
    np.testing.assert_array_equal(reused.predict(X_test), fresh.predict(X_test))
    expected = fresh.final_selection_results_["candidates"]
    actual = reused.final_selection_results_["candidates"]
    assert actual and len(actual) == len(expected)
    new_fits = int(mode in ("partial", "mixed-reordered"))
    for before, after in zip(expected, actual, strict=True):
        assert (before["index"], before["cv_scores"]) == (after["index"], after["cv_scores"])
        assert (after["fold_fits"], after["reused_folds"]) == (new_fits, 3 - new_fits)
    assert reused.fit_counts_["search_cv_calls"] == fresh.fit_counts_["search_cv_calls"] == 4
    assert reused.fit_counts_["finalist_fold_fits"] == new_fits * len(actual)


def test_finalists_do_not_reuse_a_boolean_mask_with_an_invalid_length():
    X, y = load_iris(return_X_y=True)
    splits = list(StratifiedKFold(2).split(X, y))
    train, test = splits[0]
    invalid = np.zeros(len(X) + 1, dtype=bool)
    invalid[test] = True
    model = search(
        "parameters", scoring="accuracy", final_selection="reuse_folds", error_score="raise"
    ).set_params(cv=splits, final_selection_cv=[(train, invalid)])

    with pytest.raises(IndexError):
        model.fit(X, y, callbacks=CVBudgetStopping(2))


@pytest.mark.parametrize("invalid", [False, True])
def test_finalist_matches_narrow_signed_indices_and_checks_bounds(invalid):
    X, y = load_iris(return_X_y=True)
    splits = list(StratifiedKFold(2).split(X, y))
    train, test = splits[-1]
    assert test[-1] == len(X) - 1

    def narrow(indices):
        return np.where(indices >= 128, indices - len(X), indices).astype(np.int8)

    finalist_test = narrow(test)
    if invalid:
        finalist_test = finalist_test.astype(np.int16)
        finalist_test[-2] = int(test[-2]) - 2 * len(X)
    model = search(
        "parameters", scoring="accuracy", final_selection="reuse_folds", error_score="raise"
    ).set_params(cv=splits, final_selection_cv=[(narrow(train), finalist_test)])

    if invalid:
        with pytest.raises(IndexError):
            model.fit(X, y, callbacks=CVBudgetStopping(2))
        return

    model.fit(X, y, callbacks=CVBudgetStopping(2))
    assert model.fit_counts_["finalist_fold_fits"] == 0
    for candidate in model.final_selection_results_["candidates"]:
        assert candidate["reused_folds"] == 1
        assert candidate["cv_scores"] == [
            model.cv_results_["split1_test_score"][candidate["index"]]
        ]


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize(
    ("search_score", "offset"),
    [
        ("integral_float", 2**53),
        ("integral_float", 2**63),
        ("promoted_integer", 2**53),
        ("promoted_integer", 2**63),
        ("fractional", 2**53),
        ("fractional", -(2**54)),
        ("fractional", 2**63),
    ],
)
def test_finalist_preserves_score_types_and_counts_for_partial_reuse(task, search_score, offset):
    X, y = load_iris(return_X_y=True)
    rows = np.arange(len(X))
    search_folds = [(rows[:-size], rows[-size:]) for size in (12, 14, 16)]
    final_folds = [search_folds[0], (rows[:-18], rows[-18:]), search_folds[2]]

    def score(estimator, X, y):
        complexity = estimator.max_depth if task == "parameters" else X.shape[1]
        if len(X) == 18:
            return offset + 2 * complexity + 1
        if search_score == "promoted_integer":
            return 0.25 if len(X) == 14 else complexity
        return float(complexity) + (0.25 if search_score == "fractional" else 0)

    template = search(
        task,
        scoring=score,
        generations=0,
        final_selection=True,
        final_selection_top_k=2,
        final_selection_cv=final_folds,
        error_score="raise",
    ).set_params(cv=search_folds)
    fresh = clone(template).fit(X, y)
    reused = template.set_params(final_selection="reuse_folds").fit(X, y)

    assert reused.best_index_ == fresh.best_index_
    assert reused.best_score_ == fresh.best_score_
    np.testing.assert_array_equal(reused.predict(X), fresh.predict(X))
    expected = fresh.final_selection_results_["candidates"]
    actual = reused.final_selection_results_["candidates"]
    counts = reused.fit_counts_
    calls, fits, matches = (1, 1, 2) if search_score == "fractional" else (2, 3, 0)
    assert counts["finalist_cv_calls"] == calls * len(actual)
    assert counts["finalist_fold_fits"] == fits * len(actual)
    assert counts["finalist_reused_folds"] == matches * len(actual)
    for before, after in zip(expected, actual, strict=True):
        assert before["index"] == after["index"]
        assert before["cv_scores"] == after["cv_scores"]
        assert all(type(a) is type(b) for a, b in zip(before["cv_scores"], after["cv_scores"]))
        assert (before["cv_calls"], before["fold_fits"]) == (1, 3)
        assert (after["cv_calls"], after["fold_fits"], after["reused_folds"]) == (
            calls,
            fits,
            matches,
        )


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize(
    "search_score", ["double", "integer", "promoted_integer", "promoted_double"]
)
def test_finalist_reuses_small_saved_scores_with_large_new_doubles(task, search_score):
    if search_score == "promoted_double" and np.dtype(np.longdouble) == np.dtype(float):
        pytest.skip("requires extended precision")

    X, y = load_iris(return_X_y=True)
    rows = np.arange(len(X))
    search_folds = [(rows[:-size], rows[-size:]) for size in (12, 14, 16)]
    final_folds = [search_folds[2], (rows[:-18], rows[-18:]), search_folds[0]]

    def score(estimator, X, y):
        complexity = estimator.max_depth if task == "parameters" else X.shape[1]
        if len(X) == 18:
            return float(complexity) * 2.0**90
        if search_score == "integer":
            return complexity
        if search_score == "promoted_integer" and len(X) != 14:
            return -complexity
        value = complexity + len(X) / 64
        return np.longdouble(value) if search_score == "promoted_double" and len(X) == 14 else value

    template = search(
        task,
        scoring=score,
        generations=0,
        final_selection=True,
        final_selection_top_k=2,
        final_selection_cv=final_folds,
        error_score="raise",
    ).set_params(cv=search_folds)
    fresh = clone(template).fit(X, y)
    reused = template.set_params(final_selection="reuse_folds").fit(X, y)

    assert reused.best_index_ == fresh.best_index_
    assert reused.best_score_ == fresh.best_score_
    np.testing.assert_array_equal(reused.predict(X), fresh.predict(X))
    expected = fresh.final_selection_results_["candidates"]
    actual = reused.final_selection_results_["candidates"]
    assert expected and len(expected) == len(actual)
    calls, fits, matches = (2, 4, 0) if search_score == "promoted_double" else (1, 1, 2)
    for before, after in zip(expected, actual, strict=True):
        assert before["index"] == after["index"]
        assert before["score"] == after["score"]
        assert before["cv_scores"] == after["cv_scores"]
        assert all(type(value) is float for value in after["cv_scores"])
        assert (after["cv_calls"], after["fold_fits"], after["reused_folds"]) == (
            calls,
            fits,
            matches,
        )
    counts = reused.fit_counts_
    assert counts["finalist_cv_calls"] == calls * len(actual)
    assert counts["finalist_fold_fits"] == fits * len(actual)
    assert counts["finalist_reused_folds"] == matches * len(actual)


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize(
    "precision",
    [
        "discarded_only",
        "discarded_only_with_new_fold",
        "retained",
        "new",
        "discarded_and_new",
        "new_mixed_small_scores",
        "new_mixed_large_scores",
    ],
)
def test_finalist_reuse_requires_evidence_for_extended_precision(task, precision):
    if np.finfo(np.longdouble).nmant <= np.finfo(float).nmant:
        pytest.skip("requires extended precision")

    X, y = load_iris(return_X_y=True)
    rows = np.arange(len(X))
    search_folds = [(rows[:-size], rows[-size:]) for size in (12, 14, 16)]
    final_folds = [search_folds[2], search_folds[0]]
    if precision != "discarded_only":
        final_folds.insert(1, (rows[:-18], rows[-18:]))
    if precision.startswith("new_mixed"):
        final_folds.insert(2, (rows[:-20], rows[-20:]))

    def score(estimator, X, y):
        complexity = estimator.max_depth if task == "parameters" else X.shape[1]
        value = float(complexity)
        if precision.startswith("new_mixed"):
            if len(X) in (12, 16):
                return np.longdouble(value)
            if len(X) == 18:
                return (
                    2**53 + 2 * complexity + 1
                    if precision == "new_mixed_large_scores"
                    else complexity
                )
            return value + 0.125
        if len(X) == 18 and precision in ("new", "discarded_and_new"):
            return np.nextafter(np.ldexp(np.longdouble(value), 90), np.longdouble(np.inf))
        if len(X) == 16 and precision == "retained":
            return np.nextafter(np.longdouble(value), np.longdouble(np.inf))
        if len(X) == 14 and precision.startswith("discarded"):
            return np.longdouble(value)
        return np.nextafter(value, np.inf) if len(X) == 16 else value

    model = search(
        task,
        scoring=score,
        generations=0,
        final_selection=True,
        final_selection_cv=final_folds,
        final_selection_top_k=2,
        error_score="raise",
    ).set_params(cv=search_folds)
    fresh = clone(model).fit(X, y)
    reused = model.set_params(final_selection="reuse_folds").fit(X, y)

    assert reused.best_index_ == fresh.best_index_
    assert reused.best_score_ == fresh.best_score_
    assert type(reused.best_score_) is type(fresh.best_score_)
    np.testing.assert_array_equal(reused.predict(X), fresh.predict(X))
    expected = fresh.final_selection_results_["candidates"]
    actual = reused.final_selection_results_["candidates"]
    assert expected and len(expected) == len(actual)
    reuse_counts = {
        "discarded_only": (1, 2, 0),
        "discarded_only_with_new_fold": (2, 3, 0),
        "retained": (1, 1, 2),
        "new": (1, 1, 2),
        "discarded_and_new": (1, 1, 2),
        "new_mixed_small_scores": (2, 4, 0),
        "new_mixed_large_scores": (2, 6, 0),
    }
    for before, after in zip(expected, actual, strict=True):
        assert before["index"] == after["index"]
        assert before["score"] == after["score"]
        assert before["cv_scores"] == after["cv_scores"]
        assert all(type(a) is type(b) for a, b in zip(before["cv_scores"], after["cv_scores"]))
        assert (after["cv_calls"], after["fold_fits"], after["reused_folds"]) == reuse_counts[
            precision
        ]


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize(
    "case",
    [
        "subset",
        "new_integer",
        "new_boolean",
        "new_signed_group",
        "new_unsigned_and_boolean",
        "retained_fraction",
        "new_float",
    ],
)
def test_finalist_matches_original_scalar_types_after_dropping_a_fractional_fold(task, case):
    X, y = load_iris(return_X_y=True)
    rows = np.arange(len(X))
    search_folds = [(rows[:-size], rows[-size:]) for size in (12, 14, 16)]
    final_folds = [search_folds[2], search_folds[0]]
    if case != "subset":
        final_folds.insert(1, (rows[:-18], rows[-18:]))
    if case in ("new_signed_group", "new_unsigned_and_boolean"):
        final_folds[2:2] = [(rows[:-size], rows[-size:]) for size in (20, 22)]

    def score(estimator, X, y):
        complexity = estimator.max_depth if task == "parameters" else X.shape[1]
        if len(X) == 14 or (len(X) == 16 and case == "retained_fraction"):
            return complexity + 0.25
        if case == "new_signed_group" and len(X) in (18, 20, 22):
            return True if len(X) == 20 else -(2**54 + 2 * complexity + len(X) + 1)
        if case == "new_unsigned_and_boolean":
            return 2**63 + 2 * complexity + 1 if len(X) in (18, 22) else False
        if len(X) == 18 and case == "new_float":
            return float(complexity)
        if len(X) == 18 and case == "new_boolean":
            return True
        return complexity + len(X)

    model = search(
        task,
        scoring=score,
        generations=0,
        final_selection=True,
        final_selection_cv=final_folds,
        final_selection_top_k=2,
        error_score="raise",
    ).set_params(cv=search_folds)
    fresh = clone(model).fit(X, y)
    reused = model.set_params(final_selection="reuse_folds").fit(X, y)

    assert reused.best_index_ == fresh.best_index_
    assert reused.best_score_ == fresh.best_score_
    np.testing.assert_array_equal(reused.predict(X), fresh.predict(X))
    expected = fresh.final_selection_results_["candidates"]
    actual = reused.final_selection_results_["candidates"]
    assert expected and len(expected) == len(actual)
    counts = {
        "subset": (1, 2, 0),
        "new_integer": (2, 3, 0),
        "new_boolean": (2, 3, 0),
        "new_signed_group": (2, 5, 0),
        "new_unsigned_and_boolean": (2, 8, 0),
        "retained_fraction": (1, 1, 2),
        "new_float": (1, 1, 2),
    }
    for before, after in zip(expected, actual, strict=True):
        assert before["index"] == after["index"]
        assert before["score"] == after["score"]
        assert before["cv_scores"] == after["cv_scores"]
        assert all(type(a) is type(b) for a, b in zip(before["cv_scores"], after["cv_scores"]))
        assert (after["cv_calls"], after["fold_fits"], after["reused_folds"]) == counts[case]


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
def test_candidate_fit_failure_preserves_search_time_and_raises_the_original_error(
    task, engine, monkeypatch
):
    from sklearn_genetic import _base, genetic_search

    if engine != "deap":
        pytest.importorskip(engine)
    ticks = iter((1.0, 3.5))
    monkeypatch.setattr(_base, "perf_counter", ticks.__next__)
    evaluate_cv = genetic_search.cross_validate
    completed = []

    def count_cv(*args, **kwargs):
        result = evaluate_cv(*args, **kwargs)
        completed.append(len(result["fit_time"]))
        return result

    monkeypatch.setattr(genetic_search, "cross_validate", count_cv)
    model = _cv_failure_search(task, engine, error_score="raise")
    with pytest.raises(ValueError, match="^fixture estimator rejects this candidate$"):
        model.fit(*load_iris(return_X_y=True))
    assert model.search_time_ == 2.5
    assert model.search_interrupted_
    assert model.fit_counts_["search_cv_calls"] == len(completed)
    assert model.fit_counts_["search_fold_fits"] == sum(completed)
    with pytest.raises(NotFittedError):
        check_is_fitted(model)


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize(
    "engine, resume",
    [
        ("deap", False),
        pytest.param("scipy", False, marks=requires_scipy_backend),
        *((engine, False) for engine in BACKENDS),
        ("deap", True),
        ("pymoo", True),
    ],
)
def test_search_error_preserves_newly_recorded_cv_work(task, engine, resume, tmp_path, monkeypatch):
    from sklearn_genetic import genetic_search

    if engine != "deap":
        pytest.importorskip(engine)
    failure = RuntimeError("fixture interrupts after a completed generation")

    class FailAfterGeneration(BaseCallback):
        def on_step(self, record=None, logbook=None, estimator=None):
            raise failure

    X, y = load_iris(return_X_y=True)
    model = search(task, optimizer_backend=engine, use_cache=False, error_score="raise")
    callbacks = [FailAfterGeneration()]
    saved_calls = 0
    if resume:
        checkpoint = ModelCheckpoint(tmp_path / "interrupted.pkl")
        first = clone(model).fit(X, y, callbacks=[CVBudgetStopping(6), checkpoint])
        saved_calls = first.fit_stats_["cross_validate_calls"]
        callbacks.append(checkpoint)
    evaluate_cv = genetic_search.cross_validate
    observed = []

    def count_cv(*args, **kwargs):
        result = evaluate_cv(*args, **kwargs)
        observed.append(len(result["fit_time"]))
        return result

    monkeypatch.setattr(genetic_search, "cross_validate", count_cv)
    with pytest.raises(RuntimeError, match="fixture interrupts") as error:
        model.fit(X, y, callbacks=callbacks)
    assert error.value is failure
    assert observed
    assert model.search_interrupted_
    assert model.fit_stats_["cross_validate_calls"] == saved_calls + len(observed)
    assert model.fit_counts_["search_cv_calls"] == len(observed)
    assert model.fit_counts_["search_fold_fits"] == sum(observed)
    assert model.fit_counts_["finalist_cv_calls"] == model.fit_counts_["refit_fits"] == 0
    with pytest.raises(NotFittedError):
        check_is_fitted(model)

    model.fit(X, y, callbacks=CVBudgetStopping(6))
    assert not model.search_interrupted_
    assert model.fit_counts_["search_cv_calls"] == 6


def test_search_error_before_checkpoint_restore_has_no_fresh_work(tmp_path, monkeypatch):
    from sklearn_genetic import _backend_runtime

    pytest.importorskip("pymoo")
    X, y = load_iris(return_X_y=True)
    model = search("parameters", optimizer_backend="pymoo", use_cache=False)
    checkpoint = ModelCheckpoint(tmp_path / "before-restore.pkl")
    clone(model).fit(X, y, callbacks=[CVBudgetStopping(6), checkpoint])
    assert checkpoint.load()["fit_stats"]["cross_validate_calls"] == 6

    def fail(*args, **kwargs):
        raise RuntimeError("fixture interrupts setup before restore")

    monkeypatch.setattr(_backend_runtime, "run_optimizer", fail)
    with pytest.raises(RuntimeError, match="fixture interrupts setup"):
        model.fit(X, y, callbacks=checkpoint)
    assert model.search_interrupted_
    assert model.fit_counts_["search_cv_calls"] == model.fit_counts_["search_fold_fits"] == 0


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize(
    "engine, resume",
    [
        ("deap", False),
        pytest.param("scipy", False, marks=requires_scipy_backend),
        *((engine, False) for engine in BACKENDS),
        ("deap", True),
        ("pymoo", True),
    ],
)
def test_search_error_counts_returned_cv_from_an_unrecorded_batch(
    task, engine, resume, tmp_path, monkeypatch
):
    from sklearn_genetic import genetic_search

    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    model = search(task, optimizer_backend=engine, use_cache=False, error_score="raise")
    callbacks = []
    saved_calls = 0
    if resume:
        checkpoint = ModelCheckpoint(tmp_path / "interrupted-batch.pkl")
        clone(model).fit(X, y, callbacks=[CVBudgetStopping(6), checkpoint])
        saved_calls = checkpoint.load()["logbook"][-1]["cross_validate_calls"]
        snapshot = checkpoint.checkpoint_path.read_bytes()
        callbacks = [CVBudgetStopping(6), checkpoint]

    cross_validate = genetic_search.cross_validate
    observed = []
    failure = RuntimeError("fixture interrupts the third CV call")

    def fail_third_cv(*args, **kwargs):
        if len(observed) == 2:
            raise failure
        result = cross_validate(*args, **kwargs)
        observed.append(len(result["fit_time"]))
        return result

    monkeypatch.setattr(genetic_search, "cross_validate", fail_third_cv)
    with pytest.raises(RuntimeError, match="fixture interrupts the third CV call") as error:
        model.fit(X, y, callbacks=callbacks)
    assert error.value is failure
    assert model.search_interrupted_
    assert observed == [2, 2]
    assert model.fit_counts_["search_cv_calls"] == len(observed)
    assert model.fit_counts_["search_fold_fits"] == sum(observed)
    assert model.fit_stats_["cross_validate_calls"] == saved_calls
    assert "_unrecorded_search_cv_calls" not in vars(model)
    if resume:
        assert checkpoint.checkpoint_path.read_bytes() == snapshot


def test_search_error_counts_results_delivered_before_a_threaded_candidate_fails(monkeypatch):
    from joblib import parallel_backend
    from sklearn_genetic import genetic_search

    def optimize(problem, evaluate, report):
        evaluate([(1, 1, 0, 0), (0, 1, 1, 0), (1, 0, 0, 0)])

    cross_validate = genetic_search.cross_validate

    def reject_single_feature(estimator, X, *args, **kwargs):
        if X.shape[1] == 1:
            raise RuntimeError("fixture interrupts the third candidate")
        return cross_validate(estimator, X, *args, **kwargs)

    monkeypatch.setattr(genetic_search, "cross_validate", reject_single_feature)
    model = search(
        "features",
        optimizer_backend=optimize,
        population_size=3,
        generations=0,
        n_jobs=2,
        parallel_backend="population",
        error_score="raise",
    )
    with (
        parallel_backend("threading"),
        pytest.raises(RuntimeError, match="fixture interrupts the third candidate"),
    ):
        model.fit(*load_iris(return_X_y=True))

    assert model.search_interrupted_
    assert model.fit_counts_["search_cv_calls"] == 2
    assert model.fit_counts_["search_fold_fits"] == 4
    assert model.fit_stats_["cross_validate_calls"] == 0


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_interrupted_parallel_search_counts_cv_completed_out_of_order(
    task, error_type, monkeypatch
):
    from joblib import parallel_backend
    from sklearn_genetic import genetic_search

    X, y = load_iris(return_X_y=True)
    candidates = (
        [(1,), (2,), (3,)] if task == "parameters" else [(1, 1, 0, 0), (1, 0, 0, 0), (0, 1, 1, 0)]
    )
    failure = error_type("fixture interrupts the middle candidate")
    later_finished = Event()
    cross_validate = genetic_search.cross_validate

    def optimize(problem, evaluate, report):
        evaluate(candidates)

    def interrupt_middle(estimator, values, *args, **kwargs):
        if task == "parameters":
            fail, later = estimator.max_depth == 2, estimator.max_depth == 3
        else:
            fail = values.shape[1] == 1
            later = np.array_equal(values[:, 0], X[:, 1])
        if fail:
            assert later_finished.wait(timeout=10), "the later CV call did not finish"
            raise failure
        result = cross_validate(estimator, values, *args, **kwargs)
        if later:
            later_finished.set()
        return result

    monkeypatch.setattr(genetic_search, "cross_validate", interrupt_middle)
    model = search(
        task,
        **({"param_grid": {"max_depth": Integer(1, 3)}} if task == "parameters" else {}),
        optimizer_backend=optimize,
        population_size=3,
        generations=0,
        n_jobs=2,
        parallel_backend="population",
        error_score="raise",
    )
    with parallel_backend("threading"), pytest.raises(error_type) as interrupted:
        model.fit(X, y)

    assert interrupted.value is failure
    assert model.search_interrupted_
    assert model.fit_counts_["search_cv_calls"] == 2
    assert model.fit_counts_["search_fold_fits"] == 4
    assert model.fit_stats_["cross_validate_calls"] == 0
    assert "_population_pool" not in vars(model)
    with pytest.raises(NotFittedError):
        check_is_fitted(model)


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("failure_source", ["cache_key", "timer"])
def test_later_dispatch_preparation_error_preserves_confirmed_cv_work(
    task, n_jobs, failure_source, monkeypatch
):
    from joblib import parallel_backend
    from sklearn_genetic import genetic_search

    if task == "parameters":
        candidates = [(1,), (2,), (3,), (4,)]
        options = {"param_grid": {"max_depth": Integer(1, 4)}}
    else:
        candidates = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
        options = {}
    failure = RuntimeError(f"fixture interrupts {failure_source} preparation")
    observed = []
    cross_validate = genetic_search.cross_validate

    def optimize(problem, evaluate, report):
        evaluate(candidates)

    def count_cv(*args, **kwargs):
        result = cross_validate(*args, **kwargs)
        observed.append(len(result["fit_time"]))
        return result

    class FailingTimer(TimerStopping):
        def _time_exceeded(self):
            raise failure

    monkeypatch.setattr(genetic_search, "cross_validate", count_cv)
    model = search(
        task,
        **options,
        optimizer_backend=optimize,
        population_size=4,
        generations=0,
        n_jobs=n_jobs,
        parallel_backend="population",
        error_score="raise",
    )
    if failure_source == "cache_key":
        original_key = model._individual_key

        def fail_later_key(row):
            if tuple(row) == candidates[n_jobs + 1]:
                raise failure
            return original_key(row)

        monkeypatch.setattr(model, "_individual_key", fail_later_key)
        timer = TimerStopping(60)
    else:
        timer = FailingTimer(60)

    with parallel_backend("threading"), pytest.raises(RuntimeError) as interrupted:
        model.fit(*load_iris(return_X_y=True), callbacks=timer)

    assert interrupted.value is failure
    assert observed == [2] * n_jobs
    assert model.search_interrupted_
    assert model.fit_counts_["search_cv_calls"] == len(observed)
    assert model.fit_counts_["search_fold_fits"] == sum(observed)
    assert model.fit_stats_["cross_validate_calls"] == 0
    assert not model._candidate_rows
    assert "_population_pool" not in vars(model)


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize("use_cache", [False, True])
@pytest.mark.parametrize("publication", ["rows", "counters"])
def test_result_publication_error_keeps_all_completed_cv_counts(
    task, use_cache, publication, monkeypatch
):
    from sklearn_genetic import evaluation, genetic_search

    if task == "parameters":
        first, second, last = (1,), (2,), (3,)
        options = {"param_grid": {"max_depth": Integer(1, 3)}}
    else:
        first, second, last = (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)
        options = {}
    failure = RuntimeError("fixture interrupts result publication")
    observed = []
    cross_validate = genetic_search.cross_validate

    def optimize(problem, evaluate, report):
        evaluate([first, second, first, last])

    def count_cv(*args, **kwargs):
        result = cross_validate(*args, **kwargs)
        observed.append(len(result["fit_time"]))
        return result

    monkeypatch.setattr(genetic_search, "cross_validate", count_cv)
    if publication == "rows":
        record_candidate = evaluation.record_candidate

        def fail_while_recording(estimator, record):
            if estimator._candidate_rows:
                raise failure
            return record_candidate(estimator, record)

        monkeypatch.setattr(evaluation, "record_candidate", fail_while_recording)
    else:
        record_stats = evaluation.record_fit_stats

        def fail_after_counting(*args, **kwargs):
            record_stats(*args, **kwargs)
            raise failure

        monkeypatch.setattr(evaluation, "record_fit_stats", fail_after_counting)

    model = search(
        task,
        **options,
        optimizer_backend=optimize,
        population_size=4,
        generations=0,
        use_cache=use_cache,
        error_score="raise",
    )
    with pytest.raises(RuntimeError) as interrupted:
        model.fit(*load_iris(return_X_y=True))

    assert interrupted.value is failure
    assert observed == [2] * (3 if use_cache else 4)
    assert model.search_interrupted_
    assert model.fit_counts_["search_cv_calls"] == len(observed)
    assert model.fit_counts_["search_fold_fits"] == sum(observed)
    assert model.fit_stats_["cross_validate_calls"] == (
        0 if publication == "rows" else len(observed)
    )
    assert len(model._candidate_rows) == (1 if publication == "rows" else 4)


@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
@pytest.mark.parametrize("initializer", ["smart", "random"])
def test_numeric_category_types_keep_distinct_scores_and_finalists(engine, initializer):
    from sklearn.datasets import load_wine
    from sklearn.tree import DecisionTreeClassifier

    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_wine(return_X_y=True)
    model = GASearchCV(
        DecisionTreeClassifier(random_state=7),
        param_grid={"max_features": Categorical([1, 1.0]), "max_depth": Integer(4, 4)},
        warm_start_configs=[
            {"max_features": 1, "max_depth": 4},
            {"max_features": 1.0, "max_depth": 4},
        ],
        optimizer_backend=engine,
        population_initializer=initializer,
        population_size=6,
        cv=3,
        generations=0,
        n_jobs=1,
        scoring="accuracy",
        error_score="raise",
        final_selection="reuse_folds",
        final_selection_top_k=2,
        random_state=7,
        verbose=False,
    ).fit(X, y)
    by_type = {
        type(params["max_features"]): score
        for params, score in zip(
            model.cv_results_["params"], model.cv_results_["mean_test_score"], strict=True
        )
    }
    assert set(by_type) == {int, float}
    assert by_type[float] > by_type[int]
    assert model.fit_counts_["search_cv_calls"] == 2
    assert model.fit_counts_["finalist_reused_folds"] == 6
    assert type(model.best_params_["max_features"]) is float
    assert {
        type(row["params"]["max_features"]) for row in model.final_selection_results_["candidates"]
    } == {int, float}
    if engine == "deap":
        assert model.history["unique_individuals"][0] == 2


@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
def test_single_pipeline_parameter_covers_finite_estimator_choices(engine):
    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    models = [DecisionTreeClassifier(max_depth=depth, random_state=0) for depth in (1, 3)] + [
        KNeighborsClassifier(n_neighbors=neighbors) for neighbors in (3, 7)
    ]
    pipeline = Pipeline([("scale", StandardScaler()), ("model", models[0])])
    cv = list(StratifiedKFold(2, shuffle=True, random_state=0).split(X, y))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted = GASearchCV(
            pipeline,
            param_grid={"model": Categorical(models)},
            warm_start_configs=[{"model": model} for model in models],
            optimizer_backend=engine,
            cv=cv,
            population_size=6,
            generations=2,
            n_jobs=1,
            verbose=False,
            random_state=0,
            error_score="raise",
        ).fit(X, y)
    reference = GridSearchCV(pipeline, {"model": models}, cv=cv, error_score="raise").fit(X, y)

    assert fitted.search_space_exhausted_
    assert fitted.fit_counts_["search_cv_calls"] == len(models)
    assert fitted.fit_counts_["search_fold_fits"] == len(cv) * len(models)
    assert fitted.best_score_ == reference.best_score_
    assert not any("only one parameter was provided" in str(item.message) for item in caught)


def test_categorical_identity_distinguishes_types_inside_nested_values():
    from types import SimpleNamespace

    from sklearn_genetic.optimizer_control import finite_search_space_size
    from sklearn_genetic.space import Space

    dimension = Categorical([{"values": [1]}, {"values": [1.0]}, {"values": [1]}])
    space = Space({"configuration": dimension})
    assert dimension._choice_index({"values": [1.0]}) == 1
    assert space._candidate_key([{"values": [1]}]) != space._candidate_key([{"values": [1.0]}])
    assert space._candidate_key([{"values": [1.0]}]) == space._candidate_key([dimension.choices[1]])
    assert finite_search_space_size(SimpleNamespace(space=space, use_cache=True), 3) == 2


@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
def test_finite_coverage_counts_categorical_nan_by_its_cache_key(engine):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    missing_values = [float("nan"), -float("nan"), -999.0, float("nan")]
    marker = "simpleimputer__missing_values"
    leaf = "decisiontreeclassifier__min_samples_leaf"
    model = GASearchCV(
        make_pipeline(SimpleImputer(), DecisionTreeClassifier(random_state=19)),
        param_grid={marker: Categorical(missing_values), leaf: Integer(1, 2)},
        warm_start_configs=[
            {marker: value, leaf: size} for value in missing_values[:3] for size in (1, 2)
        ],
        optimizer_backend=engine,
        population_size=8,
        generations=4,
        cv=2,
        n_jobs=1,
        scoring="accuracy",
        random_state=19,
        use_cache=True,
        error_score="raise",
        verbose=False,
    ).fit(X, y)

    assert model.search_space_exhausted_
    assert model.history["gen"] == [0]
    assert model.fit_counts_["search_cv_calls"] == len(model.fitness_cache) == 6


@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
@pytest.mark.parametrize("use_cache", [False, True])
def test_missing_value_categories_remain_distinct_in_pipeline(engine, use_cache):
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    X[0, 0] = np.nan
    parameter = "simpleimputer__missing_values"
    missing_values = [pd.NA, np.nan]
    model = GASearchCV(
        make_pipeline(
            SimpleImputer(missing_values=float("nan")), DecisionTreeClassifier(random_state=7)
        ),
        param_grid={parameter: Categorical(missing_values)},
        warm_start_configs=[{parameter: value} for value in missing_values],
        optimizer_backend=engine,
        population_size=6,
        generations=1,
        cv=2,
        scoring="accuracy",
        n_jobs=1,
        error_score="raise",
        random_state=7,
        verbose=False,
        use_cache=use_cache,
        **({"fitness_sharing": True, "local_search": True} if engine == "deap" else {}),
    ).fit(X, y)

    types = {type(params[parameter]) for params in model.cv_results_["params"]}
    assert types == {type(pd.NA), float}
    assert not model.search_space_exhausted_
    assert np.isfinite(model.cv_results_["mean_test_score"]).all()
    assert model.predict(X).shape == y.shape
    if engine == "deap":
        assert model.fit_stats_["local_refinement_candidates"] > 0
        assert any(model.history["fitness_sharing_applied"])
    if use_cache:
        assert model.fit_counts_["search_cv_calls"] == len(missing_values)
        assert model.fit_stats_["cache_hits"] + model.fit_stats_["duplicate_candidates"] > 0
    else:
        assert model.fit_counts_["search_cv_calls"] == len(model.cv_results_["params"])


@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
@pytest.mark.parametrize("nan_type", [float, np.float32])
def test_nan_category_checkpoint_preserves_cached_evaluations(
    engine, nan_type, tmp_path, monkeypatch
):
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    from sklearn_genetic import genetic_search

    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    X[::7, 0] = np.nan
    marker, depth = "simpleimputer__missing_values", "decisiontreeclassifier__max_depth"
    missing_values = [pd.NA, nan_type("nan")]
    warm_values = missing_values if nan_type is float else [pd.NA, nan_type("nan")]
    model = GASearchCV(
        make_pipeline(
            SimpleImputer(missing_values=float("nan")), DecisionTreeClassifier(random_state=7)
        ),
        param_grid={marker: Categorical(missing_values), depth: Integer(2, 3)},
        warm_start_configs=[{marker: value, depth: d} for value in warm_values for d in (2, 3)],
        optimizer_backend=engine,
        population_size=8,
        generations=2,
        cv=2,
        scoring="accuracy",
        n_jobs=1,
        error_score="raise",
        random_state=7,
        verbose=False,
    )
    checkpoint = ModelCheckpoint(tmp_path / "search.pkl")
    initial = clone(model).fit(X, y, callbacks=[ThresholdStopping(0), checkpoint])
    assert initial.fit_counts_["search_cv_calls"] == 4
    monkeypatch.setattr(
        genetic_search, "cross_validate", lambda *a, **kw: pytest.fail("unexpected CV on resume")
    )

    def resume():
        fitted = clone(model).fit(X, y, callbacks=checkpoint)
        assert fitted.fit_counts_["search_cv_calls"] == 0
        assert fitted.fit_stats_["cross_validate_calls"] == 4
        assert {type(row[marker]) for row in fitted.cv_results_["params"]} == {
            type(pd.NA),
            nan_type,
        }
        assert fitted.best_index_ == initial.best_index_
        assert fitted.best_score_ == initial.best_score_
        assert type(fitted.best_params_[marker]) is type(initial.best_params_[marker])
        np.testing.assert_array_equal(fitted.predict(X), initial.predict(X))

    saved = checkpoint.load()
    resume()

    # Older checkpoints keyed these cached values by the NaN object itself.
    saved.pop("stable_nan_categories")
    state = saved["runtime_state"] if engine == "deap" else saved
    state["fitness_cache"] = {
        tuple(
            (name, item["current_generation_params"][name]) for name in sorted((marker, depth))
        ): item
        for item in state["fitness_cache"].values()
    }
    checkpoint.checkpoint_path.write_bytes(pickle.dumps(saved))
    resume()


@pytest.mark.parametrize("engine", ["deap", "pymoo"])
@pytest.mark.parametrize("ambiguous", [False, True])
def test_categorical_checkpoint_rejects_old_ambiguous_cache_only(engine, ambiguous, tmp_path):
    from sklearn.tree import DecisionTreeClassifier

    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    choices = [1, 1.0] if ambiguous else [1, 2]
    model = GASearchCV(
        DecisionTreeClassifier(random_state=7),
        param_grid={"max_features": Categorical(choices), "max_depth": Integer(2, 5)},
        optimizer_backend=engine,
        population_size=4,
        generations=2,
        cv=2,
        n_jobs=1,
        random_state=7,
        verbose=False,
    )
    checkpoint = ModelCheckpoint(tmp_path / "categories.pkl")
    first = clone(model).fit(X, y, callbacks=[ThresholdStopping(0), checkpoint])
    original = checkpoint.load()
    assert original["typed_categorical_identity"]
    continuation = clone(model).fit(X, y, callbacks=checkpoint)
    assert len(continuation.history["gen"]) > len(first.history["gen"])

    original.pop("typed_categorical_identity")
    checkpoint.checkpoint_path.write_bytes(pickle.dumps(original))
    if ambiguous:
        with pytest.raises(ValueError, match="merged categorical scores"):
            model.fit(X, y, callbacks=checkpoint)
        assert "typed_categorical_identity" not in checkpoint.load()
    else:
        resumed = model.fit(X, y, callbacks=checkpoint)
        assert resumed.history == continuation.history
        np.testing.assert_equal(resumed.predict(X), continuation.predict(X))


class _PredictOnlyOutlier(OutlierMixin, BaseEstimator):
    fit_sizes = []

    def __init__(self, offset=0.0, column=0):
        self.offset = offset
        self.column = column

    def fit(self, X, y=None):
        self.fit_sizes.append(len(X))
        self.threshold_ = X[:, self.column].mean() + self.offset
        return self

    def predict(self, X):
        return np.where(X[:, self.column] >= self.threshold_, 1, -1).tolist()

    def fit_predict(self, X, y=None):
        return np.asarray(self.fit(X, y).predict(X))


class _ContinuousOutlier(_PredictOnlyOutlier):
    def score_samples(self, X):
        return X[:, self.column] - self.threshold_


def _counted_outlier_search(task, detector, backend="deap", generations=0, **options):
    model = _outlier_search(task, 3, backend, generations)
    if task == "parameters":
        model.set_params(param_grid={"offset": Continuous(-0.5, 0.5), "column": Integer(0, 3)})
    return model.set_params(estimator=detector, **options)


@pytest.mark.parametrize("task", ["features", "parameters"])
def test_default_outlier_scorer_never_fits_on_validation_rows(task):
    X, _ = load_iris(return_X_y=True)
    _PredictOnlyOutlier.fit_sizes.clear()
    model = _counted_outlier_search(
        task, _PredictOnlyOutlier(), final_selection=True, return_train_score=True
    )
    model.fit(X)
    counts = model.fit_counts_
    assert _PredictOnlyOutlier.fit_sizes == [100] * (
        counts["search_fold_fits"] + counts["finalist_fold_fits"]
    ) + [150]
    test = X[::7]
    expected = np.mean(np.asarray(model.predict(test)) == 1)
    assert model.score(test) == expected
    assert _PredictOnlyOutlier.fit_sizes[-1] == 150


@pytest.mark.parametrize("task", ["features", "parameters"])
def test_default_outlier_scorer_explains_unsupported_lof_mode_before_fold_fits(task, monkeypatch):
    X, _ = load_iris(return_X_y=True)
    model = _outlier_search(task, 3).set_params(estimator=LocalOutlierFactor(), error_score=np.nan)
    if task == "parameters":
        model.set_params(param_grid={"n_neighbors": Integer(3, 6), "leaf_size": Integer(20, 25)})
    monkeypatch.setattr(
        LocalOutlierFactor, "fit", lambda *a, **kw: pytest.fail("unexpected detector fit")
    )
    with pytest.raises(ValueError, match="novelty=True"):
        model.fit(X)


def test_outlier_search_can_enable_novelty_in_the_parameter_grid():
    X, _ = load_iris(return_X_y=True)
    model = _outlier_search("parameters", 3).set_params(
        estimator=LocalOutlierFactor(),
        param_grid={"novelty": Categorical([True]), "n_neighbors": Integer(3, 6)},
    )
    model.fit(X)
    assert model.best_estimator_.novelty
    assert np.isfinite(model.best_score_)


@pytest.mark.parametrize("task", ["features", "parameters"])
@pytest.mark.parametrize("engine", ["deap", "pymoo"])
@pytest.mark.parametrize("detector_type", [_PredictOnlyOutlier, _ContinuousOutlier])
def test_older_outlier_checkpoint_resumes_only_if_its_scores_are_compatible(
    task, engine, detector_type, tmp_path
):
    if engine != "deap":
        pytest.importorskip(engine)
    X, _ = load_iris(return_X_y=True)
    model = _counted_outlier_search(task, detector_type(), engine, generations=2)
    checkpoint = ModelCheckpoint(tmp_path / "outlier-history.pkl")
    first = clone(model).fit(X, callbacks=[ThresholdStopping(-1), checkpoint])
    saved = checkpoint.load()
    if detector_type is _PredictOnlyOutlier:
        restored = clone(model).fit(X, callbacks=checkpoint)
        assert len(restored.history["gen"]) > len(first.history["gen"])
    saved.pop("outlier_scoring_uses_fitted_detector")
    checkpoint.checkpoint_path.write_bytes(pickle.dumps(saved))

    if detector_type is _PredictOnlyOutlier:
        with pytest.raises(ValueError, match="Use a new checkpoint path"):
            model.fit(X, callbacks=checkpoint)
        assert "outlier_scoring_uses_fitted_detector" not in checkpoint.load()
    else:
        restored = model.fit(X, callbacks=checkpoint)
        assert len(restored.history["gen"]) > len(first.history["gen"])
        np.testing.assert_equal(
            restored.cv_results_["mean_test_score"][: len(first.cv_results_["mean_test_score"])],
            first.cv_results_["mean_test_score"],
        )


@pytest.mark.parametrize("task", ["features", "parameters"])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
def test_multimetric_search_without_refit_uses_the_explicit_objective(task, engine):
    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    model = search(
        task,
        optimizer_backend=engine,
        refit="f1",
        scoring={"accuracy": "accuracy", "f1": "f1_macro"},
        error_score="raise",
    )
    reference = clone(model).fit(X, y)
    measured = model.set_params(refit=False, objective_metric="f1").fit(X, y)

    assert not hasattr(measured, "best_estimator_")
    assert measured.best_index_ == reference.best_index_
    assert measured.best_score_ == reference.best_score_
    if task == "features":
        np.testing.assert_array_equal(
            measured.cv_results_["features"][measured.best_index_], measured.support_
        )
    else:
        assert measured.best_params_ == reference.best_params_
    assert measured.fit_counts_ == {**reference.fit_counts_, "refit_fits": 0}
    assert measured.history == reference.history
    for column in (
        "mean_test_accuracy",
        "mean_test_f1",
        "rank_test_f1",
        "features" if task == "features" else "params",
    ):
        np.testing.assert_equal(measured.cv_results_[column], reference.cv_results_[column])
    assert measured.cv_results_dataframe()["rank_test_f1"].is_monotonic_increasing


@pytest.mark.parametrize("task", ["features", "parameters"])
def test_single_metric_search_without_refit_reports_the_minimizing_candidate(task):
    X, y = load_iris(return_X_y=True)
    model = search(task, scoring="accuracy", criteria="min", refit=False, final_selection=True).fit(
        X, y
    )
    results = model.cv_results_
    assert model.best_score_ == min(results["mean_test_score"])
    assert model.best_score_ == results["mean_test_score"][model.best_index_]
    if task == "features":
        np.testing.assert_array_equal(results["features"][model.best_index_], model.support_)
    else:
        assert model.best_params_ == results["params"][model.best_index_]
    assert model.fit_counts_["refit_fits"] == model.fit_counts_["finalist_cv_calls"] == 0
    assert not hasattr(model, "best_estimator_")
    assert not hasattr(model, "final_selection_results_")


@pytest.mark.parametrize("task", ["features", "parameters"])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
def test_search_only_reports_actual_generations_after_early_stop(task, engine):
    from sklearn.exceptions import NotFittedError

    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    search_only = search(task, optimizer_backend=engine, refit=False, scoring="accuracy")
    search_only.fit(X, y, callbacks=ThresholdStopping(0))
    records = list(search_only)
    assert len(search_only) == len(records) == 1
    assert records == [search_only[-1]] == [search_only[0]]
    assert records[0]["gen"] == 0
    assert records[0]["cross_validate_calls"] == search_only.fit_counts_["search_cv_calls"]
    assert search_only.fit_counts_["refit_fits"] == 0
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        search_only.fit(X, y[:-1])
    with pytest.raises(NotFittedError):
        search_only[0]
    with pytest.raises(NotFittedError):
        search_only.cv_results_dataframe()


@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
def test_spread_initializer_orders_the_same_initial_feature_masks(engine):
    from sklearn.datasets import load_wine

    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_wine(return_X_y=True)
    base = search(
        "features",
        optimizer_backend=engine,
        population_size=8,
        refit=False,
        error_score="raise",
    ).set_params(max_features=7)
    smart = clone(base).fit(X, y, callbacks=CVBudgetStopping(8))
    spread = (
        clone(base)
        .set_params(population_initializer="spread")
        .fit(X, y, callbacks=CVBudgetStopping(8))
    )
    capped = (
        clone(base)
        .set_params(population_initializer="spread")
        .fit(X, y, callbacks=CVBudgetStopping(4))
    )
    assert capped.fit_counts_["search_cv_calls"] == 4
    assert spread.fit_counts_["search_cv_calls"] == 8
    np.testing.assert_array_equal(capped.cv_results_["n_features"], [4, 1, 7, 2])
    np.testing.assert_array_equal(
        capped.cv_results_["features"], spread.cv_results_["features"][:4]
    )
    observed = []
    for model in (smart, spread):
        results = model.cv_results_
        observed.append(
            sorted(zip(map(tuple, results["features"]), results["mean_test_score"], strict=True))
        )
    assert observed[0] == observed[1]


def test_deap_checkpoint_restores_the_spread_initializer(tmp_path):
    X, y = load_iris(return_X_y=True)
    model = search(
        "features",
        population_size=8,
        refit=False,
        population_config=PopulationConfig(initializer="spread"),
    )
    checkpoint = ModelCheckpoint(tmp_path / "spread.pkl")
    clone(model).fit(X, y, callbacks=[CVBudgetStopping(6), checkpoint])
    assert checkpoint.load()["estimator_state"]["population_initializer"] == "spread"
    resumed = (
        clone(model)
        .set_params(population_config=None, population_initializer="smart")
        .fit(X, y, callbacks=[CVBudgetStopping(4), checkpoint])
    )
    assert resumed.population_initializer == "spread"


@pytest.mark.parametrize("task, initializer", [("parameters", "random"), ("features", "spread")])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
def test_replacing_population_config_updates_fresh_and_reused_searches(task, initializer, engine):
    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    model = search(task, optimizer_backend=engine, population_size=8)
    if task == "features":
        model.set_params(max_features=4)
    fresh = clone(model).set_params(population_initializer=initializer)
    reference = fresh.fit(X, y, callbacks=CVBudgetStopping(4))

    model.fit(X, y, callbacks=CVBudgetStopping(4))
    config = PopulationConfig(initializer=initializer)
    model.set_params(population_config=config, population_initializer="smart")
    assert model.get_params()["population_config"] is config
    for updated in (model, clone(model)):
        observed = updated.fit(X, y, callbacks=CVBudgetStopping(4))
        assert observed.fit_counts_ == reference.fit_counts_
        candidates = "params" if task == "parameters" else "features"
        np.testing.assert_array_equal(
            observed.cv_results_[candidates], reference.cv_results_[candidates]
        )
        np.testing.assert_array_equal(observed.predict(X), reference.predict(X))

    # The grouped strategy retains precedence until the configuration is detached.
    model.set_params(population_initializer="smart")
    assert clone(model).get_params()["population_initializer"] == initializer
    model.set_params(population_config=None, population_initializer="smart")
    assert clone(model).get_params()["population_initializer"] == "smart"


@pytest.mark.parametrize("task, initializer", [("parameters", "spread"), ("features", "invalid")])
def test_replaced_population_config_is_validated_at_fit_time(task, initializer):
    model = search(task).set_params(population_config=PopulationConfig(initializer=initializer))
    with pytest.raises(ValueError, match="population_initializer"):
        model.fit(*load_iris(return_X_y=True))


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
def test_replacing_runtime_config_applies_to_fits_and_clones(task, engine):
    from joblib import parallel_backend

    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    config = RuntimeConfig(
        n_jobs=2,
        pre_dispatch=1,
        error_score="raise",
        return_train_score=True,
        use_cache=False,
        parallel_backend="population",
        verbose=False,
    )
    options = dict(optimizer_backend=engine, population_size=8)
    reference = search(task, runtime_config=config, **options)
    reused = search(task, **options).fit(X, y, callbacks=CVBudgetStopping(4))
    assert "mean_train_score" not in reused.cv_results_
    reused.set_params(runtime_config=config, n_jobs=1, use_cache=True)
    assert reused.get_params()["runtime_config"] is config

    candidates = "params" if task == "parameters" else "features"
    with parallel_backend("threading"):
        reference.fit(X, y, callbacks=CVBudgetStopping(4))
        for model in (reused, clone(reused)):
            model.fit(X, y, callbacks=CVBudgetStopping(4))
            assert model.fit_counts_ == reference.fit_counts_
            assert model.fit_stats_ == reference.fit_stats_
            assert model.fit_stats_["cache_hits"] == 0
            for key in (candidates, "mean_train_score", "mean_test_score"):
                np.testing.assert_array_equal(model.cv_results_[key], reference.cv_results_[key])
            np.testing.assert_array_equal(model.predict(X), reference.predict(X))

    reused.set_params(return_train_score=False)
    assert clone(reused).return_train_score is True
    reused.set_params(runtime_config=None, n_jobs=1, return_train_score=False)
    assert clone(reused).n_jobs == 1
    assert reused.fit(X, y, callbacks=CVBudgetStopping(4)).return_train_score is False
    assert "mean_train_score" not in reused.cv_results_


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize("field", ["parallel_backend", "error_score"])
def test_replaced_runtime_config_is_validated_before_cv(task, field, monkeypatch):
    from sklearn_genetic import genetic_search

    def unexpected_cv(*args, **kwargs):
        pytest.fail("invalid runtime controls reached cross-validation")

    monkeypatch.setattr(genetic_search, "cross_validate", unexpected_cv)
    config = RuntimeConfig(**{field: "invalid"})
    model = search(task).set_params(runtime_config=config)
    with pytest.raises(ValueError, match=field):
        model.fit(*load_iris(return_X_y=True))


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
def test_replacing_evolution_config_applies_to_fits_and_clones(task, engine):
    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    options = dict(optimizer_backend=engine, use_cache=False, population_size=8)
    if task == "features":
        options["population_initializer"] = "random"
    config = EvolutionConfig(
        population_size=np.int64(8),
        generations=1,
        keep_top_k=2,
        criteria="min",
        crossover_probability=0.6 if engine in {"deap", "scipy", "pymoo"} else 0.8,
    )
    reference = search(task, evolution_config=config, **options).fit(X, y)
    reused = search(task, **options).fit(X, y)
    reused.set_params(evolution_config=config, criteria="max")
    candidates = "params" if task == "parameters" else "features"
    for model in (reused, clone(reused)):
        model.fit(X, y)
        assert model.fit_counts_ == reference.fit_counts_
        assert model.history == reference.history
        np.testing.assert_array_equal(
            model.cv_results_[candidates], reference.cv_results_[candidates]
        )
        np.testing.assert_array_equal(model.predict(X), reference.predict(X))

    reused.set_params(generations=0)
    assert clone(reused).generations == config.generations
    reused.set_params(evolution_config=None, generations=0)
    assert clone(reused).generations == 0


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize("grouped", [False, True])
def test_replacing_probability_schedules_applies_to_deap_fits_and_clones(task, grouped):
    from sklearn_genetic.schedules.schedulers import ExponentialAdapter, InverseAdapter

    X, y = load_iris(return_X_y=True)
    probabilities = dict(
        crossover_probability=ExponentialAdapter(0.55, 0.2, 0.3),
        mutation_probability=InverseAdapter(0.35, 0.1, 0.4),
    )
    options = (
        {"evolution_config": EvolutionConfig(population_size=6, generations=2, **probabilities)}
        if grouped
        else probabilities
    )
    reference = search(task, use_cache=False, **clone(options, safe=False)).fit(X, y)
    reused = search(task, use_cache=False).fit(X, y).set_params(**options)
    reproduced = clone(reused)
    if grouped:
        assert reproduced.crossover_probability is reproduced.evolution_config.crossover_probability
        assert reproduced.crossover_probability is not reused.crossover_probability
        assert reproduced.crossover_adapter is reproduced.crossover_probability
    for model in (reused, reproduced):
        model.fit(X, y)
        assert model.history == reference.history
        assert model.crossover_adapter.state_dict() == reference.crossover_adapter.state_dict()
        assert model.mutation_adapter.state_dict() == reference.mutation_adapter.state_dict()
    assert reproduced.crossover_adapter.current_step > 0


@pytest.mark.parametrize("task", ["parameters", "features"])
def test_deap_resume_rebuilds_adapters_from_checkpoint(task, tmp_path):
    from sklearn_genetic.schedules.schedulers import ExponentialAdapter, InverseAdapter

    X, y = load_iris(return_X_y=True)
    config = {
        "crossover_probability": ExponentialAdapter(0.55, 0.2, 0.3),
        "mutation_probability": InverseAdapter(0.35, 0.1, 0.4),
    }
    checkpoint = ModelCheckpoint(tmp_path / "schedules.pkl")
    search(task, use_cache=False, **config).fit(X, y, callbacks=checkpoint)
    sibling = tmp_path / "reference.pkl"
    sibling.write_bytes((tmp_path / "schedules.pkl").read_bytes())
    reference = search(task, use_cache=False, **config).fit(
        X, y, callbacks=ModelCheckpoint(sibling)
    )
    observed = search(task, use_cache=False).fit(X, y, callbacks=checkpoint)
    assert observed.crossover_adapter is observed.crossover_probability
    assert observed.mutation_adapter is observed.mutation_probability
    assert observed.crossover_adapter.state_dict() == reference.crossover_adapter.state_dict()
    assert observed.mutation_adapter.state_dict() == reference.mutation_adapter.state_dict()
    assert observed.history == reference.history
    np.testing.assert_array_equal(observed.predict(X), reference.predict(X))


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
def test_replacing_optimization_config_applies_grouped_finalist_controls(task, engine):
    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    options = dict(optimizer_backend=engine, population_size=8, error_score="raise")
    old_config = OptimizationConfig(
        final_selection=True, final_selection_top_k=1, final_selection_cv=2
    )
    new_config = OptimizationConfig(
        final_selection=True, final_selection_top_k=np.int64(2), final_selection_cv=3
    )
    reference = search(task, optimization_config=new_config, **options)
    reference.fit(X, y, callbacks=CVBudgetStopping(6))
    model = search(task, optimization_config=old_config, **options)
    model.fit(X, y, callbacks=CVBudgetStopping(6))
    assert model.final_selection_results_["cv"] == 2
    model.set_params(optimization_config=new_config, final_selection=False)
    assert model.get_params()["optimization_config"] is new_config
    for candidate in (model, clone(model)):
        candidate.fit(X, y, callbacks=CVBudgetStopping(6))
        assert candidate.fit_counts_ == reference.fit_counts_
        assert candidate.fit_counts_["finalist_fold_fits"] == 6
        assert candidate.final_selection_results_["cv"] == 3
        assert (
            candidate.final_selection_results_["candidates"]
            == reference.final_selection_results_["candidates"]
        )
        np.testing.assert_array_equal(candidate.predict(X), reference.predict(X))

    model.set_params(optimization_config=None, final_selection=False)
    assert clone(model).final_selection is False


@pytest.mark.parametrize("task", ["parameters", "features"])
def test_direct_finalist_cv_override_survives_group_replacement_and_clone(task):
    X, y = load_iris(return_X_y=True)
    explicit = StratifiedKFold(4)
    initial = OptimizationConfig(final_selection=True, final_selection_cv=2)
    replacement = OptimizationConfig(final_selection=True, final_selection_cv=3)
    model = search(task, optimization_config=initial, final_selection_cv=explicit)
    model.set_params(optimization_config=replacement)
    assert model.final_selection_cv is explicit
    assert clone(model).final_selection_cv.get_n_splits() == 4

    model.set_params(final_selection_cv=None)
    model.fit(X, y, callbacks=CVBudgetStopping(4))
    assert model.final_selection_results_["cv"] == 3
    assert clone(model).final_selection_cv == 3


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize("has_override", [False, True])
def test_legacy_save_preserves_the_origin_of_finalist_cv_on_group_replacement(task, has_override):
    initial = OptimizationConfig(final_selection=True, final_selection_cv=2)
    replacement = OptimizationConfig(final_selection=True, final_selection_cv=3)
    override = StratifiedKFold(4) if has_override else None
    state = search(
        task, optimization_config=initial, final_selection_cv=override
    )._serializable_state()
    state.pop("_explicit_finalist_cv", None)
    model = search(task, final_selection_cv=5)
    model.__setstate__(state)
    model.set_params(optimization_config=replacement)
    assert model.final_selection_cv is (
        override if has_override else replacement.final_selection_cv
    )


@pytest.mark.parametrize("task", ["parameters", "features"])
def test_replaced_optimization_config_is_validated_before_cv(task, monkeypatch):
    from sklearn_genetic import genetic_search

    def unexpected_cv(*args, **kwargs):
        pytest.fail("invalid optimization controls reached cross-validation")

    monkeypatch.setattr(genetic_search, "cross_validate", unexpected_cv)
    config = OptimizationConfig(local_search=True, local_search_steps=0)
    with pytest.raises(ValueError, match="local_search_steps"):
        search(task).set_params(optimization_config=config).fit(*load_iris(return_X_y=True))


def test_spread_initializer_is_specific_to_feature_selection():
    X, y = load_iris(return_X_y=True)
    tuning = search("parameters").set_params(population_initializer="spread")
    with pytest.raises(ValueError, match="population_initializer"):
        tuning.fit(X, y)


@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
def test_adjacent_log_bounds_are_evaluated_by_public_search(engine):
    if engine != "deap":
        pytest.importorskip(engine)
    lower = 1e308
    upper = np.nextafter(lower, np.inf)
    grid = {
        "ccp_alpha": Continuous(lower, upper, distribution="log-uniform"),
        "max_depth": Integer(1, 8),
    }
    model = search(
        "parameters",
        optimizer_backend=engine,
        param_grid=grid,
        population_size=8,
        error_score="raise",
    )
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        model.fit(*load_iris(return_X_y=True), callbacks=CVBudgetStopping(8))
    assert {row["ccp_alpha"] for row in model.cv_results_["params"]} == {lower, upper}
    assert model.fit_counts_["search_cv_calls"] == 8


def test_deap_differential_evolution_accepts_adjacent_log_bounds_after_initialization():
    lower = 1e308
    upper = np.nextafter(lower, np.inf)
    grid = {
        "ccp_alpha": Continuous(lower, upper, distribution="log-uniform"),
        "max_depth": Integer(1, 8),
    }
    model = search(
        "parameters",
        param_grid=grid,
        population_size=8,
        generations=2,
        algorithm="eaDifferentialEvolution",
        use_cache=False,
        error_score="raise",
        local_search=True,
        local_search_radius=1.0,
    )
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        model.fit(*load_iris(return_X_y=True))
    assert {row["ccp_alpha"] for row in model.cv_results_["params"]} == {lower, upper}
    assert model.fit_counts_["search_cv_calls"] > model.population_size


@pytest.mark.parametrize("task", ["features", "parameters"])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
@pytest.mark.parametrize("grouped", [False, True])
def test_zero_workers_rejected_before_single_cv_call(task, engine, grouped, monkeypatch):
    from sklearn_genetic import genetic_search

    if engine != "deap":
        pytest.importorskip(engine)
    model = search(task, optimizer_backend=engine)
    if grouped:
        model.set_params(runtime_config=RuntimeConfig(n_jobs=0, verbose=False))
    else:
        model.set_params(n_jobs=0)
    monkeypatch.setattr(
        genetic_search, "cross_validate", lambda *args, **kwargs: pytest.fail("unexpected CV")
    )
    with pytest.raises(ValueError, match="n_jobs must be nonzero"):
        model.fit(*load_iris(return_X_y=True), callbacks=CVBudgetStopping(1))


@pytest.mark.parametrize("task", ["features", "parameters"])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
@pytest.mark.parametrize("budget", [1, 10])
def test_cv_budget_caps_actual_calls_and_returns_a_winner(task, engine, budget, monkeypatch):
    from sklearn_genetic import genetic_search

    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    cv = genetic_search.cross_validate
    observed = []
    caller = get_ident()

    def counted(*args, **kwargs):
        observed.append(get_ident())
        return cv(*args, **kwargs)

    monkeypatch.setattr(genetic_search, "cross_validate", counted)
    result = search(
        task,
        optimizer_backend=engine,
        n_jobs=2,
        parallel_backend="population",
        generations=4,
        use_cache=False,
        error_score="raise",
    ).fit(X, y, callbacks=CVBudgetStopping(budget))
    assert len(observed) == result.fit_counts_["search_cv_calls"] == budget
    assert result.fit_counts_["search_fold_fits"] == 2 * budget
    assert result.fit_counts_["refit_fits"] == 1
    assert result.history["cross_validate_calls"][-1] == budget
    assert result.history["partial"][-1]
    assert len(result.predict(X)) == len(y)
    if budget == 1:
        assert observed == [caller]
        assert result.fit_stats_["population_parallel_batches"] == 0
        assert result.fit_stats_["population_serial_batches"] == 1
    else:
        assert any(worker != caller for worker in observed)
        assert result.fit_stats_["population_parallel_batches"] > 0


@pytest.mark.parametrize("task", ["features", "parameters"])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
@pytest.mark.parametrize("initial, updated, expected", [(12, 3, 3), (1, 12, 8)])
@pytest.mark.parametrize("resume", [False, True])
def test_cv_budget_uses_start_callback_limit_before_dispatch(
    task, engine, initial, updated, expected, resume, tmp_path
):
    from sklearn_genetic.backends import scipy_optimizer

    if engine != "deap":
        pytest.importorskip(engine)

    class StartBudget(CVBudgetStopping):
        def on_start(self, estimator=None):
            super().on_start(estimator)
            self.max_cv_calls = updated

    X, y = load_iris(return_X_y=True)
    model = search(
        task,
        optimizer_backend=(
            scipy_optimizer(stop_on_convergence=False) if engine == "scipy" else engine
        ),
        use_cache=False,
        error_score="raise",
        generations=4,
    )
    callbacks = [CVBudgetStopping(8), StartBudget(initial)]
    saved = 0
    if resume:
        checkpoint = ModelCheckpoint(tmp_path / "start-budget.pkl")
        clone(model).fit(X, y, callbacks=[CVBudgetStopping(6), checkpoint])
        saved = checkpoint.resumable_cv_calls_
        assert saved == 6
        callbacks.append(checkpoint)
    result = model.fit(X, y, callbacks=callbacks)
    assert result.fit_counts_["search_cv_calls"] == expected
    assert result.fit_counts_["search_fold_fits"] == 2 * expected
    assert result.history["cross_validate_calls"][-1] == saved + expected
    assert len(result.predict(X)) == len(X)


@pytest.mark.parametrize("updated, exception", [(0, ValueError), (2.5, TypeError)])
def test_invalid_start_callback_budget_is_rejected_before_cv(updated, exception, monkeypatch):
    from sklearn_genetic import genetic_search

    class InvalidBudget(CVBudgetStopping):
        def on_start(self, estimator=None):
            super().on_start(estimator)
            self.max_cv_calls = updated

    monkeypatch.setattr(
        genetic_search, "cross_validate", lambda *args, **kwargs: pytest.fail("unexpected CV")
    )
    with pytest.raises(exception, match="max_cv_calls"):
        search("parameters").fit(
            *load_iris(return_X_y=True), callbacks=[CVBudgetStopping(1), InvalidBudget(8)]
        )


@pytest.mark.parametrize("task", ["features", "parameters"])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
def test_cv_budget_counts_only_fresh_calls_after_checkpoint_resume(task, engine, tmp_path):
    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    model = search(
        task, optimizer_backend=engine, use_cache=False, random_state=25, error_score="raise"
    )
    checkpoint = ModelCheckpoint(tmp_path / "cv-budget.pkl")
    first = clone(model).fit(X, y, callbacks=[CVBudgetStopping(6), checkpoint])
    assert first.fit_counts_["search_cv_calls"] == 6
    assert checkpoint.resumable_cv_calls_ == 6
    assert checkpoint.unsaved_cv_calls_ == 0
    resumed = clone(model).fit(X, y, callbacks=[CVBudgetStopping(3), checkpoint])
    assert resumed.fit_counts_["search_cv_calls"] == 3
    assert resumed.history["cross_validate_calls"][-1] == 9
    assert resumed.history["partial"] == [False, True]
    saved = checkpoint.load()["logbook"][-1]
    if engine == "deap":
        assert saved["cross_validate_calls"] == 9
        assert saved["partial"]
        assert checkpoint.resumable_cv_calls_ == 3
        assert checkpoint.unsaved_cv_calls_ == 0
    else:
        assert saved["cross_validate_calls"] == 6
        assert not saved.get("partial")
        assert checkpoint.resumable_cv_calls_ == 0
        assert checkpoint.unsaved_cv_calls_ == 3


@pytest.mark.parametrize("task", ["features", "parameters"])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
def test_checkpoint_cadence_retains_scheduled_generations(task, engine, tmp_path, monkeypatch):
    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    model = search(task, optimizer_backend=engine, use_cache=False, generations=3)
    checkpoint = ModelCheckpoint(tmp_path / "cadence.pkl", every_n_generations=2)
    written_generations = []
    replace = Path.replace

    def record_write(source, target):
        if target == checkpoint.checkpoint_path:
            with source.open("rb") as stream:
                written_generations.append(pickle.load(stream)["logbook"][-1]["gen"])
        return replace(source, target)

    monkeypatch.setattr(Path, "replace", record_write)
    first = clone(model).fit(X, y, callbacks=checkpoint)

    assert first.history["gen"] == [0, 1, 2, 3]
    assert written_generations == [0, 2]
    kept = first.history["cross_validate_calls"][2]
    assert checkpoint.resumable_cv_calls_ == kept
    assert checkpoint.unsaved_cv_calls_ == first.fit_counts_["search_cv_calls"] - kept > 0

    resumed = clone(model).fit(X, y, callbacks=checkpoint)
    assert (
        resumed.fit_stats_["cross_validate_calls"] - resumed.fit_counts_["search_cv_calls"] == kept
    )
    assert resumed.history["gen"][:3] == [0, 1, 2]
    np.testing.assert_array_equal(
        resumed.cv_results_["mean_test_score"][:kept],
        first.cv_results_["mean_test_score"][:kept],
    )
    if engine == "deap":
        assert written_generations == [0, 2, 4, 6]
    else:
        assert written_generations == [0, 2]
        assert checkpoint.resumable_cv_calls_ == 0
        assert checkpoint.unsaved_cv_calls_ == resumed.fit_counts_["search_cv_calls"]


@pytest.mark.parametrize("value, error", [(True, TypeError), (1.5, TypeError), (0, ValueError)])
def test_checkpoint_cadence_requires_positive_integer(value, error, tmp_path):
    with pytest.raises(error, match="every_n_generations"):
        ModelCheckpoint(tmp_path / "cadence.pkl", every_n_generations=value)


@pytest.mark.parametrize("task", ["features", "parameters"])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
def test_checkpoint_preserves_metadata_routing(task, engine, tmp_path, monkeypatch):
    from sklearn_genetic import genetic_search
    from sklearn import config_context
    from sklearn.metrics import accuracy_score, make_scorer

    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    X, X_test, y, _ = train_test_split(X, y, test_size=0.25, random_state=326, stratify=y)
    weights = np.arange(len(y), dtype=float) + 1
    checkpoint = ModelCheckpoint(tmp_path / "weighted.pkl")
    model = search(task, optimizer_backend=engine, use_cache=False, error_score="raise")

    with config_context(enable_metadata_routing=True):
        model.estimator.set_fit_request(sample_weight=True)
        model.set_params(scoring=make_scorer(accuracy_score).set_score_request(sample_weight=True))
        first = clone(model).fit(
            X, y, callbacks=[CVBudgetStopping(6), checkpoint], sample_weight=weights
        )
    saved = checkpoint.checkpoint_path.read_bytes()

    def unexpected_cv(*args, **kwargs):
        pytest.fail("Resume dispatched CV with a mismatched metadata routing setting")

    with config_context(enable_metadata_routing=False), monkeypatch.context() as patch:
        assert checkpoint.load()["logbook"][-1]["cross_validate_calls"] == 6
        patch.setattr(genetic_search, "cross_validate", unexpected_cv)
        with pytest.raises(ValueError, match=r"weighted\.pkl.*enable_metadata_routing=True"):
            clone(model).fit(X, y, callbacks=[checkpoint], sample_weight=weights)
    assert checkpoint.checkpoint_path.read_bytes() == saved

    with config_context(enable_metadata_routing=True):
        resumed = clone(model).fit(X, y, callbacks=[checkpoint], sample_weight=weights)
        if engine != "deap":
            expected = clone(model).fit(X, y, sample_weight=weights)
    assert resumed.fit_counts_["search_cv_calls"] > 0
    if engine == "deap":
        # DEAP starts a new population and retains the saved scores.
        np.testing.assert_equal(
            resumed.cv_results_["mean_test_score"][:6], first.cv_results_["mean_test_score"]
        )
        assert len(resumed.predict(X_test)) == len(X_test)
        return
    assert expected.history == resumed.history
    assert expected.best_index_ == resumed.best_index_
    assert resumed.fit_counts_["search_cv_calls"] + 6 == expected.fit_counts_["search_cv_calls"]
    np.testing.assert_equal(
        resumed.cv_results_["mean_test_score"], expected.cv_results_["mean_test_score"]
    )
    np.testing.assert_equal(resumed.predict(X_test), expected.predict(X_test))


@pytest.mark.parametrize("successful_writes", [0, 1])
@pytest.mark.parametrize("local_search", [False, True])
def test_checkpoint_counts_only_successful_atomic_writes(
    successful_writes, local_search, tmp_path, monkeypatch
):
    from pathlib import Path

    checkpoint = ModelCheckpoint(tmp_path / "cv-budget.pkl")
    replace = Path.replace
    writes = []

    def sometimes_fail(source, target):
        if target == checkpoint.checkpoint_path:
            writes.append(target)
            if len(writes) > successful_writes:
                raise OSError("fixture cannot replace the checkpoint")
        return replace(source, target)

    monkeypatch.setattr(Path, "replace", sometimes_fail)
    X, y = load_iris(return_X_y=True)
    budget = 7 if local_search else 10
    model = search(
        "parameters",
        use_cache=False,
        generations=0 if local_search else 3,
        local_search=local_search,
    )
    model.fit(X, y, callbacks=[CVBudgetStopping(budget), checkpoint])

    assert len(writes) == 2
    assert model.fit_counts_["search_cv_calls"] == budget
    assert checkpoint.resumable_cv_calls_ == 6 * successful_writes
    assert checkpoint.unsaved_cv_calls_ == budget - checkpoint.resumable_cv_calls_
    if successful_writes:
        assert checkpoint.load()["logbook"][-1]["cross_validate_calls"] == 6
    else:
        assert not checkpoint.checkpoint_path.exists()


@pytest.mark.parametrize("exception_type", [ValueError, RecursionError])
@pytest.mark.parametrize("engine", ["deap", "pymoo"])
def test_checkpoint_serializer_failure_keeps_previous_checkpoint(
    engine, exception_type, tmp_path, monkeypatch, caplog
):
    from types import SimpleNamespace

    from sklearn_genetic.callbacks import model_checkpoint

    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    model = search("parameters", optimizer_backend=engine, use_cache=False, generations=3)
    checkpoint = ModelCheckpoint(tmp_path / "serializer-failure.pkl")
    clone(model).fit(X, y, callbacks=[CVBudgetStopping(6), checkpoint])
    assert checkpoint.resumable_cv_calls_ == 6
    saved_bytes = checkpoint.checkpoint_path.read_bytes()

    def fail_dump(data, stream):
        stream.write(b"incomplete snapshot")
        raise exception_type("fixture cannot serialize checkpoint")

    monkeypatch.setattr(
        model_checkpoint,
        "pickle",
        SimpleNamespace(dump=fail_dump, load=pickle.load, PicklingError=pickle.PicklingError),
    )
    resumed = clone(model).fit(X, y, callbacks=[CVBudgetStopping(8), checkpoint])

    assert resumed.fit_counts_["search_cv_calls"] == 8
    assert len(resumed.predict(X)) == len(y)
    assert checkpoint.resumable_cv_calls_ == 0
    assert checkpoint.unsaved_cv_calls_ == 8
    assert checkpoint.checkpoint_path.read_bytes() == saved_bytes
    assert list(tmp_path.iterdir()) == [checkpoint.checkpoint_path]
    assert f"{exception_type.__name__}: fixture cannot serialize checkpoint" in caplog.text


@pytest.mark.parametrize("task", ["features", "parameters"])
@pytest.mark.parametrize(
    "algorithm", ["eaSimple", "eaMuPlusLambda", "eaMuCommaLambda", "eaDifferentialEvolution"]
)
def test_cv_budget_also_limits_and_checkpoints_local_refinement(
    task, algorithm, tmp_path, monkeypatch
):
    from joblib import load
    from sklearn_genetic import genetic_search
    from sklearn_genetic.callbacks import LogbookSaver, loggers

    X, y = load_iris(return_X_y=True)
    model = search(
        task,
        algorithm=algorithm,
        generations=0,
        use_cache=False,
        local_search=True,
        local_search_top_k=3,
        local_search_steps=3,
        keep_top_k=3,
        error_score="raise",
    )
    checkpoint = ModelCheckpoint(tmp_path / "local-refinement.pkl")
    logbook = LogbookSaver(tmp_path / "candidate-logbook.pkl")
    dump = loggers.dump
    saved_sizes = []

    def count_saved_rows(records, *args, **kwargs):
        saved_sizes.append(len(records))
        return dump(records, *args, **kwargs)

    monkeypatch.setattr(loggers, "dump", count_saved_rows)
    result = clone(model).fit(
        X, y, callbacks=[CVBudgetStopping(12), CVBudgetStopping(7), checkpoint, logbook]
    )
    assert result.fit_counts_["search_cv_calls"] == 7
    assert result.fit_stats_["local_refinement_candidates"] == 1
    assert result.history["local_refinements"][-1] == 1
    assert checkpoint.resumable_cv_calls_ == 7
    assert checkpoint.unsaved_cv_calls_ == 0
    assert checkpoint.load()["logbook"][-1]["cross_validate_calls"] == 7
    saved = load(logbook.checkpoint_path)
    assert saved_sizes == [6, 7]
    assert [row["index"] for row in saved] == list(range(7))
    np.testing.assert_array_equal(
        [row["score"] for row in saved], result.cv_results_["mean_test_score"]
    )

    cross_validate = genetic_search.cross_validate
    fresh_fold_fits = []

    def count_folds(*args, **kwargs):
        scores = cross_validate(*args, **kwargs)
        fresh_fold_fits.append(len(scores["fit_time"]))
        return scores

    monkeypatch.setattr(genetic_search, "cross_validate", count_folds)
    resumed = model.fit(X, y, callbacks=[CVBudgetStopping(3), checkpoint, logbook])
    assert fresh_fold_fits == [2, 2, 2]
    assert resumed.fit_counts_["search_cv_calls"] == 3
    assert resumed.fit_counts_["search_fold_fits"] == 6
    assert resumed.history["cross_validate_calls"] == [7, 10]
    assert resumed.fit_stats_["local_refinement_candidates"] == 1
    assert checkpoint.resumable_cv_calls_ == 3
    assert checkpoint.unsaved_cv_calls_ == 0
    assert saved_sizes == [6, 7, 10]
    assert len(load(logbook.checkpoint_path)) == 10
    np.testing.assert_array_equal(
        resumed.cv_results_["mean_test_score"][:7], result.cv_results_["mean_test_score"]
    )


@pytest.mark.parametrize("fail_final_save", [False, True])
def test_logbook_saver_records_cached_local_refinement(fail_final_save, tmp_path, monkeypatch):
    from joblib import load
    from sklearn_genetic import algorithms
    from sklearn_genetic.callbacks import LogbookSaver, loggers

    def known_neighbor(estimator, parent, individual_cls):
        return individual_cls(parent)

    monkeypatch.setattr(algorithms, "local_neighbor", known_neighbor)
    path = tmp_path / "candidates.pkl.gz"
    saves = []
    dump = loggers.dump

    def save(records, destination, **options):
        saves.append(len(records))
        if fail_final_save and len(saves) == 2:
            Path(destination).write_bytes(b"partial snapshot")
            raise OSError("cannot write final logbook")
        return dump(records, destination, **options)

    monkeypatch.setattr(loggers, "dump", save)
    X, y = load_iris(return_X_y=True)
    model = search("parameters", generations=0, local_search=True, use_cache=True).fit(
        X, y, callbacks=[LogbookSaver(path), CVBudgetStopping(7)]
    )
    assert model.fit_counts_["search_cv_calls"] == 6
    assert model.fit_stats_["local_refinement_candidates"] == 1
    assert model.fit_stats_["cache_hits"] == 1
    assert saves == [6, 7]
    records = load(path)
    assert len(records) == (6 if fail_final_save else 7)
    assert [row["index"] for row in records] == list(range(len(records)))
    assert set(tmp_path.iterdir()) == {path}
    assert model.predict(X).shape == y.shape


@pytest.mark.parametrize("task", ["features", "parameters"])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
def test_saved_candidate_positions_identify_same_batch_duplicates(task, engine, tmp_path):
    from joblib import load
    from sklearn_genetic.callbacks import LogbookSaver

    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    options = dict(
        cv=2,
        optimizer_backend=engine,
        population_size=6,
        generations=0,
        random_state=19,
        verbose=False,
        error_score="raise",
    )
    if task == "features":
        X = X[:, :1]
        model = GAFeatureSelectionCV(DecisionTreeClassifier(random_state=19), **options)
    else:
        model = GASearchCV(
            _TreeWithDiagnosticParameterNames(),
            param_grid={"index": Categorical(["left"])},
            **options,
        )

    saver = LogbookSaver(tmp_path / "candidate-rows.pkl")
    checkpoint = ModelCheckpoint(tmp_path / "checkpoint.pkl")
    first = clone(model).fit(X, y, callbacks=[saver, checkpoint])
    records = load(saver.checkpoint_path)
    assert first.fit_counts_["search_cv_calls"] == 1
    assert first.fit_stats_["duplicate_candidates"] == 5
    assert [row["index"] for row in records] == list(range(6))
    np.testing.assert_array_equal(
        [row["score"] for row in records], first.cv_results_["mean_test_score"]
    )

    # Reproduce the repeated indices and raw parameter layout from older files.
    snapshot = checkpoint.load()
    old_records = (
        snapshot["runtime_state"]["candidate_logbook"].chapters["parameters"]
        if engine == "deap"
        else snapshot["candidate_records"]
    )
    for row in old_records:
        row["index"] = 0 if task == "features" else "left"
        if task == "parameters":
            del row["_candidate_params"]
    checkpoint.checkpoint_path.write_bytes(pickle.dumps(snapshot))

    resumed = clone(model).fit(X, y, callbacks=checkpoint)
    restored = resumed.logbook.chapters["parameters"]
    assert [row["index"] for row in restored[:6]] == list(range(6))
    np.testing.assert_array_equal(
        resumed.cv_results_["mean_test_score"][:6], first.cv_results_["mean_test_score"]
    )
    if task == "parameters":
        assert resumed.cv_results_["param_index"] == ["left"] * len(restored)


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize(
    "algorithm", ["eaSimple", "eaMuPlusLambda", "eaMuCommaLambda", "eaDifferentialEvolution"]
)
def test_deap_stop_callback_skips_local_refinement(task, algorithm, tmp_path, monkeypatch):
    from sklearn_genetic import genetic_search

    class ObserveStop(BaseCallback):
        def on_step(self, record, logbook, estimator):
            step_calls.append(record["cross_validate_calls"])

        def on_end(self, logbook, estimator):
            end_calls.append(estimator.fit_stats_["cross_validate_calls"])

    cross_validate = genetic_search.cross_validate
    fold_counts, step_calls, end_calls = [], [], []

    def count_folds(*args, **kwargs):
        scores = cross_validate(*args, **kwargs)
        fold_counts.append(len(scores["fit_time"]))
        return scores

    monkeypatch.setattr(genetic_search, "cross_validate", count_folds)
    X, y = load_iris(return_X_y=True)
    checkpoint = ModelCheckpoint(tmp_path / "callback-stop.pkl")
    result = search(
        task,
        algorithm=algorithm,
        generations=3,
        use_cache=False,
        crossover_probability=0.0,
        mutation_probability=1.0,
        local_search=True,
        local_search_top_k=2,
        local_search_steps=2,
        final_selection=True,
        final_selection_top_k=2,
        final_selection_cv=3,
        error_score="raise",
    ).fit(
        X,
        y,
        callbacks=[DeltaThreshold(1.0, generations=2), ObserveStop(), checkpoint],
    )
    assert result.history["gen"] == [0, 1]
    assert end_calls == [step_calls[-1]]
    assert result.fit_stats_["local_refinement_candidates"] == 0
    assert result.history["cross_validate_calls"] == step_calls
    assert result.fit_counts_["search_cv_calls"] == checkpoint.resumable_cv_calls_ == step_calls[-1]
    assert checkpoint.unsaved_cv_calls_ == 0
    assert checkpoint.load()["logbook"][-1]["cross_validate_calls"] == step_calls[-1]
    assert fold_counts == [2] * step_calls[-1] + [3, 3]
    assert result.fit_counts_["finalist_cv_calls"] == 2
    assert result.fit_counts_["refit_fits"] == 1
    assert result.predict(X).shape == y.shape


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize(
    "algorithm", ["eaSimple", "eaMuPlusLambda", "eaMuCommaLambda", "eaDifferentialEvolution"]
)
@pytest.mark.parametrize(
    "step, error",
    [(None, KeyboardInterrupt), (1, KeyboardInterrupt), (1, SystemExit), (3, StopIteration)],
)
def test_deap_interruption_stops_evolution_and_local_refinement(
    task, algorithm, step, error, tmp_path, monkeypatch
):
    from sklearn_genetic import genetic_search

    class StopAfterGeneration(BaseCallback):
        def on_step(self, record, logbook, estimator):
            return logbook[-1]["gen"] == (0 if step is None else step)

    X, y = load_iris(return_X_y=True)
    X, X_test, y, _ = train_test_split(X, y, test_size=0.25, random_state=43, stratify=y)
    model = search(
        task,
        algorithm=algorithm,
        generations=3,
        use_cache=False,
        crossover_probability=0.0,
        mutation_probability=1.0,
        error_score="raise",
    )
    reference = clone(model).fit(X, y, callbacks=StopAfterGeneration())
    expected_calls = reference.fit_counts_["search_cv_calls"]
    calls = after_interrupt = 0
    interrupted = False
    original_cv = genetic_search.cross_validate

    def audited_cv(*args, **kwargs):
        nonlocal calls, after_interrupt, interrupted
        if interrupted:
            after_interrupt += 1
        elif step is None and calls == expected_calls:
            interrupted = True
            raise error("stop this search")
        calls += 1
        return original_cv(*args, **kwargs)

    events = []

    class InterruptSearch(BaseCallback):
        def on_step(self, record, logbook, estimator):
            nonlocal interrupted
            gen = logbook[-1]["gen"]
            events.append(gen)
            if gen == step:
                interrupted = True
                raise error("stop this search")

        def on_end(self, logbook, estimator):
            events.append("end")

    monkeypatch.setattr(genetic_search, "cross_validate", audited_cv)
    checkpoint = ModelCheckpoint(tmp_path / "interrupted.pkl")
    fitted = (
        clone(model)
        .set_params(local_search=True, local_search_steps=2)
        .fit(X, y, callbacks=[checkpoint, InterruptSearch()])
    )

    assert interrupted
    assert after_interrupt == 0
    assert calls == expected_calls
    assert events == [*reference.history["gen"], "end"]
    assert fitted.fit_counts_ == reference.fit_counts_
    assert fitted.fit_stats_["local_refinement_candidates"] == 0
    assert fitted.history == reference.history
    saved = checkpoint.load()["logbook"]
    assert saved.select("gen") == reference.history["gen"]
    assert saved.select("cross_validate_calls") == reference.history["cross_validate_calls"]
    assert fitted.best_index_ == reference.best_index_
    np.testing.assert_array_equal(fitted.predict(X_test), reference.predict(X_test))


@pytest.mark.parametrize(
    "algorithm", ["eaSimple", "eaMuPlusLambda", "eaMuCommaLambda", "eaDifferentialEvolution"]
)
def test_cv_budget_keeps_a_completed_deap_generation(algorithm):
    X, y = load_iris(return_X_y=True)
    model = search(
        "parameters", algorithm=algorithm, generations=3, use_cache=False, error_score="raise"
    )
    reference = clone(model).fit(X, y)
    limit = reference.history["cross_validate_calls"][1]
    assert limit > reference.history["cross_validate_calls"][0]
    capped = model.fit(X, y, callbacks=CVBudgetStopping(limit))
    assert capped.fit_counts_["search_cv_calls"] == limit
    assert capped.history == {key: values[:2] for key, values in reference.history.items()}


def test_cv_budget_leaves_cache_and_finalist_work_separate():
    X, y = load_iris(return_X_y=True)
    model = search(
        "parameters",
        use_cache=True,
        final_selection=True,
        final_selection_top_k=2,
        param_grid={"max_depth": Integer(2, 3), "criterion": Categorical(["gini"])},
        error_score="raise",
    ).fit(X, y, callbacks=CVBudgetStopping(2))
    assert model.fit_counts_["search_cv_calls"] == 2
    assert model.fit_counts_["finalist_cv_calls"] == 2
    assert model.fit_counts_["refit_fits"] == 1
    assert model.fit_stats_["evaluated_candidates"] > 2


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize(
    "algorithm", ["eaSimple", "eaMuPlusLambda", "eaMuCommaLambda", "eaDifferentialEvolution"]
)
@pytest.mark.parametrize("stop_from_callback", [True, False])
def test_timer_keeps_a_completed_deap_generation(
    task, algorithm, stop_from_callback, tmp_path, monkeypatch
):
    from sklearn_genetic import genetic_search
    from sklearn_genetic.callbacks import TimerStopping

    class StopAfterFirstGeneration(BaseCallback):
        def on_step(self, record, logbook, estimator):
            return logbook[-1]["gen"] == 1

    class NoDeadline(TimerStopping):
        def _time_exceeded(self):
            return False

    X, y = load_iris(return_X_y=True)
    model = search(task, algorithm=algorithm, use_cache=False, generations=3, error_score="raise")
    # A timer can reorder the initial feature masks. Use that same order as the timed fit.
    reference = clone(model).fit(X, y, callbacks=[NoDeadline(1), StopAfterFirstGeneration()])
    limit = reference.fit_counts_["search_cv_calls"]
    assert limit > reference.history["cross_validate_calls"][0]

    calls = 0
    original_cv = genetic_search.cross_validate

    def advance_clock(*args, **kwargs):
        nonlocal calls
        result = original_cv(*args, **kwargs)
        calls += 1
        return result

    class Deadline(TimerStopping):
        def _time_exceeded(self):
            return calls >= limit

        def on_step(self, record, logbook, estimator):
            return stop_from_callback and super().on_step(record, logbook, estimator)

    monkeypatch.setattr(genetic_search, "cross_validate", advance_clock)
    checkpoint = ModelCheckpoint(tmp_path / "deadline.pkl")
    timed = clone(model).fit(X, y, callbacks=[Deadline(limit), checkpoint])
    assert calls == limit == timed.fit_counts_["search_cv_calls"]
    assert timed.history == reference.history
    assert timed.fit_stats_ == reference.fit_stats_
    results = "features" if task == "features" else "params"
    np.testing.assert_equal(timed.cv_results_[results], reference.cv_results_[results])
    np.testing.assert_equal(timed.predict(X), reference.predict(X))
    saved = checkpoint.load()
    assert saved["logbook"][-1]["gen"] == 1
    assert not saved["logbook"][-1].get("partial", False)


@pytest.mark.parametrize("budget, expected", [(1, 1), (12, 2)])
def test_cv_budget_can_be_combined_with_a_timer(budget, expected):
    from sklearn_genetic.callbacks import TimerStopping

    class ExpiredTimer(TimerStopping):
        def _time_exceeded(self):
            return True

    X, y = load_iris(return_X_y=True)
    model = search(
        "parameters",
        n_jobs=2,
        parallel_backend="population",
        use_cache=False,
        error_score="raise",
    ).fit(X, y, callbacks=[CVBudgetStopping(budget), ExpiredTimer(1)])
    assert model.fit_counts_["search_cv_calls"] == expected


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
@pytest.mark.parametrize("wall_jump", [-60, 60])
def test_timer_uses_elapsed_time_when_system_clock_changes(task, engine, wall_jump, monkeypatch):
    from datetime import datetime, timedelta
    from sklearn_genetic import genetic_search
    from sklearn_genetic.callbacks import TimerStopping, early_stoppers

    if engine != "deap":
        pytest.importorskip(engine)
    elapsed = 0
    adjustment = 0

    class WallClock:
        @staticmethod
        def utcnow():
            return datetime(2026, 1, 1) + timedelta(seconds=elapsed + adjustment)

        @classmethod
        def now(cls, tz):
            return cls.utcnow().replace(tzinfo=tz)

    original_cv = genetic_search.cross_validate

    def timed_cv(*args, **kwargs):
        nonlocal elapsed, adjustment
        results = original_cv(*args, **kwargs)
        elapsed += 1
        if elapsed == 4:
            adjustment = wall_jump
        return results

    monkeypatch.setattr(early_stoppers, "datetime", WallClock)
    monkeypatch.setattr(early_stoppers, "monotonic", lambda: elapsed, raising=False)
    monkeypatch.setattr(genetic_search, "cross_validate", timed_cv)

    model = search(task, optimizer_backend=engine, use_cache=False, error_score="raise")
    X, y = load_iris(return_X_y=True)
    timer = TimerStopping(3)
    reference = clone(model).fit(X, y, callbacks=timer)
    assert elapsed == reference.fit_counts_["search_cv_calls"] == 3
    initial_timestamp = timer.initial_training_time
    interrupted = clone(model).fit(X, y, callbacks=timer)

    assert adjustment == wall_jump
    assert elapsed == 6
    assert timer.initial_training_time == initial_timestamp + timedelta(seconds=3)
    assert interrupted.history == reference.history
    assert interrupted.fit_counts_ == reference.fit_counts_
    assert interrupted.best_score_ == reference.best_score_
    field = "features" if task == "features" else "params"
    np.testing.assert_equal(interrupted.cv_results_[field], reference.cv_results_[field])
    np.testing.assert_array_equal(interrupted.predict(X), reference.predict(X))


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize(
    "algorithm", ["eaSimple", "eaMuPlusLambda", "eaMuCommaLambda", "eaDifferentialEvolution"]
)
@pytest.mark.parametrize("refit", [True, False])
@pytest.mark.parametrize(
    "cache, initial_time_checks", [("disabled", 0), ("disabled", 1), ("covered", 0), ("partial", 0)]
)
def test_expired_deap_resume_reuses_saved_results(
    task, algorithm, refit, initial_time_checks, cache, tmp_path, monkeypatch
):
    from sklearn_genetic import genetic_search
    from sklearn_genetic.callbacks import TimerStopping

    class Expired(TimerStopping):
        def _time_exceeded(self):
            nonlocal initial_time_checks
            initial_time_checks -= 1
            return initial_time_checks < 0

    events = []

    class Observe(BaseCallback):
        def on_start(self, estimator):
            events.append("start")

        def on_step(self, record, logbook, estimator):
            events.append("step")

        def on_end(self, logbook, estimator):
            events.append(("end", estimator.search_space_exhausted_))

    X, y = load_iris(return_X_y=True)
    covered = cache == "covered"
    model = search(
        task, algorithm=algorithm, use_cache=cache != "disabled", refit=refit, error_score="raise"
    )
    if task == "features" and covered:
        model.set_params(max_features=1)
    elif task == "parameters" and cache != "disabled":
        model.set_params(
            param_grid={
                "max_depth": Integer(2, 3 if covered else 5),
                "criterion": Categorical(["gini", "entropy"]),
            }
        )
    checkpoint = ModelCheckpoint(tmp_path / "saved.pkl")
    first = clone(model).fit(X, y, callbacks=[CVBudgetStopping(6), checkpoint])
    assert first.fit_counts_["search_cv_calls"] == (4 if covered else 6)
    assert first.search_space_exhausted_ is covered
    original = checkpoint.checkpoint_path.read_bytes()

    def unexpected_cv(*args, **kwargs):
        pytest.fail("cross-validation started after the deadline")

    monkeypatch.setattr(genetic_search, "cross_validate", unexpected_cv)
    resumed = clone(model).fit(X, y, callbacks=[Expired(1), Observe(), checkpoint])

    assert events == ["start", ("end", covered)]
    assert checkpoint.checkpoint_path.read_bytes() == original
    assert resumed.search_space_exhausted_ is covered
    assert resumed.fit_counts_["search_cv_calls"] == 0
    assert resumed.fit_counts_["search_fold_fits"] == 0
    assert resumed.fit_counts_["refit_fits"] == int(refit)
    assert resumed.history == first.history
    assert resumed.fit_stats_ == first.fit_stats_
    assert resumed.best_index_ == first.best_index_
    assert resumed.best_score_ == first.best_score_
    field = "features" if task == "features" else "params"
    np.testing.assert_equal(resumed.cv_results_[field], first.cv_results_[field])
    if refit:
        np.testing.assert_equal(resumed.predict(X), first.predict(X))
    elif task == "features":
        np.testing.assert_array_equal(resumed.support_, first.support_)


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize(
    "algorithm", ["eaSimple", "eaMuPlusLambda", "eaMuCommaLambda", "eaDifferentialEvolution"]
)
def test_timer_omits_deap_generation_with_no_evaluated_candidates(task, algorithm, tmp_path):
    from sklearn_genetic.callbacks import TimerStopping

    steps = []

    class Deadline(TimerStopping):
        def __init__(self):
            super().__init__(1)
            self.checks_after_initial_generation = 0

        def _time_exceeded(self):
            if steps:
                self.checks_after_initial_generation += 1
            return self.checks_after_initial_generation > 1

        def on_step(self, record, logbook, estimator):
            steps.append(len(logbook) - 1)
            return False

    X, y = load_iris(return_X_y=True)
    model = search(
        task,
        algorithm=algorithm,
        mutation_probability=1.0,
        crossover_probability=0.0,
        use_cache=False,
        error_score="raise",
    )
    checkpoint = ModelCheckpoint(tmp_path / "last-generation.pkl")
    fitted = model.fit(X, y, callbacks=[Deadline(), checkpoint])

    assert fitted.fit_counts_["search_cv_calls"] == model.population_size
    assert steps == [0]
    assert fitted.history["gen"] == [0]
    assert not any(fitted.history.get("partial", ()))
    assert len(fitted) == 1
    assert checkpoint.load()["logbook"][-1]["gen"] == 0


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize(
    "algorithm", ["eaSimple", "eaMuPlusLambda", "eaMuCommaLambda", "eaDifferentialEvolution"]
)
@pytest.mark.parametrize("refit", [True, False])
def test_deap_resume_stops_when_verified_cache_has_full_coverage(task, algorithm, refit, tmp_path):
    X, y = load_iris(return_X_y=True)
    model = search(task, algorithm=algorithm, refit=refit, use_cache=True, error_score="raise")
    if task == "features":
        model.set_params(max_features=1)
    else:
        model.set_params(
            param_grid={"max_depth": Integer(2, 3), "criterion": Categorical(["gini", "entropy"])}
        )
    checkpoint = ModelCheckpoint(tmp_path / "finite.pkl")
    first = clone(model).fit(X, y, callbacks=checkpoint)
    assert first.search_space_exhausted_
    assert first.fit_counts_["search_cv_calls"] == 4
    before = checkpoint.checkpoint_path.read_bytes()
    events = []

    class Observe(BaseCallback):
        def on_start(self, estimator):
            events.append("start")

        def on_step(self, record, logbook, estimator):
            events.append("step")

        def on_end(self, logbook, estimator):
            events.append(("end", estimator.search_space_exhausted_))

    resumed = clone(model).fit(X, y, callbacks=[Observe(), checkpoint])

    assert events == ["start", ("end", True)]
    assert checkpoint.checkpoint_path.read_bytes() == before
    assert resumed.search_space_exhausted_
    assert resumed.history == first.history
    assert resumed.fit_stats_ == first.fit_stats_
    assert resumed.fit_counts_["search_cv_calls"] == 0
    assert resumed.fit_counts_["refit_fits"] == int(refit)
    assert resumed.best_index_ == first.best_index_
    assert resumed.best_score_ == first.best_score_
    field = "features" if task == "features" else "params"
    np.testing.assert_equal(resumed.cv_results_[field], first.cv_results_[field])
    if refit:
        np.testing.assert_array_equal(resumed.predict(X), first.predict(X))


class _DisableCacheAtStart(BaseCallback):
    def on_start(self, estimator):
        estimator.use_cache = False


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize(
    "algorithm", ["eaSimple", "eaMuPlusLambda", "eaMuCommaLambda", "eaDifferentialEvolution"]
)
def test_deap_finite_coverage_uses_cache_setting_after_start(task, algorithm):
    X, y = load_iris(return_X_y=True)
    model = search(
        task,
        algorithm=algorithm,
        population_size=16,
        population_initializer="random",
        generations=3,
        error_score="raise",
    )
    if task == "features":
        model.set_params(max_features=1)
    else:
        model.set_params(
            param_grid={"max_depth": Integer(2, 3), "criterion": Categorical(["gini", "entropy"])}
        )
    reference = clone(model).set_params(use_cache=False).fit(X, y)
    actual = model.fit(X, y, callbacks=_DisableCacheAtStart())

    assert not actual.search_space_exhausted_
    assert actual.history == reference.history
    assert actual.fit_counts_ == reference.fit_counts_
    assert actual.fit_counts_["search_cv_calls"] >= actual.population_size
    np.testing.assert_array_equal(actual.predict(X), reference.predict(X))


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
def test_disabling_cache_on_resume_requests_fresh_cv_after_saved_coverage(task, engine, tmp_path):
    from sklearn_genetic.backends import scipy_optimizer

    if engine != "deap":
        pytest.importorskip(engine)
    backend = scipy_optimizer(stop_on_convergence=False) if engine == "scipy" else engine
    X, y = load_iris(return_X_y=True)
    model = search(
        task, optimizer_backend=backend, population_size=8, generations=4, error_score="raise"
    )
    if task == "features":
        model.set_params(max_features=1)
    else:
        model.set_params(
            param_grid={"max_depth": Integer(2, 3), "criterion": Categorical(["gini", "entropy"])}
        )
    checkpoint = ModelCheckpoint(tmp_path / "covered.pkl")
    first = clone(model).fit(X, y, callbacks=checkpoint)
    assert first.search_space_exhausted_
    assert first.fit_counts_["search_cv_calls"] == 4

    resumed = clone(model).fit(
        X, y, callbacks=[_DisableCacheAtStart(), CVBudgetStopping(12), checkpoint]
    )

    assert not resumed.search_space_exhausted_
    assert resumed.fit_counts_["search_cv_calls"] == 12
    assert resumed.history["cross_validate_calls"][-1] == 16
    np.testing.assert_array_equal(resumed.predict(X), first.predict(X))


def test_deap_resume_keeps_candidate_rows_for_callable_refit(tmp_path):
    X, y = load_iris(return_X_y=True)
    selected_rows = []

    def choose_last_candidate(results):
        row = len(results["mean_test_score"]) - 1
        selected_rows.append(row)
        return row

    model = search("parameters", refit=choose_last_candidate, use_cache=True).set_params(
        param_grid={"max_depth": Integer(2, 3), "criterion": Categorical(["gini", "entropy"])}
    )
    checkpoint = ModelCheckpoint(tmp_path / "callable.pkl")
    clone(model).fit(X, y, callbacks=checkpoint)
    resumed = clone(model).fit(X, y, callbacks=checkpoint)

    assert selected_rows[1] > selected_rows[0]
    assert resumed.best_index_ == selected_rows[1]


@pytest.mark.parametrize("task", ["parameters", "features"])
def test_expired_deap_resume_attempts_first_batch_if_every_saved_fit_failed(task, tmp_path):
    from sklearn_genetic.callbacks import TimerStopping

    class Expired(TimerStopping):
        def _time_exceeded(self):
            return True

    X, y = load_iris(return_X_y=True)
    model = _cv_failure_search(task, "deap", all_fail=True)
    checkpoint = ModelCheckpoint(tmp_path / "failed.pkl")
    with pytest.warns(FitFailedWarning), pytest.raises(ValueError, match="All search candidates"):
        clone(model).fit(X, y, callbacks=checkpoint)
    retry = clone(model)
    with pytest.warns(FitFailedWarning), pytest.raises(ValueError, match="All search candidates"):
        retry.fit(X, y, callbacks=[Expired(1), checkpoint])
    assert retry.fit_counts_["search_cv_calls"] == 1
    assert retry.cv_results_["all_fits_failed"].all()
    assert retry.history["cross_validate_calls"][-1] == 7


@pytest.mark.parametrize(
    "budget, exception", [(0, ValueError), (-1, ValueError), (1.5, TypeError), (True, TypeError)]
)
def test_cv_budget_validates_the_positive_call_limit(budget, exception):
    with pytest.raises(exception, match="max_cv_calls"):
        CVBudgetStopping(budget)


@pytest.mark.parametrize("task", ["features", "parameters"])
@pytest.mark.parametrize(
    "refit, objective, message",
    [
        (False, None, "set objective_metric"),
        (False, "missing", "set objective_metric"),
        ("accuracy", "f1", "must match the named refit metric"),
        (lambda results: 0, None, "set objective_metric"),
        (lambda results: 0, "missing", "set objective_metric"),
    ],
    ids=(
        "missing-objective",
        "unknown-objective",
        "conflicting-objective",
        "callable-refit-missing-objective",
        "callable-refit-unknown-objective",
    ),
)
def test_multimetric_objective_validation_happens_before_cv(
    task, refit, objective, message, monkeypatch
):
    from sklearn_genetic import genetic_search

    X, y = load_iris(return_X_y=True)
    model = search(
        task,
        scoring={"accuracy": "accuracy", "f1": "f1_macro"},
        refit=refit,
        objective_metric=objective,
    )
    monkeypatch.setattr(
        genetic_search, "cross_validate", lambda *a, **kw: pytest.fail("unexpected CV")
    )
    with pytest.raises(ValueError, match=message):
        model.fit(X, y)


def _select_most_accurate(results):
    return np.argmax(results["mean_test_accuracy"])


@pytest.mark.parametrize("task", ["features", "parameters"])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
@pytest.mark.parametrize("first_metric", ["accuracy", "score"])
@pytest.mark.parametrize("error_score", [np.nan, "raise"])
def test_combined_callable_scorer_explains_named_scorer_configuration(
    task, engine, first_metric, error_score
):
    if engine != "deap":
        pytest.importorskip(engine)

    def combined(estimator, X, y):
        return {first_metric: estimator.score(X, y), "constant": 0.5}

    model = search(task, scoring=combined, optimizer_backend=engine, error_score=error_score)
    with pytest.raises(ValueError, match=r"single numeric score.*dictionary of named scorers"):
        model.fit(*load_iris(return_X_y=True), callbacks=CVBudgetStopping(1))
    assert not hasattr(model, "cv_results_")


@pytest.mark.parametrize("engine", ["deap", "pymoo"])
@pytest.mark.parametrize("refit", [False, _select_most_accurate], ids=["no-refit", "callable"])
def test_multimetric_checkpoint_preserves_the_optimization_objective(engine, refit, tmp_path):
    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    checkpoint = ModelCheckpoint(tmp_path / "objective.pkl")
    model = search(
        "parameters",
        optimizer_backend=engine,
        refit=refit,
        objective_metric="f1",
        scoring={"accuracy": "accuracy", "f1": "f1_macro"},
    )
    first = clone(model).fit(X, y, callbacks=[ThresholdStopping(0), checkpoint])
    resume = clone(model).fit(X, y, callbacks=checkpoint)
    assert len(resume.history["gen"]) > len(first.history["gen"])
    assert resume.refit_metric == "f1"
    assert resume.fit_counts_["refit_fits"] == int(callable(refit))
    assert resume.cv_results_dataframe()["rank_test_f1"].is_monotonic_increasing
    if callable(refit):
        assert resume.best_index_ == _select_most_accurate(resume.cv_results_)
        assert not hasattr(resume, "best_score_")

    # A different selected scorer would make cached fitness incomparable.
    for choices in (
        {"refit": "accuracy", "objective_metric": None},
        {"refit": False, "objective_metric": "accuracy"},
    ):
        changed = clone(model).set_params(**choices)
        with pytest.raises(ValueError, match="optimization metric|search configuration"):
            changed.fit(X, y, callbacks=checkpoint)


@pytest.mark.parametrize("task", ["features", "parameters"])
@pytest.mark.parametrize("engine", ["deap", "pymoo"])
def test_single_metric_refit_callable_chooses_the_fitted_candidate(task, engine):
    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    calls = []

    def scoring(estimator, X, y):
        return X.shape[1] if task == "features" else estimator.max_depth

    def select(results):
        calls.append(results)
        return np.argmin(results["mean_test_score"])

    model = search(task, optimizer_backend=engine, scoring=scoring, refit=select).fit(X, y)
    assert len(calls) == 1
    assert calls[0] is model.cv_results_
    original = int(np.argmax(model.cv_results_["mean_test_score"]))
    assert model.best_index_ != original
    assert model.best_index_ == int(np.argmin(model.cv_results_["mean_test_score"]))
    assert not hasattr(model, "best_score_")
    assert model.final_selection_results_["changed"]
    assert model.fit_counts_["refit_fits"] == 1
    assert model.fit_counts_["finalist_cv_calls"] == 0

    if task == "features":
        mask = model.cv_results_["features"][model.best_index_]
        np.testing.assert_equal(model.get_support(), mask)
        expected = clone(model.estimator).fit(X[:, mask], y).predict(X[:, mask])
    else:
        params = model.cv_results_["params"][model.best_index_]
        assert model.best_params_ == params
        expected = clone(model.estimator).set_params(**params).fit(X, y).predict(X)
    np.testing.assert_equal(model.predict(X), expected)
    assert model.score(X, y) == model.cv_results_["mean_test_score"][model.best_index_]


@pytest.mark.parametrize("engine", ["deap", "custom"])
@pytest.mark.parametrize("keep_original", [True, False])
def test_feature_callable_refit_diagnostics_compare_with_the_archived_winner(engine, keep_original):
    def optimize(problem, evaluate, report):
        proposals = [(1, 1, 0, 0), (0, 1, 0, 0)]
        report([handle for _, handle in evaluate(proposals)])

    X, y = load_iris(return_X_y=True)
    template = GAFeatureSelectionCV(
        DummyClassifier(),
        cv=2,
        max_features=2,
        optimizer_backend=engine if engine == "deap" else optimize,
        population_initializer="random",
        population_size=6,
        generations=0,
        random_state=3,
        use_cache=False,
        n_jobs=1,
        verbose=False,
    )
    reference = clone(template).fit(X, y)
    assert reference.cv_results_["rank_test_score"][reference.best_index_] == 1
    assert reference.cv_results_["rank_test_score"][0] == 1
    assert reference.n_features_ < reference.cv_results_["n_features"][0]
    chosen_index = reference.best_index_ if keep_original else 0
    selected = template.set_params(refit=lambda results: chosen_index).fit(X, y)

    diagnostics = selected.final_selection_results_
    assert diagnostics["original_best_index"] == reference.best_index_
    assert diagnostics["original_best_score"] == reference.best_score_
    np.testing.assert_array_equal(
        diagnostics["original_best_params"]["features"], reference.get_support()
    )
    assert diagnostics["changed"] is not keep_original
    assert selected.best_index_ == chosen_index
    np.testing.assert_array_equal(
        selected.get_support(), reference.cv_results_["features"][chosen_index]
    )
    assert selected.fit_counts_ == reference.fit_counts_


@pytest.mark.parametrize("task", ["features", "parameters"])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
def test_multimetric_callable_refit_keeps_the_search_objective(task, engine):
    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    calls = []

    def capacity(estimator, X, y):
        return X.shape[1] if task == "features" else estimator.max_depth

    def compactness(estimator, X, y):
        return -capacity(estimator, X, y)

    def select(results):
        calls.append(results)
        return np.argmax(results["mean_test_compactness"])

    model = search(
        task, optimizer_backend=engine, scoring={"capacity": capacity, "compactness": compactness}
    )
    reference = clone(model).set_params(refit="capacity").fit(X, y)
    selected = model.set_params(refit=select, objective_metric="capacity").fit(X, y)

    assert len(calls) == 1 and calls[0] is selected.cv_results_
    assert selected.fit_counts_ == reference.fit_counts_
    assert selected.history == reference.history
    assert selected.best_index_ == np.argmax(selected.cv_results_["mean_test_compactness"])
    assert selected.final_selection_results_["changed"]
    assert not hasattr(selected, "best_score_")
    columns = ["mean_test_capacity", "mean_test_compactness", "rank_test_capacity"]
    columns.append("features" if task == "features" else "params")
    for column in columns:
        np.testing.assert_equal(selected.cv_results_[column], reference.cv_results_[column])

    table = selected.cv_results_dataframe(preserve_index=True)
    winner = table.loc[selected.best_index_]
    objective = selected.cv_results_["mean_test_capacity"][selected.best_index_]
    assert winner["mean_test_capacity"] == objective
    if task == "features":
        mask = selected.cv_results_["features"][selected.best_index_]
        np.testing.assert_equal(winner["features"], mask)
        np.testing.assert_equal(selected.get_support(), mask)
        assert selected.n_features_ == int(mask.sum())
        expected = clone(selected.estimator).fit(X[:, mask], y).predict(X[:, mask])
    else:
        params = selected.cv_results_["params"][selected.best_index_]
        assert winner["params"] == params
        assert selected.best_params_ == params
        expected = clone(selected.estimator).set_params(**params).fit(X, y).predict(X)
    np.testing.assert_equal(selected.predict(X), expected)
    assert selected.score(X, y) == selected.cv_results_["mean_test_capacity"][selected.best_index_]


@pytest.mark.parametrize("weighted_objective", [True, False])
def test_multimetric_callable_refit_routes_score_metadata(weighted_objective):
    from sklearn import config_context
    from sklearn.metrics import accuracy_score, get_scorer, make_scorer

    X, y = load_iris(return_X_y=True)
    weights = np.arange(len(y)) + 1
    with config_context(enable_metadata_routing=True):
        scorer = make_scorer(accuracy_score).set_score_request(sample_weight=weighted_objective)
        f1 = get_scorer("f1_macro").set_score_request(sample_weight=True)
        model = search(
            "parameters",
            scoring={"accuracy": scorer, "f1": f1},
            objective_metric="accuracy",
            refit=lambda results: 0,
            generations=0,
        ).fit(X, y)
        if weighted_objective:
            assert model.score(X, y, sample_weight=weights) == accuracy_score(
                y, model.predict(X), sample_weight=weights
            )
        else:
            with pytest.raises(TypeError, match="sample_weight"):
                model.score(X, y, sample_weight=weights)


@pytest.mark.parametrize("task", ["features", "parameters"])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
@pytest.mark.parametrize("weighted_fit", [False, True])
def test_fit_routes_separate_training_and_evaluation_weights(task, engine, weighted_fit):
    from sklearn import config_context
    from sklearn.metrics import accuracy_score, make_scorer

    if engine != "deap":
        pytest.importorskip(engine)
    rows = np.arange(36)
    X = np.column_stack([rows + offset * len(rows) for offset in range(4)])
    y = rows % 2
    training = rows + 1
    evaluation = rows[::-1] + 100
    search_cv = list(StratifiedKFold(2).split(X, y))
    final_cv = list(StratifiedKFold(3).split(X, y))
    observed_fits = []
    observed_evaluations = []

    class WeightCheckingTree(DecisionTreeClassifier):
        def fit(self, X, y, sample_weight=None, check_input=True):
            expected = training[X[:, 0] % len(rows)] if weighted_fit else None
            np.testing.assert_array_equal(sample_weight, expected)
            observed_fits.append(len(y))
            return super().fit(X, y, sample_weight=sample_weight, check_input=check_input)

    def weighted_accuracy(y_true, y_pred, sample_weight=None):
        observed_evaluations.append(np.asarray(sample_weight).tolist())
        return accuracy_score(y_true, y_pred, sample_weight=sample_weight)

    with config_context(enable_metadata_routing=True):
        estimator = WeightCheckingTree(random_state=17).set_fit_request(
            sample_weight="training_weight" if weighted_fit else False
        )
        scorer = make_scorer(weighted_accuracy).set_score_request(sample_weight="evaluation_weight")
        scoring = {"acc": scorer, "secondary": "balanced_accuracy"} if weighted_fit else scorer
        model = search(
            task,
            optimizer_backend=engine,
            generations=0,
            scoring=scoring,
            refit="acc" if weighted_fit else True,
            final_selection=True,
            final_selection_top_k=2,
            final_selection_cv=final_cv,
            error_score="raise",
        ).set_params(estimator=estimator, cv=search_cv)
        metadata = {"evaluation_weight": evaluation}
        if weighted_fit:
            metadata["training_weight"] = training
        model.fit(X, y, **metadata)

    counts = model.fit_counts_
    expected_weights = [evaluation[test].tolist() for _, test in search_cv] * counts[
        "search_cv_calls"
    ]
    expected_weights += [evaluation[test].tolist() for _, test in final_cv] * counts[
        "finalist_cv_calls"
    ]
    assert observed_evaluations == expected_weights
    assert observed_fits == (
        [len(train) for train, _ in search_cv] * counts["search_cv_calls"]
        + [len(train) for train, _ in final_cv] * counts["finalist_cv_calls"]
        + [len(rows)]
    )
    assert counts["refit_fits"] == 1


@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
def test_feature_refit_reports_the_selected_search_row(engine):
    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    model = search(
        "features",
        optimizer_backend=engine,
        scoring={"accuracy": "accuracy", "f1": "f1_macro"},
        refit="f1",
    ).fit(X, y)
    index = model.best_index_
    np.testing.assert_equal(model.cv_results_["features"][index], model.get_support())
    assert model.best_score_ == model.cv_results_["mean_test_f1"][index]
    assert model.best_score_ == model.history["fitness_best"][-1]
    assert model.n_features_ == model.cv_results_["n_features"][index]
    assert model.fit_counts_["refit_fits"] == 1

    model.set_params(refit=False, objective_metric="f1").fit(X, y)
    assert model.best_index_ == index
    assert model.best_score_ == model.cv_results_["mean_test_f1"][index]
    assert not hasattr(model, "best_estimator_")
    assert model.n_features_ == int(model.support_.sum())


@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
def test_feature_pipeline_with_refit_disabled(engine):
    import pandas as pd
    from sklearn.pipeline import make_pipeline
    from sklearn.tree import DecisionTreeClassifier

    if engine != "deap":
        pytest.importorskip(engine)
    iris = load_iris()
    frame = pd.DataFrame(iris.data, columns=iris.feature_names, index=2 * np.arange(len(iris.data)))
    selector = search("features", optimizer_backend=engine, refit=False).set_output(
        transform="pandas"
    )
    pipeline = make_pipeline(selector, DecisionTreeClassifier(random_state=7)).fit(
        frame, iris.target
    )
    expected = frame.loc[:, selector.support_]
    assert selector.n_features_in_ == frame.shape[1]
    np.testing.assert_array_equal(selector.get_support(), selector.support_)
    np.testing.assert_array_equal(selector.get_feature_names_out(), expected.columns)
    pd.testing.assert_frame_equal(selector.transform(frame), expected)
    reference = DecisionTreeClassifier(random_state=7).fit(expected, iris.target)
    np.testing.assert_array_equal(pipeline.predict(frame), reference.predict(expected))
    assert selector.fit_counts_["refit_fits"] == 0
    assert not hasattr(selector, "best_estimator_")


@pytest.mark.parametrize(
    "fold_means, masks, expected",
    [
        ([0.8, 0.8], [[1, 0], [0, 1]], 1),
        ([0.7, 0.8, 0.95], [[0, 1], [1, 0], [1, 0]], 1),
    ],
    ids=["tied-distinct-masks", "same-mask-with-different-scores"],
)
def test_feature_refit_reports_the_archived_evaluation(fold_means, masks, expected):
    from sklearn.tree import DecisionTreeClassifier

    def optimizer(problem, evaluate, report):
        report([handle for _, handle in evaluate(masks)])

    scores = iter(score for score in fold_means for _ in range(2))

    def scoring(estimator, X, y):
        return next(scores)

    X, y = load_iris(return_X_y=True)
    model = GAFeatureSelectionCV(
        DecisionTreeClassifier(random_state=7),
        cv=2,
        scoring=scoring,
        optimizer_backend=optimizer,
        population_size=len(masks),
        keep_top_k=2,
        max_features=1,
        generations=0,
        use_cache=False,
        n_jobs=1,
        random_state=7,
        verbose=False,
    ).fit(X[:, :2], y)
    assert model.best_index_ == expected
    assert model.best_score_ == fold_means[expected]
    np.testing.assert_equal(model.get_support(), masks[expected])


@pytest.mark.parametrize("score,expected", [(0.8, 128), (0.7, 129)])
def test_archived_feature_row_checks_scores_before_reusing_mask_comparisons(score, expected):
    from types import SimpleNamespace

    target = np.array([True, False, False])
    other = np.array([False, True, False])
    rows = [{"features": target, "score": 0.3}, {"features": other, "score": 0.8}] * 64 + [
        {"features": target, "score": 0.8},
        {"features": target, "score": 0.95},
        {"features": target, "score": 0.8},
    ]
    search = SimpleNamespace(
        refit_metric="score",
        fit_stats_={"unique_candidates": 2},
        cv_results_={
            "mean_test_score": [row["score"] for row in rows],
            "rank_test_score": [*([68, 2] * 64), 2, 1, 2],
        },
    )
    assert (
        GAFeatureSelectionCV._archived_feature_row(search, rows, score, target.copy()) == expected
    )


@pytest.mark.parametrize("task", ["features", "parameters"])
@pytest.mark.parametrize("criteria", ["min", "max"])
def test_large_optimizer_archive_preserves_equal_score_order(task, criteria):
    X, y = load_iris(return_X_y=True)
    population = [*range(24), 24, 2, 25]
    scores = [*(i % 4 for i in range(24)), 2, 3, 0]
    direction = 1 if criteria == "max" else -1
    fold_scores = iter(direction * score for score in scores for _ in range(2))

    def scorer(estimator, X, y):
        return next(fold_scores)

    def optimize(problem, evaluate, report):
        report([handle for _, handle in evaluate(candidates)])

    if task == "features":
        X = np.tile(X[:, :1], (1, 26))
        masks = [tuple(int(column == index) for column in range(26)) for index in range(26)]
        candidates = [masks[index] for index in population]
        options = {}
    else:
        candidates = [(index + 1,) for index in population]
        options = {"param_grid": {"max_depth": Integer(1, 26)}}
    model = search(
        task,
        optimizer_backend=optimize,
        population_size=len(candidates),
        generations=0,
        keep_top_k=24,
        use_cache=False,
        criteria=criteria,
        scoring=scorer,
        **options,
    ).fit(X, y)

    # New equal-ranked elites enter first, unless they tie the filled archive's
    # last entry. An already archived candidate cannot replace its first score.
    order = [*range(23, 2, -4), 24, *range(22, 1, -4), *range(21, 0, -4), *range(20, 0, -4)]
    archive_indices = (
        [mask.index(1) for mask in model._hof]
        if task == "features"
        else [depth - 1 for (depth,) in model._hof]
    )
    assert archive_indices == order
    assert model.fit_counts_["search_cv_calls"] == len(candidates)


@pytest.mark.parametrize("task", ["features", "parameters"])
@pytest.mark.parametrize(
    "select, error",
    [
        (lambda r: False, TypeError),
        (lambda r: 1.0, TypeError),
        (lambda r: -1, IndexError),
        (lambda r: len(r["mean_test_score"]), IndexError),
    ],
)
def test_refit_callable_rejects_invalid_indices_before_the_full_data_fit(task, select, error):
    X, y = load_iris(return_X_y=True)
    model = search(task, scoring="accuracy", refit=select, generations=0)
    with pytest.raises(error, match="index.*cv_results_"):
        model.fit(X, y)
    assert model.fit_counts_["refit_fits"] == 0


def test_callable_refit_and_finalist_selection_conflict_before_cv(monkeypatch):
    from sklearn_genetic import genetic_search

    X, y = load_iris(return_X_y=True)
    model = search("features", refit=lambda r: 0, final_selection=True)
    monkeypatch.setattr(
        genetic_search, "cross_validate", lambda *a, **kw: pytest.fail("unexpected CV")
    )
    with pytest.raises(ValueError, match="Callable refit requires final_selection=False"):
        model.fit(X, y)


@pytest.mark.parametrize("task", ["features", "parameters"])
@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("final_selection_top_k", 0),
        ("final_selection_top_k", 1.5),
        ("final_selection_cv", 1),
        ("final_selection_cv", np.int64(151)),
        ("final_selection_cv", "stratified"),
        ("final_selection_cv", 2.5),
    ],
)
def test_invalid_finalist_settings_from_set_params_fail_before_cv(
    task, parameter, value, monkeypatch
):
    from sklearn_genetic import genetic_search

    def unexpected_cv(*args, **kwargs):
        pytest.fail("invalid finalist settings reached candidate cross-validation")

    monkeypatch.setattr(genetic_search, "cross_validate", unexpected_cv)
    model = search(task, scoring="accuracy", final_selection=True)
    model.set_params(**{parameter: value})
    with pytest.raises(ValueError, match=parameter):
        model.fit(*load_iris(return_X_y=True))


@pytest.mark.parametrize("task", ["features", "parameters"])
@pytest.mark.parametrize(("final_selection", "refit"), [(False, True), (True, False)])
def test_unused_finalist_cv_does_not_block_search(task, final_selection, refit):
    X, y = load_iris(return_X_y=True)
    result = search(
        task,
        generations=0,
        final_selection=final_selection,
        refit=refit,
        final_selection_cv=1,
    ).fit(X, y)
    assert result.search_time_ >= 0
    assert result.fit_counts_["search_cv_calls"] > 0
    assert result.fit_counts_["finalist_cv_calls"] == 0


@pytest.mark.parametrize("task", ["features", "parameters"])
@pytest.mark.parametrize("splitter", [KFold, StratifiedKFold, GroupKFold])
def test_standard_cv_keeps_the_independent_arrays_created_by_split(task, splitter, monkeypatch):
    from sklearn_genetic import genetic_search

    X, y = load_iris(return_X_y=True)
    cv = splitter(3)
    fit_params = {"groups": np.arange(len(X)) // 5} if splitter is GroupKFold else {}
    expected_folds = list(cv.split(X, y, **fit_params))
    original_copy = genetic_search.deepcopy
    fold_copies = []

    def track_copy(value, *args, **kwargs):
        if (
            type(value) is tuple
            and len(value) == 2
            and all(type(row) is np.ndarray for row in value)
        ):
            fold_copies.append(value)
        return original_copy(value, *args, **kwargs)

    monkeypatch.setattr(genetic_search, "deepcopy", track_copy)
    result = search(
        task,
        generations=0,
        final_selection="reuse_folds",
        final_selection_cv=splitter(3),
        final_selection_top_k=2,
        error_score="raise",
    ).set_params(cv=cv)
    result.fit(X, y, **fit_params)
    assert not fold_copies
    np.testing.assert_equal(result._cv_splits, expected_folds)
    assert result.fit_counts_["finalist_reused_folds"] == 6
    assert result.fit_counts_["finalist_fold_fits"] == 0


class _BufferReusingCV:
    def __init__(self, folds):
        self.folds = folds

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.folds)

    def split(self, X=None, y=None, groups=None):
        train, test = map(np.empty_like, self.folds[0])
        for next_train, next_test in self.folds:
            train[:] = next_train
            test[:] = next_test
            yield train, test


@pytest.mark.parametrize("task", ["features", "parameters"])
@pytest.mark.parametrize("cv_input", ["splitter", "iterator", "replaced_method"])
@pytest.mark.parametrize("final_selection", [True, "reuse_folds"])
@pytest.mark.parametrize("phase", ["search", "finalists"])
def test_cross_validation_preserves_indices_from_reused_buffers(
    task, cv_input, final_selection, phase
):
    X, y = load_iris(return_X_y=True)
    folds = list(StratifiedKFold(3, shuffle=True, random_state=7).split(X, y))
    final_folds = list(reversed(folds))

    def source(splits):
        cv = _BufferReusingCV(splits)
        if cv_input == "replaced_method":
            native = KFold(len(splits))
            native.split = cv.split
            return native
        return cv if cv_input == "splitter" else cv.split()

    model = search(
        task,
        generations=0,
        scoring="accuracy",
        error_score="raise",
        final_selection=final_selection,
        final_selection_top_k=2,
    )
    expected = clone(model).set_params(cv=folds, final_selection_cv=final_folds).fit(X, y)
    actual = model.set_params(
        cv=source(folds) if phase == "search" else folds,
        final_selection_cv=source(final_folds) if phase == "finalists" else final_folds,
    ).fit(X, y)

    for split in range(len(folds)):
        key = f"split{split}_test_score"
        np.testing.assert_array_equal(actual.cv_results_[key], expected.cv_results_[key])
    np.testing.assert_equal(
        actual.final_selection_results_["candidates"],
        expected.final_selection_results_["candidates"],
    )
    assert actual.fit_counts_ == expected.fit_counts_
    np.testing.assert_array_equal(actual.predict(X), expected.predict(X))


@pytest.mark.parametrize("historical", [False, True])
def test_deap_resume_keeps_the_fold_materialization_used_for_saved_scores(historical, tmp_path):
    X, y = load_iris(return_X_y=True)
    folds = list(StratifiedKFold(3, shuffle=True, random_state=7).split(X, y))
    original_folds = [folds[-1]] * len(folds) if historical else folds
    input_cv = original_folds if historical else _BufferReusingCV(folds)
    model = search("parameters", scoring="accuracy", generations=2, error_score="raise").set_params(
        cv=input_cv
    )
    checkpoint = ModelCheckpoint(tmp_path / "cv.pkl")
    first = clone(model).fit(X, y, callbacks=[ThresholdStopping(0), checkpoint])
    if historical:
        # Before the fix a splitter reusing arrays scored the last fold every time.
        saved = checkpoint.load()
        saved["estimator_state"]["cv"] = _BufferReusingCV(folds)
        saved["runtime_state"].pop("copy_cv_indices")
        saved["runtime_state"].pop("search_cv_splits", None)
        saved["runtime_state"].pop("training_data_hash", None)
        checkpoint.checkpoint_path.write_bytes(pickle.dumps(saved))

    resumed = model.set_params(cv=_BufferReusingCV(folds)).fit(X, y, callbacks=checkpoint)
    assert resumed.fit_counts_["search_cv_calls"] > 0
    assert len(resumed.history["gen"]) > len(first.history["gen"])
    if historical:
        assert "training_data_hash" not in checkpoint.load()["runtime_state"]
    for fold, (train, test) in enumerate(original_folds):
        scores = [
            clone(model.estimator)
            .set_params(**params)
            .fit(X[train], y[train])
            .score(X[test], y[test])
            for params in resumed.cv_results_["params"]
        ]
        np.testing.assert_array_equal(resumed.cv_results_[f"split{fold}_test_score"], scores)


@pytest.mark.parametrize("task", ["features", "parameters"])
@pytest.mark.parametrize("changed", ["rows", "targets", "groups", "weights"])
def test_deap_checkpoint_rejects_changed_training_inputs(task, changed, tmp_path, monkeypatch):
    from sklearn_genetic import genetic_search

    X, y = load_iris(return_X_y=True)
    groups = np.arange(len(X)) % 9
    weights = np.linspace(1.0, 2.0, len(X))
    model = search(task, scoring="accuracy", error_score="raise").set_params(cv=GroupKFold(3))
    checkpoint = ModelCheckpoint(tmp_path / "training.pkl")
    clone(model).fit(
        X, y, groups=groups, sample_weight=weights, callbacks=[ThresholdStopping(0), checkpoint]
    )
    saved = checkpoint.checkpoint_path.read_bytes()

    X, y, groups, weights = (array.copy() for array in (X, y, groups, weights))
    if changed == "rows":
        X[[0, -1]] = X[[-1, 0]]
    elif changed == "targets":
        y[0] = 1
    elif changed == "groups":
        groups[0] = 1
    else:
        weights[0] *= 2
    monkeypatch.setattr(
        genetic_search, "cross_validate", lambda *a, **kw: pytest.fail("unexpected CV")
    )
    with pytest.raises(ValueError, match="Checkpoint training data differs"):
        clone(model).fit(X, y, groups=groups, sample_weight=weights, callbacks=checkpoint)
    assert checkpoint.checkpoint_path.read_bytes() == saved


@pytest.mark.parametrize("task", ["features", "parameters"])
def test_deap_checkpoint_preserves_stateful_search_folds(task, tmp_path):
    X, y = load_iris(return_X_y=True)
    splitter = StratifiedKFold(3, shuffle=True, random_state=np.random.RandomState(7))
    model = search(task, scoring="accuracy", error_score="raise").set_params(cv=splitter)
    checkpoint = ModelCheckpoint(tmp_path / "stateful-cv.pkl")
    first = clone(model).fit(X, y, callbacks=[ThresholdStopping(0), checkpoint])
    resumed = clone(model).fit(X.copy(), y.copy(), callbacks=checkpoint)
    assert resumed.fit_counts_["search_cv_calls"] > 0
    for saved, restored in zip(first._cv_splits, resumed._cv_splits, strict=True):
        np.testing.assert_equal(saved, restored)
    with checkpoint.checkpoint_path.open("rb") as stream:
        runtime = pickle.load(stream)["runtime_state"]
    assert runtime["training_data_hash"]
    assert "_deap_checkpoint_data_hash" not in vars(resumed)


@pytest.mark.parametrize("task", ["features", "parameters"])
@pytest.mark.parametrize(
    "option, saved_value, requested_value",
    [
        ("criteria", "min", "max"),
        ("error_score", 0.25, "raise"),
        ("return_train_score", False, True),
    ],
)
def test_deap_checkpoint_restores_evaluation_options(
    task, option, saved_value, requested_value, tmp_path
):
    X, y = load_iris(return_X_y=True)
    model = search(task, scoring="accuracy").set_params(**{option: saved_value})
    checkpoint = ModelCheckpoint(tmp_path / "options.pkl")
    clone(model).fit(X, y, callbacks=[ThresholdStopping(0), checkpoint])
    resumed = clone(model).set_params(**{option: requested_value}).fit(X, y, callbacks=checkpoint)
    assert getattr(resumed, option) == saved_value
    assert checkpoint.load()["estimator_state"][option] == saved_value
    if option == "return_train_score":
        assert "mean_train_score" not in resumed.cv_results_


def test_deap_checkpoint_restores_the_feature_limit(tmp_path):
    X, y = load_iris(return_X_y=True)
    model = search("features", scoring="accuracy").set_params(max_features=2)
    checkpoint = ModelCheckpoint(tmp_path / "mask-size.pkl")
    clone(model).fit(X, y, callbacks=[ThresholdStopping(0), checkpoint])
    resumed = clone(model).set_params(max_features=1).fit(X, y, callbacks=checkpoint)
    assert resumed.max_features == checkpoint.load()["estimator_state"]["max_features"] == 2
    assert np.max(resumed.cv_results_["n_features"]) <= resumed.max_features
    assert resumed.support_.sum() <= resumed.max_features


def _outlier_search(task, cv, backend="deap", generations=0):
    estimator = IsolationForest(n_estimators=3, random_state=13, n_jobs=1)
    options = dict(
        cv=cv,
        generations=generations,
        optimizer_backend=backend,
        population_size=5,
        random_state=19,
        n_jobs=1,
        verbose=False,
        error_score="raise",
    )
    if task == "features":
        return GAFeatureSelectionCV(estimator, max_features=2, **options)
    return GASearchCV(
        estimator,
        param_grid={"max_samples": Continuous(0.5, 0.95), "max_features": Continuous(0.5, 1.0)},
        **options,
    )


class _EmptyCVSplitter:
    def split(self, X, y=None, groups=None):
        return iter(())


class _CustomCVWithoutGroups:
    def split(self, X, y=None):
        return StratifiedKFold(2).split(X, y)


@pytest.mark.parametrize("task", ["features", "parameters"])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
@pytest.mark.parametrize("grouped_stage", ["search", "finalists"])
def test_mixed_grouped_and_custom_cv_matches_explicit_folds(task, engine, grouped_stage):
    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    X, X_test, y, _ = train_test_split(X, y, test_size=0.25, stratify=y, random_state=71)
    groups = np.arange(len(y)) % 9
    plain, grouped = _CustomCVWithoutGroups(), GroupKFold(3)
    plain_folds = list(plain.split(X, y))
    grouped_folds = list(grouped.split(X, y, groups=groups))
    if grouped_stage == "search":
        cv, final_cv = grouped, plain
        expected_cv, expected_final_cv = grouped_folds, plain_folds
    else:
        cv, final_cv = plain, grouped
        expected_cv, expected_final_cv = plain_folds, grouped_folds
    model = search(
        task, optimizer_backend=engine, scoring="accuracy", error_score="raise", random_state=71
    ).set_params(final_selection=True, final_selection_top_k=2)
    reference = clone(model).set_params(cv=expected_cv, final_selection_cv=expected_final_cv)
    reference.fit(X, y, groups=groups, callbacks=CVBudgetStopping(4))
    model.set_params(cv=cv, final_selection_cv=final_cv)
    model.fit(X, y, groups=groups, callbacks=CVBudgetStopping(4))

    assert model.fit_counts_ == reference.fit_counts_
    assert model.fit_counts_["search_cv_calls"] == 4
    for fold in range(len(expected_cv)):
        label = f"split{fold}_test_score"
        np.testing.assert_array_equal(model.cv_results_[label], reference.cv_results_[label])
    np.testing.assert_array_equal(model.predict(X_test), reference.predict(X_test))
    assert model.best_index_ == reference.best_index_
    for actual, expected in zip(
        model.final_selection_results_["candidates"],
        reference.final_selection_results_["candidates"],
        strict=True,
    ):
        assert actual["index"] == expected["index"]
        np.testing.assert_array_equal(actual["cv_scores"], expected["cv_scores"])


def test_custom_search_cv_does_not_drop_groups_for_inactive_final_selection():
    X, y = load_iris(return_X_y=True)
    model = search("parameters").set_params(
        cv=_CustomCVWithoutGroups(), final_selection=False, final_selection_cv=GroupKFold(2)
    )

    with pytest.raises(TypeError, match="unexpected keyword argument 'groups'"):
        model.fit(X, y, groups=np.arange(len(y)) % 6)


@pytest.mark.parametrize("task", ["features", "parameters"])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
@pytest.mark.parametrize("cv_option", ["cv", "final_selection_cv"])
@pytest.mark.parametrize("source", ["list", "generator", "splitter"])
def test_empty_cv_reports_the_input_option_before_scoring_its_folds(
    task, engine, cv_option, source, monkeypatch
):
    from sklearn_genetic import genetic_search

    if engine != "deap":
        pytest.importorskip(engine)
    empty = {"list": list, "generator": lambda: iter(()), "splitter": _EmptyCVSplitter}[source]()
    options = {cv_option: empty}
    if cv_option == "final_selection_cv":
        options["final_selection"] = True
    model = search(task, optimizer_backend=engine).set_params(**options)
    original = genetic_search.cross_validate
    calls = []

    def counted(*args, **kwargs):
        calls.append(len(kwargs["cv"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(genetic_search, "cross_validate", counted)
    with pytest.raises(
        ValueError, match=rf"^{cv_option} must provide at least one train/test split"
    ):
        model.fit(*load_iris(return_X_y=True), callbacks=CVBudgetStopping(1))

    expected = [2] if cv_option == "final_selection_cv" and source != "list" else []
    assert calls == expected
    if expected:
        results = model.final_selection_results_
        assert results["failed"] is True
        assert results["time_seconds"] >= 0.0
        assert results["selected_index"] is None
        assert results["candidates"] == []
        assert "failed_candidate" not in results
        assert model.fit_counts_["finalist_cv_calls"] == 0


@pytest.mark.parametrize("task", ["features", "parameters"])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
@pytest.mark.parametrize("partition", ["training", "validation"])
@pytest.mark.parametrize("kind", ["positive", "negative", "mask"])
@pytest.mark.parametrize("cv_option", ["cv", "final_selection_cv"])
def test_invalid_explicit_cv_rows_are_rejected_before_search(
    task, engine, partition, kind, cv_option, monkeypatch
):
    from sklearn_genetic import genetic_search

    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    train, test = list(StratifiedKFold(2).split(X, y))[-1]
    indices = train if partition == "training" else test
    if kind == "mask":
        invalid = np.zeros(len(X) + 1, dtype=bool)
        invalid[indices] = True
    else:
        invalid = indices.copy()
        invalid[-1] = len(X) if kind == "positive" else -len(X) - 1
        if kind == "negative":
            invalid = invalid.tolist()
    fold = (invalid, test) if partition == "training" else (train, invalid)
    folds = ((train, test), fold) if kind == "negative" else [(train, test), fold]
    model = search(
        task,
        optimizer_backend=engine,
        scoring="accuracy",
        error_score="raise",
        final_selection=True,
    ).set_params(**{cv_option: folds})
    original = genetic_search.cross_validate
    calls = []

    def counted(*args, **kwargs):
        calls.append(len(kwargs["cv"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(genetic_search, "cross_validate", counted)
    with pytest.raises(IndexError) as error:
        model.fit(X, y, callbacks=CVBudgetStopping(1))

    assert not calls
    assert str(error.value).startswith(f"{cv_option} fold 2 {partition} ")


@pytest.mark.parametrize("task", ["features", "parameters"])
@pytest.mark.parametrize(
    "engine", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
@pytest.mark.parametrize("form", ["empty", "incomplete", "extra", "scalar_array"])
@pytest.mark.parametrize("cv_option", ["cv", "final_selection_cv"])
def test_malformed_explicit_cv_fold_is_rejected_before_search(
    task, engine, form, cv_option, monkeypatch
):
    from sklearn_genetic import genetic_search

    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    train, test = next(StratifiedKFold(2).split(X, y))
    malformed = {
        "empty": (),
        "incomplete": [train],
        "extra": np.stack((train, test, test)),
        "scalar_array": np.array(0),
    }[form]
    model = search(
        task,
        optimizer_backend=engine,
        scoring="accuracy",
        error_score="raise",
        final_selection=True,
    ).set_params(**{cv_option: [(train, test), malformed]})
    original = genetic_search.cross_validate
    calls = []

    def counted(*args, **kwargs):
        calls.append(len(kwargs["cv"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(genetic_search, "cross_validate", counted)
    with pytest.raises(ValueError, match=rf"^{cv_option} fold 2 must contain exactly two"):
        model.fit(X, y, callbacks=CVBudgetStopping(1))

    assert not calls


@pytest.mark.parametrize("task", ["features", "parameters"])
@pytest.mark.parametrize("cv_option", ["cv", "final_selection_cv"])
@pytest.mark.parametrize("dtype", [np.float64, object])
def test_invalid_cv_array_selector_dtype_is_rejected_before_search(
    task, cv_option, dtype, monkeypatch
):
    from sklearn_genetic import genetic_search

    X, y = load_iris(return_X_y=True)
    train, test = next(StratifiedKFold(2).split(X, y))
    model = search(task, final_selection=True, error_score="raise").set_params(
        **{cv_option: [(train, test), (train, test.astype(dtype))]}
    )
    monkeypatch.setattr(
        genetic_search, "cross_validate", lambda *args, **kwargs: pytest.fail("unexpected CV")
    )
    with pytest.raises(
        ValueError,
        match=rf"^{cv_option} fold 2 validation selector must use an integer or Boolean dtype",
    ):
        model.fit(X, y, callbacks=CVBudgetStopping(1))


@pytest.mark.parametrize("task", ["features", "parameters"])
@pytest.mark.parametrize("source", ["generator", "splitter"])
@pytest.mark.parametrize("cv_option", ["cv", "final_selection_cv"])
@pytest.mark.parametrize(
    ("damage", "exception", "detail"),
    [
        ("pair", ValueError, "must contain exactly two"),
        ("index", IndexError, "training indices"),
        ("mask", IndexError, "validation mask"),
        ("dtype", ValueError, "training selector must use an integer or Boolean dtype"),
    ],
)
def test_invalid_generated_folds_fail_before_scoring_their_stage(
    task, source, cv_option, damage, exception, detail, monkeypatch
):
    from sklearn_genetic import genetic_search

    X, y = load_iris(return_X_y=True)
    train, test = next(StratifiedKFold(2).split(X, y))
    invalid = {
        "pair": (train,),
        "index": (np.append(train, len(X)), test),
        "mask": (train, np.ones(len(X) + 1, dtype=bool)),
        "dtype": (train.astype(float), test),
    }[damage]
    folds = [(train, test), invalid]

    class Splitter:
        def split(self, X, y=None, groups=None):
            return iter(folds)

    model = search(task, final_selection=True).set_params(
        **{cv_option: iter(folds) if source == "generator" else Splitter()}
    )
    original = genetic_search.cross_validate
    calls = []

    def counted(*args, **kwargs):
        assert kwargs["cv"] is model._cv_splits and cv_option == "final_selection_cv"
        calls.append(len(kwargs["cv"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(genetic_search, "cross_validate", counted)
    with pytest.raises(exception, match=rf"^{cv_option} fold 2 {detail}"):
        model.fit(X, y, callbacks=CVBudgetStopping(1))

    assert calls == ([2] if cv_option == "final_selection_cv" else [])


@pytest.mark.parametrize("task", ["parameters", "features"])
@pytest.mark.parametrize("field", ["cv_splits", "search_cv_splits", "outlier_cv_splits"])
@pytest.mark.parametrize(
    ("damage", "error", "message"),
    [
        ("container", ValueError, "must contain materialized"),
        ("empty", ValueError, "must provide at least one"),
        ("pair", ValueError, "fold 2 must contain exactly two"),
        ("index", IndexError, "fold 2 training indices"),
        ("mask", IndexError, "fold 2 validation mask"),
    ],
)
def test_resume_rejects_malformed_saved_search_folds_before_cv(
    task, field, damage, error, message, tmp_path, monkeypatch
):
    from sklearn_genetic import genetic_search

    engine = "pymoo" if field == "cv_splits" else "deap"
    if engine == "pymoo":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    folds = list(StratifiedKFold(2).split(X, y))
    model = search(task, optimizer_backend=engine, use_cache=False).set_params(cv=folds)
    checkpoint = ModelCheckpoint(tmp_path / "splits.pkl")
    clone(model).fit(X, y, callbacks=[CVBudgetStopping(6), checkpoint])
    snapshot = checkpoint.load()
    target = snapshot if engine == "pymoo" else snapshot["runtime_state"]
    if field == "outlier_cv_splits":
        target[field] = target.pop("search_cv_splits")
    if damage == "container":
        target[field] = None
    elif damage == "empty":
        target[field] = []
    elif damage == "pair":
        target[field][1] = (folds[1][0],)
    elif damage == "index":
        target[field][1] = (np.array([len(X)]), folds[1][1])
    else:
        target[field][1] = (folds[1][0], np.array([True, False]))
    contents = pickle.dumps(snapshot)
    checkpoint.checkpoint_path.write_bytes(contents)

    monkeypatch.setattr(
        genetic_search, "cross_validate", lambda *args, **kwargs: pytest.fail("unexpected CV")
    )
    with pytest.raises(error, match=rf"^checkpoint {field} {message}"):
        clone(model).fit(X, y, callbacks=checkpoint)
    assert checkpoint.checkpoint_path.read_bytes() == contents


@pytest.mark.parametrize("task", ["features", "parameters"])
@pytest.mark.parametrize("splitter", ["stratified", "grouped", "explicit", 3, None])
def test_outlier_search_uses_the_configured_cv(task, splitter):
    X, labels = load_iris(return_X_y=True)
    groups, y = None, None
    if splitter == "stratified":
        y = labels
        cv = StratifiedKFold(3, shuffle=True, random_state=7)
        splits = list(cv.split(X, y))
    elif splitter == "grouped":
        groups = np.arange(len(X)) % 12
        cv = GroupKFold(3)
        splits = list(cv.split(X, groups=groups))
    elif splitter == "explicit":
        cv = splits = list(KFold(3, shuffle=True, random_state=7).split(X))
    else:
        cv = splitter
        splits = list(KFold(cv or 5).split(X))
    fitted = _outlier_search(task, cv).fit(X, y, groups=groups)
    assert fitted.n_splits_ == len(splits)
    assert fitted.fit_counts_["search_fold_fits"] == (
        len(splits) * fitted.fit_counts_["search_cv_calls"]
    )

    for i, (train, test) in enumerate(splits):
        if task == "features":
            model = clone(fitted.estimator)
            candidate_X = X[:, fitted.cv_results_["features"][0]]
        else:
            model = clone(fitted.estimator).set_params(**fitted.cv_results_["params"][0])
            candidate_X = X
        expected = model.fit(candidate_X[train]).score_samples(candidate_X[test]).mean()
        np.testing.assert_allclose(fitted.cv_results_[f"split{i}_test_score"][0], expected)


@pytest.mark.parametrize("task", ["features", "parameters"])
@pytest.mark.parametrize(
    "backend", ["deap", pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
def test_outlier_checkpoint_reuses_the_recorded_custom_folds(task, backend, tmp_path, monkeypatch):
    if backend != "deap":
        pytest.importorskip(backend)
    X, _ = load_iris(return_X_y=True)
    groups = np.arange(len(X)) % 12
    search = _outlier_search(task, GroupKFold(3), backend, generations=2)
    checkpoint = ModelCheckpoint(tmp_path / "outlier.pkl")
    first = clone(search).fit(X, groups=groups, callbacks=[ThresholdStopping(-1), checkpoint])

    def unexpected_split(*args, **kwargs):
        pytest.fail("a resume should use the exact saved folds")

    monkeypatch.setattr(GroupKFold, "split", unexpected_split)
    resumed = clone(search).fit(X, groups=groups, callbacks=checkpoint)
    assert first.n_splits_ == resumed.n_splits_ == 3
    assert len(first.history["gen"]) < len(resumed.history["gen"])
    assert resumed.fit_counts_["search_fold_fits"] == 3 * resumed.fit_counts_["search_cv_calls"]


def test_older_deap_outlier_checkpoint_keeps_its_original_folds(tmp_path):
    X, _ = load_iris(return_X_y=True)
    checkpoint = ModelCheckpoint(tmp_path / "outlier-legacy.pkl")
    original = _outlier_search("parameters", 5).fit(X, callbacks=checkpoint)
    snapshot = checkpoint.load()
    # Historical checkpoints held the caller's CV object but used default KFold.
    snapshot["estimator_state"]["cv"] = GroupKFold(3)
    checkpoint.checkpoint_path.write_bytes(pickle.dumps(snapshot))

    resumed = _outlier_search("parameters", GroupKFold(3)).fit(X, callbacks=checkpoint)
    assert resumed.n_splits_ == original.n_splits_ == 5
    np.testing.assert_array_equal(
        resumed.cv_results_["mean_test_score"][: len(original.cv_results_["params"])],
        original.cv_results_["mean_test_score"],
    )


def _outlier_target_presence_score(estimator, X, y=None):
    return 1.0 if y is None else 0.25


@pytest.mark.parametrize("task", ["features", "parameters"])
@pytest.mark.parametrize("backend", ["deap", "pymoo"])
def test_outlier_checkpoint_keeps_legacy_target_policy(task, backend, tmp_path):
    if backend != "deap":
        pytest.importorskip(backend)
    X, _ = load_iris(return_X_y=True)
    checkpoint = ModelCheckpoint(tmp_path / "outlier-target.pkl")
    search = _outlier_search(task, 3, backend, generations=3).set_params(
        scoring=_outlier_target_presence_score
    )
    zeros = np.zeros(len(X))
    clone(search).fit(X, zeros, callbacks=[ThresholdStopping(-1), checkpoint])
    saved = checkpoint.load()
    assert saved["preserves_missing_outlier_target"]
    with pytest.raises(ValueError, match="data.*differs"):
        clone(search).fit(X, callbacks=checkpoint)

    # Before the policy was recorded, fit(X) and fit(X, zeros) had the same
    # targets, scores and data fingerprint.
    saved.pop("preserves_missing_outlier_target")
    checkpoint.checkpoint_path.write_bytes(pickle.dumps(saved))
    resumed = clone(search).fit(X, callbacks=checkpoint)
    np.testing.assert_array_equal(resumed.y_, zeros)
    assert set(resumed.cv_results_["mean_test_score"]) == {0.25}
    assert resumed.fit_counts_["search_cv_calls"] > 0
    assert not checkpoint.load()["preserves_missing_outlier_target"]

    fresh = clone(search).fit(X)
    assert fresh.y_ is None
    assert set(fresh.cv_results_["mean_test_score"]) == {1.0}


@pytest.mark.parametrize(
    "backend", [pytest.param("scipy", marks=requires_scipy_backend), *BACKENDS]
)
@pytest.mark.parametrize("task", ["features", "parameters"])
@pytest.mark.parametrize(
    "refit, final_selection",
    [(True, False), (False, True), (False, "reuse_folds")],
    ids=["refit", "search-only-finalists", "search-only-reuse-folds"],
)
def test_bundle_stops_at_finite_coverage_and_validates_terminal_checkpoint(
    backend, task, refit, final_selection, tmp_path, monkeypatch
):
    from sklearn_genetic import genetic_search

    pytest.importorskip(backend)
    X, y = load_iris(return_X_y=True)
    model = search(
        task,
        optimizer_backend=backend,
        population_size=8,
        generations=5,
        refit=refit,
        final_selection=final_selection,
        error_score="raise",
    )
    if task == "features":
        model.set_params(max_features=1)
    else:
        model.set_params(
            param_grid={"max_depth": Integer(2, 3), "criterion": Categorical(["gini", "entropy"])}
        )
    checkpoint = ModelCheckpoint(tmp_path / "search.pkl")
    events = []

    class RecordCoverage(BaseCallback):
        def on_step(self, record, logbook, estimator):
            events.append((record["gen"], estimator.search_space_exhausted_))

    fitted = clone(model).fit(X, y, callbacks=[checkpoint, RecordCoverage()])
    assert fitted.search_space_exhausted_
    assert events == [(0, True)]
    assert fitted.history["gen"] == [0]
    assert fitted.fit_stats_["cross_validate_calls"] == 4
    assert fitted.fit_counts_["search_cv_calls"] == 4
    assert fitted.fit_counts_["search_fold_fits"] == 8
    assert fitted.fit_counts_["refit_fits"] == int(refit)
    assert fitted.fit_counts_["finalist_cv_calls"] == 0
    assert checkpoint.load()["resumable"]

    monkeypatch.setattr(
        genetic_search, "cross_validate", lambda *a, **kw: pytest.fail("unexpected CV on resume")
    )
    restored = clone(model).fit(X, y, callbacks=[checkpoint, RecordCoverage()])
    assert events == [(0, True)]
    assert restored.search_space_exhausted_
    assert restored.history == fitted.history
    assert restored.fit_stats_ == fitted.fit_stats_
    assert restored.fit_counts_["search_cv_calls"] == 0
    assert restored.fit_counts_["search_fold_fits"] == 0
    assert restored.fit_counts_["refit_fits"] == int(refit)
    assert restored.fit_counts_["finalist_cv_calls"] == 0
    assert restored.best_index_ == fitted.best_index_
    assert restored.best_score_ == fitted.best_score_
    if refit:
        np.testing.assert_equal(restored.predict(X), fitted.predict(X))
    elif task == "features":
        np.testing.assert_array_equal(restored.support_, fitted.support_)
    else:
        assert restored.best_params_ == fitted.best_params_

    if backend in ("scipy", "pymoo"):
        saved = checkpoint.load()
        _, replies, survivors = saved["solver_state"][0]
        saved["solver_state"][0] = ("wrong signature", replies, survivors)
        checkpoint.checkpoint_path.write_bytes(pickle.dumps(saved))
        with pytest.raises(ValueError, match="candidate proposals"):
            clone(model).fit(X, y, callbacks=checkpoint)


def test_coverage_counts_reset_between_fits_and_respect_cache_and_final_selection():
    pytest.importorskip("pymoo")
    X, y = load_iris(return_X_y=True)
    model = search("features", optimizer_backend="pymoo", population_size=6, generations=2)
    model.set_params(max_features=1).fit(X, y)
    assert model.search_space_exhausted_
    assert model.history["gen"] == [0]

    for overrides in (
        {"use_cache": False},
        {"final_selection": True},
        {"use_cache": False, "refit": False, "final_selection": True},
    ):
        refitted = clone(model).set_params(**overrides).fit(X, y)
        assert not refitted.search_space_exhausted_
        assert refitted.history["gen"] == [0, 1, 2]

    model.set_params(max_features=2, generations=0).fit(X, y)
    assert not model.search_space_exhausted_


def test_resume_accumulates_coverage_until_the_last_unseen_mask(tmp_path):
    pytest.importorskip("pymoo")
    X, y = load_iris(return_X_y=True)
    model = search("features", optimizer_backend="pymoo", generations=10)
    expected = clone(model).fit(X, y)
    assert expected.search_space_exhausted_
    assert len(expected.history["gen"]) > 1
    checkpoint = ModelCheckpoint(tmp_path / "partial-coverage.pkl")
    first = clone(model).fit(X, y, callbacks=[ThresholdStopping(0), checkpoint])
    assert not first.search_space_exhausted_
    assert first.fit_stats_["cross_validate_calls"] < 10

    resumed = clone(model).fit(X, y, callbacks=checkpoint)
    assert resumed.search_space_exhausted_
    assert resumed.fit_stats_["cross_validate_calls"] == 10
    assert resumed.fit_stats_ == expected.fit_stats_
    assert resumed.history == expected.history
    np.testing.assert_equal(resumed.predict(X), expected.predict(X))


@pytest.mark.parametrize(
    "backend, unverified_field",
    [
        ("deap", None),
        ("deap", "training_data_hash"),
        ("deap", "copy_cv_indices"),
        ("pymoo", None),
    ],
    ids=["deap", "deap-unverified-inputs", "deap-unverified-indices", "pymoo"],
)
@pytest.mark.parametrize("task", ["features", "parameters"])
def test_fit_counts_measure_only_current_work_after_checkpoint_resume(
    backend, unverified_field, task, tmp_path, monkeypatch
):
    from sklearn_genetic import _base, genetic_search

    if backend != "deap":
        pytest.importorskip(backend)
    search_elapsed = iter((1.0, 8.0, 11.0, 11.125))
    monkeypatch.setattr(_base, "perf_counter", search_elapsed.__next__)
    X, y = load_iris(return_X_y=True)
    checkpoint = ModelCheckpoint(tmp_path / "counts.pkl")
    model = search(
        task,
        optimizer_backend=backend,
        generations=0,
        final_selection="reuse_folds",
        final_selection_top_k=2,
        error_score="raise",
    ).fit(X, y, callbacks=checkpoint)
    assert model.search_time_ == 7.0
    assert model.fit_counts_["finalist_reused_folds"] == 4
    assert checkpoint.resumable_cv_calls_ == model.fit_counts_["search_cv_calls"]
    assert checkpoint.unsaved_cv_calls_ == 0

    if unverified_field is not None:
        saved = checkpoint.load()
        saved["runtime_state"].pop(unverified_field)
        checkpoint.checkpoint_path.write_bytes(pickle.dumps(saved))

    cv = genetic_search.cross_validate
    observed_folds = []

    def counted(*args, **kwargs):
        result = cv(*args, **kwargs)
        observed_folds.append(len(result["fit_time"]))
        return result

    monkeypatch.setattr(genetic_search, "cross_validate", counted)
    model.fit(X, y, callbacks=checkpoint)
    assert model.search_time_ == 0.125
    counts = model.fit_counts_
    assert counts["search_cv_calls"] + counts["finalist_cv_calls"] == len(observed_folds)
    assert checkpoint.resumable_cv_calls_ == counts["search_cv_calls"]
    assert checkpoint.unsaved_cv_calls_ == 0
    assert counts["search_fold_fits"] + counts["finalist_fold_fits"] == sum(observed_folds)
    assert counts["refit_fits"] == 1
    if unverified_field is not None:
        assert counts["finalist_cv_calls"] == 2
        assert counts["finalist_fold_fits"] == 4
        assert counts["finalist_reused_folds"] == 0
    else:
        assert counts["finalist_cv_calls"] == counts["finalist_fold_fits"] == 0
        assert counts["finalist_reused_folds"] == 4
    if backend != "deap":
        assert counts["search_cv_calls"] == 0


@pytest.mark.parametrize("task", ["features", "parameters"])
def test_deap_checkpoint_revalidates_only_unmatched_finalist_folds(task, tmp_path):
    X, y = load_iris(return_X_y=True)
    first, second, third = StratifiedKFold(3, shuffle=True, random_state=7).split(X, y)
    updated = list(StratifiedKFold(3, shuffle=True, random_state=21).split(X, y))
    final_folds = [third, updated[1], first]
    model = search(
        task, scoring="accuracy", final_selection="reuse_folds", error_score="raise"
    ).set_params(cv=_BufferReusingCV([first, second, third]), final_selection_cv=final_folds)
    checkpoint = ModelCheckpoint(tmp_path / "partial-finalists.pkl")
    clone(model).fit(X, y, callbacks=[ThresholdStopping(0), checkpoint])
    resumed = clone(model).fit(X, y, callbacks=checkpoint)

    finalists = resumed.final_selection_results_["candidates"]
    assert len(finalists) == 3
    assert resumed.fit_counts_["finalist_cv_calls"] == len(finalists)
    assert resumed.fit_counts_["finalist_fold_fits"] == len(finalists)
    assert resumed.fit_counts_["finalist_reused_folds"] == 2 * len(finalists)
    for row in finalists:
        estimator = clone(model.estimator)
        if task == "features":
            features = X[:, row["params"]["features"]]
        else:
            features = X
            estimator.set_params(**row["params"])
        expected = [
            clone(estimator).fit(features[train], y[train]).score(features[test], y[test])
            for train, test in final_folds
        ]
        np.testing.assert_array_equal(row["cv_scores"], expected)


@pytest.mark.parametrize(
    ("index_kinds", "python_selectors"),
    [
        pytest.param("biu", True, id="current"),
        pytest.param("biu", False, id="numpy_only"),
        pytest.param("iu", False, id="integer_only"),
        pytest.param("", False, id="raw"),
    ],
)
@pytest.mark.parametrize(
    "encoding", ["indices", "shared_masks", "mixed", "python_positions", "python_masks"]
)
def test_checkpoint_resume_accepts_cloned_explicit_finalist_folds(
    tmp_path, index_kinds, python_selectors, encoding
):
    pytest.importorskip("pymoo")
    X, y = load_iris(return_X_y=True)
    search_folds = list(StratifiedKFold(2, shuffle=True, random_state=5).split(X, y))
    new_folds = list(StratifiedKFold(3, shuffle=True, random_state=11).split(X, y))
    final_folds = [search_folds[0], *new_folds[1:]]
    if encoding in {"shared_masks", "mixed", "python_masks"}:
        rows = np.arange(len(X))
        search_folds = [(np.isin(rows, train), np.isin(rows, test)) for train, test in search_folds]
        if encoding == "shared_masks":
            final_folds = [
                search_folds[0],
                *((np.isin(rows, train), np.isin(rows, test)) for train, test in new_folds[1:]),
            ]
    if encoding.startswith("python_"):
        search_folds = [(train.tolist(), tuple(test.tolist())) for train, test in search_folds]
        final_folds[0] = search_folds[0]
    model = search(
        "parameters",
        optimizer_backend="pymoo",
        final_selection="reuse_folds",
        final_selection_cv=final_folds,
        error_score="raise",
    ).set_params(cv=search_folds)
    reference = clone(model).fit(X, y)
    checkpoint = ModelCheckpoint(tmp_path / "folds.pkl")
    clone(model).fit(X, y, callbacks=[ThresholdStopping(0), checkpoint])

    if not python_selectors:
        from sklearn_genetic._backend_runtime import _config_hash

        model = clone(model)
        saved = checkpoint.load()
        digest = _config_hash(model, index_kinds=index_kinds, python_selectors=False)
        saved["context"] = (saved["context"][0], digest, digest, *saved["context"][3:])
        checkpoint.checkpoint_path.write_bytes(pickle.dumps(saved))

    resumed = model.fit(X, y, callbacks=checkpoint)
    assert resumed.history == reference.history
    assert (
        resumed.final_selection_results_["candidates"]
        == reference.final_selection_results_["candidates"]
    )
    assert all(row["reused_folds"] == 1 for row in resumed.final_selection_results_["candidates"])
    np.testing.assert_equal(resumed.predict(X), reference.predict(X))

    changed = clone(model).set_params(final_selection_cv=list(reversed(final_folds)))
    with pytest.raises(ValueError, match="search configuration"):
        changed.fit(X, y, callbacks=checkpoint)
    if encoding.startswith("python_"):
        train, test = search_folds[0]
        changed = clone(model).set_params(cv=[(train[::-1], test), *search_folds[1:]])
        with pytest.raises(ValueError, match="search configuration"):
            changed.fit(X, y, callbacks=checkpoint)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("task", ["features", "parameters"])
def test_catalog_fit(backend, task, tmp_path):
    pytest.importorskip(backend)
    X, y = load_iris(return_X_y=True)
    model = search(task, optimizer_backend=backend, scoring={"f1": "f1_macro"}, refit="f1")
    model.fit(X, y)
    again = clone(model).fit(X, y)
    np.testing.assert_array_equal(
        model.cv_results_["mean_test_f1"], again.cv_results_["mean_test_f1"]
    )
    key = "features" if task == "features" else "params"
    assert np.array_equal(model.cv_results_[key], again.cv_results_[key])
    assert len(model.cv_results_[key]) == 18
    assert len(model.history["gen"]) == 3
    assert len(model.predict(X)) == len(y)
    if task == "features":
        assert all(1 <= sum(mask) <= 2 for mask in model.cv_results_[key])
    checkpoint = ModelCheckpoint(tmp_path / "progress.pkl")
    model.fit(X, y, callbacks=[ThresholdStopping(0), checkpoint])
    assert model.history["gen"] == [0]
    assert checkpoint.load()["resumable"] is True
    model.fit(X, y, callbacks=checkpoint)
    assert model.history == again.history
    np.testing.assert_equal(model.cv_results_[key], again.cv_results_[key])


@pytest.mark.parametrize(
    "engine", ["scipy", "nevergrad", "cmaes", "pymoo", "TwoPointsDE", "custom-sigma"]
)
@pytest.mark.parametrize("task", ["features", "parameters"])
def test_resume_multiple_boundaries_without_repeating_cv(engine, task, tmp_path, monkeypatch):
    from sklearn_genetic import genetic_search

    backend = {"TwoPointsDE": "nevergrad", "custom-sigma": "cmaes"}.get(engine, engine)
    pytest.importorskip(backend)
    configured = engine
    if engine != backend:
        optimize = importlib.import_module(f"sklearn_genetic._{backend}_backend").optimize
        options = {"strategy": engine} if backend == "nevergrad" else {"sigma": 0.12}
        configured = partial(optimize, **options)
    X, y = load_iris(return_X_y=True)
    model = search(
        task,
        optimizer_backend=configured,
        use_cache=task == "features",
        generations=3,
        scoring={"f1": "f1_macro"},
        refit="f1",
    )
    reference = clone(model).fit(X, y)
    checkpoint = ModelCheckpoint(tmp_path / "search.pkl")
    steps = []
    calls = 0
    cv = genetic_search.cross_validate

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return cv(*args, **kwargs)

    class StopAfterOneReport(BaseCallback):
        def on_step(self, record, logbook, estimator):
            steps.append(record["gen"])
            return True

    monkeypatch.setattr(genetic_search, "cross_validate", counted)
    for _ in reference.history["gen"]:
        restored = clone(model).fit(X, y, callbacks=[checkpoint, StopAfterOneReport()])
    assert steps == reference.history["gen"]
    restored.fit(X, y, callbacks=checkpoint)
    assert restored.history == reference.history
    assert restored.fit_stats_ == reference.fit_stats_
    assert calls == reference.fit_stats_["cross_validate_calls"]
    for field in ("features" if task == "features" else "params", "mean_test_f1", "split0_test_f1"):
        np.testing.assert_equal(restored.cv_results_[field], reference.cv_results_[field])
    np.testing.assert_equal(restored.predict(X), reference.predict(X))


@pytest.mark.parametrize("task", ["parameters", "features"])
def test_nevergrad_resume_preserves_cv_rng_in_fresh_process(tmp_path, task):
    pytest.importorskip("nevergrad")
    script = dedent("""
        import json
        import os
        import sys
        from pathlib import Path
        import numpy as np
        from sklearn.datasets import load_iris
        from sklearn_genetic import genetic_search
        from sklearn_genetic.callbacks import ModelCheckpoint
        from sklearn_genetic.tests.test_optimizer_backends import search

        def score(estimator, X, y):
            return np.random.random()

        directory, phase, task = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
        checkpoint = ModelCheckpoint(directory / "search.pkl")
        if phase == "crash":
            cv = genetic_search.cross_validate
            calls = 0

            def interrupt_after_cv(*args, **kwargs):
                global calls
                result = cv(*args, **kwargs)
                calls += 1
                if calls == 7:  # generation zero is durable; the next batch is not
                    os._exit(86)
                return result

            genetic_search.cross_validate = interrupt_after_cv
        callbacks = [] if phase == "control" else [checkpoint]
        model = search(task, optimizer_backend="nevergrad", scoring=score, use_cache=False)
        model.fit(*load_iris(return_X_y=True), callbacks=callbacks)
        (directory / phase).write_text(json.dumps(list(model.cv_results_["mean_test_score"])))
    """)
    for phase in ("control", "crash", "resume"):
        child = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path), phase, task],
            cwd=Path(__file__).resolve().parents[2],
            check=phase != "crash",
        )
        if phase == "crash":
            assert child.returncode == 86
    assert (tmp_path / "control").read_bytes() == (tmp_path / "resume").read_bytes()


@pytest.mark.parametrize(
    "engine,damage,error",
    [
        ("scipy", "environment", "environment"),
        ("cmaes", "generation", "boundary"),
        ("pymoo", "generation", "boundary"),
        ("scipy", "candidate", "candidate proposals"),
        ("pymoo", "survivor", "survivors"),
        ("pymoo", "logbook", "logbook.*boundary"),
        ("nevergrad", "missing nevals", "logbook.*boundary"),
        ("cmaes", "candidate_records", "candidate_records.*boundary"),
        ("scipy", "missing replies", "solver_state.*boundary"),
        ("cmaes", "native payload type", "solver_state.*boundary"),
        ("nevergrad", "native payload bytes", "solver_state.*unreadable"),
    ],
)
def test_incompatible_checkpoint_fails_before_cv(engine, damage, error, tmp_path, monkeypatch):
    from sklearn_genetic import genetic_search

    pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    model = search("parameters", optimizer_backend=engine)
    callback = ModelCheckpoint(tmp_path / "search.pkl")
    model.fit(X, y, callbacks=[callback, ThresholdStopping(0)])
    saved = callback.load()
    state = saved["solver_state"]
    if damage == "environment":
        saved["context"] = (*saved["context"][:3], ())
    elif damage == "generation":
        saved["solver_state"] = (7, state[1]) if engine == "cmaes" else []
    elif damage in {"logbook", "candidate_records"}:
        saved[damage] = None
    elif damage == "missing nevals":
        del saved["logbook"][0]["nevals"]
    elif damage == "missing replies":
        signature, _, survivors = state[0]
        state[0] = (signature, None, survivors)
    elif damage in {"native payload type", "native payload bytes"}:
        payload = None if damage == "native payload type" else b""
        saved["solver_state"] = (state[0], payload)
    else:
        key, replies, survivors = state[0]
        if damage == "candidate":
            key = "different"
        else:
            survivors = survivors[::-1]
        state[0] = (key, replies, survivors)
    contents = pickle.dumps(saved)
    callback.checkpoint_path.write_bytes(contents)
    monkeypatch.setattr(
        genetic_search, "cross_validate", lambda *a, **kw: pytest.fail("unexpected CV")
    )
    with pytest.raises(ValueError, match=error):
        model.fit(X, y, callbacks=callback)
    assert callback.checkpoint_path.read_bytes() == contents


@pytest.mark.parametrize("engine", [pytest.param("scipy", marks=requires_scipy_backend), "pymoo"])
@pytest.mark.parametrize("damage", [False, True])
def test_feature_checkpoint_replay_verifies_existing_proposal_signatures(
    engine, damage, tmp_path, monkeypatch
):
    from sklearn_genetic import _optimizer_checkpoint, genetic_search
    from sklearn_genetic.backends import scipy_optimizer

    pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    X = np.tile(X, (1, 65))
    optimizer = scipy_optimizer(stop_on_convergence=False) if engine == "scipy" else engine
    template = search("features", optimizer_backend=optimizer, use_cache=False)
    checkpoint = ModelCheckpoint(tmp_path / "feature-checkpoint.pkl")
    clone(template).fit(X, y, callbacks=[CVBudgetStopping(6), checkpoint])

    saved = checkpoint.load()
    signature, replies, survivors = saved["solver_state"][0]
    canonical_hash = _optimizer_checkpoint.joblib_hash
    assert signature == canonical_hash(
        tuple(tuple(row) for row in saved["problem"].initial_population)
    )
    if damage:
        saved["solver_state"][0] = ("different", replies, survivors)
        checkpoint.checkpoint_path.write_bytes(pickle.dumps(saved))

    cross_validate = genetic_search.cross_validate
    calls = []
    proposal_hashes_before_cv = []

    def count_cv(*args, **kwargs):
        calls.append(1)
        return cross_validate(*args, **kwargs)

    def count_hash(value):
        if not calls and isinstance(value, tuple) and value and type(value[0]) is tuple:
            proposal_hashes_before_cv.append(1)
        return canonical_hash(value)

    monkeypatch.setattr(genetic_search, "cross_validate", count_cv)
    monkeypatch.setattr(_optimizer_checkpoint, "joblib_hash", count_hash)
    if damage:
        with pytest.raises(ValueError, match="candidate proposals"):
            clone(template).fit(X, y, callbacks=checkpoint)
        assert not calls
        assert proposal_hashes_before_cv == [1]
    else:
        resumed = clone(template).fit(X, y, callbacks=[CVBudgetStopping(6), checkpoint])
        assert len(calls) == resumed.fit_counts_["search_cv_calls"] == 6
        assert not proposal_hashes_before_cv
        new_masks = tuple(tuple(map(int, mask)) for mask in resumed.cv_results_["features"][6:12])
        assert checkpoint.load()["solver_state"][1][0] == canonical_hash(new_masks)


@pytest.mark.parametrize("backend", BACKENDS)
def test_missing_extra(backend, monkeypatch):
    for name in tuple(sys.modules):
        if name.startswith(backend + "."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.delitem(sys.modules, backend, raising=False)

    class AbsentPackage:
        def find_spec(self, name, *args):
            if name == backend:
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            return None

    monkeypatch.setattr(sys, "meta_path", [AbsentPackage(), *sys.meta_path])
    adapter = importlib.import_module(f"sklearn_genetic._{backend}_backend")
    with pytest.raises(ImportError, match=rf"sklearn-genetic-opt\[{backend}\]"):
        adapter.optimize(None, None, None)


def test_missing_checkpoint_extra_shows_install_command(monkeypatch, tmp_path):
    from importlib.metadata import PackageNotFoundError
    from sklearn_genetic import _optimizer_checkpoint

    def missing(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(_optimizer_checkpoint, "version", missing)
    with pytest.raises(ImportError, match=r"sklearn-genetic-opt\[cmaes\]"):
        search("parameters", optimizer_backend="cmaes").fit(
            *load_iris(return_X_y=True), callbacks=ModelCheckpoint(tmp_path / "missing.pkl")
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_fixed_space_and_priors(backend):
    pytest.importorskip(backend)
    X, y = load_iris(return_X_y=True)
    grid = dict(
        max_depth=Integer(2, 2), criterion=Categorical(["gini"]), ccp_alpha=Continuous(0, 0)
    )
    model = search("parameters", optimizer_backend=backend, param_grid=grid).fit(X, y)
    assert model.history["gen"] == [0]
    grid["criterion"] = Categorical(["gini", "entropy"], priors=[0.2, 0.8])
    with pytest.raises(ValueError, match="Unsupported dimension"):
        search("parameters", optimizer_backend=backend, param_grid=grid).fit(X, y)


def test_cmaes_wide_integers_and_flat_objective():
    pytest.importorskip("cmaes")
    from sklearn_genetic._cmaes_backend import optimize
    from sklearn_genetic.backends import OptimizationProblem

    problem = OptimizationProblem(
        "parameters",
        ((0, 0.5),) * 4,
        ("i", "x"),
        (Integer(0, 10**12), Continuous(0, 1)),
        None,
        3,
        0.8,
        22,
    )
    observed = []

    def evaluate(rows):
        rows = list(rows)
        observed.extend(rows)
        return [(0, row) for row in rows]

    with np.errstate(invalid="raise", divide="raise"):
        optimize(problem, evaluate, lambda handles: None)
    assert len(observed) == 16
    assert all(isinstance(i, int) and 0 <= i <= 10**12 and 0 <= x <= 1 for i, x in observed)


def test_cmaes_boundary_initialization():
    pytest.importorskip("cmaes")
    from sklearn_genetic._cmaes_backend import optimize
    from sklearn_genetic.backends import OptimizationProblem

    problem = OptimizationProblem(
        "parameters", ((0,),) * 4, ("x",), (Integer(0, 1),), None, 6, 0.8, 429
    )
    batches = []
    with np.errstate(invalid="raise", divide="raise"):
        optimize(problem, lambda rows: [(row[0], row) for row in rows], batches.append)
    assert len(batches) == 7


def test_cmaes_binary_prior_uses_exact_counts_and_keeps_mask_repair(monkeypatch):
    from types import SimpleNamespace
    from sklearn_genetic._cmaes_backend import optimize
    from sklearn_genetic._mixed_space import MixedSpace
    from sklearn_genetic.backends import OptimizationProblem

    cmaes = pytest.importorskip("cmaes")
    initial = [(1, 0, 1, 0), (0, 1, 1, 0), (0, 0, 1, 1)] * 4
    problem = OptimizationProblem("features", tuple(initial), (), (), 2, 1, 0.8, 19)
    prior = []

    class ScriptedSolver:
        def __init__(self, **kwargs):
            prior.append(kwargs["cat_param"])

        def ask(self):
            return SimpleNamespace(x=None, z=None, c=np.array([[False, True]] * 4))

        def tell(self, solutions):
            assert len(solutions) == len(initial)

    monkeypatch.setattr(cmaes, "CatCMAwM", ScriptedSolver)
    observed = []

    def evaluate(rows):
        observed.extend(rows)
        return [(0.0, genes) for genes in rows]

    optimize(problem, evaluate, lambda handles: None)
    first_elite = np.asarray(initial[:3])
    expected_prior = [(np.bincount(first_elite[:, i], minlength=2) / 3 + 0.5) / 2 for i in range(4)]
    np.testing.assert_array_equal(prior[0], expected_prior)
    repaired = MixedSpace(problem).decode([1, 1, 1, 1])
    assert observed[len(initial) :] == [repaired] * len(initial)
    assert sum(repaired) == problem.max_features


def test_pymoo_saturated_choice_preserves_batch_budget():
    pytest.importorskip("pymoo")
    from sklearn_genetic._pymoo_backend import optimize
    from sklearn_genetic.backends import OptimizationProblem

    problem = OptimizationProblem(
        "parameters",
        (({"a": 0},),) * 5,
        ("x",),
        (Categorical([{"a": 0}, {"b": 1}]),),
        None,
        3,
        0,
        73,
    )
    evaluated = []
    history = []

    def evaluate(rows):
        evaluated.extend(rows)
        return [(float("a" in row[0]), row) for row in rows]

    optimize(problem, evaluate, history.append)
    assert len(evaluated) == 20
    assert all(len(survivors) == 5 for survivors in history)
    assert all(row[0] in problem.dimensions[0].choices for row in evaluated)


def test_pymoo_wide_integer_offsets():
    pytest.importorskip("pymoo")
    from sklearn_genetic._pymoo_backend import optimize
    from sklearn_genetic.backends import OptimizationProblem

    initial = ((2**38,), (2**39,), (2**40,), (2**41,))
    problem = OptimizationProblem(
        "parameters", initial, ("x",), (Integer(0, 2**42),), None, 4, 0.8, 13
    )
    rows = []

    def evaluate(batch):
        rows.extend(batch)
        return [(-row[0], row) for row in batch]

    optimize(problem, evaluate, lambda population: None)
    assert len(rows) == 20 and all(isinstance(row[0], int) and 0 <= row[0] <= 2**42 for row in rows)


def test_nevergrad_continuous_search_completes_long_budget():
    pytest.importorskip("nevergrad")
    from sklearn_genetic._nevergrad_backend import optimize
    from sklearn_genetic.backends import OptimizationProblem

    dims = tuple(Continuous(0.001, 10, distribution="log-uniform") for _ in range(3))
    problem = OptimizationProblem(
        "parameters", ((1.0, 1.0, 1.0),) * 8, ("a", "b", "c"), dims, None, 14, 0.8, 41
    )

    def evaluate(batch):
        return [(sum((x - 0.3) ** 2 for x in row), row) for row in batch]

    batches = []
    optimize(problem, evaluate, batches.append)
    assert len(batches) == 15
    assert all(len(population) == 8 for population in batches)


def test_nevergrad_long_mixed_run_with_boundary_start():
    pytest.importorskip("nevergrad")
    from sklearn_genetic._nevergrad_backend import optimize
    from sklearn_genetic.backends import OptimizationProblem

    dims = (Continuous(0, 1), Integer(1, 4), Categorical([None, "x", "y"]))
    problem = OptimizationProblem(
        "parameters", ((0.0, 1, "y"),) * 8, ("x", "y", "z"), dims, None, 15, 0.8, 12
    )
    batches = []

    def evaluate(rows):
        return [(x**2 + n + (c == "y"), (x, n, c)) for x, n, c in rows]

    optimize(problem, evaluate, batches.append)
    assert len(batches) == 16
