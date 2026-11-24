import pickle
import random
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from threading import Barrier

import numpy as np
import pytest
from joblib import parallel_backend
from scipy import __version__ as scipy_version
from sklearn import config_context, get_config
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.exceptions import NotFittedError
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

from sklearn_genetic import GAFeatureSelectionCV, GASearchCV
from sklearn_genetic import evaluation
from sklearn_genetic.backends import scipy_optimizer
from sklearn_genetic.callbacks import (
    BaseCallback,
    CVBudgetStopping,
    ModelCheckpoint,
    ThresholdStopping,
    TimerStopping,
)
from sklearn_genetic.space import Categorical, Continuous, Integer

requires_scipy_backend = pytest.mark.skipif(
    np.lib.NumpyVersion(scipy_version) < "1.15.0", reason="requires the scipy optimizer extra"
)


def search(kind, **kwargs):
    model = DecisionTreeClassifier(random_state=33)
    options = dict(population_size=6, generations=2, n_jobs=1, random_state=33, verbose=False)
    options.update(kwargs)
    if kind == "features":
        return GAFeatureSelectionCV(model, 2, max_features=2, **options)
    grid = options.pop(
        "param_grid",
        dict(
            criterion=Categorical(["gini", "entropy"]),
            ccp_alpha=Continuous(1e-5, 0.1, distribution="log-uniform"),
            max_depth=Integer(1, 5),
        ),
    )
    return GASearchCV(model, 2, grid, **options)


@pytest.mark.parametrize("kind", ["parameters", "features"])
def test_default_and_explicit_deap_agree(kind):
    X, y = load_iris(return_X_y=True)
    default = search(kind).fit(X, y)
    explicit = search(kind, optimizer_backend="deap").fit(X, y)
    assert default.history == explicit.history
    assert default.fit_stats_ == explicit.fit_stats_
    np.testing.assert_array_equal(
        default.cv_results_["mean_test_score"], explicit.cv_results_["mean_test_score"]
    )
    assert list(default) == list(explicit)
    np.testing.assert_array_equal(default.predict(X), explicit.predict(X))


@pytest.mark.parametrize("kind", ["parameters", "features"])
def test_deap_preserves_caller_creator_classes_after_failed_and_successful_fit(kind, monkeypatch):
    from deap import creator

    names = {
        "FitnessMax": type("CallerFitness", (), {}),
        "Individual": type("CallerIndividual", (list,), {}),
    }
    for name, value in names.items():
        monkeypatch.setitem(vars(creator), name, value)

    X, y = load_iris(return_X_y=True)
    model = search(kind, generations=0)

    def fail_population():
        raise ValueError("initial population unavailable")

    with monkeypatch.context() as scoped:
        scoped.setattr(model, "_initialize_population", fail_population)
        with pytest.raises(ValueError, match="initial population unavailable"):
            model.fit(X, y)
    assert all(getattr(creator, name) is value for name, value in names.items())

    model.fit(X, y)
    assert len(model.predict(X)) == len(y)
    assert all(getattr(creator, name) is value for name, value in names.items())


def test_overlapping_deap_searches_keep_independent_individuals():
    X, y = load_iris(return_X_y=True)
    ready = Barrier(2)

    class WaitForOtherSearch(BaseCallback):
        def on_start(self, estimator=None):
            ready.wait(timeout=5)

    def fit(kind, criteria):
        return search(kind, criteria=criteria, generations=1, error_score="raise").fit(
            X, y, callbacks=WaitForOtherSearch()
        )

    with ThreadPoolExecutor(2) as executor:
        parameters = executor.submit(fit, "parameters", "min")
        features = executor.submit(fit, "features", "max")
        tuning, selection = parameters.result(timeout=15), features.result(timeout=15)

    assert tuning.best_score_ == min(tuning.cv_results_["mean_test_score"])
    assert selection.best_score_ == max(selection.cv_results_["mean_test_score"])
    assert len(tuning.predict(X)) == len(selection.predict(X)) == len(y)


def test_deap_feature_selection_can_fit_nested_deap_hyperparameter_search():
    X, y = load_iris(return_X_y=True)
    inner = search("parameters", population_size=3, generations=0, error_score="raise")
    outer = search("features", population_size=3, generations=0, error_score="raise")
    outer.set_params(estimator=inner).fit(X, y)
    assert len(outer.predict(X)) == len(y)
    assert outer.fit_counts_["search_cv_calls"] > 0


@pytest.mark.parametrize("kind", ["parameters", "features"])
@pytest.mark.parametrize("backend", ["deap", pytest.param("scipy", marks=requires_scipy_backend)])
def test_repeated_fit_without_refit_discards_the_previous_winner(kind, backend):
    X, y = load_iris(return_X_y=True)
    model = search(
        kind,
        optimizer_backend=backend,
        generations=0,
        final_selection=True,
        final_selection_top_k=2,
    ).fit(X, y)
    assert hasattr(model, "best_estimator_")
    assert model.final_selection_results_["enabled"]

    model.set_params(refit=False).fit(X, y[::-1])
    check_is_fitted(model)
    assert model.fit_counts_["refit_fits"] == 0
    assert model.fit_counts_["finalist_fold_fits"] == 0
    assert model.fit_counts_["finalist_reused_folds"] == 0
    fresh = clone(model).fit(X, y[::-1])
    np.testing.assert_equal(
        model.cv_results_["mean_test_score"], fresh.cv_results_["mean_test_score"]
    )
    assert model.fit_stats_ == fresh.fit_stats_
    for name in ("best_estimator_", "estimator_", "refit_time_", "final_selection_results_"):
        assert not hasattr(model, name)
    for name in ("best_params_", "best_score_", "best_index_"):
        assert getattr(model, name, None) == getattr(fresh, name, None)

    model.set_params(refit=True)
    with pytest.raises(NotFittedError):
        model.predict(X)
    assert len(model.fit(X, y).predict(X)) == len(y)


@pytest.mark.parametrize("kind", ["parameters", "features"])
@pytest.mark.parametrize(
    "backend",
    ["deap", pytest.param("scipy", marks=requires_scipy_backend), "nevergrad", "cmaes", "pymoo"],
)
def test_refit_error_keeps_cv_records_without_publishing_a_partial_estimator(kind, backend):
    if backend != "deap":
        pytest.importorskip(backend)
    X, y = load_iris(return_X_y=True)
    full_train_size = len(X)
    reject_full_fit = False

    class FullDataFailureTree(DecisionTreeClassifier):
        def fit(self, X, y, **fit_params):
            super().fit(X, y, **fit_params)
            if reject_full_fit and len(X) == full_train_size:
                raise ValueError("refit failed after creating fitted state")
            return self

    model = search(
        kind,
        optimizer_backend=backend,
        generations=0,
        final_selection=True,
        final_selection_top_k=2,
        final_selection_cv=3,
        error_score="raise",
    ).set_params(estimator=FullDataFailureTree(random_state=33))
    model.fit(X, y)
    predictions = model.predict(X)
    scores = model.cv_results_["mean_test_score"]
    counts = model.fit_counts_.copy()
    reject_full_fit = True

    with pytest.raises(ValueError, match="refit failed after creating fitted state"):
        model.fit(X, y)
    assert model.fit_counts_ == {**counts, "refit_fits": 0}
    np.testing.assert_array_equal(model.cv_results_["mean_test_score"], scores)
    assert model.final_selection_results_["candidates"]
    for name in ("best_estimator_", "estimator_", "refit_time_"):
        assert not hasattr(model, name)
    with pytest.raises(NotFittedError):
        check_is_fitted(model)
    with pytest.raises(NotFittedError):
        model.predict(X)

    reject_full_fit = False
    np.testing.assert_array_equal(model.fit(X, y).predict(X), predictions)


@pytest.mark.parametrize("kind", ["parameters", "features"])
@pytest.mark.parametrize(
    "backend",
    ["deap", pytest.param("scipy", marks=requires_scipy_backend), "nevergrad", "cmaes", "pymoo"],
)
@pytest.mark.parametrize("materialize", [False, True])
def test_failed_fit_before_search_discards_previous_diagnostics(kind, backend, materialize):
    if backend != "deap":
        pytest.importorskip(backend)
    iris = load_iris(as_frame=True)
    model = search(kind, optimizer_backend=backend, generations=0).fit(iris.data, iris.target)
    if materialize:
        assert len(model.logbook.chapters["parameters"]) > 0
        assert len(model.hof) > 0

    values = iris.data.to_numpy()[:, :-1]
    model.set_params(final_selection_top_k=0)
    with pytest.raises(ValueError, match="final_selection_top_k"):
        model.fit(values, iris.target)

    result_attributes = (
        "fit_stats_",
        "fit_counts_",
        "search_time_",
        "search_space_exhausted_",
        "scorer_",
        "multimetric_",
        "n_splits_",
        "logbook",
        "hof",
        "history",
        "cv_results_",
        "best_estimator_",
    )
    assert [name for name in result_attributes if hasattr(model, name)] == []
    with pytest.raises(NotFittedError):
        check_is_fitted(model)

    model.set_params(final_selection_top_k=3).fit(values, iris.target)
    fresh = clone(model).fit(values, iris.target)
    assert model.fit_stats_ == fresh.fit_stats_
    assert model.fit_counts_ == fresh.fit_counts_
    assert model.history == fresh.history
    np.testing.assert_array_equal(
        model.cv_results_["mean_test_score"], fresh.cv_results_["mean_test_score"]
    )
    np.testing.assert_array_equal(model.predict(values), fresh.predict(values))
    assert len(model.logbook.chapters["parameters"]) == len(fresh.logbook.chapters["parameters"])
    if kind == "features":
        assert not hasattr(model, "feature_names_in_")
        assert model.n_features_in_ == values.shape[1]


@pytest.mark.parametrize("kind", ["parameters", "features"])
@pytest.mark.parametrize("backend", ["deap", "plugin"])
def test_population_fit_reuses_threads_and_closes_them(kind, backend, monkeypatch):
    pools = []

    class TrackedPool(ThreadPoolExecutor):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.closed = False
            pools.append(self)

        def shutdown(self, **kwargs):
            super().shutdown(**kwargs)
            self.closed = True

    class InspectState(BaseCallback):
        def on_step(self, record, logbook, estimator):
            assert estimator._population_pool[1] is pools[-1]
            assert "_population_pool" not in estimator.__getstate__()
            return False

    def repeat_initial(problem, evaluate, report):
        for _ in range(problem.generations + 1):
            report([handle for _, handle in evaluate(problem.initial_population)])

    monkeypatch.setattr(evaluation, "ThreadPoolExecutor", TrackedPool)
    options = dict(
        use_cache=False, optimizer_backend=repeat_initial if backend == "plugin" else backend
    )
    X, y = load_iris(return_X_y=True)
    reference = search(kind, **options).fit(X, y)
    parallel = search(kind, **options, n_jobs=2, parallel_backend="population").fit(
        X, y, callbacks=InspectState()
    )

    assert len(pools) == 1 and pools[0].closed
    assert "_population_pool" not in vars(parallel)
    assert parallel.fit_stats_["population_parallel_batches"] >= 2
    assert (
        parallel.fit_stats_["cross_validate_calls"] == reference.fit_stats_["cross_validate_calls"]
    )
    np.testing.assert_equal(
        parallel.cv_results_["mean_test_score"], reference.cv_results_["mean_test_score"]
    )
    np.testing.assert_equal(parallel.predict(X), reference.predict(X))


@pytest.mark.parametrize("integer_only", [False, True])
def test_population_fit_keeps_configured_joblib_backend(monkeypatch, integer_only):
    def unexpected_pool(**kwargs):
        pytest.fail("The configured joblib process backend should handle this search")

    monkeypatch.setattr(evaluation, "ThreadPoolExecutor", unexpected_pool)
    X, y = load_iris(return_X_y=True)
    model = search("parameters", n_jobs=2, generations=0, parallel_backend="population")
    if integer_only:
        model.set_params(param_grid={"max_depth": Integer(1, 5), "min_samples_leaf": Integer(1, 3)})
    with parallel_backend(backend="loky", n_jobs=2):
        fitted = model.fit(X, y)
    assert fitted.fit_stats_["population_parallel_batches"] == 1
    assert len(fitted.predict(X)) == len(y)


@pytest.mark.parametrize("kind", ["parameters", "features"])
@pytest.mark.parametrize("joblib_backend", [None, "loky"])
def test_population_fit_propagates_sklearn_config_and_metadata(kind, joblib_backend):
    X, y = load_iris(return_X_y=True)
    weights = np.arange(len(y), dtype=float) + 1

    def weighted_score(y_true, y_pred, sample_weight=None):
        config = get_config()
        assert config["enable_metadata_routing"]
        assert config["working_memory"] == 17
        assert sample_weight is not None
        return np.average(y_true == y_pred, weights=sample_weight)

    def create_search(n_jobs):
        model = search(kind, n_jobs=n_jobs, parallel_backend="population", error_score="raise")
        model.estimator.set_fit_request(sample_weight="train_weights")
        return model.set_params(
            scoring=make_scorer(weighted_score).set_score_request(sample_weight="score_weights")
        )

    with config_context(enable_metadata_routing=True, working_memory=17):
        metadata = dict(train_weights=weights, score_weights=weights[::-1])
        expected = create_search(1).fit(X, y, **metadata)
        selected_backend = parallel_backend(joblib_backend) if joblib_backend else nullcontext()
        with selected_backend:
            actual = create_search(2).fit(X, y, **metadata)

    assert actual.fit_stats_["population_parallel_batches"] > 0
    assert actual.fit_counts_ == expected.fit_counts_
    assert actual.history == expected.history
    assert actual.best_index_ == expected.best_index_
    np.testing.assert_equal(
        actual.cv_results_["mean_test_score"], expected.cv_results_["mean_test_score"]
    )
    np.testing.assert_equal(actual.predict(X), expected.predict(X))


def test_population_fit_cleans_up_after_scoring_error():
    def failing_score(estimator, X, y):
        raise RuntimeError("scoring failed")

    X, y = load_iris(return_X_y=True)
    model = search("features", n_jobs=2, scoring=failing_score, error_score="raise")
    with pytest.raises(RuntimeError, match="scoring failed"):
        model.fit(X, y)
    assert "_population_pool" not in vars(model)
    assert model.set_params(scoring="accuracy").fit(X, y).predict(X).shape == y.shape


@pytest.mark.parametrize("kind", ["parameters", "features"])
@pytest.mark.parametrize("parallel", ["cv", "population"])
@requires_scipy_backend
def test_scipy_results_and_repeated_fit(kind, parallel):
    X, y = load_iris(return_X_y=True)
    est = clone(search(kind)).set_params(
        optimizer_backend="scipy",
        scoring={"acc": "accuracy", "f1": "f1_macro"},
        refit="f1",
        n_jobs=2,
        parallel_backend=parallel,
        return_train_score=True,
    )
    assert clone(est).get_params()["optimizer_backend"] == "scipy"
    est.fit(X, y)
    previous = est.cv_results_["mean_test_f1"].copy()
    history = est.history.copy()
    assert len(previous) == est.fit_stats_["evaluated_candidates"]
    assert history["cross_validate_calls"][-1] == est.fit_stats_["cross_validate_calls"]
    assert len(list(est)) == len(history["gen"])
    assert "split1_train_acc" in est.cv_results_
    if kind == "features":
        masks = np.asarray(est.cv_results_["features"])
        assert np.all((masks.sum(axis=1) >= 1) & (masks.sum(axis=1) <= 2))
        np.testing.assert_array_equal(est.transform(X), X[:, est.support_])
    else:
        assert set(est.best_params_) == set(est.param_grid)
        assert est.best_score_ == pytest.approx(max(previous))
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(est)).predict(X), est.predict(X))
    est.fit(X, y)
    np.testing.assert_array_equal(est.cv_results_["mean_test_f1"], previous)
    assert est.history == history


@pytest.mark.parametrize("kind", ["features", "parameters"])
@requires_scipy_backend
def test_scipy_checkpoint_and_legacy_progress_file(kind, tmp_path):
    checkpoint = ModelCheckpoint(tmp_path / "progress.pkl")
    est = search(kind, optimizer_backend="scipy")
    X, y = load_iris(return_X_y=True)
    est.fit(X, y, callbacks=[ThresholdStopping(0.0), checkpoint])
    saved = checkpoint.load()
    assert saved["resumable"] is True
    assert len(saved["candidate_records"]) == est.fit_stats_["evaluated_candidates"] == 6
    assert saved["logbook"][0]["gen"] == 0
    est.fit(X, y, callbacks=checkpoint)
    assert len(est.history["gen"]) > 1
    saved.update(resumable=False, solver_state=None, context=None)
    contents = pickle.dumps(saved)
    checkpoint.checkpoint_path.write_bytes(contents)
    with pytest.raises(ValueError, match="does not support exact resume"):
        est.fit(X, y, callbacks=checkpoint)
    assert checkpoint.checkpoint_path.read_bytes() == contents


@pytest.mark.parametrize(
    "option",
    [
        {"algorithm": "eaSimple"},
        {"mutation_probability": 0.3},
        {"local_search": True},
        {"param_grid": {"max_depth": Integer(2**70, 2**70 + 2)}},
        {"population_size": 4},
        {"generations": -1},
        {"crossover_probability": 1.1},
        {"param_grid": {"criterion": Categorical(["entropy", "gini"], priors=[0.2, 0.8])}},
    ],
)
@requires_scipy_backend
def test_unsupported_scipy_options(option):
    with pytest.raises(ValueError, match="DEAP-only|SciPy"):
        search("parameters", optimizer_backend="scipy", **option).fit(*load_iris(return_X_y=True))


def test_base_install_and_old_scipy(monkeypatch):
    from sklearn_genetic import _scipy_backend

    monkeypatch.setattr(_scipy_backend, "scipy_version", "1.11.4")
    X, y = load_iris(return_X_y=True)
    est = search("features", generations=0).fit(X, y)
    with pytest.raises(ImportError, match=r"sklearn-genetic-opt\[scipy\]"):
        est.set_params(optimizer_backend="scipy").fit(X, y)
    with pytest.raises(ValueError, match="optimizer_backend"):
        est.set_params(optimizer_backend="other").fit(X, y)


@requires_scipy_backend
def test_scipy_propagates_scoring_errors():
    def broken_scorer(model, X, y):
        raise ValueError("score unavailable")

    with pytest.raises(ValueError, match="score unavailable"):
        search(
            "features", optimizer_backend="scipy", scoring=broken_scorer, error_score="raise"
        ).fit(*load_iris(return_X_y=True))


@requires_scipy_backend
def test_scipy_minimize_cache_and_rounded_log_bounds():
    X, y = load_iris(return_X_y=True)
    bounds = Continuous(100.0, np.nextafter(100.0, np.inf), distribution="log-uniform")
    est = search(
        "parameters",
        optimizer_backend="scipy",
        criteria="min",
        param_grid={"ccp_alpha": bounds, "max_depth": Integer(2, 2)},
        warm_start_configs=[{"ccp_alpha": 100.0, "max_depth": 2}],
    ).fit(X, y)
    assert est.cv_results_["params"][0]["ccp_alpha"] == 100.0
    assert est.best_score_ == min(est.cv_results_["mean_test_score"])
    assert est.fit_stats_["cross_validate_calls"] < len(est.cv_results_["params"])


@pytest.mark.parametrize("kind", ["features", "parameters"])
@requires_scipy_backend
def test_scipy_refit_modes(kind):
    X, y = load_iris(return_X_y=True)
    est = search(kind, optimizer_backend="scipy", refit=False, generations=0)
    est.fit(X, y)
    assert "mean_test_score" in est.cv_results_
    assert not hasattr(est, "best_estimator_")
    est.set_params(
        refit=True, final_selection="reuse_folds", final_selection_top_k=2, error_score="raise"
    ).fit(X, y)
    assert est.best_score_ == pytest.approx(est.cv_results_["mean_test_score"][est.best_index_])
    assert len(est.predict(X)) == len(X)


def test_legacy_fitted_state_defaults_to_deap():
    est = search("features")
    state = est.__getstate__()
    del state["optimizer_backend"]
    est.set_params(optimizer_backend="scipy")
    est.__setstate__(state)
    assert clone(est).optimizer_backend == "deap"


def test_feature_operators_preserve_seeded_repairs():
    from sklearn_genetic.utils.tools import cxUniform, mutFlipBit

    # Legacy operator output and next draw, including the empty-mask repair.
    state = random.getstate()
    try:
        random.seed(17)
        assert mutFlipBit([1, 1, 1], 1) == ([0, 1, 0],)
        assert random.random() == 0.17468829278727938
        random.seed(2)
        assert cxUniform([0, 0, 1], [1, 0, 0], 0.5) == ([1, 0, 0], [1, 0, 1])
        assert random.random() == 0.3610574739836072
    finally:
        random.setstate(state)


@requires_scipy_backend
def test_scipy_runner_uses_plain_candidates_and_opaque_evaluation_handles():
    from sklearn_genetic._scipy_backend import run_scipy

    proposals = []
    generations = []

    def evaluate(candidates):
        batch = []
        for genes in candidates:
            assert type(genes) is list
            handle = len(proposals)
            proposals.append(genes.copy())
            batch.append((genes[0] ** 2, handle))
        return iter(batch)

    run_scipy(
        [[value] for value in range(6)],
        evaluate,
        generations.append,
        dimensions=(Integer(0, 5),),
        max_features=None,
        generations=1,
        recombination=0.8,
        seed=11,
    )
    assert len(proposals) == 12
    assert generations[0] == list(range(6))
    assert len(generations[1]) == 6
    assert all(0 <= handle < len(proposals) for handle in generations[1])
    assert min(proposals[h][0] for h in generations[1]) == 0


def test_history_accepts_generation_rows_without_deap():
    from sklearn_genetic._base import generation_history

    history = generation_history(iter([{"gen": 3, "fitness": 0.8}, {"gen": 5, "fitness": 0.9}]))
    assert history["gen"] == [3, 5]
    assert history["fitness"] == [0.8, 0.9]
    assert history["fitness_best_observed"] == [None, None]


@requires_scipy_backend
def test_identical_scipy_survivors_keep_separate_handles():
    from itertools import count
    from sklearn_genetic._scipy_backend import run_scipy

    numbers = count()
    reports = []
    run_scipy(
        [[0]] * 6,
        lambda candidates: ((0.0, next(numbers)) for _ in candidates),
        reports.append,
        dimensions=(Integer(0, 0),),
        max_features=None,
        generations=1,
        recombination=0.8,
        seed=11,
    )
    assert reports[0] == list(range(6))
    assert len(set(reports[1])) == 6


@requires_scipy_backend
def test_scipy_finds_a_valid_trial_after_several_all_invalid_populations():
    from sklearn_genetic._scipy_backend import run_scipy

    reports = []
    batches = []

    def evaluate(rows):
        batch = len(batches)
        batches.append(rows)
        return [(np.inf if batch < 3 else 0.0, (batch, index)) for index in range(len(rows))]

    with np.errstate(over="raise", invalid="raise"):
        run_scipy(
            [[value] for value in range(6)],
            evaluate,
            reports.append,
            dimensions=(Integer(0, 5),),
            max_features=None,
            generations=5,
            recombination=0.8,
            seed=11,
        )
    assert [len(rows) for rows in batches] == [6] * 4
    assert [len(report) for report in reports] == [6] * 4
    assert set(reports[-1]) == {(3, index) for index in range(6)}


def _tied_score(estimator, X, y):
    return 1.0


def _huge_tied_score(estimator, X, y):
    return 1e308


@requires_scipy_backend
@pytest.mark.parametrize("kind", ["parameters", "features"])
def test_scipy_replays_saved_extreme_losses_before_applying_convergence(
    kind, tmp_path, monkeypatch
):
    from sklearn_genetic import _scipy_backend

    X, y = load_iris(return_X_y=True)
    model = search(
        kind, optimizer_backend="scipy", generations=4, scoring=_huge_tied_score, use_cache=False
    )
    checkpoint = ModelCheckpoint(tmp_path / "plateau.pkl")
    with monkeypatch.context() as historical:
        # An older solver's overflowing variance let equal losses continue.
        historical.setattr(_scipy_backend, "_losses_converged", lambda energies: False)
        recorded = clone(model).fit(X, y, callbacks=[CVBudgetStopping(18), checkpoint])
    with np.errstate(over="raise", invalid="raise"):
        resumed = clone(model).fit(X, y, callbacks=checkpoint)

    assert resumed.fit_counts_["search_cv_calls"] == 0
    assert resumed.history == recorded.history
    assert resumed.history["gen"] == [0, 1, 2]
    np.testing.assert_array_equal(resumed.predict(X), recorded.predict(X))


@requires_scipy_backend
@pytest.mark.parametrize("kind", ["parameters", "features"])
def test_scipy_can_search_past_equal_losses_within_cv_budget(kind, tmp_path):
    X, y = load_iris(return_X_y=True)
    options = dict(generations=4, scoring=_tied_score, use_cache=False, error_score="raise")
    default = search(kind, optimizer_backend="scipy", **options)
    configured = search(
        kind, optimizer_backend=scipy_optimizer(stop_on_convergence=False), **options
    )
    default.fit(X, y, callbacks=CVBudgetStopping(19))
    expected = clone(configured).fit(X, y, callbacks=CVBudgetStopping(19))
    key = "features" if kind == "features" else "params"
    assert default.fit_counts_["search_cv_calls"] == 12
    assert expected.fit_counts_["search_cv_calls"] == 19
    assert default.history["gen"] == [0, 1]
    assert expected.history["partial"] == [False, False, False, True]
    np.testing.assert_equal(expected.cv_results_[key][:12], default.cv_results_[key])

    checkpoint = ModelCheckpoint(tmp_path / "scipy.pkl")
    first = clone(configured).fit(X, y, callbacks=[ThresholdStopping(1), checkpoint])
    assert first.fit_counts_["search_cv_calls"] == 6
    with pytest.raises(ValueError, match="configuration differs"):
        clone(configured).set_params(optimizer_backend=scipy_optimizer()).fit(
            X, y, callbacks=checkpoint
        )
    resumed = clone(configured).fit(X, y, callbacks=[CVBudgetStopping(13), checkpoint])
    assert resumed.fit_counts_["search_cv_calls"] == 13
    assert resumed.fit_stats_["cross_validate_calls"] == 19
    np.testing.assert_equal(resumed.cv_results_[key], expected.cv_results_[key])
    np.testing.assert_equal(resumed.predict(X), expected.predict(X))


@requires_scipy_backend
def test_configured_scipy_retains_validation_and_finite_space_stopping():
    X, y = load_iris(return_X_y=True)
    backend = scipy_optimizer(stop_on_convergence=False)
    with pytest.raises(ValueError, match="SciPy requires population_size >= 5"):
        search("parameters", optimizer_backend=backend, population_size=4).fit(X, y)
    fixed = search(
        "parameters",
        optimizer_backend=backend,
        param_grid={"max_depth": Integer(2, 2)},
        generations=4,
    ).fit(X, y)
    assert fixed.search_space_exhausted_
    assert fixed.history["gen"] == [0]
    assert fixed.fit_counts_["search_cv_calls"] == 1


def _initial_population_only(problem, evaluate, report):
    # One evaluation per initial row. Different rows can have identical scores.
    scored = evaluate(problem.initial_population)
    report([handle for _, handle in scored])


@pytest.mark.parametrize("cap", [None, 2])
def test_binary_batch_preserves_scalar_mask_repairs(cap):
    from sklearn_genetic._mixed_space import MixedSpace
    from sklearn_genetic.backends import OptimizationProblem

    supplied = np.asfortranarray(
        [[1, 0, 1, 0, 0], [0, 0, 0, 0, 0], [1, 0, 1, 1, 1], [1, 0, 1, 1, 1]], dtype=bool
    )
    problem = OptimizationProblem(
        task="features",
        initial_population=((1, 0, 0, 0, 0),),
        parameter_names=(),
        dimensions=(),
        max_features=cap,
        generations=2,
        crossover_probability=0.8,
        seed=17,
    )
    decoder = MixedSpace(problem)
    original = supplied.copy()
    batch = decoder.decode_masks(supplied)

    assert batch.dtype == np.uint8
    assert batch.tolist() == [decoder.decode(mask) for mask in original]
    assert all(1 <= count <= (cap or supplied.shape[1]) for count in batch.sum(axis=1))
    np.testing.assert_array_equal(supplied, original)


def test_pymoo_submits_repaired_feature_proposals_as_integer_tuples(monkeypatch):
    pytest.importorskip("pymoo")
    from sklearn_genetic import _backend_runtime

    validate = _backend_runtime._valid_feature_mask
    proposals = []

    def record(row, limit):
        assert type(row) is tuple
        assert all(type(bit) is int for bit in row)
        assert 1 <= sum(row) <= limit
        proposals.append(row)
        return validate(row, limit)

    monkeypatch.setattr(_backend_runtime, "_valid_feature_mask", record)
    X, y = load_iris(return_X_y=True)
    X = np.tile(X, (1, 8))
    fitted = search("features", optimizer_backend="pymoo", generations=1, use_cache=False).fit(X, y)

    assert len(proposals) == fitted.population_size
    np.testing.assert_array_equal(
        fitted.cv_results_["features"][fitted.population_size :], proposals
    )


@pytest.mark.parametrize(
    "engine",
    [pytest.param("scipy", marks=requires_scipy_backend), "pymoo", "cmaes", "nevergrad", "plugin"],
)
@pytest.mark.parametrize("checkpointing", [False, True])
def test_backend_reuses_native_initial_mask_validation(
    engine, checkpointing, tmp_path, monkeypatch
):
    from sklearn_genetic import _backend_runtime
    from sklearn_genetic.callbacks import ModelCheckpoint

    backend = _initial_population_only if engine == "plugin" else engine
    if engine != "plugin":
        pytest.importorskip(engine)
    mask_validation = _backend_runtime._valid_feature_mask
    checked = []

    def check(row, limit):
        checked.append(row)
        return mask_validation(row, limit)

    monkeypatch.setattr(_backend_runtime, "_valid_feature_mask", check)
    X, y = load_iris(return_X_y=True)
    X = np.tile(X, (1, 18))
    model = search("features", optimizer_backend=backend, generations=0, use_cache=False)
    callbacks = ModelCheckpoint(tmp_path / "first.pkl") if checkpointing else None
    fitted = model.fit(X, y, callbacks=callbacks)

    assert fitted.fit_counts_["search_cv_calls"] == model.population_size
    assert fitted.support_.sum() <= model.max_features
    assert not checked


def test_backend_validates_a_custom_initial_feature_population(monkeypatch):
    from sklearn_genetic import genetic_search

    class InvalidInitialMask(GAFeatureSelectionCV):
        def _repair_individual(self, individual):
            individual[:] = [0] * len(individual)
            return individual

    monkeypatch.setattr(
        genetic_search,
        "cross_validate",
        lambda *a, **kw: pytest.fail("unexpected cross-validation"),
    )
    model = InvalidInitialMask(
        DecisionTreeClassifier(random_state=27),
        cv=2,
        population_size=6,
        optimizer_backend=_initial_population_only,
        random_state=27,
        verbose=False,
    )
    with pytest.raises(ValueError, match="nonempty binary mask"):
        model.fit(*load_iris(return_X_y=True))


def test_backend_validates_alternative_masks_after_initial_population():
    def invalid_trial(problem, evaluate, report):
        _initial_population_only(problem, evaluate, report)
        evaluate([(0,) * len(problem.initial_population[0])])

    with pytest.raises(ValueError, match="nonempty binary mask"):
        search("features", optimizer_backend=invalid_trial, use_cache=False).fit(
            *load_iris(return_X_y=True)
        )


@pytest.mark.parametrize("override", [False, True])
def test_plugin_validates_repeated_canonical_tuple_once_per_batch(override, monkeypatch):
    from sklearn_genetic import _backend_runtime

    original = _backend_runtime._valid_feature_mask
    checked = []

    def check(row, limit):
        checked.append(tuple(row))
        return original(row, limit)

    monkeypatch.setattr(_backend_runtime, "_valid_feature_mask", check)
    first, repeated = (1, 0, 0, 0), (0, 1, 0, 0)

    def optimize(problem, evaluate, report):
        responses = evaluate([list(first), repeated, repeated])
        report([handle for _, handle in responses])

    class ObservingSelector(GAFeatureSelectionCV):
        def evaluate_population(self, individuals):
            assert all(type(mask) is list for mask in individuals)
            assert individuals[1] is not individuals[2]
            return super().evaluate_population(individuals)

    selector = ObservingSelector if override else GAFeatureSelectionCV
    model = selector(
        DecisionTreeClassifier(random_state=27),
        cv=2,
        population_size=3,
        generations=0,
        optimizer_backend=optimize,
        random_state=27,
        verbose=False,
    ).fit(*load_iris(return_X_y=True))

    assert checked == [first, repeated]
    assert model.fit_counts_["search_cv_calls"] == 2
    assert model.fit_stats_["duplicate_candidates"] == 1
    np.testing.assert_array_equal(model.cv_results_["features"], [first, repeated, repeated])


@pytest.mark.parametrize("kind", ["features", "parameters"])
def test_plugin_switch_clone_and_legacy_result_views(kind, tmp_path):
    X, y = load_iris(return_X_y=True)
    model = clone(search(kind, optimizer_backend=_initial_population_only, keep_top_k=3))
    model.fit(X, y)
    first = model.cv_results_["mean_test_score"]
    assert model.history["gen"] == [0]
    assert len(first) == 6
    assert "logbook" not in vars(model)
    model.set_params(keep_top_k=1)
    if kind == "features":
        assert model.hof.maxsize == 3
        model.hof.clear()
    book = model.logbook
    model.evaluate(list(model._hof[0]))
    assert book.chapters["parameters"] is model._candidate_rows
    assert len(book) == len(model._candidate_rows) == 7
    model.logbook.record(marker=9)
    model.logbook.header = ["marker"]
    path = tmp_path / "estimator.pkl"
    model.save(path)
    restored = search(kind)
    restored.load(path)
    assert restored.logbook.header == ["marker"]
    assert restored.logbook[-1] == {"marker": 9}
    if kind == "features":
        assert len(restored.hof) == 0
    model.set_params(optimizer_backend="deap").fit(X, y)
    assert "_candidate_rows" not in vars(model)
    reference = search(kind, keep_top_k=1).fit(X, y)
    assert model.history == reference.history
    model.set_params(optimizer_backend=_initial_population_only).fit(X, y)
    np.testing.assert_equal(first, model.cv_results_["mean_test_score"])


@pytest.mark.parametrize(
    "action, message",
    [
        (lambda p, e, r: e([]), "nonempty batch"),
        (lambda p, e, r: e([[]]), "coordinates"),
        (lambda p, e, r: e([[-100] * len(p.initial_population[0])]), "parameter space"),
        (lambda p, e, r: e(p.initial_population * 4), "budget"),
        (lambda p, e, r: r([0]), "handles from this fit"),
        (lambda p, e, r: None, "report survivors"),
        (lambda p, e, r: e(p.initial_population), "report survivors"),
    ],
)
def test_plugin_contract_errors(action, message):
    with pytest.raises(ValueError, match=message):
        search("parameters", optimizer_backend=action).fit(*load_iris(return_X_y=True))


def test_plugin_limits_generations():
    def overreport(problem, evaluate, report):
        handles = [h for _, h in evaluate(problem.initial_population)]
        report(handles)
        report(handles)

    with pytest.raises(ValueError, match="generation reports"):
        search("features", generations=0, optimizer_backend=overreport).fit(
            *load_iris(return_X_y=True)
        )


@pytest.mark.parametrize("width,limit", [(256, None), (257, 1)])
def test_plugin_validates_numpy_masks_without_integer_overflow(width, limit):
    X, y = load_iris(return_X_y=True)
    X = np.tile(X[:, :1], (1, width))

    def dense_mask(problem, evaluate, report):
        report([evaluate([np.ones(width, dtype=np.uint8)])[0][1]])

    model = GAFeatureSelectionCV(
        DecisionTreeClassifier(random_state=7),
        cv=2,
        max_features=limit,
        generations=0,
        population_size=2,
        optimizer_backend=dense_mask,
        random_state=7,
        verbose=False,
    )
    if limit is None:
        assert model.fit(X, y).support_.sum() == width
    else:
        with pytest.raises(ValueError, match="max_features"):
            model.fit(X, y)
        assert model.fit_stats_["cross_validate_calls"] == 0


@pytest.mark.parametrize(
    "mask,valid",
    [
        ([0, 1, 0, 1], True),
        ([False, np.int8(1), np.bool_(False), True], True),
        ((False, np.int8(1), np.bool_(False), True), True),
        ([0, 0, 0, 0], False),
        ((0, 0, 0, 0), False),
        ([0, 1, 1, 1], False),
        ([-1, 2, 0, 0], False),
        ([0, 1, 0.0, 1], False),
    ],
)
def test_plugin_enforces_binary_feature_masks_before_cv(mask, valid):
    def optimizer(problem, evaluate, report):
        report([evaluate([mask])[0][1]])

    X, y = load_iris(return_X_y=True)
    model = search("features", optimizer_backend=optimizer, generations=0)
    if valid:
        np.testing.assert_array_equal(model.fit(X, y).support_, mask)
        assert model.fit_counts_["search_cv_calls"] == 1
    else:
        with pytest.raises(ValueError, match="nonempty binary mask within max_features"):
            model.fit(X, y)
        assert model.fit_stats_["cross_validate_calls"] == 0


def test_entry_point_discovery_and_deap_free_fits(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys
    from textwrap import dedent

    # A separately installed distribution's module and standard metadata on sys.path.
    (tmp_path / "example_engine.py").write_text(dedent("""
        def optimize(problem, evaluate, report):
            from random import Random
            rng = Random(problem.seed)
            parents = problem.initial_population
            observed = evaluate(parents)
            report([handle for _, handle in observed])
            for _ in range(problem.generations):
                chosen = min(range(len(parents)), key=lambda i: observed[i][0])
                trials = [parents[chosen], rng.choice(parents)]
                observed_trials = evaluate(trials)
                report([handle for _, handle in observed_trials])
    """))
    info = tmp_path / "example_engine-1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: example-engine\nVersion: 1\n")
    (info / "entry_points.txt").write_text(
        "[sklearn_genetic.optimizers]\nindependent = example_engine:optimize\n"
        "unselected = missing_module:optimize\n"
    )
    # Start before sklearn_genetic has been imported; fail on even a transient DEAP import.
    code = dedent("""
        import pickle
        import sys
        import tempfile
        from pathlib import Path
        class RejectDeap:
            def find_spec(self, fullname, path, target=None):
                if fullname.split(".")[0] == "deap":
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, RejectDeap())
        from sklearn.datasets import load_iris
        from sklearn.tree import DecisionTreeClassifier
        from sklearn_genetic import GAFeatureSelectionCV, GASearchCV
        from sklearn_genetic.space import Integer
        from sklearn_genetic.callbacks import DeltaThreshold, ModelCheckpoint, LogbookSaver
        import numpy as np
        from scipy import __version__ as scipy_version

        X, y = load_iris(return_X_y=True)
        engines = ["independent"]
        if np.lib.NumpyVersion(scipy_version) >= "1.15.0":
            engines.append("scipy")
        for engine in engines:
            opts = dict(cv=2, optimizer_backend=engine, population_size=5,
                        generations=1, random_state=33, verbose=False)
            for model in (GAFeatureSelectionCV(DecisionTreeClassifier(random_state=3), **opts),
                          GASearchCV(DecisionTreeClassifier(random_state=3), param_grid={"max_depth": Integer(1, 12)}, **opts)):
                with tempfile.TemporaryDirectory() as tmp:
                    snapshot = ModelCheckpoint(Path(tmp) / "snapshot.pkl")
                    saver = LogbookSaver(Path(tmp) / "candidates.pkl")
                    model.fit(X, y, callbacks=[DeltaThreshold(2, 2), snapshot, saver])
                    assert len(snapshot.load()["logbook"]) == 2
                assert model.history["gen"] == [0, 1]
                results = model.cv_results_dataframe()
                assert len(results) == (7 if engine == "independent" else 10)
                assert model.fit_stats_["cross_validate_calls"] <= len(results)
                np.testing.assert_equal(model.predict(X), pickle.loads(pickle.dumps(model)).predict(X))
        assert "deap" not in sys.modules
    """)
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(map(str, (tmp_path, root)))}
    subprocess.run([sys.executable, "-c", code], cwd=root, env=env, check=True)


def test_plugin_budget_bounds_generator_consumption():
    def overrun(problem, evaluate, report):
        def rows():
            for _ in range(problem.max_evaluations + 1):
                yield problem.initial_population[0]
            pytest.fail("Read beyond the first excess proposal")

        evaluate(rows())

    with pytest.raises(ValueError, match="budget"):
        search("parameters", generations=0, optimizer_backend=overrun).fit(
            *load_iris(return_X_y=True)
        )


def test_retained_plugin_callbacks_close_after_fit():
    callbacks = []

    def optimize(problem, evaluate, report):
        callbacks.extend([evaluate, report])
        _initial_population_only(problem, evaluate, report)

    model = search("parameters", optimizer_backend=optimize).fit(*load_iris(return_X_y=True))
    for callback in callbacks:
        with pytest.raises(RuntimeError, match="closed"):
            callback([])
    assert len(model.cv_results_["params"]) == 6


@pytest.mark.parametrize(
    "name, expected, message",
    [
        (123, ValueError, "optimizer_backend"),
        ("duplicate", ValueError, "Multiple optimizer"),
        ("invalid", TypeError, "must export a callable"),
    ],
)
def test_invalid_plugin_registration(monkeypatch, name, expected, message):
    from importlib.metadata import EntryPoint
    from sklearn_genetic import backends

    entry = EntryPoint(group="sklearn_genetic.optimizers", name=str(name), value="math:pi")
    monkeypatch.setattr(
        backends, "entry_points", lambda **kw: [entry] * (2 if name == "duplicate" else 1)
    )
    with pytest.raises(expected, match=message):
        search("parameters", optimizer_backend=name).fit(*load_iris(return_X_y=True))


class _ResumableOptimizer:
    """A third-party engine retaining an elite handle and an independent RNG."""

    def __call__(self, problem, evaluate, report):
        rng = random.Random(problem.seed)
        scored = evaluate(problem.initial_population)
        best = min(scored, key=lambda pair: pair[0])
        state = (1, rng.getstate(), best)
        report([handle for _, handle in scored], state=state)
        self.resume(problem, evaluate, report, state)

    def resume(self, problem, evaluate, report, state):
        generation, rng_state, best = state
        rng = random.Random()
        rng.setstate(rng_state)
        for gen in range(generation, problem.generations + 1):
            proposals = rng.choices(problem.initial_population, k=len(problem.initial_population))
            scored = evaluate(proposals)
            best = min([best, *scored], key=lambda pair: pair[0])
            report([best[1], scored[-1][1]], state=(gen + 1, rng.getstate(), best))


@pytest.mark.parametrize("kind", ["parameters", "features"])
def test_plugin_checkpoint_continues_exact_trajectory(kind, tmp_path):
    X, y = load_iris(return_X_y=True)
    options = dict(optimizer_backend=_ResumableOptimizer(), use_cache=False, generations=3)
    expected = search(kind, **options).fit(X, y)
    checkpoint = ModelCheckpoint(tmp_path / "resume.pkl")
    original = search(kind, **options).fit(X, y, callbacks=[checkpoint, ThresholdStopping(0)])
    saved = checkpoint.load()
    assert saved["resumable"]
    assert len(saved["logbook"]) == 1
    assert len(original.history["gen"]) == 1
    original_bytes = checkpoint.checkpoint_path.read_bytes()
    with pytest.raises(ValueError, match="configuration differs"):
        search(kind, **options).fit(X, y[::-1], callbacks=checkpoint)
    assert checkpoint.checkpoint_path.read_bytes() == original_bytes
    result = search(kind, **options).fit(X, y, callbacks=checkpoint)
    assert result.history == expected.history
    assert result.fit_stats_ == expected.fit_stats_
    for key in ("features" if kind == "features" else "params", "mean_test_score"):
        np.testing.assert_equal(result.cv_results_[key], expected.cv_results_[key])
    np.testing.assert_equal(result.predict(X), expected.predict(X))
    assert len(checkpoint.load()["logbook"]) == 4


@pytest.mark.parametrize("kind", ["parameters", "features"])
@pytest.mark.parametrize("engine", ["scipy", _ResumableOptimizer()])
def test_callbacks_share_generation_events(kind, engine, tmp_path):
    if engine == "scipy" and np.lib.NumpyVersion(scipy_version) < "1.15.0":
        pytest.skip("requires the SciPy optimizer extra")
    from sklearn_genetic.callbacks import ConsecutiveStopping, DeltaThreshold, LogbookSaver
    from joblib import load

    X, y = load_iris(return_X_y=True)
    events = []

    class Lifecycle(BaseCallback):
        def on_start(self, estimator):
            events.append("start")

        def on_step(self, record, logbook, estimator):
            assert record is logbook[-1]
            assert logbook.select("gen") == list(range(record["gen"] + 1))
            assert logbook.select("gen", "fitness")[1][-1] == record["fitness"]
            events.append(record["gen"])

        def on_end(self, logbook, estimator):
            events.append("end")

    model = search(kind, optimizer_backend=engine, generations=5)
    path = tmp_path / "candidates.pkl"
    model.fit(
        X,
        y,
        callbacks=[Lifecycle(), ConsecutiveStopping(2), DeltaThreshold(10, 2), LogbookSaver(path)],
    )
    assert events == ["start", 0, 1, "end"]
    assert len(load(path)) == model.fit_stats_["evaluated_candidates"] == 12
    assert "logbook" not in vars(model)


@pytest.mark.parametrize("kind", ["parameters", "features"])
@pytest.mark.parametrize(
    "engine",
    ["deap", pytest.param("scipy", marks=requires_scipy_backend), "nevergrad", "cmaes", "pymoo"],
)
def test_logbook_saver_keeps_previous_output_on_partial_write(
    kind, engine, tmp_path, monkeypatch, caplog
):
    from joblib import load
    from sklearn_genetic.callbacks import LogbookSaver, loggers

    if engine != "deap":
        pytest.importorskip(engine)
    X, y = load_iris(return_X_y=True)
    path = tmp_path / ("candidates.pkl.gz" if kind == "features" else "candidates.pkl")
    saver = LogbookSaver(path)
    dump = loggers.dump
    saved_before_failure = []
    sizes = []

    def partial_write(records, filename, **options):
        sizes.append(len(records))
        if len(sizes) == 1:
            return dump(records, filename, **options)
        saved_before_failure.append(path.read_bytes())
        Path(filename).write_bytes(b"incomplete logbook")
        raise OSError("simulated storage failure")

    backend = scipy_optimizer(stop_on_convergence=False) if engine == "scipy" else engine
    model = search(kind, optimizer_backend=backend, use_cache=False, generations=1)
    with monkeypatch.context() as scoped:
        scoped.setattr(loggers, "dump", partial_write)
        model.fit(X, y, callbacks=saver)
    assert sizes[0] == 6 and len(sizes) == 2 and sizes[1] > sizes[0]
    assert path.read_bytes() == saved_before_failure[0]
    assert len(load(path)) == 6
    if path.suffix == ".gz":
        assert path.read_bytes().startswith(b"\x1f\x8b")
    assert set(tmp_path.iterdir()) == {path}
    assert model.predict(X).shape == y.shape
    assert str(path) in caplog.text
    assert "simulated storage failure" in caplog.text

    assert not saver.on_step(estimator=model)
    assert len(load(path)) == sizes[-1] == model.fit_stats_["evaluated_candidates"]
    assert set(tmp_path.iterdir()) == {path}


@pytest.mark.parametrize("kind", ["parameters", "features"])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_timer_stops_inside_batch_before_refit(kind, n_jobs, tmp_path):
    class Expired(TimerStopping):
        def _time_exceeded(self):
            return True

    X, y = load_iris(return_X_y=True)
    checkpoint = ModelCheckpoint(tmp_path / "partial.pkl")
    est = search(kind, optimizer_backend=_ResumableOptimizer(), use_cache=False, n_jobs=n_jobs).fit(
        X, y, callbacks=[Expired(1), checkpoint]
    )
    assert est.history["partial"] == [True]
    assert est.history["population_size"] == [n_jobs]
    assert est.fit_stats_["cross_validate_calls"] == n_jobs
    assert len(est.predict(X)) == len(X)
    assert checkpoint.load()["resumable"] is False
    assert checkpoint.resumable_cv_calls_ == 0
    assert checkpoint.unsaved_cv_calls_ == n_jobs
    assert "_optimizer_snapshot" not in vars(est)
    assert "_evaluation_timers" not in vars(est)
    assert "_population_pool" not in vars(est)


def test_timer_keeps_last_complete_solver_state(tmp_path):
    class Deadline(TimerStopping):
        def on_start(self, estimator):
            self.armed = False
            self.checks = 0

        def on_step(self, record, logbook, estimator):
            self.armed = True
            return False

        def _time_exceeded(self):
            if self.armed:
                self.checks += 1
            return self.checks > 2

    X, y = load_iris(return_X_y=True)
    checkpoint = ModelCheckpoint(tmp_path / "resume.pkl")
    model = search("parameters", optimizer_backend=_ResumableOptimizer(), use_cache=False)
    model.fit(X, y, callbacks=[Deadline(1), checkpoint])
    assert model.history["partial"] == [False, True]
    assert model.fit_stats_["evaluated_candidates"] == 7
    assert checkpoint.load()["resumable"]
    assert len(checkpoint.load()["candidate_records"]) == 6
    assert model.fit(X, y, callbacks=checkpoint).history["gen"] == [0, 1, 2]


def test_plugin_resume_uses_original_folds_and_mutable_rng(tmp_path):
    from sklearn.model_selection import KFold

    X, y = load_iris(return_X_y=True)

    def make():
        return GAFeatureSelectionCV(
            DecisionTreeClassifier(random_state=7),
            cv=KFold(2, shuffle=True, random_state=np.random.RandomState(19)),
            optimizer_backend=_ResumableOptimizer(),
            population_size=5,
            generations=2,
            random_state=5,
            final_selection="reuse_folds",
            error_score="raise",
            verbose=False,
        )

    reference = make().fit(X, y)
    checkpoint = ModelCheckpoint(tmp_path / "folds.pkl")
    make().fit(X, y, callbacks=[ThresholdStopping(0), checkpoint])
    for restored in (make(), make()):
        restored.fit(X, y, callbacks=checkpoint)
        assert restored.history == reference.history
        assert restored.fit_stats_ == reference.fit_stats_
        assert restored.best_score_ == reference.best_score_
        assert all(
            row["reused_folds"] == 2 for row in restored.final_selection_results_["candidates"]
        )


def test_first_existing_checkpoint_is_loaded_and_formats_are_checked(tmp_path):
    X, y = load_iris(return_X_y=True)
    absent = ModelCheckpoint(tmp_path / "absent.pkl")
    saved = ModelCheckpoint(tmp_path / "saved.pkl")
    search("parameters").fit(X, y, callbacks=[saved, ThresholdStopping(0)])
    with pytest.raises(ValueError, match="historical DEAP"):
        search("parameters", optimizer_backend=_ResumableOptimizer()).fit(
            X, y, callbacks=[absent, saved]
        )
    assert not absent.checkpoint_path.exists()
    model = search("parameters", optimizer_backend=_ResumableOptimizer())
    model.fit(X, y, callbacks=absent)
    with pytest.raises(ValueError, match="cannot resume a DEAP"):
        search("parameters").fit(X, y, callbacks=absent)
    contents = absent.load()
    contents["optimizer_checkpoint_version"] = 2
    absent.checkpoint_path.write_bytes(pickle.dumps(contents))
    with pytest.raises(ValueError, match="Unsupported optimizer checkpoint version"):
        model.fit(X, y, callbacks=absent)


@pytest.mark.parametrize(
    "payload", [None, {}, [], [1], False, {"unrelated": 1}, {"estimator_state": []}]
)
@pytest.mark.parametrize("kind", ["parameters", "features"])
@pytest.mark.parametrize("plugin", [False, True])
def test_malformed_existing_checkpoint_fails_before_cv(
    payload, kind, plugin, tmp_path, monkeypatch
):
    from sklearn_genetic import genetic_search

    fresh = ModelCheckpoint(tmp_path / "fresh.pkl")
    existing = ModelCheckpoint(tmp_path / "invalid.pkl")
    contents = pickle.dumps(payload)
    existing.checkpoint_path.write_bytes(contents)
    engine = _ResumableOptimizer() if plugin else "deap"
    model = search(kind, optimizer_backend=engine)
    monkeypatch.setattr(
        genetic_search, "cross_validate", lambda *args, **kwargs: pytest.fail("unexpected CV")
    )

    with pytest.raises(ValueError, match=r"Invalid search checkpoint.*invalid\.pkl"):
        model.fit(*load_iris(return_X_y=True), callbacks=[fresh, existing])

    assert existing.checkpoint_path.read_bytes() == contents
    assert not fresh.checkpoint_path.exists()


def test_resumable_plugin_requires_state_only_with_checkpoint(tmp_path):
    class MissingState(_ResumableOptimizer):
        def resume(self, problem, evaluate, report, state):
            report([state[-1][1]])

    model = search("parameters", optimizer_backend=MissingState())
    X, y = load_iris(return_X_y=True)
    assert model.fit(X, y).history["gen"] == [0, 1]
    with pytest.raises(ValueError, match="serializable search settings"):
        model.fit(X, y, callbacks=ModelCheckpoint(tmp_path / "local.pkl"))


@pytest.mark.parametrize("kind", ["parameters", "features"])
@pytest.mark.parametrize(
    "field",
    [
        "resumable",
        "backend",
        "context",
        "problem",
        "solver_state",
        "owner",
        "elites",
        "logbook",
        "candidate_records",
        "fitness_cache",
        "fit_stats",
        "cv_splits",
        "telemetry",
        "best_observed",
        "rng",
        "fit_stats.cross_validate_calls",
        "fit_stats.population_serial_batches",
        "fit_stats.evaluated_candidates",
    ],
)
def test_incomplete_optimizer_checkpoint_names_missing_field(field, kind, tmp_path, monkeypatch):
    from sklearn_genetic import genetic_search

    X, y = load_iris(return_X_y=True)
    checkpoint = ModelCheckpoint(tmp_path / "incomplete.pkl")
    model = search(kind, optimizer_backend=_ResumableOptimizer(), use_cache=False)
    clone(model).fit(X, y, callbacks=[CVBudgetStopping(6), checkpoint])
    snapshot = checkpoint.load()
    if field.startswith("fit_stats."):
        del snapshot["fit_stats"][field.split(".", 1)[1]]
    else:
        del snapshot[field]
    contents = pickle.dumps(snapshot)
    checkpoint.checkpoint_path.write_bytes(contents)
    monkeypatch.setattr(
        genetic_search, "cross_validate", lambda *args, **kwargs: pytest.fail("unexpected CV")
    )

    with pytest.raises(ValueError, match=r"Invalid search checkpoint.*incomplete\.pkl") as error:
        model.fit(X, y, callbacks=checkpoint)

    assert field in str(error.value)
    assert checkpoint.checkpoint_path.read_bytes() == contents


@pytest.mark.parametrize("engine", ["scipy", "nevergrad", "cmaes", "pymoo"])
@pytest.mark.parametrize("kind", ["parameters", "features"])
@pytest.mark.parametrize("value", [-1, True, 1.5, None])
def test_invalid_optimizer_checkpoint_counter_is_rejected_before_cv(
    engine, kind, value, tmp_path, monkeypatch
):
    from sklearn_genetic import genetic_search

    pytest.importorskip(engine, minversion="1.15" if engine == "scipy" else None)
    X, y = load_iris(return_X_y=True)
    checkpoint = ModelCheckpoint(tmp_path / "bad-counter.pkl")
    model = search(kind, optimizer_backend=engine, use_cache=False, n_jobs=1)
    clone(model).fit(X, y, callbacks=[CVBudgetStopping(6), checkpoint])
    snapshot = checkpoint.load()
    snapshot["fit_stats"]["cache_hits"] = value
    contents = pickle.dumps(snapshot)
    checkpoint.checkpoint_path.write_bytes(contents)
    monkeypatch.setattr(
        genetic_search, "cross_validate", lambda *args, **kwargs: pytest.fail("unexpected CV")
    )

    with pytest.raises(ValueError, match=r"bad-counter\.pkl.*fit_stats\.cache_hits.*integer"):
        clone(model).fit(X, y, callbacks=[CVBudgetStopping(3), checkpoint])

    assert checkpoint.checkpoint_path.read_bytes() == contents


class _MissingContinuation(_ResumableOptimizer):
    def resume(self, problem, evaluate, report, state):
        report([state[-1][1]])


def test_missing_continuation_preserves_valid_checkpoint(tmp_path):
    path = ModelCheckpoint(tmp_path / "state.pkl")
    est = search("parameters", optimizer_backend=_MissingContinuation())
    with pytest.raises(ValueError, match="non-None state"):
        est.fit(*load_iris(return_X_y=True), callbacks=path)
    assert path.load()["resumable"]
    assert len(path.load()["logbook"]) == 1


def test_resume_preserves_identity_keyed_categorical_cache(tmp_path):
    from sklearn.dummy import DummyClassifier
    from sklearn.pipeline import Pipeline

    X, y = load_iris(return_X_y=True)

    def make():
        models = [DecisionTreeClassifier(random_state=7), DummyClassifier()]
        return GASearchCV(
            Pipeline([("model", models[0])]),
            cv=2,
            param_grid={"model": Categorical(models)},
            population_size=5,
            generations=2,
            optimizer_backend=_ResumableOptimizer(),
            random_state=9,
            verbose=False,
        )

    reference = make().fit(X, y)
    checkpoint = ModelCheckpoint(tmp_path / "categories.pkl")
    make().fit(X, y, callbacks=[checkpoint, ThresholdStopping(0)])
    recovered = make().fit(X, y, callbacks=checkpoint)
    assert recovered.fit_stats_ == reference.fit_stats_
    assert recovered.history == reference.history
    np.testing.assert_equal(recovered.predict(X), reference.predict(X))


class _GlobalRngOptimizer(_ResumableOptimizer):
    def resume(self, problem, evaluate, report, state):
        next_gen, _, best = state
        for gen in range(next_gen, problem.generations + 1):
            proposals = [
                random.choice(problem.initial_population) for _ in problem.initial_population
            ]
            proposals[0] = problem.initial_population[np.random.randint(len(proposals))]
            scored = evaluate(proposals)
            report([best[1], scored[0][1]], state=(gen + 1, None, best))


def test_resume_restores_proposal_rng_after_callback_lifecycle(tmp_path):
    class Observer(BaseCallback):
        def on_start(self, estimator):
            random.random()
            np.random.rand()

        def on_step(self, record, logbook, estimator):
            self.on_start(estimator)

    def make():
        return search("parameters", optimizer_backend=_GlobalRngOptimizer(), use_cache=False)

    X, y = load_iris(return_X_y=True)
    reference = make().fit(X, y, callbacks=Observer())
    checkpoint = ModelCheckpoint(tmp_path / "rng.pkl")
    make().fit(X, y, callbacks=[checkpoint, Observer(), ThresholdStopping(0)])
    restored = make().fit(X, y, callbacks=[checkpoint, Observer()])
    assert restored.cv_results_["params"] == reference.cv_results_["params"]
    assert restored.history == reference.history
