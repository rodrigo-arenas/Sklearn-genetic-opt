import pickle
import random

import numpy as np
import pytest
from scipy import __version__ as scipy_version
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from sklearn_genetic import GAFeatureSelectionCV, GASearchCV
from sklearn_genetic.callbacks import ModelCheckpoint, ThresholdStopping
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
def test_checkpoint_requires_deap_before_io(kind, tmp_path):
    path = tmp_path / "old.pkl"
    path.write_bytes(b"historical checkpoint placeholder")
    est = search(kind, optimizer_backend="scipy")
    X, y = load_iris(return_X_y=True)
    for callback in (ModelCheckpoint(path), ThresholdStopping(0.9)):
        with pytest.raises(ValueError, match="Callbacks.*deap"):
            est.fit(X, y, callbacks=callback)
    assert path.read_bytes() == b"historical checkpoint placeholder"


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
