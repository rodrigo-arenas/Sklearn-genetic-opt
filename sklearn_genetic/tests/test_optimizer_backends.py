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


def _initial_population_only(problem, evaluate, report):
    # One evaluation per initial row. Different rows can have identical scores.
    scored = evaluate(problem.initial_population)
    report([handle for _, handle in scored])


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
        class RejectDeap:
            def find_spec(self, fullname, path, target=None):
                if fullname.split(".")[0] == "deap":
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, RejectDeap())
        from sklearn.datasets import load_iris
        from sklearn.tree import DecisionTreeClassifier
        from sklearn_genetic import GAFeatureSelectionCV, GASearchCV
        from sklearn_genetic.space import Integer
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
                          GASearchCV(DecisionTreeClassifier(random_state=3), param_grid={"max_depth": Integer(1, 5)}, **opts)):
                model.fit(X, y)
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
