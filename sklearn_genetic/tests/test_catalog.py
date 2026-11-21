from functools import partial
import importlib
import pickle
from pathlib import Path
import subprocess
from textwrap import dedent
import sys

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import load_iris

from sklearn_genetic.callbacks import BaseCallback, ModelCheckpoint, ThresholdStopping
from sklearn_genetic.space import Categorical, Continuous, Integer
from sklearn_genetic.tests.test_optimizer_backends import search

BACKENDS = ["nevergrad", "cmaes", "pymoo"]


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


def test_nevergrad_resume_preserves_cv_rng_in_fresh_process(tmp_path):
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

        directory, phase = Path(sys.argv[1]), sys.argv[2]
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
        model = search("parameters", optimizer_backend="nevergrad", scoring=score, use_cache=False)
        model.fit(*load_iris(return_X_y=True), callbacks=callbacks)
        (directory / phase).write_text(json.dumps(list(model.cv_results_["mean_test_score"])))
    """)
    for phase in ("control", "crash", "resume"):
        child = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path), phase],
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
