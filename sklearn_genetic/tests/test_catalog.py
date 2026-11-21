import importlib
import sys

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import load_iris

from sklearn_genetic.callbacks import ModelCheckpoint, ThresholdStopping
from sklearn_genetic.space import Categorical, Continuous, Integer
from sklearn_genetic.tests.test_optimizer_backends import search

BACKENDS = ["nevergrad", "cmaes"]


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
    assert checkpoint.load()["resumable"] is False


@pytest.mark.parametrize("backend", BACKENDS)
def test_missing_extra(backend, monkeypatch):
    monkeypatch.setitem(sys.modules, backend, None)
    adapter = importlib.import_module(f"sklearn_genetic._{backend}_backend")
    with pytest.raises(ImportError, match=rf"sklearn-genetic-opt\[{backend}\]"):
        adapter.optimize(None, None, None)


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
