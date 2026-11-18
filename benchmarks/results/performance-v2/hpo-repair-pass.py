"""Measure the HPO repair pass immediately inside uncached CV evaluation."""

import inspect
import json
import sys
from textwrap import dedent
from time import perf_counter
from unittest.mock import patch

import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Categorical, Continuous, Integer


def tree_grid():
    return dict(
        criterion=Categorical(["gini", "entropy"]),
        max_depth=Integer(1, 8),
        min_weight_fraction_leaf=Continuous(0.0, 0.25),
        min_samples_leaf=Integer(1, 12),
    )


def run(X, y, seed, evaluate, cache):
    search = GASearchCV(
        DecisionTreeClassifier(random_state=11),
        param_grid=tree_grid(),
        cv=2,
        scoring="accuracy",
        population_size=8,
        generations=2,
        use_cache=cache,
        random_state=seed,
        n_jobs=1,
        verbose=False,
    )
    with patch.object(GASearchCV, "_evaluate_individual", evaluate):
        start = perf_counter()
        search.fit(X, y)
        elapsed = perf_counter() - start
    return search, elapsed


if __name__ == "__main__":
    original = GASearchCV._evaluate_individual
    source = dedent(inspect.getsource(original))
    repair = "    self._repair_individual(individual)\n"
    assert source.count(repair) == 1
    namespace = {}
    exec(source.replace(repair, ""), original.__globals__, namespace)
    variants = {"current": original, "single_pass": namespace["_evaluate_individual"]}
    X, y = load_iris(return_X_y=True)
    rows = []
    with threadpool_limits(1):
        for cache in (True, False):
            for evaluate in variants.values():
                run(X, y, 7, evaluate, cache)
            for seed in (3, 19, 41, 53):
                for order in (list(variants), list(reversed(variants))):
                    fits, times = {}, {}
                    for name in order:
                        fits[name], times[name] = run(X, y, seed, variants[name], cache)
                    a, b = fits.values()
                    assert a.cv_results_["params"] == b.cv_results_["params"]
                    for i in range(2):
                        np.testing.assert_array_equal(
                            a.cv_results_[f"split{i}_test_score"],
                            b.cv_results_[f"split{i}_test_score"],
                        )
                    assert a.fit_stats_ == b.fit_stats_ and a.best_params_ == b.best_params_
                    np.testing.assert_array_equal(a.predict(X), b.predict(X))
                    rows.append(
                        dict(
                            cache=cache,
                            seed=seed,
                            order=order,
                            seconds=times,
                            cv_calls=a.fit_stats_["cross_validate_calls"],
                        )
                    )
        audits = []
        repair_method = GASearchCV._repair_individual
        for cache in (True, False):
            for seed in (3, 19, 41, 53):
                samples = []

                def measured(search, individual):
                    if sys._getframe(1).f_code is original.__code__:
                        before = tuple(individual)
                        start = perf_counter()
                        result = repair_method(search, individual)
                        samples.append(perf_counter() - start)
                        assert before == tuple(individual)
                        return result
                    return repair_method(search, individual)

                with patch.object(GASearchCV, "_repair_individual", measured):
                    fitted, seconds = run(X, y, seed, original, cache)
                assert len(samples) == fitted.fit_stats_["cross_validate_calls"]
                audits.append(
                    dict(
                        seed=seed,
                        cache=cache,
                        calls=len(samples),
                        repair_seconds=sum(samples),
                        fit_seconds=seconds,
                    )
                )
    print(json.dumps(dict(pairs=rows, attribution=audits), indent=2, allow_nan=False))
