"""Paired engine comparison. Run python -m benchmarks.benchmark_optimizer_backends."""

import json
import platform
from importlib.metadata import version
from time import perf_counter
from unittest.mock import patch

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from sklearn_genetic import GAFeatureSelectionCV, GASearchCV
from sklearn_genetic.space import Categorical, Continuous, Integer


def measure(task, backend, seed):
    X, y = load_breast_cancer(return_X_y=True)
    X, test_X, y, test_y = train_test_split(X, y, stratify=y, test_size=0.25, random_state=seed)
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    options = dict(
        estimator=DecisionTreeClassifier(random_state=seed),
        cv=cv,
        scoring="balanced_accuracy",
        random_state=seed,
        population_size=8,
        generations=2,
        n_jobs=1,
        use_cache=False,
        verbose=False,
        diversity_control=False,
        optimizer_backend=backend,
    )
    if backend == "deap":
        options["algorithm"] = "eaDifferentialEvolution"
    if task == "parameters":
        search = GASearchCV(
            param_grid={
                "max_depth": Integer(2, 10),
                "ccp_alpha": Continuous(1e-5, 1e-1, distribution="log-uniform"),
                "criterion": Categorical(["gini", "entropy"]),
            },
            **options,
        )
    else:
        search = GAFeatureSelectionCV(max_features=12, **options)

    cv_times = []

    def timed_cv(*args, **kwargs):
        start = perf_counter()
        result = cross_validate(*args, **kwargs)
        cv_times.append(perf_counter() - start)
        return result

    with patch("sklearn_genetic.genetic_search.cross_validate", timed_cv):
        start = perf_counter()
        search.fit(X, y)
        elapsed = perf_counter() - start
    if len(cv_times) != 24 or search.fit_stats_["cross_validate_calls"] != len(cv_times):
        raise RuntimeError(
            f"Unmatched evaluation budget: {task}, {backend}, {seed}, {len(cv_times)}"
        )
    return dict(
        task=task,
        backend=backend,
        seed=seed,
        cv_calls=len(cv_times),
        best_cv=float(np.max(search.cv_results_["mean_test_score"])),
        held_out=balanced_accuracy_score(test_y, search.predict(test_X)),
        features=int(search.support_.sum()) if task == "features" else None,
        wall_ms=elapsed * 1000,
        outside_cv_and_refit_ms=(elapsed - sum(cv_times) - search.refit_time_) * 1000,
    )


if __name__ == "__main__":
    rows = []
    with threadpool_limits(1):
        for backend in ("deap", "scipy"):
            measure("parameters", backend, 41)  # warm up imports
        for i, seed in enumerate((17, 29, 43, 61)):
            for task in ("parameters", "features"):
                for backend in (("scipy", "deap") if i % 2 else ("deap", "scipy")):
                    rows.append(measure(task, backend, seed))
    print(
        json.dumps(
            dict(
                python=platform.python_version(),
                packages={
                    name: version(name) for name in ("deap", "scipy", "numpy", "scikit-learn")
                },
                runs=rows,
            ),
            indent=2,
        )
    )
