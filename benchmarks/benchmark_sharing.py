"""Benchmark original pairwise sharing versus bounded vectorized sharing."""

import json
import time
from types import SimpleNamespace
import numpy as np
from sklearn_genetic.optimizer_control import niche_counts, individual_distance, sharing_value
from sklearn_genetic.space import Space, Continuous, Integer, Categorical
from sklearn_genetic import GASearchCV, GAFeatureSelectionCV
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits
import sklearn_genetic.optimizer_control as control


def reference(population, estimator):
    # Literal accumulation from d6b2f3a: Python 3.12's sum() compensates
    # floating-point error and is not equivalent to repeated += here.
    counts = []
    for left in population:
        count = 0.0
        for right in population:
            count += sharing_value(
                individual_distance(estimator, left, right),
                estimator.sharing_radius,
                estimator.sharing_alpha,
            )
        counts.append(max(count, 1.0))
    return counts


def main():
    rows = []

    rng = np.random.default_rng(42)
    for size in (50, 150, 300):
        for kind in ("features", "mixed"):
            estimator = SimpleNamespace(sharing_radius=0.8, sharing_alpha=1.0)
            if kind == "features":
                population = rng.integers(0, 2, size=(size, 100)).tolist()
            else:
                estimator.space = Space(
                    {"x": Continuous(0.0, 1.0), "y": Integer(0, 10), "z": Categorical(["a", "b"])}
                )
                population = [
                    [float(rng.random()), int(rng.integers(0, 11)), str(rng.choice(["a", "b"]))]
                    for _ in range(size)
                ]
            times = {}
            results = {}
            for name, fn in (("before", reference), ("after", niche_counts)):
                durations = []
                for _ in range(3):
                    start = time.perf_counter()
                    results[name] = fn(population, estimator)
                    durations.append(time.perf_counter() - start)
                times[name] = float(np.median(durations))
            np.testing.assert_allclose(results["before"], results["after"], rtol=1e-12, atol=1e-12)
            rows.append(
                dict(size=size, kind=kind, **times, speedup=times["before"] / times["after"])
            )
    with threadpool_limits(limits=1):
        X, y = make_classification(n_samples=300, n_features=80, n_informative=8, random_state=240)
        for kind in ("features", "hpo"):
            for seed in (201, 202):
                times = {}
                scores = {}
                evaluations = {}
                for name, fn in (("before", reference), ("after", niche_counts)):
                    control.niche_counts = fn
                    kwargs = dict(
                        population_size=50,
                        generations=4,
                        n_jobs=1,
                        cv=3,
                        random_state=seed,
                        fitness_sharing=True,
                        sharing_radius=0.8,
                        verbose=False,
                    )
                    model = DecisionTreeClassifier(max_depth=4, random_state=seed)
                    if kind == "features":
                        search = GAFeatureSelectionCV(model, max_features=40, **kwargs)
                    else:
                        search = GASearchCV(
                            model,
                            param_grid={
                                "max_depth": Integer(2, 10),
                                "min_samples_leaf": Integer(1, 12),
                                "max_features": Continuous(0.1, 1.0),
                                "criterion": Categorical(["gini", "entropy"]),
                            },
                            **kwargs,
                        )
                    start = time.perf_counter()
                    search.fit(X, y)
                    times[name] = time.perf_counter() - start
                    scores[name] = float(max(search.cv_results_["mean_test_score"]))
                    evaluations[name] = search.fit_stats_["cross_validate_calls"]
                rows.append(
                    dict(
                        kind=kind,
                        seed=seed,
                        scope="fit",
                        times=times,
                        scores=scores,
                        evaluations=evaluations,
                        speedup=times["before"] / times["after"],
                    )
                )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
