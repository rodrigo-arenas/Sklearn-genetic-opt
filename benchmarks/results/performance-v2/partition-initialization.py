"""Compare initial feature masks with the same downstream genetic algorithm."""

import json
from pathlib import Path
from random import Random
from runpy import run_path
from time import perf_counter

import numpy as np
from sklearn.base import clone
from sklearn.metrics import get_scorer
from threadpoolctl import threadpool_limits

from sklearn_genetic import GAFeatureSelectionCV


class FourColumnStart(GAFeatureSelectionCV):
    partitioned = False

    def _initialize_population(self):
        population = super()._initialize_population()
        # Keep evolution's RNG state identical at the end of initialization.
        rng = Random(self.random_state)
        width = len(population[0])
        masks = {}
        while len(masks) < len(population):
            perm = rng.sample(range(width), width)
            for offset in range(0, width - 3 if self.partitioned else 1, 4):
                masks[tuple(sorted(perm[offset : offset + 4]))] = None
        for individual, columns in zip(population, masks):
            individual[:] = [int(i in columns) for i in range(width)]
        return population


class PartitionStart(FourColumnStart):
    partitioned = True


def compare(X, y, test, labels, cv, model, metric, seed, *args, **kwargs):
    rows = {}
    methods = [GAFeatureSelectionCV, FourColumnStart, PartitionStart]
    Random(seed).shuffle(methods)
    for cls in methods:
        search = cls(
            clone(model),
            max_features=4,
            population_size=6,
            generations=3,
            scoring=metric,
            cv=cv,
            random_state=seed,
            error_score="raise",
            n_jobs=1,
            verbose=False,
        )
        start = perf_counter()
        search.fit(X, y)
        elapsed = perf_counter() - start
        observed = {}
        for mask, score in zip(
            search.cv_results_["features"], search.cv_results_["mean_test_score"]
        ):
            observed.setdefault(tuple(np.flatnonzero(mask).tolist()), score)
        assert len(observed) == search.fit_stats_["cross_validate_calls"] and len(observed) >= 24
        scores = {}
        for budget in (6, 24):
            winner = max(list(observed)[:budget], key=lambda m: (observed[m], -len(m)))
            fitted = clone(model).fit(X[:, winner], y)
            scores[budget] = get_scorer(metric)(fitted, test[:, winner], labels)
        rows[cls.__name__] = dict(
            candidates=len(observed), full_fit_seconds=elapsed, holdout=scores
        )
    return dict(seed=seed, methods=rows)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    run_real = run_path(str(here / "known-support.py"))["run_real"]
    run_real.__globals__["compare"] = compare
    with threadpool_limits(1):
        rows = [
            run_real(task, split, seed)
            for task in ("wine", "diabetes")
            for split in (8231, 8232)
            for seed in (8241, 8242)
        ]
    print(json.dumps(rows, indent=2, allow_nan=False))
