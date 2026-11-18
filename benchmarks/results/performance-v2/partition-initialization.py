"""Compare initial feature masks with the same downstream genetic algorithm."""

import argparse
import json
from pathlib import Path
from random import Random
from runpy import run_path
from time import perf_counter

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from sklearn_genetic import GAFeatureSelectionCV


class FourColumnStart(GAFeatureSelectionCV):
    partitioned = False
    mask_size = 4

    def _initialize_population(self):
        population = super()._initialize_population()
        # Keep evolution's RNG state identical at the end of initialization.
        rng = Random(self.random_state)
        width = len(population[0])
        masks = {}
        while len(masks) < len(population):
            perm = rng.sample(range(width), width)
            stop = width - self.mask_size + 1 if self.partitioned else 1
            for offset in range(0, stop, self.mask_size):
                masks[tuple(sorted(perm[offset : offset + self.mask_size]))] = None
        for individual, columns in zip(population, masks):
            individual[:] = [int(i in columns) for i in range(width)]
        return population


class PartitionStart(FourColumnStart):
    partitioned = True


class TwoColumnStart(FourColumnStart):
    mask_size = 2


class BalancedSmartStart(GAFeatureSelectionCV):
    def _initialize_population(self):
        population = super()._initialize_population()
        width = len(population[0])
        exposure = [0] * width
        rng = Random(self.random_state)
        for individual in population:
            order = rng.sample(range(width), width)
            order.sort(key=exposure.__getitem__)
            chosen = set(order[: sum(individual)])
            individual[:] = [int(i in chosen) for i in range(width)]
            for i in chosen:
                exposure[i] += 1
        assert max(exposure) - min(exposure) <= 1
        return population


def compare(
    X,
    y,
    test,
    labels,
    cv,
    model,
    metric,
    seed,
    *args,
    target=None,
    balanced=False,
    signal=None,
    methods=None,
    **kwargs,
):
    rows = {}
    if methods is None:
        methods = [
            GAFeatureSelectionCV,
            FourColumnStart,
            PartitionStart if target is None else TwoColumnStart,
        ]
        if balanced:
            methods = [GAFeatureSelectionCV, BalancedSmartStart]
    methods = list(methods)
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
        if hasattr(search, "union_count_"):
            rows[cls.__name__]["full_fit_unions"] = search.union_count_
        if balanced:
            initial = list(observed)[:6]
            rows[cls.__name__]["initial_sizes"] = [len(mask) for mask in initial]
            rows[cls.__name__]["initial_coverage"] = len(set().union(*initial))
            prefix = list(observed)[:24]
            rows[cls.__name__]["signal_trace"] = [
                [col for col in signal if col in mask] for mask in prefix
            ]
            rows[cls.__name__]["columns_seen_by_24"] = len(set().union(*prefix))
            rows[cls.__name__]["winning_signals_at_24"] = sorted(set(winner).intersection(signal))
        if target is not None:
            rows[cls.__name__]["first_target_by_24"] = next(
                (i for i, score in enumerate(list(observed.values())[:24], 1) if score >= target),
                None,
            )
    return dict(seed=seed, methods=rows)


def sparse_regression(width, seed, balanced=False, methods=None):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(1100, 64))
    y = features[:, 0] + features[:, 1] + 0.5 * rng.normal(size=len(features))
    order = rng.permutation(width)
    X = features[:, order]
    folds = list(KFold(3, shuffle=True, random_state=seed).split(X[:240]))
    result = compare(
        X[:240],
        y[:240],
        X[240:],
        y[240:],
        folds,
        Ridge(),
        "r2",
        seed,
        target=0.75,
        balanced=balanced,
        methods=methods,
        signal=np.flatnonzero(order < 2).tolist(),
    )
    return dict(width=width, **result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--sparse-regression", action="store_true")
    modes.add_argument("--balanced-coverage", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    run_real = run_path(str(here / "known-support.py"))["run_real"]
    run_real.__globals__["compare"] = compare
    with threadpool_limits(1):
        if args.balanced_coverage:
            rows = [
                sparse_regression(width, seed, balanced=True)
                for width in (16, 64)
                for seed in range(8421, 8429)
            ]
        elif args.sparse_regression:
            rows = [
                sparse_regression(width, seed) for width in (16, 64) for seed in range(8301, 8305)
            ]
        else:
            rows = [
                run_real(task, split, seed)
                for task in ("wine", "diabetes")
                for split in (8231, 8232)
                for seed in (8241, 8242)
            ]
    print(json.dumps(rows, indent=2, allow_nan=False))
