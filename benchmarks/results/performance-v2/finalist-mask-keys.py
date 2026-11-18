"""Measure Boolean mask key construction within the actual finalist ranking."""

import argparse
import json
from statistics import median
from time import perf_counter
from timeit import timeit

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from sklearn_genetic import GAFeatureSelectionCV


class TupleRanking(GAFeatureSelectionCV):
    def _finalist_key(self, index):
        return tuple(sorted(self._candidate_params_from_index(index).items()))


def measure(width, copies):
    rng = np.random.default_rng(9110 + width)
    masks = [rng.random(width) < 0.25 for _ in range(64)]
    ranks = rng.integers(1, 8, size=len(masks))
    results = dict(
        features=[mask for mask in masks for _ in range(copies)],
        rank_test_score=np.repeat(ranks, copies),
        n_features=np.repeat([mask.sum() for mask in masks], copies),
    )
    searches = [
        cls(DummyClassifier(), final_selection_top_k=8)
        for cls in (TupleRanking, GAFeatureSelectionCV)
    ]
    for search in searches:
        search.cv_results_ = results
        search.refit_metric = "score"
    expected, actual = [s._top_candidate_indices() for s in searches]
    assert expected == actual
    times = [[], []]
    for repeat in range(9):
        for i in ((0, 1) if repeat % 2 else (1, 0)):
            times[i].append(timeit(searches[i]._top_candidate_indices, number=10) / 10)
    return dict(
        width=width,
        copies=copies,
        rows=len(results["features"]),
        winner_indices=[int(i) for i in actual],
        median_us=[median(t) * 1e6 for t in times],
    )


def paired_fit(width, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((240, width))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    folds = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(X[:160], y[:160]))
    methods = (TupleRanking, GAFeatureSelectionCV)
    rows = {}
    outcomes = []
    for cls in methods if seed % 2 else reversed(methods):
        search = cls(
            DecisionTreeClassifier(max_depth=3, random_state=seed),
            max_features=16,
            cv=folds[:1],
            final_selection_cv=folds,
            final_selection="reuse_folds",
            error_score="raise",
            final_selection_top_k=8,
            scoring="accuracy",
            n_jobs=1,
            random_state=seed,
            population_size=8,
            generations=3,
            verbose=False,
        )
        started = perf_counter()
        search.fit(X[:160], y[:160])
        elapsed = perf_counter() - started
        finalists = search.final_selection_results_["candidates"]
        outcomes.append(
            (
                [row["index"] for row in finalists],
                [row["cv_scores"] for row in finalists],
                search.best_features_.tolist(),
                search.predict(X[160:]).tolist(),
                search.fit_stats_["cross_validate_calls"],
            )
        )
        rows[cls.__name__] = dict(
            fit_seconds=elapsed,
            ranking_seconds=median(
                timeit(search._top_candidate_indices, number=3) / 3 for _ in range(3)
            ),
        )
    assert outcomes[0] == outcomes[1]
    return dict(width=width, seed=seed, cv_calls=outcomes[0][-1], runs=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-fit", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(1):
        rows = (
            [paired_fit(w, s) for w in (256, 4096) for s in range(9141, 9147)]
            if args.full_fit
            else [measure(w, c) for w in (16, 256, 4096) for c in (1, 8)]
        )
    print(json.dumps(rows, indent=2))
