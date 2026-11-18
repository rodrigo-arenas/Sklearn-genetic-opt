"""Measure Boolean mask key construction within the actual finalist ranking."""

import json
from statistics import median
from timeit import timeit

import numpy as np
from sklearn.dummy import DummyClassifier

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


if __name__ == "__main__":
    print(json.dumps([measure(w, c) for w in (16, 256, 4096) for c in (1, 8)], indent=2))
