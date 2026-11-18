"""Measure where sequential searches spend time in the finite Wine tree grid."""

import argparse
import json
from contextlib import ExitStack
from functools import wraps
from time import perf_counter
from unittest.mock import patch

from sklearn.datasets import load_wine
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.model_selection import _search, _validation
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from sklearn_genetic import GASearchCV, genetic_search
from sklearn_genetic.space import Categorical, Integer


def instrument(function, samples):
    @wraps(function)
    def timed(*args, **kwargs):
        start = perf_counter()
        result = function(*args, **kwargs)
        samples.append(perf_counter() - start)
        return result

    return timed


def run(seed, genetic, X, y, folds, breakdown=False):
    common = dict(cv=folds, scoring="accuracy", n_jobs=1, random_state=seed, error_score="raise")
    tree = DecisionTreeClassifier(random_state=0)
    if genetic:
        search = GASearchCV(
            tree,
            param_grid={"max_depth": Integer(2, 15), "criterion": Categorical(["gini", "entropy"])},
            population_size=8,
            generations=4,
            verbose=False,
            **common,
        )
    else:
        search = RandomizedSearchCV(
            tree,
            {"max_depth": list(range(2, 16)), "criterion": ["gini", "entropy"]},
            n_iter=28,
            **common,
        )
    fold_times, candidate_times = [], []
    stages = {}
    with ExitStack() as stack:
        module = _validation if genetic else _search
        stack.enter_context(
            patch.object(module, "_fit_and_score", instrument(module._fit_and_score, fold_times))
        )
        if genetic:
            if breakdown:
                for name in (
                    "indexable",
                    "is_classifier",
                    "check_cv",
                    "check_scoring",
                    "_manual_routing",
                    "clone",
                    "Parallel",
                    "_aggregate_score_dicts",
                    "_normalize_score_results",
                ):
                    stages[name] = []
                    stack.enter_context(
                        patch.object(
                            _validation, name, instrument(getattr(_validation, name), stages[name])
                        )
                    )
            stack.enter_context(
                patch.object(
                    genetic_search,
                    "cross_validate",
                    instrument(genetic_search.cross_validate, candidate_times),
                )
            )
        start = perf_counter()
        search.fit(X, y)
        total = perf_counter() - start
    candidates = (
        search.fit_stats_["cross_validate_calls"] if genetic else len(search.cv_results_["params"])
    )
    assert len(fold_times) == 3 * candidates
    if genetic:
        assert len(candidate_times) == candidates
    result = dict(
        seed=seed,
        method="genetic" if genetic else "random",
        candidates=candidates,
        fit_seconds=total,
        fold_seconds=sum(fold_times),
        candidate_cv_seconds=sum(candidate_times) if genetic else None,
    )
    if stages:
        assert len(stages["clone"]) == len(fold_times)
        result["stages"] = {k: dict(calls=len(v), seconds=sum(v)) for k, v in stages.items()}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--breakdown", action="store_true")
    args = parser.parse_args()
    X, y = load_wine(return_X_y=True)
    rows = []
    with threadpool_limits(1):
        for seed in range(1091, 1096):
            train, _, labels, _ = train_test_split(
                X, y, test_size=0.25, stratify=y, random_state=seed
            )
            folds = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(train, labels))
            for genetic in (bool(seed % 2), not bool(seed % 2)):
                rows.append(run(seed, genetic, train, labels, folds, args.breakdown))
    print(json.dumps(rows, indent=2))
