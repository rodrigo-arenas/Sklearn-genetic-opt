"""Replay finalist CV with a score ceiling of one and one reusable fold."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from benchmarks.benchmark_fit import SCENARIOS, FitCounters, count_cross_validate_calls, make_cv
from sklearn_genetic import GAFeatureSelectionCV, GASearchCV


def replay(search, features):
    best = (-np.inf, 0)
    winner = None
    reads = []
    rows = search.final_selection_results_["candidates"]
    for row in rows:
        scores = row["cv_scores"]
        assert row["reused_folds"] == 1 and all(np.isfinite(s) and s <= 1 for s in scores)
        for n in range(1, len(scores) + 1):
            optimistic = np.mean(scores[:n] + [1.0] * (len(scores) - n))
            if optimistic < best[0]:
                break
            if n == len(scores):
                rank = (optimistic, -sum(row["params"]["features"]) if features else 0)
                if rank > best:
                    best, winner = rank, row["index"]
        reads.append(n - 1)
    assert (winner, best[0]) == (search.best_index_, search.best_score_)
    return dict(
        winner=winner,
        score=best[0],
        new_folds_read=reads,
        full_new_folds=sum(len(row["cv_scores"]) - 1 for row in rows),
    )


def promote(search, features, train, labels, test, target, metric):
    rows = search.final_selection_results_["candidates"]
    assert len(rows) == 8 and all(len(r["cv_scores"]) == 5 and r["reused_folds"] == 1 for r in rows)

    def rank(row, folds):
        return (
            np.mean(row["cv_scores"][:folds]),
            -sum(row["params"]["features"]) if features else 0,
        )

    top = sorted(range(8), key=lambda i: rank(rows[i], 2), reverse=True)[:4]
    arms = {}
    for name, pool in (
        ("full", rows),
        ("promoted", [rows[i] for i in sorted(top)]),
        ("first_five", rows[:5]),
    ):
        winner = max(pool, key=lambda row: rank(row, 5))
        model = clone(search.estimator)
        X, holdout = train, test
        if features:
            mask = np.asarray(winner["params"]["features"], dtype=bool)
            X, holdout = train[:, mask], test[:, mask]
        else:
            model.set_params(**winner["params"])
        score = get_scorer(metric)(model.fit(X, labels), holdout, target)
        arms[name] = dict(index=winner["index"], cv=winner["score"], holdout=score)
    assert arms["full"]["index"] == search.best_index_
    return arms


def time_shortlists(search, train, labels, test, target, metric, seed, features):
    runs = {}
    for k in ((5, 8) if seed % 2 else (8, 5)):
        fitted = clone(search).set_params(final_selection_top_k=k)
        counts = FitCounters()
        with count_cross_validate_calls(counts):
            start = perf_counter()
            fitted.fit(train, labels)
            elapsed = perf_counter() - start
        finalist_folds = counts.cv_fold_fits - fitted.fit_stats_["cross_validate_calls"]
        assert finalist_folds == 4 * k
        holdout = fitted.transform(test) if features else test
        runs[str(k)] = dict(
            finalists=[row["index"] for row in fitted.final_selection_results_["candidates"]],
            seconds=elapsed,
            folds=counts.cv_fold_fits,
            finalist_folds=finalist_folds,
            index=fitted.best_index_,
            cv=fitted.best_score_,
            holdout=get_scorer(metric)(fitted.best_estimator_, holdout, target),
        )
    return runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--promote", action="store_true")
    modes.add_argument("--time-shortlists", action="store_true")
    args = parser.parse_args()
    records = []
    with threadpool_limits(1):
        for name in ("classification_lr", "regression_ridge"):
            task = SCENARIOS[name]
            X, y = task.loader()
            for seed in (
                range(8711, 8717) if args.promote or args.time_shortlists else range(8601, 8604)
            ):
                train, test, labels, target = train_test_split(
                    X,
                    y,
                    test_size=0.25,
                    random_state=seed,
                    stratify=y if task.task == "classification" else None,
                )
                folds = list(make_cv(task, 5, seed).split(train, labels))
                options = dict(
                    cv=folds[:1],
                    final_selection="reuse_folds",
                    final_selection_cv=folds,
                    final_selection_top_k=8,
                    scoring=task.scoring,
                    error_score="raise",
                    population_size=6,
                    generations=2,
                    n_jobs=1,
                    random_state=seed,
                    verbose=False,
                )
                for features in (False, True):
                    model = task.estimator_builder(seed)
                    search = (
                        GAFeatureSelectionCV(model, max_features=train.shape[1] // 2, **options)
                        if features
                        else GASearchCV(model, param_grid=task.param_grid_builder(), **options)
                    )
                    if args.time_shortlists:
                        runs = time_shortlists(
                            search, train, labels, test, target, task.scoring, seed, features
                        )
                        records.append(dict(task=name, features=features, seed=seed, runs=runs))
                        continue
                    search.fit(train, labels)
                    result = (
                        promote(search, features, train, labels, test, target, task.scoring)
                        if args.promote
                        else replay(search, features)
                    )
                    records.append(dict(task=name, features=features, seed=seed, **result))
    print(json.dumps(records, indent=2))
