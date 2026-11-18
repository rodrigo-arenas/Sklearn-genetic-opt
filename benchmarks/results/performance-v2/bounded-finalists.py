"""Replay finalist CV with a score ceiling of one and one reusable fold."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from benchmarks.benchmark_fit import SCENARIOS, make_cv
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


if __name__ == "__main__":
    records = []
    with threadpool_limits(1):
        for name in ("classification_lr", "regression_ridge"):
            task = SCENARIOS[name]
            X, y = task.loader()
            for seed in range(8601, 8604):
                train, _, labels, _ = train_test_split(
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
                    search.fit(train, labels)
                    records.append(
                        dict(task=name, features=features, seed=seed, **replay(search, features))
                    )
    print(json.dumps(records, indent=2))
