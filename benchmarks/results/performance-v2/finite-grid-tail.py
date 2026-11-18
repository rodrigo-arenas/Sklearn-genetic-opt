"""Measure the proposals needed to finish a nearly covered finite Wine grid."""

import json
from time import perf_counter

from sklearn.datasets import load_wine
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Categorical, Integer


def run(seed, generations, X, y, test, labels, folds):
    search = GASearchCV(
        DecisionTreeClassifier(random_state=0),
        param_grid={"max_depth": Integer(2, 15), "criterion": Categorical(["gini", "entropy"])},
        population_size=8,
        generations=generations,
        random_state=seed,
        cv=folds,
        scoring="accuracy",
        error_score="raise",
        n_jobs=1,
        verbose=False,
    )
    start = perf_counter()
    search.fit(X, y)
    elapsed = perf_counter() - start
    proposals = search.cv_results_["params"]
    first = {}
    for i, params in enumerate(proposals, 1):
        first.setdefault((int(params["max_depth"]), params["criterion"]), i)
    assert len(first) == search.fit_stats_["cross_validate_calls"]
    assert search.search_space_exhausted_ == (len(first) == 28)
    return dict(
        seed=seed,
        limit=generations,
        seconds=elapsed,
        generations=search.history["gen"][-1],
        proposals=len(proposals),
        discovery_positions=list(first.values()),
        best_params=search.best_params_,
        cv=search.best_score_,
        holdout=search.score(test, labels),
    )


if __name__ == "__main__":
    X, y = load_wine(return_X_y=True)
    rows = []
    with threadpool_limits(1):
        for seed in range(1091, 1101):
            train, test, target, labels = train_test_split(
                X, y, test_size=0.25, stratify=y, random_state=seed
            )
            folds = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(train, target))
            for cap in ((4, 40) if seed % 2 else (40, 4)):
                rows.append(run(seed, cap, train, target, test, labels, folds))
    print(json.dumps(rows, indent=2))
