"""Compare complete and bounded five-fold accuracy on fixed HPO shortlists."""

import argparse
import json
from pathlib import Path
from runpy import run_path
from time import perf_counter

import numpy as np
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from sklearn_genetic import GASearchCV

tree_grid = run_path(str(Path(__file__).with_name("hpo-repair-pass.py")))["tree_grid"]


def score_shortlist(search, indices, X, y, splits, batch_size):
    start = perf_counter()
    best = -np.inf
    winner = None
    fits = 0
    for index in indices:
        model = clone(search.estimator).set_params(**search.cv_results_["params"][index])
        scores = []
        for offset in range(0, len(splits), batch_size):
            scores.extend(
                cross_val_score(
                    model,
                    X,
                    y,
                    scoring="accuracy",
                    cv=splits[offset : offset + batch_size],
                    n_jobs=1,
                    error_score="raise",
                )
            )
            remaining = len(splits) - len(scores)
            if remaining and (sum(scores) + remaining) / len(splits) < best - 1e-12:
                break
        fits += len(scores)
        if len(scores) == len(splits) and np.mean(scores) > best:
            best, winner = float(np.mean(scores)), int(index)
    return dict(winner=winner, accuracy=best, fits=fits, seconds=perf_counter() - start)


def compare(load, seed, forest=False):
    X, y = load(return_X_y=True)
    search = GASearchCV(
        (
            RandomForestClassifier(n_estimators=12, n_jobs=1, random_state=seed)
            if forest
            else DecisionTreeClassifier(random_state=seed)
        ),
        param_grid=tree_grid(),
        population_size=8,
        generations=2,
        final_selection_top_k=8,
        cv=2,
        scoring="accuracy",
        n_jobs=1,
        random_state=seed,
        verbose=False,
    ).fit(X, y)
    indices = search._top_candidate_indices()
    splits = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y))
    arms = {}
    policies = [("full", 5), ("bounded", 1)]
    if forest:
        policies.append(("block", 3))
        offset = seed % 3
        policies = policies[offset:] + policies[:offset]
    elif seed % 2:
        policies.reverse()
    for name, size in policies:
        arms[name] = score_shortlist(search, indices, X, y, splits, size)
    assert len(indices) == 8 and arms["full"]["fits"] == 40
    assert all(
        arms["full"][key] == arm[key] for arm in arms.values() for key in ("accuracy", "winner")
    )
    return dict(dataset=load.__name__, seed=seed, **arms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forest", action="store_true")
    forest = parser.parse_args().forest
    with threadpool_limits(1):
        rows = [
            compare(load, seed, forest)
            for load in (load_iris, load_wine, load_breast_cancer)
            for seed in (range(10251, 10254) if forest else range(10131, 10135))
        ]
    print(json.dumps(rows, indent=2, allow_nan=False))
