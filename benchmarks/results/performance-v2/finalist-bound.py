"""Compare complete and bounded five-fold accuracy on fixed HPO shortlists."""

import json
from pathlib import Path
from runpy import run_path
from time import perf_counter

import numpy as np
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from sklearn_genetic import GASearchCV

tree_grid = run_path(str(Path(__file__).with_name("hpo-repair-pass.py")))["tree_grid"]


def score_shortlist(search, indices, X, y, splits, bounded):
    start = perf_counter()
    best = -np.inf
    winner = None
    fits = 0
    for index in indices:
        model = clone(search.estimator).set_params(**search.cv_results_["params"][index])
        if bounded:
            scores = []
            for fold in splits:
                scores.extend(
                    cross_val_score(
                        model, X, y, scoring="accuracy", cv=[fold], n_jobs=1, error_score="raise"
                    )
                )
                remaining = len(splits) - len(scores)
                if remaining and (sum(scores) + remaining) / len(splits) < best - 1e-12:
                    break
        else:
            scores = cross_val_score(
                model, X, y, scoring="accuracy", cv=splits, n_jobs=1, error_score="raise"
            )
        fits += len(scores)
        if len(scores) == len(splits) and np.mean(scores) > best:
            best, winner = float(np.mean(scores)), int(index)
    return dict(winner=winner, accuracy=best, fits=fits, seconds=perf_counter() - start)


def compare(load, seed):
    X, y = load(return_X_y=True)
    search = GASearchCV(
        DecisionTreeClassifier(random_state=seed),
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
    for bounded in ((True, False) if seed % 2 else (False, True)):
        arms["bounded" if bounded else "full"] = score_shortlist(
            search, indices, X, y, splits, bounded
        )
    assert len(indices) == 8 and arms["full"]["fits"] == 40
    assert all(arms["full"][key] == arms["bounded"][key] for key in ("accuracy", "winner"))
    return dict(dataset=load.__name__, seed=seed, **arms)


if __name__ == "__main__":
    with threadpool_limits(1):
        rows = [
            compare(load, seed)
            for load in (load_iris, load_wine, load_breast_cancer)
            for seed in (10131, 10132, 10133, 10134)
        ]
    print(json.dumps(rows, indent=2, allow_nan=False))
