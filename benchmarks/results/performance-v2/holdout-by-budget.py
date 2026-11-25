"""Run from the repository root with the package installed editable."""

import json
import platform
import random
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from functools import lru_cache

import numpy as np
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score, train_test_split
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path.cwd()))
from benchmarks.benchmark_fit import SCENARIOS, make_cv
from sklearn_genetic import GASearchCV, GAFeatureSelectionCV


def paired_run(task, seed, features):
    sc = SCENARIOS[task]
    X, y = sc.loader()
    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=y if sc.task == "classification" else None,
    )
    folds = list(make_cv(sc, 3, seed).split(Xtr, ytr))
    model = sc.estimator_builder(seed)
    space = sc.param_grid_builder()
    columns = X.shape[1]
    limit = columns // 2
    options = dict(
        cv=folds,
        scoring=sc.scoring,
        population_size=6,
        generations=2,
        n_jobs=1,
        random_state=seed,
        error_score="raise",
        verbose=False,
    )
    if features:
        search = GAFeatureSelectionCV(model, max_features=limit, **options)
    else:
        search = GASearchCV(model, param_grid=space, **options)
    search.fit(Xtr, ytr)
    results = search.cv_results_
    candidates = results["features" if features else "params"]

    def candidate_key(candidate):
        return tuple(map(bool, candidate)) if features else tuple(sorted(candidate.items()))

    archive = {}
    for candidate, score in zip(candidates, results["mean_test_score"]):
        key = candidate_key(candidate)
        archive.setdefault(key, float(score))
    assert len(archive) == search.fit_stats_["cross_validate_calls"]
    initial = dict(list(archive.items())[:6])
    assert len({candidate_key(candidate) for candidate in candidates[:6]}) == len(initial) == 6
    assert max(initial.values()) == search.history["fitness_best"][0]

    def prepare(key, data):
        if features:
            return clone(model), data[:, np.asarray(key)]
        return clone(model).set_params(**dict(key)), data

    control = initial.copy()
    random.seed(seed + 10000)
    np.random.seed(seed + 10000)
    while len(control) < len(archive):
        if features:
            selected = set(random.sample(range(columns), random.randint(1, limit)))
            key = tuple(i in selected for i in range(columns))
        else:
            key = tuple(sorted((name, dim.sample()) for name, dim in space.items()))
        if key in control:
            continue
        est, data = prepare(key, Xtr)
        control[key] = float(
            cross_val_score(
                est, data, ytr, cv=folds, scoring=sc.scoring, n_jobs=1, error_score="raise"
            ).mean()
        )

    @lru_cache(maxsize=None)
    def holdout_score(candidate):
        estimator, train = prepare(candidate, Xtr)
        _, test = prepare(candidate, Xte)
        estimator.fit(train, ytr)
        return float(get_scorer(sc.scoring)(estimator, test, yte))

    def checkpoints(pool):
        items = list(pool.items())
        trajectory = []
        for count in (6, 12, 18, 24, len(items)):
            candidate, score = max(items[:count], key=lambda item: item[1])
            trajectory.append(
                dict(unique=count, candidate=candidate, cv=score, holdout=holdout_score(candidate))
            )
        return trajectory

    assert max(archive.values()) == max(search.history["fitness_best"])
    return dict(
        task=task,
        seed=seed,
        search="features" if features else "hpo",
        scoring=sc.scoring,
        folds=len(folds),
        genetic=checkpoints(archive),
        random=checkpoints(control),
    )


with threadpool_limits(1):
    rows = [
        paired_run(task, seed, features)
        for task in ("classification_lr", "regression_ridge")
        for seed in (1201, 1202, 1203)
        for features in (False, True)
    ]
result = dict(
    source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    python=platform.python_version(),
    versions={p: version(p) for p in ["scikit-learn", "numpy", "scipy", "deap"]},
    rows=rows,
)
path = Path(sys.argv[1])
path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
