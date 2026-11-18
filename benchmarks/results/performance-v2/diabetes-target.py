"""Time complete feature selection policies up to a prespecified CV R² target."""

import argparse
import json
import sys
from contextlib import redirect_stdout
from itertools import permutations
from pathlib import Path
from runpy import run_path
from time import perf_counter

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from sklearn_genetic import GAFeatureSelectionCV
from sklearn_genetic.callbacks import ThresholdStopping

residual_selector = run_path(str(Path(__file__).with_name("filter-portfolio.py")))[
    "residual_selector"
]


def run(policy, X, y, test, labels, cv, seed, target):
    start = perf_counter()
    model = make_pipeline(StandardScaler(), Ridge())
    winner = None
    score = -np.inf
    calls = 0
    if policy != "genetic":
        selector = residual_selector(4) if policy == "residual" else SelectKBest(f_regression, k=4)
        proposal = make_pipeline(selector, model)
        score = cross_val_score(proposal, X, y, cv=cv, scoring="r2", error_score="raise").mean()
        calls += 1
        winner = proposal
    fallback = score < target
    if fallback:
        ga = GAFeatureSelectionCV(
            model,
            cv=cv,
            scoring="r2",
            max_features=4,
            population_size=6,
            generations=3,
            random_state=seed,
            error_score="raise",
            n_jobs=1,
            verbose=False,
        )
        with redirect_stdout(sys.stderr):
            ga.fit(X, y, callbacks=ThresholdStopping(target, metric="fitness_max"))
        mask = np.all(np.asarray(ga.cv_results_["features"]) == ga.best_features_, axis=1)
        ga_score = ga.cv_results_["mean_test_score"][np.flatnonzero(mask)[0]]
        calls += ga.fit_stats_["cross_validate_calls"]
        if ga_score >= score:
            score, winner = ga_score, ga
    if not isinstance(winner, GAFeatureSelectionCV):
        winner.fit(X, y)
    seconds = perf_counter() - start
    return dict(
        policy=policy,
        seed=seed,
        ran_genetic=bool(fallback),
        folds=calls * len(cv),
        cv=float(score),
        reached=bool(score >= target),
        seconds=seconds,
        holdout=winner.score(test, labels),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensitivity", action="store_true")
    sensitivity = parser.parse_args().sensitivity
    X, y = load_diabetes(return_X_y=True)
    rows = []
    with threadpool_limits(1):
        for index, order in enumerate(permutations(("genetic", "univariate", "residual"))):
            seed = (9901 if sensitivity else 9791) + index
            train, test, targets, labels = train_test_split(X, y, test_size=0.25, random_state=seed)
            cv = list(KFold(3, shuffle=True, random_state=seed).split(train))
            thresholds = (0.40, 0.45, 0.50) if sensitivity else (0.45,)
            offset = index % len(thresholds)
            for target in thresholds[offset:] + thresholds[:offset]:
                for policy in order:
                    rows.append(
                        dict(
                            target=target,
                            **run(policy, train, targets, test, labels, cv, seed, target),
                        )
                    )
    print(json.dumps(rows, indent=2, allow_nan=False))
