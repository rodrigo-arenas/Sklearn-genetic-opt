"""Paired feature-mutation pilot. Run from the repository root."""

import json
import platform
import random
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from benchmarks.benchmark_fit import SCENARIOS, make_cv
from sklearn_genetic import GAFeatureSelectionCV


class ExchangeMutation(GAFeatureSelectionCV):
    """Keep the parent mask size while exchanging one active and inactive column."""

    def mutate(self, individual):
        selected = [i for i, bit in enumerate(individual) if bit]
        available = [i for i, bit in enumerate(individual) if not bit]
        assert selected and available  # All pilot masks select 1 to half the columns.
        individual[random.choice(selected)] = 0
        individual[random.choice(available)] = 1
        return (individual,)


def paired_run(name, seed):
    scenario = SCENARIOS[name]
    X, y = scenario.loader()
    train, holdout, labels, targets = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=y if scenario.task == "classification" else None,
    )
    folds = list(make_cv(scenario, 3, seed).split(train, labels))
    model = scenario.estimator_builder(seed)
    archives = {}
    for strategy in (GAFeatureSelectionCV, ExchangeMutation):
        search = strategy(
            model,
            cv=folds,
            scoring=scenario.scoring,
            population_size=6,
            generations=3,
            max_features=X.shape[1] // 2,
            n_jobs=1,
            random_state=seed,
            verbose=False,
            error_score="raise",
        )
        search.fit(train, labels)
        scores = {}
        for mask, score in zip(
            search.cv_results_["features"], search.cv_results_["mean_test_score"], strict=True
        ):
            scores.setdefault(tuple(map(bool, mask)), float(score))
        assert len(scores) == search.fit_stats_["cross_validate_calls"]
        archives[strategy.__name__] = list(scores.items())
    baseline, exchange = archives.values()
    assert baseline[:6] == exchange[:6] and min(len(baseline), len(exchange)) >= 24
    result = {}
    for method, entries in archives.items():
        checkpoints = []
        for budget in (6, 12, 24):
            mask, score = max(entries[:budget], key=lambda item: item[1])
            fitted = clone(model).fit(train[:, mask], labels)
            external = get_scorer(scenario.scoring)(fitted, holdout[:, mask], targets)
            checkpoints.append(
                dict(
                    unique=budget,
                    columns=np.flatnonzero(mask).tolist(),
                    cv=score,
                    holdout=float(external),
                )
            )
        result[method] = dict(total_unique=len(entries), checkpoints=checkpoints)
    return dict(task=name, seed=seed, scoring=scenario.scoring, results=result)


if __name__ == "__main__":
    with threadpool_limits(1):
        rows = [
            paired_run(task, seed)
            for task in ("classification_lr", "regression_ridge", "regression_friedman_ridge")
            for seed in (1501, 1502, 1503)
        ]
    data = dict(
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        python=platform.python_version(),
        versions={p: version(p) for p in ("scikit-learn", "numpy", "deap", "scipy")},
        rows=rows,
    )
    Path(sys.argv[1]).write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
