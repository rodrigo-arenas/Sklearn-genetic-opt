"""Compare feature searches on independently generated datasets with known support."""

import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sklearn_genetic import GAFeatureSelectionCV


def run_pair(seed, interaction):
    rng = np.random.default_rng(seed)
    signal = sorted(rng.choice(16, size=2, replace=False).tolist())

    def sample(n):
        X = rng.standard_normal((n, 16))
        a, b = X[:, signal].T
        positive = (a > 0) ^ (b > 0) if interaction else a + b > 0
        return X, (positive ^ (rng.random(n) < 0.05)).astype(int)

    X, y = sample(300)
    test, labels = sample(1000)
    cv = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(X, y))
    model = make_pipeline(StandardScaler(), SVC(C=2))
    search = GAFeatureSelectionCV(
        model,
        cv=cv,
        scoring="balanced_accuracy",
        max_features=4,
        population_size=6,
        generations=3,
        random_state=seed,
        n_jobs=1,
        verbose=False,
        error_score="raise",
    ).fit(X, y)
    log = search.cv_results_
    genetic = {}
    for mask, score in zip(log["features"], log["mean_test_score"], strict=True):
        genetic.setdefault(tuple(np.flatnonzero(mask).tolist()), float(score))
    assert len(genetic) == search.fit_stats_["cross_validate_calls"] and len(genetic) >= 24
    assert len(set(tuple(np.flatnonzero(mask)) for mask in log["features"][:6])) == 6
    genetic = dict(list(genetic.items())[:24])
    control = dict(list(genetic.items())[:6])
    draw = np.random.default_rng(seed + 10000)
    while len(control) < len(genetic):
        columns = tuple(sorted(draw.choice(16, size=draw.integers(1, 5), replace=False).tolist()))
        if columns not in control:
            control[columns] = float(
                cross_val_score(
                    model,
                    X[:, columns],
                    y,
                    cv=cv,
                    scoring="balanced_accuracy",
                    n_jobs=1,
                    error_score="raise",
                ).mean()
            )

    def evaluate(columns):
        fitted = clone(model).fit(X[:, columns], y)
        return dict(
            columns=list(columns),
            recovered=len(set(columns).intersection(signal)),
            holdout=float(balanced_accuracy_score(labels, fitted.predict(test[:, columns]))),
        )

    def summary(history):
        winner = max(history, key=lambda mask: (history[mask], -len(mask)))
        return dict(
            **evaluate(winner),
            cv=history[winner],
            proposals=[dict(columns=list(mask), cv=score) for mask, score in history.items()],
        )

    univariate = SelectKBest(f_classif, k=2).fit(X, y).get_support(indices=True).tolist()
    return dict(
        seed=seed,
        task="xor" if interaction else "additive",
        signal=signal,
        genetic=summary(genetic),
        random=summary(control),
        filter=evaluate(univariate),
        known_support=evaluate(signal),
    )


if __name__ == "__main__":
    with threadpool_limits(1):
        rows = [
            run_pair(seed, interaction)
            for seed in range(2101, 2105)
            for interaction in (False, True)
        ]
    payload = dict(
        python=platform.python_version(),
        versions={p: version(p) for p in ("scikit-learn", "numpy", "deap")},
        rows=rows,
    )
    Path(sys.argv[1]).write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
