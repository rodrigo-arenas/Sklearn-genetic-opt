"""Compare feature proposal policies at matched budgets on synthetic and real datasets."""

import argparse
import json
import platform
import sys
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.datasets import load_diabetes, load_wine
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sklearn_genetic import GAFeatureSelectionCV


def forward_history(width, budget, seed, cv_score):
    history = {}
    parent = ()
    order_rng = np.random.default_rng(seed)
    while len(history) < budget and len(parent) < min(4, width):
        extensions = {}
        for column in order_rng.permutation(width).tolist():
            if column in parent:
                continue
            mask = tuple(sorted((*parent, column)))
            extensions[mask] = history[mask] = cv_score(mask)
            if len(history) == budget:
                break
        parent = max(extensions, key=extensions.get)
    return history


def run_pair(seed, interaction, pair_coverage=False, score_guided=False, forward=False):
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
    result = compare(
        X,
        y,
        test,
        labels,
        cv,
        model,
        "balanced_accuracy",
        seed,
        pair_coverage,
        score_guided,
        signal,
        forward=forward,
    )
    return dict(task="xor" if interaction else "additive", signal=signal, **result)


def run_real(task, split_seed, seed, forward=False):
    classification = task == "wine"
    X, y = (load_wine if classification else load_diabetes)(return_X_y=True)
    X, test, y, labels = train_test_split(
        X, y, test_size=0.25, random_state=split_seed, stratify=y if classification else None
    )
    splitter = StratifiedKFold if classification else KFold
    cv = list(splitter(3, shuffle=True, random_state=split_seed).split(X, y))
    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=300) if classification else Ridge()
    )
    metric = "balanced_accuracy" if classification else "r2"
    return dict(
        task=task,
        split_seed=split_seed,
        metric=metric,
        **compare(X, y, test, labels, cv, model, metric, seed, False, not forward, forward=forward),
    )


def compare(
    X,
    y,
    test,
    labels,
    cv,
    model,
    metric,
    seed,
    pair_coverage,
    score_guided,
    signal=None,
    forward=False,
):
    search = GAFeatureSelectionCV(
        model,
        cv=cv,
        scoring=metric,
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

    def cv_score(columns):
        return float(
            cross_val_score(
                model,
                X[:, columns],
                y,
                cv=cv,
                scoring=metric,
                n_jobs=1,
                error_score="raise",
            ).mean()
        )

    while len(control) < len(genetic):
        columns = tuple(
            sorted(draw.choice(X.shape[1], size=draw.integers(1, 5), replace=False).tolist())
        )
        if columns not in control:
            control[columns] = cv_score(columns)

    def evaluate(columns):
        fitted = clone(model).fit(X[:, columns], y)
        result = dict(
            columns=list(columns),
            holdout=float(get_scorer(metric)(fitted, test[:, columns], labels)),
        )
        if signal is not None:
            result["recovered"] = len(set(columns).intersection(signal))
        return result

    def summary(history):
        winner = max(history, key=lambda mask: (history[mask], -len(mask)))
        return dict(
            **evaluate(winner),
            cv=history[winner],
            proposals=[dict(columns=list(mask), cv=score) for mask, score in history.items()],
        )

    result = dict(seed=seed, genetic=summary(genetic), random=summary(control))
    if forward:
        history = forward_history(X.shape[1], len(genetic), seed, cv_score)
        assert len(history) == len(genetic)
        result["forward"] = summary(history)
    if signal is not None:
        univariate = SelectKBest(f_classif, k=2).fit(X, y).get_support(indices=True).tolist()
        result.update(filter=evaluate(univariate), known_support=evaluate(signal))

    if pair_coverage or score_guided:
        history = dict(list(genetic.items())[:6])
        candidates = {size: list(combinations(range(X.shape[1]), size)) for size in range(1, 5)}
        pool_rng = np.random.default_rng(seed + 20000)
        for random_mask in list(control)[6:]:
            available = [mask for mask in candidates[len(random_mask)] if mask not in history]
            pool = pool_rng.choice(len(available), size=min(16, len(available)), replace=False)
            proposals = [available[idx] for idx in pool]
            if score_guided:
                # Two observations at the running mean shrink sparse column estimates.
                center = np.mean(list(history.values()))
                evidence = np.zeros(X.shape[1])
                observations = np.zeros(X.shape[1])
                for mask, score in history.items():
                    evidence[list(mask)] += score - center
                    observations[list(mask)] += 1
                weights = evidence / (2 + observations)
                mask = max(proposals, key=lambda candidate: weights[list(candidate)].sum())
            else:
                covered = {pair for mask in history for pair in combinations(mask, 2)}
                mask = max(
                    proposals,
                    key=lambda candidate: sum(
                        pair not in covered for pair in combinations(candidate, 2)
                    ),
                )
            history[mask] = cv_score(mask)
        result["score_guided" if score_guided else "pair_coverage"] = summary(history)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    policies = parser.add_mutually_exclusive_group()
    policies.add_argument(
        "--pair-coverage", action="store_true", help="Test pair coverage on six new dataset seeds."
    )
    policies.add_argument(
        "--score-guided", action="store_true", help="Rank proposals using earlier mask CV scores."
    )
    policies.add_argument(
        "--real-transfer",
        action="store_true",
        help="Use Wine and Diabetes. Compare score-guided proposals unless --forward is set.",
    )
    parser.add_argument("--forward", action="store_true", help="Compare greedy forward selection.")
    args = parser.parse_args()
    seeds = range(2201, 2207) if args.pair_coverage else range(2101, 2105)
    if args.score_guided:
        seeds = range(3201, 3207)
    with threadpool_limits(1):
        if args.real_transfer:
            rows = [
                run_real(task, split_seed, seed, forward=args.forward)
                for task in ("wine", "diabetes")
                for split_seed in (3301, 3302)
                for seed in (3311, 3312, 3313)
            ]
        else:
            rows = [
                run_pair(seed, interaction, args.pair_coverage, args.score_guided, args.forward)
                for seed in seeds
                for interaction in (False, True)
            ]
    payload = dict(
        python=platform.python_version(),
        versions={p: version(p) for p in ("scikit-learn", "numpy", "deap")},
        rows=rows,
    )
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
