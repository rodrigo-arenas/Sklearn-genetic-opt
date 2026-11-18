"""Trade four fixed-mask CV evaluations for fold-local univariate filters."""

import argparse
import json
from functools import partial
from pathlib import Path
from runpy import run_path
from time import perf_counter

import numpy as np
from sklearn.base import clone, is_classifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, f_regression
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from sklearn_genetic import GAFeatureSelectionCV


def compare(
    X,
    y,
    test,
    labels,
    cv,
    model,
    metric,
    seed,
    *args,
    one_slot=False,
    cap=4,
    residual=False,
    **kwargs,
):
    ga = GAFeatureSelectionCV(
        clone(model),
        cv=cv,
        scoring=metric,
        max_features=cap,
        population_size=6,
        generations=3,
        random_state=seed,
        error_score="raise",
        n_jobs=1,
        verbose=False,
    )
    start = perf_counter()
    ga.fit(X, y)
    full_ga_seconds = perf_counter() - start
    scores = {}
    for mask, score in zip(ga.cv_results_["features"], ga.cv_results_["mean_test_score"]):
        scores.setdefault(tuple(np.flatnonzero(mask).tolist()), float(score))
    assert len(scores) == ga.fit_stats_["cross_validate_calls"] and len(scores) >= 24
    genetic = [(v, len(m), m) for m, v in list(scores.items())[:24]]
    filters = []
    scorer = f_classif if is_classifier(model) else f_regression
    start = perf_counter()
    for k in np.linspace(1, cap, 4).round().astype(int).tolist():
        candidate = make_pipeline(SelectKBest(scorer, k=k), clone(model))
        score = cross_val_score(
            candidate, X, y, cv=cv, scoring=metric, n_jobs=1, error_score="raise"
        ).mean()
        filters.append((float(score), k, None))
    filter_cv_seconds = perf_counter() - start
    arms = {}
    policies = [
        ("genetic", genetic),
        ("filters", filters),
        ("portfolio", genetic[:20] + filters),
    ]
    if one_slot:
        policies.append(("one_filter", genetic[:23] + filters[-1:]))
    if residual:
        selector = SelectFromModel(
            make_pipeline(StandardScaler(), OrthogonalMatchingPursuit(n_nonzero_coefs=cap)),
            importance_getter="named_steps.orthogonalmatchingpursuit.coef_",
            threshold=1e-12,
        )
        residual_model = make_pipeline(selector, clone(model))
        score = cross_val_score(
            residual_model, X, y, cv=cv, scoring=metric, n_jobs=1, error_score="raise"
        ).mean()
        policies.append(("one_residual", genetic[:23] + [(float(score), cap, None)]))
    for name, candidates in policies:
        score, k, mask = max(candidates, key=lambda c: (c[0], -c[1]))
        if mask is None:
            candidate = (
                residual_model
                if name == "one_residual"
                else make_pipeline(SelectKBest(scorer, k=k), clone(model))
            )
            fitted = candidate.fit(X, y)
            selected = np.flatnonzero(fitted[0].get_support()).tolist()
            held = get_scorer(metric)(fitted, test, labels)
        else:
            selected = list(mask)
            held = get_scorer(metric)(clone(model).fit(X[:, mask], y), test[:, mask], labels)
        arms[name] = dict(
            folds=len(cv) * len(candidates),
            filter=mask is None,
            selected=selected,
            cv=score,
            holdout=held,
        )
    return dict(
        seed=seed,
        arms=arms,
        full_ga_candidates=len(scores),
        full_ga_seconds=full_ga_seconds,
        filter_cv_seconds=filter_cv_seconds,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one-slot", action="store_true")
    args = parser.parse_args()
    seeds = range(9351, 9357) if args.one_slot else range(9241, 9245)
    here = Path(__file__).resolve().parent
    known = run_path(str(here / "known-support.py"))
    sparse = run_path(str(here / "partition-initialization.py"))["sparse_regression"]
    pair, real = known["run_pair"], known["run_real"]
    for builder in (sparse, pair, real):
        builder.__globals__["compare"] = partial(compare, one_slot=args.one_slot)
    with threadpool_limits(1):
        rows = [dict(task="sparse", **sparse(64, s)) for s in seeds]
        rows += [pair(s, True) for s in seeds]
        rows += [real(task, s, s + 10) for task in ("wine", "diabetes") for s in seeds]
        if args.one_slot:
            rows += [pair(s, False) for s in seeds]
    print(json.dumps(rows, indent=2, allow_nan=False))
