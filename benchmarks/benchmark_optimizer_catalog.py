"""Run from the checkout: python -m benchmarks.benchmark_optimizer_catalog --help."""

import argparse
import csv
import json
import platform
from contextlib import ExitStack
from functools import partial
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
from sklearn.base import clone
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_wine,
    make_classification,
    make_regression,
    make_friedman1,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import get_scorer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from sklearn_genetic import GAFeatureSelectionCV, GASearchCV
from sklearn_genetic.space import Categorical, Continuous, Integer


def measure(case, backend, variant, seed, cache, pilot, target=32):
    from sklearn_genetic._cmaes_backend import optimize as cmaes
    from sklearn_genetic._nevergrad_backend import optimize as nevergrad
    from sklearn_genetic._pymoo_backend import optimize as pymoo

    classification = not case.startswith(("diabetes", "linear", "friedman"))
    masks = case.startswith(("sparse", "linear")) or case in ("cancer", "wine-mask")
    if case.startswith(("sparse", "calibration-cls")):
        difficulty = {
            "sparse-easy": (1.6, 0.01),
            "sparse-hard": (0.65, 0.12),
            "calibration-cls": (1.0, 0.06),
        }
        separation, label_noise = difficulty[case]
        X, y = make_classification(
            n_samples=300,
            n_features=40,
            n_informative=6,
            n_redundant=4,
            class_sep=separation,
            flip_y=label_noise,
            random_state=20_000 + seed,
        )
    elif case.startswith("linear"):
        rank, noise = {
            "linear-easy": (None, 5),
            "linear-hard": (8, 8),
            "linear-calibration": (16, 5),
        }[case]
        X, y = make_regression(
            n_samples=280,
            n_features=36,
            n_informative=8,
            effective_rank=rank,
            noise=noise,
            random_state=30_000 + seed,
        )
    elif case == "friedman":
        X, y = make_friedman1(n_samples=300, n_features=24, noise=1.5, random_state=40_000 + seed)
    elif case.startswith("diabetes"):
        X, y = load_diabetes(return_X_y=True, scaled=False)
    elif case in ("wine", "wine-mask"):
        X, y = load_wine(return_X_y=True)
    else:
        X, y = load_breast_cancer(return_X_y=True)
    X, hold_X, y, hold_y = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y if classification else None
    )
    scoring = "balanced_accuracy" if classification else "neg_root_mean_squared_error"
    model = (DecisionTreeClassifier if classification else DecisionTreeRegressor)(random_state=seed)
    if case.startswith("sparse"):
        model = LogisticRegression(solver="liblinear", random_state=seed)
    elif case.startswith("linear"):
        model = Ridge(alpha=0.1)
    elif case == "diabetes-svr":
        model = make_pipeline(StandardScaler(), SVR())
    engine = backend
    prob = 0.8
    if pilot:
        if backend == "cmaes":
            engine = partial(cmaes, sigma=float(variant))
        elif backend == "nevergrad":
            engine = partial(nevergrad, strategy=variant)
        elif backend == "pymoo":
            prob, engine = float(variant), pymoo
    size = 8
    generations = target // size - 1
    options = dict(
        estimator=model,
        cv=(StratifiedKFold if classification else KFold)(3, shuffle=True, random_state=seed),
        random_state=seed,
        scoring=scoring,
        optimizer_backend=engine,
        n_jobs=1,
        population_size=size,
        generations=generations,
        crossover_probability=prob,
        use_cache=cache,
        refit=False,
        verbose=False,
    )
    if masks:
        search = GAFeatureSelectionCV(max_features=max(2, X.shape[1] // 3), **options)
    elif case == "diabetes-svr":
        grid = {
            name: Continuous(lo, hi, distribution="log-uniform")
            for name, lo, hi in (
                ("svr__C", 0.1, 1000),
                ("svr__gamma", 0.0001, 1),
                ("svr__epsilon", 0.01, 10),
            )
        }
        search = GASearchCV(param_grid=grid, **options)
    else:
        grid = dict(
            max_depth=Integer(2, 12),
            min_samples_leaf=Integer(1, 12),
            ccp_alpha=Continuous(1e-6, 0.1, distribution="log-uniform"),
            criterion=Categorical(
                ["gini", "entropy"] if classification else ["squared_error", "absolute_error"]
            ),
        )
        search = GASearchCV(param_grid=grid, **options)
    trace = []
    cv_seconds = 0
    cv_calls = 0
    original = type(search)._evaluate_individual

    def clock_cv(*args, **kwargs):
        nonlocal cv_seconds, cv_calls
        cv_calls += 1
        before = perf_counter()
        result = cross_validate(*args, **kwargs)
        cv_seconds += perf_counter() - before
        return result

    def tracked(self, candidate, *args, **kwargs):
        result = original(self, candidate, *args, **kwargs)
        if result[2]:
            trace.append(
                (float(result[1]["score"]), tuple(candidate), perf_counter() - start, cv_seconds)
            )
        return result

    with ExitStack() as stack:
        stack.enter_context(patch("sklearn_genetic.genetic_search.cross_validate", clock_cv))
        stack.enter_context(patch.object(type(search), "_evaluate_individual", tracked))
        start = perf_counter()
        search.fit(X, y)
        total = perf_counter() - start
    if not trace or not cv_calls == len(trace) == search.fit_stats_["cross_validate_calls"]:
        raise RuntimeError(f"CV count mismatch: {case}, {backend}, {seed}")
    return (trace, model, X, y, hold_X, hold_y, scoring, masks, search, total)


def summarize(run, budget):
    trace, model, X, y, hold_X, hold_y, scoring, masks, search, total = run
    observed = trace[:budget]
    best, coords, _, _ = max(observed, key=lambda x: (x[0], -sum(x[1]) if masks else 0))
    if masks:
        mask = np.array(coords, dtype=bool)
        X, hold_X = X[:, mask], hold_X[:, mask]
    else:
        model = clone(model).set_params(**dict(zip(search.space.parameters, coords)))
    model.fit(X, y)
    held_out = get_scorer(scoring)(model, hold_X, hold_y)
    if scoring == "balanced_accuracy":
        baseline = 1.0 / len(np.unique(y))
        skill = (held_out - baseline) / (1 - baseline)
    else:
        baseline = -np.sqrt(np.mean((hold_y - np.mean(y)) ** 2))
        skill = 1 - held_out / baseline
    elapsed, cv_time = observed[-1][2:]
    return dict(
        cv_calls=budget,
        full_cv_calls=len(trace),
        full_proposals=search.fit_stats_["evaluated_candidates"],
        full_cache_hits=search.fit_stats_["cache_hits"],
        full_duplicates=search.fit_stats_["duplicate_candidates"],
        first_best_call=1 + next(i for i, row in enumerate(observed) if row[0] == best),
        unique_prefix=len({x[1] for x in observed}),
        best_cv=best,
        best_first_eight=max(x[0] for x in observed[:8]),
        mean_running_best=np.maximum.accumulate([x[0] for x in observed]).mean(),
        holdout=held_out,
        holdout_baseline=baseline,
        holdout_skill=skill,
        selected=X.shape[1] if masks else "",
        prefix_ms=1000 * elapsed,
        prefix_outside_cv_ms=1000 * (elapsed - cv_time),
        full_outside_cv_ms=1000 * (total - trace[-1][3]),
        full_ms=1000 * total,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--budget", type=int, choices=(32, 96), default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    engines = (
        (
            [("nevergrad", v) for v in ("TwoPointsDE", "OnePlusOne")]
            + [("cmaes", v) for v in (".15", ".25", ".4")]
            + [("pymoo", v) for v in (".5", ".8", "1")]
        )
        if args.pilot
        else [(k, "default") for k in ("deap", "scipy", "nevergrad", "cmaes", "pymoo")]
    )
    seeds = (7, 19, 31) if args.pilot else (41, 59, 73, 97, 113)
    cases = (
        ("wine", "wine-mask", "calibration-cls", "linear-calibration")
        if args.pilot
        else (
            "tumor-params",
            "cancer",
            "diabetes",
            "diabetes-svr",
            "sparse-easy",
            "sparse-hard",
            "linear-easy",
            "linear-hard",
            "friedman",
        )
    )
    partial_path = args.output.with_suffix(args.output.suffix + ".partial")
    count = 0
    writer = None
    with threadpool_limits(1), partial_path.open("w", newline="") as out:
        # Pay import and solver setup costs once outside the timed runs.
        for engine, variant in engines:
            measure("wine", engine, variant, 3, False, args.pilot)
        for cache in ((False,) if args.pilot else (False, True)):
            for case in cases:
                for i, seed in enumerate(seeds):
                    order = engines[i % len(engines) :] + engines[: i % len(engines)]
                    runs = [
                        (name, v, measure(case, name, v, seed, cache, args.pilot, args.budget))
                        for name, v in order
                    ]
                    budget = min(args.budget, *(len(run[0]) for _, _, run in runs))
                    for backend, variant, run in runs:
                        row = dict(
                            case=case,
                            target=args.budget,
                            seed=seed,
                            cache=cache,
                            backend=backend,
                            variant=variant,
                            **summarize(run, budget),
                        )
                        if writer is None:
                            writer = csv.DictWriter(out, fieldnames=list(row), lineterminator="\n")
                            writer.writeheader()
                        writer.writerow(row)
                        count += 1
                    out.flush()
    partial_path.replace(args.output)
    print(
        json.dumps(
            dict(
                python=platform.python_version(),
                platform=platform.platform(),
                packages={
                    p: version(p)
                    for p in (
                        "scikit-learn",
                        "numpy",
                        "deap",
                        "scipy",
                        "nevergrad",
                        "cmaes",
                        "pymoo",
                    )
                },
                rows=count,
            )
        )
    )


if __name__ == "__main__":
    main()
