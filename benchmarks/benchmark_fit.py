"""Benchmark GA fit-time mechanics and holdout model quality.

Run from the repository root, for example:

    python benchmarks/benchmark_fit.py --quick
    python benchmarks/benchmark_fit.py --generations 8 --population-size 12 --runs 3
    python benchmarks/benchmark_fit.py --output-json benchmarks/results.json

The benchmark intentionally lives outside the test suite. It measures wall time,
actual cross-validation calls, duplicate/cache reuse, invalid feature masks skipped,
and holdout classification metrics for GASearchCV and GAFeatureSelectionCV.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="sklearn-genetic-bench-mpl-"))

from sklearn_genetic import GAFeatureSelectionCV, GASearchCV
from sklearn_genetic import genetic_search
from sklearn_genetic.space import Categorical, Continuous


@dataclass
class FitCounters:
    cross_validate_calls: int = 0


@contextmanager
def count_cross_validate_calls(counters: FitCounters):
    original_cross_validate = genetic_search.cross_validate

    def counted_cross_validate(*args, **kwargs):
        counters.cross_validate_calls += 1
        return original_cross_validate(*args, **kwargs)

    genetic_search.cross_validate = counted_cross_validate
    try:
        yield
    finally:
        genetic_search.cross_validate = original_cross_validate


def build_classifier(random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear",
                    max_iter=500,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_gasearch(
    *,
    random_state: int,
    cv: StratifiedKFold,
    population_size: int,
    generations: int,
    n_jobs: int | None,
) -> GASearchCV:
    return GASearchCV(
        estimator=build_classifier(random_state),
        cv=cv,
        scoring="roc_auc",
        population_size=population_size,
        generations=generations,
        tournament_size=3,
        elitism=True,
        param_grid={
            "clf__C": Continuous(1e-3, 10.0, distribution="log-uniform"),
            "clf__class_weight": Categorical([None, "balanced"]),
        },
        verbose=False,
        n_jobs=n_jobs,
        return_train_score=False,
        use_cache=True,
    )


def build_feature_selector(
    *,
    random_state: int,
    cv: StratifiedKFold,
    population_size: int,
    generations: int,
    n_jobs: int | None,
    max_features: int,
) -> GAFeatureSelectionCV:
    return GAFeatureSelectionCV(
        estimator=build_classifier(random_state),
        cv=cv,
        scoring="roc_auc",
        population_size=population_size,
        generations=generations,
        tournament_size=3,
        elitism=True,
        max_features=max_features,
        verbose=False,
        n_jobs=n_jobs,
        return_train_score=False,
        use_cache=True,
    )


def holdout_metrics(estimator, X_test, y_test) -> dict[str, float]:
    predictions = estimator.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
    }

    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(X_test)
        if probabilities.shape[1] == 2:
            metrics["roc_auc"] = roc_auc_score(y_test, probabilities[:, 1])

    return metrics


def get_log_records(estimator) -> list[dict[str, Any]]:
    return list(estimator.logbook.chapters["parameters"])


def summarize_fit_mechanics(estimator, counters: FitCounters) -> dict[str, int]:
    records = get_log_records(estimator)
    evaluated_records = len(records)
    invalid_feature_masks_skipped = 0

    for record in records:
        fit_time = np.asarray(record.get("fit_time", []))
        score = record.get("score")
        if fit_time.size and np.all(fit_time == 0) and score in {-100000, 100000}:
            invalid_feature_masks_skipped += 1

    duplicate_or_cache_reuses = (
        evaluated_records - counters.cross_validate_calls - invalid_feature_masks_skipped
    )

    return {
        "evaluated_records": evaluated_records,
        "actual_cross_validate_calls": counters.cross_validate_calls,
        "duplicate_or_cache_reuses": max(0, duplicate_or_cache_reuses),
        "invalid_feature_masks_skipped": invalid_feature_masks_skipped,
        "fitness_cache_size": len(getattr(estimator, "fitness_cache", {})),
    }


def run_one_benchmark(
    *,
    name: str,
    estimator,
    X_train,
    X_test,
    y_train,
    y_test,
    n_jobs: int | None,
    run_index: int,
) -> dict[str, Any]:
    counters = FitCounters()
    started_at = time.perf_counter()

    with count_cross_validate_calls(counters):
        estimator.fit(X_train, y_train)

    fit_seconds = time.perf_counter() - started_at

    result = {
        "scenario": name,
        "run": run_index,
        "n_jobs": n_jobs,
        "fit_seconds": fit_seconds,
        **summarize_fit_mechanics(estimator, counters),
        "best_score": getattr(estimator, "best_score_", None),
        "holdout_metrics": holdout_metrics(estimator, X_test, y_test),
    }

    if hasattr(estimator, "support_"):
        result["selected_features"] = int(np.sum(estimator.support_))

    return result


def aggregate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault((result["scenario"], str(result["n_jobs"])), []).append(result)

    summaries = []
    for (scenario, n_jobs), items in grouped.items():
        metric_names = sorted(
            {metric_name for item in items for metric_name in item["holdout_metrics"].keys()}
        )
        summary = {
            "scenario": scenario,
            "n_jobs": n_jobs,
            "runs": len(items),
            "fit_seconds_mean": float(np.mean([item["fit_seconds"] for item in items])),
            "fit_seconds_std": float(np.std([item["fit_seconds"] for item in items])),
            "actual_cross_validate_calls_mean": float(
                np.mean([item["actual_cross_validate_calls"] for item in items])
            ),
            "duplicate_or_cache_reuses_mean": float(
                np.mean([item["duplicate_or_cache_reuses"] for item in items])
            ),
            "invalid_feature_masks_skipped_mean": float(
                np.mean([item["invalid_feature_masks_skipped"] for item in items])
            ),
        }

        for metric_name in metric_names:
            values = [
                item["holdout_metrics"][metric_name]
                for item in items
                if metric_name in item["holdout_metrics"]
            ]
            summary[f"{metric_name}_mean"] = float(np.mean(values))
            summary[f"{metric_name}_std"] = float(np.std(values))

        summaries.append(summary)

    return summaries


def print_summary_table(summaries: list[dict[str, Any]]) -> None:
    columns = [
        "scenario",
        "n_jobs",
        "runs",
        "fit_seconds_mean",
        "actual_cross_validate_calls_mean",
        "duplicate_or_cache_reuses_mean",
        "invalid_feature_masks_skipped_mean",
        "accuracy_mean",
        "roc_auc_mean",
        "balanced_accuracy_mean",
        "f1_mean",
    ]

    print("\nBenchmark summary")
    print("=================")
    print("\t".join(columns))
    for summary in summaries:
        row = []
        for column in columns:
            value = summary.get(column, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            row.append(str(value))
        print("\t".join(row))


def parse_n_jobs(value: str) -> int | None:
    if value.lower() in {"none", "null"}:
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=1, help="Number of repeated runs per scenario.")
    parser.add_argument("--generations", type=int, default=4, help="GA generations per run.")
    parser.add_argument("--population-size", type=int, default=8, help="GA population size.")
    parser.add_argument("--cv-splits", type=int, default=3, help="Cross-validation splits.")
    parser.add_argument(
        "--n-jobs",
        nargs="+",
        default=["1", "-1"],
        help="One or more n_jobs values to compare. Use 'none' for None.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a smaller benchmark for quick local smoke checks.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save raw results and aggregate summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.runs = 1
        args.generations = min(args.generations, 2)
        args.population_size = min(args.population_size, 5)

    n_jobs_values = [parse_n_jobs(value) for value in args.n_jobs]

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=42,
    )

    results: list[dict[str, Any]] = []

    for run_index in range(args.runs):
        random_state = 42 + run_index
        random.seed(random_state)
        np.random.seed(random_state)
        cv = StratifiedKFold(
            n_splits=args.cv_splits,
            shuffle=True,
            random_state=random_state,
        )

        for n_jobs in n_jobs_values:
            results.append(
                run_one_benchmark(
                    name="GASearchCV",
                    estimator=build_gasearch(
                        random_state=random_state,
                        cv=cv,
                        population_size=args.population_size,
                        generations=args.generations,
                        n_jobs=n_jobs,
                    ),
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    n_jobs=n_jobs,
                    run_index=run_index,
                )
            )

            results.append(
                run_one_benchmark(
                    name="GAFeatureSelectionCV",
                    estimator=build_feature_selector(
                        random_state=random_state,
                        cv=cv,
                        population_size=args.population_size,
                        generations=args.generations,
                        n_jobs=n_jobs,
                        max_features=max(2, X_train.shape[1] // 3),
                    ),
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    n_jobs=n_jobs,
                    run_index=run_index,
                )
            )

    summaries = aggregate_results(results)
    print_summary_table(summaries)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps({"results": results, "summaries": summaries}, indent=2),
            encoding="utf-8",
        )
        print(f"\nSaved benchmark results to {args.output_json}")


if __name__ == "__main__":
    main()
