"""Benchmark GA fit-time mechanics and holdout model quality.

Run from the repository root, for example:

    python benchmarks/benchmark_fit.py --quick
    python benchmarks/benchmark_fit.py --scenarios classification_lr regression_ridge
    python benchmarks/benchmark_fit.py --parallel-backends auto cv --runs 3
    python benchmarks/benchmark_fit.py --label current --output-json benchmarks/current.json
    python benchmarks/benchmark_fit.py --compare-json benchmarks/baseline.json

The benchmark intentionally lives outside the test suite. It measures wall time,
candidate evaluation counters, serial/parallel strategies, and holdout model
quality metrics for GASearchCV and GAFeatureSelectionCV.
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
from typing import Any, Callable

import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, make_classification, make_friedman1
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="sklearn-genetic-bench-mpl-"))

from sklearn_genetic import (
    EvolutionConfig,
    GAFeatureSelectionCV,
    GASearchCV,
    OptimizationConfig,
    PopulationConfig,
    RuntimeConfig,
)
from sklearn_genetic import genetic_search
from sklearn_genetic.space import Categorical, Continuous, Integer


@dataclass(frozen=True)
class Scenario:
    name: str
    task: str
    scoring: str
    loader: Callable[[], tuple[np.ndarray, np.ndarray]]
    estimator_builder: Callable[[int], object]
    param_grid_builder: Callable[[], dict[str, object]]


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


def classification_data() -> tuple[np.ndarray, np.ndarray]:
    return load_breast_cancer(return_X_y=True)


def regression_data() -> tuple[np.ndarray, np.ndarray]:
    return load_diabetes(return_X_y=True)


def synthetic_classification_data() -> tuple[np.ndarray, np.ndarray]:
    return make_classification(
        n_samples=1800,
        n_features=45,
        n_informative=14,
        n_redundant=10,
        n_repeated=2,
        n_clusters_per_class=3,
        class_sep=0.9,
        flip_y=0.03,
        random_state=42,
    )


def synthetic_regression_data() -> tuple[np.ndarray, np.ndarray]:
    return make_friedman1(n_samples=1800, n_features=25, noise=1.0, random_state=42)


def logistic_pipeline(random_state: int) -> Pipeline:
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


def random_forest_classifier(random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(random_state=random_state, n_jobs=1)


def hist_gradient_boosting_classifier(random_state: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(random_state=random_state, early_stopping=False)


def ridge_pipeline(random_state: int) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])


SCENARIOS = {
    "classification_lr": Scenario(
        name="classification_lr",
        task="classification",
        scoring="roc_auc",
        loader=classification_data,
        estimator_builder=logistic_pipeline,
        param_grid_builder=lambda: {
            "clf__C": Continuous(1e-4, 100.0, distribution="log-uniform"),
            "clf__class_weight": Categorical([None, "balanced"]),
        },
    ),
    "classification_rf": Scenario(
        name="classification_rf",
        task="classification",
        scoring="roc_auc",
        loader=classification_data,
        estimator_builder=random_forest_classifier,
        param_grid_builder=lambda: {
            "n_estimators": Integer(80, 240),
            "max_depth": Integer(2, 14),
            "min_samples_split": Integer(2, 12),
            "min_samples_leaf": Integer(1, 8),
            "max_features": Categorical(["sqrt", "log2", None]),
            "ccp_alpha": Continuous(0.0, 0.03),
        },
    ),
    "regression_ridge": Scenario(
        name="regression_ridge",
        task="regression",
        scoring="r2",
        loader=regression_data,
        estimator_builder=ridge_pipeline,
        param_grid_builder=lambda: {
            "ridge__alpha": Continuous(1e-4, 1000.0, distribution="log-uniform"),
            "ridge__fit_intercept": Categorical([True, False]),
        },
    ),
    "classification_rf_synthetic": Scenario(
        name="classification_rf_synthetic",
        task="classification",
        scoring="roc_auc",
        loader=synthetic_classification_data,
        estimator_builder=random_forest_classifier,
        param_grid_builder=lambda: {
            "n_estimators": Integer(100, 320),
            "max_depth": Integer(3, 18),
            "min_samples_split": Integer(2, 16),
            "min_samples_leaf": Integer(1, 10),
            "max_features": Categorical(["sqrt", "log2", None]),
            "ccp_alpha": Continuous(0.0, 0.02),
        },
    ),
    "classification_gbm": Scenario(
        name="classification_gbm",
        task="classification",
        scoring="roc_auc",
        loader=classification_data,
        estimator_builder=hist_gradient_boosting_classifier,
        param_grid_builder=lambda: {
            "learning_rate": Continuous(0.01, 0.3, distribution="log-uniform"),
            "max_iter": Integer(50, 300),
            "max_depth": Integer(2, 8),
            "min_samples_leaf": Integer(5, 50),
            "l2_regularization": Continuous(1e-6, 1.0, distribution="log-uniform"),
            "max_features": Continuous(0.3, 1.0),
            "max_leaf_nodes": Integer(15, 127),
        },
    ),
    "classification_gbm_synthetic": Scenario(
        name="classification_gbm_synthetic",
        task="classification",
        scoring="roc_auc",
        loader=synthetic_classification_data,
        estimator_builder=hist_gradient_boosting_classifier,
        param_grid_builder=lambda: {
            "learning_rate": Continuous(0.01, 0.3, distribution="log-uniform"),
            "max_iter": Integer(50, 300),
            "max_depth": Integer(2, 8),
            "min_samples_leaf": Integer(5, 50),
            "l2_regularization": Continuous(1e-6, 1.0, distribution="log-uniform"),
            "max_features": Continuous(0.3, 1.0),
            "max_leaf_nodes": Integer(15, 127),
        },
    ),
    "regression_friedman_ridge": Scenario(
        name="regression_friedman_ridge",
        task="regression",
        scoring="r2",
        loader=synthetic_regression_data,
        estimator_builder=ridge_pipeline,
        param_grid_builder=lambda: {
            "ridge__alpha": Continuous(1e-4, 1000.0, distribution="log-uniform"),
            "ridge__fit_intercept": Categorical([True, False]),
        },
    ),
}


def make_cv(scenario: Scenario, cv_splits: int, random_state: int):
    if scenario.task == "classification":
        return StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    return KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)


def build_gasearch(
    *,
    scenario: Scenario,
    random_state: int,
    cv,
    population_size: int,
    generations: int,
    n_jobs: int | None,
    parallel_backend: str,
    population_initializer: str,
    final_selection: bool = False,
    final_selection_top_k: int = 3,
    final_selection_cv=None,
) -> GASearchCV:
    evolution_config = EvolutionConfig(
        population_size=population_size,
        generations=generations,
        tournament_size=3,
        elitism=True,
    )
    population_config = PopulationConfig(initializer=population_initializer)
    runtime_config = RuntimeConfig(
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        verbose=False,
        return_train_score=False,
        use_cache=True,
    )
    optimization_config = OptimizationConfig(
        local_search=True,
        local_search_top_k=2,
        local_search_steps=1,
        local_search_radius=0.15,
        diversity_control=True,
        diversity_threshold=0.25,
        diversity_stagnation_generations=4,
        diversity_mutation_boost=1.5,
        random_immigrants_fraction=0.10,
        adaptive_selection=True,
        selection_pressure_min=2,
        selection_pressure_max=5,
        offspring_diversity_retries=3,
        fitness_sharing=True,
        sharing_radius=0.35,
        final_selection=final_selection,
        final_selection_top_k=final_selection_top_k,
        final_selection_cv=final_selection_cv,
    )
    return GASearchCV(
        estimator=scenario.estimator_builder(random_state),
        cv=cv,
        scoring=scenario.scoring,
        param_grid=scenario.param_grid_builder(),
        evolution_config=evolution_config,
        population_config=population_config,
        runtime_config=runtime_config,
        optimization_config=optimization_config,
    )


def build_feature_selector(
    *,
    scenario: Scenario,
    random_state: int,
    cv,
    population_size: int,
    generations: int,
    n_jobs: int | None,
    parallel_backend: str,
    population_initializer: str,
    max_features: int,
) -> GAFeatureSelectionCV:
    evolution_config = EvolutionConfig(
        population_size=population_size,
        generations=generations,
        tournament_size=3,
        elitism=True,
    )
    population_config = PopulationConfig(initializer=population_initializer)
    runtime_config = RuntimeConfig(
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        verbose=False,
        return_train_score=False,
        use_cache=True,
    )
    optimization_config = OptimizationConfig(
        local_search=True,
        local_search_top_k=2,
        local_search_steps=1,
        local_search_radius=0.15,
        diversity_control=True,
        diversity_threshold=0.25,
        diversity_stagnation_generations=4,
        diversity_mutation_boost=1.5,
        random_immigrants_fraction=0.10,
        adaptive_selection=True,
        selection_pressure_min=2,
        selection_pressure_max=5,
        offspring_diversity_retries=3,
        fitness_sharing=True,
        sharing_radius=0.35,
    )
    return GAFeatureSelectionCV(
        estimator=scenario.estimator_builder(random_state),
        cv=cv,
        scoring=scenario.scoring,
        max_features=max_features,
        evolution_config=evolution_config,
        population_config=population_config,
        runtime_config=runtime_config,
        optimization_config=optimization_config,
    )


def holdout_metrics(estimator, X_test, y_test, task: str) -> dict[str, float]:
    predictions = estimator.predict(X_test)

    if task == "regression":
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        return {
            "r2": r2_score(y_test, predictions),
            "rmse": rmse,
            "mae": mean_absolute_error(y_test, predictions),
        }

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


def metric_gap(train_metrics: dict[str, float], holdout: dict[str, float]) -> dict[str, float]:
    gaps = {}
    for metric_name, train_value in train_metrics.items():
        if metric_name not in holdout:
            continue

        if metric_name == "rmse":
            gaps[f"{metric_name}_gap"] = holdout[metric_name] - train_value
        else:
            gaps[f"{metric_name}_gap"] = train_value - holdout[metric_name]

    return gaps


def get_log_records(estimator) -> list[dict[str, Any]]:
    return list(estimator.logbook.chapters["parameters"])


def fallback_fit_mechanics(estimator, counters: FitCounters) -> dict[str, int]:
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
        "evaluated_candidates": evaluated_records,
        "unique_candidates": counters.cross_validate_calls + invalid_feature_masks_skipped,
        "cross_validate_calls": counters.cross_validate_calls,
        "cache_hits": 0,
        "duplicate_candidates": max(0, duplicate_or_cache_reuses),
        "skipped_invalid_candidates": invalid_feature_masks_skipped,
        "population_parallel_batches": 0,
        "population_serial_batches": 0,
    }


def summarize_fit_mechanics(estimator, counters: FitCounters) -> dict[str, int]:
    stats = getattr(estimator, "fit_stats_", None)
    if stats is None:
        stats = fallback_fit_mechanics(estimator, counters)

    summary = dict(stats)
    summary["actual_cross_validate_calls"] = summary["cross_validate_calls"]
    summary["duplicate_or_cache_reuses"] = summary["cache_hits"] + summary["duplicate_candidates"]
    summary["fitness_cache_size"] = len(getattr(estimator, "fitness_cache", {}))
    return summary


def summarize_optimizer_telemetry(estimator) -> dict[str, float | int | None]:
    history = getattr(estimator, "history", {})
    generations = history.get("gen", [])
    unique_ratios = history.get("unique_individual_ratio", [])
    genotype_diversities = history.get("genotype_diversity", [])
    stagnation_generations = history.get("stagnation_generations", [])
    best_generations = history.get("best_generation", [])
    fitness_best = history.get("fitness_best", [])
    fitness = history.get("fitness", [])
    fitness_max = history.get("fitness_max", [])
    selection_pressure = [
        value for value in history.get("selection_pressure", []) if value is not None
    ]

    return {
        "generations_ran": int(generations[-1]) if generations else None,
        "best_generation": int(best_generations[-1]) if best_generations else None,
        "initial_fitness_best": float(fitness_best[0]) if fitness_best else None,
        "final_fitness_best": float(fitness_best[-1]) if fitness_best else None,
        "fitness_best_improvement": (
            float(fitness_best[-1] - fitness_best[0]) if len(fitness_best) >= 2 else None
        ),
        "final_population_fitness": float(fitness[-1]) if fitness else None,
        "final_population_fitness_max": float(fitness_max[-1]) if fitness_max else None,
        "final_unique_individual_ratio": (float(unique_ratios[-1]) if unique_ratios else None),
        "mean_unique_individual_ratio": (float(np.mean(unique_ratios)) if unique_ratios else None),
        "final_genotype_diversity": (
            float(genotype_diversities[-1]) if genotype_diversities else None
        ),
        "mean_genotype_diversity": (
            float(np.mean(genotype_diversities)) if genotype_diversities else None
        ),
        "final_stagnation_generations": (
            int(stagnation_generations[-1]) if stagnation_generations else None
        ),
        "mean_selection_pressure": (
            float(np.mean(selection_pressure)) if selection_pressure else None
        ),
    }


def run_one_benchmark(
    *,
    label: str,
    scenario: Scenario,
    estimator_name: str,
    estimator,
    X_train,
    X_test,
    y_train,
    y_test,
    n_jobs: int | None,
    parallel_backend: str,
    population_initializer: str,
    run_index: int,
) -> dict[str, Any]:
    counters = FitCounters()
    started_at = time.perf_counter()

    with count_cross_validate_calls(counters):
        estimator.fit(X_train, y_train)

    fit_seconds = time.perf_counter() - started_at
    train_metrics = holdout_metrics(estimator, X_train, y_train, scenario.task)
    test_metrics = holdout_metrics(estimator, X_test, y_test, scenario.task)

    result = {
        "label": label,
        "scenario": scenario.name,
        "task": scenario.task,
        "estimator": estimator_name,
        "run": run_index,
        "n_jobs": n_jobs,
        "parallel_backend": parallel_backend,
        "population_initializer": population_initializer,
        "fit_seconds": fit_seconds,
        **summarize_fit_mechanics(estimator, counters),
        **summarize_optimizer_telemetry(estimator),
        "best_score": getattr(estimator, "best_score_", None),
        "train_metrics": train_metrics,
        "holdout_metrics": test_metrics,
        "generalization_gap": metric_gap(train_metrics, test_metrics),
    }

    final_selection_results = getattr(estimator, "final_selection_results_", None)
    if final_selection_results is not None:
        result.update(
            {
                "final_selection_enabled": final_selection_results["enabled"],
                "final_selection_changed": final_selection_results["changed"],
                "final_selection_top_k": final_selection_results["top_k"],
                "final_selection_cv": final_selection_results["cv"],
                "final_selection_time_seconds": final_selection_results["time_seconds"],
                "final_selection_original_score": final_selection_results["original_best_score"],
                "final_selection_selected_score": final_selection_results["selected_score"],
            }
        )

    if hasattr(estimator, "support_"):
        result["selected_features"] = int(np.sum(estimator.support_))

    return result


def group_key(result: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        result["label"],
        result["scenario"],
        result["estimator"],
        str(result["n_jobs"]),
        result["parallel_backend"],
        result["population_initializer"],
    )


def aggregate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(group_key(result), []).append(result)

    def mean_optional(items: list[dict[str, Any]], key: str) -> float | None:
        values = [item[key] for item in items if item.get(key) is not None]
        return float(np.mean(values)) if values else None

    summaries = []
    for (
        label,
        scenario,
        estimator,
        n_jobs,
        parallel_backend,
        population_initializer,
    ), items in grouped.items():
        metric_names = sorted(
            {metric_name for item in items for metric_name in item["holdout_metrics"].keys()}
        )
        train_metric_names = sorted(
            {metric_name for item in items for metric_name in item.get("train_metrics", {}).keys()}
        )
        gap_metric_names = sorted(
            {metric_name for item in items for metric_name in item.get("generalization_gap", {})}
        )
        summary = {
            "label": label,
            "scenario": scenario,
            "estimator": estimator,
            "n_jobs": n_jobs,
            "parallel_backend": parallel_backend,
            "population_initializer": population_initializer,
            "runs": len(items),
            "fit_seconds_mean": float(np.mean([item["fit_seconds"] for item in items])),
            "fit_seconds_std": float(np.std([item["fit_seconds"] for item in items])),
            "actual_cross_validate_calls_mean": float(
                np.mean([item["actual_cross_validate_calls"] for item in items])
            ),
            "duplicate_or_cache_reuses_mean": float(
                np.mean([item["duplicate_or_cache_reuses"] for item in items])
            ),
            "skipped_invalid_candidates_mean": float(
                np.mean([item["skipped_invalid_candidates"] for item in items])
            ),
            "population_parallel_batches_mean": float(
                np.mean([item["population_parallel_batches"] for item in items])
            ),
            "generations_ran_mean": mean_optional(items, "generations_ran"),
            "best_generation_mean": mean_optional(items, "best_generation"),
            "initial_fitness_best_mean": mean_optional(items, "initial_fitness_best"),
            "final_fitness_best_mean": mean_optional(items, "final_fitness_best"),
            "fitness_best_improvement_mean": mean_optional(items, "fitness_best_improvement"),
            "final_population_fitness_mean": mean_optional(items, "final_population_fitness"),
            "final_population_fitness_max_mean": mean_optional(
                items, "final_population_fitness_max"
            ),
            "final_unique_individual_ratio_mean": mean_optional(
                items, "final_unique_individual_ratio"
            ),
            "mean_unique_individual_ratio_mean": mean_optional(
                items, "mean_unique_individual_ratio"
            ),
            "final_genotype_diversity_mean": mean_optional(items, "final_genotype_diversity"),
            "mean_genotype_diversity_mean": mean_optional(items, "mean_genotype_diversity"),
            "final_stagnation_generations_mean": mean_optional(
                items, "final_stagnation_generations"
            ),
            "mean_selection_pressure_mean": mean_optional(items, "mean_selection_pressure"),
            "final_selection_changed_mean": mean_optional(items, "final_selection_changed"),
            "final_selection_time_seconds_mean": mean_optional(
                items, "final_selection_time_seconds"
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

        for metric_name in train_metric_names:
            values = [
                item["train_metrics"][metric_name]
                for item in items
                if metric_name in item.get("train_metrics", {})
            ]
            summary[f"train_{metric_name}_mean"] = float(np.mean(values))
            summary[f"train_{metric_name}_std"] = float(np.std(values))

        for metric_name in gap_metric_names:
            values = [
                item["generalization_gap"][metric_name]
                for item in items
                if metric_name in item.get("generalization_gap", {})
            ]
            summary[f"{metric_name}_mean"] = float(np.mean(values))
            summary[f"{metric_name}_std"] = float(np.std(values))

        summaries.append(summary)

    return summaries


def print_summary_table(summaries: list[dict[str, Any]]) -> None:
    columns = [
        "label",
        "scenario",
        "estimator",
        "n_jobs",
        "parallel_backend",
        "population_initializer",
        "runs",
        "fit_seconds_mean",
        "actual_cross_validate_calls_mean",
        "duplicate_or_cache_reuses_mean",
        "skipped_invalid_candidates_mean",
        "generations_ran_mean",
        "best_generation_mean",
        "final_fitness_best_mean",
        "fitness_best_improvement_mean",
        "final_unique_individual_ratio_mean",
        "mean_genotype_diversity_mean",
        "final_stagnation_generations_mean",
        "mean_selection_pressure_mean",
        "final_selection_changed_mean",
        "final_selection_time_seconds_mean",
        "accuracy_mean",
        "roc_auc_mean",
        "balanced_accuracy_mean",
        "f1_mean",
        "r2_mean",
        "rmse_mean",
        "rmse_gap_mean",
        "mae_mean",
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


def comparison_key(summary: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        summary["scenario"],
        summary["estimator"],
        str(summary["n_jobs"]),
        summary["parallel_backend"],
        summary["population_initializer"],
    )


def print_comparison_table(current: list[dict[str, Any]], baseline: list[dict[str, Any]]) -> None:
    baseline_by_key = {comparison_key(summary): summary for summary in baseline}
    comparable = [
        (summary, baseline_by_key[comparison_key(summary)])
        for summary in current
        if comparison_key(summary) in baseline_by_key
    ]

    if not comparable:
        print("\nNo comparable baseline rows found.")
        return

    columns = [
        "scenario",
        "estimator",
        "n_jobs",
        "parallel_backend",
        "population_initializer",
        "fit_seconds_delta",
        "fit_seconds_ratio",
        "accuracy_delta",
        "roc_auc_delta",
        "r2_delta",
        "rmse_delta",
    ]
    print("\nComparison against baseline")
    print("===========================")
    print("\t".join(columns))
    for summary, base in comparable:
        row = {
            "scenario": summary["scenario"],
            "estimator": summary["estimator"],
            "n_jobs": summary["n_jobs"],
            "parallel_backend": summary["parallel_backend"],
            "population_initializer": summary["population_initializer"],
            "fit_seconds_delta": summary["fit_seconds_mean"] - base["fit_seconds_mean"],
            "fit_seconds_ratio": summary["fit_seconds_mean"] / base["fit_seconds_mean"],
        }
        for metric_name in ["accuracy", "roc_auc", "r2", "rmse"]:
            current_key = f"{metric_name}_mean"
            if current_key in summary and current_key in base:
                row[f"{metric_name}_delta"] = summary[current_key] - base[current_key]

        print(
            "\t".join(
                (
                    f"{row[column]:.4f}"
                    if isinstance(row.get(column), float)
                    else str(row.get(column, ""))
                )
                for column in columns
            )
        )


def parse_n_jobs(value: str) -> int | None:
    if value.lower() in {"none", "null"}:
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="current", help="Label stored in benchmark results.")
    parser.add_argument("--runs", type=int, default=1, help="Number of repeated runs per scenario.")
    parser.add_argument("--generations", type=int, default=10, help="GA generations per run.")
    parser.add_argument("--population-size", type=int, default=12, help="GA population size.")
    parser.add_argument("--cv-splits", type=int, default=3, help="Cross-validation splits.")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=sorted(SCENARIOS),
        default=[
            "classification_lr",
            "classification_rf",
            "regression_ridge",
            "classification_rf_synthetic",
            "regression_friedman_ridge",
        ],
        help="Benchmark scenarios to run.",
    )
    parser.add_argument(
        "--estimators",
        nargs="+",
        choices=["gasearch", "feature_selection"],
        default=["gasearch", "feature_selection"],
        help="GA estimators to benchmark.",
    )
    parser.add_argument(
        "--n-jobs",
        nargs="+",
        default=["-1"],
        help="One or more n_jobs values to compare. Use 'none' for None.",
    )
    parser.add_argument(
        "--parallel-backends",
        nargs="+",
        choices=["auto", "population", "cv"],
        default=["auto"],
        help="Parallel strategy values to compare.",
    )
    parser.add_argument(
        "--population-initializers",
        nargs="+",
        choices=["smart", "random"],
        default=["smart"],
        help="Initial population strategies to compare.",
    )
    parser.add_argument(
        "--final-selection",
        action="store_true",
        help="Re-evaluate the top GA candidates before selecting the final estimator.",
    )
    parser.add_argument(
        "--final-selection-top-k",
        type=int,
        default=3,
        help="Number of top GA candidates to re-evaluate when final selection is enabled.",
    )
    parser.add_argument(
        "--final-selection-cv-splits",
        type=int,
        default=None,
        help=(
            "Optional CV split count for the final GA candidate re-evaluation. "
            "Defaults to the main CV splitter."
        ),
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
    parser.add_argument(
        "--compare-json",
        type=Path,
        default=None,
        help="Optional previous benchmark JSON to compare against.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.runs = 1
        args.generations = min(args.generations, 2)
        args.population_size = min(args.population_size, 5)

    n_jobs_values = [parse_n_jobs(value) for value in args.n_jobs]

    results: list[dict[str, Any]] = []

    for scenario_name in args.scenarios:
        scenario = SCENARIOS[scenario_name]
        X, y = scenario.loader()
        stratify = y if scenario.task == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            stratify=stratify,
            random_state=42,
        )

        for run_index in range(args.runs):
            random_state = 42 + run_index
            random.seed(random_state)
            np.random.seed(random_state)
            cv = make_cv(scenario, args.cv_splits, random_state)

            for parallel_backend in args.parallel_backends:
                for population_initializer in args.population_initializers:
                    for n_jobs in n_jobs_values:
                        if "gasearch" in args.estimators:
                            results.append(
                                run_one_benchmark(
                                    label=args.label,
                                    scenario=scenario,
                                    estimator_name="GASearchCV",
                                    estimator=build_gasearch(
                                        scenario=scenario,
                                        random_state=random_state,
                                        cv=cv,
                                        population_size=args.population_size,
                                        generations=args.generations,
                                        n_jobs=n_jobs,
                                        parallel_backend=parallel_backend,
                                        population_initializer=population_initializer,
                                        final_selection=args.final_selection,
                                        final_selection_top_k=args.final_selection_top_k,
                                        final_selection_cv=args.final_selection_cv_splits,
                                    ),
                                    X_train=X_train,
                                    X_test=X_test,
                                    y_train=y_train,
                                    y_test=y_test,
                                    n_jobs=n_jobs,
                                    parallel_backend=parallel_backend,
                                    population_initializer=population_initializer,
                                    run_index=run_index,
                                )
                            )

                        if "feature_selection" in args.estimators:
                            results.append(
                                run_one_benchmark(
                                    label=args.label,
                                    scenario=scenario,
                                    estimator_name="GAFeatureSelectionCV",
                                    estimator=build_feature_selector(
                                        scenario=scenario,
                                        random_state=random_state,
                                        cv=cv,
                                        population_size=args.population_size,
                                        generations=args.generations,
                                        n_jobs=n_jobs,
                                        parallel_backend=parallel_backend,
                                        population_initializer=population_initializer,
                                        max_features=max(2, X_train.shape[1] // 3),
                                    ),
                                    X_train=X_train,
                                    X_test=X_test,
                                    y_train=y_train,
                                    y_test=y_test,
                                    n_jobs=n_jobs,
                                    parallel_backend=parallel_backend,
                                    population_initializer=population_initializer,
                                    run_index=run_index,
                                )
                            )

    summaries = aggregate_results(results)
    print_summary_table(summaries)

    if args.compare_json is not None:
        baseline_payload = json.loads(args.compare_json.read_text(encoding="utf-8"))
        print_comparison_table(summaries, baseline_payload["summaries"])

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps({"results": results, "summaries": summaries}, indent=2),
            encoding="utf-8",
        )
        print(f"\nSaved benchmark results to {args.output_json}")


if __name__ == "__main__":
    main()
