"""Compare GASearchCV against sklearn hyperparameter search methods.

Run from the repository root, for example:

    python benchmarks/benchmark_search_methods.py --quick
    python benchmarks/benchmark_search_methods.py --methods gasearch randomized grid
    python benchmarks/benchmark_search_methods.py --scenarios classification_lr regression_ridge
    python benchmarks/benchmark_search_methods.py --label current --output-json benchmarks/search-current.json
    python benchmarks/benchmark_search_methods.py --compare-json benchmarks/search-baseline.json

The benchmark measures solution quality and solution time for comparable
hyperparameter search workflows. It reports wall time, evaluated candidates,
cross-validation effort, best CV score, holdout metrics, and best parameters.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import randint, loguniform, uniform
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
    RandomizedSearchCV,
    train_test_split,
)

from benchmark_fit import (
    SCENARIOS,
    Scenario,
    build_gasearch,
    holdout_metrics,
    make_cv,
    metric_gap,
)
from sklearn_genetic.space import Categorical, Continuous, Integer

HALVING_MIN_RESOURCES = 100


def numeric_grid(lower: float, upper: float, grid_points: int, log_scale: bool) -> list[float]:
    if log_scale:
        return [float(value) for value in np.geomspace(lower, upper, num=grid_points)]

    return [float(value) for value in np.linspace(lower, upper, num=grid_points)]


def sklearn_param_grid(param_grid: dict[str, object], grid_points: int) -> dict[str, list[Any]]:
    grid = {}
    for parameter, dimension in param_grid.items():
        if isinstance(dimension, Integer):
            values = np.linspace(dimension.lower, dimension.upper, num=grid_points)
            grid[parameter] = sorted({int(round(value)) for value in values})
        elif isinstance(dimension, Continuous):
            grid[parameter] = numeric_grid(
                dimension.lower,
                dimension.upper,
                grid_points,
                log_scale=dimension.distribution == "log-uniform" and dimension.lower > 0,
            )
        elif isinstance(dimension, Categorical):
            grid[parameter] = list(dimension.choices)
        else:
            raise TypeError(f"Unsupported search dimension for {parameter}: {type(dimension)!r}")

    return grid


def sklearn_param_distributions(param_grid: dict[str, object]) -> dict[str, object]:
    distributions = {}
    for parameter, dimension in param_grid.items():
        if isinstance(dimension, Integer):
            distributions[parameter] = randint(dimension.lower, dimension.upper + 1)
        elif isinstance(dimension, Continuous):
            if dimension.distribution == "log-uniform" and dimension.lower > 0:
                distributions[parameter] = loguniform(dimension.lower, dimension.upper)
            else:
                distributions[parameter] = uniform(
                    loc=dimension.lower,
                    scale=dimension.upper - dimension.lower,
                )
        elif isinstance(dimension, Categorical):
            distributions[parameter] = list(dimension.choices)
        else:
            raise TypeError(f"Unsupported search dimension for {parameter}: {type(dimension)!r}")

    return distributions


def build_searcher(
    *,
    method: str,
    scenario: Scenario,
    random_state: int,
    cv,
    n_jobs: int | None,
    n_iter: int,
    grid_points: int,
    ga_population_size: int,
    ga_generations: int,
):
    estimator = scenario.estimator_builder(random_state)
    ga_space = scenario.param_grid_builder()

    if method == "gasearch":
        return build_gasearch(
            scenario=scenario,
            random_state=random_state,
            cv=cv,
            population_size=ga_population_size,
            generations=ga_generations,
            n_jobs=n_jobs,
            parallel_backend="auto",
            population_initializer="smart",
        )

    if method == "grid":
        return GridSearchCV(
            estimator=estimator,
            param_grid=sklearn_param_grid(ga_space, grid_points),
            scoring=scenario.scoring,
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
            return_train_score=False,
        )

    if method == "randomized":
        return RandomizedSearchCV(
            estimator=estimator,
            param_distributions=sklearn_param_distributions(ga_space),
            n_iter=n_iter,
            scoring=scenario.scoring,
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
            random_state=random_state,
            return_train_score=False,
        )

    if method == "halving_grid":
        return HalvingGridSearchCV(
            estimator=estimator,
            param_grid=sklearn_param_grid(ga_space, grid_points),
            scoring=scenario.scoring,
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
            factor=2,
            aggressive_elimination=False,
            min_resources=HALVING_MIN_RESOURCES,
            random_state=random_state,
        )

    if method == "halving_random":
        return HalvingRandomSearchCV(
            estimator=estimator,
            param_distributions=sklearn_param_distributions(ga_space),
            n_candidates=n_iter,
            scoring=scenario.scoring,
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
            factor=2,
            aggressive_elimination=False,
            min_resources=HALVING_MIN_RESOURCES,
            random_state=random_state,
        )

    raise ValueError(f"Unknown search method: {method}")


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()

    return value


def evaluated_candidates(searcher) -> int | None:
    cv_results = getattr(searcher, "cv_results_", None)
    if cv_results is None:
        return None

    return len(cv_results.get("params", []))


def cv_fit_time_summary(searcher) -> dict[str, float | None]:
    cv_results = getattr(searcher, "cv_results_", None)
    if cv_results is None or "mean_fit_time" not in cv_results:
        return {"mean_cv_fit_time": None, "std_cv_fit_time": None}

    return {
        "mean_cv_fit_time": float(np.mean(cv_results["mean_fit_time"])),
        "std_cv_fit_time": float(np.mean(cv_results["std_fit_time"])),
    }


def run_one_benchmark(
    *,
    label: str,
    scenario: Scenario,
    method: str,
    searcher,
    X_train,
    X_test,
    y_train,
    y_test,
    n_jobs: int | None,
    cv_splits: int,
    run_index: int,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    searcher.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started_at

    candidates = evaluated_candidates(searcher)
    best_estimator = getattr(searcher, "best_estimator_", searcher)
    train_metrics = holdout_metrics(best_estimator, X_train, y_train, scenario.task)
    test_metrics = holdout_metrics(best_estimator, X_test, y_test, scenario.task)

    result = {
        "label": label,
        "scenario": scenario.name,
        "task": scenario.task,
        "method": method,
        "run": run_index,
        "n_jobs": n_jobs,
        "cv_splits": cv_splits,
        "fit_seconds": fit_seconds,
        "evaluated_candidates": candidates,
        "candidate_cv_evaluations": candidates * cv_splits if candidates is not None else None,
        "best_score": getattr(searcher, "best_score_", None),
        "best_params": to_jsonable(getattr(searcher, "best_params_", None)),
        "refit_time": getattr(searcher, "refit_time_", None),
        "train_metrics": train_metrics,
        "holdout_metrics": test_metrics,
        "generalization_gap": metric_gap(train_metrics, test_metrics),
        **cv_fit_time_summary(searcher),
    }

    fit_stats = getattr(searcher, "fit_stats_", None)
    if fit_stats is not None:
        result.update(
            {
                "fit_stats_evaluated_candidates": fit_stats["evaluated_candidates"],
                "unique_candidates": fit_stats["unique_candidates"],
                "cache_hits": fit_stats["cache_hits"],
                "duplicate_candidates": fit_stats["duplicate_candidates"],
                "skipped_invalid_candidates": fit_stats["skipped_invalid_candidates"],
                "random_immigrants": fit_stats["random_immigrants"],
                "local_refinement_candidates": fit_stats["local_refinement_candidates"],
            }
        )

    history = getattr(searcher, "history", None)
    if history:
        fitness_best = history.get("fitness_best", [])
        unique_ratios = history.get("unique_individual_ratio", [])
        genotype_diversities = history.get("genotype_diversity", [])
        result.update(
            {
                "generations_ran": int(history["gen"][-1]) if history.get("gen") else None,
                "best_generation": (
                    int(history["best_generation"][-1]) if history.get("best_generation") else None
                ),
                "initial_fitness_best": float(fitness_best[0]) if fitness_best else None,
                "final_fitness_best": float(fitness_best[-1]) if fitness_best else None,
                "fitness_best_improvement": (
                    float(fitness_best[-1] - fitness_best[0]) if len(fitness_best) >= 2 else None
                ),
                "final_unique_individual_ratio": (
                    float(unique_ratios[-1]) if unique_ratios else None
                ),
                "mean_genotype_diversity": (
                    float(np.mean(genotype_diversities)) if genotype_diversities else None
                ),
            }
        )

    return result


def group_key(result: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        result["label"],
        result["scenario"],
        result["method"],
        str(result["n_jobs"]),
    )


def aggregate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(group_key(result), []).append(result)

    def mean_optional(items: list[dict[str, Any]], key: str) -> float | None:
        values = [item[key] for item in items if item.get(key) is not None]
        return float(np.mean(values)) if values else None

    summaries = []
    for (label, scenario, method, n_jobs), items in grouped.items():
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
            "method": method,
            "n_jobs": n_jobs,
            "runs": len(items),
            "fit_seconds_mean": float(np.mean([item["fit_seconds"] for item in items])),
            "fit_seconds_std": float(np.std([item["fit_seconds"] for item in items])),
            "evaluated_candidates_mean": mean_optional(items, "evaluated_candidates"),
            "candidate_cv_evaluations_mean": mean_optional(items, "candidate_cv_evaluations"),
            "best_score_mean": mean_optional(items, "best_score"),
            "mean_cv_fit_time_mean": mean_optional(items, "mean_cv_fit_time"),
            "refit_time_mean": mean_optional(items, "refit_time"),
            "unique_candidates_mean": mean_optional(items, "unique_candidates"),
            "cache_hits_mean": mean_optional(items, "cache_hits"),
            "random_immigrants_mean": mean_optional(items, "random_immigrants"),
            "local_refinement_candidates_mean": mean_optional(items, "local_refinement_candidates"),
            "generations_ran_mean": mean_optional(items, "generations_ran"),
            "best_generation_mean": mean_optional(items, "best_generation"),
            "final_fitness_best_mean": mean_optional(items, "final_fitness_best"),
            "fitness_best_improvement_mean": mean_optional(items, "fitness_best_improvement"),
            "final_unique_individual_ratio_mean": mean_optional(
                items, "final_unique_individual_ratio"
            ),
            "mean_genotype_diversity_mean": mean_optional(items, "mean_genotype_diversity"),
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
        "method",
        "n_jobs",
        "runs",
        "fit_seconds_mean",
        "evaluated_candidates_mean",
        "candidate_cv_evaluations_mean",
        "best_score_mean",
        "final_fitness_best_mean",
        "fitness_best_improvement_mean",
        "final_unique_individual_ratio_mean",
        "mean_genotype_diversity_mean",
        "accuracy_mean",
        "roc_auc_mean",
        "balanced_accuracy_mean",
        "f1_mean",
        "r2_mean",
        "rmse_mean",
        "rmse_gap_mean",
        "mae_mean",
    ]

    print("\nSearch method benchmark summary")
    print("===============================")
    print("\t".join(columns))
    for summary in summaries:
        row = []
        for column in columns:
            value = summary.get(column, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            row.append(str(value))
        print("\t".join(row))


def comparison_key(summary: dict[str, Any]) -> tuple[str, str, str]:
    return (summary["scenario"], summary["method"], str(summary["n_jobs"]))


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
        "method",
        "n_jobs",
        "fit_seconds_delta",
        "fit_seconds_ratio",
        "best_score_delta",
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
            "method": summary["method"],
            "n_jobs": summary["n_jobs"],
            "fit_seconds_delta": summary["fit_seconds_mean"] - base["fit_seconds_mean"],
            "fit_seconds_ratio": summary["fit_seconds_mean"] / base["fit_seconds_mean"],
        }
        for metric_name in ["best_score", "accuracy", "roc_auc", "r2", "rmse"]:
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
    parser.add_argument("--n-iter", type=int, default=60, help="Random search candidate budget.")
    parser.add_argument(
        "--ga-population-size",
        type=int,
        default=12,
        help="GASearchCV population size.",
    )
    parser.add_argument(
        "--ga-generations",
        type=int,
        default=10,
        help="GASearchCV generation count.",
    )
    parser.add_argument(
        "--grid-points", type=int, default=5, help="Grid points per numeric dimension."
    )
    parser.add_argument("--cv-splits", type=int, default=3, help="Cross-validation splits.")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=sorted(SCENARIOS),
        default=["classification_lr", "regression_ridge"],
        help="Benchmark scenarios to run.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["gasearch", "grid", "randomized", "halving_grid", "halving_random"],
        default=["gasearch", "randomized", "grid"],
        help="Search methods to compare.",
    )
    parser.add_argument(
        "--n-jobs",
        nargs="+",
        default=["-1"],
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
        args.n_iter = min(args.n_iter, 6)
        args.ga_population_size = min(args.ga_population_size, 5)
        args.ga_generations = min(args.ga_generations, 2)
        args.grid_points = min(args.grid_points, 3)
        args.scenarios = args.scenarios[:1]

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

            for n_jobs in n_jobs_values:
                for method in args.methods:
                    searcher = build_searcher(
                        method=method,
                        scenario=scenario,
                        random_state=random_state,
                        cv=cv,
                        n_jobs=n_jobs,
                        n_iter=args.n_iter,
                        grid_points=args.grid_points,
                        ga_population_size=args.ga_population_size,
                        ga_generations=args.ga_generations,
                    )
                    results.append(
                        run_one_benchmark(
                            label=args.label,
                            scenario=scenario,
                            method=method,
                            searcher=searcher,
                            X_train=X_train,
                            X_test=X_test,
                            y_train=y_train,
                            y_test=y_test,
                            n_jobs=n_jobs,
                            cv_splits=args.cv_splits,
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
            json.dumps(
                {"results": to_jsonable(results), "summaries": to_jsonable(summaries)}, indent=2
            ),
            encoding="utf-8",
        )
        print(f"\nSaved benchmark results to {args.output_json}")


if __name__ == "__main__":
    main()
