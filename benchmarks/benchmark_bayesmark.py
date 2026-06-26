"""Bayesmark-style comparison of GASearchCV against Optuna and random search.

This benchmark reproduces the experiment design popularized by Uber's
`bayesmark <https://github.com/uber/bayesmark>`_ suite, which Optuna and other
hyperparameter-optimization frameworks use to compare optimizers. The idea is
simple and framework-agnostic:

* take a handful of small, standard scikit-learn datasets,
* tune a handful of standard scikit-learn estimators over fixed search spaces,
* give every optimizer the *same evaluation budget*,
* repeat across several seeds, and
* report the best cross-validation score each optimizer reached.

The datasets and the per-model search spaces below are copied from bayesmark's
``API_CONFIG`` so the comparison matches what the Optuna team benchmarks
against. The only adaptation is that ``logit``-warped parameters are searched on
their natural bounded range (neither sklearn-genetic-opt, Optuna's default
samplers, nor scipy expose a logit warp), which keeps every optimizer on an
equal footing.

Optuna is an *optional* benchmarking dependency. If it is not installed the
script still runs and simply skips the Optuna column::

    pip install sklearn-genetic-opt[benchmark]

Run from the repository root, for example::

    python benchmarks/benchmark_bayesmark.py --quick
    python benchmarks/benchmark_bayesmark.py --datasets wine iris --models svm knn
    python benchmarks/benchmark_bayesmark.py --budget 80 --seeds 5 \
        --optimizers gasearch optuna randomized --output-json benchmarks/bayesmark.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_wine,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="sklearn-genetic-bayesmark-"))

from sklearn_genetic import (
    EvolutionConfig,
    GASearchCV,
    OptimizationConfig,
    PopulationConfig,
    RuntimeConfig,
)
from sklearn_genetic.space import Categorical, Continuous, Integer

try:  # Optuna is an optional benchmarking dependency.
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:  # pragma: no cover - exercised only without the extra installed
    HAS_OPTUNA = False


# ---------------------------------------------------------------------------
# Datasets (bayesmark DATA_LOADERS, minus boston which sklearn removed)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Dataset:
    name: str
    task: str  # "classification" or "regression"
    loader: Callable[[], tuple[np.ndarray, np.ndarray]]


DATASETS: dict[str, Dataset] = {
    "iris": Dataset("iris", "classification", lambda: load_iris(return_X_y=True)),
    "wine": Dataset("wine", "classification", lambda: load_wine(return_X_y=True)),
    "breast": Dataset("breast", "classification", lambda: load_breast_cancer(return_X_y=True)),
    "digits": Dataset("digits", "classification", lambda: load_digits(return_X_y=True)),
    "diabetes": Dataset("diabetes", "regression", lambda: load_diabetes(return_X_y=True)),
}


# ---------------------------------------------------------------------------
# Search spaces (verbatim from bayesmark API_CONFIG)
# Each entry: (param_name, type, space, (low, high))
#   type  in {"real", "int"}
#   space in {"log", "linear", "logit"}  (logit searched on its natural range)
# ---------------------------------------------------------------------------
Spec = tuple[str, str, str, tuple[float, float]]


@dataclass(frozen=True)
class Model:
    name: str
    needs_scaling: bool
    space: list[Spec]
    # builders return a configured estimator for the given task/seed
    classifier: Callable[[int], Any] | None
    regressor: Callable[[int], Any] | None


MODELS: dict[str, Model] = {
    "knn": Model(
        name="knn",
        needs_scaling=True,
        space=[
            ("n_neighbors", "int", "linear", (1, 25)),
            ("p", "int", "linear", (1, 4)),
        ],
        classifier=lambda seed: KNeighborsClassifier(),
        regressor=lambda seed: KNeighborsRegressor(),
    ),
    "svm": Model(
        name="svm",
        needs_scaling=True,
        space=[
            ("C", "real", "log", (1.0, 1e3)),
            ("gamma", "real", "log", (1e-4, 1e-3)),
            ("tol", "real", "log", (1e-5, 1e-1)),
        ],
        classifier=lambda seed: SVC(kernel="rbf", random_state=seed),
        regressor=lambda seed: SVR(kernel="rbf"),
    ),
    "dt": Model(
        name="dt",
        needs_scaling=False,
        space=[
            ("max_depth", "int", "linear", (1, 15)),
            ("min_samples_split", "real", "logit", (0.01, 0.99)),
            ("min_samples_leaf", "real", "logit", (0.01, 0.49)),
            ("min_weight_fraction_leaf", "real", "logit", (0.01, 0.49)),
            ("max_features", "real", "logit", (0.01, 0.99)),
            ("min_impurity_decrease", "real", "linear", (0.0, 0.5)),
        ],
        classifier=lambda seed: DecisionTreeClassifier(random_state=seed),
        regressor=lambda seed: DecisionTreeRegressor(random_state=seed),
    ),
    "rf": Model(
        name="rf",
        needs_scaling=False,
        space=[
            ("max_depth", "int", "linear", (1, 15)),
            ("max_features", "real", "logit", (0.01, 0.99)),
            ("min_samples_split", "real", "logit", (0.01, 0.99)),
            ("min_samples_leaf", "real", "logit", (0.01, 0.49)),
            ("min_weight_fraction_leaf", "real", "logit", (0.01, 0.49)),
            ("min_impurity_decrease", "real", "linear", (0.0, 0.5)),
        ],
        classifier=lambda seed: RandomForestClassifier(
            n_estimators=50, random_state=seed, n_jobs=1
        ),
        regressor=lambda seed: RandomForestRegressor(n_estimators=50, random_state=seed, n_jobs=1),
    ),
    "ada": Model(
        name="ada",
        needs_scaling=False,
        space=[
            ("n_estimators", "int", "linear", (10, 100)),
            ("learning_rate", "real", "log", (1e-4, 1e1)),
        ],
        classifier=lambda seed: AdaBoostClassifier(random_state=seed),
        regressor=lambda seed: AdaBoostRegressor(random_state=seed),
    ),
    "mlp": Model(
        name="mlp",
        needs_scaling=True,
        space=[
            ("hidden_layer_sizes", "int", "linear", (50, 200)),
            ("alpha", "real", "log", (1e-5, 1e1)),
            ("learning_rate_init", "real", "log", (1e-5, 1e-1)),
            ("tol", "real", "log", (1e-5, 1e-1)),
            ("validation_fraction", "real", "logit", (0.1, 0.9)),
        ],
        classifier=lambda seed: MLPClassifier(solver="adam", max_iter=200, random_state=seed),
        regressor=lambda seed: MLPRegressor(solver="adam", max_iter=200, random_state=seed),
    ),
    "lasso": Model(
        name="lasso",
        needs_scaling=True,
        space=[
            ("C", "real", "log", (1e-2, 1e2)),
            ("intercept_scaling", "real", "log", (1e-2, 1e2)),
        ],
        classifier=lambda seed: LogisticRegression(
            penalty="l1", solver="liblinear", max_iter=1000, random_state=seed
        ),
        regressor=lambda seed: Lasso(max_iter=5000, random_state=seed),
    ),
    "linear": Model(
        name="linear",
        needs_scaling=True,
        space=[
            ("C", "real", "log", (1e-2, 1e2)),
            ("intercept_scaling", "real", "log", (1e-2, 1e2)),
        ],
        classifier=lambda seed: LogisticRegression(
            penalty="l2", solver="liblinear", max_iter=1000, random_state=seed
        ),
        regressor=lambda seed: Ridge(random_state=seed),
    ),
}

# bayesmark's "lasso"/"linear" regressors are penalized linear models that do not
# accept ``intercept_scaling``; restrict their tuned space to the shared parameter.
REGRESSION_SPACE_OVERRIDES: dict[str, list[Spec]] = {
    "lasso": [("alpha", "real", "log", (1e-2, 1e2))],
    "linear": [("alpha", "real", "log", (1e-2, 1e2))],
}


# ---------------------------------------------------------------------------
# Search-space converters (one internal spec -> each optimizer's format)
# ---------------------------------------------------------------------------
def is_integer_param(spec: Spec) -> bool:
    return spec[1] == "int"


def to_sklearn_genetic_space(specs: list[Spec], prefix: str) -> dict[str, Any]:
    grid: dict[str, Any] = {}
    for name, kind, space, (low, high) in specs:
        key = f"{prefix}{name}"
        if kind == "int":
            grid[key] = Integer(int(low), int(high))
        elif space == "log":
            grid[key] = Continuous(low, high, distribution="log-uniform")
        else:  # linear or logit -> uniform on the natural range
            grid[key] = Continuous(low, high, distribution="uniform")
    return grid


def to_scipy_distributions(specs: list[Spec], prefix: str) -> dict[str, Any]:
    distributions: dict[str, Any] = {}
    for name, kind, space, (low, high) in specs:
        key = f"{prefix}{name}"
        if kind == "int":
            distributions[key] = randint(int(low), int(high) + 1)
        elif space == "log":
            distributions[key] = loguniform(low, high)
        else:
            distributions[key] = uniform(low, high - low)
    return distributions


def suggest_with_optuna(trial: "optuna.Trial", specs: list[Spec], prefix: str) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, kind, space, (low, high) in specs:
        key = f"{prefix}{name}"
        if kind == "int":
            params[key] = trial.suggest_int(name, int(low), int(high))
        elif space == "log":
            params[key] = trial.suggest_float(name, low, high, log=True)
        else:
            params[key] = trial.suggest_float(name, low, high)
    return params


# ---------------------------------------------------------------------------
# Task wiring
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Task:
    dataset: Dataset
    model: Model
    scoring: str
    higher_is_better: bool


def build_task(dataset: Dataset, model: Model) -> Task | None:
    if dataset.task == "classification":
        if model.classifier is None:
            return None
        return Task(dataset, model, scoring="accuracy", higher_is_better=True)
    if model.regressor is None:
        return None
    return Task(dataset, model, scoring="neg_mean_squared_error", higher_is_better=True)


def task_space(task: Task) -> list[Spec]:
    if task.dataset.task == "regression" and task.model.name in REGRESSION_SPACE_OVERRIDES:
        return REGRESSION_SPACE_OVERRIDES[task.model.name]
    return task.model.space


def base_estimator(task: Task, seed: int) -> Any:
    builder = (
        task.model.classifier if task.dataset.task == "classification" else task.model.regressor
    )
    return builder(seed)


def make_pipeline(task: Task, seed: int) -> tuple[Any, str]:
    """Return (estimator, param_prefix). Scaled models get a Pipeline."""
    estimator = base_estimator(task, seed)
    if task.model.needs_scaling:
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
        return pipeline, "model__"
    return estimator, ""


def make_cv(task: Task, seed: int, splits: int):
    if task.dataset.task == "classification":
        return StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    return KFold(n_splits=splits, shuffle=True, random_state=seed)


# ---------------------------------------------------------------------------
# Optimizer runs (each returns best CV score under the shared budget)
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    optimizer: str
    best_score: float
    evaluations: int
    seconds: float


def run_randomized(task: Task, X, y, seed: int, budget: int, cv_splits: int) -> RunResult:
    estimator, prefix = make_pipeline(task, seed)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=to_scipy_distributions(task_space(task), prefix),
        n_iter=budget,
        scoring=task.scoring,
        cv=make_cv(task, seed, cv_splits),
        random_state=seed,
        n_jobs=-1,
        refit=False,
        error_score="raise",
    )
    started = time.perf_counter()
    search.fit(X, y)
    elapsed = time.perf_counter() - started
    return RunResult("randomized", float(search.best_score_), budget, elapsed)


def run_optuna(task: Task, X, y, seed: int, budget: int, cv_splits: int) -> RunResult:
    specs = task_space(task)
    cv = make_cv(task, seed, cv_splits)

    def objective(trial: "optuna.Trial") -> float:
        estimator, prefix = make_pipeline(task, seed)
        params = suggest_with_optuna(trial, specs, prefix)
        estimator.set_params(**params)
        scores = cross_val_score(estimator, X, y, scoring=task.scoring, cv=cv, n_jobs=-1)
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    started = time.perf_counter()
    study.optimize(objective, n_trials=budget, show_progress_bar=False)
    elapsed = time.perf_counter() - started
    return RunResult("optuna", float(study.best_value), budget, elapsed)


def run_gasearch(task: Task, X, y, seed: int, budget: int, cv_splits: int) -> RunResult:
    estimator, prefix = make_pipeline(task, seed)
    population_size, generations = ga_budget(budget)
    search = GASearchCV(
        estimator=estimator,
        param_grid=to_sklearn_genetic_space(task_space(task), prefix),
        scoring=task.scoring,
        cv=make_cv(task, seed, cv_splits),
        evolution_config=EvolutionConfig(
            population_size=population_size,
            generations=generations,
            tournament_size=3,
            elitism=True,
            keep_top_k=3,
        ),
        population_config=PopulationConfig(initializer="smart"),
        runtime_config=RuntimeConfig(
            n_jobs=-1, parallel_backend="auto", use_cache=True, verbose=False
        ),
        optimization_config=OptimizationConfig(
            local_search=True,
            local_search_top_k=2,
            local_search_steps=1,
            diversity_control=True,
            random_immigrants_fraction=0.10,
            fitness_sharing=True,
        ),
    )
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X, y)
    elapsed = time.perf_counter() - started
    evaluations = int(
        search.fit_stats_.get("evaluated_candidates", population_size * (generations + 1))
    )
    return RunResult("gasearch", float(search.best_score_), evaluations, elapsed)


def ga_budget(budget: int) -> tuple[int, int]:
    """Split an evaluation budget into (population_size, generations).

    ``eaMuPlusLambda`` evaluates ``population_size * (generations + 1)`` candidates
    before caching/deduplication, so we size the grid to land near ``budget``.
    """
    population_size = max(8, int(round(math.sqrt(budget * 2))))
    generations = max(2, round(budget / population_size) - 1)
    return population_size, generations


OPTIMIZERS: dict[str, Callable[..., RunResult]] = {
    "randomized": run_randomized,
    "optuna": run_optuna,
    "gasearch": run_gasearch,
}


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------
@dataclass
class TaskReport:
    dataset: str
    model: str
    task: str
    scoring: str
    by_optimizer: dict[str, dict[str, float]] = field(default_factory=dict)


def aggregate(scores: list[float], evals: list[int], seconds: list[float]) -> dict[str, float]:
    return {
        "mean_score": statistics.fmean(scores),
        "std_score": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        "best_score": max(scores),
        "mean_evaluations": statistics.fmean(evals),
        "mean_seconds": statistics.fmean(seconds),
        "n_seeds": len(scores),
    }


def display_score(value: float, scoring: str) -> float:
    # Show MSE as a positive, human-friendly number for regression.
    if scoring.startswith("neg_"):
        return -value
    return value


def format_markdown(reports: list[TaskReport], optimizers: list[str]) -> str:
    header = ["dataset", "model", "metric", *optimizers, "winner"]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for report in reports:
        metric = "accuracy" if report.scoring == "accuracy" else "MSE (lower=better)"
        cells = [report.dataset, report.model, metric]
        present = {name: stats for name, stats in report.by_optimizer.items()}
        for name in optimizers:
            stats = present.get(name)
            if stats is None:
                cells.append("—")
                continue
            shown = display_score(stats["mean_score"], report.scoring)
            std = stats["std_score"]
            cells.append(f"{shown:.4f} ± {std:.4f}")
        # Winner is always by the maximized objective (higher raw score is better).
        if present:
            winner = max(present.items(), key=lambda kv: kv[1]["mean_score"])[0]
        else:
            winner = "—"
        cells.append(f"**{winner}**")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None, choices=list(DATASETS), help="Datasets to run."
    )
    parser.add_argument(
        "--models", nargs="+", default=None, choices=list(MODELS), help="Models to tune."
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=["gasearch", "optuna", "randomized"],
        choices=list(OPTIMIZERS),
        help="Optimizers to compare.",
    )
    parser.add_argument(
        "--budget", type=int, default=80, help="Function-evaluation budget per optimizer."
    )
    parser.add_argument(
        "--seeds", type=int, default=3, help="Number of random seeds to average over."
    )
    parser.add_argument("--cv-splits", type=int, default=3, help="Cross-validation folds.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast smoke matrix: a few datasets/models, smaller budget and seed count.",
    )
    parser.add_argument(
        "--output-json", type=Path, default=None, help="Write full results as JSON."
    )
    return parser.parse_args(argv)


def resolve_matrix(args: argparse.Namespace) -> tuple[list[str], list[str], int, int]:
    if args.quick:
        datasets = args.datasets or ["wine", "breast"]
        models = args.models or ["svm", "knn", "rf"]
        budget = args.budget if args.budget != 80 else 24
        seeds = args.seeds if args.seeds != 3 else 2
        return datasets, models, budget, seeds
    datasets = args.datasets or ["iris", "wine", "breast", "digits", "diabetes"]
    models = args.models or ["knn", "svm", "dt", "rf", "ada"]
    return datasets, models, args.budget, args.seeds


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dataset_names, model_names, budget, seeds = resolve_matrix(args)

    optimizers = list(args.optimizers)
    if "optuna" in optimizers and not HAS_OPTUNA:
        print(
            "Optuna is not installed; skipping it. Install with: pip install sklearn-genetic-opt[benchmark]\n"
        )
        optimizers = [name for name in optimizers if name != "optuna"]

    print(
        f"Bayesmark-style benchmark | datasets={dataset_names} models={model_names} "
        f"optimizers={optimizers} budget={budget} seeds={seeds} cv={args.cv_splits}\n"
    )

    reports: list[TaskReport] = []
    for dataset_name in dataset_names:
        dataset = DATASETS[dataset_name]
        X, y = dataset.loader()
        for model_name in model_names:
            task = build_task(dataset, MODELS[model_name])
            if task is None:
                continue
            report = TaskReport(dataset_name, model_name, dataset.task, task.scoring)
            for optimizer in optimizers:
                runner = OPTIMIZERS[optimizer]
                scores, evals, seconds = [], [], []
                for seed in range(seeds):
                    try:
                        result = runner(task, X, y, seed, budget, args.cv_splits)
                    except Exception as error:  # noqa: BLE001 - keep the matrix going
                        print(
                            f"  ! {dataset_name}/{model_name}/{optimizer} seed={seed} failed: {error}"
                        )
                        continue
                    scores.append(result.best_score)
                    evals.append(result.evaluations)
                    seconds.append(result.seconds)
                if scores:
                    report.by_optimizer[optimizer] = aggregate(scores, evals, seconds)
            reports.append(report)
            summary = " | ".join(
                f"{name}={display_score(stats['mean_score'], task.scoring):.4f}"
                for name, stats in report.by_optimizer.items()
            )
            print(f"  {dataset_name:>9} / {model_name:<6} -> {summary}")

    print("\n## Results\n")
    print(format_markdown(reports, optimizers))

    if args.output_json:
        payload = {
            "config": {
                "budget": budget,
                "seeds": seeds,
                "cv_splits": args.cv_splits,
                "optimizers": optimizers,
            },
            "results": [
                {
                    "dataset": report.dataset,
                    "model": report.model,
                    "task": report.task,
                    "scoring": report.scoring,
                    "optimizers": report.by_optimizer,
                }
                for report in reports
            ],
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
