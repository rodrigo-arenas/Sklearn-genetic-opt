"""CASH benchmark: combined algorithm selection + hyperparameter tuning.

This is the regime where evolutionary search has a genuine *structural* reason to
do well: the optimizer must pick **which estimator family** to use *and* tune that
family's hyperparameters at the same time. The search space is highly conditional
— ``svm__gamma`` only matters when the kernel SVM is chosen, ``rf__max_depth``
only when the random forest is chosen, and so on.

It is a deliberately *fair* fight:

* **Optuna** gets its native, define-by-run conditional API — it suggests the
  family first and then only that family's parameters. Conditional spaces are one
  of Optuna's headline strengths, so it is not handicapped.
* **GASearchCV** and **RandomizedSearchCV** get the same problem through a flat
  encoding: a single ``CASHClassifier`` meta-estimator that exposes every family's
  hyperparameters as flat parameters and routes to the selected family at ``fit``.
  Both pay the flat-encoding tax of carrying inactive genes.

The dataset is chosen to have **real headroom** (it does not saturate near 1.0),
so there is room for the optimizers to actually separate, and the decision
boundary is non-linear so the estimator family genuinely matters.

Optuna and SciPy are optional benchmarking dependencies::

    pip install sklearn-genetic-opt[benchmark]

Run from the repository root, for example::

    python benchmarks/benchmark_cash.py --quick
    python benchmarks/benchmark_cash.py --dataset synth --budget 100 --seeds 3
    python benchmarks/benchmark_cash.py --dataset digits --optimizers gasearch optuna randomized \
        --output-json benchmarks/cash.json
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import load_digits, make_classification
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="sklearn-genetic-cash-"))

from sklearn_genetic import (
    EvolutionConfig,
    GASearchCV,
    OptimizationConfig,
    PopulationConfig,
    RuntimeConfig,
)

# Reuse the spec -> optimizer-space converters from the bayesmark benchmark.
from benchmark_bayesmark import to_scipy_distributions, to_sklearn_genetic_space

try:  # Optuna is an optional benchmarking dependency.
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:  # pragma: no cover - exercised only without the extra installed
    HAS_OPTUNA = False


# ---------------------------------------------------------------------------
# Family search spaces (single source of truth)
# Spec entry: (param_name, type, space, payload)  -- same convention as bayesmark.
# The flat parameter name exposed by CASHClassifier is f"{family}_{param}".
# ---------------------------------------------------------------------------
FAMILY_SPACES: dict[str, list[tuple[str, str, str, Any]]] = {
    "svm": [
        ("svm_C", "real", "log", (1e-2, 1e3)),
        ("svm_gamma", "real", "log", (1e-5, 1e1)),
    ],
    "rf": [
        ("rf_n_estimators", "int", "linear", (50, 150)),
        ("rf_max_depth", "cat", "choice", [None, 4, 8, 16, 32]),
        ("rf_max_features", "real", "linear", (0.1, 1.0)),
        ("rf_min_samples_leaf", "int", "linear", (1, 20)),
    ],
    "histgb": [
        ("histgb_learning_rate", "real", "log", (1e-3, 5e-1)),
        ("histgb_max_iter", "int", "linear", (50, 150)),
        ("histgb_max_leaf_nodes", "int", "linear", (15, 127)),
        ("histgb_l2_regularization", "real", "log", (1e-6, 1e1)),
    ],
    "logreg": [
        ("logreg_C", "real", "log", (1e-3, 1e2)),
    ],
    "knn": [
        ("knn_n_neighbors", "int", "linear", (1, 40)),
        ("knn_p", "int", "linear", (1, 2)),
        ("knn_weights", "cat", "choice", ["uniform", "distance"]),
    ],
}

FAMILIES = list(FAMILY_SPACES)
SCALE_SENSITIVE = {"svm", "logreg", "knn"}


# ---------------------------------------------------------------------------
# CASH meta-estimator: flat parameters, routes to the selected family at fit.
# Every parameter is declared explicitly so sklearn's get_params/clone see them.
# ---------------------------------------------------------------------------
class CASHClassifier(ClassifierMixin, BaseEstimator):
    """One estimator that selects a family and its hyperparameters via flat params."""

    def __init__(
        self,
        estimator: str = "rf",
        random_state: int | None = None,
        # svm
        svm_C: float = 1.0,
        svm_gamma: float = 1e-2,
        # rf
        rf_n_estimators: int = 100,
        rf_max_depth: int | None = None,
        rf_max_features: float = 0.5,
        rf_min_samples_leaf: int = 1,
        # histgb
        histgb_learning_rate: float = 0.1,
        histgb_max_iter: int = 100,
        histgb_max_leaf_nodes: int = 31,
        histgb_l2_regularization: float = 1e-6,
        # logreg
        logreg_C: float = 1.0,
        # knn
        knn_n_neighbors: int = 5,
        knn_p: int = 2,
        knn_weights: str = "uniform",
    ):
        self.estimator = estimator
        self.random_state = random_state
        self.svm_C = svm_C
        self.svm_gamma = svm_gamma
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.rf_max_features = rf_max_features
        self.rf_min_samples_leaf = rf_min_samples_leaf
        self.histgb_learning_rate = histgb_learning_rate
        self.histgb_max_iter = histgb_max_iter
        self.histgb_max_leaf_nodes = histgb_max_leaf_nodes
        self.histgb_l2_regularization = histgb_l2_regularization
        self.logreg_C = logreg_C
        self.knn_n_neighbors = knn_n_neighbors
        self.knn_p = knn_p
        self.knn_weights = knn_weights

    def _build_family(self):
        name = self.estimator
        if name == "svm":
            return SVC(
                C=self.svm_C, gamma=self.svm_gamma, kernel="rbf", random_state=self.random_state
            )
        if name == "rf":
            return RandomForestClassifier(
                n_estimators=self.rf_n_estimators,
                max_depth=self.rf_max_depth,
                max_features=self.rf_max_features,
                min_samples_leaf=self.rf_min_samples_leaf,
                random_state=self.random_state,
                n_jobs=1,
            )
        if name == "histgb":
            return HistGradientBoostingClassifier(
                learning_rate=self.histgb_learning_rate,
                max_iter=self.histgb_max_iter,
                max_leaf_nodes=self.histgb_max_leaf_nodes,
                l2_regularization=self.histgb_l2_regularization,
                early_stopping=False,
                random_state=self.random_state,
            )
        if name == "logreg":
            return LogisticRegression(
                C=self.logreg_C, max_iter=2000, random_state=self.random_state
            )
        if name == "knn":
            return KNeighborsClassifier(
                n_neighbors=self.knn_n_neighbors, p=self.knn_p, weights=self.knn_weights
            )
        raise ValueError(f"Unknown estimator family: {name!r}")

    def fit(self, X, y):
        model = self._build_family()
        if self.estimator in SCALE_SENSITIVE:
            model = make_pipeline(StandardScaler(), model)
        self.model_ = model.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)


# ---------------------------------------------------------------------------
# Flat encoding for GASearchCV and RandomizedSearchCV
# ---------------------------------------------------------------------------
def flat_specs() -> list[tuple[str, str, str, Any]]:
    specs: list[tuple[str, str, str, Any]] = [("estimator", "cat", "choice", list(FAMILIES))]
    for family in FAMILIES:
        specs.extend(FAMILY_SPACES[family])
    return specs


# ---------------------------------------------------------------------------
# Datasets with real headroom (non-saturating, non-linear boundary)
# ---------------------------------------------------------------------------
def load_synth() -> tuple[np.ndarray, np.ndarray]:
    return make_classification(
        n_samples=1500,
        n_features=25,
        n_informative=9,
        n_redundant=5,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=3,
        class_sep=0.9,
        flip_y=0.02,
        random_state=0,
    )


DATASETS: dict[str, Callable[[], tuple[np.ndarray, np.ndarray]]] = {
    "synth": load_synth,
    "digits": lambda: load_digits(return_X_y=True),
}


# ---------------------------------------------------------------------------
# Optimizer runs (equal evaluation budget; return best CV accuracy)
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    optimizer: str
    best_score: float
    evaluations: int
    seconds: float


def make_cv(seed: int, splits: int) -> StratifiedKFold:
    return StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)


def run_randomized(X, y, seed: int, budget: int, cv_splits: int) -> RunResult:
    search = RandomizedSearchCV(
        estimator=CASHClassifier(random_state=seed),
        param_distributions=to_scipy_distributions(flat_specs(), ""),
        n_iter=budget,
        scoring="accuracy",
        cv=make_cv(seed, cv_splits),
        random_state=seed,
        n_jobs=-1,
        refit=False,
        error_score=np.nan,
    )
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X, y)
    return RunResult("randomized", float(search.best_score_), budget, time.perf_counter() - started)


def run_optuna(X, y, seed: int, budget: int, cv_splits: int) -> RunResult:
    cv = make_cv(seed, cv_splits)

    def suggest_family(trial: "optuna.Trial", family: str) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for name, kind, space, payload in FAMILY_SPACES[family]:
            if kind == "cat":
                params[name] = trial.suggest_categorical(name, list(payload))
            elif kind == "int":
                low, high = payload
                params[name] = trial.suggest_int(name, int(low), int(high))
            elif space == "log":
                low, high = payload
                params[name] = trial.suggest_float(name, low, high, log=True)
            else:
                low, high = payload
                params[name] = trial.suggest_float(name, low, high)
        return params

    def objective(trial: "optuna.Trial") -> float:
        family = trial.suggest_categorical("estimator", FAMILIES)
        params = {"estimator": family, **suggest_family(trial, family)}
        clf = CASHClassifier(random_state=seed, **params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(clf, X, y, scoring="accuracy", cv=cv, n_jobs=-1)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    started = time.perf_counter()
    study.optimize(objective, n_trials=budget, show_progress_bar=False)
    return RunResult("optuna", float(study.best_value), budget, time.perf_counter() - started)


def ga_budget(budget: int) -> tuple[int, int]:
    population_size = max(10, int(round(math.sqrt(budget * 2))))
    generations = max(2, round(budget / population_size) - 1)
    return population_size, generations


def run_gasearch(X, y, seed: int, budget: int, cv_splits: int) -> RunResult:
    population_size, generations = ga_budget(budget)
    search = GASearchCV(
        estimator=CASHClassifier(random_state=seed),
        param_grid=to_sklearn_genetic_space(flat_specs(), ""),
        scoring="accuracy",
        cv=make_cv(seed, cv_splits),
        evolution_config=EvolutionConfig(
            population_size=population_size,
            generations=generations,
            tournament_size=3,
            elitism=True,
            keep_top_k=3,
        ),
        population_config=PopulationConfig(initializer="smart"),
        runtime_config=RuntimeConfig(
            n_jobs=-1, parallel_backend="auto", use_cache=True, verbose=False, error_score=np.nan
        ),
        optimization_config=OptimizationConfig(
            local_search=True,
            local_search_top_k=2,
            local_search_steps=1,
            diversity_control=True,
            random_immigrants_fraction=0.15,
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


OPTIMIZERS: dict[str, Callable[..., RunResult]] = {
    "randomized": run_randomized,
    "optuna": run_optuna,
    "gasearch": run_gasearch,
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def aggregate(scores: list[float], evals: list[int], seconds: list[float]) -> dict[str, float]:
    return {
        "mean_score": statistics.fmean(scores),
        "std_score": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        "best_score": max(scores),
        "mean_evaluations": statistics.fmean(evals),
        "mean_seconds": statistics.fmean(seconds),
        "n_seeds": len(scores),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset", default="synth", choices=list(DATASETS), help="Dataset to use."
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=["gasearch", "optuna", "randomized"],
        choices=list(OPTIMIZERS),
        help="Optimizers to compare.",
    )
    parser.add_argument("--budget", type=int, default=100, help="Evaluation budget per optimizer.")
    parser.add_argument("--seeds", type=int, default=3, help="Random seeds to average over.")
    parser.add_argument("--cv-splits", type=int, default=3, help="Cross-validation folds.")
    parser.add_argument(
        "--quick", action="store_true", help="Fast smoke run: small budget and a single seed."
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Write results as JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    budget = 24 if args.quick else args.budget
    seeds = 1 if args.quick else args.seeds

    optimizers = list(args.optimizers)
    if "optuna" in optimizers and not HAS_OPTUNA:
        print(
            "Optuna is not installed; skipping it. Install with: pip install sklearn-genetic-opt[benchmark]\n"
        )
        optimizers = [name for name in optimizers if name != "optuna"]

    X, y = DATASETS[args.dataset]()
    print(
        f"CASH benchmark | dataset={args.dataset} shape={X.shape} families={FAMILIES} "
        f"flat_dims={len(flat_specs())} | optimizers={optimizers} budget={budget} "
        f"seeds={seeds} cv={args.cv_splits}\n"
    )

    # Equal-compute comparison. GASearchCV's realized evaluation count (population
    # x generations, plus local-search refinements) is not known until it runs, and
    # usually exceeds the nominal budget. To keep the comparison fair, GASearchCV
    # runs first each seed and the other optimizers are given its *actual* number of
    # evaluations rather than the nominal budget.
    collected: dict[str, dict[str, list]] = {
        opt: {"scores": [], "evals": [], "seconds": []} for opt in optimizers
    }
    for seed in range(seeds):
        matched_budget = budget
        if "gasearch" in optimizers:
            result = run_gasearch(X, y, seed, budget, args.cv_splits)
            matched_budget = result.evaluations
            collected["gasearch"]["scores"].append(result.best_score)
            collected["gasearch"]["evals"].append(result.evaluations)
            collected["gasearch"]["seconds"].append(result.seconds)
            print(
                f"  {'gasearch':<10} seed={seed} best={result.best_score:.4f} "
                f"evals={result.evaluations} time={result.seconds:.1f}s  "
                f"(other optimizers matched to {matched_budget} evals)"
            )
        for optimizer in optimizers:
            if optimizer == "gasearch":
                continue
            result = OPTIMIZERS[optimizer](X, y, seed, matched_budget, args.cv_splits)
            collected[optimizer]["scores"].append(result.best_score)
            collected[optimizer]["evals"].append(result.evaluations)
            collected[optimizer]["seconds"].append(result.seconds)
            print(
                f"  {optimizer:<10} seed={seed} best={result.best_score:.4f} "
                f"evals={result.evaluations} time={result.seconds:.1f}s"
            )

    results: dict[str, dict[str, float]] = {
        opt: aggregate(c["scores"], c["evals"], c["seconds"]) for opt, c in collected.items()
    }

    print("\n## Results\n")
    header = ["optimizer", "mean accuracy ± std", "best", "mean evals"]
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")
    winner = max(results.items(), key=lambda kv: kv[1]["mean_score"])[0]
    for name in optimizers:
        stats = results[name]
        label = f"**{name}**" if name == winner else name
        print(
            f"| {label} | {stats['mean_score']:.4f} ± {stats['std_score']:.4f} | "
            f"{stats['best_score']:.4f} | {stats['mean_evaluations']:.0f} |"
        )

    print("\nMean wall time per run:")
    for name in optimizers:
        print(
            f"  {name:<10} {results[name]['mean_seconds']:.1f}s  (~{results[name]['mean_evaluations']:.0f} evals)"
        )

    if args.output_json:
        payload = {
            "config": {
                "dataset": args.dataset,
                "shape": list(X.shape),
                "families": FAMILIES,
                "flat_dims": len(flat_specs()),
                "budget": budget,
                "seeds": seeds,
                "cv_splits": args.cv_splits,
                "optimizers": optimizers,
            },
            "results": results,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
