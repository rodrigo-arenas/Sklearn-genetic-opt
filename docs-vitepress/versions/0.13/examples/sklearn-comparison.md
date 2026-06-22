---
title: Comparing GASearchCV With sklearn Search Methods
description: Side-by-side comparison of GASearchCV, RandomizedSearchCV, and GridSearchCV on a binary classification task — solution quality, search cost, and runtime.
---
# Comparing GASearchCV With sklearn Search Methods

This example compares `GASearchCV` with `RandomizedSearchCV` and `GridSearchCV` on the same classification problem. The goal is not to declare one method universally best — it is to show how to compare solution quality, search cost, and runtime fairly.

## Problem Setup

Breast cancer binary classification with a scaled logistic-regression pipeline. The search space includes continuous and categorical choices.

```python
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn_genetic import EvolutionConfig, GASearchCV, OptimizationConfig, PopulationConfig, RuntimeConfig
from sklearn_genetic.callbacks import ConsecutiveStopping, DeltaThreshold, TimerStopping
from sklearn_genetic.schedules import ExponentialAdapter, InverseAdapter
from sklearn_genetic.space import Categorical, Continuous

warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
```

## Shared Helpers

Each method receives the same estimator family and train/test split. We report both CV score and holdout metrics.

```python
def make_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(solver="liblinear", max_iter=500, random_state=RANDOM_STATE)),
    ])


def evaluate_classifier(estimator):
    predictions = estimator.predict(X_test)
    probabilities = estimator.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def summarize_search(name, estimator, fit_seconds):
    cv_results = getattr(estimator, "cv_results_", {})
    evaluated_candidates = len(cv_results.get("params", []))
    row = {
        "method": name,
        "fit_seconds": fit_seconds,
        "evaluated_candidates": evaluated_candidates,
        "estimated_cv_evaluations": evaluated_candidates * cv.get_n_splits(),
        "best_cv_score": getattr(estimator, "best_score_", None),
    }
    row.update(evaluate_classifier(estimator))
    return row
```

## RandomizedSearchCV

Random search samples a fixed number of candidates. It is often a strong baseline for continuous spaces.

```python
randomized_search = RandomizedSearchCV(
    estimator=make_model(),
    param_distributions={
        "logistic__C": loguniform(1e-3, 30.0),
        "logistic__class_weight": [None, "balanced"],
    },
    n_iter=16,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    refit=True,
)

started_at = time.perf_counter()
randomized_search.fit(X_train, y_train)
randomized_seconds = time.perf_counter() - started_at
```

## GridSearchCV

Grid search is deterministic and easy to reason about. It becomes expensive when every additional dimension multiplies the candidate count.

```python
grid_search = GridSearchCV(
    estimator=make_model(),
    param_grid={
        "logistic__C": np.geomspace(1e-3, 30.0, num=8),
        "logistic__class_weight": [None, "balanced"],
    },
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    refit=True,
)

started_at = time.perf_counter()
grid_search.fit(X_train, y_train)
grid_seconds = time.perf_counter() - started_at
```

## GASearchCV

The genetic version uses the same parameter region with `sklearn-genetic-opt` spaces and enables optimizer controls that help in mixed search spaces.

```python
ga_search = GASearchCV(
    estimator=make_model(),
    param_grid={
        "logistic__C": Continuous(1e-3, 30.0, distribution="log-uniform"),
        "logistic__class_weight": Categorical([None, "balanced"]),
    },
    scoring="roc_auc",
    cv=cv,
    evolution_config=EvolutionConfig(
        population_size=10,
        generations=8,
        crossover_probability=ExponentialAdapter(initial_value=0.8, end_value=0.4, adaptive_rate=0.15),
        mutation_probability=InverseAdapter(initial_value=0.25, end_value=0.08, adaptive_rate=0.25),
        tournament_size=3,
        elitism=True,
        keep_top_k=3,
    ),
    population_config=PopulationConfig(
        initializer="smart",
        warm_start_configs=[{"logistic__C": 1.0, "logistic__class_weight": None}],
    ),
    runtime_config=RuntimeConfig(n_jobs=-1, parallel_backend="auto", use_cache=True, verbose=True),
    optimization_config=OptimizationConfig(
        local_search=True,
        local_search_top_k=2,
        local_search_steps=1,
        diversity_control=True,
        random_immigrants_fraction=0.10,
        fitness_sharing=True,
    ),
)

callbacks = [
    DeltaThreshold(threshold=0.0005, generations=5, metric="fitness_best"),
    ConsecutiveStopping(generations=7, metric="fitness_best"),
    TimerStopping(total_seconds=90),
]

started_at = time.perf_counter()
ga_search.fit(X_train, y_train, callbacks=callbacks)
ga_seconds = time.perf_counter() - started_at
```

## Compare Results

Candidate budgets are not exactly identical, so the table includes evaluated candidates and estimated CV evaluations.

```python
comparison = pd.DataFrame([
    summarize_search("RandomizedSearchCV", randomized_search, randomized_seconds),
    summarize_search("GridSearchCV", grid_search, grid_seconds),
    summarize_search("GASearchCV", ga_search, ga_seconds),
]).sort_values("roc_auc", ascending=False)

print(comparison.to_string())
```

## Reading GA-Specific Telemetry

The sklearn searchers expose `cv_results_`. `GASearchCV` also exposes `fit_stats_` and `history`, which help explain search behavior.

```python
# Summary of evaluation mechanics
print(ga_search.fit_stats_)
# {'evaluated_candidates': 92, 'unique_candidates': 87, 'cache_hits': 5, ...}

# Per-generation telemetry
history = pd.DataFrame(ga_search.history)
print(history[["gen", "fitness", "fitness_max", "unique_individual_ratio", "genotype_diversity"]].tail())
```

## Practical Notes

- Compare methods using both quality metrics and search cost.
- `RandomizedSearchCV` is a strong baseline for continuous spaces.
- `GridSearchCV` is useful when the grid is small and deliberately chosen.
- `GASearchCV` becomes more attractive as the space gets mixed, conditional, rugged, or expensive — where smarter exploration pays off.
- For repeatable conclusions, run several seeds or use the repository benchmark script: `python benchmarks/benchmark_search_methods.py --runs 3`.

## See Also

- [When to Use](../guide/when-to-use) — decision guide for choosing a search method
- [Understanding Cross-Validation](../guide/understand-cv) — reading the generation log
- [Advanced Optimizer Control](../guide/advanced-optimizer-control) — diversity, fitness sharing, local search
