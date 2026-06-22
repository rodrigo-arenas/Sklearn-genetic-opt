---
title: Multi-Metric Search on Iris
description: Track multiple scorers simultaneously, choose which metric drives the refit, and inspect per-metric cv_results_ after fitting.
---

# Multi-Metric Search on Iris

This example shows how to run `GASearchCV` with multiple scorers and inspect per-metric results after fitting.

## Setup

```python
import warnings
from pprint import pprint

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn_genetic import (
    EvolutionConfig, GASearchCV, OptimizationConfig, PopulationConfig, RuntimeConfig,
)
from sklearn_genetic.callbacks import ConsecutiveStopping, DeltaThreshold, TimerStopping
from sklearn_genetic.schedules import ExponentialAdapter, InverseAdapter
from sklearn_genetic.space import Categorical, Continuous, Integer

warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42

iris = load_iris(as_frame=True)
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
```

## Define Multiple Metrics

A multi-metric search receives a dictionary of scorers. The `refit` parameter decides which metric is used to choose `best_params_` and refit `best_estimator_`.

```python
scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": make_scorer(balanced_accuracy_score),
    "f1_macro": make_scorer(f1_score, average="macro"),
}
```

Setting `refit="balanced_accuracy"` selects the final model by class-balanced behavior.

## Configure GASearchCV

```python
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(solver="saga", max_iter=1200, random_state=RANDOM_STATE)),
])

param_grid = {
    "logistic__C": Continuous(1e-3, 30.0, distribution="log-uniform"),
    "logistic__l1_ratio": Continuous(0.0, 1.0),
    "logistic__class_weight": Categorical([None, "balanced"]),
    "logistic__max_iter": Integer(1000, 1500),
}

search = GASearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring=scoring,
    refit="balanced_accuracy",   # drives best_params_ and best_estimator_
    cv=cv,
    evolution_config=EvolutionConfig(
        population_size=12,
        generations=10,
        crossover_probability=ExponentialAdapter(initial_value=0.8, end_value=0.4, adaptive_rate=0.15),
        mutation_probability=InverseAdapter(initial_value=0.25, end_value=0.08, adaptive_rate=0.25),
        tournament_size=3,
        elitism=True,
        keep_top_k=3,
    ),
    population_config=PopulationConfig(
        initializer="smart",
        warm_start_configs=[{
            "logistic__C": 1.0,
            "logistic__l1_ratio": 0.0,
            "logistic__class_weight": None,
            "logistic__max_iter": 1200,
        }],
    ),
    runtime_config=RuntimeConfig(n_jobs=-1, parallel_backend="auto", use_cache=True, verbose=True),
    optimization_config=OptimizationConfig(
        local_search=True,
        local_search_top_k=2,
        local_search_steps=1,
        local_search_radius=0.20,
        diversity_control=True,
        diversity_threshold=0.30,
        diversity_stagnation_generations=3,
        diversity_mutation_boost=1.8,
        random_immigrants_fraction=0.10,
        fitness_sharing=True,
        sharing_radius=0.40,
    ),
)

callbacks = [
    DeltaThreshold(threshold=0.001, generations=5, metric="fitness_best"),
    ConsecutiveStopping(generations=7, metric="fitness_best"),
    TimerStopping(total_seconds=90),
]

search.fit(X_train, y_train, callbacks=callbacks)
```

## Best Parameters and Test Metrics

Because `refit="balanced_accuracy"`, `best_params_` and `best_estimator_` are selected by the CV rank of that metric.

```python
print("Refit metric:", search.refit_metric)
print("Best balanced-accuracy CV score:", round(search.best_score_, 4))
pprint(search.best_params_)

predictions = search.predict(X_test)
test_metrics = {
    "accuracy": accuracy_score(y_test, predictions),
    "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
    "f1_macro": f1_score(y_test, predictions, average="macro"),
}
print(test_metrics)
```

## Explore Multi-Metric cv_results_

For multi-metric searches, `cv_results_` contains one set of columns per metric.

```python
results = pd.DataFrame(search.cv_results_)
metric_columns = [
    "mean_test_accuracy", "rank_test_accuracy",
    "mean_test_balanced_accuracy", "rank_test_balanced_accuracy",
    "mean_test_f1_macro", "rank_test_f1_macro",
]
param_columns = [c for c in results.columns if c.startswith("param_")]

results[metric_columns + param_columns].sort_values("rank_test_balanced_accuracy").head()
```

## Comparing Metric Rankings

The same `cv_results_` can point to different candidate rankings. This example shows the best row for each metric without rerunning the search.

```python
best_rows = []
for metric_name in ["accuracy", "balanced_accuracy", "f1_macro"]:
    row = results.sort_values(f"rank_test_{metric_name}").iloc[0]
    best_rows.append({
        "metric": metric_name,
        "mean_test_score": row[f"mean_test_{metric_name}"],
        "C": row["param_logistic__C"],
        "l1_ratio": row["param_logistic__l1_ratio"],
        "class_weight": row["param_logistic__class_weight"],
    })

pd.DataFrame(best_rows)
```

## Optimizer Telemetry

With multi-metric scoring, the GA optimizes a single scalar fitness — the selected `refit` metric. Telemetry explains how the optimizer moved through the space while optimizing that metric.

```python
print(search.fit_stats_)

history = pd.DataFrame(search.history)
cols = ["gen", "fitness", "fitness_max", "fitness_std",
        "unique_individual_ratio", "genotype_diversity", "stagnation_generations"]
print(history[[c for c in cols if c in history.columns]].tail())
```

## Practical Notes

- Set `refit` to the metric that should define the final model before fitting.
- `best_score_`, `best_params_`, and `best_estimator_` follow the `refit` metric, not every metric at once.
- Use `cv_results_` to inspect tradeoffs between metrics after fitting.
- Use `fit_stats_` and `history` to understand optimizer cost, diversity, stagnation, and convergence.

## See Also

- [Multi-Metric Optimization Guide](../guide/multi-metric) — full guide with scoring dict details
- [GASearchCV API](../api/gasearchcv) — `scoring` and `refit` parameter reference
- [Understanding Cross-Validation](../guide/understand-cv) — reading the generation log
