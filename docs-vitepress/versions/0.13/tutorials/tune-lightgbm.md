---
title: Tuning LightGBM With GASearchCV
description: Optimize 9 LightGBM hyperparameters including num_leaves, learning rate, and subsampling using a genetic algorithm. Includes parameter interaction scatter plots and 3-way comparison.
---

# Tuning LightGBM With GASearchCV

LightGBM grows trees leaf-wise rather than depth-wise. This means `num_leaves` is the primary complexity control, and it interacts with `max_depth` in a non-obvious way: if `num_leaves > 2^max_depth`, LightGBM silently clips the value. Random search samples `num_leaves` and `max_depth` independently and wastes much of its budget on invalid combinations. A genetic algorithm discovers the valid region quickly and concentrates effort there.

LightGBM is also faster per tree than XGBoost, so we can afford more generations for the same wall-clock time.

## Prerequisites

```bash
pip install sklearn-genetic-opt lightgbm
```

## Setup

```python
import warnings
from pprint import pprint
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import loguniform, randint, uniform
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from lightgbm import LGBMClassifier

from sklearn_genetic import (
    EvolutionConfig, GASearchCV, OptimizationConfig, PopulationConfig, RuntimeConfig,
)
from sklearn_genetic.callbacks import ConsecutiveStopping, TimerStopping
from sklearn_genetic.schedules import ExponentialAdapter, InverseAdapter
from sklearn_genetic.space import Continuous, Integer

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
```

## Baseline Model

```python
def evaluate(name, estimator, X_eval, y_eval):
    predictions = estimator.predict(X_eval)
    probabilities = estimator.predict_proba(X_eval)[:, 1]
    return {
        "name": name,
        "accuracy": round(accuracy_score(y_eval, predictions), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_eval, predictions), 4),
        "roc_auc": round(roc_auc_score(y_eval, probabilities), 4),
    }


baseline = LGBMClassifier(verbose=-1, random_state=RANDOM_STATE)
baseline.fit(X_train, y_train)
baseline_metrics = evaluate("LightGBM defaults", baseline, X_test, y_test)
print(baseline_metrics)
# {'name': 'LightGBM defaults', 'accuracy': 0.9649, 'balanced_accuracy': 0.9623, 'roc_auc': 0.9947}
```

:::tip Always pass `verbose=-1` to LGBMClassifier
Without it, LightGBM prints a training log for every tree in every cross-validation fold. A 3-fold search across 20 generations of 20 candidates with 100 trees each produces 12,000 log lines. Add `verbose=-1` to the estimator constructor and it is silenced.
:::

## Search Space

```python
param_grid = {
    # Volume of boosting
    "n_estimators":      Integer(50, 500),

    # Tree complexity — these two interact: num_leaves > 2^max_depth is invalid
    "num_leaves":        Integer(20, 150),           # primary complexity control (leaf-wise growth)
    "max_depth":         Integer(3, 12),             # upper bound on tree depth (-1 = unlimited)

    # Step size
    "learning_rate":     Continuous(0.005, 0.3, distribution="log-uniform"),

    # Subsampling
    "subsample":         Continuous(0.6, 1.0),       # row fraction per tree (requires subsample_freq > 0)
    "colsample_bytree":  Continuous(0.5, 1.0),       # feature fraction per tree

    # Leaf regularization
    "min_child_samples": Integer(5, 50),             # min samples required in a leaf

    # L1 / L2 regularization
    "reg_alpha":         Continuous(1e-5, 10.0, distribution="log-uniform"),
    "reg_lambda":        Continuous(1e-5, 10.0, distribution="log-uniform"),
}
```

:::info `num_leaves` and `max_depth` interaction
LightGBM's leaf-wise algorithm can grow trees with up to `num_leaves` leaves. If `max_depth` is also set, the effective leaf count is `min(num_leaves, 2^max_depth)`. The search space covers both — the GA will naturally explore the boundary and learn that the useful region is where `num_leaves ≤ 2^max_depth`. Random search has no mechanism to discover this constraint.
:::

## Configure GASearchCV

```python
callbacks = [
    ConsecutiveStopping(generations=12, metric="fitness_best"),
    TimerStopping(total_seconds=360),
]

ga_search = GASearchCV(
    estimator=LGBMClassifier(verbose=-1, random_state=RANDOM_STATE, n_jobs=1),
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    evolution_config=EvolutionConfig(
        population_size=20,
        generations=30,       # LightGBM is faster per eval — afford more generations
        crossover_probability=ExponentialAdapter(
            initial_value=0.8, end_value=0.4, adaptive_rate=0.15
        ),
        mutation_probability=InverseAdapter(
            initial_value=0.25, end_value=0.05, adaptive_rate=0.20
        ),
        tournament_size=3,
        elitism=True,
        keep_top_k=3,
    ),
    population_config=PopulationConfig(
        initializer="smart",
        warm_start_configs=[{
            "n_estimators":      100,
            "num_leaves":        31,       # LightGBM default
            "max_depth":         6,        # use a bounded value (default is -1/unlimited)
            "learning_rate":     0.1,
            "subsample":         0.8,
            "colsample_bytree":  0.8,
            "min_child_samples": 20,
            "reg_alpha":         1e-5,
            "reg_lambda":        1e-5,
        }],
    ),
    runtime_config=RuntimeConfig(
        n_jobs=-1,
        parallel_backend="cv",    # LightGBM uses OpenMP threads internally
        use_cache=True,
        verbose=True,
    ),
    optimization_config=OptimizationConfig(
        local_search=True,
        local_search_top_k=2,
        local_search_steps=1,
        local_search_radius=0.2,
        diversity_control=True,
        diversity_threshold=0.30,
        diversity_stagnation_generations=3,
        diversity_mutation_boost=1.8,
        random_immigrants_fraction=0.10,
        fitness_sharing=True,
        sharing_radius=0.35,
    ),
)
```

## Fit and Results

```python
started_at = time.perf_counter()
ga_search.fit(X_train, y_train, callbacks=callbacks)
ga_seconds = time.perf_counter() - started_at

print(f"\nBest CV ROC AUC: {ga_search.best_score_:.4f}")
print(f"Search time:     {ga_seconds:.0f}s")
pprint(ga_search.best_params_)
```

### Evaluation Mechanics

```python
print(ga_search.fit_stats_)
# {
#   'evaluated_candidates': 380,
#   'unique_candidates':    375,
#   'cache_hits':           5,
#   'random_immigrants':    28,
# }
```

### Generation Telemetry

```python
history = pd.DataFrame(ga_search.history)
cols = ["gen", "fitness", "fitness_max", "fitness_std",
        "unique_individual_ratio", "genotype_diversity", "stagnation_generations"]
print(history[[c for c in cols if c in history.columns]].to_string())
```

## Fitness Evolution

```python
ax = history.plot(
    x="gen",
    y=["fitness_best", "fitness_max", "fitness"],
    marker="o",
    figsize=(9, 4),
)
ax.set_title("LightGBM GA Search — Fitness over Generations")
ax.set_xlabel("Generation")
ax.set_ylabel("ROC AUC (CV)")
ax.legend(["best so far", "generation max", "generation mean"])
plt.tight_layout()
plt.show()
```

## `num_leaves` vs `max_depth` Interaction

This scatter plot shows every evaluated candidate colored by its CV score. The constraint `num_leaves ≤ 2^max_depth` appears as an upper boundary — candidates above it have lower scores because LightGBM clips `num_leaves` internally.

```python
cv_results = pd.DataFrame(ga_search.cv_results_)
cv_results = cv_results.dropna(subset=["mean_test_score"])

# Compute the 2^max_depth boundary
depth_range = np.arange(3, 13)
boundary_leaves = 2 ** depth_range

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(
    cv_results["param_max_depth"],
    cv_results["param_num_leaves"],
    c=cv_results["mean_test_score"],
    cmap="RdYlGn",
    s=50,
    alpha=0.8,
    edgecolors="none",
)
ax.plot(depth_range, boundary_leaves, "k--", linewidth=1.5, label="2^max_depth boundary")
plt.colorbar(scatter, ax=ax, label="Mean CV ROC AUC")
ax.set_xlabel("max_depth")
ax.set_ylabel("num_leaves")
ax.set_title("Evaluated Candidates — num_leaves vs max_depth")
ax.legend()
plt.tight_layout()
plt.show()
```

High-scoring candidates (green) cluster below the dashed boundary, confirming that the GA learns the constraint region faster than random search.

## Compare with RandomizedSearchCV

```python
randomized_search = RandomizedSearchCV(
    estimator=LGBMClassifier(verbose=-1, random_state=RANDOM_STATE, n_jobs=1),
    param_distributions={
        "n_estimators":      randint(50, 501),
        "num_leaves":        randint(20, 151),
        "max_depth":         randint(3, 13),
        "learning_rate":     loguniform(0.005, 0.3),
        "subsample":         uniform(0.6, 0.4),
        "colsample_bytree":  uniform(0.5, 0.5),
        "min_child_samples": randint(5, 51),
        "reg_alpha":         loguniform(1e-5, 10.0),
        "reg_lambda":        loguniform(1e-5, 10.0),
    },
    n_iter=25,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

started_at = time.perf_counter()
randomized_search.fit(X_train, y_train)
rs_seconds = time.perf_counter() - started_at

rs_metrics = evaluate("RandomizedSearchCV", randomized_search, X_test, y_test)
ga_metrics = evaluate("GASearchCV", ga_search, X_test, y_test)

comparison = pd.DataFrame([baseline_metrics, rs_metrics, ga_metrics])
comparison["best_cv_score"] = [
    None,
    round(randomized_search.best_score_, 4),
    round(ga_search.best_score_, 4),
]
comparison["fit_seconds"] = [None, round(rs_seconds, 1), round(ga_seconds, 1)]
print(comparison.to_string(index=False))
```

Expected output (approximate):

```
                name  accuracy  balanced_accuracy  roc_auc  best_cv_score  fit_seconds
  LightGBM defaults    0.9649             0.9623   0.9947           None         None
  RandomizedSearchCV   0.9708             0.9665   0.9953         0.9928         12.4
       GASearchCV      0.9766             0.9742   0.9970         0.9961         41.8
```

## Practical Notes

- **`verbose=-1`** is not optional — omitting it generates thousands of log lines during search.
- **`parallel_backend="cv"`** prevents OpenMP thread explosion. LightGBM by default uses all available cores; combining with candidate-level parallelism saturates the CPU.
- **`num_leaves` constraint** — the GA learns it; random search wastes budget on the infeasible region. The scatter plot makes this visible.
- LightGBM's default `learning_rate` is `0.1`, same as XGBoost, but useful ranges extend lower (`0.005` is often competitive with more trees).
- `min_child_samples` (LightGBM) is analogous to `min_child_weight` in XGBoost but counts samples directly, not their weighted sum. The range `[5, 50]` is appropriate for datasets with hundreds to thousands of rows per leaf.

## See Also

- [Tune XGBoost](./tune-xgboost) — depth-wise boosting with similar workflow
- [Tune CatBoost](./tune-catboost) — CatBoost-specific parameters
- [Adaptive Schedules](../guide/adapters) — how `ExponentialAdapter` and `InverseAdapter` work
- [GASearchCV API](../api/gasearchcv)
