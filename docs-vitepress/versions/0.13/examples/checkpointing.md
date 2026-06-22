---
title: Checkpointing and Persistence
description: Write intermediate checkpoints, save and reload a fitted GASearchCV search object, and inspect checkpoint contents.
---
# Checkpointing and Persistence

Long-running searches should be able to write intermediate checkpoints, save the fitted search object, and reload it later for inspection or prediction.

## The Two Persistence Mechanisms

| Mechanism | When to use |
|-----------|-------------|
| `ModelCheckpoint` callback | Progress recovery and audit trails **during** a fit |
| `search.save()` / `search.load()` | Reusing a fully fitted search object **after** training |

## Setup

```python
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn_genetic import (
    EvolutionConfig, GASearchCV, OptimizationConfig, PopulationConfig, RuntimeConfig,
)
from sklearn_genetic.callbacks import ConsecutiveStopping, ModelCheckpoint, TimerStopping
from sklearn_genetic.plots import plot_fitness_evolution
from sklearn_genetic.schedules import ExponentialAdapter, InverseAdapter
from sklearn_genetic.space import Categorical, Continuous, Integer

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
```

## Configure File Paths

```python
artifact_dir = Path("ga_artifacts")
artifact_dir.mkdir(exist_ok=True)

checkpoint_path = artifact_dir / "breast_cancer_ga_checkpoint.pkl"
saved_search_path = artifact_dir / "breast_cancer_ga_search.pkl"
```

## Search Configuration

```python
param_grid = {
    "n_estimators": Integer(40, 160),
    "max_depth": Integer(2, 12),
    "min_samples_leaf": Integer(1, 8),
    "max_features": Continuous(0.25, 1.0),
    "criterion": Categorical(["gini", "entropy", "log_loss"]),
    "class_weight": Categorical([None, "balanced"]),
}

callbacks = [
    ModelCheckpoint(checkpoint_path),         # writes after every generation
    ConsecutiveStopping(generations=6, metric="fitness_best"),
    TimerStopping(total_seconds=180),
]

search = GASearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=1),
    cv=cv,
    scoring="roc_auc",
    param_grid=param_grid,
    evolution_config=EvolutionConfig(
        population_size=14,
        generations=10,
        crossover_probability=ExponentialAdapter(initial_value=0.85, end_value=0.45, adaptive_rate=0.08),
        mutation_probability=InverseAdapter(initial_value=0.18, end_value=0.55, adaptive_rate=0.12),
        keep_top_k=4,
    ),
    population_config=PopulationConfig(
        initializer="smart",
        warm_start_configs=[{
            "n_estimators": 80,
            "max_depth": 5,
            "min_samples_leaf": 2,
            "max_features": 0.7,
            "criterion": "gini",
            "class_weight": None,
        }],
    ),
    runtime_config=RuntimeConfig(
        use_cache=True,
        parallel_backend="auto",
        n_jobs=-1,
        verbose=False,
    ),
    optimization_config=OptimizationConfig(
        local_search=True,
        local_search_top_k=2,
        local_search_steps=2,
        local_search_radius=0.12,
        diversity_control=True,
        diversity_threshold=0.2,
        diversity_stagnation_generations=3,
        diversity_mutation_boost=1.8,
        random_immigrants_fraction=0.15,
        fitness_sharing=True,
        sharing_radius=0.25,
    ),
    refit=True,
)
```

## Fit With Checkpointing

```python
search.fit(X_train, y_train, callbacks=callbacks)
# Checkpoint save in ga_artifacts/breast_cancer_ga_checkpoint.pkl  (printed each generation)

print(search.best_params_)
print("Best CV ROC AUC:", round(search.best_score_, 4))
```

## Evaluate on Test Set

```python
y_pred = search.predict(X_test)
y_proba = search.predict_proba(X_test)[:, 1]

pd.Series({
    "accuracy": accuracy_score(y_test, y_pred),
    "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba),
}).to_frame("test_score")
```

## Fit Statistics

```python
pd.Series(search.fit_stats_).to_frame("value")
# evaluated_candidates, unique_candidates, cache_hits, random_immigrants, ...
```

## Plot Fitness Evolution

```python
import matplotlib.pyplot as plt
plot_fitness_evolution(search)
plt.show()
```

## Inspect Checkpoint Contents

The checkpoint stores a dictionary with two keys: `estimator_state` and `logbook`.

```python
checkpoint = ModelCheckpoint(checkpoint_path).load()

print(checkpoint.keys())         # dict_keys(['estimator_state', 'logbook'])
print(len(checkpoint["logbook"])) # number of generations completed

# Optimizer config captured in the checkpoint
print(sorted(checkpoint["estimator_state"].keys()))
```

The `estimator_state` is intentionally lightweight — it captures the search configuration, not the fitted model. Use `save` / `load` for the full fitted object.

## Save and Reload the Fitted Search

```python
# Save after training is complete
search.save(saved_search_path)

# Reload into a fresh GASearchCV instance
restored_search = GASearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=1),
    cv=cv,
    scoring="roc_auc",
    param_grid=param_grid,
)
restored_search.load(saved_search_path)

# The restored object makes identical predictions
restored_predictions = restored_search.predict(X_test)
assert (restored_predictions == y_pred).all()
print("Restored best score:", round(restored_search.best_score_, 4))
```

## Practical Notes

- Use `ModelCheckpoint` for progress recovery and audit trails during a fit.
- Use `save` and `load` for fitted search objects that need to be reused for prediction or later analysis.
- Store checkpoints outside temporary notebook directories for long runs.
- Keep `random_state` fixed across the estimator, splitter, and search inputs to produce repeatable artifacts.

## See Also

- [Callbacks](../guide/callbacks) — `ModelCheckpoint` and other callbacks
- [Reproducibility](../guide/reproducibility) — fixing seeds for repeatable runs
- [Callbacks API](../api/callbacks) — `ModelCheckpoint` parameter reference
