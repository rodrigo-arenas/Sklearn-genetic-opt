---
title: Tuning XGBoost With GASearchCV
description: Optimize 9 XGBoost hyperparameters — learning rate, depth, subsampling, and regularization — using a genetic algorithm on breast cancer data. Includes 3-way comparison and feature importance.
---

# Tuning XGBoost With GASearchCV

XGBoost has at least 9 hyperparameters that interact in non-linear ways. `learning_rate` and `n_estimators` must be balanced together — more trees with a lower rate often outperforms fewer trees with a high rate, but the optimal combination depends on `max_depth`, `subsample`, and the regularization terms simultaneously. Random search treats each parameter independently and misses these interactions. A genetic algorithm explores the joint space and captures them.

This tutorial tunes all 9 parameters, compares against a random-search baseline, and shows feature importance after tuning.

## Prerequisites

```bash
pip install sklearn-genetic-opt xgboost
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from sklearn_genetic import (
    EvolutionConfig, GASearchCV, OptimizationConfig, PopulationConfig, RuntimeConfig,
)
from sklearn_genetic.callbacks import ConsecutiveStopping, TimerStopping
from sklearn_genetic.schedules import ExponentialAdapter, InverseAdapter
from sklearn_genetic.space import Categorical, Continuous, Integer

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Class balance — Train: {y_train.mean():.2f}, Test: {y_test.mean():.2f}")
```

## Baseline Model

Establish a reference point with XGBoost defaults before searching.

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


baseline = XGBClassifier(
    tree_method="hist",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=1,
)
baseline.fit(X_train, y_train)
baseline_metrics = evaluate("XGBoost defaults", baseline, X_test, y_test)
print(baseline_metrics)
# {'name': 'XGBoost defaults', 'accuracy': 0.9649, 'balanced_accuracy': 0.9613, 'roc_auc': 0.9934}
```

## Search Space

Nine parameters with ranges grounded in XGBoost's own documentation and common community practice. Every `log-uniform` param spans multiple orders of magnitude — log-uniform sampling gives equal probability to each order of magnitude rather than biasing toward large values.

```python
param_grid = {
    # Tree structure
    "n_estimators":     Integer(50, 500),           # number of boosting rounds
    "max_depth":        Integer(3, 10),              # max tree depth; deeper = more complex
    "min_child_weight": Integer(1, 10),              # min sum of instance weight in a leaf

    # Subsampling — prevents overfitting by using a fraction of data/features
    "subsample":        Continuous(0.6, 1.0),        # row subsampling per tree
    "colsample_bytree": Continuous(0.5, 1.0),        # column subsampling per tree

    # Step size
    "learning_rate":    Continuous(0.01, 0.3, distribution="log-uniform"),

    # Regularization
    "gamma":            Continuous(0.0, 0.5),        # min loss reduction to make a split
    "reg_alpha":        Continuous(1e-5, 10.0, distribution="log-uniform"),  # L1
    "reg_lambda":       Continuous(1e-5, 10.0, distribution="log-uniform"),  # L2
}
```

:::tip Why log-uniform for regularization?
`reg_alpha` and `reg_lambda` are useful across many orders of magnitude — `1e-5`, `0.01`, `1.0`, `10.0` are all valid. A uniform distribution would spend 99.9% of samples above `0.01`, effectively never exploring the low end. `log-uniform` gives equal probability to each decade.
:::

## Configure GASearchCV

Key decisions in this configuration:

- **`parallel_backend="cv"`** — XGBoost uses its own internal thread pool (`n_jobs=1` in the estimator). With `parallel_backend="cv"`, sklearn-genetic-opt parallelises at the CV-fold level instead of the candidate level, avoiding CPU oversubscription.
- **`warm_start_configs`** — seeds the first population with XGBoost's documented defaults so the search starts from a known good region.
- **`log-uniform` lower bounds in warm_start** — use `1e-5` (not `0.0`) to stay within the sampling distribution.

```python
callbacks = [
    ConsecutiveStopping(generations=10, metric="fitness_best"),
    TimerStopping(total_seconds=300),
]

ga_search = GASearchCV(
    estimator=XGBClassifier(
        tree_method="hist",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
    ),
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    evolution_config=EvolutionConfig(
        population_size=25,
        generations=20,
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
            "n_estimators":     100,
            "max_depth":        6,
            "min_child_weight": 1,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "learning_rate":    0.1,
            "gamma":            0.0,
            "reg_alpha":        1e-5,
            "reg_lambda":       1.0,
        }],
    ),
    runtime_config=RuntimeConfig(
        n_jobs=-1,
        parallel_backend="cv",
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

:::warning CPU oversubscription with XGBoost
XGBoost spawns threads internally for tree building. If `n_jobs=-1` in the estimator AND `parallel_backend="population"` in RuntimeConfig, you get `n_workers × n_xgb_threads` total threads — often 4–8× your CPU count. Use `n_jobs=1` in the XGBClassifier and `parallel_backend="cv"` to let sklearn-genetic-opt handle parallelism at the fold level instead.
:::

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
#   'evaluated_candidates': 312,
#   'unique_candidates':    308,
#   'cache_hits':           4,
#   'random_immigrants':    24,
#   'local_refinement_candidates': 4,
# }
```

### Generation-by-Generation Telemetry

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
ax.set_title("XGBoost GA Search — Fitness over Generations")
ax.set_xlabel("Generation")
ax.set_ylabel("ROC AUC (CV)")
ax.legend(["best so far", "generation max", "generation mean"])
plt.tight_layout()
plt.show()
```

## Feature Importance

After tuning, the best estimator is a fitted XGBoost model. Plot its feature importances to understand which measurements drive predictions.

```python
importances = ga_search.best_estimator_.feature_importances_
feat_df = pd.Series(importances, index=data.feature_names).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 10))
feat_df.tail(20).plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Top-20 Feature Importances (Gain) — Tuned XGBoost")
ax.set_xlabel("Importance (gain)")
plt.tight_layout()
plt.show()
```

## Compare with RandomizedSearchCV

Use the same evaluation budget (roughly equal number of CV evaluations) to make the comparison fair.

```python
randomized_search = RandomizedSearchCV(
    estimator=XGBClassifier(
        tree_method="hist",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
    ),
    param_distributions={
        "n_estimators":     randint(50, 501),
        "max_depth":        randint(3, 11),
        "min_child_weight": randint(1, 11),
        "subsample":        uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.5, 0.5),
        "learning_rate":    loguniform(0.01, 0.3),
        "gamma":            uniform(0.0, 0.5),
        "reg_alpha":        loguniform(1e-5, 10.0),
        "reg_lambda":       loguniform(1e-5, 10.0),
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
  XGBoost defaults    0.9649             0.9613   0.9934           None         None
  RandomizedSearchCV  0.9708             0.9680   0.9947         0.9921         18.3
       GASearchCV     0.9825             0.9810   0.9975         0.9958         52.1
```

## Practical Notes

- **`tree_method='hist'`** cuts per-tree build time significantly. Always use it unless you have a reason not to.
- **`eval_metric='logloss'`** suppresses XGBoost's default metric warning when `use_label_encoder` is absent.
- **`parallel_backend="cv"`** with `n_jobs=1` in the estimator is the correct pairing for any estimator that manages its own threads.
- **`reg_alpha` and `reg_lambda` lower bounds** in `warm_start_configs` must be `≥ 1e-5` (the log-uniform lower bound), not `0.0`.
- GA particularly pays off here because `learning_rate × n_estimators × max_depth` form a three-way interaction that random search cannot efficiently navigate.
- Check `fit_stats_["cache_hits"]` — a non-zero value means the cache is working and duplicate candidates from convergence are being recycled.

## See Also

- [Tune LightGBM](./tune-lightgbm) — similar workflow for LightGBM's leaf-wise trees
- [Tune CatBoost](./tune-catboost) — CatBoost-specific parameters
- [Advanced Optimizer Control](../guide/advanced-optimizer-control) — diversity, fitness sharing, local search in detail
- [Adaptive Schedules](../guide/adapters) — `ExponentialAdapter` and `InverseAdapter` explained
- [GASearchCV API](../api/gasearchcv)
