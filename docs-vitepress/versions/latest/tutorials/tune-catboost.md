---
title: Tuning CatBoost With GASearchCV
description: Optimize 7 CatBoost hyperparameters — including bagging_temperature and border_count, which have no XGBoost/LightGBM equivalent — using a genetic algorithm on breast cancer data.
---

# Tuning CatBoost With GASearchCV

CatBoost's ordered boosting algorithm introduces several parameters that have no direct equivalent in XGBoost or LightGBM: `bagging_temperature` controls data perturbation via Bayesian bootstrap weights, `border_count` sets the granularity of numeric feature binning, and `colsample_bylevel` samples features per tree level rather than per tree. These parameters interact with regularization in ways that make joint optimization — what GA does — more effective than independent search.

## Prerequisites

```bash
pip install sklearn-genetic-opt catboost
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
from catboost import CatBoostClassifier

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


baseline = CatBoostClassifier(verbose=0, random_state=RANDOM_STATE)
baseline.fit(X_train, y_train)
baseline_metrics = evaluate("CatBoost defaults", baseline, X_test, y_test)
print(baseline_metrics)
# {'name': 'CatBoost defaults', 'accuracy': 0.9708, 'balanced_accuracy': 0.9680, 'roc_auc': 0.9951}
```

:::warning Always pass `verbose=0` to CatBoostClassifier
Without it, CatBoost prints a detailed training table for every iteration of every cross-validation fold. A 3-fold search with 100 iterations per candidate produces thousands of output lines. Unlike LightGBM's `verbose=-1`, CatBoost uses `verbose=0` (integer, not negative).
:::

## Search Space

CatBoost's parameter names and their valid ranges differ from XGBoost/LightGBM in important ways.

```python
param_grid = {
    # Boosting volume
    "iterations":           Integer(50, 500),         # number of trees (= n_estimators elsewhere)

    # Step size — CatBoost default is 0.03, much lower than XGBoost/LightGBM
    "learning_rate":        Continuous(0.01, 0.3, distribution="log-uniform"),

    # Tree complexity
    "depth":                Integer(4, 10),            # max tree depth (CatBoost default: 6)

    # Regularization
    "l2_leaf_reg":          Continuous(1e-3, 10.0, distribution="log-uniform"),  # L2 reg on leaf weights

    # Sampling — per level, not per tree
    "colsample_bylevel":    Continuous(0.5, 1.0),      # feature fraction per tree level

    # Bayesian bootstrap (incompatible with subsample — see note below)
    "bagging_temperature":  Continuous(0.0, 1.0),      # 0 = uniform weights, 1 = exponential

    # Numeric feature granularity
    "border_count":         Integer(32, 255),           # bins per numeric feature
}
```

:::info CatBoost-specific parameters explained

**`bagging_temperature`** — CatBoost's default bootstrap on CPU is Bayesian, not subsampling. Each training sample gets a weight drawn from an exponential distribution scaled by `bagging_temperature`. At `0.0` all weights are equal (no perturbation); at `1.0` the distribution is exponential and provides strong regularization through data perturbation. This replaces `subsample` — do not include both in the same search.

**`border_count`** — the number of splits evaluated for each numeric feature (analogous to `max_bin` in XGBoost). Higher values give finer-grained splits and better accuracy but slower training. The default is 254; searching from 32 to 255 covers fast-but-coarse through the full default granularity.

**`colsample_bylevel`** — unlike XGBoost/LightGBM where feature subsampling is per tree, CatBoost subsamples features at each tree level. Combined with `bagging_temperature` this provides two complementary regularization axes.
:::

## Configure GASearchCV

CatBoost is slower per iteration than LightGBM, so we use fewer generations and rely on `TimerStopping` as the primary budget control.

```python
callbacks = [
    ConsecutiveStopping(generations=8, metric="fitness_best"),
    TimerStopping(total_seconds=300),
]

ga_search = GASearchCV(
    estimator=CatBoostClassifier(verbose=0, random_state=RANDOM_STATE),
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    evolution_config=EvolutionConfig(
        population_size=20,
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
            "iterations":          100,
            "learning_rate":       0.03,    # CatBoost default — note: lower than XGBoost!
            "depth":               6,
            "l2_leaf_reg":         3.0,
            "colsample_bylevel":   1.0,
            "bagging_temperature": 1.0,
            "border_count":        254,
        }],
    ),
    runtime_config=RuntimeConfig(
        n_jobs=-1,
        parallel_backend="cv",    # CatBoost manages its own thread pool
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

:::tip GPU acceleration
If a CUDA-capable GPU is available, CatBoost can be orders of magnitude faster:
```python
CatBoostClassifier(verbose=0, task_type="GPU", random_state=RANDOM_STATE)
```
GPU training uses `SymmetricTree` structure, which may affect optimal `depth` and `border_count` values. Consider expanding the search space if switching to GPU.
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
#   'evaluated_candidates': 240,
#   'unique_candidates':    238,
#   'cache_hits':           2,
#   'random_immigrants':    18,
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
ax.set_title("CatBoost GA Search — Fitness over Generations")
ax.set_xlabel("Generation")
ax.set_ylabel("ROC AUC (CV)")
ax.legend(["best so far", "generation max", "generation mean"])
plt.tight_layout()
plt.show()
```

## `border_count` vs `bagging_temperature` Interaction

`border_count` controls split precision; `bagging_temperature` controls regularization strength through data perturbation. High `border_count` with low `bagging_temperature` can lead to overfitting; the GA discovers the balance.

```python
cv_results = pd.DataFrame(ga_search.cv_results_)
cv_results = cv_results.dropna(subset=["mean_test_score"])

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(
    cv_results["param_border_count"].astype(float),
    cv_results["param_bagging_temperature"].astype(float),
    c=cv_results["mean_test_score"],
    cmap="RdYlGn",
    s=60,
    alpha=0.8,
    edgecolors="none",
)
plt.colorbar(scatter, ax=ax, label="Mean CV ROC AUC")
ax.set_xlabel("border_count")
ax.set_ylabel("bagging_temperature")
ax.set_title("Evaluated Candidates — border_count vs bagging_temperature")
plt.tight_layout()
plt.show()
```

## Compare with RandomizedSearchCV

```python
randomized_search = RandomizedSearchCV(
    estimator=CatBoostClassifier(verbose=0, random_state=RANDOM_STATE),
    param_distributions={
        "iterations":          randint(50, 501),
        "learning_rate":       loguniform(0.01, 0.3),
        "depth":               randint(4, 11),
        "l2_leaf_reg":         loguniform(1e-3, 10.0),
        "colsample_bylevel":   uniform(0.5, 0.5),
        "bagging_temperature": uniform(0.0, 1.0),
        "border_count":        randint(32, 256),
    },
    n_iter=20,
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
  CatBoost defaults    0.9708             0.9680   0.9951           None         None
  RandomizedSearchCV   0.9766             0.9742   0.9958         0.9936         38.5
       GASearchCV      0.9825             0.9810   0.9972         0.9962         72.4
```

## Practical Notes

- **`verbose=0` is not optional** — it is an integer, not a boolean. `verbose=False` also works.
- **CatBoost default `learning_rate` is `0.03`**, not `0.1` like XGBoost/LightGBM. Reflect this in `warm_start_configs`.
- **Do not mix `subsample` and `bagging_temperature`** — on CPU, CatBoost uses Bayesian bootstrap by default (`bootstrap_type='Bayesian'`), which uses `bagging_temperature`. Adding `subsample` to the search space while on Bayesian bootstrap will raise a parameter conflict error. If you want to use `subsample`, also search `bootstrap_type=Categorical(['Bernoulli', 'MVS'])`.
- **`border_count` impacts speed**: 255 bins is up to 8× slower than 32 on wide datasets. If search time is a bottleneck, fix `border_count=64` and remove it from the search space.
- **`parallel_backend="cv"`** — CatBoost, like XGBoost and LightGBM, manages its own thread pool. CV-level parallelism avoids oversubscription.

## See Also

- [Tune XGBoost](./tune-xgboost) — depth-wise boosting
- [Tune LightGBM](./tune-lightgbm) — leaf-wise boosting with `num_leaves` interaction
- [Advanced Optimizer Control](../guide/advanced-optimizer-control) — diversity, fitness sharing, local search
- [GASearchCV API](../api/gasearchcv)
