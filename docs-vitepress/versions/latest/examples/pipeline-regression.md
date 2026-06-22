---
title: Pipeline Regression Tuning
description: Tune a scikit-learn Pipeline containing StandardScaler and GradientBoostingRegressor using GASearchCV — pipeline parameter naming, regression scorers, and search visualization.
---
:::warning Development version
This is the **latest (dev)** documentation. It may contain unreleased features or breaking changes. For the stable release, use [version 0.13](/versions/0.13/).
:::


# Pipeline Regression Tuning

This example shows how to tune a scikit-learn `Pipeline` with `GASearchCV`. Pipeline parameters use the standard sklearn double-underscore syntax: `regressor__max_depth`.

## Setup

```python
import warnings
from pprint import pprint

import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn_genetic import (
    EvolutionConfig, GASearchCV, OptimizationConfig, PopulationConfig, RuntimeConfig,
)
from sklearn_genetic.callbacks import ConsecutiveStopping, DeltaThreshold, TimerStopping
from sklearn_genetic.plots import plot_fitness_evolution, plot_search_space
from sklearn_genetic.schedules import ExponentialAdapter, InverseAdapter
from sklearn_genetic.space import Categorical, Continuous, Integer

warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42

data = load_diabetes(as_frame=True)
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE
)
cv = KFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)

print(f"Training shape: {X_train.shape}")  # (309, 10)
print(f"Test shape: {X_test.shape}")       # (133, 10)
```

## Baseline Pipeline

```python
def make_pipeline(**regressor_kwargs):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", GradientBoostingRegressor(random_state=RANDOM_STATE, **regressor_kwargs)),
    ])


def regression_metrics(estimator, X_eval, y_eval):
    predictions = estimator.predict(X_eval)
    rmse = mean_squared_error(y_eval, predictions) ** 0.5
    return {
        "r2": r2_score(y_eval, predictions),
        "rmse": rmse,
        "mae": mean_absolute_error(y_eval, predictions),
    }


baseline = make_pipeline()
baseline.fit(X_train, y_train)
baseline_metrics = regression_metrics(baseline, X_test, y_test)
print(baseline_metrics)
# {'r2': 0.430, 'rmse': 55.46, 'mae': 44.72}
```

## Pipeline Search Space

Parameter names use the sklearn `step__param` convention.

```python
param_grid = {
    "regressor__n_estimators": Integer(40, 180),
    "regressor__learning_rate": Continuous(0.01, 0.20, distribution="log-uniform"),
    "regressor__max_depth": Integer(1, 4),
    "regressor__min_samples_leaf": Integer(1, 12),
    "regressor__subsample": Continuous(0.65, 1.0),
    "regressor__loss": Categorical(["squared_error", "absolute_error", "huber"]),
}
```

:::tip Regression scorers
For metrics where smaller is better, use sklearn's negative scorer: `"neg_root_mean_squared_error"`. The GA maximizes the fitness value, so negative RMSE increases as RMSE decreases.
:::

## Configure GASearchCV

```python
search = GASearchCV(
    estimator=make_pipeline(),
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    criteria="max",
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
            "regressor__n_estimators": 100,
            "regressor__learning_rate": 0.05,
            "regressor__max_depth": 2,
            "regressor__min_samples_leaf": 4,
            "regressor__subsample": 0.85,
            "regressor__loss": "squared_error",
        }],
    ),
    runtime_config=RuntimeConfig(
        n_jobs=-1,
        parallel_backend="auto",
        use_cache=True,
        verbose=True,
        return_train_score=False,
    ),
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
    DeltaThreshold(threshold=0.01, generations=5, metric="fitness_best"),
    ConsecutiveStopping(generations=7, metric="fitness_best"),
    TimerStopping(total_seconds=120),
]

search.fit(X_train, y_train, callbacks=callbacks)
```

## Evaluate Predictions

`GASearchCV` refits the best pipeline automatically, so you can call `predict` directly on the search object.

```python
print("Best CV negative RMSE:", round(search.best_score_, 4))
pprint(search.best_params_)

ga_metrics = regression_metrics(search, X_test, y_test)
pd.DataFrame([baseline_metrics, ga_metrics], index=["baseline", "ga_pipeline"])
```

## Search Cost and Telemetry

```python
print(search.fit_stats_)
# {'evaluated_candidates': 134, 'cache_hits': 1, 'random_immigrants': 3,
#  'local_refinement_candidates': 2, ...}

history = pd.DataFrame(search.history)
cols = ["gen", "fitness", "fitness_max", "unique_individual_ratio",
        "genotype_diversity", "stagnation_generations"]
print(history[[c for c in cols if c in history.columns]].tail())
```

## Visualize the Search

```python
import matplotlib.pyplot as plt

plot_fitness_evolution(search)
plt.show()

# Show how two parameters were sampled
plot_search_space(search, features=["regressor__learning_rate", "regressor__max_depth"])
plt.show()
```

## Practical Notes

- Use pipeline parameter names exactly as sklearn expects them (`step__param`).
- For regression losses where larger is better only after negation, use sklearn's negative scorers such as `"neg_root_mean_squared_error"`.
- Compare holdout metrics, not only CV fitness.
- If the search revisits many candidates, inspect `cache_hits` in `fit_stats_` and consider stronger diversity controls or a larger search space.

## See Also

- [Pipeline Tuning Guide](../guide/pipeline-tuning) — pipeline parameter naming and step configuration
- [Search Space API](../api/space) — `Continuous`, `Integer`, `Categorical` reference
- [Plots API](../api/plots) — `plot_fitness_evolution` and `plot_search_space` reference
