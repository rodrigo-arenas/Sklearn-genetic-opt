---
title: GASearchCV
description: API reference for GASearchCV — genetic algorithm hyperparameter search for scikit-learn estimators.
---

:::warning Development version
You are reading the **latest (dev)** docs. For the stable version, see [0.13](/versions/0.13/).
:::

# GASearchCV

Genetic algorithm hyperparameter search for scikit-learn estimators.

`GASearchCV` implements the scikit-learn estimator interface and follows the same patterns as `GridSearchCV` and `RandomizedSearchCV`. After calling `fit`, it exposes `best_params_`, `best_score_`, `cv_results_`, `predict`, `predict_proba`, and `score`.

## Class Signature

```python
from sklearn_genetic import GASearchCV

GASearchCV(
    estimator,
    param_grid,
    *,
    scoring=None,
    cv=5,
    refit=True,
    verbose=0,
    error_score=np.nan,
    return_train_score=False,
    random_state=None,
    evolution_config=None,
    population_config=None,
    runtime_config=None,
    optimization_config=None,
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `estimator` | estimator | — | A scikit-learn estimator with a `fit` method |
| `param_grid` | dict | — | Mapping from parameter name to `Integer`, `Continuous`, or `Categorical` |
| `scoring` | str or dict | `None` | Metric(s) to evaluate. `None` uses the estimator's default scorer |
| `cv` | int or CV splitter | `5` | Cross-validation strategy |
| `refit` | bool or str | `True` | Metric to use when refitting the best model. If `scoring` is a dict, pass the metric name |
| `verbose` | int | `0` | Verbosity level. `1` = generation log |
| `error_score` | float or `"raise"` | `np.nan` | Score to use when a candidate raises an exception |
| `return_train_score` | bool | `False` | Include training scores in `cv_results_` |
| `random_state` | int, RandomState or `None` | `None` | Seeds the whole search (population init, mutation, crossover, immigrants) at `fit` time for reproducible runs. `None` is non-deterministic |
| `evolution_config` | `EvolutionConfig` | `None` | Controls population size, generations, crossover/mutation rates, elitism |
| `population_config` | `PopulationConfig` | `None` | Controls initialization strategy, warm starts, diversity |
| `runtime_config` | `RuntimeConfig` | `None` | Controls parallelism, caching, verbosity |
| `optimization_config` | `OptimizationConfig` | `None` | Controls local search, fitness sharing |

## Attributes After `fit`

| Attribute | Description |
|-----------|-------------|
| `best_params_` | Parameter setting that gave the best mean cross-validated score |
| `best_score_` | Mean cross-validated score of the best estimator |
| `best_estimator_` | Estimator fitted with `best_params_` on the full training data (if `refit=True`) |
| `cv_results_` | Dict with per-candidate results, compatible with `pd.DataFrame` |
| `history` | List of per-generation dicts with fitness and diversity telemetry |
| `logbook` | DEAP logbook — same data as `history` in DEAP's format |
| `fit_stats_` | Dict with evaluation counters (cache hits, skipped candidates, etc.) |
| `support_` | *(Not applicable for GASearchCV — see GAFeatureSelectionCV)* |

## Methods

| Method | Description |
|--------|-------------|
| `fit(X, y, callbacks=None)` | Run the genetic search |
| `predict(X)` | Predict using `best_estimator_` |
| `predict_proba(X)` | Predict class probabilities (if estimator supports it) |
| `score(X, y)` | Score using `best_estimator_` |
| `get_params()` / `set_params()` | Standard sklearn estimator interface |

## Example

```python
from sklearn_genetic import EvolutionConfig, GASearchCV, PopulationConfig, RuntimeConfig
from sklearn_genetic.space import Categorical, Continuous, Integer

search = GASearchCV(
    estimator=your_estimator,
    param_grid={
        "param1": Integer(1, 100),
        "param2": Continuous(0.01, 1.0, distribution="log-uniform"),
        "param3": Categorical(["a", "b", "c"]),
    },
    cv=5,
    scoring="roc_auc",
    evolution_config=EvolutionConfig(population_size=20, generations=15),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, use_cache=True),
)

search.fit(X_train, y_train)
print(search.best_params_)
print(search.best_score_)
```

## See Also

- [Basic Usage](../guide/basic-usage) — step-by-step tutorial
- [Config Objects](./config) — `EvolutionConfig`, `PopulationConfig`, `RuntimeConfig`, `OptimizationConfig`
- [Search Space](./space) — `Integer`, `Continuous`, `Categorical`
- [GAFeatureSelectionCV](./gafeatureselectioncv) — feature selection variant
