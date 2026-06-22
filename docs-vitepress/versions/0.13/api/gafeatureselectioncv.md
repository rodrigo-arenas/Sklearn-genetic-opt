---
title: GAFeatureSelectionCV
description: API reference for GAFeatureSelectionCV — genetic algorithm feature selection for scikit-learn estimators.
---

# GAFeatureSelectionCV

Genetic algorithm wrapper-based feature selection with cross-validation.

`GAFeatureSelectionCV` selects a subset of columns from the input data that maximises the cross-validation score of the given estimator. After fitting, it exposes `support_` (a boolean mask of selected features) and behaves as a fitted scikit-learn estimator for prediction.

## Class Signature

```python
from sklearn_genetic import GAFeatureSelectionCV

GAFeatureSelectionCV(
    estimator,
    *,
    cv=5,
    scoring=None,
    refit=True,
    verbose=0,
    keep_top_k=1,
    elitism=True,
    error_score=np.nan,
    return_train_score=False,
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
| `cv` | int or CV splitter | `5` | Cross-validation strategy |
| `scoring` | str | `None` | Metric to evaluate. `None` uses the estimator's default scorer |
| `refit` | bool | `True` | Refit the best estimator on the full training data after search |
| `verbose` | int | `0` | Verbosity level |
| `keep_top_k` | int | `1` | Number of hall-of-fame individuals to keep across generations |
| `elitism` | bool | `True` | Whether to carry over the best individual to the next generation |
| `error_score` | float or `"raise"` | `np.nan` | Score to assign when a candidate raises an exception |
| `evolution_config` | `EvolutionConfig` | `None` | Population size, generations, crossover/mutation rates |
| `population_config` | `PopulationConfig` | `None` | Initialization strategy and diversity settings |
| `runtime_config` | `RuntimeConfig` | `None` | Parallelism, caching, verbosity |
| `optimization_config` | `OptimizationConfig` | `None` | Local search, fitness sharing |

## Attributes After `fit`

| Attribute | Description |
|-----------|-------------|
| `support_` | Boolean mask — `True` for selected features |
| `best_score_` | Mean CV score of the best feature subset |
| `best_estimator_` | Estimator fitted with the selected features on the full training data |
| `cv_results_` | Dict with per-candidate results |
| `history` | List of per-generation dicts |
| `logbook` | DEAP logbook |
| `fit_stats_` | Evaluation counters |
| `n_features_` | Number of selected features |

## Methods

| Method | Description |
|--------|-------------|
| `fit(X, y, callbacks=None)` | Run the genetic feature selection |
| `transform(X)` | Return `X` with only the selected features |
| `predict(X)` | Predict using `best_estimator_` on selected features |
| `predict_proba(X)` | Predict class probabilities |
| `score(X, y)` | Score using `best_estimator_` on selected features |

## Example

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn_genetic import EvolutionConfig, GAFeatureSelectionCV, PopulationConfig, RuntimeConfig

X, y = load_iris(return_X_y=True)
noise = np.random.uniform(0, 10, size=(X.shape[0], 10))
X = np.hstack((X, noise))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

selector = GAFeatureSelectionCV(
    estimator=SVC(gamma="auto"),
    cv=3,
    scoring="accuracy",
    evolution_config=EvolutionConfig(population_size=30, generations=20, keep_top_k=2, elitism=True),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1),
)

selector.fit(X_train, y_train)

print("Selected features:", selector.support_)
print("CV accuracy:", round(selector.best_score_, 4))
print("Test accuracy:", round(selector.score(X_test, y_test), 4))
```

## See Also

- [Basic Usage](../guide/basic-usage) — feature selection tutorial
- [Config Objects](./config) — configuration objects
- [GASearchCV](./gasearchcv) — hyperparameter tuning variant
