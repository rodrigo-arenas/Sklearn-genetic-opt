---
title: How to Tune an Entire scikit-learn Pipeline with Genetic Algorithms
description: Tune preprocessing and model hyperparameters together by building a scikit-learn Pipeline and running GASearchCV across the full parameter space.
---

:::warning Development version
You are reading the **latest (dev)** docs. For the stable version, see [stable](/stable/).
:::

**Estimated reading time:** 8 minutes  
**Difficulty:** Intermediate  
**Prerequisites:** [Getting Started with GASearchCV](./basic-usage), basic sklearn Pipeline knowledge

# Pipeline Tuning with GASearchCV

scikit-learn `Pipeline` objects let you chain preprocessing steps and an estimator into a single object. `GASearchCV` tunes pipelines the same way it tunes plain estimators — the only difference is the parameter naming convention.

## Prerequisites

- Completed [Basic Usage](./basic-usage)
- Familiarity with `sklearn.pipeline.Pipeline`

## Parameter Naming Inside a Pipeline

Pipeline parameters follow the pattern `stepname__paramname` (two underscores). The step name is the string you assigned when creating the pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", GradientBoostingRegressor()),
])

# Discover all tunable parameters:
print(list(pipe.get_params().keys()))
# -> ['scaler', 'regressor', 'scaler__copy', ..., 'regressor__n_estimators', ...]
```

Use `pipe.get_params().keys()` to discover the exact names before writing `param_grid`.

## Full Example: Gradient Boosting Regression Pipeline

```python
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn_genetic import EvolutionConfig, GASearchCV, PopulationConfig, RuntimeConfig
from sklearn_genetic.space import Categorical, Continuous, Integer

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", GradientBoostingRegressor(random_state=42)),
])

param_grid = {
    "regressor__n_estimators": Integer(50, 300),
    "regressor__learning_rate": Continuous(0.01, 0.3, distribution="log-uniform"),
    "regressor__max_depth": Integer(2, 6),
    "regressor__min_samples_leaf": Integer(1, 20),
    "regressor__subsample": Continuous(0.5, 1.0),
    "scaler__with_std": Categorical([True, False]),
}

search = GASearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring="neg_root_mean_squared_error",
    evolution_config=EvolutionConfig(population_size=20, generations=15),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, use_cache=True),
)

search.fit(X_train, y_train)

print("Best CV RMSE:", round(-search.best_score_, 4))
print("Best parameters:", search.best_params_)

y_pred = search.predict(X_test)
```

## Using Preset Search Spaces

Instead of writing every dimension by hand, you can start from a [preset](../api/presets)
search space and pass `prefix` to match your pipeline step name:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn_genetic import GASearchCV
from sklearn_genetic.presets import random_forest_classifier_space

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)

pipe = Pipeline(
    [
        ("scale", StandardScaler()),
        ("model", RandomForestClassifier(random_state=0, n_jobs=1)),
    ]
)

# prefix="model__" produces keys like model__n_estimators, model__max_depth
param_grid = random_forest_classifier_space(prefix="model__")

search = GASearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=3,
    population_size=10,
    generations=5,
    random_state=0,
)
search.fit(X_train, y_train)
print("Best ROC AUC:", round(search.best_score_, 4))
```

The prefix must match the step name — `"model"` → `prefix="model__"`. See
[Estimator Presets](../api/presets) for the full list of available presets and
profiles (`"fast"`, `"balanced"`, `"wide"`).

## Tips & Gotchas

- Always call `pipe.get_params().keys()` first — it is easy to misspell a step name.
- Preprocessing parameters (e.g., `scaler__with_std`) are part of the same search space and can be tuned alongside model parameters.
- For nested pipelines (a pipeline inside a pipeline), the naming chain extends: `outer_step__inner_step__paramname`.

## See Also

- [Gradient Boosting Hyperparameter Tuning](../tutorials/tune-gradient-boosting) — pipeline-friendly estimator
- [MLflow Integration](./mlflow) — track pipeline experiments
- [Multi-Metric Optimization](./multi-metric) — evaluate pipelines on multiple metrics
- [Common Hyperparameter Tuning Mistakes](./common-mistakes) — avoid data leakage in pipelines
- [Examples: Pipeline Regression](../examples/pipeline-regression) — full regression pipeline example
