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

## Tips & Gotchas

- Always call `pipe.get_params().keys()` first — it is easy to misspell a step name.
- Preprocessing parameters (e.g., `scaler__with_std`) are part of the same search space and can be tuned alongside model parameters.
- For nested pipelines (a pipeline inside a pipeline), the naming chain extends: `outer_step__inner_step__paramname`.

## Using Preset Search Spaces

Manually defining `param_grid` for every estimator gets repetitive. `sklearn_genetic.presets` ships pre-built parameter grids for common models — pass them directly to `GASearchCV` to skip the boilerplate.

### Available Presets

| Preset | Estimator | Type |
|---|---|---|
| `GRADIENT_BOOSTING_REG` / `GRADIENT_BOOSTING_CLF` | `GradientBoostingRegressor` / `GradientBoostingClassifier` | Regression / Classification |
| `RANDOM_FOREST_REG` / `RANDOM_FOREST_CLF` | `RandomForestRegressor` / `RandomForestClassifier` | Regression / Classification |
| `SVR` / `SVC` | `SVR` / `SVC` | Regression / Classification |
| `LOGISTIC_REGRESSION` | `LogisticRegression` | Classification |
| `RIDGE` / `LASSO` / `ELASTIC_NET` | `Ridge` / `Lasso` / `ElasticNet` | Regression |
| `KNN_REGRESSOR` / `KNN_CLASSIFIER` | `KNeighborsRegressor` / `KNeighborsClassifier` | Regression / Classification |
| `DECISION_TREE_REG` / `DECISION_TREE_CLF` | `DecisionTreeRegressor` / `DecisionTreeClassifier` | Regression / Classification |

See the [Preset Search Spaces API reference](../api/presets) for the full list and parameter ranges.

### Using a Preset with a Pipeline

Presets use bare parameter names (e.g., `n_estimators`). Inside a `Pipeline`, parameters must be prefixed with the step name (`regressor__n_estimators`). Build the prefixed grid with a dict comprehension:

```python
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn_genetic import GASearchCV
from sklearn_genetic.presets import GRADIENT_BOOSTING_REG

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", GradientBoostingRegressor(random_state=42)),
])

# Prefix each preset key with the pipeline step name
param_grid = {
    f"regressor__{k}": v for k, v in GRADIENT_BOOSTING_REG.items()
}

search = GASearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
)

search.fit(X_train, y_train)
print("Best parameters:", search.best_params_)
```

:::tip
You can mix preset parameters with custom ones in the same `param_grid`. Add pipeline-specific parameters (like `scaler__with_std`) alongside the prefixed preset entries.
:::

## See Also

- [Gradient Boosting Hyperparameter Tuning](../tutorials/tune-gradient-boosting) — pipeline-friendly estimator
- [MLflow Integration](./mlflow) — track pipeline experiments
- [Multi-Metric Optimization](./multi-metric) — evaluate pipelines on multiple metrics
- [Common Hyperparameter Tuning Mistakes](./common-mistakes) — avoid data leakage in pipelines
- [Examples: Pipeline Regression](../examples/pipeline-regression) — full regression pipeline example
