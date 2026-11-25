---
title: How to Tune an Entire scikit-learn Pipeline with Genetic Algorithms
description: Tune preprocessing and model hyperparameters together by building a scikit-learn Pipeline and running GASearchCV across the full parameter space.
---

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

### Choose a configured pipeline step

Use the step name as the parameter and give `Categorical` a list of configured
estimators. This lets you offer a finite set of model choices whose parameters
depend on the model or kernel. See [conditional choices for an SVM pipeline](../examples/conditional-pipeline-choices).

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

## Cache fitted preprocessing with `Pipeline.memory`

When many candidates share an expensive transformer, set `memory` on the
scikit-learn pipeline. Candidates that use the same transformer settings and
training inputs can reuse the fitted transformer and its transformed training
data. Keep the transformer inside the pipeline so it is fitted on each fold's
training rows. scikit-learn fits the final pipeline step for every fresh CV fold.

This example compares cold-cache and uncached fits with the same splits, seed,
and search settings. Replace the transformer and dataset with your own workload
when deciding whether to enable caching.

```python
from tempfile import TemporaryDirectory
from time import perf_counter

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from sklearn_genetic import GASearchCV
from sklearn_genetic.callbacks import CVBudgetStopping
from sklearn_genetic.space import Continuous

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
cv = list(StratifiedKFold(3, shuffle=True, random_state=42).split(X_train, y_train))

with TemporaryDirectory() as cache_dir:
    for memory in (None, cache_dir):
        pipe = Pipeline([
            ("pca", PCA(n_components=24, svd_solver="full")),
            ("classifier", LogisticRegression(max_iter=500, random_state=42)),
        ], memory=memory)
        search = GASearchCV(
            pipe,
            param_grid={"classifier__C": Continuous(0.001, 100, distribution="log-uniform")},
            cv=cv,
            scoring="accuracy",
            population_size=8,
            generations=4,
            random_state=42,
            n_jobs=1,
            verbose=False,
        )
        start = perf_counter()
        search.fit(X_train, y_train, callbacks=CVBudgetStopping(16))
        seconds = perf_counter() - start
        label = "cached" if memory else "uncached"
        folds = search.fit_counts_["search_fold_fits"]
        holdout = search.score(X_test, y_test)
        print(f"{label}: {seconds:.3f} seconds, {folds} fold fits")
        print(f"CV accuracy: {search.best_score_:.4f}, holdout accuracy: {holdout:.4f}")
```

`GASearchCV(use_cache=True)` reuses whole candidate evaluations when the same
parameters recur. `Pipeline.memory` can save preprocessing work even when each
candidate uses different model parameters. Compare total elapsed `fit` time as
well as CV fold counts, since a reused transformer still counts as part of a
fold fit. Seed randomized transformers so a cached fit can be compared with a
reproducible uncached fit. A cheap transformer or many distinct preprocessing
settings can make hashing and disk I/O cost more than they save. The temporary
directory in the example deletes cached data when the comparison finishes.
See the [scikit-learn Pipeline API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).

## Using Preset Search Spaces

Manually defining `param_grid` for every estimator gets repetitive. `sklearn_genetic` ships pre-built parameter grids for common models — call a preset function and pass the result directly to `GASearchCV` to skip the boilerplate.

### Available Presets

Each preset is a factory function that returns a ready-to-use `param_grid` dict. Pass `profile="fast"`, `"balanced"` (default), or `"wide"` to control the search range.

| Preset function | Estimator | Type |
|---|---|---|
| `random_forest_regressor_space` / `random_forest_classifier_space` | `RandomForestRegressor` / `RandomForestClassifier` | Regression / Classification |
| `hist_gradient_boosting_regressor_space` / `hist_gradient_boosting_classifier_space` | `HistGradientBoostingRegressor` / `HistGradientBoostingClassifier` | Regression / Classification |
| `xgboost_regressor_space` / `xgboost_classifier_space` | `XGBRegressor` / `XGBClassifier` | Regression / Classification |
| `svc_space` | `SVC` | Classification |
| `logistic_regression_space` | `LogisticRegression` | Classification |

See the [Preset Search Spaces API reference](../api/presets) for the full list and parameter ranges.

### Using a Preset with a Pipeline

Every preset function accepts a `prefix` argument. Inside a `Pipeline`, set `prefix` to the step name followed by `__` so the returned keys are already pipeline-ready — no manual dict comprehension needed:

```python
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn_genetic import GASearchCV, random_forest_regressor_space

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", RandomForestRegressor(random_state=42, n_jobs=1)),
])

# prefix="regressor__" matches the "regressor" step, e.g. regressor__n_estimators
param_grid = random_forest_regressor_space(prefix="regressor__")

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
