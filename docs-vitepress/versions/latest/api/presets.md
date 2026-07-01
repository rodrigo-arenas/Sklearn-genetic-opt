---
title: Estimator Presets
description: Starter search spaces for common scikit-learn estimators.
---

:::warning Development version
You are reading the **latest (dev)** docs. For the stable version, see [stable](/stable/).
:::

# Estimator Presets

Preset functions return ordinary `param_grid` dictionaries using `Integer`,
`Continuous`, and `Categorical` dimensions. They are meant as strong starting
points, not universal best settings.

```python
from sklearn.ensemble import RandomForestClassifier

from sklearn_genetic import GASearchCV, random_forest_classifier_space

search = GASearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=random_forest_classifier_space(profile="balanced"),
    scoring="roc_auc",
    cv=5,
    random_state=42,
)
```

## Available presets

| Function | Estimator |
|----------|-----------|
| `random_forest_classifier_space` | `RandomForestClassifier` |
| `random_forest_regressor_space` | `RandomForestRegressor` |
| `hist_gradient_boosting_classifier_space` | `HistGradientBoostingClassifier` |
| `hist_gradient_boosting_regressor_space` | `HistGradientBoostingRegressor` |
| `logistic_regression_space` | `LogisticRegression` with `solver="saga"` and `penalty="elasticnet"` |
| `svc_space` | `SVC` |
| `xgboost_classifier_space` | `xgboost.XGBClassifier` |
| `xgboost_regressor_space` | `xgboost.XGBRegressor` |

## Profiles

Each preset accepts `profile="fast"`, `"balanced"`, or `"wide"`:

```python
param_grid = random_forest_classifier_space(profile="fast")
```

| Profile | Use when |
|---------|----------|
| `fast` | You want a quick first run or a notebook demo |
| `balanced` | You want a practical default for real tuning |
| `wide` | You have more budget and want broader exploration |

## Discovering presets

Two helpers let you list the available presets and profiles from Python — handy
in a notebook or for building a UI:

```python
from sklearn_genetic import list_preset_profiles, list_preset_spaces

list_preset_profiles()
# ['balanced', 'fast', 'wide']

list_preset_spaces()
# ['hist_gradient_boosting_classifier_space', ..., 'xgboost_regressor_space']
```

Both return sorted lists, and every name from `list_preset_spaces()` is
importable from `sklearn_genetic`.

## Pipeline prefixes

Use `prefix` when tuning an estimator inside a scikit-learn `Pipeline`:

```python
from sklearn_genetic import svc_space

param_grid = svc_space(prefix="model__")
```

This returns parameter names such as `model__C`, `model__kernel`, and
`model__gamma`.

### Full example: tuning a preset inside a `Pipeline`

The `prefix` must match the **step name** of the estimator in the pipeline. If
the step is named `"model"`, use `prefix="model__"` so the keys become
`model__n_estimators`, `model__max_depth`, and so on:

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

# prefix="model__" matches the "model" step, e.g. model__n_estimators
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

:::tip
The prefix is just the pipeline step name followed by `__`. If your step is
called `"clf"`, use `prefix="clf__"`. See [Tuning scikit-learn Pipelines](../guide/pipeline-tuning) for more.
:::

## XGBoost

The XGBoost presets cover the common nine-parameter tree booster space used in
the XGBoost tutorials and recipes:

```python
from xgboost import XGBClassifier

from sklearn_genetic import GASearchCV, RuntimeConfig, xgboost_classifier_space

search = GASearchCV(
    estimator=XGBClassifier(tree_method="hist", eval_metric="logloss", n_jobs=1),
    param_grid=xgboost_classifier_space(profile="balanced"),
    scoring="roc_auc",
    cv=5,
    runtime_config=RuntimeConfig(n_jobs=-1, parallel_backend="cv"),
    random_state=42,
)
```

:::warning XGBoost threading
Set `n_jobs=1` on the XGBoost estimator and parallelize with
`parallel_backend="cv"` on `GASearchCV` to avoid CPU oversubscription.
:::

## See Also

- [Search Space](./space) — native dimensions and sklearn/scipy conversion
- [Tuning scikit-learn Pipelines](../guide/pipeline-tuning)
- [Choosing the Right Search Space](../guide/choosing-search-spaces)
- [XGBoost Hyperparameter Tuning](../tutorials/tune-xgboost)
