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

## Pipeline prefixes

Use `prefix` when tuning an estimator inside a scikit-learn `Pipeline`:

```python
from sklearn_genetic import svc_space

param_grid = svc_space(prefix="model__")
```

This returns parameter names such as `model__C`, `model__kernel`, and
`model__gamma`.

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
