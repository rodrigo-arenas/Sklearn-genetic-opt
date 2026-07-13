---
title: "Tune SGDRegressor with Genetic Algorithms"
description: "Copy-paste recipe to perform genetic feature selection and hyperparameter tuning for SGDRegressor on regression tasks."
---

:::warning Development version
You are reading the **latest (development)** docs. For stable documentation, see [stable](/stable/).
:::

# Tune SGDRegressor

**Time:** 5 min | **Difficulty:** Intermediate

## What This Solves

`SGDRegressor` is a linear model that is sensitive to feature scaling and hyperparameter selection. This recipe demonstrates a two-stage optimization workflow using genetic algorithms:

1. **Feature selection** with `GAFeatureSelectionCV`
2. **Hyperparameter tuning** with `GASearchCV`

The example uses a synthetic regression dataset generated with `make_regression()`, performs feature scaling using `StandardScaler`, optimizes the model using **RMSE** during cross-validation, and evaluates the final model using the **R² score** on the test set.

## Recipe

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from sklearn_genetic import GASearchCV, GAFeatureSelectionCV
from sklearn_genetic.space import Categorical, Continuous, Integer
from sklearn_genetic.config import (
    RuntimeConfig,
    EvolutionConfig,
    OptimizationConfig,
)

# Generate a synthetic regression dataset
X, y = make_regression(
    n_samples=1000,
    n_features=20,
    noise=0.1,
    random_state=42,
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

# Stage 1: Feature Selection
selector = GAFeatureSelectionCV(
    estimator=SGDRegressor(random_state=42),
    cv=cv,
    scoring="neg_root_mean_squared_error",
    evolution_config=EvolutionConfig(
        population_size=20,
        generations=15,
        elitism=True,
    ),
    optimization_config=OptimizationConfig(
        diversity_threshold=1.0,
        sharing_alpha=5.0,
    ),
    runtime_config=RuntimeConfig(
        n_jobs=-1,
        verbose=True,
    ),
    random_state=42,
    local_search_steps=3,
)

selector.fit(X_train, y_train)

mask = selector.support_

print(
    f"Stage 1: {mask.sum()} features selected "
    f"(from {X_train.shape[1]})"
)

X_train_sel = X_train[:, mask]
X_test_sel = X_test[:, mask]

# Stage 2: Hyperparameter Tuning
param_grid = {
    "alpha": Continuous(0.0, 0.01),
    "l1_ratio": Continuous(0.0, 1.0),
    "max_iter": Integer(500, 5000),
    "tol": Continuous(1e-6, 1e-2),
    "eta0": Continuous(0.0, 1.0),
    "n_iter_no_change": Integer(5, 50),
    "penalty": Categorical([
        "l2",
        "l1",
        "elasticnet",
    ]),
    "learning_rate": Categorical([
        "constant",
        "optimal",
        "invscaling",
        "adaptive",
    ]),
    "average": Categorical([
        True,
        False,
    ]),
}

ga = GASearchCV(
    estimator=SGDRegressor(random_state=42),
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    evolution_config=EvolutionConfig(
        population_size=20,
        generations=15,
        elitism=True,
    ),
    runtime_config=RuntimeConfig(
        n_jobs=-1,
        verbose=True,
    ),
    random_state=42,
)

ga.fit(X_train_sel, y_train)

pred = ga.predict(X_test_sel)

print("R² on Test Set:", round(r2_score(y_test, pred), 4))
print("Best Parameters:", ga.best_params_)
```

## Key Points

- **Scale the features before optimization**: `SGDRegressor` is sensitive to the scale of the input features. Always apply `StandardScaler` before feature selection and hyperparameter tuning.
- **Two-stage optimization**: First select informative features using `GAFeatureSelectionCV`, then tune the estimator using `GASearchCV`.
- **Cross-validation metric**: The genetic algorithms optimize the model using `neg_root_mean_squared_error`.
- **Final evaluation**: The tuned model is evaluated on the held-out test set using the **R² score**.

## See Also

- [Tune for MAE](../metrics/mae) — MAE as alternative metric
- [Tune for RMSE](../metrics/rmse) — custom RMSE scorer
- [LightGBM Regressor](./lightgbm-regressor) — fastest training option