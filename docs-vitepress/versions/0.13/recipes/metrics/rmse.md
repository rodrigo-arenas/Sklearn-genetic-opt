---
title: "Tune Hyperparameters for RMSE (Root Mean Squared Error)"
description: "Recipe to optimize hyperparameters for RMSE in regression using neg_root_mean_squared_error scorer."
---

# Tune for RMSE (Root Mean Squared Error)

**Time:** 5 min | **Difficulty:** Beginner

## What This Solves

RMSE penalizes large errors more than MAE (squared → squared root). Use it when large prediction errors are disproportionately costly. This recipe shows how to use sklearn's built-in `neg_root_mean_squared_error`.

## Recipe

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split

from sklearn_genetic import GASearchCV, EvolutionConfig, RuntimeConfig
from sklearn_genetic.space import Continuous, Integer

X, y = make_regression(
    n_samples=1000, n_features=20, n_informative=10, noise=0.5, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    "n_estimators":  Integer(50, 300),
    "max_depth":     Integer(2, 8),
    "learning_rate": Continuous(0.01, 0.3, distribution="log-uniform"),
    "subsample":     Continuous(0.5, 1.0),
    "max_features":  Continuous(0.3, 1.0),
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

ga = GASearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",   # sklearn ≥ 0.24
    cv=cv,
    evolution_config=EvolutionConfig(population_size=20, generations=15, elitism=True),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=True),
    random_state=42,
)
ga.fit(X_train, y_train)

pred = ga.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, pred))

print(f"Best CV score (neg RMSE): {ga.best_score_:.4f}")
print(f"Best CV RMSE:             {-ga.best_score_:.4f}")
print(f"Test RMSE:                {test_rmse:.4f}")
print("Best params:", ga.best_params_)
```

## Older sklearn (< 0.24)

`neg_root_mean_squared_error` was added in sklearn 0.24. For earlier versions:

```python
from sklearn.metrics import make_scorer, mean_squared_error
import numpy as np

rmse_scorer = make_scorer(
    lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)),
    greater_is_better=True
)
ga = GASearchCV(..., scoring=rmse_scorer, ...)
```

## Key Points

- **`neg_root_mean_squared_error`**: Already square-rooted, so units match the target. No extra square root needed when reading `best_score_`.
- **RMSE vs `neg_mean_squared_error`**: MSE uses squared units (hard to interpret). RMSE restores the original unit scale.
- **Negate `best_score_`**: `actual_rmse = -ga.best_score_`.

## See Also

- [Tune for MAE](./mae) — when you want equal penalty for all errors
- [Tune LGBMRegressor](../regression/lightgbm-regressor) — RMSE for LightGBM
- [Tune XGBRegressor](../regression/xgboost-regressor) — RMSE for XGBoost
