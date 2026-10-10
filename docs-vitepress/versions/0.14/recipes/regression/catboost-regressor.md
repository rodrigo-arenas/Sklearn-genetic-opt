---
title: "Tune CatBoostRegressor with Genetic Algorithms"
description: "Copy-paste recipe to tune CatBoost for regression tasks including ordered boosting and bagging_temperature."
---

# Tune CatBoostRegressor

**Time:** 5 min | **Difficulty:** Intermediate

## What This Solves

CatBoost regression uses the same unique params (`bagging_temperature`, `border_count`) as the classifier. This recipe shows the minimal working setup with RMSE scoring.

## Recipe

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import numpy as np

from sklearn_genetic import GASearchCV, EvolutionConfig, RuntimeConfig
from sklearn_genetic.space import Continuous, Integer

X, y = make_regression(n_samples=2000, n_features=20, n_informative=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    "iterations":          Integer(50, 400),
    "depth":               Integer(3, 10),
    "learning_rate":       Continuous(0.01, 0.3, distribution="log-uniform"),
    "l2_leaf_reg":         Continuous(1e-3, 10.0, distribution="log-uniform"),
    "bagging_temperature": Continuous(0.0, 1.0),
    "border_count":        Integer(32, 255),
}

cv = KFold(n_splits=3, shuffle=True, random_state=42)

ga = GASearchCV(
    estimator=CatBoostRegressor(
        random_seed=42,
        thread_count=1,   # ← required
        verbose=0,
        loss_function="RMSE",
    ),
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    evolution_config=EvolutionConfig(
        population_size=15,
        generations=10,
        elitism=True,
    ),
    runtime_config=RuntimeConfig(
        n_jobs=-1,
        parallel_backend="cv",
        verbose=True,
    ),
    random_state=42,
)
ga.fit(X_train, y_train)

pred = ga.predict(X_test)
print("Test RMSE:", round(np.sqrt(mean_squared_error(y_test, pred)), 4))
print("Best params:", ga.best_params_)
```

## Key Points

- **`loss_function="RMSE"`**: Explicitly set the CatBoost internal loss for regression. Default is `RMSE`, but being explicit avoids surprises.
- **`thread_count=1`**: CatBoost's version of `n_jobs=1` — prevents CPU oversubscription.
- **`bagging_temperature=0`**: Disables bootstrap. Useful if you have very little data and want every sample in every tree.

## See Also

- [CatBoost Hyperparameter Tuning](../../tutorials/tune-catboost) — full tutorial
- [XGBoost Regressor](./xgboost-regressor) — depth-wise alternative
- [LightGBM Regressor](./lightgbm-regressor) — fastest training option
