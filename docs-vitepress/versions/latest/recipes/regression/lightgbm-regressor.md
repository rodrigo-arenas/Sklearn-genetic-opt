---
title: "Tune LGBMRegressor with Genetic Algorithms"
description: "Copy-paste recipe to tune LightGBM for regression with RMSE scoring and min_child_samples guidance."
---

:::warning Development version
You are reading the **latest (development)** docs. For stable documentation, see [stable](/stable/).
:::

# Tune LGBMRegressor

**Time:** 5 min | **Difficulty:** Intermediate

## What This Solves

LightGBM regression has the same leaf-wise growth as the classifier. The critical difference: `min_child_samples` controls leaf size, which directly affects prediction smoothness on regression targets.

## Recipe

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import numpy as np

from sklearn_genetic import GASearchCV, EvolutionConfig, RuntimeConfig
from sklearn_genetic.space import Continuous, Integer

X, y = make_regression(n_samples=2000, n_features=20, n_informative=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    "n_estimators":      Integer(50, 400),
    "num_leaves":        Integer(20, 150),
    "max_depth":         Integer(3, 12),
    "min_child_samples": Integer(5, 100),     # key for regression smoothness
    "subsample":         Continuous(0.5, 1.0),
    "colsample_bytree":  Continuous(0.4, 1.0),
    "learning_rate":     Continuous(0.01, 0.3, distribution="log-uniform"),
    "reg_alpha":         Continuous(1e-5, 10.0, distribution="log-uniform"),
    "reg_lambda":        Continuous(1e-5, 10.0, distribution="log-uniform"),
}

cv = KFold(n_splits=3, shuffle=True, random_state=42)

ga = GASearchCV(
    estimator=LGBMRegressor(
        random_state=42,
        n_jobs=1,      # ← required
        verbose=-1,
    ),
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    evolution_config=EvolutionConfig(
        population_size=15,
        generations=10,
        elitism=True,
        keep_top_k=3,
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
test_rmse = np.sqrt(mean_squared_error(y_test, pred))
print("Best CV RMSE (neg):", round(ga.best_score_, 4))
print("Test RMSE:", round(test_rmse, 4))
print("Best params:", ga.best_params_)
```

## Key Points

- **`min_child_samples` for regression**: Minimum samples in a leaf. Higher values (50–100) prevent overfitting on noisy regression targets.
- **Wider `min_child_samples` range for regression**: Classification typically uses 5–50; regression with noisy targets benefits from searching up to 100+.
- **`n_jobs=1` on LGBMRegressor**: Same threading issue as LGBMClassifier — use `parallel_backend="cv"`.

## See Also

- [LightGBM Hyperparameter Tuning](../../tutorials/tune-lightgbm) — full tutorial
- [XGBoost Regressor](./xgboost-regressor) — depth-wise alternative
- [Tune for RMSE](../metrics/rmse) — RMSE scoring setup
