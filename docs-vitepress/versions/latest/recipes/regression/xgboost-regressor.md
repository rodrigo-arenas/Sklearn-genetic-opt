---
title: "Tune XGBRegressor with Genetic Algorithms"
description: "Copy-paste recipe to tune XGBoost for regression tasks with RMSE scoring and the n_jobs=1 CPU fix."
---

:::warning Development version
You are reading the **latest (development)** docs. For stable documentation, see [stable](/stable/).
:::

# Tune XGBRegressor

**Time:** 5 min | **Difficulty:** Intermediate

## What This Solves

The XGBoost regressor has the same hyperparameter interactions as the classifier, but uses `reg:squarederror` as the objective and MAE/RMSE as the evaluation metric. This recipe shows the correct setup.

## Recipe

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import numpy as np

from sklearn_genetic import GASearchCV, EvolutionConfig, RuntimeConfig
from sklearn_genetic.space import Continuous, Integer

X, y = make_regression(n_samples=2000, n_features=20, n_informative=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    "n_estimators":     Integer(50, 350),
    "max_depth":        Integer(2, 10),
    "min_child_weight": Integer(1, 12),
    "subsample":        Continuous(0.5, 1.0),
    "colsample_bytree": Continuous(0.4, 1.0),
    "learning_rate":    Continuous(0.01, 0.3, distribution="log-uniform"),
    "gamma":            Continuous(1e-4, 1.0, distribution="log-uniform"),
    "reg_alpha":        Continuous(1e-5, 10.0, distribution="log-uniform"),
    "reg_lambda":       Continuous(1e-5, 10.0, distribution="log-uniform"),
}

cv = KFold(n_splits=3, shuffle=True, random_state=42)

ga = GASearchCV(
    estimator=XGBRegressor(
        tree_method="hist",
        eval_metric="rmse",
        random_state=42,
        n_jobs=1,         # ← required: prevent CPU oversubscription
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

- **`n_jobs=1` on estimator**: Same oversubscription fix as the classifier — see [XGBoost Classifier recipe](../classification/xgboost-classifier).
- **`neg_root_mean_squared_error`**: Available in sklearn ≥ 0.24. Use `neg_mean_squared_error` and take sqrt for earlier versions.
- **`eval_metric="rmse"`**: Sets XGBoost's internal eval metric for early stopping (if used). Doesn't affect the CV scoring metric.

## See Also

- [XGBoost Hyperparameter Tuning](../../tutorials/tune-xgboost) — full classifier tutorial (same params apply to regressor)
- [Tune for RMSE](../metrics/rmse) — custom RMSE scorer
- [LightGBM Regressor](./lightgbm-regressor) — leaf-wise regression alternative
