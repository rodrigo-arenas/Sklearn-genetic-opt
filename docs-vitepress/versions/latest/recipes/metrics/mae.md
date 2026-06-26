---
title: "Tune Hyperparameters for MAE (Mean Absolute Error)"
description: "Recipe to optimize hyperparameters for MAE in regression, with the negated scorer pattern and result interpretation."
---

:::warning Development version
You are reading the **latest (development)** docs. For stable documentation, see [stable](/stable/).
:::

# Tune for MAE (Mean Absolute Error)

**Time:** 5 min | **Difficulty:** Beginner

## What This Solves

MAE is more interpretable than RMSE (same units as the target, no squared penalty for outliers). sklearn maximizes scores, so you must use `neg_mean_absolute_error` — this recipe shows the correct setup.

## Recipe

```python
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, train_test_split

from sklearn_genetic import GASearchCV, EvolutionConfig, RuntimeConfig
from sklearn_genetic.space import Categorical, Continuous, Integer

X, y = make_regression(
    n_samples=1000, n_features=20, n_informative=10, noise=0.3, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    "n_estimators":      Integer(50, 300),
    "max_depth":         Integer(3, 20),
    "min_samples_leaf":  Integer(1, 20),
    "max_features":      Continuous(0.1, 1.0),
    "bootstrap":         Categorical([True, False]),
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

ga = GASearchCV(
    estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",   # sklearn negates minimization objectives
    cv=cv,
    evolution_config=EvolutionConfig(population_size=20, generations=15, elitism=True),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=True),
    random_state=42,
)
ga.fit(X_train, y_train)

pred = ga.predict(X_test)
test_mae = mean_absolute_error(y_test, pred)

print(f"Best CV score (neg MAE): {ga.best_score_:.4f}")
print(f"Best CV MAE:             {-ga.best_score_:.4f}")    # negate to get MAE
print(f"Test MAE:                {test_mae:.4f}")
print("Best params:", ga.best_params_)
```

## Key Points

- **`neg_mean_absolute_error`**: sklearn maximizes scoring functions, so it negates MAE. The best score will be a large negative number close to 0.
- **Negate to read**: `test_mae = -ga.best_score_` gives the actual MAE.
- **`KFold` not `StratifiedKFold`**: For regression there are no classes to stratify.
- **MAE vs RMSE**: MAE is robust to outliers (linear penalty). RMSE penalizes large errors more (squared). Choose based on whether large errors matter disproportionately.

## See Also

- [Tune for RMSE](./rmse) — RMSE when large errors matter more
- [Tune RandomForestRegressor](../regression/random-forest-regressor) — RF regression recipe
- [Tune XGBRegressor](../regression/xgboost-regressor) — XGBoost regression recipe
