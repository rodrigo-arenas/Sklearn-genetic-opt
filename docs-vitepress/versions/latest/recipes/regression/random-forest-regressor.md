---
title: "Tune RandomForestRegressor with Genetic Algorithms"
description: "Copy-paste recipe to tune RandomForestRegressor for regression tasks with MAE scoring and min_samples_leaf guidance."
---

:::warning Development version
You are reading the **latest (development)** docs. For stable documentation, see [stable](/stable/).
:::

# Tune RandomForestRegressor

**Time:** 5 min | **Difficulty:** Beginner

## What This Solves

Regression Random Forest has the same hyperparameters as the classifier variant, but `min_samples_leaf` matters more — it directly controls prediction smoothness. This recipe tunes for MAE.

## Recipe

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from sklearn_genetic import GASearchCV, EvolutionConfig, RuntimeConfig
from sklearn_genetic.space import Categorical, Continuous, Integer

X, y = make_regression(n_samples=1000, n_features=20, n_informative=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    "n_estimators":      Integer(50, 300),
    "max_depth":         Integer(3, 20),
    "min_samples_split": Integer(2, 20),
    "min_samples_leaf":  Integer(1, 20),      # key for regression smoothness
    "max_features":      Continuous(0.1, 1.0),
    "bootstrap":         Categorical([True, False]),
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

ga = GASearchCV(
    estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=cv,
    evolution_config=EvolutionConfig(
        population_size=20,
        generations=15,
        elitism=True,
    ),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=True),
    random_state=42,
)
ga.fit(X_train, y_train)

pred = ga.predict(X_test)
print("Best CV MAE (neg):", round(ga.best_score_, 4))
print("Test MAE:", round(mean_absolute_error(y_test, pred), 4))
print("Best params:", ga.best_params_)
```

## Key Points

- **`min_samples_leaf` for regression**: Higher values smooth the output. For noisy targets, try `min_samples_leaf=5–20`. For clean data, `1–3` is fine.
- **`neg_mean_absolute_error`**: sklearn scores are always maximized. MAE is minimized, so use the negated version.
- **`KFold` not `StratifiedKFold`**: Stratification is for classification (balancing class proportions). For regression, use `KFold`.
- **`max_features` as float**: Searching 0.1–1.0 covers `sqrt` (~0.22 for 20 features) and `log2` (~0.22) and beyond.

## See Also

- [Tune for MAE](../metrics/mae) — MAE scoring in depth
- [Tune for RMSE](../metrics/rmse) — when to use RMSE instead
- [Random Forest Classifier](../classification/random-forest-classifier) — classification variant
