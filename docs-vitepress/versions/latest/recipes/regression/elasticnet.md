---
title: "Tune ElasticNet with Genetic Algorithms"
description: "Copy-paste recipe to tune ElasticNet alpha and l1_ratio with log-uniform sampling using GASearchCV."
---

:::warning Development version
You are reading the **latest (development)** docs. For stable documentation, see [stable](/stable/).
:::

# Tune ElasticNet

**Time:** 5 min | **Difficulty:** Beginner

## What This Solves

`ElasticNet` blends L1 and L2 regularization via `alpha` (strength) and `l1_ratio` (blend). These two parameters interact: the right `l1_ratio` depends on `alpha`. A 2D search finds the productive combination.

## Recipe

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import numpy as np

from sklearn_genetic import GASearchCV, EvolutionConfig, RuntimeConfig
from sklearn_genetic.space import Continuous

X, y = make_regression(
    n_samples=1000, n_features=30, n_informative=10, noise=0.5, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# StandardScaler required — ElasticNet is not scale-invariant
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("enet", ElasticNet(max_iter=5000, random_state=42)),
])

param_grid = {
    "enet__alpha":    Continuous(1e-4, 10.0, distribution="log-uniform"),
    "enet__l1_ratio": Continuous(0.0, 1.0),
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

ga = GASearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
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
print("Best CV MSE (neg):", round(ga.best_score_, 4))
print("Test RMSE:", round(np.sqrt(mean_squared_error(y_test, pred)), 4))
print("Best params:", ga.best_params_)
print(f"  → l1_ratio={ga.best_params_['enet__l1_ratio']:.3f}: "
      f"{'mostly L1 (Lasso)' if ga.best_params_['enet__l1_ratio'] > 0.7 else 'mostly L2 (Ridge)' if ga.best_params_['enet__l1_ratio'] < 0.3 else 'balanced blend'}")
```

## Key Points

- **`alpha` spans orders of magnitude**: Use log-uniform (1e-4 to 10) so small values like 0.001 get fair coverage alongside 1.0 and 10.0.
- **`l1_ratio=0` is Ridge, `l1_ratio=1` is Lasso**: Values in between blend sparsity (L1) with stability (L2).
- **Scale first**: `ElasticNet` penalizes large coefficients — unscaled features with large magnitudes get artificially shrunk.
- **`max_iter=5000`**: Low `alpha` values can require many iterations to converge. Increase if you see `ConvergenceWarning`.

## See Also

- [Choosing the Right Search Space](../../guide/choosing-search-spaces) — when to use log-uniform vs uniform
- [Tune for RMSE](../metrics/rmse) — RMSE scoring setup
- [Tune for MAE](../metrics/mae) — MAE as alternative metric
