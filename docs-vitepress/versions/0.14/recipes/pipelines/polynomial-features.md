---
title: "Tune Polynomial Features Degree as a Hyperparameter"
description: "Recipe to search over PolynomialFeatures degree and interaction_only as pipeline hyperparameters with GASearchCV."
---

# Tune Polynomial Features Degree

**Time:** 5 min | **Difficulty:** Beginner

## What This Solves

`PolynomialFeatures` degree is often set by hand. Including it as a search parameter lets the genetic algorithm find the right balance between feature expressiveness and overfitting.

:::warning Feature count explosion
Degree 2 with 20 features → 231 features. Degree 3 with 20 features → 1771 features. Keep input features small or use `interaction_only=True`.
:::

## Recipe

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn_genetic import GASearchCV, EvolutionConfig, RuntimeConfig
from sklearn_genetic.space import Categorical, Continuous, Integer

X, y = make_regression(
    n_samples=500, n_features=8, n_informative=5, noise=0.3, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = Pipeline([
    ("poly",   PolynomialFeatures(include_bias=False)),
    ("scaler", StandardScaler()),
    ("ridge",  Ridge()),
])

param_grid = {
    "poly__degree":          Integer(1, 3),
    "poly__interaction_only": Categorical([True, False]),
    "ridge__alpha":          Continuous(1e-3, 100.0, distribution="log-uniform"),
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

ga = GASearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    cv=cv,
    evolution_config=EvolutionConfig(population_size=15, generations=12, elitism=True),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=True),
    random_state=42,
)
ga.fit(X_train, y_train)

print("Best CV MSE (neg):", round(ga.best_score_, 4))
print("Best degree:", ga.best_params_["poly__degree"])
print("Best interaction_only:", ga.best_params_["poly__interaction_only"])
print("Best alpha:", round(ga.best_params_["ridge__alpha"], 4))
```

## Key Points

- **`include_bias=False`**: Prevents a constant column from being added (Ridge/Lasso handle this via the intercept).
- **`interaction_only=True`**: Only creates products of distinct features (no `x^2`). Reduces explosion: 8 features → 36 columns vs 45 with full degree 2.
- **Scale after polynomial expansion**: The polynomial features will have different scales than the originals — `StandardScaler` after `PolynomialFeatures` is critical.
- **Keep input features small**: With 8 features and degree ≤ 3, the expansion is manageable. With 20+ features, restrict to `interaction_only=True`.

## See Also

- [Tuning scikit-learn Pipelines](../../guide/pipeline-tuning) — full guide
- [Tune ElasticNet](../regression/elasticnet) — regularized linear regression
- [Preprocessing + Estimator Pipeline](./preprocessing-pipeline) — simpler pipeline pattern
