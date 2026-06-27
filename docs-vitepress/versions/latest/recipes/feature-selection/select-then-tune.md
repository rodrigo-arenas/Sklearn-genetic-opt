---
title: "Combine Feature Selection and Hyperparameter Tuning"
description: "Two-stage recipe: use GAFeatureSelectionCV to select features, then GASearchCV to tune the estimator on the reduced feature set."
---

:::warning Development version
You are reading the **latest (development)** docs. For stable documentation, see [stable](/stable/).
:::

# Combine Feature Selection + Hyperparameter Tuning

**Time:** 8 min | **Difficulty:** Intermediate

## What This Solves

Running hyperparameter search on all features wastes compute on noise. Running feature selection with default hyperparameters means the selection is biased. The two-stage approach: (1) select features with default params, (2) retune on the selected subset.

:::warning Avoid leaking test data
Apply `selector.support_` to train data only, then apply the same mask to the test set. Never fit anything on the test set.
:::

## Recipe

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn_genetic import GAFeatureSelectionCV, GASearchCV, EvolutionConfig, RuntimeConfig
from sklearn_genetic.space import Categorical, Continuous, Integer

X, y = make_classification(
    n_samples=1000, n_features=30, n_informative=10, n_redundant=10, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Stage 1: Feature selection with default estimator params
selector = GAFeatureSelectionCV(
    estimator=RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    cv=cv,
    scoring="roc_auc",
    population_size=20,
    generations=15,
    elitism=True,
    verbose=False,
    n_jobs=-1,
    random_state=42,
)
selector.fit(X_train, y_train)
mask = selector.support_
print(f"Stage 1: {mask.sum()} features selected (from {X_train.shape[1]})")

# Apply mask
X_train_sel = X_train[:, mask]
X_test_sel  = X_test[:, mask]

# Stage 2: Hyperparameter tuning on the reduced feature set
param_grid = {
    "n_estimators":      Integer(50, 300),
    "max_depth":         Integer(3, 20),
    "min_samples_leaf":  Integer(1, 15),
    "max_features":      Continuous(0.1, 1.0),
    "class_weight":      Categorical([None, "balanced"]),
}

ga = GASearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    evolution_config=EvolutionConfig(population_size=20, generations=15, elitism=True),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=True),
    random_state=42,
)
ga.fit(X_train_sel, y_train)

print(f"Stage 2 Best CV ROC AUC: {ga.best_score_:.4f}")
print(f"Test ROC AUC: {ga.score(X_test_sel, y_test):.4f}")
```

## Key Points

- **Same `cv` object for both stages**: Ensures the same fold splits are used.
- **Apply mask to train set, then test set**: Never fit the selector or tuner on the test set.
- **Stage 2 can use a different estimator**: Select features with a fast estimator (RF), then tune a more expensive one (XGBoost) on the reduced set.
- **Third stage (optional)**: Validate on a second estimator to confirm the selected features generalize — see the [Feature Selection Tutorial](../../tutorials/feature-selection).

## See Also

- [Feature Selection Tutorial](../../tutorials/feature-selection) — 3-stage workflow with second-estimator validation
- [Feature Selection on 50+ Columns](./high-dimensional) — scaling to large feature sets
- [Feature Selection Methods Compared](../../guide/feature-selection-guide) — wrapper vs filter
