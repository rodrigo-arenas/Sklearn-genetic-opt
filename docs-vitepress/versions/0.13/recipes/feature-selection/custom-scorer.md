---
title: "Feature Selection with a Custom Scorer (Feature-Count Penalty)"
description: "Recipe to add a feature-count penalty to the feature selection objective — preferring parsimonious models when two feature sets score equally."
---

# Feature Selection with a Custom Scorer

**Time:** 8 min | **Difficulty:** Advanced

## What This Solves

When two feature subsets score equally on the CV metric, you usually prefer the smaller one (interpretability, inference cost). A custom scorer adds a penalty for the number of features selected.

## Recipe

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.base import clone

from sklearn_genetic import GAFeatureSelectionCV

X, y = make_classification(
    n_samples=800, n_features=30, n_informative=8, n_redundant=10, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

def penalized_roc_auc(estimator, X, y):
    """ROC AUC minus a small penalty proportional to the fraction of features used."""
    proba = estimator.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba)
    n_features_used = X.shape[1]           # after masking, this IS the selected count
    n_total = X_train.shape[1]
    feature_fraction = n_features_used / n_total
    penalty_weight = 0.05                  # tune this: higher = more parsimonious
    return auc - penalty_weight * feature_fraction

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

selector = GAFeatureSelectionCV(
    estimator=RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    cv=cv,
    scoring=penalized_roc_auc,            # custom callable
    population_size=20,
    generations=15,
    elitism=True,
    verbose=True,
    n_jobs=-1,
    random_state=42,
)
selector.fit(X_train, y_train)

mask = selector.support_
print(f"Features selected: {mask.sum()} / {X_train.shape[1]}")
print(f"Best penalized score: {selector.best_score_:.4f}")

# Evaluate on test set with standard AUC (no penalty)
X_test_sel = X_test[:, mask]
rf = clone(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
rf.fit(X_train[:, mask], y_train)
proba = rf.predict_proba(X_test_sel)[:, 1]
print(f"Test ROC AUC (no penalty): {roc_auc_score(y_test, proba):.4f}")
```

## Key Points

- **`scoring` as a callable**: Receives `(estimator, X, y)`. `X` is already masked to the selected features — check `X.shape[1]` to count selected features.
- **`penalty_weight`**: Tune between 0.01 (light preference) and 0.1 (strong preference for small sets). Too high and the search converges to using zero features.
- **Test evaluation**: Evaluate on the test set using standard (un-penalized) AUC for an honest comparison.

## See Also

- [Write a Custom Scoring Function](../advanced/custom-scorer) — `make_scorer` and callable scorer patterns
- [Feature Selection on 50+ Columns](./high-dimensional) — scaling to large feature sets
- [Feature Selection Tutorial](../../tutorials/feature-selection) — complete end-to-end workflow
