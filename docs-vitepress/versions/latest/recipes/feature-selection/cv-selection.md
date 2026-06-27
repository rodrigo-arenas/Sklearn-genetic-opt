---
title: "Feature Selection with Per-Fold Cross-Validation"
description: "Recipe to avoid data leakage by applying feature selection inside cross-validation folds using a Pipeline."
---

:::warning Development version
You are reading the **latest (development)** docs. For stable documentation, see [stable](/stable/).
:::

# Feature Selection with Cross-Validation (Leakage-Free)

**Time:** 8 min | **Difficulty:** Advanced

## What This Solves

Running `GAFeatureSelectionCV` once on the full training set and then evaluating on the same split causes data leakage: the selection has seen all training labels. The correct approach is to run selection inside each CV fold.

:::warning Data leakage is subtle here
If you select features on the full training set before running cross-validation, the feature selection has seen labels from the validation fold — the CV score is optimistically biased.
:::

## Two Approaches

### Approach 1: Use GAFeatureSelectionCV's Built-in CV (Recommended)

`GAFeatureSelectionCV` already runs cross-validation internally. Its `best_score_` is an honest CV estimate of the feature subset, not a train-set score:

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn_genetic import GAFeatureSelectionCV

X, y = make_classification(n_samples=800, n_features=30, n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# CV is applied during selection — this is already leakage-free on X_train
selector = GAFeatureSelectionCV(
    estimator=RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    cv=cv,
    scoring="roc_auc",
    population_size=20,
    generations=15,
    elitism=True,
    verbose=True,
    n_jobs=-1,
    random_state=42,
)
selector.fit(X_train, y_train)

print(f"CV ROC AUC (honest): {selector.best_score_:.4f}")
print(f"Features selected: {selector.support_.sum()}")
```

### Approach 2: Feature Selection Inside a Pipeline

Wrap selection in a `Pipeline` and use `cross_val_score` to get per-fold selection. This is the safest approach but expensive:

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn_genetic import GAFeatureSelectionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

X, y = make_classification(n_samples=800, n_features=30, n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

selector = GAFeatureSelectionCV(
    estimator=RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
    cv=inner_cv,
    scoring="roc_auc",
    population_size=15,
    generations=10,
    verbose=False,
    n_jobs=-1,
    random_state=42,
)

# Note: GAFeatureSelectionCV implements the sklearn Estimator interface
# and can be used in cross_val_score for nested CV
scores = cross_val_score(selector, X_train, y_train, cv=outer_cv, scoring="roc_auc", n_jobs=1)
print(f"Nested CV ROC AUC: {scores.mean():.4f} ± {scores.std():.4f}")
```

## Key Points

- **Approach 1 is sufficient for most use cases**: `GAFeatureSelectionCV` already does CV internally. Fit on `X_train`, evaluate on `X_test`.
- **Approach 2 (nested CV)**: Only needed when you want an unbiased estimate of the *selection process* itself, not just the selected features. Very expensive.
- **Never fit `selector` on all data**: Always hold out a test set before calling `.fit()`.

## See Also

- [Feature Selection Tutorial](../../tutorials/feature-selection) — end-to-end workflow
- [Cross-Validation in Hyperparameter Search](../../guide/understand-cv) — CV concepts and common pitfalls
- [Common Hyperparameter Tuning Mistakes](../../guide/common-mistakes) — data leakage section
