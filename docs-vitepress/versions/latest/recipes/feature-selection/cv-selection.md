---
title: "Feature Selection with Per-Fold Cross-Validation"
description: "Recipe to evaluate feature selection with a holdout or outer cross-validation folds."
---

:::warning Development version
You are reading the **latest (development)** docs. For stable documentation, see [stable](/stable/).
:::

# Feature Selection with Cross-Validation (Leakage-Free)

**Time:** 8 min | **Difficulty:** Advanced

## What This Solves

Selecting features on all rows before cross-validation leaks labels from each validation fold. Split off a test set first, or run the selector inside each outer CV fold when estimating the whole selection procedure.

:::warning Data leakage is subtle here
If you select features on the full training set before running cross-validation, the feature selection has seen labels from the validation fold — the CV score is optimistically biased.
:::

## Two Approaches

### Approach 1: Use GAFeatureSelectionCV's Built-in CV and a Holdout

`GAFeatureSelectionCV` uses internal CV to compare candidate masks. The highest search CV score is useful for choosing a mask, but optimistic as a performance estimate because many masks were compared. Evaluate the selected model on the untouched test set:

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn_genetic import GAFeatureSelectionCV

X, y = make_classification(n_samples=800, n_features=30, n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# CV is applied during selection; X_test remains untouched.
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

print(f"Best search CV ROC AUC: {np.nanmax(selector.cv_results_['mean_test_score']):.4f}")
print(f"Features selected: {selector.support_.sum()}")
print(f"Holdout ROC AUC: {selector.score(X_test, y_test):.4f}")
```

### Approach 2: Nested CV Around Feature Selection

Pass the selector to `cross_val_score` so it refits feature selection within each outer training fold. This estimates the selection procedure, but is expensive:

```python
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

- **Approach 1 is sufficient for most use cases**: `GAFeatureSelectionCV` does CV internally to select features. Fit on `X_train`, then report performance on untouched `X_test`.
- **Approach 2 (nested CV)**: Only needed when you want an unbiased estimate of the *selection process* itself, not just the selected features. Very expensive.
- **Never fit `selector` on all data**: Always hold out a test set before calling `.fit()`.

## See Also

- [Feature Selection Tutorial](../../tutorials/feature-selection) — end-to-end workflow
- [Cross-Validation in Hyperparameter Search](../../guide/understand-cv) — CV concepts and common pitfalls
- [Common Hyperparameter Tuning Mistakes](../../guide/common-mistakes) — data leakage section
