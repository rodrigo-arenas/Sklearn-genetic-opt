---
title: "Feature Selection on 50+ Columns with Genetic Algorithms"
description: "Copy-paste recipe to run GAFeatureSelectionCV on high-dimensional datasets with a threshold strategy."
---

:::warning Development version
You are reading the **latest (development)** docs. For stable documentation, see [stable](/stable/).
:::

# Feature Selection on 50+ Columns

**Time:** 5 min | **Difficulty:** Intermediate

## What This Solves

High-dimensional datasets (50+ features) can overwhelm simpler feature selectors. `GAFeatureSelectionCV` uses a genetic algorithm to find the compact feature subset that maximizes cross-validated score — more efficient than searching all 2^50+ subsets.

## Recipe

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn_genetic import GAFeatureSelectionCV

# 50 features — only 10 are informative
X, y = make_classification(
    n_samples=800,
    n_features=50,
    n_informative=10,
    n_redundant=15,
    n_repeated=5,
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

selector = GAFeatureSelectionCV(
    estimator=RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    cv=cv,
    scoring="roc_auc",
    population_size=30,   # larger population for 50 binary variables
    generations=20,
    tournament_size=3,
    elitism=True,
    keep_top_k=4,
    verbose=True,
    n_jobs=-1,
    random_state=42,
)
selector.fit(X_train, y_train)

selected = selector.support_
print(f"Features selected: {selected.sum()} / {len(selected)}")
print(f"Selected indices:  {np.where(selected)[0].tolist()}")
print(f"Best CV ROC AUC:   {selector.best_score_:.4f}")

# Apply selection to test set
X_test_selected = X_test[:, selected]
test_score = selector.score(X_test, y_test)
print(f"Test ROC AUC:      {test_score:.4f}")
```

## Key Points

- **`population_size=30`+**: For 50 binary feature bits, the population needs to be large enough to cover the space. Scale up for more features.
- **`generations=20`+**: More features = more generations needed. Consider `ConsecutiveStopping` to end early when stalled.
- **`selector.support_`**: Boolean mask of selected features. Use `X[:, selector.support_]` to apply it.
- **Redundant features**: The genetic search naturally drops highly correlated features that don't add independent signal.

## Add Early Stopping

```python
from sklearn_genetic.callbacks import ConsecutiveStopping

selector.fit(
    X_train, y_train,
    callbacks=[ConsecutiveStopping(generations=5, metric="fitness_best")]
)
```

## See Also

- [Feature Selection Tutorial](../../tutorials/feature-selection) — 3-stage workflow: select, retune, validate
- [Combine Feature Selection + Tuning](./select-then-tune) — two-stage pipeline
- [Feature Selection Methods Compared](../../guide/feature-selection-guide) — wrapper vs filter vs embedded
