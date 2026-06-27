---
title: "Write a Custom Scoring Function for Hyperparameter Search"
description: "Recipe to define a custom scorer using make_scorer or a callable for metrics not in sklearn's built-in scoring list."
---

:::warning Development version
You are reading the **latest (development)** docs. For stable documentation, see [stable](/stable/).
:::

# Write a Custom Scoring Function

**Time:** 8 min | **Difficulty:** Advanced

## What This Solves

`GASearchCV` accepts any callable as `scoring`. Use a custom scorer when you need: a metric not in sklearn's list, a business metric (revenue, cost), or a metric that depends on something beyond `(y_true, y_pred)`.

## Pattern 1: `make_scorer` (Recommended for Standard Metrics)

```python
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn_genetic import GASearchCV, EvolutionConfig, RuntimeConfig
from sklearn_genetic.space import Continuous, Integer

X, y = make_classification(n_samples=1000, n_features=20, weights=[0.8, 0.2], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# F2 score: weights recall 2× more than precision (use when false negatives are costly)
f2_scorer = make_scorer(fbeta_score, beta=2)

ga = GASearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid={
        "n_estimators": Integer(50, 200),
        "max_depth":    Integer(3, 15),
        "max_features": Continuous(0.1, 1.0),
    },
    scoring=f2_scorer,   # ← custom scorer
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    evolution_config=EvolutionConfig(population_size=15, generations=10, elitism=True),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=True),
    random_state=42,
)
ga.fit(X_train, y_train)
print(f"Best CV F2: {ga.best_score_:.4f}")
```

## Pattern 2: Callable `(estimator, X, y)` (For Complex Metrics)

Use a callable when you need access to the estimator or probabilities:

```python
from sklearn.metrics import roc_auc_score

def profit_scorer(estimator, X, y):
    """Business metric: $5 profit per true positive, $2 cost per false positive."""
    pred = estimator.predict(X)
    tp = ((pred == 1) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    return 5 * tp - 2 * fp   # higher is better

ga = GASearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid={
        "n_estimators": Integer(50, 200),
        "max_depth":    Integer(3, 15),
        "max_features": Continuous(0.1, 1.0),
    },
    scoring=profit_scorer,   # ← callable
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    evolution_config=EvolutionConfig(population_size=15, generations=10, elitism=True),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=True),
    random_state=42,
)
ga.fit(X_train, y_train)
print(f"Best CV profit score: {ga.best_score_:.2f}")
```

## Pattern 3: Custom Scorer for Unsupervised Metrics

For anomaly detection where `y` is not available during training:

```python
from sklearn.metrics import roc_auc_score
import numpy as np

# y_test available at scoring time for evaluation, but not used in training
def anomaly_scorer(estimator, X, y):
    scores = estimator.score_samples(X)
    return roc_auc_score(y, -scores)   # negative: lower score = more anomalous

# Use with IsolationForest — see the Isolation Forest tutorial
```

## Key Points

- **`make_scorer`**: Wraps a `(y_true, y_pred)` or `(y_true, y_score)` function. Use `needs_proba=True` for probability-based metrics.
- **Callable `(estimator, X, y)`**: Full access to the fitted estimator and the validation fold. Returns a float (higher = better).
- **Always maximized**: Both `make_scorer` and callables are maximized. Use `greater_is_better=False` in `make_scorer` to negate a minimization metric.
- **Thread safety**: Callable scorers are called from parallel worker processes. Avoid shared mutable state.

## See Also

- [Isolation Forest Hyperparameter Tuning](../../tutorials/isolation-forest) — custom scorer from `score_samples`
- [Feature Selection with Custom Scorer](../feature-selection/custom-scorer) — penalized feature count
- [Tune for F1 Score](../metrics/f1-binary) — `make_scorer` for F1
