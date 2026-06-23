---
title: Outlier Detection
description: Use GASearchCV to tune outlier-detection estimators such as IsolationForest and LocalOutlierFactor.
---

# Outlier Detection

`GASearchCV` supports scikit-learn outlier-detection estimators via a scorer adapter. These estimators do not use a class label `y` during fit, so a custom scorer is needed.

## Prerequisites

- Completed [Basic Usage](./basic-usage)

## Using Outlier Estimators

Wrap an outlier estimator with a scorer that evaluates anomaly scores:

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, make_scorer
import numpy as np

from sklearn_genetic import EvolutionConfig, GASearchCV, PopulationConfig, RuntimeConfig
from sklearn_genetic.space import Categorical, Continuous, Integer

# Synthetic dataset with 5% outliers
X, y = make_classification(
    n_samples=500, n_features=10, weights=[0.95, 0.05],
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def outlier_roc_auc(estimator, X, y):
    # Negate: score_samples is lower for anomalies, but roc_auc_score expects
    # higher values for the positive class (y=1 = outlier)
    scores = -estimator.score_samples(X)
    return roc_auc_score(y, scores)

search = GASearchCV(
    estimator=IsolationForest(random_state=42),
    param_grid={
        "n_estimators": Integer(50, 200),
        "max_samples": Continuous(0.5, 1.0),
        "contamination": Continuous(0.01, 0.2),
        "max_features": Continuous(0.3, 1.0),
    },
    cv=3,
    scoring=make_scorer(outlier_roc_auc, needs_proba=False),
    evolution_config=EvolutionConfig(population_size=15, generations=10),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1),
)

search.fit(X_train, y_train)

print("Best parameters:", search.best_params_)
```

## Tips & Gotchas

- Outlier estimators fit on `X` only — pass `y` to `fit` but expect the estimator to ignore it.
- Use `make_scorer` with a custom function that computes the metric from `score_samples` or `decision_function`.
- `contamination` is typically the most impactful parameter to tune.

## Next Steps

- [Isolation Forest Tutorial](../tutorials/isolation-forest) — full end-to-end walkthrough with contour plots, ROC curve, and a 3-way comparison against baseline and random search.
- [Callbacks](./callbacks) — add early stopping to avoid over-tuning contamination.
- [Troubleshooting](./troubleshooting) — diagnose flat fitness when the space is too narrow.
