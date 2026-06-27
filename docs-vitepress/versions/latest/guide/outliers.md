---
title: "Tuning Outlier Detection Models: IsolationForest and LocalOutlierFactor"
description: Use GASearchCV to tune IsolationForest and LocalOutlierFactor hyperparameters with a custom scorer, handling the unique challenges of outlier detection.
---

:::warning Development version
You are reading the **latest (dev)** docs. For the stable version, see [stable](/stable/).
:::

# Tuning Outlier Detection Models: IsolationForest and LocalOutlierFactor

`GASearchCV` supports scikit-learn outlier-detection estimators. These estimators fit on `X` only — they never see `y` during training — so the standard `scoring="roc_auc"` string cannot be used directly. Instead, you define a custom scorer that calls the estimator's anomaly scoring method and compares the result against ground-truth labels.

## Prerequisites

- Completed [Basic Usage](./basic-usage)
- Ground-truth labels for your anomalies (at minimum for evaluation; the estimator itself is unsupervised)

## The Custom Scorer Pattern

The general pattern for any outlier estimator is to pass a callable scorer with the sklearn estimator signature:

```python
from sklearn.metrics import roc_auc_score

def outlier_roc_auc(estimator, X, y):
    # Replace .score_samples with the method appropriate for your estimator
    # (see table below)
    scores = -estimator.score_samples(X)
    return roc_auc_score(y, scores)

scoring = outlier_roc_auc
```

Two details matter:

- **Negate the score** — `score_samples` and `decision_function` return *lower* values for anomalies, but `roc_auc_score` expects *higher* values for the positive class (`y=1` = outlier). Negating aligns them. Omitting the negation causes the GA to silently optimise in the wrong direction.
- **Use the estimator-aware scorer signature** — the callable receives `(estimator, X, y)`, so it can call `score_samples` directly. Do not wrap this pattern with `make_scorer`, which is intended for functions that receive `y_true` and predictions.

### Which scoring method to use

| Estimator | Method | Notes |
|-----------|--------|-------|
| `IsolationForest` | `score_samples` | Always available; independent of `contamination` |
| `LocalOutlierFactor(novelty=True)` | `score_samples` | Requires `novelty=True` at construction |
| `LocalOutlierFactor(novelty=False)` | `negative_outlier_factor_` | Attribute set after fit; only usable on training data |
| `OneClassSVM` | `score_samples` | Available since sklearn 0.24 |
| `EllipticEnvelope` | `score_samples` | Also has `decision_function` |

:::warning LocalOutlierFactor default mode cannot score new data
Standard LOF (`novelty=False`) computes scores using the training neighbourhood. Calling `score_samples` on test data raises an error. Use `LocalOutlierFactor(novelty=True)` when you need to score unseen points — which is always the case inside cross-validation.
:::

## IsolationForest Example

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn_genetic import EvolutionConfig, GASearchCV, PopulationConfig, RuntimeConfig
from sklearn_genetic.plots import plot_cv_scores, plot_score_landscape
from sklearn_genetic.space import Continuous, Integer

# Synthetic dataset: two normal clusters + 5% uniform outliers
X_normal, _ = make_blobs(n_samples=475, centers=2, cluster_std=0.8, random_state=42)
rng = np.random.default_rng(42)
X_outliers = rng.uniform(low=-6, high=6, size=(25, 2))

X = np.vstack([X_normal, X_outliers])
y = np.array([0] * 475 + [1] * 25)   # 0 = normal, 1 = outlier

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


def outlier_roc_auc(estimator, X, y):
    scores = -estimator.score_samples(X)
    return roc_auc_score(y, scores)


search = GASearchCV(
    estimator=IsolationForest(random_state=42),
    random_state=42,
    param_grid={
        "n_estimators":  Integer(50, 300),
        "max_samples":   Continuous(0.05, 0.80),
        "contamination": Continuous(0.01, 0.20),
        "max_features":  Continuous(0.5, 1.0),
    },
    cv=cv,
    scoring=outlier_roc_auc,
    evolution_config=EvolutionConfig(population_size=15, generations=12),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=True),
)

search.fit(X_train, y_train)

print("Best CV ROC AUC:", round(search.best_score_, 4))
print("Best parameters:", search.best_params_)
```

Anomaly searches can look stable even when the scorer is sensitive to the thresholding parameters. The landscape view shows how the best regions depend on `max_samples` and `contamination`:

```python
import matplotlib.pyplot as plt

plot_score_landscape(search, x="max_samples", y="contamination")
plt.show()
```

![IsolationForest score landscape](/images/outliers_isolation_forest_score_landscape.png)

Use the fold-level plot to check whether top candidates are consistently strong or just lucky on one split:

```python
plot_cv_scores(search, top_k=5, label_params=["max_samples", "contamination"])
plt.show()
```

![IsolationForest CV scores](/images/outliers_isolation_forest_cv_scores.png)

## LocalOutlierFactor Example

LOF requires `novelty=True` for use inside cross-validation. The scorer is identical.

```python
from sklearn.neighbors import LocalOutlierFactor

lof_search = GASearchCV(
    estimator=LocalOutlierFactor(novelty=True),
    random_state=42,
    param_grid={
        "n_neighbors":       Integer(5, 50),
        "contamination":     Continuous(0.01, 0.20),
        "leaf_size":         Integer(10, 60),
    },
    cv=cv,
    scoring=outlier_roc_auc,
    evolution_config=EvolutionConfig(population_size=15, generations=10),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=True),
)

lof_search.fit(X_train, y_train)

print("Best CV ROC AUC:", round(lof_search.best_score_, 4))
print("Best parameters:", lof_search.best_params_)
```

## Tips & Gotchas

- **Always negate the score** — `score_samples` is lower for anomalies. Pass `-score_samples` to `roc_auc_score` when `y=1` is the outlier class. Getting this wrong produces a valid-looking search that quietly minimises detection quality.
- **Pass a callable scorer directly** — outlier estimators have no `predict_proba`, and the scorer needs access to the estimator so it can call `score_samples`.
- **Use `StratifiedKFold`** — with 5% outliers, plain `KFold` can produce folds with very few anomalies, making the AUC estimate noisy and the fitness signal unreliable.
- **`contamination` affects `predict`, not `score_samples`** — tuning it improves the hard-decision boundary but not the ranking score. If your downstream use only needs ranking (e.g., a top-K alert list), fix `contamination` based on domain knowledge and remove it from the search space.
- **`score_samples` is independent of `contamination`** for IsolationForest — you can optimise the scorer freely without worrying that contamination is circularly influencing the metric used to select it.
- **LOF with `novelty=False` cannot score new data** — attempting `score_samples` on test data raises an error. Always set `novelty=True` when using LOF inside `GASearchCV`.

## Next Steps

- [Isolation Forest Tutorial](../tutorials/isolation-forest) — full end-to-end walkthrough with contour plots, ROC curve, and a 3-way comparison against baseline and random search.
- [Callbacks](./callbacks) — add early stopping to avoid over-tuning on a noisy anomaly scorer.
- [Troubleshooting](./troubleshooting) — diagnose flat fitness when the search space is too narrow.
