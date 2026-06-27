---
title: "Tune RandomForestClassifier with Genetic Algorithms"
description: "Copy-paste recipe to tune 7 RandomForestClassifier hyperparameters jointly using GASearchCV, including class_weight as a search parameter."
---

:::warning Development version
You are reading the **latest (development)** docs. For stable documentation, see [stable](/stable/).
:::

# Tune RandomForestClassifier

**Time:** 5 min | **Difficulty:** Beginner

## What This Solves

You have a `RandomForestClassifier` and want to tune `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `bootstrap`, and `class_weight` jointly — something GridSearchCV's combinatorial explosion makes impractical.

## Recipe

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn_genetic import GASearchCV, EvolutionConfig, RuntimeConfig
from sklearn_genetic.space import Categorical, Continuous, Integer

X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=10, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

param_grid = {
    "n_estimators":      Integer(50, 300),
    "max_depth":         Integer(3, 20),
    "min_samples_split": Integer(2, 20),
    "min_samples_leaf":  Integer(1, 10),
    "max_features":      Continuous(0.1, 1.0),
    "bootstrap":         Categorical([True, False]),
    "class_weight":      Categorical([None, "balanced", "balanced_subsample"]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

ga = GASearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    evolution_config=EvolutionConfig(
        population_size=20,
        generations=15,
        elitism=True,
        keep_top_k=3,
    ),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=True),
    random_state=42,
)
ga.fit(X_train, y_train)

print("Best ROC AUC (CV):", round(ga.best_score_, 4))
print("Best params:", ga.best_params_)
print("Test ROC AUC:", round(ga.score(X_test, y_test), 4))
```

## Key Points

- **`class_weight` as a param**: Pass `None`, `"balanced"`, or `"balanced_subsample"` via `Categorical`. The search picks the best weighting for your data automatically.
- **`max_features` as float**: Using `Continuous(0.1, 1.0)` searches the fraction of features more smoothly than the discrete `"sqrt"`/`"log2"` strings.
- **`n_jobs=-1` on estimator + search**: Safe for Random Forest because it uses a shared memory pool (not per-worker threads). For XGBoost/LightGBM, set `n_jobs=1` on the estimator instead.

## Adapt This Recipe

To tune for F1 instead of ROC-AUC:
```python
ga = GASearchCV(..., scoring="f1", ...)
```

To add a time budget:
```python
from sklearn_genetic.callbacks import TimerStopping
ga.fit(X_train, y_train, callbacks=[TimerStopping(total_seconds=120)])
```

## See Also

- [Random Forest Hyperparameter Tuning](../../tutorials/tune-random-forest) — full tutorial with baseline comparison and visualizations
- [Tune for ROC-AUC](../metrics/roc-auc) — scoring recipe
- [Tune for Imbalanced Data](../../tutorials/imbalanced-classification) — using `class_weight` and `balanced_accuracy`
