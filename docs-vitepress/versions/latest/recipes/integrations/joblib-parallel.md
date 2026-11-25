---
title: "Parallelize Hyperparameter Search with Joblib"
description: "Configure GASearchCV n_jobs and parallel_backend to parallelize candidates, CV folds, or estimator training with joblib."
---

:::warning Development version
You are reading the **latest (development)** docs. For stable documentation, see [stable](/stable/).
:::

# Parallelize with Joblib

**Time:** 5 min | **Difficulty:** Beginner

## What This Solves

`GASearchCV` uses joblib for parallelism. This recipe shows how to choose the right `n_jobs`, `parallel_backend`, and estimator-level `n_jobs` to avoid CPU oversubscription.

## Choose Where to Use Worker Processes or Threads

| Work to parallelize | Search settings | Estimator settings |
| --- | --- | --- |
| Candidate evaluations | `n_jobs>1` or `-1`, `parallel_backend="auto"` or `"population"` | One worker per fit |
| CV folds | `n_jobs>1` or `-1`, `parallel_backend="cv"` | One worker per fit |
| Estimator training | `n_jobs=1` | Multiple workers, if supported |

With candidate parallelism, workers need distinct configurations that require
CV. With fold parallelism, one candidate's folds can run at the same time.
Both options pass the estimator's training configuration through to each fold.

## Recipe: sklearn Estimator (Use `"population"`)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn_genetic import GASearchCV, EvolutionConfig, RuntimeConfig
from sklearn_genetic.space import Continuous, Integer
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

ga = GASearchCV(
    estimator=RandomForestClassifier(
        random_state=42,
        n_jobs=1,          # keep estimator-level parallelism disabled
    ),
    param_grid={
        "n_estimators": Integer(50, 200),
        "max_depth":    Integer(3, 15),
        "max_features": Continuous(0.1, 1.0),
    },
    scoring="roc_auc",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    evolution_config=EvolutionConfig(population_size=20, generations=15),
    runtime_config=RuntimeConfig(
        n_jobs=-1,
        parallel_backend="population",  # parallelize candidate evaluation
    ),
    random_state=42,
)
ga.fit(X_train, y_train)
print("Best ROC AUC:", round(ga.best_score_, 4))
```

## Recipe: External GBM (Use `"cv"`)

```python
from xgboost import XGBClassifier
from sklearn_genetic import GASearchCV, EvolutionConfig, RuntimeConfig
from sklearn_genetic.space import Continuous, Integer
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

ga = GASearchCV(
    estimator=XGBClassifier(
        tree_method="hist",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,          # allocate workers to CV folds
    ),
    param_grid={
        "n_estimators": Integer(50, 200),
        "max_depth":    Integer(2, 8),
        "learning_rate": Continuous(0.01, 0.3, distribution="log-uniform"),
    },
    scoring="roc_auc",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    evolution_config=EvolutionConfig(population_size=15, generations=10),
    runtime_config=RuntimeConfig(
        n_jobs=-1,
        parallel_backend="cv",  # parallelize folds instead of candidates
    ),
    random_state=42,
)
ga.fit(X_train, y_train)
print("Best ROC AUC:", round(ga.best_score_, 4))
```

Use `n_jobs=-1` to request the available workers at the chosen layer. Set a
smaller explicit count to compare settings or share the machine with other
jobs. Check the estimator's worker parameter, such as `n_jobs` for
RandomForest and XGBoost or `thread_count` for CatBoost.

## See Also

- [Advanced Optimizer Control](../../guide/advanced-optimizer-control) — full parallelism configuration
- [Troubleshoot slow fits](../../guide/troubleshooting#search-is-slow)
- [XGBoost Classifier Recipe](../classification/xgboost-classifier) — oversubscription fix
- [LightGBM Classifier Recipe](../classification/lightgbm-classifier) — same fix for LightGBM
