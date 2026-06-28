---
title: "Parallelize Hyperparameter Search with Joblib"
description: "Recipe to configure GASearchCV parallelism using n_jobs and loky vs threading backends, with oversubscription avoidance."
---

:::warning Development version
You are reading the **latest (development)** docs. For stable documentation, see [stable](/stable/).
:::

# Parallelize with Joblib

**Time:** 5 min | **Difficulty:** Beginner

## What This Solves

`GASearchCV` uses joblib for parallelism. This recipe shows how to choose the right `n_jobs`, `parallel_backend`, and estimator-level `n_jobs` to avoid CPU oversubscription.

## The Two Parallelism Modes

| `parallel_backend` | What parallelizes | Use when |
|-------------------|-------------------|----------|
| `"auto"` (default) or `"population"` | Evaluates multiple candidates at once | Estimator training is single-threaded, or estimator-level parallelism is disabled |
| `"cv"` | Evaluates multiple CV folds at once | Estimator manages its own threads (XGBoost, LightGBM, CatBoost) |

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
        n_jobs=1,          # ← MUST be 1 for XGBoost/LightGBM/CatBoost
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

## Key Points

- **`n_jobs=-1`**: Uses all available cores.
- **XGBoost/LightGBM/CatBoost**: Always set `n_jobs=1` (or `thread_count=1` for CatBoost) on the estimator. Use `parallel_backend="cv"`.
- **RF/sklearn models**: Use `parallel_backend="population"` with `n_jobs=-1` on the search and `n_jobs=1` on the estimator to parallelize candidate evaluation without nested parallelism.

## See Also

- [Advanced Optimizer Control](../../guide/advanced-optimizer-control) — full parallelism configuration
- [XGBoost Classifier Recipe](../classification/xgboost-classifier) — oversubscription fix
- [LightGBM Classifier Recipe](../classification/lightgbm-classifier) — same fix for LightGBM
