---
title: "Tune XGBClassifier with Genetic Algorithms"
description: "Copy-paste recipe to tune 9 XGBoost hyperparameters with GASearchCV. Includes the n_jobs=1 / parallel_backend='cv' fix for CPU oversubscription."
---

:::warning Development version
You are reading the **latest (development)** docs. For stable documentation, see [stable](/stable/).
:::

# Tune XGBClassifier

**Time:** 5 min | **Difficulty:** Intermediate

## What This Solves

XGBoost has 9 interacting hyperparameters and its own internal threading. This recipe shows the minimal setup to tune them jointly without oversubscribing your CPUs.

:::warning CPU oversubscription
Set `n_jobs=1` on `XGBClassifier` and `parallel_backend="cv"` on `GASearchCV`. Otherwise XGBoost's internal threads × search workers = CPU overload.
:::

## Recipe

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from sklearn_genetic import GASearchCV, EvolutionConfig, RuntimeConfig
from sklearn_genetic.space import Continuous, Integer

X, y = make_classification(
    n_samples=2000, n_features=20, n_informative=10, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

param_grid = {
    "n_estimators":     Integer(50, 350),
    "max_depth":        Integer(2, 10),
    "min_child_weight": Integer(1, 12),
    "subsample":        Continuous(0.5, 1.0),
    "colsample_bytree": Continuous(0.4, 1.0),
    "learning_rate":    Continuous(0.01, 0.3, distribution="log-uniform"),
    "gamma":            Continuous(1e-4, 1.0, distribution="log-uniform"),
    "reg_alpha":        Continuous(1e-5, 10.0, distribution="log-uniform"),
    "reg_lambda":       Continuous(1e-5, 10.0, distribution="log-uniform"),
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

ga = GASearchCV(
    estimator=XGBClassifier(
        tree_method="hist",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,          # ← required: let the search handle parallelism
    ),
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    evolution_config=EvolutionConfig(
        population_size=15,
        generations=10,
        elitism=True,
        keep_top_k=3,
    ),
    runtime_config=RuntimeConfig(
        n_jobs=-1,
        parallel_backend="cv",   # ← parallelize at fold level, not candidate level
        verbose=True,
    ),
    random_state=42,
)
ga.fit(X_train, y_train)

print("Best ROC AUC (CV):", round(ga.best_score_, 4))
print("Best params:", ga.best_params_)
print("Test ROC AUC:", round(ga.score(X_test, y_test), 4))
```

## Key Points

- **`tree_method="hist"`**: Dramatically faster tree building. Always use it.
- **`n_jobs=1` on estimator + `parallel_backend="cv"`**: XGBoost spawns internal threads. Setting `n_jobs=-1` on the estimator AND parallelizing candidates produces `workers × xgb_threads` threads — far more than your core count.
- **Log-uniform for rates and regularization**: `learning_rate`, `gamma`, `reg_alpha`, `reg_lambda` span orders of magnitude — log-uniform gives equal weight to each decade.

## See Also

- [XGBoost Hyperparameter Tuning](../../tutorials/tune-xgboost) — full tutorial with interaction visualization
- [LightGBM Classifier](./lightgbm-classifier) — often faster, similar API
- [Seed with Known-Good Params](../advanced/warm-start) — warm-start the XGBoost defaults
