---
title: "Tune CatBoostClassifier with Genetic Algorithms"
description: "Copy-paste recipe to tune CatBoost hyperparameters including bagging_temperature and border_count using GASearchCV."
---

# Tune CatBoostClassifier

**Time:** 5 min | **Difficulty:** Intermediate

## What This Solves

CatBoost has unique hyperparameters (`bagging_temperature`, `border_count`) that don't exist in XGBoost or LightGBM. This recipe shows how to tune them alongside the standard depth/rate/regularization params.

:::warning CPU oversubscription
Set `thread_count=1` on `CatBoostClassifier` and use `parallel_backend="cv"`.
:::

## Recipe

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, train_test_split
from catboost import CatBoostClassifier

from sklearn_genetic import GASearchCV, EvolutionConfig, RuntimeConfig
from sklearn_genetic.space import Continuous, Integer

X, y = make_classification(
    n_samples=2000, n_features=20, n_informative=10, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

param_grid = {
    "iterations":          Integer(50, 400),
    "depth":               Integer(3, 10),
    "learning_rate":       Continuous(0.01, 0.3, distribution="log-uniform"),
    "l2_leaf_reg":         Continuous(1e-3, 10.0, distribution="log-uniform"),
    "bagging_temperature": Continuous(0.0, 1.0),
    "border_count":        Integer(32, 255),
    "random_strength":     Continuous(0.0, 1.0),
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

ga = GASearchCV(
    estimator=CatBoostClassifier(
        random_seed=42,
        thread_count=1,   # ← required: prevent CPU oversubscription
        verbose=0,
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
        parallel_backend="cv",
        verbose=True,
    ),
    random_state=42,
)
ga.fit(X_train, y_train)

print("Best ROC AUC (CV):", round(ga.best_score_, 4))
print("Best params:", ga.best_params_)
```

## Key Points

- **`bagging_temperature`**: Controls Bayesian bootstrap intensity (0 = no bootstrap, 1 = standard). Different from XGBoost's `subsample`.
- **`border_count`**: Number of splits for numerical features. 128 is a good default; search 32–255 for datasets where feature quantization matters.
- **`random_strength`**: Adds noise to splits during tree growth — prevents early overfitting on small datasets.
- **`thread_count=1` not `n_jobs=1`**: CatBoost uses `thread_count` instead of `n_jobs`.

## See Also

- [CatBoost Hyperparameter Tuning](../../tutorials/tune-catboost) — full tutorial
- [XGBoost Classifier](./xgboost-classifier) — alternative gradient booster
- [LightGBM Classifier](./lightgbm-classifier) — fastest training, leaf-wise growth
