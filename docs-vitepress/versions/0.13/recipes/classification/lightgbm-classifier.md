---
title: "Tune LGBMClassifier with Genetic Algorithms"
description: "Copy-paste recipe to tune LightGBM hyperparameters including num_leaves and max_depth interaction using GASearchCV."
---

# Tune LGBMClassifier

**Time:** 5 min | **Difficulty:** Intermediate

## What This Solves

LightGBM uses leaf-wise tree growth. Its `num_leaves` and `max_depth` interact: a high `num_leaves` with unconstrained depth overfits. This recipe searches the joint space to find the productive region.

:::warning CPU oversubscription
Same as XGBoost: set `n_jobs=1` on `LGBMClassifier` and use `parallel_backend="cv"`.
:::

## Recipe

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, train_test_split
from lightgbm import LGBMClassifier

from sklearn_genetic import GASearchCV, EvolutionConfig, RuntimeConfig
from sklearn_genetic.space import Continuous, Integer

X, y = make_classification(
    n_samples=2000, n_features=20, n_informative=10, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

param_grid = {
    "n_estimators":      Integer(50, 400),
    "num_leaves":        Integer(20, 150),
    "max_depth":         Integer(3, 12),
    "min_child_samples": Integer(5, 50),
    "subsample":         Continuous(0.5, 1.0),
    "colsample_bytree":  Continuous(0.4, 1.0),
    "learning_rate":     Continuous(0.01, 0.3, distribution="log-uniform"),
    "reg_alpha":         Continuous(1e-5, 10.0, distribution="log-uniform"),
    "reg_lambda":        Continuous(1e-5, 10.0, distribution="log-uniform"),
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

ga = GASearchCV(
    estimator=LGBMClassifier(
        random_state=42,
        n_jobs=1,           # ← required
        verbose=-1,         # suppress LightGBM output
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

- **`num_leaves` is the primary complexity control**: LightGBM grows leaf-wise, not depth-wise. `num_leaves=31` (default) is a good starting lower bound; search up to 150+ for complex datasets.
- **Constrain `max_depth` to prevent runaway depth**: Without a `max_depth` limit, leaf-wise growth can produce extremely deep trees with a high `num_leaves`.
- **`min_child_samples`**: Controls regularization. Higher values (20–50) prevent overfitting on small datasets; lower values (5–10) are fine with large data.
- **`verbose=-1`**: Suppresses per-tree output that would flood the console during search.

## See Also

- [LightGBM Hyperparameter Tuning](../../tutorials/tune-lightgbm) — full tutorial with num_leaves/max_depth scatter plot
- [XGBoost Classifier](./xgboost-classifier) — depth-wise alternative
- [CatBoost Classifier](./catboost-classifier) — best for categorical columns
