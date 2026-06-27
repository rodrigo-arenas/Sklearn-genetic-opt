---
title: "Tune HistGradientBoostingClassifier with Genetic Algorithms"
description: "Copy-paste recipe to tune scikit-learn's HistGradientBoostingClassifier including max_leaf_nodes and l2_regularization."
---

:::warning Development version
You are reading the **latest (development)** docs. For stable documentation, see [stable](/stable/).
:::

# Tune HistGradientBoostingClassifier

**Time:** 5 min | **Difficulty:** Beginner

## What This Solves

`HistGradientBoostingClassifier` is scikit-learn's fast native gradient booster (no extra install). Its key param is `max_leaf_nodes`, not `max_depth` — this recipe shows the right search space.

## Recipe

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn_genetic import GASearchCV, EvolutionConfig, RuntimeConfig
from sklearn_genetic.space import Continuous, Integer

X, y = make_classification(
    n_samples=2000, n_features=20, n_informative=10, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

param_grid = {
    "max_iter":          Integer(50, 400),
    "max_leaf_nodes":    Integer(15, 255),    # primary complexity control
    "max_depth":         Integer(3, 10),
    "min_samples_leaf":  Integer(10, 50),
    "learning_rate":     Continuous(0.01, 0.3, distribution="log-uniform"),
    "l2_regularization": Continuous(0.0, 1.0),
    "max_features":      Continuous(0.3, 1.0),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

ga = GASearchCV(
    estimator=HistGradientBoostingClassifier(random_state=42),
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    evolution_config=EvolutionConfig(
        population_size=20,
        generations=12,
        elitism=True,
    ),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=True),
    random_state=42,
)
ga.fit(X_train, y_train)

print("Best ROC AUC (CV):", round(ga.best_score_, 4))
print("Best params:", ga.best_params_)
```

## Key Points

- **`max_leaf_nodes` vs `max_depth`**: HistGBM is leaf-wise. `max_leaf_nodes` is the primary complexity knob. Setting both constrains to the stricter of the two.
- **`min_samples_leaf=20` default**: Much higher than the classic GBM default of 1 — helps regularization. Search 10–100.
- **Built-in missing value handling**: No imputation step needed in a Pipeline.
- **No `n_jobs` on estimator needed**: HistGBM uses OpenMP threading managed at the C level; it doesn't conflict with sklearn's joblib the way XGBoost does.

## See Also

- [Gradient Boosting Hyperparameter Tuning](../../tutorials/tune-gradient-boosting) — HistGBM vs classic GBM comparison
- [XGBoost Classifier](./xgboost-classifier) — external library alternative
- [LightGBM Classifier](./lightgbm-classifier) — fastest external library option
