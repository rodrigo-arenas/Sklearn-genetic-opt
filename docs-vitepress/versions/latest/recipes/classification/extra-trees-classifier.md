---
title: "Tune ExtraTreesClassifier with Genetic Algorithms"
description: "Copy-paste recipe to tune ExtraTreesClassifier hyperparameters and understand how it differs from RandomForestClassifier."
---

:::warning Development version
You are reading the **latest (development)** docs. For stable documentation, see [stable](/stable/).
:::

# Tune ExtraTreesClassifier

**Time:** 5 min | **Difficulty:** Beginner

## What This Solves

`ExtraTreesClassifier` uses random split thresholds instead of optimal splits. This makes it faster and higher-variance than Random Forest. The right `max_features` value differs from RF — this recipe shows the productive range.

## Recipe

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import ExtraTreesClassifier

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
    "max_depth":         Integer(5, 30),       # ET often needs deeper trees
    "min_samples_split": Integer(2, 20),
    "min_samples_leaf":  Integer(1, 10),
    "max_features":      Continuous(0.1, 1.0), # often benefits from more features than RF
    "bootstrap":         Categorical([True, False]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

ga = GASearchCV(
    estimator=ExtraTreesClassifier(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    evolution_config=EvolutionConfig(
        population_size=15,
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

- **Higher variance, faster training**: Extra Trees uses random splits (not optimal splits), so each tree trains much faster. More trees are needed to get the same bias.
- **Deeper trees are fine**: Random splits don't overfit the same way optimal splits do — `max_depth=None` or 20+ is often productive.
- **`max_features=1.0` can win**: Unlike RF where `sqrt` features is usually optimal, ET often benefits from considering all features (since splits are random anyway).
- **`bootstrap=False` often preferred**: ET's randomness comes from split thresholds, not bootstrap samples — turning off bootstrap gives lower variance.

## See Also

- [Tune RandomForestClassifier](./random-forest-classifier) — optimal splits, standard baseline
- [Random Forest Hyperparameter Tuning](../../tutorials/tune-random-forest) — which params matter
