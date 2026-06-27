---
title: "Tune Hyperparameters for F1 Score (Binary Classification)"
description: "Recipe to optimize hyperparameters for F1 score in binary classification, with confusion matrix evaluation."
---

:::warning Development version
You are reading the **latest (development)** docs. For stable documentation, see [stable](/stable/).
:::

# Tune for F1 Score (Binary Classification)

**Time:** 5 min | **Difficulty:** Beginner

## What This Solves

`roc_auc` is threshold-agnostic. When you care about the actual prediction quality at a fixed threshold (especially in imbalanced problems), optimize for F1 directly. This recipe shows the setup and how to evaluate the result.

## Recipe

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn_genetic import GASearchCV, EvolutionConfig, RuntimeConfig
from sklearn_genetic.space import Categorical, Continuous, Integer

X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=8,
    weights=[0.75, 0.25],   # 75/25 imbalance
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

param_grid = {
    "n_estimators":  Integer(50, 300),
    "max_depth":     Integer(3, 20),
    "max_features":  Continuous(0.1, 1.0),
    "class_weight":  Categorical([None, "balanced", "balanced_subsample"]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

ga = GASearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    scoring="f1",          # optimizes for binary F1 (positive class = 1)
    cv=cv,
    evolution_config=EvolutionConfig(population_size=20, generations=15, elitism=True),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=True),
    random_state=42,
)
ga.fit(X_train, y_train)

pred = ga.predict(X_test)
print("Best CV F1:", round(ga.best_score_, 4))
print("\nClassification report:")
print(classification_report(y_test, pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, pred))
print("\nBest params:", ga.best_params_)
```

## Scoring Options for F1

| `scoring=` | What it optimizes |
|-----------|------------------|
| `"f1"` | F1 for the positive class (binary) |
| `"f1_macro"` | Unweighted average F1 across all classes |
| `"f1_weighted"` | F1 weighted by class support |
| `"f1_micro"` | Micro-average F1 (= accuracy for balanced data) |

## Key Points

- **`class_weight` as a param**: For imbalanced data, let the search decide whether weighting helps F1 — often `"balanced"` wins.
- **`StratifiedKFold`**: Critical for imbalanced data — ensures each fold has the right class proportions.
- **F1 vs ROC-AUC**: F1 depends on the 0.5 threshold. If your production threshold differs, optimize ROC-AUC and set the threshold at inference time.

## See Also

- [Tune for ROC-AUC](./roc-auc) — threshold-agnostic alternative
- [Tune for Balanced Accuracy](./balanced-accuracy) — when class balance matters most
- [Hyperparameter Tuning for Imbalanced Datasets](../../tutorials/imbalanced-classification) — full imbalanced tutorial
