---
title: "Tune Hyperparameters for Balanced Accuracy"
description: "Recipe to optimize hyperparameters for balanced accuracy in imbalanced classification, with class_weight as a search parameter."
---

# Tune for Balanced Accuracy

**Time:** 5 min | **Difficulty:** Beginner

## What This Solves

Standard accuracy collapses to the majority class on imbalanced data. Balanced accuracy averages per-class recall, so a model that predicts everything as the majority class scores 0.5 — not 0.95. This recipe shows the setup.

## Recipe

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn_genetic import GASearchCV, EvolutionConfig, RuntimeConfig
from sklearn_genetic.space import Categorical, Continuous, Integer

# Severe 95/5 imbalance
X, y = make_classification(
    n_samples=2000,
    n_features=20,
    n_informative=8,
    weights=[0.95, 0.05],
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
    "min_samples_leaf": Integer(1, 20),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

ga = GASearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    scoring="balanced_accuracy",
    cv=cv,
    evolution_config=EvolutionConfig(population_size=20, generations=15, elitism=True),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=True),
    random_state=42,
)
ga.fit(X_train, y_train)

pred = ga.predict(X_test)
print("Best CV balanced accuracy:", round(ga.best_score_, 4))
print("Test balanced accuracy:", round(balanced_accuracy_score(y_test, pred), 4))
print("Test standard accuracy:", round((pred == y_test).mean(), 4))
print("\nClassification report:")
print(classification_report(y_test, pred))
print("\nBest class_weight:", ga.best_params_["class_weight"])
```

## Key Points

- **`balanced_accuracy`**: Macro-average of per-class recall. 0.5 = predicting all one class, 1.0 = perfect.
- **`class_weight` as a param**: For severe imbalance, `"balanced"` typically wins. Let the search confirm.
- **`StratifiedKFold` mandatory**: Without stratification, some folds may have zero minority class samples — CV becomes useless.
- **Compare to standard accuracy**: Always report both — a large gap confirms your imbalance handling is working.

## See Also

- [Hyperparameter Tuning for Imbalanced Datasets](../../tutorials/imbalanced-classification) — full tutorial with confusion matrices
- [Tune for F1 Score](./f1-binary) — alternative imbalance metric
- [Tune for ROC-AUC](./roc-auc) — ranking metric, threshold-agnostic
