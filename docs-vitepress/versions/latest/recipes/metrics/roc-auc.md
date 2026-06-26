---
title: "Tune Hyperparameters for ROC-AUC"
description: "Recipe to optimize hyperparameters for ROC-AUC scoring with probability calibration guidance."
---

:::warning Development version
You are reading the **latest (development)** docs. For stable documentation, see [stable](/stable/).
:::

# Tune for ROC-AUC

**Time:** 5 min | **Difficulty:** Beginner

## What This Solves

ROC-AUC measures ranking quality across all thresholds — the right metric when you want to rank predictions (fraud scoring, medical risk) rather than classify at a fixed threshold. This recipe shows the minimal setup.

## Recipe

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn_genetic import GASearchCV, EvolutionConfig, RuntimeConfig
from sklearn_genetic.space import Continuous, Integer

X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=10, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

param_grid = {
    "n_estimators":  Integer(50, 300),
    "max_depth":     Integer(2, 8),
    "learning_rate": Continuous(0.01, 0.3, distribution="log-uniform"),
    "subsample":     Continuous(0.5, 1.0),
    "max_features":  Continuous(0.3, 1.0),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

ga = GASearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    evolution_config=EvolutionConfig(population_size=20, generations=15, elitism=True),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=True),
    random_state=42,
)
ga.fit(X_train, y_train)

proba = ga.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, proba)
print("Best CV ROC AUC:", round(ga.best_score_, 4))
print("Test ROC AUC:", round(test_auc, 4))
```

## Estimators That Need `probability=True`

Some estimators don't output probabilities by default. For `roc_auc` scoring, they need probabilities:

```python
from sklearn.svm import SVC
svc = SVC(probability=True, random_state=42)   # ← required for roc_auc
```

Estimators that work without modification: `RandomForestClassifier`, `GradientBoostingClassifier`, `XGBClassifier`, `LGBMClassifier`, `LogisticRegression`.

## Key Points

- **Threshold-agnostic**: ROC-AUC measures ranking quality — not accuracy at the 0.5 threshold. Use it when downstream decisions vary by risk score, not a single cutoff.
- **`predict_proba` required**: `roc_auc` scoring uses probabilities, not hard predictions. `SVC` needs `probability=True`.
- **ROC-AUC range**: 0.5 = random, 1.0 = perfect. For imbalanced data, also check Precision-Recall AUC.

## See Also

- [Tune for F1 Score](./f1-binary) — threshold-dependent alternative
- [Tune for Balanced Accuracy](./balanced-accuracy) — better for severe imbalance
- [Isolation Forest Hyperparameter Tuning](../../tutorials/isolation-forest) — custom ROC-AUC scorer for unsupervised models
