---
title: "Tune SVC (Support Vector Classifier) with Genetic Algorithms"
description: "Copy-paste recipe to tune SVC hyperparameters C, kernel, and gamma using GASearchCV with mandatory StandardScaler in a Pipeline."
---

:::warning Development version
You are reading the **latest (development)** docs. For stable documentation, see [stable](/stable/).
:::

# Tune SVC (Support Vector Classifier)

**Time:** 5 min | **Difficulty:** Beginner

## What This Solves

`SVC` with an RBF kernel has a strong `C`–`gamma` interaction: the right `gamma` depends on `C`. Grid search wastes most of its budget in bad (C, gamma) pairs. A genetic search explores the joint space and finds the productive ridge efficiently.

:::warning Always scale features
`SVC` is not scale-invariant. Running it without `StandardScaler` gives meaningless results — always wrap in a `Pipeline`.
:::

:::warning O(n²) training cost
`SVC` scales quadratically with the number of samples. For datasets larger than ~10,000 rows, use `LinearSVC` or `SGDClassifier` instead.
:::

## Recipe

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn_genetic import GASearchCV, EvolutionConfig, RuntimeConfig
from sklearn_genetic.space import Categorical, Continuous

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(probability=True, random_state=42)),
])

param_grid = {
    "svc__C":      Continuous(1e-2, 100.0, distribution="log-uniform"),
    "svc__kernel": Categorical(["rbf", "linear"]),
    "svc__gamma":  Continuous(1e-4, 1.0, distribution="log-uniform"),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

ga = GASearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    evolution_config=EvolutionConfig(
        population_size=15,
        generations=12,
        elitism=True,
        keep_top_k=3,
    ),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=True),
    random_state=42,
)
ga.fit(X_train, y_train)

print("Best ROC AUC (CV):", round(ga.best_score_, 4))
print("Best params:", ga.best_params_)
print("Test ROC AUC:", round(ga.score(X_test, y_test), 4))
```

## Key Points

- **`probability=True`**: Required for `roc_auc` scoring. Adds a calibration step — slightly slower but necessary for probability outputs.
- **`gamma` only matters for RBF**: For `linear` kernel, `gamma` is ignored. The search wastes some evaluations on `(linear, gamma=x)` combos, but it's harmless.
- **Step prefix `svc__`**: Pipeline parameters need the `stepname__paramname` prefix.
- **Log-uniform for C and gamma**: Both span orders of magnitude, so log-uniform sampling is critical.

## See Also

- [SVM Hyperparameter Tuning](../../tutorials/tune-svm) — full tutorial with C–gamma interaction visualization
- [Tune a Preprocessing + Estimator Pipeline](../pipelines/preprocessing-pipeline) — step prefix patterns
- [Choosing the Right Search Space](../../guide/choosing-search-spaces) — when to use log-uniform
