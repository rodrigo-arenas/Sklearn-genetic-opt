---
title: "Tune a Preprocessing + Estimator Pipeline with Genetic Algorithms"
description: "Recipe to tune hyperparameters inside a scikit-learn Pipeline using the stepname__paramname prefix pattern."
---

# Tune a Preprocessing + Estimator Pipeline

**Time:** 5 min | **Difficulty:** Beginner

## What This Solves

When your model needs preprocessing (scaling, encoding), the preprocessing must be inside a `Pipeline` to avoid data leakage. `GASearchCV` works with pipelines using `stepname__paramname` prefixes.

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
    ("svc",    SVC(probability=True, random_state=42)),
])

# Parameters use the "stepname__paramname" prefix
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
    evolution_config=EvolutionConfig(population_size=15, generations=12, elitism=True),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=True),
    random_state=42,
)
ga.fit(X_train, y_train)

print("Best ROC AUC (CV):", round(ga.best_score_, 4))
print("Best params:", ga.best_params_)
print("Test ROC AUC:", round(ga.score(X_test, y_test), 4))
```

## Key Points

- **`stepname__paramname` prefix**: The step name (e.g. `"svc"`) is the key you set in the `Pipeline` constructor list. Use double underscore `__` to separate step from param.
- **Scaler is fitted inside CV folds**: The `Pipeline` prevents fitting the `StandardScaler` on the validation fold — no leakage.
- **`ga.best_estimator_`**: Returns the full fitted pipeline (scaler + estimator). Use it directly for inference.

## Tuning Scaler Choice

You can also search over the scaler type:

```python
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn_genetic.space import Categorical

param_grid = {
    "scaler": Categorical([StandardScaler(), MinMaxScaler(), RobustScaler()]),
    "svc__C": Continuous(0.01, 100.0, distribution="log-uniform"),
    "svc__gamma": Continuous(1e-4, 1.0, distribution="log-uniform"),
}
```

## See Also

- [Tuning scikit-learn Pipelines](../../guide/pipeline-tuning) — complete guide
- [SVC Classifier Recipe](../classification/svm-classifier) — SVC with scaling
- [ColumnTransformer Pipeline](./column-transformer) — mixed-type features
