---
title: "Tune Imputer Strategy as a Hyperparameter"
description: "Recipe to search over mean, median, and most_frequent imputation strategies inside a scikit-learn Pipeline."
---

# Tune Imputer Strategy as a Hyperparameter

**Time:** 5 min | **Difficulty:** Beginner

## What This Solves

The right imputation strategy depends on your data distribution. Instead of picking `mean` or `median` by hand, include `strategy` as a search parameter — the genetic algorithm will evaluate which one actually helps downstream performance.

## Recipe

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn_genetic import GASearchCV, EvolutionConfig, RuntimeConfig
from sklearn_genetic.space import Categorical, Integer

X, y = load_breast_cancer(return_X_y=True)

# Inject 20% missing values to simulate a real dataset
rng = np.random.default_rng(42)
mask = rng.random(X.shape) < 0.2
X = X.astype(float)
X[mask] = np.nan

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipe = Pipeline([
    ("imputer", SimpleImputer()),
    ("clf",     RandomForestClassifier(random_state=42, n_jobs=-1)),
])

param_grid = {
    "imputer__strategy": Categorical(["mean", "median", "most_frequent"]),
    "clf__n_estimators": Integer(50, 200),
    "clf__max_depth":    Integer(3, 15),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

ga = GASearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    evolution_config=EvolutionConfig(population_size=15, generations=10, elitism=True),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=True),
    random_state=42,
)
ga.fit(X_train, y_train)

print("Best ROC AUC (CV):", round(ga.best_score_, 4))
print("Best imputer strategy:", ga.best_params_["imputer__strategy"])
print("Best estimator params:", {k: v for k, v in ga.best_params_.items() if k.startswith("clf")})
```

## Key Points

- **`imputer__strategy` prefix**: The step name is `"imputer"`, so the prefix is `imputer__`.
- **Imputer fitted inside CV folds**: Because it's in the `Pipeline`, the imputer doesn't see the validation fold's values during fitting — no leakage.
- **`most_frequent`**: The only strategy that works for categorical features encoded as strings/integers.

## See Also

- [Tuning scikit-learn Pipelines](../../guide/pipeline-tuning) — full guide
- [ColumnTransformer Pipeline](./column-transformer) — mixed-type features
- [Common Hyperparameter Tuning Mistakes](../../guide/common-mistakes) — imputation leakage pitfall
