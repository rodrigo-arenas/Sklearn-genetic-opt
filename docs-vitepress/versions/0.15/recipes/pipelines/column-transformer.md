---
title: "Tune a ColumnTransformer Pipeline with Genetic Algorithms"
description: "Recipe to tune hyperparameters in a Pipeline with mixed numeric and categorical features using ColumnTransformer."
---

# Tune a ColumnTransformer Pipeline

**Time:** 8 min | **Difficulty:** Intermediate

## What This Solves

Real datasets have mixed types (numeric + categorical). A `ColumnTransformer` applies different preprocessing per column type. This recipe shows how to tune the estimator inside such a pipeline using the `preprocessor__transformer__param` prefix chain.

## Recipe

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn_genetic import GASearchCV, EvolutionConfig, RuntimeConfig
from sklearn_genetic.space import Categorical, Continuous, Integer

# Synthetic mixed-type data
rng = np.random.default_rng(42)
n = 500
X = pd.DataFrame({
    "age":     rng.integers(18, 80, n).astype(float),
    "income":  rng.normal(50000, 20000, n),
    "score":   rng.uniform(0, 100, n),
    "region":  rng.choice(["north", "south", "east", "west"], n),
    "product": rng.choice(["A", "B", "C"], n),
})
y = (X["income"] > 50000).astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

numeric_features = ["age", "income", "score"]
categorical_features = ["region", "product"]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer,  numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", RandomForestClassifier(random_state=42, n_jobs=-1)),
])

# Param path: "stepname__paramname" for direct params,
#             "preprocessor__transformer__step__param" for nested
param_grid = {
    "clf__n_estimators":  Integer(50, 300),
    "clf__max_depth":     Integer(3, 20),
    "clf__max_features":  Continuous(0.1, 1.0),
    "clf__class_weight":  Categorical([None, "balanced"]),
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
```

## Key Points

- **`clf__n_estimators` prefix**: Top-level pipeline step `"clf"` → use `clf__`.
- **Nested prefix for preprocessor params**: To tune e.g. the imputer strategy inside the numeric transformer: `"preprocessor__num__imputer__strategy"`.
- **`handle_unknown="ignore"`**: Prevents the `OneHotEncoder` from erroring on unseen categories in the test set.
- **DataFrame input**: `ColumnTransformer` accepts DataFrames with string column names.

## See Also

- [Tuning scikit-learn Pipelines](../../guide/pipeline-tuning) — full guide
- [Tune Imputer Strategy](./imputer-strategy) — tuning imputation inside a pipeline
- [Preprocessing + Estimator Pipeline](./preprocessing-pipeline) — simpler single-transformer case
