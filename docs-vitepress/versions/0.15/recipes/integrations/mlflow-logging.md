---
title: "Log Every Hyperparameter Search Candidate to MLflow"
description: "Recipe to use MlflowCallback to log every GASearchCV candidate as a child run, with parameter and metric tracking."
---

# Log Every Candidate to MLflow

**Time:** 5 min | **Difficulty:** Beginner

## What This Solves

You want every evaluated hyperparameter configuration tracked in MLflow — not just the best one. `MlflowCallback` logs each candidate as a child run automatically.

:::info Prerequisites
```bash
pip install sklearn-genetic-opt[mlflow]
```
Start the MLflow tracking server (or use the default local `./mlruns` directory).
:::

## Recipe

```python
import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn_genetic import GASearchCV, EvolutionConfig, RuntimeConfig
from sklearn_genetic.callbacks import MlflowCallback
from sklearn_genetic.space import Categorical, Continuous, Integer

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

param_grid = {
    "n_estimators": Integer(50, 300),
    "max_depth":    Integer(3, 20),
    "max_features": Continuous(0.1, 1.0),
    "class_weight": Categorical([None, "balanced"]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

ga = GASearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    evolution_config=EvolutionConfig(population_size=15, generations=10, elitism=True),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=True),
    random_state=42,
)

# Start an MLflow parent run
with mlflow.start_run(run_name="rf-genetic-search"):
    mlflow.log_params({
        "population_size": 15,
        "generations": 10,
        "scoring": "roc_auc",
    })

    ga.fit(
        X_train, y_train,
        callbacks=[MlflowCallback()],  # logs each candidate as a child run
    )

    # Log final best result
    mlflow.log_metric("best_cv_roc_auc", ga.best_score_)
    mlflow.log_metric("test_roc_auc", ga.score(X_test, y_test))
    mlflow.log_params(ga.best_params_)

print("Best CV ROC AUC:", round(ga.best_score_, 4))
print("MLflow UI: mlflow ui --port 5000")
```

## What Gets Logged

Each child run captures:
- All hyperparameter values tried
- The CV score for that candidate
- The generation number

The parent run captures:
- Your search config (population, generations, scoring)
- Best CV score and test score
- Best hyperparameter values

## Key Points

- **Nested runs**: `MlflowCallback` creates child runs under the active parent `mlflow.start_run()`. Start a parent run first.
- **No parent run**: If you call `ga.fit()` outside a `with mlflow.start_run()` block, the callback creates a new top-level run per candidate.
- **MLflow UI**: `mlflow ui` in the project directory opens the experiment browser at `http://localhost:5000`.

## See Also

- [MLflow Integration Guide](../../guide/mlflow) — full MLflow setup with experiment names and tracking URI
- [MLflow Experiment Tracking Example](../../examples/mlflow-tracking) — annotated example
