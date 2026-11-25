---
title: "Log Every Hyperparameter Search Candidate to MLflow"
description: "Track evaluated GASearchCV candidates in MLflow with MLflowConfig, group runs under a parent, and log the winner separately."
---

# Log Every Candidate to MLflow

**Time:** 5 min | **Difficulty:** Beginner

## What This Solves

Compare candidate hyperparameters and cross-validation scores under a single
MLflow parent run. The built-in logger is `MLflowConfig`, passed as
`log_config` to the search, not as a callback to `fit`.

:::info Prerequisites
Install the tracking extra with `python -m pip install "sklearn-genetic-opt[mlflow]"`.
This example writes to a local SQLite database and does not need a server.
:::

## Recipe

```python
import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn_genetic import GASearchCV
from sklearn_genetic.mlflow_log import MLflowConfig
from sklearn_genetic.space import Integer

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
tracking_uri = "sqlite:///search-tracking.db"
mlflow.set_tracking_uri(tracking_uri)
logger = MLflowConfig(
    tracking_uri=tracking_uri,
    experiment="tree-tuning",
    run_name="candidate",
)
search = GASearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid={"max_depth": Integer(2, 12)},
    cv=3, scoring="roc_auc",
    population_size=5, generations=1,
    n_jobs=1, random_state=42, verbose=False,
    log_config=logger,
)

with mlflow.start_run(run_name="tree-search") as parent:
    search.fit(X_train, y_train)
    mlflow.log_params(search.best_params_)
    mlflow.log_metric("best_cv_roc_auc", search.best_score_)
    mlflow.log_metric("test_roc_auc", search.score(X_test, y_test))
    print("Parent run ID:", parent.info.run_id)
```

## What Gets Logged

Expand `tree-search` in the MLflow UI. Each child named `candidate` records
the parameters and a metric named `score` (the mean CV objective). Cached
repeats do not create extra runs. Fold scores and generation numbers are not
logged automatically. The parent metrics exist because the example writes
them explicitly.

To inspect the database from the directory where you ran the example:

```bash
mlflow ui --backend-store-uri sqlite:///search-tracking.db
```

## See Also

- [MLflow API](../../api/mlflow) for tags and candidate model artifacts
- [MLflow experiment tracking example](../../examples/mlflow-tracking) to log the refitted winner and its dataset
