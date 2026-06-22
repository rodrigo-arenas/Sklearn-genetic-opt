---
title: MLflow API
description: API reference for MLflowCallback and MLflow integration utilities.
---

# MLflow API

```python
from sklearn_genetic.callbacks import MLflowCallback
```

Requires MLflow: `pip install sklearn-genetic-opt[mlflow]`.

## MLflowCallback

Logs per-generation metrics to an active MLflow run.

```python
MLflowCallback(
    log_params=True,
    log_metrics=True,
    prefix="",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_params` | bool | `True` | Log best parameters to the MLflow run at the end of the search |
| `log_metrics` | bool | `True` | Log per-generation metrics as MLflow step metrics |
| `prefix` | str | `""` | Prefix to add to all logged metric names |

### Logged metrics (per generation)

| Metric | Description |
|--------|-------------|
| `fitness` | Mean population fitness |
| `fitness_best` | Best fitness found so far |
| `fitness_std` | Standard deviation of fitness |
| `genotype_diversity` | Population diversity (0–1) |
| `stagnation_generations` | Consecutive stagnant generations |

### Usage

```python
import mlflow
from sklearn_genetic.callbacks import MLflowCallback

mlflow.set_experiment("my-experiment")

with mlflow.start_run():
    search.fit(X_train, y_train, callbacks=[MLflowCallback()])

    # Log additional artifacts
    mlflow.log_metric("holdout_roc_auc", roc_auc_score(y_test, search.predict_proba(X_test)[:, 1]))
    mlflow.sklearn.log_model(search.best_estimator_, "best_model")
```

## See Also

- [MLflow Integration Guide](../guide/mlflow) — full tutorial
- [Callbacks](./callbacks) — all available callbacks
