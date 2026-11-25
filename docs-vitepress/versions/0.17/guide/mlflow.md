---
title: MLflow Integration
description: Log sklearn-genetic-opt experiments to MLflow using MLflowConfig — track hyperparameter search runs, compare experiments, and save fitted models.
---

**Estimated reading time:** 3 minutes
**Difficulty:** Intermediate  
**Prerequisites:** [Getting Started with GASearchCV](./basic-usage), MLflow installed (`pip install sklearn-genetic-opt[mlflow]`)

# MLflow Integration

`sklearn-genetic-opt` has built-in integration with MLflow via `MLflowConfig`. It records a run for each freshly evaluated candidate. Group these runs under an explicit parent to compare configurations from one search in the MLflow UI.

## Track a search

Install the optional tracking dependency:

```bash
python -m pip install "sklearn-genetic-opt[mlflow]"
```

Use [the runnable MLflow recipe](../recipes/integrations/mlflow-logging) to
track a small tree search in SQLite. For a tracking server, use its URL in
place of the recipe's SQLite URI. Pass the resulting `MLflowConfig` as
`log_config` when constructing the search. Set `run_name` to the label
you want on **candidate** runs.

## Organize candidate runs and the winner

- **Search summary:** Open a parent with `mlflow.start_run(...)` around
  `search.fit(...)` in the same experiment. Log the best parameters, holdout
  metrics and any parent tags explicitly inside that block. Without an
  active parent, each freshly evaluated candidate gets a separate run.
- **Candidate comparisons:** The logger records parameters and one metric,
  `score`, the mean search CV objective. It does not log the generation,
  every fold or every secondary scorer. For per-candidate results across
  metrics, inspect `search.cv_results_`.
- **Model artifacts:** `save_models=True` logs the configured estimator for
  each candidate. Cross-validation fits separate clones, so do not use that
  artifact as a fitted winner. Leave the flag off to save space; explicitly
  log `search.best_estimator_` after a successful full-data refit. See the
  [MLflow 3 example](../examples/mlflow-tracking) for a working pattern.
- **Parallelism:** Configuring candidate logging disables population-level
  parallel evaluation. If appropriate, use `parallel_backend="cv"` to run
  folds in parallel within each candidate.

## See also

- [MLflowConfig reference](../api/mlflow) for configuration and tags
- [Multi-metric optimization](./multi-metric) to choose the objective sent to MLflow
