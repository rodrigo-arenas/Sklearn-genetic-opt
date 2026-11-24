---
title: Multi-Metric Optimization
description: Track multiple scoring metrics in GASearchCV or GAFeatureSelectionCV and choose the optimization objective with or without refitting.
---

:::warning Development version
You are reading the **latest (dev)** docs. For the stable version, see [stable](/stable/).
:::

# Multi-Metric Optimization

`GASearchCV` and `GAFeatureSelectionCV` can record several metrics while using
one metric to guide the evolutionary search.

## Prerequisites

- Completed [Basic Usage](./basic-usage)

## How It Works

Pass a dictionary to `scoring` where each key is a metric name and each value is a scorer. Set `refit` to the metric name that should drive the search:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn_genetic import EvolutionConfig, GASearchCV, PopulationConfig, RuntimeConfig
from sklearn_genetic.space import Categorical, Continuous, Integer

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

scoring = {
    "roc_auc": "roc_auc",
    "f1": make_scorer(f1_score),
    "accuracy": "accuracy",
}

search = GASearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid={
        "n_estimators": Integer(50, 200),
        "max_depth": Integer(2, 10),
        "min_samples_split": Integer(2, 10),
        "max_features": Categorical(["sqrt", "log2"]),
        "ccp_alpha": Continuous(0.0, 0.02),
    },
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    scoring=scoring,
    refit="roc_auc",   # optimize and refit on ROC-AUC
    evolution_config=EvolutionConfig(population_size=20, generations=12),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, use_cache=True),
)

search.fit(X_train, y_train)

print("Best ROC-AUC (refit metric):", round(search.best_score_, 4))
print("Best parameters:", search.best_params_)
```

A standalone callable scorer returns one numeric score per fold. If a
callable produces several metrics, `fit` reports a configuration error.
Supply those metrics as separate entries in `scoring`. Each entry accepts a
registered scorer name or a callable returning one score.
See the [executable example](../tutorials/optimizer-backends#configure-a-combined-scorer).

## Inspecting All Metrics

After fitting, `cv_results_` contains columns for every metric:

```python
import pandas as pd

results = pd.DataFrame(search.cv_results_)
# Columns include: mean_test_roc_auc, mean_test_f1, mean_test_accuracy, rank_test_*
print(results[["params", "mean_test_roc_auc", "mean_test_f1", "mean_test_accuracy"]].head())

# Best configuration by a different metric:
best_by_f1 = results.sort_values("rank_test_f1").iloc[0]
print("Best params by F1:", best_by_f1["params"])
```

The validation columns are available by default. Use `return_train_score=True`
when you also want to compare training and validation scores for each metric.
This adds scoring work per fold. See
[how to measure the cost](./troubleshooting#check-the-cost-of-training-scores).

## Search without refitting

To inspect the candidates before committing to a full-data fit, set
`refit=False` and name the metric that should guide the search in
`objective_metric`. All metrics are still available in `cv_results_`.

```python
from sklearn.base import clone

report_only = clone(search).set_params(refit=False, objective_metric="roc_auc")
report_only.fit(X_train, y_train)
print(report_only.cv_results_dataframe(columns=["mean_test_roc_auc", "mean_test_f1"], top_k=5))
```

`cv_results_dataframe()` sorts candidates by the objective metric, even when
`columns` omits its rank column. Keep the same objective when resuming a
checkpoint so that saved and new scores are comparable.
The search [warns if no candidate has a finite mean for the objective](./troubleshooting#no-candidate-has-a-finite-objective-score).

## Choose a candidate by several metrics

Set `objective_metric` to guide the search and pass a
[refit function](./advanced-optimizer-control#choose-a-refit-candidate-with-a-function)
to select an index from `cv_results_` after search. For example, optimize
balanced accuracy and choose the shallowest tree within one percentage point
of the best score. `best_index_` identifies that row, and `score(X, y)` uses
the named objective on the refitted model.

## Tips & Gotchas

- With a metric name for `refit`, `best_params_` and `best_score_` refer to that metric.
- The evolutionary search uses the named `refit` metric or `objective_metric`
  as its fitness signal. Inspect the other metrics in `cv_results_`.
- For binary classification, `make_scorer(f1_score)` defaults to `average="binary"`. For multi-class, pass `average="macro"` or `"weighted"`.

## Next Steps

- [Callbacks](./callbacks) — stop the search when the `refit` metric plateaus.
- [Troubleshooting](./troubleshooting#multi-metric-best_params_-is-not-what-i-expected) — debugging multi-metric searches.
