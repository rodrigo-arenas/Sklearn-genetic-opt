---
title: Multi-Metric Optimization
description: Track multiple scoring metrics during a GASearchCV search and choose which one drives optimization.
---

# Multi-Metric Optimization

`GASearchCV` supports multi-metric evaluation — you can track several metrics simultaneously while optimizing (refitting) on one of them.

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

## Tips & Gotchas

- `best_params_` and `best_score_` always refer to the `refit` metric, not the others.
- The evolutionary search only uses the `refit` metric as the fitness signal. Other metrics are recorded in `cv_results_` but do not influence which individuals survive.
- For binary classification, `make_scorer(f1_score)` defaults to `average="binary"`. For multi-class, pass `average="macro"` or `"weighted"`.

## Next Steps

- [Callbacks](./callbacks) — stop the search when the `refit` metric plateaus.
- [Troubleshooting](./troubleshooting#multi-metric-best_params_-is-not-what-i-expected) — debugging multi-metric searches.
