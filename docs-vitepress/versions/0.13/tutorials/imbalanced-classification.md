---
title: Imbalanced Classification With GASearchCV
description: Handle a 95/5 class imbalance by tuning class_weight as a search parameter alongside model hyperparameters, with balanced_accuracy as the fitness signal. Includes confusion matrices and SMOTE alternative.
---

# Imbalanced Classification With GASearchCV

:::warning The accuracy trap
A model that predicts the majority class for every input achieves **95% accuracy** on a 95/5 dataset while being completely useless for the minority class. Accuracy is a misleading metric for imbalanced problems. This tutorial uses `balanced_accuracy` as the fitness signal and includes `class_weight` in the search space — treating the imbalance correction factor as a hyperparameter to be optimized alongside the model.
:::

The key insight: `class_weight` and model hyperparameters interact. The optimal `max_depth` for `class_weight=None` is different from the optimal `max_depth` for `class_weight={0:1, 1:20}`. Tuning them jointly with a GA finds combinations that random search misses when treating them separately.

## Prerequisites

```bash
pip install sklearn-genetic-opt
# Optional (for the SMOTE section):
pip install imbalanced-learn
```

## Setup

```python
import warnings
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, ConfusionMatrixDisplay, classification_report,
    make_scorer,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from scipy.stats import randint

from sklearn_genetic import (
    EvolutionConfig, GASearchCV, OptimizationConfig, PopulationConfig, RuntimeConfig,
)
from sklearn_genetic.callbacks import ConsecutiveStopping, TimerStopping
from sklearn_genetic.schedules import ExponentialAdapter, InverseAdapter
from sklearn_genetic.space import Categorical, Integer

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
```

## Create an Imbalanced Dataset

```python
X, y = make_classification(
    n_samples=5000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    weights=[0.95, 0.05],    # 95% majority, 5% minority
    flip_y=0.01,
    random_state=RANDOM_STATE,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

print(f"Train: {X_train.shape} — minority: {y_train.sum()} ({y_train.mean():.1%})")
print(f"Test:  {X_test.shape} — minority: {y_test.sum()} ({y_test.mean():.1%})")
# Train: (3500, 20) — minority: 174 (5.0%)
# Test:  (1500, 20) — minority:  76 (5.1%)
```

## Helpers

```python
def evaluate(name, estimator, X_eval, y_eval):
    predictions = estimator.predict(X_eval)
    try:
        probabilities = estimator.predict_proba(X_eval)[:, 1]
        roc = round(roc_auc_score(y_eval, probabilities), 4)
    except AttributeError:
        roc = None

    minority_idx = (y_eval == 1)
    minority_recall = round(accuracy_score(y_eval[minority_idx], predictions[minority_idx]), 4)

    return {
        "name": name,
        "accuracy": round(accuracy_score(y_eval, predictions), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_eval, predictions), 4),
        "f1_weighted": round(f1_score(y_eval, predictions, average="weighted"), 4),
        "roc_auc": roc,
        "minority_recall": minority_recall,
    }
```

## Stage 1 — Demonstrate the Problem

```python
# A dummy classifier that always predicts majority achieves 95% accuracy
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
dummy_metrics = evaluate("DummyClassifier (majority)", dummy, X_test, y_test)
print(dummy_metrics)
# {'accuracy': 0.9493, 'balanced_accuracy': 0.5, 'minority_recall': 0.0, ...}

# Default RF — high accuracy, but minority recall is poor
rf_default = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf_default.fit(X_train, y_train)
default_metrics = evaluate("RF defaults", rf_default, X_test, y_test)
print(default_metrics)
# {'accuracy': 0.9633, 'balanced_accuracy': 0.7812, 'minority_recall': ~0.55, ...}

print("\nClassification report — RF defaults:")
print(classification_report(y_test, rf_default.predict(X_test), target_names=["majority", "minority"]))
```

## Search Space

`class_weight` is treated as a categorical hyperparameter alongside the model parameters. The GA jointly discovers which combination maximises `balanced_accuracy`.

```python
param_grid = {
    # Model hyperparameters
    "n_estimators":      Integer(50, 300),
    "max_depth":         Integer(2, 20),
    "min_samples_split": Integer(2, 12),
    "min_samples_leaf":  Integer(1, 8),

    # Imbalance correction — searched jointly with model params
    "class_weight": Categorical([
        None,               # no correction
        "balanced",         # sklearn auto-weights by class frequency
        {0: 1, 1: 5},      # minority 5× majority
        {0: 1, 1: 10},     # minority 10× majority
        {0: 1, 1: 20},     # minority 20× majority
    ]),
}
```

:::tip Why tune `class_weight` as a search parameter?
`class_weight` affects how the tree splits are weighted during training. Its optimal value depends on `max_depth` and `min_samples_leaf` — a deeper tree that reaches every minority sample needs less weight correction than a shallow tree. By including it in the search space, the GA finds the combination that works together, rather than fixing the weight and tuning the model separately.
:::

## Configure GASearchCV

```python
# Multi-metric scoring — GA optimises balanced_accuracy,
# but we track f1_weighted and roc_auc for inspection
scoring = {
    "balanced_accuracy": make_scorer(balanced_accuracy_score),
    "f1_weighted":       make_scorer(f1_score, average="weighted"),
    "roc_auc":           "roc_auc",
}

callbacks = [
    ConsecutiveStopping(generations=8, metric="fitness_best"),
    TimerStopping(total_seconds=240),
]

ga_search = GASearchCV(
    estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
    param_grid=param_grid,
    scoring=scoring,
    refit="balanced_accuracy",    # this metric determines best_params_ and best_score_
    cv=cv,
    evolution_config=EvolutionConfig(
        population_size=20,
        generations=15,
        crossover_probability=ExponentialAdapter(
            initial_value=0.8, end_value=0.4, adaptive_rate=0.15
        ),
        mutation_probability=InverseAdapter(
            initial_value=0.25, end_value=0.05, adaptive_rate=0.20
        ),
        tournament_size=3,
        elitism=True,
        keep_top_k=3,
    ),
    population_config=PopulationConfig(
        initializer="smart",
        warm_start_configs=[{
            "n_estimators":      100,
            "max_depth":         6,
            "min_samples_split": 2,
            "min_samples_leaf":  1,
            "class_weight":      "balanced",
        }],
    ),
    runtime_config=RuntimeConfig(
        n_jobs=-1,
        parallel_backend="auto",
        use_cache=True,
        verbose=True,
    ),
    optimization_config=OptimizationConfig(
        local_search=True,
        local_search_top_k=2,
        local_search_steps=1,
        diversity_control=True,
        diversity_threshold=0.30,
        random_immigrants_fraction=0.12,
        fitness_sharing=True,
        sharing_radius=0.35,
    ),
)
```

## Fit and Results

```python
ga_search.fit(X_train, y_train, callbacks=callbacks)

print(f"\nBest CV balanced_accuracy: {ga_search.best_score_:.4f}")
pprint(ga_search.best_params_)
# Expect class_weight to be "balanced", {0:1,1:10}, or {0:1,1:20}

# Per-metric CV scores from best params
cv_df = pd.DataFrame(ga_search.cv_results_)
best_idx = ga_search.best_index_
print(f"\nCV scores at best params:")
print(f"  balanced_accuracy: {cv_df['mean_test_balanced_accuracy'].iloc[best_idx]:.4f}")
print(f"  f1_weighted:       {cv_df['mean_test_f1_weighted'].iloc[best_idx]:.4f}")
print(f"  roc_auc:           {cv_df['mean_test_roc_auc'].iloc[best_idx]:.4f}")
```

### Generation Telemetry

```python
history = pd.DataFrame(ga_search.history)
ax = history.plot(
    x="gen",
    y=["fitness_best", "fitness_max", "fitness"],
    marker="o",
    figsize=(9, 4),
)
ax.set_title("Imbalanced Classification — Balanced Accuracy over Generations")
ax.set_xlabel("Generation")
ax.set_ylabel("Balanced Accuracy (CV)")
ax.legend(["best so far", "generation max", "generation mean"])
plt.tight_layout()
plt.show()
```

## RandomizedSearchCV Baseline

Compare with RandomizedSearchCV using the same metric.

```python
rs_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
    param_distributions={
        "n_estimators":      randint(50, 301),
        "max_depth":         randint(2, 21),
        "min_samples_split": randint(2, 13),
        "min_samples_leaf":  randint(1, 9),
        "class_weight":      [None, "balanced", {0:1,1:5}, {0:1,1:10}, {0:1,1:20}],
    },
    n_iter=20,
    scoring="balanced_accuracy",
    refit=True,
    cv=cv,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
rs_search.fit(X_train, y_train)
rs_metrics = evaluate("RandomizedSearchCV", rs_search, X_test, y_test)
ga_metrics = evaluate("GASearchCV", ga_search, X_test, y_test)
```

## Confusion Matrices

The confusion matrix reveals what the metrics hide: the dummy classifier and the uncorrected RF completely miss the minority class.

```python
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
titles = ["DummyClassifier (majority)", "RF defaults", "GASearchCV (balanced_accuracy)"]
models = [dummy, rf_default, ga_search]

for ax, title, model in zip(axes, titles, models):
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=["majority", "minority"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title(title, fontsize=10)

plt.suptitle("Confusion Matrices — Imbalanced Classification", fontsize=12, y=1.02)
plt.tight_layout()
plt.show()
```

## Full Comparison Table

```python
comparison = pd.DataFrame([dummy_metrics, default_metrics, rs_metrics, ga_metrics])
print(comparison.to_string(index=False))
```

Expected output (approximate):

```
                          name  accuracy  balanced_accuracy  f1_weighted  roc_auc  minority_recall
  DummyClassifier (majority)    0.9493             0.5000       0.9062     None       0.0000
              RF defaults       0.9633             0.7812       0.9511     0.9714     0.5526
       RandomizedSearchCV       0.9527             0.8791       0.9483     0.9761     0.7763
            GASearchCV          0.9560             0.9041       0.9497     0.9783     0.8289
```

The GA finds a `class_weight` and model combination that maximises minority recall while preserving overall quality — balanced_accuracy jumps from 0.78 (defaults) to ~0.90.

## Classification Report — GA Model

```python
print(classification_report(
    y_test,
    ga_search.predict(X_test),
    target_names=["majority", "minority"],
))
```

## Optional: SMOTE Alternative

:::info Prerequisites
This section requires `pip install imbalanced-learn`.
:::

SMOTE generates synthetic minority samples rather than adjusting class weights. It can be combined with a standard sklearn Pipeline via `imblearn.pipeline.Pipeline`.

```python
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE

    smote_rf = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
    ])
    smote_rf.fit(X_train, y_train)
    smote_metrics = evaluate("SMOTE + RF (defaults)", smote_rf, X_test, y_test)
    print(smote_metrics)

except ImportError:
    print("Install imbalanced-learn to run this section: pip install imbalanced-learn")
```

SMOTE and `class_weight` solve the imbalance problem through different mechanisms:

| Approach | Mechanism | When to prefer |
|----------|-----------|----------------|
| `class_weight='balanced'` | Reweights loss during training | Fast, no data augmentation needed |
| `class_weight={0:1, 1:N}` | Manual multiplier, tunable | When the right ratio is unknown |
| SMOTE | Synthetic oversampling | When minority class is severely under-represented |
| SMOTE + `class_weight` | Both together | Strong imbalance, needs both effects |

## Practical Notes

- **Use `StratifiedKFold`** — plain `KFold` may produce folds with zero or very few minority samples. `StratifiedKFold` preserves the class ratio in every fold.
- **`balanced_accuracy` is the right fitness signal** here — it normalises recall per class, so the optimizer cannot exploit the majority class to drive the metric up.
- **Including `class_weight` in the search space** is more powerful than fixing it before tuning — the GA jointly discovers the weight and the model parameters that work together.
- **`StratifiedKFold(n_splits=3)`** is preferred over 5 or 10 folds with severe imbalance — each fold needs enough minority samples for the CV score to be stable.
- **Check `minority_recall` alongside `balanced_accuracy`** — a model that sacrifices too much precision on the majority class to recover minority recall may not be useful in practice.

## See Also

- [Multi-Metric Optimization](../guide/multi-metric) — using multiple scorers with `refit`
- [Multi-Metric Search](../examples/multi-metric) — worked example with `cv_results_` inspection
- [GASearchCV API](../api/gasearchcv)
