---
title: Feature Selection With Noisy Data
description: Use GAFeatureSelectionCV to find a compact, informative feature subset when the input has both real and noise features.
---

# Feature Selection With Noisy Data

This example adds synthetic noise features to the Iris dataset to make feature selection realistic. A useful selector should keep a small subset of the original measurements and discard most noise columns.

## Setup

```python
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn_genetic import (
    EvolutionConfig, GAFeatureSelectionCV, OptimizationConfig, PopulationConfig, RuntimeConfig,
)
from sklearn_genetic.callbacks import ConsecutiveStopping, DeltaThreshold, TimerStopping
from sklearn_genetic.schedules import ExponentialAdapter, InverseAdapter

warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)
```

## Add Noise Features

```python
iris = load_iris(as_frame=True)
X_original = iris.data
y = iris.target

noise = pd.DataFrame(
    rng.normal(size=(X_original.shape[0], 12)),
    columns=[f"noise_{i:02d}" for i in range(12)],
)
X = pd.concat([X_original, noise], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

print(f"Original features: {X_original.shape[1]}")  # 4
print(f"Noise features: {noise.shape[1]}")          # 12
print(f"Total features: {X.shape[1]}")              # 16
```

## Baseline With All Features

```python
def make_svc_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=2.0, gamma="scale", random_state=RANDOM_STATE)),
    ])


def evaluate(estimator, X_eval, y_eval):
    predictions = estimator.predict(X_eval)
    return {
        "accuracy": accuracy_score(y_eval, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_eval, predictions),
    }


baseline = make_svc_pipeline()
baseline.fit(X_train, y_train)
baseline_metrics = evaluate(baseline, X_test, y_test)
print(baseline_metrics)
# {'accuracy': 0.822, 'balanced_accuracy': 0.822}
```

## Configure GAFeatureSelectionCV

`GAFeatureSelectionCV` searches over binary masks — `1` means the feature is selected, `0` means it is excluded.

```python
selector = GAFeatureSelectionCV(
    estimator=make_svc_pipeline(),
    cv=cv,
    scoring="balanced_accuracy",
    max_features=6,           # prefer compact subsets
    evolution_config=EvolutionConfig(
        population_size=20,
        generations=15,
        crossover_probability=ExponentialAdapter(initial_value=0.8, end_value=0.4, adaptive_rate=0.15),
        mutation_probability=InverseAdapter(initial_value=0.30, end_value=0.08, adaptive_rate=0.25),
        tournament_size=3,
        elitism=True,
        keep_top_k=3,
    ),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, parallel_backend="auto", use_cache=True, verbose=True),
    optimization_config=OptimizationConfig(
        local_search=True,
        local_search_top_k=2,
        local_search_steps=1,
        local_search_radius=0.15,
        diversity_control=True,
        diversity_threshold=0.30,
        diversity_stagnation_generations=3,
        diversity_mutation_boost=1.8,
        random_immigrants_fraction=0.10,
        fitness_sharing=True,
        sharing_radius=0.40,
    ),
)

callbacks = [
    DeltaThreshold(threshold=0.001, generations=5, metric="fitness_best"),
    ConsecutiveStopping(generations=7, metric="fitness_best"),
    TimerStopping(total_seconds=90),
]

selector.fit(X_train, y_train, callbacks=callbacks)
```

## Inspect Selected Features

The fitted selector exposes `support_`, just like sklearn feature selectors.

```python
selected_features = X_train.columns[selector.support_]
selected_summary = pd.DataFrame({
    "feature": X_train.columns,
    "selected": selector.support_,
    "kind": ["original" if c in X_original.columns else "noise" for c in X_train.columns],
})

print(f"Selected {len(selected_features)} of {X_train.shape[1]} features")
print(selected_summary[selected_summary["selected"]])
```

## Telemetry

```python
# Evaluation mechanics
print(selector.fit_stats_)
# {'evaluated_candidates': 182, 'cache_hits': 0, 'random_immigrants': 16, ...}

# Per-generation stats
history = pd.DataFrame(selector.history)
telemetry_cols = [
    "gen", "fitness", "fitness_max", "fitness_std",
    "unique_individual_ratio", "genotype_diversity", "stagnation_generations",
]
print(history[[c for c in telemetry_cols if c in history.columns]])
```

```python
import matplotlib.pyplot as plt

ax = history.plot(
    x="gen", y=["fitness_best", "fitness_max", "fitness"],
    marker="o", figsize=(8, 4),
)
ax.set_title("Feature-selection fitness over generations")
ax.set_xlabel("Generation")
ax.set_ylabel("Balanced accuracy")
plt.show()
```

## Compare Results

```python
selector_metrics = evaluate(selector, X_test, y_test)
pd.DataFrame(
    [baseline_metrics, selector_metrics],
    index=["all_features", "selected_features"],
)
# selected_features typically improves on the noisy-input baseline

print(classification_report(y_test, selector.predict(X_test), target_names=iris.target_names))
```

## Practical Notes

- `max_features` is a useful way to make the selection prefer compact solutions.
- If many candidates are skipped as invalid, increase `max_features` or reduce mutation strength.
- If diversity drops quickly, use `diversity_control`, `random_immigrants_fraction`, and `fitness_sharing` before simply adding more generations.
- Always compare with an all-feature baseline. A smaller selected subset is only useful if quality remains acceptable.

## See Also

- [GAFeatureSelectionCV API](../api/gafeatureselectioncv) — full parameter reference
- [Advanced Optimizer Control](../guide/advanced-optimizer-control) — diversity control, fitness sharing, local search
- [Adaptive Schedules](../guide/adapters) — how `ExponentialAdapter` and `InverseAdapter` work
