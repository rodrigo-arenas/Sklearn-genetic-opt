---
title: Advanced Random Forest Tuning
description: Tour of advanced optimizer controls — smart initialization, warm starts, diversity control, fitness sharing, local search, and adaptive schedules — applied to a RandomForestClassifier on breast cancer data.
---

# Advanced Random Forest Tuning

This example is a guided tour of advanced optimizer controls. We tune a `RandomForestClassifier`, inspect telemetry, compare against a random-search baseline, and apply the same ideas to feature selection.

## Problem Setup

```python
import warnings
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from scipy.stats import randint

from sklearn_genetic import (
    EvolutionConfig, GAFeatureSelectionCV, GASearchCV,
    OptimizationConfig, PopulationConfig, RuntimeConfig,
)
from sklearn_genetic.callbacks import ConsecutiveStopping, DeltaThreshold, TimerStopping
from sklearn_genetic.schedules import ExponentialAdapter, InverseAdapter
from sklearn_genetic.space import Categorical, Continuous, Integer

warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

print(f"Training shape: {X_train.shape}")  # (398, 30)
print(f"Test shape: {X_test.shape}")       # (171, 30)
```

## Baseline Model

Train a plain random forest to give a reference point.

```python
def evaluate_classifier(estimator, X_eval, y_eval):
    predictions = estimator.predict(X_eval)
    probabilities = estimator.predict_proba(X_eval)[:, 1]
    return {
        "accuracy": accuracy_score(y_eval, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_eval, predictions),
        "roc_auc": roc_auc_score(y_eval, probabilities),
    }


baseline = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)
baseline.fit(X_train, y_train)
baseline_metrics = evaluate_classifier(baseline, X_test, y_test)
print(baseline_metrics)
# {'accuracy': 0.9357, 'balanced_accuracy': 0.9298, 'roc_auc': 0.9913}
```

## Search Space

```python
param_grid = {
    "n_estimators": Integer(40, 140),
    "max_depth": Integer(2, 12),
    "min_samples_split": Integer(2, 12),
    "min_samples_leaf": Integer(1, 8),
    "max_features": Categorical(["sqrt", "log2", None]),
    "ccp_alpha": Continuous(0.0, 0.03),
}
```

## Configure GASearchCV

This configuration demonstrates:

- `PopulationConfig(initializer="smart")` — smarter initial population using estimator defaults, stratified categorical choices, and Latin hypercube sampling for numeric dimensions.
- `warm_start_configs` — inject a known reasonable configuration into the first population.
- `RuntimeConfig(parallel_backend="auto")` — let the estimator decide whether to parallelize candidate evaluation or cross-validation.
- `OptimizationConfig(local_search=True)` — short refinement around best candidates at the end.
- `OptimizationConfig(diversity_control=True)` — increases mutation pressure and injects random candidates when the population collapses too early.
- `OptimizationConfig(fitness_sharing=True)` — reduces crowding so similar candidates do not dominate selection.
- Adaptive schedules — crossover and mutation probabilities evolve over generations.

```python
callbacks = [
    ConsecutiveStopping(generations=10, metric="fitness_best"),
    TimerStopping(total_seconds=240),
]

ga_search = GASearchCV(
    estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    evolution_config=EvolutionConfig(
        population_size=20,
        generations=15,
        crossover_probability=ExponentialAdapter(initial_value=0.8, end_value=0.4, adaptive_rate=0.15),
        mutation_probability=InverseAdapter(initial_value=0.25, end_value=0.05, adaptive_rate=0.2),
        tournament_size=3,
        elitism=True,
        keep_top_k=3,
    ),
    population_config=PopulationConfig(
        initializer="smart",
        warm_start_configs=[{
            "n_estimators": 100,
            "max_depth": 6,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "ccp_alpha": 0.0,
        }],
    ),
    runtime_config=RuntimeConfig(
        n_jobs=-1,
        parallel_backend="auto",
        use_cache=True,
        verbose=True,
        return_train_score=False,
    ),
    optimization_config=OptimizationConfig(
        local_search=True,
        local_search_top_k=2,
        local_search_steps=1,
        local_search_radius=0.2,
        diversity_control=True,
        diversity_threshold=0.35,
        diversity_stagnation_generations=3,
        diversity_mutation_boost=1.8,
        random_immigrants_fraction=0.15,
        fitness_sharing=True,
        sharing_radius=0.35,
        sharing_alpha=1.0,
    ),
)

ga_search.fit(X_train, y_train, callbacks=callbacks)
```

## Inspect Results and Telemetry

```python
print("Best CV ROC AUC:", round(ga_search.best_score_, 4))
pprint(ga_search.best_params_)

ga_metrics = evaluate_classifier(ga_search, X_test, y_test)
pd.DataFrame([baseline_metrics, ga_metrics], index=["baseline", "ga_search"])
```

```python
# Evaluation mechanics summary
print(ga_search.fit_stats_)
# {'evaluated_candidates': 462, 'unique_candidates': 460, 'cache_hits': 2,
#  'random_immigrants': 36, 'local_refinement_candidates': 2, ...}
```

```python
# Per-generation telemetry
history = pd.DataFrame(ga_search.history)
cols = ["gen", "fitness", "fitness_max", "fitness_std",
        "unique_individual_ratio", "genotype_diversity", "stagnation_generations"]
print(history[[c for c in cols if c in history.columns]].tail())
```

### Visualizing Dynamics

```python
import matplotlib.pyplot as plt

# Fitness chart
ax = history.plot(
    x="gen", y=["fitness_best", "fitness_max", "fitness"],
    marker="o", figsize=(8, 4),
)
ax.set_title("Fitness over generations")
ax.set_xlabel("Generation")
ax.set_ylabel("ROC AUC")
plt.show()

# Diversity chart
ax = history.plot(
    x="gen", y=["unique_individual_ratio", "genotype_diversity"],
    marker="o", figsize=(8, 4),
)
ax.set_title("Population diversity over generations")
plt.show()
```

:::tip Reading diversity charts
If diversity curves drop to zero early while fitness stalls, the search is over-exploiting one region. Try `diversity_control=True`, a larger `random_immigrants_fraction`, or `fitness_sharing=True`.
:::

## Compare With RandomizedSearchCV

```python
randomized_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
    param_distributions={
        "n_estimators": randint(40, 141),
        "max_depth": randint(2, 13),
        "min_samples_split": randint(2, 13),
        "min_samples_leaf": randint(1, 9),
        "max_features": ["sqrt", "log2", None],
        "ccp_alpha": np.linspace(0.0, 0.03, 20),
    },
    n_iter=12,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

randomized_search.fit(X_train, y_train)
randomized_metrics = evaluate_classifier(randomized_search, X_test, y_test)

pd.DataFrame(
    [baseline_metrics, randomized_metrics, ga_metrics],
    index=["baseline", "randomized_search", "ga_search"],
)
```

## Feature Selection With GAFeatureSelectionCV

Apply the same optimizer ideas to feature selection. The individual becomes a binary mask instead of a hyperparameter vector.

```python
feature_selector = GAFeatureSelectionCV(
    estimator=RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1,
        **ga_search.best_params_,    # reuse best params from above
    ),
    scoring="roc_auc",
    cv=cv,
    max_features=10,
    evolution_config=EvolutionConfig(population_size=14, generations=10),
    population_config=PopulationConfig(initializer="smart"),
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
        local_search_radius=0.15,
        diversity_control=True,
        diversity_threshold=0.30,
        random_immigrants_fraction=0.10,
        fitness_sharing=True,
        sharing_radius=0.40,
    ),
)

feature_selector.fit(X_train, y_train, callbacks=[TimerStopping(total_seconds=120)])

selected_features = X_train.columns[feature_selector.support_]
print(f"Selected {len(selected_features)} features: {selected_features.tolist()}")
```

```python
# Compare all four models
selector_metrics = evaluate_classifier(feature_selector, X_test, y_test)
print(classification_report(y_test, feature_selector.predict(X_test), target_names=data.target_names))

pd.DataFrame(
    [baseline_metrics, randomized_metrics, ga_metrics, selector_metrics],
    index=["baseline", "randomized_search", "ga_search", "feature_selector"],
)
```

## Practical Takeaways

- Start with `PopulationConfig(initializer="smart")` — it usually gives better early coverage.
- Use `fit_stats_` to understand the cost: evaluated candidates, unique candidates, cache hits, skipped invalid masks, and cross-validation calls.
- Use `history` to decide whether the optimizer is exploring enough. Low diversity plus stalled fitness suggests stronger mutation, fitness sharing, random immigrants, or a larger population.
- Use `OptimizationConfig(local_search=True)` when the GA already finds good regions and you want a final exploitation pass.
- Keep a sklearn baseline nearby. It is the simplest way to check whether the advanced optimizer improves quality enough to justify extra search time.

## See Also

- [Advanced Optimizer Control](../guide/advanced-optimizer-control) — full guide to all optimizer controls
- [Adaptive Schedules](../guide/adapters) — how `ExponentialAdapter` and `InverseAdapter` work
- [GAFeatureSelectionCV API](../api/gafeatureselectioncv) — feature selection API reference
