---
title: Advanced Optimizer Control
description: Diversity control, fitness sharing, local search, adaptive schedules, and telemetry for difficult search spaces.
---

:::warning Development version
You are reading the **latest (dev)** docs. For the stable version, see [0.13](/versions/0.13/).
:::

# Advanced Optimizer Control

When default settings produce premature convergence or poor search quality, these controls give you finer-grained command over the evolutionary process. They are **optional** — the defaults are conservative and work well for most searches. Add them one at a time when telemetry shows the optimizer needs them.

## Prerequisites

- Completed [Basic Usage](./basic-usage)
- Familiar with [Understanding Cross-Validation](./understand-cv)

## When to Use These Controls

Check `pd.DataFrame(search.history)` after a run. Add controls if you see:

| Symptom | Likely cause | Remedy |
|---------|-------------|--------|
| `unique_individual_ratio` drops to near 0 in a few generations | Population collapsed | Enable `diversity_control` |
| `genotype_diversity` is low while score still improves | Slow convergence, not stuck | Let it run or increase `population_size` |
| `stagnation_generations` grows for 5+ generations | Local optimum | Enable `diversity_control`, fitness sharing, or local search |
| Multiple high-scoring but similar candidates | Single dominant region | Enable `fitness_sharing` to keep multiple niches alive |
| Final solution is good but nearby configs might be better | Under-exploited region | Enable `local_search` |

## Configuration Objects

Advanced settings live in `OptimizationConfig`. All parameters are disabled (or set to low-impact defaults) unless explicitly changed.

```python
from sklearn_genetic import OptimizationConfig
```

## Full Example: Hyperparameter Search

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn_genetic import (
    EvolutionConfig,
    GASearchCV,
    OptimizationConfig,
    PopulationConfig,
    RuntimeConfig,
)
from sklearn_genetic.schedules import ExponentialAdapter, InverseAdapter
from sklearn_genetic.space import Categorical, Integer

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

param_grid = {
    "n_estimators": Integer(50, 250),
    "max_depth": Integer(2, 20),
    "min_samples_split": Integer(2, 20),
    "min_samples_leaf": Integer(1, 10),
    "max_features": Categorical(["sqrt", "log2", None]),
    "criterion": Categorical(["gini", "entropy", "log_loss"]),
}

crossover_schedule = InverseAdapter(
    initial_value=0.8,
    end_value=0.6,
    adaptive_rate=0.05,
)
mutation_schedule = ExponentialAdapter(
    initial_value=0.1,
    end_value=0.25,
    adaptive_rate=0.08,
)

search = GASearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=1),
    param_grid=param_grid,
    cv=cv,
    scoring="roc_auc",
    evolution_config=EvolutionConfig(
        population_size=24,
        generations=18,
        crossover_probability=crossover_schedule,
        mutation_probability=mutation_schedule,
        tournament_size=3,
        elitism=True,
        keep_top_k=4,
    ),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, parallel_backend="auto", verbose=True),
    optimization_config=OptimizationConfig(
        # Diversity control
        diversity_control=True,
        diversity_threshold=0.18,
        diversity_stagnation_generations=4,
        diversity_mutation_boost=1.8,
        random_immigrants_fraction=0.15,
        # Fitness sharing
        fitness_sharing=True,
        sharing_radius=0.25,
        sharing_alpha=1.0,
        # Local search
        local_search=True,
        local_search_top_k=2,
        local_search_steps=4,
        local_search_radius=0.08,
    ),
)

search.fit(X_train, y_train)

y_pred = search.predict(X_test)
y_proba = search.predict_proba(X_test)[:, 1]

print(search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
```

## Reading Optimizer Telemetry

After fitting, convert `history` to a DataFrame to diagnose the search:

```python
history = pd.DataFrame(search.history)

columns = [
    "gen",
    "fitness_best",
    "fitness_max",
    "unique_individual_ratio",
    "genotype_diversity",
    "stagnation_generations",
    "mutation_probability",
    "diversity_control_triggered",
    "random_immigrants",
    "duplicate_replacements",
    "fitness_sharing_applied",
    "mean_niche_count",
    "max_niche_count",
    "local_refinements",
]

print(history[columns])
print(search.fit_stats_)
```

**Key telemetry fields:**

| Field | Meaning |
|-------|---------|
| `unique_individual_ratio` | Fraction of distinct individuals. Low → population collapsed |
| `genotype_diversity` | Average per-gene diversity. Low → structurally similar candidates |
| `stagnation_generations` | Generations since `fitness_best` last improved |
| `diversity_control_triggered` | Whether diversity control fired this generation |
| `random_immigrants` | Number of random candidates injected |
| `duplicate_replacements` | Duplicate offspring replaced before evaluation |
| `fitness_sharing_applied` | Whether niche-aware selection was active |
| `mean_niche_count` / `max_niche_count` | Crowding during selection |
| `local_refinements` | Neighbor candidates evaluated by local search (usually non-zero only in the final row) |

## Feature Selection Example

The same controls work with `GAFeatureSelectionCV`. In feature selection, `local_search_radius` controls the fraction of feature bits flipped when creating local neighbors.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn_genetic import (
    EvolutionConfig,
    GAFeatureSelectionCV,
    OptimizationConfig,
    PopulationConfig,
    RuntimeConfig,
)
from sklearn_genetic.schedules import ExponentialAdapter

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

selector = GAFeatureSelectionCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=1),
    cv=cv,
    scoring="roc_auc",
    max_features=18,
    evolution_config=EvolutionConfig(
        population_size=30,
        generations=16,
        crossover_probability=0.8,
        mutation_probability=ExponentialAdapter(
            initial_value=0.1, end_value=0.25, adaptive_rate=0.08
        ),
        keep_top_k=4,
    ),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, parallel_backend="auto", verbose=True),
    optimization_config=OptimizationConfig(
        diversity_control=True,
        diversity_threshold=0.2,
        diversity_stagnation_generations=4,
        random_immigrants_fraction=0.15,
        fitness_sharing=True,
        sharing_radius=0.2,
        local_search=True,
        local_search_top_k=2,
        local_search_steps=5,
        local_search_radius=0.1,
    ),
)

selector.fit(X_train, y_train)

print("Selected features:", selector.best_features_)
print("Test score:", selector.score(X_test, y_test))
```

## Recommended Workflow

Start simple and add controls based on telemetry:

1. **Default run** — use `PopulationConfig(initializer="smart")`, `crossover_probability=0.8`, `mutation_probability=0.1`. Inspect history.
2. **If diversity collapses early** — enable `OptimizationConfig(diversity_control=True)`.
3. **If one candidate family dominates** — enable `fitness_sharing=True`.
4. **If the final region looks close but not fully refined** — enable `local_search=True`.
5. **If stagnation persists** — add adaptive schedules: slowly increase mutation over generations.

## Tuning Guidelines

| Parameter | Practical starting range | Notes |
|-----------|------------------------|-------|
| `diversity_threshold` | 0.1–0.3 | Higher triggers earlier |
| `diversity_mutation_boost` | 1.5–2.5 | Too high = next generation is random |
| `random_immigrants_fraction` | 0.05–0.2 | Larger helps rugged spaces; may slow convergence |
| `sharing_radius` | 0.15–0.35 | Smaller = only penalize very similar candidates |
| `local_search_radius` | 0.05–0.15 | Larger = behaves more like an extra mutation phase |
| `local_search_steps` | 1–5 | Each step = extra CV calls |

## Practical Notes

- Fitness sharing only changes temporary selection pressure; it does not alter `best_score_`, `cv_results_`, or raw cross-validation scores.
- Local search evaluates extra candidates **after** the genetic search completes, so it can improve final quality but increases total runtime.
- Always keep a final holdout set separate from the CV folds for honest model assessment.

## Next Steps

- [Adaptive Schedules](./adapters) — schedule mutation and crossover rates over generations
- [Troubleshooting](./troubleshooting) — diagnose convergence problems using `fit_stats_` and `history`
- [Config Objects API](../api/config) — full parameter reference for `OptimizationConfig`
