---
title: "Advanced Optimizer Control: Diversity, Fitness Sharing, and Local Search"
description: Control population diversity, enable fitness sharing and local search refinement, and interpret optimizer telemetry to improve genetic algorithm hyperparameter search.
---

# Advanced Optimizer Control: Diversity, Fitness Sharing, and Local Search

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
import numpy as np
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
from sklearn_genetic.plots import plot_diversity, plot_optimizer_events
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
    random_state=42,
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

The same telemetry is easier to reason about visually. `plot_diversity` shows whether the population collapsed or recovered, while `plot_optimizer_events` marks the generations where diversity control, immigrant injection, sharing, or local search changed the search process.

```python
import matplotlib.pyplot as plt

plot_diversity(search)
plt.show()

plot_optimizer_events(search)
plt.show()
```

![Advanced optimizer diversity](/images/advanced_optimizer_diversity.png)

![Advanced optimizer events](/images/advanced_optimizer_events.png)

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
    random_state=42,
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

## Cheaper search with full-training finalist validation

Both searches can explore using fewer training rows per CV fold, then re-evaluate a small number of distinct finalists on the original folds. `subsample_cv` changes only the training indices; every validation row stays in place. The evolutionary operators continue to select and generate candidates using these cheaper scores.

Use a separate outer test set to measure whether the speed/quality tradeoff suits your data:

```python
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn_genetic import GASearchCV, GAFeatureSelectionCV, subsample_cv
from sklearn_genetic.space import Continuous

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
full_cv = list(StratifiedKFold(3, shuffle=True, random_state=42).split(X_train, y_train))
search_cv = subsample_cv(full_cv, 0.25, stratify=y_train, random_state=42)
model = make_pipeline(StandardScaler(), SVC())
common = dict(
    cv=search_cv,
    scoring="accuracy",
    population_size=8,
    generations=4,
    random_state=42,
    n_jobs=1,
    verbose=False,
    final_selection=True,
    final_selection_top_k=4,
    final_selection_cv=full_cv,
)
hpo = GASearchCV(
    clone(model),
    param_grid={
        "svc__C": Continuous(0.01, 1000, distribution="log-uniform"),
        "svc__gamma": Continuous(0.0001, 1, distribution="log-uniform"),
    },
    **common,
).fit(X_train, y_train)
features = GAFeatureSelectionCV(clone(model), **common).fit(X_train, y_train)
print(hpo.best_score_, hpo.score(X_test, y_test))
print(features.best_score_, features.score(X_test, y_test), features.support_.sum())
```

For models where each fold is expensive, another option is to replace the `subsample_cv(...)` line with `search_cv = full_cv[:1]` and set `final_selection_top_k=8` in `common`. The search then scores candidates on one original fold and validates the eight best distinct candidates on all three folds. This can save more time than shortening each training fold, but preliminary rankings use less validation data and can miss the best candidate. In a [six-split five-fold SVC benchmark](https://github.com/rodrigo-arenas/Sklearn-genetic-opt/blob/master/benchmarks/PERFORMANCE_V2.md#does-promoting-more-finalists-recover-accuracy), this pattern made HPO 4.09–4.81× faster with a mean accuracy loss of 0.00583 on synthetic data. Feature selection was 3.48–3.92× faster but lost mean accuracy on all three datasets. Expanding from eight to 24 finalists did not consistently recover quality. Compare held-out accuracy and the complete fitting time on your own data before choosing either shortcut.

`cv_results_`, `history` and the hall of fame retain the original search scores and candidates. `final_selection_results_` records each promoted candidate's original and final scores, full-validation fold scores, the winner and validation time. `best_score_` describes the final-validation winner, so it need not equal a value in `cv_results_`. Feature selection breaks equal final scores by preferring fewer columns. Invalid NaN finalists cannot replace valid candidates; if every finalist score is NaN, fitting raises an error.

Final selection runs with `refit=True`. Its extra CV evaluations and final refit are additional to the evolutionary budget; generation stopping callbacks do not cap those stages. Include them when timing a workflow. An explicit non-None `final_selection_cv` overrides the corresponding `OptimizationConfig` value. Both settings and explicit fold arrays survive cloning and checkpoints.

Use `stratify=y_train` for classification, and omit it for regression. Sampling that drops a class raises an error; increase the fraction in that case. The discarded rows do not need to contain every class. A fraction of 1 keeps the original training folds. This is opt-in: ordinary searches still use full training folds. Smaller training sets can change candidate rankings, and full validation of finalists cannot recover a good candidate discarded earlier. Measure held-out quality and total time; small datasets or models with high fixed training cost may gain little.
