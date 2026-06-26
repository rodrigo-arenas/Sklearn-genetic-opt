---
title: Migrating from 0.12 to 0.13
description: Step-by-step guide for upgrading sklearn-genetic-opt from 0.12.x to 0.13.0, covering breaking changes and new APIs.
---

:::warning Development version
You are reading the **latest (dev)** docs. For the stable version, see [stable](/stable/).
:::

# Migrating from 0.12 to 0.13

0.13.0 introduces significant improvements to search quality, parallelism, and configurability. Three breaking changes affect default behavior; existing code will run without errors but **results will differ** unless you pin the old defaults explicitly.

## Prerequisites

- sklearn-genetic-opt 0.12.x installed in your current environment
- Python ≥ 3.12

Upgrade with:

```bash
pip install --upgrade sklearn-genetic-opt
```

---

## Breaking Change 1 — Probability Defaults Swapped

`crossover_probability` and `mutation_probability` defaults have been **swapped** to align with standard genetic algorithm conventions.

| Parameter | 0.12 default | 0.13 default |
|-----------|-------------|-------------|
| `crossover_probability` | `0.2` | `0.8` |
| `mutation_probability` | `0.8` | `0.1` |

The new defaults produce better exploration on typical hyperparameter spaces. The old values were inverted relative to DEAP's intended semantics — high mutation with low crossover led to random-walk behavior rather than guided search.

### What to do

**Option A — Accept the new defaults (recommended).** Delete any explicit `crossover_probability=0.2` or `mutation_probability=0.8` arguments and rerun your search. Expect different (generally better) results.

**Option B — Preserve old behavior exactly.** Pin the old values:

```python
# 0.12 code
evolved_estimator = GASearchCV(
    estimator=clf,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    crossover_probability=0.2,   # was the default; now must be explicit
    mutation_probability=0.8,    # was the default; now must be explicit
)
```

Or with the new config objects:

```python
from sklearn_genetic import EvolutionConfig, GASearchCV

evolved_estimator = GASearchCV(
    estimator=clf,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    evolution_config=EvolutionConfig(
        crossover_probability=0.2,
        mutation_probability=0.8,
    ),
)
```

---

## Breaking Change 2 — Diversity Control On by Default

Population diversity control is now **enabled by default** with a stricter threshold.

| Parameter | 0.12 default | 0.13 default |
|-----------|-------------|-------------|
| `diversity_control` | `False` | `True` |
| `diversity_threshold` | `0.1` | `0.25` |

When `diversity_control=True`, the optimizer monitors `genotype_diversity` each generation. If diversity drops below `diversity_threshold`, it injects random immigrants and boosts mutation rate to escape premature convergence. This is almost always beneficial, but it means the optimizer may behave differently from 0.12 even with identical data and seeds.

### What to do

**Option A — Accept the new defaults (recommended).** Your search will be more robust against premature convergence. `fit_stats_["random_immigrants"]` tells you how many times the control triggered.

**Option B — Disable diversity control to match 0.12 behavior exactly:**

```python
# Flat parameters (backward compatible)
evolved_estimator = GASearchCV(
    estimator=clf,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    diversity_control=False,
    diversity_threshold=0.1,
)
```

Or with config objects:

```python
from sklearn_genetic import GASearchCV, OptimizationConfig

evolved_estimator = GASearchCV(
    estimator=clf,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    optimization_config=OptimizationConfig(
        diversity_control=False,
        diversity_threshold=0.1,
    ),
)
```

---

## Breaking Change 3 — Single-Objective Fitness for `GASearchCV`

`GASearchCV` now optimizes a **single objective**: cross-validation score only.

In 0.12, the fitness function included a second objective (`novelty_score`) based on Hamming distance between candidates. This Pareto-dominance approach frequently preferred diverse-but-weaker candidates over better-scoring ones, degrading search quality on large or irregular spaces.

`GAFeatureSelectionCV` is **unaffected** — it retains its two-objective fitness (CV score + feature count).

### What to do

No code changes are needed. The internal fitness representation changed but `GASearchCV`'s public API (`best_params_`, `best_score_`, `cv_results_`, `history`) is unchanged.

If you were relying on Pareto diversity implicitly (e.g., inspecting the second objective in custom callbacks), that value no longer exists. Use `OptimizationConfig(fitness_sharing=True)` if you need niche-based diversity pressure.

:::tip Checklist for callbacks
If you wrote a custom callback that reads from `record` or `logbook`, verify it does not reference `"fitness_var"` or a second fitness dimension. The generation record shape changed — see [Custom Callbacks](./custom-callback) for the updated field list.
:::

---

## New API: Config Objects

0.13 introduces grouped configuration dataclasses. **Flat keyword arguments still work**, so existing code needs no changes. Config objects are the recommended API for new code because they provide IDE autocomplete, type hints, and cleaner diffs.

```python
# 0.12 style — still valid
evolved_estimator = GASearchCV(
    estimator=clf,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    population_size=20,
    generations=40,
    mutation_probability=0.1,
    crossover_probability=0.8,
    tournament_size=3,
    elitism=True,
    keep_top_k=1,
    n_jobs=-1,
    verbose=True,
    use_cache=True,
)
```

```python
# 0.13 style — equivalent, using config objects
from sklearn_genetic import (
    EvolutionConfig,
    GASearchCV,
    OptimizationConfig,
    PopulationConfig,
    RuntimeConfig,
)

evolved_estimator = GASearchCV(
    estimator=clf,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    evolution_config=EvolutionConfig(
        population_size=20,
        generations=40,
        mutation_probability=0.1,
        crossover_probability=0.8,
        tournament_size=3,
        elitism=True,
        keep_top_k=1,
    ),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1, verbose=True, use_cache=True),
    optimization_config=OptimizationConfig(diversity_control=True),
)
```

:::info Mixing styles
You can mix flat parameters and config objects. If both are provided, the config object takes precedence over the corresponding flat parameter.
:::

---

## New Features Worth Adopting

These are opt-in additions — nothing breaks if you ignore them, but they improve search quality and observability.

### Smart Initialization

The default initializer is now `"smart"` (Latin hypercube sampling + estimator defaults + stratified categorical values). This reduces the chance of a bad initial population:

```python
# 0.12 behavior — random initialization
population_config=PopulationConfig(initializer="random")

# 0.13 default — smart initialization (no action needed)
population_config=PopulationConfig(initializer="smart")
```

### Parallel Candidate Evaluation

Unique candidates within a generation are now evaluated in parallel. Control the strategy with `parallel_backend`:

```python
RuntimeConfig(n_jobs=-1, parallel_backend="auto")   # default: parallel across candidates
RuntimeConfig(n_jobs=-1, parallel_backend="cv")      # parallel across CV folds instead
RuntimeConfig(n_jobs=-1, parallel_backend="population")  # explicit population-level
```

### Evaluation Counters

`fit_stats_` provides post-fit counters to understand search efficiency:

```python
search.fit(X_train, y_train)
print(search.fit_stats_)
# {
#     "evaluated_candidates": 420,
#     "unique_candidates": 310,
#     "cache_hits": 110,
#     "random_immigrants": 12,
#     ...
# }
```

### Generation Telemetry

`history` now includes diversity and stagnation fields per generation:

```python
import pandas as pd

history = pd.DataFrame(search.history)
print(history[["gen", "fitness_best", "genotype_diversity", "stagnation_generations"]])
```

---

## Quick Migration Checklist

- [ ] Upgraded package: `pip install --upgrade sklearn-genetic-opt`
- [ ] Reviewed default probability change — add explicit `crossover_probability=0.2, mutation_probability=0.8` if you need identical 0.12 results
- [ ] Reviewed diversity control change — add `diversity_control=False` if you need identical 0.12 results
- [ ] Checked custom callbacks for second-objective references (`fitness_var`, dual fitness arrays)
- [ ] (Optional) Migrated flat params to config objects for cleaner code

## Next Steps

- [Advanced Optimizer Control](./advanced-optimizer-control) — diversity control, fitness sharing, and local search
- [Callbacks](./callbacks) — built-in early-stopping and logging callbacks
- [Custom Callbacks](./custom-callback) — updated record field reference
- [Changelog](../release-notes) — full list of additions and fixes in 0.13.0
