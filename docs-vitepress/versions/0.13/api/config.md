---
title: Config Objects
description: API reference for EvolutionConfig, PopulationConfig, RuntimeConfig, and OptimizationConfig.
---

# Config Objects

sklearn-genetic-opt groups advanced settings into four configuration objects. Passing `None` (the default) uses sensible defaults for each.

## EvolutionConfig

Controls the core evolutionary algorithm parameters.

```python
from sklearn_genetic import EvolutionConfig

EvolutionConfig(
    population_size=50,
    generations=80,
    crossover_probability=0.8,
    mutation_probability=0.1,
    tournament_size=3,
    elitism=True,
    keep_top_k=1,
    criteria="max",
    algorithm="eaMuPlusLambda",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `population_size` | int | `50` | Number of individuals per generation |
| `generations` | int | `80` | Number of generations to run |
| `crossover_probability` | float or adapter | `0.8` | Probability of crossover between two parents. Can be a fixed float or an [adapter schedule](./schedules) |
| `mutation_probability` | float or adapter | `0.1` | Per-individual mutation probability. Can be a fixed float or an [adapter schedule](./schedules) |
| `tournament_size` | int | `3` | Number of individuals competing in tournament selection |
| `elitism` | bool | `True` | Carry the best individual unchanged to the next generation |
| `keep_top_k` | int | `1` | Size of the hall of fame — best `k` individuals are preserved across generations |
| `criteria` | `"max"` or `"min"` | `"max"` | Whether to maximize or minimize the scoring metric |
| `algorithm` | str | `"eaMuPlusLambda"` | DEAP algorithm: `"eaMuPlusLambda"`, `"eaMuCommaLambda"`, or `"eaSimple"` |

## PopulationConfig

Controls how the initial population is created and seeded.

```python
from sklearn_genetic import PopulationConfig

PopulationConfig(
    initializer="smart",
    warm_start_configs=None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initializer` | `"smart"` or `"random"` | `"smart"` | Population initialization strategy. `"smart"` uses Latin hypercube sampling, estimator defaults, stratified categoricals, and duplicate avoidance |
| `warm_start_configs` | list of dicts | `None` | Specific parameter configurations to include in the initial population. Each dict must have keys matching `param_grid`. Invalid configs are silently skipped |

## RuntimeConfig

Controls parallelism, caching, and runtime behavior.

```python
from sklearn_genetic import RuntimeConfig

RuntimeConfig(
    n_jobs=None,
    pre_dispatch="2*n_jobs",
    error_score=np.nan,
    return_train_score=False,
    use_cache=True,
    parallel_backend="auto",
    verbose=True,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_jobs` | int or None | `None` | Number of parallel jobs. `None` = 1, `-1` = all available cores |
| `pre_dispatch` | str or int | `"2*n_jobs"` | Number of jobs pre-dispatched for parallel execution |
| `error_score` | float or `"raise"` | `np.nan` | Score to assign when a candidate's fit raises an exception. Use `"raise"` to surface exceptions during debugging |
| `return_train_score` | bool | `False` | Include training scores in `cv_results_` |
| `use_cache` | bool | `True` | Cache evaluated candidates and reuse scores when the same configuration appears again |
| `parallel_backend` | str | `"auto"` | Parallelism strategy: `"auto"` or `"population"` (parallel across candidates in each generation), `"cv"` (parallel across CV folds within each candidate) |
| `verbose` | bool | `True` | Print the per-generation log during fit |

:::tip Nested parallelism
If your estimator already uses parallelism (e.g., `RandomForestClassifier(n_jobs=-1)`), set `parallel_backend="cv"` or set the estimator's `n_jobs=1` to avoid oversubscribing the CPU.
:::

## OptimizationConfig

Optional quality controls for the main GA loop. All features here are disabled by default except `diversity_control`.

```python
from sklearn_genetic import OptimizationConfig

OptimizationConfig(
    # Diversity control (enabled by default)
    diversity_control=True,
    diversity_threshold=0.25,
    diversity_stagnation_generations=5,
    diversity_mutation_boost=2.0,
    random_immigrants_fraction=0.1,

    # Adaptive tournament selection
    adaptive_selection=False,
    selection_pressure_min=2,
    selection_pressure_max=None,
    offspring_diversity_retries=0,

    # Fitness sharing
    fitness_sharing=False,
    sharing_radius=0.2,
    sharing_alpha=1.0,

    # Local search
    local_search=False,
    local_search_top_k=1,
    local_search_steps=1,
    local_search_radius=0.1,

    # Final re-evaluation
    final_selection=False,
    final_selection_top_k=3,
    final_selection_cv=None,
)
```

### Diversity Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `diversity_control` | bool | `True` | Monitor population diversity and inject random immigrants when it drops below `diversity_threshold` |
| `diversity_threshold` | float | `0.25` | Genotype diversity level below which intervention triggers. Values 0.1–0.3 are typical |
| `diversity_stagnation_generations` | int | `5` | Also trigger diversity control after this many stagnant generations, even if diversity is above the threshold |
| `diversity_mutation_boost` | float | `2.0` | Multiplier applied to `mutation_probability` when diversity control triggers. Values 1.5–2.5 are typical |
| `random_immigrants_fraction` | float | `0.1` | Fraction of offspring replaced with random individuals when diversity control triggers |

### Adaptive Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adaptive_selection` | bool | `False` | Adjust tournament size based on current diversity and stagnation |
| `selection_pressure_min` | int | `2` | Minimum tournament size when diversity is low |
| `selection_pressure_max` | int or None | `None` | Maximum tournament size. `None` uses `tournament_size` from `EvolutionConfig` |
| `offspring_diversity_retries` | int | `0` | Number of times to retry generating offspring that are distinct from parents |

### Fitness Sharing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fitness_sharing` | bool | `False` | Apply niche-aware selection pressure: reduce the effective fitness of candidates that are too similar to each other |
| `sharing_radius` | float | `0.2` | Niche radius — normalized distance below which two individuals share fitness. Values 0.15–0.35 are typical |
| `sharing_alpha` | float | `1.0` | Sharing function exponent. Higher values narrow the sharing effect to very similar individuals |

### Local Search

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `local_search` | bool | `False` | After the genetic search, evaluate neighbors of the top `local_search_top_k` hall-of-fame individuals |
| `local_search_top_k` | int | `1` | Number of hall-of-fame individuals to refine |
| `local_search_steps` | int | `1` | Number of neighbor evaluations per hall-of-fame individual. Each step adds cross-validation calls |
| `local_search_radius` | float | `0.1` | Step size as a fraction of the parameter range. Values 0.05–0.15 keep refinement local |

### Final Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `final_selection` | bool | `False` | Re-evaluate the top `final_selection_top_k` candidates after the GA and select the best before refitting |
| `final_selection_top_k` | int | `3` | Number of top candidates to re-evaluate |
| `final_selection_cv` | CV splitter or None | `None` | Cross-validation strategy for final selection. `None` reuses the main `cv` |

## See Also

- [Advanced Optimizer Control](../guide/advanced-optimizer-control) — guide with worked examples
- [Schedules](./schedules) — adaptive crossover/mutation rate schedules
