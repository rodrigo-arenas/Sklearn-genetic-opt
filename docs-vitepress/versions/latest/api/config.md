---
title: Config Objects
description: API reference for EvolutionConfig, PopulationConfig, RuntimeConfig, and OptimizationConfig.
---

:::warning Development version
You are reading the **latest (dev)** docs. For the stable version, see [0.13](/versions/0.13/).
:::

# Config Objects

sklearn-genetic-opt groups advanced settings into four configuration objects. Passing `None` (the default) uses sensible defaults for each.

## EvolutionConfig

Controls the evolutionary algorithm parameters.

```python
from sklearn_genetic import EvolutionConfig

EvolutionConfig(
    population_size=50,
    generations=40,
    crossover_probability=0.8,
    mutation_probability=0.1,
    tournament_size=3,
    elitism=True,
    keep_top_k=1,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `population_size` | int | `50` | Number of individuals per generation |
| `generations` | int | `40` | Number of generations to run |
| `crossover_probability` | float or schedule | `0.8` | Probability of crossover between two parents |
| `mutation_probability` | float or schedule | `0.1` | Probability of mutating a gene |
| `tournament_size` | int | `3` | Number of individuals competing in tournament selection |
| `elitism` | bool | `True` | Whether to carry the best individual to the next generation |
| `keep_top_k` | int | `1` | Number of hall-of-fame individuals to preserve |

## PopulationConfig

Controls how the initial population is created and how diversity is maintained.

```python
from sklearn_genetic import PopulationConfig

PopulationConfig(
    initializer="smart",
    warm_start_configs=None,
    diversity_control=True,
    diversity_threshold=0.25,
    random_immigrants_fraction=0.05,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initializer` | `"smart"` or `"random"` | `"smart"` | Population initialization strategy. `"smart"` uses Latin hypercube + estimator defaults |
| `warm_start_configs` | list of dicts | `None` | Specific parameter configurations to include in the initial population |
| `diversity_control` | bool | `True` | Monitor genotype diversity and inject immigrants when it drops below `diversity_threshold` |
| `diversity_threshold` | float | `0.25` | Diversity level below which random immigrants are injected |
| `random_immigrants_fraction` | float | `0.05` | Fraction of population to replace with random immigrants when diversity is low |

## RuntimeConfig

Controls parallelism, caching, and runtime behavior.

```python
from sklearn_genetic import RuntimeConfig

RuntimeConfig(
    n_jobs=1,
    parallel_backend="auto",
    use_cache=False,
    verbose=False,
    error_score=np.nan,
    pre_dispatch="2*n_jobs",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_jobs` | int | `1` | Number of parallel jobs. `-1` uses all available cores |
| `parallel_backend` | str | `"auto"` | Parallelism strategy: `"auto"` (population-level), `"population"`, or `"cv"` (CV-level) |
| `use_cache` | bool | `False` | Cache evaluated candidates and reuse scores on repeated configurations |
| `verbose` | bool | `False` | Print per-generation log during fit |
| `error_score` | float or `"raise"` | `np.nan` | Score to assign when a candidate's fit raises an exception |

## OptimizationConfig

Controls local search and fitness sharing.

```python
from sklearn_genetic import OptimizationConfig

OptimizationConfig(
    local_search=False,
    local_search_step=0.1,
    fitness_sharing=False,
    fitness_sharing_alpha=1.0,
    fitness_sharing_sigma=0.5,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `local_search` | bool | `False` | Refine hall-of-fame individuals by evaluating their neighbors |
| `local_search_step` | float | `0.1` | Step size for local search, as a fraction of the parameter range |
| `fitness_sharing` | bool | `False` | Reduce fitness of similar individuals to encourage niche exploration |
| `fitness_sharing_alpha` | float | `1.0` | Sharing function exponent |
| `fitness_sharing_sigma` | float | `0.5` | Niche radius for fitness sharing |

## See Also

- [Advanced Optimizer Control](../guide/advanced-optimizer-control) — guide with examples
- [Schedules](./schedules) — adaptive crossover/mutation schedules
