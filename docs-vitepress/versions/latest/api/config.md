---
title: Config Objects
description: "API reference for EvolutionConfig, PopulationConfig, RuntimeConfig and OptimizationConfig, which group advanced settings for both search estimators."
---

:::warning Development version
You are reading the **latest (dev)** docs. For the stable version, see [stable](/stable/).
:::

# Config Objects

sklearn-genetic-opt groups advanced settings into four configuration objects. Passing `None` (the default) uses sensible defaults for each.

Select the engine using the estimator's keyword-only `optimizer_backend`. Population size, generations, archive size and scoring direction apply across engines. Crossover support varies by engine. Mutation, tournament, elitism, schedules and the choice of DEAP algorithm are DEAP-specific. See the [backend compatibility table](./optimizer-backends).

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
| `population_size` | int | `50` | Positive number of individuals per generation. Some engines require a higher minimum |
| `generations` | int | `80` | Nonnegative number of evolutionary generations. `0` skips evolution after initialization |
| `crossover_probability` | float or adapter | `0.8` | Probability of crossover between two parents. Can be a fixed float or an [adapter schedule](./schedules) |
| `mutation_probability` | float or adapter | `0.1` | Per-individual mutation probability. Can be a fixed float or an [adapter schedule](./schedules) |
| `tournament_size` | int | `3` | Number of individuals competing in tournament selection |
| `elitism` | bool | `True` | Carry the best individual unchanged to the next generation |
| `keep_top_k` | int | `1` | Positive size of the hall of fame |
| `criteria` | `"max"` or `"min"` | `"max"` | Whether to maximize or minimize the scoring metric |
| `algorithm` | str | `"eaMuPlusLambda"` | Evolutionary algorithm: `"eaMuPlusLambda"`, `"eaMuCommaLambda"`, `"eaSimple"`, or `"eaDifferentialEvolution"` (requires at least four individuals) |

The search accepts Python and NumPy integer scalars for `population_size`,
`generations` and `keep_top_k`. Boolean values raise `ValueError` before
candidate evaluation. With `generations=0`, configured final selection and
refit can still run after initialization.

Use `set_params(evolution_config=...)` to replace the group for the next
fresh fit. The group takes precedence over individual settings while it is
attached. To choose individual settings, remove the group in the same call.
DEAP starts a fresh fit with the selected probability schedules and restores
the saved schedule state when it resumes a checkpoint.

```python
search.set_params(evolution_config=None, population_size=20, generations=10)
```

## PopulationConfig

Controls how the initial population is created and seeded.

```python
from sklearn_genetic import PopulationConfig

PopulationConfig(
    initializer="smart",
    warm_start_configs=[],
    warm_start_masks=None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initializer` | `"smart"`, `"random"` or `"spread"` (feature selection) | `"smart"` | `"smart"` uses Latin hypercube sampling, estimator defaults and stratified categoricals for tuning. `"spread"` [varies the sizes of early feature masks](./gafeatureselectioncv#initial-mask-order-under-a-cv-budget). |
| `warm_start_configs` | list of dicts | empty list | Seeds for `GASearchCV` in input order. `"smart"` takes distinct completed candidates until the population is full. Invalid entries encountered during initialization raise `ValueError`. |
| `warm_start_masks` | list, tuple, or 2D array of masks | `None` | [Seeds for `GAFeatureSelectionCV`](./gafeatureselectioncv#start-from-known-feature-subsets). Each mask follows input column order and contains Boolean or integer 0/1 values. The first distinct masks occupy the first slots of the population. |

Replace the grouped configuration with `set_params(population_config=...)`
to change the initialization strategy for the next `fit`. For `GASearchCV`,
this also replaces the grouped warm starts. While `population_config` is set,
its `initializer` and any grouped warm starts take precedence over the direct
parameters. To choose a direct strategy again, detach the group and set the
initializer in the same call.

```python
search.set_params(population_config=None, population_initializer="random")
```

For `GASearchCV` with direct parameters, `set_params(warm_start_configs=[...])`
replaces the seeds and `set_params(warm_start_configs=None)` removes them.
For `GAFeatureSelectionCV`, replace the group to supply a new list or set
`warm_start_masks=None` in the new group to remove the seeds.

## RuntimeConfig

Controls parallelism, caching, and runtime behavior.

```python
import numpy as np

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
| `n_jobs` | int or None | `None` | Number of parallel jobs. `None` uses the joblib default, usually 1. `-1` uses all available cores. `0` raises before candidate evaluation. |
| `pre_dispatch` | str or int | `"2*n_jobs"` | Number of jobs pre-dispatched for parallel execution |
| `error_score` | float or `"raise"` | `np.nan` | Score to assign when a candidate's fit raises an exception. Use `"raise"` to surface exceptions during debugging |
| `return_train_score` | bool | `False` | Include training scores in `cv_results_` |
| `use_cache` | bool | `True` | Cache evaluated candidates and reuse scores when the same configuration appears again |
| `parallel_backend` | str | `"auto"` | Parallelism strategy: `"auto"` or `"population"` (parallel across candidates in each generation), `"cv"` (parallel across CV folds within each candidate) |
| `verbose` | bool | `True` | Print the per-generation log during fit |

Replace `runtime_config` with `set_params` to update these settings for the
next fresh fit. While `runtime_config` is set, its values take precedence
over direct parameters. To set individual options again, remove the group in
the same call. Unspecified options keep their current values.

```python
search.set_params(runtime_config=None, n_jobs=1, use_cache=True)
```

Population evaluation reuses local worker threads during a fit when joblib selects
its standard thread backend. Custom joblib backends and nested joblib scheduling
retain their own dispatch. This can reduce the overhead of evaluating small batches.
Candidate workers inherit the active scikit-learn configuration, including
`enable_metadata_routing` for [separate training and validation weights](../guide/understand-cv#separate-training-and-validation-weights).

:::tip Nested parallelism
Set the search's `n_jobs=1` when the estimator owns the worker budget.
To parallelize candidates or folds instead, set the estimator's worker count
to 1, when it exposes that setting. The `"cv"` mode schedules folds and leaves
the estimator's worker setting in effect. See the
[parallelism settings by layer](../guide/troubleshooting#choose-a-parallelism-layer).
:::

## OptimizationConfig

Finalist selection also applies to alternative engines. The diversity, fitness-sharing, immigrant and local-search operators in this section belong to DEAP.

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
| `local_search_steps` | int | `1` | Number of refinement steps per hall-of-fame individual. An accepted improvement becomes the starting point for the next step; each step adds cross-validation calls |
| `local_search_radius` | float | `0.1` | Step size as a fraction of the parameter range. Values 0.05–0.15 keep refinement local |

### Final Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `final_selection` | bool or `"reuse_folds"` | `False` | Re-evaluate the top `final_selection_top_k` candidates after the GA and select the best before refitting. For deterministic runs, `"reuse_folds"` with `error_score="raise"` reuses matching folds from the current fit |
| `final_selection_top_k` | int | `3` | Number of top candidates to re-evaluate |
| `final_selection_cv` | int, CV splitter, fold iterable or None | `None` | Cross-validation strategy for final selection. `None` reuses the main `cv` |

When final selection and refit are enabled, an integer `final_selection_cv`
must be at least 2 and at most the number of samples. Both estimators check
this before evaluating search candidates. They read custom splitters and
fold iterables when evaluating finalists.

Use `set_params(optimization_config=...)` to replace the group for the next
fresh fit. An explicit `final_selection_cv` keeps precedence when the group
changes. Pass `final_selection_cv=None` to use the group's CV again. The
group takes precedence for the other optimization settings while attached.

## See Also

- [Advanced Optimizer Control](../guide/advanced-optimizer-control) — guide with worked examples
- [Schedules](./schedules) — adaptive crossover/mutation rate schedules
