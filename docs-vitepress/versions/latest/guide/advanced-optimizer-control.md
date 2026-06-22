---
title: Advanced Optimizer Control
description: Diversity control, local search, fitness sharing, adaptive schedules, and optimizer telemetry for harder search spaces.
---

:::warning Development version
You are reading the **latest (dev)** docs. For the stable version, see [0.13](/versions/0.13/).
:::

# Advanced Optimizer Control

When default settings produce premature convergence or poor search quality, these advanced controls give you finer-grained control over the evolutionary process.

## Prerequisites

- Completed [Basic Usage](./basic-usage)
- Familiar with [Understanding Cross-Validation](./understand-cv)

## Configuration Objects

Advanced settings are grouped into configuration objects:

| Object | Controls |
|--------|---------|
| `EvolutionConfig` | Population size, generations, crossover/mutation rates, elitism |
| `PopulationConfig` | Initialization strategy, warm starts, diversity thresholds |
| `RuntimeConfig` | Parallelism, caching, verbosity, error handling |
| `OptimizationConfig` | Local search, fitness sharing, diversity control parameters |

## Diversity Control

Diversity control prevents premature convergence by monitoring the population's genotype diversity and injecting random immigrants when it drops below a threshold.

Diversity control is **enabled by default** as of 0.13.0 (`diversity_control=True`, `diversity_threshold=0.25`).

```python
from sklearn_genetic import EvolutionConfig, GASearchCV, OptimizationConfig, PopulationConfig, RuntimeConfig

search = GASearchCV(
    estimator=your_estimator,
    param_grid=your_param_grid,
    cv=your_cv,
    scoring="roc_auc",
    evolution_config=EvolutionConfig(population_size=30, generations=20),
    population_config=PopulationConfig(
        initializer="smart",
        diversity_control=True,
        diversity_threshold=0.25,       # trigger when diversity < 25%
        random_immigrants_fraction=0.2, # inject 20% of population as immigrants
    ),
    runtime_config=RuntimeConfig(n_jobs=-1, use_cache=True),
)
```

## Fitness Sharing

Fitness sharing reduces the fitness of individuals that are too similar to each other, promoting exploration of different regions of the search space:

```python
from sklearn_genetic import OptimizationConfig

opt_config = OptimizationConfig(
    fitness_sharing=True,
    fitness_sharing_alpha=1.0,   # sharing function exponent
    fitness_sharing_sigma=0.5,   # niche radius
)
```

## Local Search

Local search refines the hall-of-fame solutions after each generation by evaluating their neighbors:

```python
opt_config = OptimizationConfig(
    local_search=True,
    local_search_step=0.1,  # step size relative to parameter range
)
```

## Adaptive Schedules

Crossover and mutation rates can be scheduled to change over generations:

```python
from sklearn_genetic.schedules import ExponentialAdapter, InverseAdapter

evolution_config = EvolutionConfig(
    population_size=30,
    generations=25,
    mutation_probability=ExponentialAdapter(initial=0.3, end=0.05, adaptive_rate=0.05),
    crossover_probability=InverseAdapter(initial=0.9, end=0.6, adaptive_rate=0.05),
)
```

## Telemetry: Reading `history`

After fitting, `pd.DataFrame(search.history)` contains one row per generation with these key columns:

| Column | Meaning |
|--------|---------|
| `genotype_diversity` | Average fraction of distinct values per gene position (0–1) |
| `unique_individual_ratio` | Fraction of distinct individuals in the population |
| `stagnation_generations` | Consecutive generations without `fitness_best` improvement |
| `diversity_control_triggered` | Whether diversity control fired this generation |
| `random_immigrants` | Number of immigrants injected |

```python
import pandas as pd

history = pd.DataFrame(search.history)
print(history[[
    "gen", "fitness_best", "genotype_diversity",
    "unique_individual_ratio", "stagnation_generations",
    "diversity_control_triggered",
]])
```

## Tips & Gotchas

- If `genotype_diversity` drops below `diversity_threshold` in the first few generations, increase `population_size` or `random_immigrants_fraction`.
- Local search adds evaluations — monitor `fit_stats_["local_refinement_candidates"]` to see how many extra evaluations it produced.
- Fitness sharing and local search together can significantly increase runtime. Enable one at a time and measure the benefit.

## Next Steps

- [Troubleshooting](./troubleshooting) — diagnose convergence problems using `fit_stats_` and `history`.
- [Reproducibility](./reproducibility) — make results reproducible across runs.
