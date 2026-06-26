---
title: Cross-Validation in Genetic Algorithm Hyperparameter Search
description: How cross-validation interacts with evolutionary search, how to read the generation log, and how to interpret convergence plots.
---

:::warning Development version
You are reading the **latest (dev)** docs. For the stable version, see [stable](/stable/).
:::

**Estimated reading time:** 8 minutes  
**Difficulty:** Intermediate  
**Prerequisites:** [Getting Started with GASearchCV](./basic-usage), basic cross-validation knowledge

# Cross-Validation in Genetic Algorithm Hyperparameter Search

This tutorial explains how `GASearchCV` evaluates candidate hyperparameters and how cross-validation fits into the evolutionary search process.

## Prerequisites

- Completed [Basic Usage](./basic-usage)
- Basic familiarity with scikit-learn cross-validation

## Key Parameters

`cv`
: The cross-validation strategy. Can be an integer or any compatible scikit-learn cross-validator such as `KFold`, `StratifiedKFold`, or `RepeatedKFold`.

`scoring`
: The metric used to evaluate each candidate. For classification: `"accuracy"`, `"roc_auc"`, `"f1"`. For regression: `"r2"`, `"neg_root_mean_squared_error"`. See [scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html) for the full list.

## Evolutionary Algorithm Background

A genetic algorithm is a metaheuristic optimization method inspired by natural selection. In sklearn-genetic-opt:

- **Individual** — one candidate solution (one set of hyperparameters).
- **Population** — a group of individuals evaluated in the same generation.
- **Generation** — one iteration of the evolutionary process.
- **Fitness value** — the cross-validation score used to compare individuals.

Each generation, individuals are selected by fitness, crossed over to create offspring, and mutated to introduce variation. The process repeats until the configured number of generations is reached or a callback stops the search.

## The Generation Log

When `RuntimeConfig(verbose=True)` is set, each generation prints a log row:

```
gen  nevals  fitness  fitness_std  fitness_best  fitness_max  fitness_min  div    unique  stag  events
1    20      0.932    0.012        0.961         0.961        0.901        0.87   1.00    0     
2    18      0.944    0.008        0.972         0.972        0.923        0.81   0.95    0     dup=2
```

| Column | Meaning |
|--------|---------|
| `gen` | Generation number |
| `nevals` | Individuals evaluated this generation (may be less than `population_size` if cache hits occur) |
| `fitness` | Average CV score across the population |
| `fitness_std` | Standard deviation of CV scores |
| `fitness_best` | Best score found so far across all generations |
| `fitness_max` | Best score in this generation |
| `fitness_min` | Worst score in this generation |
| `div` | Genotype diversity — fraction of distinct values per gene position (1.0 = fully diverse) |
| `unique` | Unique individual ratio — fraction of distinct individuals in the population |
| `stag` | Stagnation generations — consecutive generations without improvement in `fitness_best` |
| `events` | Optimizer interventions: `div` (diversity control), `imm=N` (N immigrants injected), `dup=N` (N duplicates replaced), `share` (fitness sharing applied) |

## How Candidates Are Evaluated

1. A generation of individuals (parameter configurations) is produced by crossover and mutation.
2. Duplicate individuals within the generation are detected and removed (only unique ones are evaluated).
3. Unique candidates are evaluated in parallel (when `n_jobs > 1` and `parallel_backend` is `"auto"` or `"population"`).
4. Each candidate runs its own cross-validation sequentially (to avoid nested parallelism).
5. Scores are stored in the fitness cache. If the same configuration appears in a later generation, the cached score is reused (a cache hit).

## Visualizing the CV Process

```python
import pandas as pd
from sklearn_genetic.plots import plot_fitness_evolution

# After fitting:
history = pd.DataFrame(evolved_estimator.history)
print(history[["gen", "fitness", "fitness_best", "genotype_diversity", "stagnation_generations"]])

plot_fitness_evolution(evolved_estimator)
```

The `plot_fitness_evolution` chart shows:
- The mean fitness (blue line) — average quality of the population over generations.
- The best fitness (orange line) — the best solution found so far.

A healthy search shows the mean and best fitness rising together, then plateauing as the population converges.

## Tips & Gotchas

- Use `StratifiedKFold` for classification tasks to keep class balance across folds.
- A noisy scoring metric (high `fitness_std`) can mislead the optimizer. Increase `cv` folds to reduce variance.
- The `use_cache=True` option in `RuntimeConfig` is strongly recommended — it avoids re-evaluating previously seen configurations.

## See Also

- [Common Hyperparameter Tuning Mistakes](./common-mistakes) — avoid CV-related mistakes
- [Early Stopping with Callbacks](./callbacks) — stop search when CV plateaus
- [Advanced Optimizer Control](./advanced-optimizer-control) — diversity and stagnation telemetry
- [Reproducibility & Checkpointing](./reproducibility) — seed your CV splitter
- [API: GASearchCV](../api/gasearchcv) — full parameter reference
