---
title: Algorithms
description: API reference for the DEAP-based evolutionary algorithm implementations used internally by GASearchCV.
---

:::warning Development version
You are reading the **latest (dev)** docs. For the stable version, see [stable](/stable/).
:::

# Algorithms

The `sklearn_genetic.algorithms` module provides the DEAP-based evolutionary algorithm implementations used internally by `GASearchCV` and `GAFeatureSelectionCV`.

These are low-level functions — you typically do not call them directly. They are exposed for advanced users who want to inspect or extend the evolutionary logic.

```python
from sklearn_genetic.algorithms import eaMuPlusLambda, eaMuCommaLambda, eaSimple, eaDifferentialEvolution
```

## eaMuPlusLambda

The (μ + λ) evolutionary strategy. The next generation is selected from the union of parents and offspring. This is the default algorithm used by `GASearchCV`.

- **Pros:** Parents compete with offspring; with `elitism=True`, archived best candidates retain survivor slots.
- **Cons:** Can converge prematurely if the initial population is too small.

## eaMuCommaLambda

The (μ, λ) evolutionary strategy. The next generation is selected from offspring only. Parents do not survive directly.

- **Pros:** Better exploration — avoids stagnation around a local optimum.
- **Cons:** Without `elitism=True`, offspring-only survival can lose the best individual between generations.

## eaSimple

A simple generational genetic algorithm. Each generation fully replaces the previous one.

- **Pros:** Simple and fast.
- **Cons:** Without `elitism=True`, generational replacement can lose the best individual.

## eaDifferentialEvolution

An optional differential-evolution strategy that generates one trial per
parent and retains the better candidate. It requires a population of at least
four. Both search classes accept `algorithm="eaDifferentialEvolution"`.
Numeric dimensions use their declared linear or log domain; categorical
choices and feature masks stay within their valid domains. It is an
alternative to benchmark for your task, not a generally faster or more
accurate replacement for the default algorithm.

## See Also

- [DEAP documentation](https://deap.readthedocs.io/en/master/api/algo.html) — upstream algorithm documentation
- [Advanced Optimizer Control](../guide/advanced-optimizer-control) — configure algorithm behavior via `EvolutionConfig`

## Diversity telemetry

`history["genotype_diversity"]` is the mean probability that two distinct population
members have different values at a gene. Identical populations score zero;
populations with distinct values at every gene score one. Balanced binary genes
score approximately 0.5 even in large populations. `unique_individual_ratio`
continues to describe the fraction of distinct complete candidates.

This corrects the older distinct-allele normalization that made large feature
populations appear collapsed. Custom diversity thresholds may need recalibration.
Local integer refinement now samples directly from the bounded neighborhood,
excluding the parent's value when alternatives exist, with constant memory.

## Elites and progress

With `elitism=True`, the mu/lambda algorithms reserve up to `keep_top_k` survivor
slots for copies of the best archived candidates. With `elitism=False`, all
survivors are selected from the algorithm's candidate pool. In both cases,
`fitness_best` tracks the best evaluated candidate in the hall of fame, while
`fitness_min`, `fitness_max` and `fitness` describe the current population.
