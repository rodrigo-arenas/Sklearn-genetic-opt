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
from sklearn_genetic.algorithms import eaMuPlusLambda, eaMuCommaLambda, eaSimple
```

## eaMuPlusLambda

The (μ + λ) evolutionary strategy. The next generation is selected from the union of parents and offspring. This is the default algorithm used by `GASearchCV`.

- **Pros:** Elitist — the best individuals are never lost.
- **Cons:** Can converge prematurely if the initial population is too small.

## eaMuCommaLambda

The (μ, λ) evolutionary strategy. The next generation is selected from offspring only. Parents do not survive directly.

- **Pros:** Better exploration — avoids stagnation around a local optimum.
- **Cons:** Can lose the best individual between generations (non-elitist).

## eaSimple

A simple generational genetic algorithm. Each generation fully replaces the previous one.

- **Pros:** Simple and fast.
- **Cons:** No explicit elitism — the best solution can be lost.

## See Also

- [DEAP documentation](https://deap.readthedocs.io/en/master/api/algo.html) — upstream algorithm documentation
- [Advanced Optimizer Control](../guide/advanced-optimizer-control) — configure algorithm behavior via `EvolutionConfig`
