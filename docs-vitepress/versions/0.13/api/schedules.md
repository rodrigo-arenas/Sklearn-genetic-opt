---
title: Schedules
description: API reference for adaptive crossover and mutation rate schedules.
---

# Schedules

Schedules let `crossover_probability` and `mutation_probability` in `EvolutionConfig` change adaptively over generations rather than staying constant.

```python
from sklearn_genetic.schedules import ExponentialAdapter, InverseAdapter
```

## ExponentialAdapter

Decays a value exponentially from `initial` toward `end` over generations.

```python
ExponentialAdapter(initial, end, adaptive_rate)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `initial` | float | Starting value (generation 0) |
| `end` | float | Minimum value |
| `adaptive_rate` | float | Controls decay speed. Larger values decay faster |

**Example:** decay mutation from 0.3 → 0.05 over many generations:

```python
from sklearn_genetic import EvolutionConfig
from sklearn_genetic.schedules import ExponentialAdapter

evolution_config = EvolutionConfig(
    population_size=30,
    generations=25,
    mutation_probability=ExponentialAdapter(initial=0.3, end=0.05, adaptive_rate=0.05),
)
```

## InverseAdapter

Decays a value inversely proportional to the generation number.

```python
InverseAdapter(initial, end, adaptive_rate)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `initial` | float | Starting value |
| `end` | float | Minimum value |
| `adaptive_rate` | float | Controls how fast the decay flattens |

## Typical Patterns

| Phase | Crossover | Mutation |
|-------|-----------|---------|
| Early (exploration) | high (0.8–0.9) | high (0.2–0.4) |
| Late (exploitation) | medium (0.6–0.75) | low (0.05–0.1) |

Use schedules to start with high exploration and shift toward exploitation as generations progress.

## See Also

- [Advanced Optimizer Control](../guide/advanced-optimizer-control) — full example with schedules
- [Config Objects](./config) — `EvolutionConfig` where schedules are passed
