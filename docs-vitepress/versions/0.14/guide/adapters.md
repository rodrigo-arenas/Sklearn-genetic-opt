---
title: Adaptive Mutation and Crossover Schedules for Better Convergence
description: Use ExponentialAdapter, InverseAdapter, and PotentialAdapter to anneal mutation and crossover rates from exploration to exploitation during genetic search.
---

# Adaptive Mutation and Crossover Schedules for Better Convergence

By default, `crossover_probability` and `mutation_probability` in `EvolutionConfig` stay fixed for the entire search. Adapters let them change automatically over generations — a useful tool for searches that need strong early exploration but precise late-stage refinement.

## Prerequisites

- Completed [Basic Usage](./basic-usage)

## Why Adaptive Schedules?

The recommended starting point is:
- **High crossover (0.8)** — recombines good parameter combinations from high-fitness parents. This is the primary search operator.
- **Low mutation (0.1)** — provides exploratory perturbation without disrupting good solutions.

But a fixed strategy can be suboptimal across the full run. Early generations benefit from more exploration; late generations benefit from more refinement. Adapters let you express this:

| Stage | Crossover | Mutation |
|-------|-----------|---------|
| Early (exploration) | High (0.8–0.9) | Low→Medium (0.1→0.2) |
| Late (refinement) | Slightly lower (0.6–0.8) | Still moderate |

## Available Adapters

| Adapter | Shape | When to use |
|---------|-------|-------------|
| `ConstantAdapter` | Flat | Explicit fixed value (same as passing a float) |
| `ExponentialAdapter` | Fast initial change, gradual flattening | Most common choice |
| `InverseAdapter` | Moderate initial change | Smoother transitions |
| `PotentialAdapter` | Very fast initial drop | Short searches |

All adapters take the same three parameters:

```python
AdapterName(
    initial_value=...,  # value at generation 0
    end_value=...,      # target value (approached but not necessarily reached)
    adaptive_rate=...,  # speed of change; larger = faster
)
```

## Example: Decaying Mutation, Stable Crossover

```python
from sklearn_genetic.schedules import ExponentialAdapter

# Crossover: start high, decay slightly — keeps recombination strong
crossover_adapter = ExponentialAdapter(
    initial_value=0.8,
    end_value=0.6,
    adaptive_rate=0.1,
)

# Mutation: start low, increase gently to escape stagnation in later generations
mutation_adapter = ExponentialAdapter(
    initial_value=0.1,
    end_value=0.2,
    adaptive_rate=0.1,
)
```

Pass them to `EvolutionConfig`:

```python
from sklearn_genetic import EvolutionConfig

evolution_config = EvolutionConfig(
    population_size=20,
    generations=25,
    crossover_probability=crossover_adapter,
    mutation_probability=mutation_adapter,
)
```

## Full Working Example

```python
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn_genetic import EvolutionConfig, GASearchCV, PopulationConfig, RuntimeConfig
from sklearn_genetic.schedules import ExponentialAdapter, InverseAdapter
from sklearn_genetic.space import Categorical, Continuous, Integer

data = load_digits()
n_samples = len(data.images)
X = data.images.reshape((n_samples, -1))
y = data["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

crossover_adapter = ExponentialAdapter(initial_value=0.8, end_value=0.6, adaptive_rate=0.1)
mutation_adapter = ExponentialAdapter(initial_value=0.1, end_value=0.2, adaptive_rate=0.1)

param_grid = {
    "min_weight_fraction_leaf": Continuous(0.01, 0.5, distribution="log-uniform"),
    "bootstrap": Categorical([True, False]),
    "max_depth": Integer(2, 30),
    "max_leaf_nodes": Integer(2, 35),
    "n_estimators": Integer(100, 300),
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

evolved_estimator = GASearchCV(
    estimator=RandomForestClassifier(random_state=42),
    cv=cv,
    scoring="accuracy",
    param_grid=param_grid,
    evolution_config=EvolutionConfig(
        population_size=20,
        generations=25,
        crossover_probability=crossover_adapter,
        mutation_probability=mutation_adapter,
    ),
    population_config=PopulationConfig(initializer="smart"),
    runtime_config=RuntimeConfig(n_jobs=-1),
)

evolved_estimator.fit(X_train, y_train)
print(evolved_estimator.best_params_)
print("Accuracy:", accuracy_score(y_test, evolved_estimator.predict(X_test)))
```

## Visualizing the Schedule

Preview a schedule before committing to a long run:

```python
from sklearn_genetic.schedules import ExponentialAdapter
import matplotlib.pyplot as plt

adapter = ExponentialAdapter(initial_value=0.1, end_value=0.2, adaptive_rate=0.1)
values = [adapter.step() for _ in range(30)]

plt.plot(values)
plt.xlabel("Generation")
plt.ylabel("Mutation probability")
plt.title("ExponentialAdapter: 0.1 → 0.2")
plt.grid(True, alpha=0.3)
plt.show()
```

## Tips & Gotchas

- Keep `crossover_probability + mutation_probability ≤ 1.0` at every generation when both are scheduled.
- Start with `adaptive_rate=0.1` and adjust — smaller values give more gradual change.
- If `initial_value > end_value`, the adapter decays. If `initial_value < end_value`, it ascends.
- For most searches, fixed `crossover_probability=0.8, mutation_probability=0.1` works well. Only add adapters when you observe stagnation in the generation log.

## Next Steps

- [Schedules API](../api/schedules) — full API reference for all four adapter types
- [Advanced Optimizer Control](./advanced-optimizer-control) — combine schedules with diversity control and fitness sharing
