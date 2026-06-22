---
title: Schedules (Adapters)
description: API reference for adaptive crossover and mutation rate schedules — ConstantAdapter, ExponentialAdapter, InverseAdapter, PotentialAdapter.
---

:::warning Development version
You are reading the **latest (dev)** docs. For the stable version, see [0.13](/versions/0.13/).
:::

# Schedules (Adapters)

Adapters are schedule objects that let `crossover_probability` and `mutation_probability` in `EvolutionConfig` change automatically over generations instead of staying at a fixed value.

```python
from sklearn_genetic.schedules import (
    ConstantAdapter,
    ExponentialAdapter,
    InverseAdapter,
    PotentialAdapter,
)
```

All adapters share three parameters:

| Parameter | Description |
|-----------|-------------|
| `initial_value` | Probability used at generation 0 |
| `end_value` | Target value the schedule approaches over time |
| `adaptive_rate` | How quickly the schedule reaches `end_value`. Larger = faster |

Call `.step()` on an adapter to advance it one generation and get its current value. This is handled automatically by the optimizer — you only need to construct the adapter and pass it to `EvolutionConfig`.

## ConstantAdapter

Returns the same value every generation. This is what the optimizer uses internally when you pass a plain float.

```python
ConstantAdapter(initial_value=0.8)
```

## ExponentialAdapter

Decays (or ascends) exponentially from `initial_value` toward `end_value`:

$$p(t) = (p_0 - p_f) \cdot e^{-\alpha t} + p_f$$

```python
adapter = ExponentialAdapter(initial_value=0.8, end_value=0.2, adaptive_rate=0.1)
for _ in range(3):
    print(adapter.step())  # 0.8, 0.74, 0.69
```

**Use case:** Decay mutation from high exploration (0.3) → low exploitation (0.05) over the run.

## InverseAdapter

Decays with inverse decay — changes fast early, then slows:

$$p(t) = \frac{p_0 - p_f}{1 + \alpha t} + p_f$$

```python
adapter = InverseAdapter(initial_value=0.8, end_value=0.2, adaptive_rate=0.1)
for _ in range(3):
    print(adapter.step())  # 0.8, 0.75, 0.70
```

**Use case:** Slow the change in crossover rate after the first few generations.

## PotentialAdapter

Decays with a potential (geometric) form — the fastest initial drop:

$$p(t) = (p_0 - p_f)(1 - \alpha)^t + p_f$$

```python
adapter = PotentialAdapter(initial_value=0.8, end_value=0.2, adaptive_rate=0.1)
for _ in range(3):
    print(adapter.step())  # 0.8, 0.26, 0.21
```

**Use case:** Fast early convergence for short searches.

## Full Example

```python
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score

from sklearn_genetic import EvolutionConfig, GASearchCV, PopulationConfig, RuntimeConfig
from sklearn_genetic.schedules import ExponentialAdapter
from sklearn_genetic.space import Categorical, Continuous, Integer

data = load_digits()
n_samples = len(data.images)
X = data.images.reshape((n_samples, -1))
y = data["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# High crossover to recombine good combinations; slight decay as population converges
crossover_adapter = ExponentialAdapter(initial_value=0.8, end_value=0.6, adaptive_rate=0.1)
# Low mutation to refine; slight increase later to escape stagnation
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

## Tips & Gotchas

- Keep `crossover_probability + mutation_probability ≤ 1.0` at all generations when both are scheduled.
- If `initial_value > end_value`, the adapter decays. If `initial_value < end_value`, it ascends.
- The default `initial_value=0.8` for crossover and `initial_value=0.1` for mutation are good starting points.
- Verify the schedule shape before a long run by calling `.step()` in a loop and printing the values.

## See Also

- [Adapters Guide](../guide/adapters) — worked tutorial with when to use each adapter
- [Config Objects](./config) — `EvolutionConfig` where adapters are passed
- [Advanced Optimizer Control](../guide/advanced-optimizer-control) — full example with schedules and diversity control
