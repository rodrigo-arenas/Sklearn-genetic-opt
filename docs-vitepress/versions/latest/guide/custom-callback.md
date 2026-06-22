---
title: Custom Callbacks
description: Write your own callback to add custom logic — stopping conditions, logging, or side effects — after each generation.
---

:::warning Development version
You are reading the **latest (dev)** docs. For the stable version, see [0.13](/versions/0.13/).
:::

# Custom Callbacks

You can extend the built-in callbacks or write your own from scratch by subclassing `BaseCallback`.

## Prerequisites

- Read [Callbacks](./callbacks) first

## The BaseCallback Interface

```python
from sklearn_genetic.callbacks.base import BaseCallback

class MyCallback(BaseCallback):
    def on_start(self, estimator=None):
        """Called once before generation 0. Return None."""
        pass

    def on_step(self, record=None, logbook=None, estimator=None):
        """Called after each generation.
        Return True to STOP the search, False to continue."""
        return False

    def on_end(self, logbook=None, estimator=None):
        """Called after the final generation. Return None."""
        pass
```

**Key rule:** `on_step` returning `True` **stops** the search. `False` (or `None`) continues.

### Parameters available in `on_step`

| Parameter | Type | Description |
|-----------|------|-------------|
| `record` | dict | Current generation stats: `gen`, `nevals`, `fitness`, `fitness_std`, `fitness_best`, `fitness_max`, `fitness_min`, `genotype_diversity`, `stagnation_generations`, etc. |
| `logbook` | DEAP Logbook | All previous generation records. Use `logbook.select("fitness")` to get a list of values |
| `estimator` | `GASearchCV` | The search estimator with all its attributes |

## Example: Stop After N Generations Below a Threshold

This callback stops the search if more than `N` fitness values in the history fall below a threshold:

```python
from sklearn_genetic.callbacks.base import BaseCallback


class DummyThreshold(BaseCallback):
    def __init__(self, threshold, N, metric="fitness"):
        self.threshold = threshold
        self.N = N
        self.metric = metric

    def on_start(self, estimator=None):
        print("Search starting!")

    def on_step(self, record=None, logbook=None, estimator=None):
        if len(logbook) <= self.N:
            return False  # not enough history yet

        # Get the last N+1 metric values
        stats = logbook.select(self.metric)[-(self.N + 1):]
        n_below = sum(1 for x in stats if x < self.threshold)

        if n_below > self.N:
            return True  # stop

        return False

    def on_end(self, logbook=None, estimator=None):
        print(f"Search ended after {len(logbook)} generations.")
```

Use it like any other callback:

```python
callback = DummyThreshold(threshold=0.85, N=4, metric="fitness")
evolved_estimator.fit(X_train, y_train, callbacks=callback)
```

## Example: Log to a File

```python
import json
from sklearn_genetic.callbacks.base import BaseCallback


class JsonLogger(BaseCallback):
    def __init__(self, path):
        self.path = path
        self._file = None

    def on_start(self, estimator=None):
        self._file = open(self.path, "w")

    def on_step(self, record=None, logbook=None, estimator=None):
        if record is not None:
            # record contains only JSON-serializable types
            self._file.write(json.dumps({
                "gen": record.get("gen"),
                "fitness_best": record.get("fitness_best"),
                "genotype_diversity": record.get("genotype_diversity"),
            }) + "\n")
            self._file.flush()
        return False

    def on_end(self, logbook=None, estimator=None):
        if self._file:
            self._file.close()
```

## Example: Alert When Score Plateaus

```python
from sklearn_genetic.callbacks.base import BaseCallback


class PlateauAlert(BaseCallback):
    """Send an alert when the best score hasn't improved for N generations."""

    def __init__(self, patience=10):
        self.patience = patience

    def on_step(self, record=None, logbook=None, estimator=None):
        stagnation = record.get("stagnation_generations", 0) if record else 0
        if stagnation >= self.patience:
            print(
                f"[PlateauAlert] Best score has not improved for "
                f"{stagnation} generations. Consider stopping early."
            )
        return False
```

## Tips & Gotchas

- Always return `False` from `on_step` unless you explicitly want to stop the search.
- Call `print(record.keys())` inside `on_step` during development to see all available fields.
- The `logbook` is a DEAP `Logbook` object. Use `logbook.select("metric_name")` to get a Python list of values across all generations.
- You can combine custom callbacks with built-in ones in the same list.
- Heavy computation in `on_step` (e.g., writing to a remote server) can slow down the search. For non-critical logging, consider buffering and flushing only in `on_end`.

## Next Steps

- [Callbacks](./callbacks) — all built-in callbacks and how to combine them
- [MLflow Integration](./mlflow) — log experiments to MLflow
