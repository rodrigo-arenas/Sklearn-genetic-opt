---
title: Custom Callbacks
description: Write your own callback to add custom logic after each generation of the genetic search.
---

:::warning Development version
You are reading the **latest (dev)** docs. For the stable version, see [0.13](/versions/0.13/).
:::

# Custom Callbacks

You can write a callback that runs after every generation and can stop the search, log data, or trigger any side effect.

## Prerequisites

- Read [Callbacks](./callbacks) first

## Callback Interface

A callback is a callable that receives the current logbook entry (a dictionary of generation statistics) and returns `True` to continue or `False` to stop:

```python
def my_callback(record):
    # record is a dict with keys like:
    # gen, nevals, fitness, fitness_std, fitness_best, genotype_diversity, ...
    return True  # returning False stops the search
```

Or subclass `BaseCallback`:

```python
from sklearn_genetic.callbacks import BaseCallback

class MyCallback(BaseCallback):
    def __init__(self, threshold):
        self.threshold = threshold

    def on_step(self, record, logbook):
        # record: current generation stats (dict)
        # logbook: full history so far (list of dicts)
        if record["fitness_best"] >= self.threshold:
            print(f"Target reached: {record['fitness_best']:.4f}")
            return False  # stop
        return True
```

Pass it to `fit` like any other callback:

```python
search.fit(X_train, y_train, callbacks=[MyCallback(threshold=0.98)])
```

## Example: Logging to a File

```python
import json
from sklearn_genetic.callbacks import BaseCallback

class FileLogger(BaseCallback):
    def __init__(self, path):
        self.path = path
        self._file = None

    def on_start(self, search):
        self._file = open(self.path, "w")

    def on_step(self, record, logbook):
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()
        return True

    def on_end(self, search):
        if self._file:
            self._file.close()
```

## Tips & Gotchas

- Return `False` (or `None`) to stop the search at the end of the current generation. The estimator will be refitted on the best individual found so far.
- The `record` dict contains all columns visible in the generation log. Print `record.keys()` in the first call to see what's available.
- Use `on_start` / `on_end` hooks for setup and teardown (opening files, starting timers, etc.).

## Next Steps

- [Callbacks](./callbacks) — the full list of built-in callbacks.
- [MLflow Integration](./mlflow) — log experiments to MLflow using `MLflowCallback`.
