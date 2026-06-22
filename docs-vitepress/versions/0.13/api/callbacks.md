---
title: Callbacks API
description: API reference for all built-in callbacks — ThresholdStopping, ConsecutiveStopping, DeltaThreshold, TimerStopping, ProgressBar, LogbookSaver, ModelCheckpoint, TensorBoard, and BaseCallback.
---

# Callbacks API

```python
from sklearn_genetic.callbacks import (
    ThresholdStopping,
    ConsecutiveStopping,
    DeltaThreshold,
    TimerStopping,
    ProgressBar,
    LogbookSaver,
    ModelCheckpoint,
    TensorBoard,
    BaseCallback,
)
```

Callbacks are passed to `fit(X, y, callbacks=[...])`. After each generation, `on_step` is called on every callback. If any callback returns `True` from `on_step`, the search stops at the end of that generation.

Available metric names for stopping callbacks: `"fitness"`, `"fitness_std"`, `"fitness_best"`, `"fitness_max"`, `"fitness_min"`.

## BaseCallback

Base class for custom callbacks. Subclass and override the methods you need.

```python
class BaseCallback:
    def on_start(self, estimator=None): ...      # called before generation 0
    def on_step(self, record=None, logbook=None, estimator=None) -> bool: ...  # True = stop
    def on_end(self, logbook=None, estimator=None): ...  # called after the last generation
```

`on_step` returning `True` stops the search. `False` (or `None`) continues.

## ThresholdStopping

Stops when the metric reaches or exceeds a target value.

```python
ThresholdStopping(threshold, metric="fitness")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | — | Stop when `metric ≥ threshold` |
| `metric` | str | `"fitness"` | History column to watch |

**Example:** stop when mean population accuracy exceeds 0.98:

```python
ThresholdStopping(threshold=0.98, metric="fitness_max")
```

## ConsecutiveStopping

Stops if the metric fails to improve for `N` consecutive generations.

```python
ConsecutiveStopping(generations, metric="fitness")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `generations` | int | — | Number of non-improving consecutive generations before stopping |
| `metric` | str | `"fitness"` | History column to watch |

**Example:** stop if `fitness_best` hasn't improved in 8 generations:

```python
ConsecutiveStopping(generations=8, metric="fitness_best")
```

## DeltaThreshold

Stops when the spread (max − min) of the metric across the last `N` generations falls below a threshold — detects convergence.

```python
DeltaThreshold(threshold, generations=2, metric="fitness")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | — | Stop when `max(metric[-N:]) - min(metric[-N:]) ≤ threshold` |
| `generations` | int | `2` | Window size |
| `metric` | str | `"fitness"` | History column to watch |

## TimerStopping

Stops after a wall-clock time limit. The current generation finishes before stopping.

```python
TimerStopping(total_seconds)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `total_seconds` | float | Stop after this many elapsed seconds |

## ProgressBar

Displays a tqdm progress bar. No parameters required.

```python
ProgressBar()
```

## LogbookSaver

Saves the generation logbook to a file after each generation using `joblib.dump`.

```python
LogbookSaver(checkpoint_path)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `checkpoint_path` | str or Path | File path for the pickled logbook |

Restore with:

```python
from joblib import load
logbook = load("./logbook.pkl")
```

## ModelCheckpoint

Saves the full estimator state (logbook + model state) to a pickle file after each generation.

```python
ModelCheckpoint(checkpoint_path)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `checkpoint_path` | str or Path | File path for the checkpoint |

Restore with:

```python
import pickle

with open("./checkpoint.pkl", "rb") as f:
    checkpoint = pickle.load(f)

estimator_state = checkpoint["estimator_state"]
logbook = checkpoint["logbook"]
```

## TensorBoard

Logs per-generation fitness metrics to TensorBoard.

```python
TensorBoard(writer)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `writer` | `tf.summary.FileWriter` | TensorFlow summary writer |

Requires TensorFlow: `pip install sklearn-genetic-opt[all]`.

```python
import tensorflow as tf
from sklearn_genetic.callbacks import TensorBoard

writer = tf.summary.create_file_writer("./logs")
callback = TensorBoard(writer=writer)
```

## See Also

- [Callbacks Guide](../guide/callbacks) — usage examples and combining callbacks
- [Custom Callbacks](../guide/custom-callback) — writing your own callback
