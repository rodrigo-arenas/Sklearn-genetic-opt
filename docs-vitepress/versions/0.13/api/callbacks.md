---
title: Callbacks API
description: API reference for built-in callbacks — ConsecutiveStopping, DeltaThreshold, TimerStopping, ProgressBar, LogbookSaver, TensorBoard, and MLflowCallback.
---

# Callbacks API

```python
from sklearn_genetic.callbacks import (
    ConsecutiveStopping,
    DeltaThreshold,
    TimerStopping,
    ProgressBar,
    LogbookSaver,
    TensorBoard,
    MLflowCallback,
    BaseCallback,
)
```

## BaseCallback

Base class for custom callbacks. Override `on_step` (required), `on_start` (optional), and `on_end` (optional).

```python
class BaseCallback:
    def on_start(self, search): ...
    def on_step(self, record, logbook) -> bool: ...
    def on_end(self, search): ...
```

Return `False` from `on_step` to stop the search.

## ConsecutiveStopping

Stops if a metric does not improve for `N` consecutive generations.

```python
ConsecutiveStopping(generations, metric="fitness_best")
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `generations` | int | Number of consecutive non-improving generations before stopping |
| `metric` | str | History column to watch (e.g., `"fitness_best"`, `"fitness"`) |

## DeltaThreshold

Stops when the per-generation improvement falls below a threshold.

```python
DeltaThreshold(threshold, metric="fitness")
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `threshold` | float | Minimum improvement required to continue |
| `metric` | str | History column to watch |

## TimerStopping

Stops after a wall-clock time limit.

```python
TimerStopping(total_seconds)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `total_seconds` | float | Maximum elapsed time in seconds |

## ProgressBar

Displays a tqdm progress bar showing generations completed.

```python
ProgressBar()
```

Requires `tqdm` (included in the core dependencies).

## LogbookSaver

Saves the generation logbook to a file after each generation.

```python
LogbookSaver(checkpoint_path)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `checkpoint_path` | str or Path | File path for the pickled logbook |

## TensorBoard

Logs per-generation fitness metrics to TensorBoard.

```python
TensorBoard(writer)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `writer` | `tf.summary.FileWriter` | TensorFlow summary writer |

Requires TensorFlow: `pip install sklearn-genetic-opt[all]`.

## MLflowCallback

Logs per-generation metrics to an active MLflow run.

```python
MLflowCallback()
```

Must be used inside an `mlflow.start_run()` context. Requires MLflow: `pip install sklearn-genetic-opt[mlflow]`.

## See Also

- [Callbacks Guide](../guide/callbacks) — usage examples
- [Custom Callbacks](../guide/custom-callback) — writing your own callback
