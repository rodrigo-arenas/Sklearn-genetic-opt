---
title: Examples
description: End-to-end examples for sklearn-genetic-opt — hyperparameter search, feature selection, pipelines, multi-metric, MLflow, checkpointing, and plotting.
---
# Examples

End-to-end runnable examples from real use cases. Each example is self-contained — you can copy the code directly into a script or notebook.

## Hyperparameter Search

| Example | What it covers |
|---------|---------------|
| [Comparing Search Methods](./sklearn-comparison) | Side-by-side: GASearchCV vs RandomizedSearchCV vs GridSearchCV |
| [Advanced Random Forest Tuning](./advanced-rf) | Smart initialization, warm starts, diversity control, fitness sharing, local search, adaptive schedules |
| [Pipeline Regression](./pipeline-regression) | Pipeline parameter naming, regression scorers, search visualization |

## Feature Selection

| Example | What it covers |
|---------|---------------|
| [Feature Selection With Noisy Data](./feature-selection) | GAFeatureSelectionCV on Iris with 12 added noise features |
| [Advanced RF + Feature Selection](./advanced-rf#feature-selection-with-gafeatureselectioncv) | Feature selection after hyperparameter tuning |

## Multi-Metric and Refit

| Example | What it covers |
|---------|---------------|
| [Multi-Metric Search on Iris](./multi-metric) | Multiple scorers, choosing refit metric, inspecting cv_results_ per metric |

## Experiment Tracking

| Example | What it covers |
|---------|---------------|
| [MLflow 3 Experiment Tracking](./mlflow-tracking) | Parent/child runs, dataset inputs, logged models, model lifecycle tags |

## Visualization

| Example | What it covers |
|---------|---------------|
| [Plotting Gallery](./plotting-gallery) | `plot_fitness_evolution`, `plot_history`, `plot_search_space` |

## Persistence

| Example | What it covers |
|---------|---------------|
| [Checkpointing and Persistence](./checkpointing) | `ModelCheckpoint`, `save`, `load`, inspecting checkpoint contents |
