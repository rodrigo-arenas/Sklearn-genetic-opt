---
title: Examples
description: End-to-end examples for sklearn-genetic-opt — hyperparameter search, feature selection, pipelines, multi-metric, MLflow, checkpointing, and plotting.
---
:::warning Development version
This is the **latest (dev)** documentation. It may contain unreleased features or breaking changes. For the stable release, use [version 0.13](/versions/0.13/).
:::


# Examples

End-to-end runnable examples from real use cases. Each example is self-contained — you can copy the code directly into a script or notebook.

:::tip Looking for deeper tutorials?
The [Tutorials](/versions/latest/tutorials/) section covers comprehensive walkthroughs for XGBoost, LightGBM, CatBoost, multi-stage feature selection, and imbalanced classification — each with baseline comparisons, visualizations, and practical notes.
:::

## Hyperparameter Search

| Example | What it covers |
|---------|---------------|
| [Comparing Search Methods](./sklearn-comparison) | Side-by-side: GASearchCV vs RandomizedSearchCV vs GridSearchCV |
| [Advanced Random Forest Tuning](./advanced-rf) | Smart initialization, warm starts, diversity control, fitness sharing, local search, adaptive schedules |
| [Pipeline Regression](./pipeline-regression) | Pipeline parameter naming, regression scorers, search visualization |

## Feature Selection

| Example | What it covers |
|---------|---------------|
| [Finding the Signal in 60 Columns](./feature-selection) | GAFeatureSelectionCV recovers the signal from a dataset that is two-thirds noise, beating the all-features baseline |
| [Advanced RF + Feature Selection](./advanced-rf#feature-selection-with-gafeatureselectioncv) | Feature selection after hyperparameter tuning |

## Multi-Metric and Refit

| Example | What it covers |
|---------|---------------|
| [Multi-Metric Search on Imbalanced Data](./multi-metric) | Multiple scorers that genuinely disagree, choosing the refit metric, inspecting per-metric `cv_results_` |

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
