---
title: Tutorials
description: In-depth tutorials for sklearn-genetic-opt covering XGBoost, LightGBM, CatBoost, comprehensive feature selection, and imbalanced classification.
---

# Tutorials

Step-by-step tutorials for common real-world scenarios. Each tutorial is self-contained and includes a baseline comparison, runnable code, visualizations, and practical notes.

These go deeper than the [Examples](../examples/) section — every tutorial covers the full workflow from raw data to a tuned, evaluated model.

## Gradient Boosting Libraries

| Tutorial | What it covers |
|----------|---------------|
| [Tune XGBoost](./tune-xgboost) | 9-parameter XGBoost search, adaptive schedules, feature importance, 3-way comparison |
| [Tune LightGBM](./tune-lightgbm) | 9-parameter LightGBM search, `num_leaves`/`max_depth` interaction, parameter scatter plots |
| [Tune CatBoost](./tune-catboost) | 7-parameter CatBoost search, `bagging_temperature`, `border_count`, GPU tip |

## Feature Selection

| Tutorial | What it covers |
|----------|---------------|
| [Comprehensive Feature Selection](./feature-selection) | 3-stage workflow: select on 50 features, retune on selected subset, validate with a second estimator |

## Imbalanced Data

| Tutorial | What it covers |
|----------|---------------|
| [Imbalanced Classification](./imbalanced-classification) | 95/5 imbalance, `class_weight` as search param, `balanced_accuracy` scoring, confusion matrices |

## Outlier Detection

| Tutorial | What it covers |
|----------|---------------|
| [Isolation Forest](./isolation-forest) | Custom scorer from `score_samples`, 4-param search, anomaly contour plots, ROC curve |

## See Also

- [Examples](../examples/) — shorter end-to-end examples for common use cases
- [User Guide](../guide/when-to-use) — decision guide for choosing a search method
- [API Reference](../api/gasearchcv) — full parameter documentation
