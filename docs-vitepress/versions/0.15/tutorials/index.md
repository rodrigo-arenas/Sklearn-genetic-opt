---
title: "Hyperparameter Tuning Tutorials: Random Forest, XGBoost, LightGBM, and More"
description: "End-to-end hyperparameter tuning tutorials for scikit-learn models, XGBoost, LightGBM, CatBoost, and more — using genetic algorithms with sklearn-genetic-opt."
---

# Hyperparameter Tuning Tutorials

Step-by-step tutorials for common real-world scenarios. Each tutorial is self-contained and includes a baseline comparison, runnable code, visualizations, and practical notes.

::: tip Tutorials vs Examples
**Tutorials** are end-to-end walkthroughs of a complete real-world task — from raw data to a tuned, evaluated model. **[Examples](../examples/)** are shorter, focused recipes that each demonstrate a single feature you can drop into your own code.
:::

## scikit-learn Estimators

| Tutorial | Difficulty | What it covers |
|----------|-----------|---------------|
| [Random Forest Hyperparameter Tuning](./tune-random-forest) | Intermediate | 7-parameter joint search, which params matter, classification and regression, baseline comparison |
| [Gradient Boosting Hyperparameter Tuning](./tune-gradient-boosting) | Intermediate | HistGradientBoosting vs classic GBM, max_leaf_nodes vs max_depth, speed comparison |
| [Logistic Regression Hyperparameter Tuning](./tune-logistic-regression) | Beginner | C, penalty, solver compatibility, multi-penalty search with SAGA |
| [SVM Hyperparameter Tuning (C, kernel, gamma)](./tune-svm) | Intermediate | C–gamma interaction, Pipeline with StandardScaler, RBF vs linear kernel, scaling limits |

## Gradient Boosting Libraries

| Tutorial | Difficulty | What it covers |
|----------|-----------|---------------|
| [XGBoost Hyperparameter Tuning](./tune-xgboost) | Intermediate | 9-parameter XGBoost search, adaptive schedules, feature importance, 3-way comparison |
| [LightGBM Hyperparameter Tuning](./tune-lightgbm) | Intermediate | 9-parameter LightGBM search, `num_leaves`/`max_depth` interaction, parameter scatter plots |
| [CatBoost Hyperparameter Tuning](./tune-catboost) | Intermediate | 7-parameter CatBoost search, `bagging_temperature`, `border_count`, GPU tip |

## Feature Selection

| Tutorial | Difficulty | What it covers |
|----------|-----------|---------------|
| [Feature Selection with Genetic Algorithms](./feature-selection) | Advanced | 3-stage workflow: select on 50 features, retune on selected subset, validate with a second estimator |

## Imbalanced Data

| Tutorial | Difficulty | What it covers |
|----------|-----------|---------------|
| [Hyperparameter Tuning for Imbalanced Datasets](./imbalanced-classification) | Intermediate | 95/5 imbalance, `class_weight` as search param, `balanced_accuracy` scoring, confusion matrices |

## Outlier Detection

| Tutorial | Difficulty | What it covers |
|----------|-----------|---------------|
| [Isolation Forest Hyperparameter Tuning](./isolation-forest) | Advanced | Custom scorer from `score_samples`, 4-param search, anomaly contour plots, ROC curve |

## Not Sure Where to Start?

:::tip Recommended reading order
1. [How Hyperparameter Optimization Works](../guide/how-hyperparameter-optimization-works) — theory and method comparison
2. [When to Use Genetic Algorithm Search](../guide/when-to-use) — decide if GASearchCV fits your problem
3. [Getting Started with GASearchCV](../guide/basic-usage) — run your first search
4. Pick the tutorial for your model above
:::

## See Also

- [Examples](../examples/) — shorter end-to-end examples for common use cases
- [Comparisons](../comparisons/) — honest benchmarks: GA vs Random vs Bayesian
- [Common Hyperparameter Tuning Mistakes](../guide/common-mistakes) — avoid the most frequent pitfalls
- [Choosing the Right Search Space](../guide/choosing-search-spaces) — define good parameter bounds
- [API Reference](../api/gasearchcv) — full parameter documentation
