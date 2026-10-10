---
title: "Hyperparameter Tuning Recipes: Copy-Paste Solutions"
description: "Quick, copy-paste ready hyperparameter tuning recipes for scikit-learn, XGBoost, LightGBM, CatBoost, and more. Each recipe solves one specific problem."
---

:::warning Development version
You are reading the **latest (development)** docs. For stable documentation, see [stable](/stable/).
:::

# Hyperparameter Tuning Recipes

Recipes are short, copy-paste ready solutions to specific problems. Each recipe answers exactly one question — run it, adapt it, move on.

::: tip Recipes vs Tutorials vs Examples
| | Recipes | [Tutorials](../tutorials/) | [Examples](../examples/) |
|---|---|---|---|
| **Purpose** | Solve one specific problem | End-to-end conceptual walkthrough | Demonstrate a library feature |
| **Length** | 5–10 min | 20–40 min | 5–15 min |
| **Code** | Copy-paste ready | Guided walkthrough | Feature-focused snippet |
:::

## Classification

| Recipe | What it solves |
|--------|---------------|
| [Tune RandomForestClassifier](./classification/random-forest-classifier) | 7-parameter joint search with `class_weight` |
| [Tune LogisticRegression](./classification/logistic-regression) | Solver/penalty compatibility, multi-class |
| [Tune SVC](./classification/svm-classifier) | C/gamma with mandatory scaling in a Pipeline |
| [Tune XGBClassifier](./classification/xgboost-classifier) | 9 params, CPU oversubscription fix, `n_jobs=1` |
| [Tune LGBMClassifier](./classification/lightgbm-classifier) | `num_leaves`/`max_depth` interaction |
| [Tune CatBoostClassifier](./classification/catboost-classifier) | Categorical columns, `bagging_temperature` |
| [Tune HistGradientBoostingClassifier](./classification/histgbm-classifier) | Fast sklearn GBM, `max_leaf_nodes` |
| [Tune ExtraTreesClassifier](./classification/extra-trees-classifier) | High variance vs RF, `max_features` |

## Regression

| Recipe | What it solves |
|--------|---------------|
| [Tune RandomForestRegressor](./regression/random-forest-regressor) | `min_samples_leaf` vs `min_samples_split`, MAE scoring |
| [Tune XGBRegressor](./regression/xgboost-regressor) | Regression with custom `eval_metric` |
| [Tune LGBMRegressor](./regression/lightgbm-regressor) | RMSE scoring, `min_child_samples` |
| [Tune CatBoostRegressor](./regression/catboost-regressor) | Regression with ordered boosting |
| [Tune SGDRegressor](./regression/sgd-regressor) | `learning_rate`/`eta0` interaction, log-uniform |
| [Tune ElasticNet](./regression/elasticnet) | `l1_ratio`/`alpha` interaction, log-uniform |

## Feature Selection

| Recipe | What it solves |
|--------|---------------|
| [Select Features on 50+ Columns](./feature-selection/high-dimensional) | Threshold strategy for large feature sets |
| [Combine Feature Selection + Hyperparameter Tuning](./feature-selection/select-then-tune) | Two-stage: select features, then retune |
| [Use a Custom Scorer for Feature Selection](./feature-selection/custom-scorer) | Wrap `roc_auc` with a feature-count penalty |
| [Feature Selection with Cross-Validation](./feature-selection/cv-selection) | Per-fold selection to avoid leakage |

## Pipelines

| Recipe | What it solves |
|--------|---------------|
| [Tune a Preprocessing + Estimator Pipeline](./pipelines/preprocessing-pipeline) | `StandardScaler` + `SVC` with step prefix |
| [Tune a ColumnTransformer Pipeline](./pipelines/column-transformer) | Mixed types, `preprocessor__...` param paths |
| [Tune Imputer Strategy as a Hyperparameter](./pipelines/imputer-strategy) | Search over `mean`/`median`/`most_frequent` |
| [Tune Polynomial Features Degree](./pipelines/polynomial-features) | Include degree and interaction-only as params |

## Scoring Metrics

| Recipe | What it solves |
|--------|---------------|
| [Tune for F1 Score (Binary)](./metrics/f1-binary) | `scoring="f1"`, thresholds, confusion matrix |
| [Tune for ROC-AUC](./metrics/roc-auc) | `scoring="roc_auc"`, probability calibration |
| [Tune for Balanced Accuracy](./metrics/balanced-accuracy) | Class imbalance, `class_weight` as a param |
| [Tune for MAE (Regression)](./metrics/mae) | `scoring="neg_mean_absolute_error"` |
| [Tune for RMSE (Regression)](./metrics/rmse) | Custom `neg_root_mean_squared_error` scorer |

## Integrations

| Recipe | What it solves |
|--------|---------------|
| [Log Every Candidate to MLflow](./integrations/mlflow-logging) | `MlflowCallback`, nested runs, param logging |
| [Parallelize with Joblib](./integrations/joblib-parallel) | `n_jobs`, `loky` vs `threading` backend |
| [Run in a Jupyter Notebook](./integrations/jupyter-notebook) | `verbose=True`, tqdm progress, plot inline |

## Advanced

| Recipe | What it solves |
|--------|---------------|
| [Seed a Search with Known-Good Params](./advanced/warm-start) | `warm_start_configs` to skip cold start |
| [Stop Early When Fitness Plateaus](./advanced/early-stopping-consecutive) | `ConsecutiveStopping` callback setup |
| [Stop After a Time Budget](./advanced/time-budget) | `TimerStopping` for wall-clock limits |
| [Resume a Stopped Search](./advanced/checkpointing) | Save/load state, continue from checkpoint |
| [Write a Custom Scoring Function](./advanced/custom-scorer) | `make_scorer` + unsupervised metrics |

## See Also

- [Tutorials](../tutorials/) — end-to-end model walkthroughs with explanations
- [Examples](../examples/) — focused library feature demonstrations  
- [API Reference](../api/gasearchcv) — full parameter documentation
- [Common Mistakes](../guide/common-mistakes) — pitfalls to avoid
