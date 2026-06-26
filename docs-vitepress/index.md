---
layout: home
title: sklearn-genetic-opt — Genetic Algorithm Hyperparameter Tuning for scikit-learn
description: "Tune Random Forest, XGBoost, LightGBM, CatBoost, and any scikit-learn estimator using genetic algorithms. Evolutionary hyperparameter search and feature selection powered by DEAP."
titleTemplate: Evolutionary hyperparameter tuning for scikit-learn

hero:
  name: sklearn-genetic-opt
  text: Genetic Algorithm Hyperparameter Tuning
  tagline: Tune Random Forest, XGBoost, LightGBM, and any scikit-learn estimator with evolutionary search — handles parameter interactions that GridSearchCV and RandomizedSearchCV miss.
  image:
    src: /logo.png
    alt: sklearn-genetic-opt logo
  actions:
    - theme: brand
      text: Get Started
      link: /stable/
    - theme: alt
      text: How It Works
      link: /versions/latest/guide/how-hyperparameter-optimization-works
    - theme: alt
      text: View on GitHub
      link: https://github.com/rodrigo-arenas/Sklearn-genetic-opt

features:
  - title: Drop-in GridSearchCV Replacement
    details: GASearchCV follows the same fit/predict/best_params_ API as GridSearchCV — replace it in one line and keep your entire sklearn workflow.
  - title: Handles Parameter Interactions
    details: Genetic algorithms evaluate complete configurations, not individual parameters — naturally finding learning_rate × n_estimators and other cross-parameter sweet spots.
  - title: Wrapper-Based Feature Selection
    details: GAFeatureSelectionCV finds the compact feature subset that maximises cross-validated score — outperforms filter methods on datasets with feature interactions.
  - title: Smart Initialization
    details: Latin hypercube sampling, estimator defaults, and warm-start seeds produce a better initial population than pure random — fewer wasted evaluations.
  - title: Early Stopping
    details: Built-in callbacks (ConsecutiveStopping, DeltaThreshold, TimerStopping) stop the search automatically when it converges — no manual iteration counting.
  - title: Diversity Control
    details: Adaptive mutation/crossover schedules, random immigrants, and fitness sharing prevent the population from converging to a local optimum.
  - title: MLflow & TensorBoard
    details: Every candidate is logged as a child run in MLflow — compare experiments, track parameter distributions, and store fitted models automatically.
  - title: XGBoost, LightGBM, CatBoost
    details: Detailed tutorials for every major gradient boosting library — 7–9 parameters tuned jointly, with CPU oversubscription tips and baseline comparisons.
  - title: Comprehensive Educational Docs
    details: Not just API docs — tutorials for Random Forest, XGBoost, LightGBM, Logistic Regression, SVM, feature selection, imbalanced data, outlier detection, and more.
---
