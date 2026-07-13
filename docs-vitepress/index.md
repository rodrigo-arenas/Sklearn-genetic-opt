---
layout: home
title: sklearn-genetic-opt — Evolutionary Hyperparameter Tuning for scikit-learn
description: "Genetic algorithm hyperparameter search and feature selection for scikit-learn. Tune Random Forest, XGBoost, LightGBM, SVM and any sklearn estimator with GASearchCV — the drop-in GridSearchCV replacement powered by evolutionary algorithms."
titleTemplate: Evolutionary hyperparameter tuning for scikit-learn

hero:
  name: sklearn-genetic-opt
  text: Hyperparameter Tuning for scikit-learn
  tagline: Find better parameters faster. Evolutionary search handles cross-parameter interactions that GridSearchCV and RandomizedSearchCV miss — with feature selection, callbacks, and MLflow built in.
  image:
    src: /brand/icon.svg
    alt: sklearn-genetic-opt logo
  actions:
    - theme: brand
      text: Get Started →
      link: /stable/
    - theme: alt
      text: Browse Recipes
      link: /versions/latest/recipes/
    - theme: alt
      text: View on GitHub
      link: https://github.com/rodrigo-arenas/Sklearn-genetic-opt

features:
  - icon: 🔁
    title: Drop-in sklearn Replacement
    details: GASearchCV follows the same fit / predict / best_params_ API as GridSearchCV. Replace it in one line and keep your entire pipeline unchanged.
    link: /versions/latest/guide/basic-usage
    linkText: Quick Start
  - icon: 🧬
    title: Evolves Whole Configurations
    details: Genetic operators evaluate complete hyperparameter sets — naturally discovering learning_rate × n_estimators interactions that one-at-a-time search misses.
    link: /versions/latest/guide/how-hyperparameter-optimization-works
    linkText: How It Works
  - icon: 🎯
    title: Wrapper Feature Selection
    details: GAFeatureSelectionCV finds the compact feature subset that maximises CV score — outperforms filter methods when features interact.
    link: /versions/latest/tutorials/feature-selection
    linkText: Feature Selection Tutorial
  - icon: 📊
    title: Built-in Monitoring
    details: Callbacks for early stopping, MLflow child-run logging, TensorBoard, and per-generation telemetry — out of the box.
    link: /versions/latest/guide/callbacks
    linkText: Callbacks Guide
---
