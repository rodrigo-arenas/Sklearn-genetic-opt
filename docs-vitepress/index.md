---
layout: home
title: "sklearn-genetic-opt | scikit-learn Tuning & Feature Selection"
titleTemplate: false
description: "Tune scikit-learn hyperparameters with GASearchCV and select features with GAFeatureSelectionCV. Install the Python package and compare five optimization backends."

hero:
  name: sklearn-genetic-opt
  text: Evolutionary tuning and feature selection
  tagline: Tune hyperparameters or select features by recombining candidates and scoring them with cross-validation.
  actions:
    - theme: brand
      text: Run Your First Search
      link: /versions/0.16/guide/basic-usage
    - theme: alt
      text: Compare Optimizers
      link: /versions/0.16/benchmarks/optimizer-catalog
    - theme: alt
      text: Browse Recipes
      link: /versions/0.16/recipes/
    - theme: alt
      text: View on GitHub
      link: https://github.com/rodrigo-arenas/Sklearn-genetic-opt

features:
  - title: Familiar sklearn API
    details: GASearchCV supports fit, predict, and best_params_ in the scikit-learn search workflow. Define parameter ranges and a validation budget to get started.
    link: /versions/0.16/guide/basic-usage
    linkText: Quick Start
  - title: Evolves Whole Configurations
    details: Genetic operators recombine promising parameter configurations and retain strong candidates across generations. Compare against a random-search baseline on your data.
    link: /versions/0.16/guide/how-hyperparameter-optimization-works
    linkText: How It Works
  - title: Wrapper Feature Selection
    details: GAFeatureSelectionCV evaluates feature subsets through cross-validation. Control subset size and check the result on untouched data.
    link: /versions/0.16/tutorials/feature-selection
    linkText: Feature Selection Tutorial
  - title: Budgeted Search and Checkpoints
    details: Cap fresh cross-validation calls and recover saved work across training sessions. The tutorial shows how to count search, finalist, and refit work separately.
    link: /versions/latest/tutorials/budgeted-optimizer-search
    linkText: Budget and Resume a Search
---
