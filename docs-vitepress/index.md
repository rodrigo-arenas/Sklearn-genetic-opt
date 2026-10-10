---
layout: home
title: Evolutionary Hyperparameter Tuning and Feature Selection
description: "Tune scikit-learn hyperparameters and select feature subsets with evolutionary algorithms. Learn when to use GASearchCV and GAFeatureSelectionCV, with runnable examples and API reference."

hero:
  name: sklearn-genetic-opt
  text: Hyperparameter Tuning for scikit-learn
  tagline: Search hyperparameters and feature subsets with evolutionary algorithms. Use familiar scikit-learn estimators, cross-validation, and scoring APIs.
  image:
    src: /brand/icon.svg
    alt: sklearn-genetic-opt logo
  actions:
    - theme: brand
      text: Get Started →
      link: /versions/0.14/
    - theme: alt
      text: Browse Recipes
      link: /versions/0.14/recipes/
    - theme: alt
      text: View on GitHub
      link: https://github.com/rodrigo-arenas/Sklearn-genetic-opt

features:
  - icon: 🔁
    title: Familiar sklearn API
    details: GASearchCV supports fit, predict, and best_params_ in the scikit-learn search workflow. Define parameter ranges and a validation budget to get started.
    link: /versions/0.14/guide/basic-usage
    linkText: Quick Start
  - icon: 🧬
    title: Evolves Whole Configurations
    details: Genetic operators recombine promising parameter configurations and retain strong candidates across generations. Compare against a random-search baseline on your data.
    link: /versions/0.14/guide/how-hyperparameter-optimization-works
    linkText: How It Works
  - icon: 🎯
    title: Wrapper Feature Selection
    details: GAFeatureSelectionCV evaluates feature subsets through cross-validation. Control subset size and check the result on untouched data.
    link: /versions/0.14/tutorials/feature-selection
    linkText: Feature Selection Tutorial
  - icon: 📊
    title: Built-in Monitoring
    details: Use callbacks for early stopping and inspect each generation. Optional integrations include MLflow and TensorBoard.
    link: /versions/0.14/guide/callbacks
    linkText: Callbacks Guide
---
