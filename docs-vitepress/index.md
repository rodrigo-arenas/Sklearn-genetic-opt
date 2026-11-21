---
layout: home
title: Genetic Algorithm Hyperparameter Tuning for scikit-learn
description: "Tune scikit-learn hyperparameters and select features with genetic algorithms. GASearchCV and GAFeatureSelectionCV: examples, tutorials and API."

hero:
  name: sklearn-genetic-opt
  text: Evolutionary search for scikit-learn
  tagline: Tune hyperparameters or select features by recombining candidates and scoring them with cross-validation.
  actions:
    - theme: brand
      text: Run Your First Search
      link: /versions/0.16/guide/basic-usage
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
  - title: Built-in Monitoring
    details: Use callbacks for early stopping and inspect each generation. Optional integrations include MLflow and TensorBoard.
    link: /versions/0.16/guide/callbacks
    linkText: Callbacks Guide
---
