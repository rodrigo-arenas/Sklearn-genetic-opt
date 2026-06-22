---
layout: home
title: sklearn-genetic-opt
titleTemplate: Evolutionary hyperparameter tuning for scikit-learn

hero:
  name: sklearn-genetic-opt
  text: Evolutionary hyperparameter tuning
  tagline: Tune scikit-learn models and select features using genetic algorithms powered by DEAP.
  image:
    src: /logo.png
    alt: sklearn-genetic-opt logo
  actions:
    - theme: brand
      text: Get Started
      link: /versions/0.13/
    - theme: alt
      text: View on GitHub
      link: https://github.com/rodrigo-arenas/Sklearn-genetic-opt

features:
  - title: GASearchCV
    details: Hyperparameter search across classification, regression, and outlier-detection estimators using evolutionary operators.
  - title: GAFeatureSelectionCV
    details: Wrapper-based feature selection with cross-validation — find the compact subset that maximises your score.
  - title: Smart Initialization
    details: Latin hypercube sampling, estimator defaults, and warm-start seeds produce a better initial population than pure random.
  - title: Diversity Control
    details: Adaptive mutation/crossover, random immigrants, and fitness sharing prevent premature convergence.
  - title: Callbacks & MLflow
    details: Early stopping, progress bars, checkpoints, TensorBoard, and MLflow 3 logging out of the box.
  - title: scikit-learn Compatible
    details: Follows the familiar fit/predict/best_params_ API — drop it in wherever you'd use GridSearchCV.
---
