---
layout: home
title: sklearn-genetic-opt
titleTemplate: Evolutionary hyperparameter tuning for scikit-learn

hero:
  name: sklearn-genetic-opt
  text: Evolutionary hyperparameter tuning
  tagline: Tune XGBoost, LightGBM, CatBoost, and any scikit-learn estimator using genetic algorithms — evolutionary hyperparameter search and wrapper-based feature selection, powered by DEAP.
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
  - title: Boost Library Support
    details: Works with XGBoost, LightGBM, and CatBoost out of the box — comprehensive tutorials tuning 7–9 hyperparameters with parameter interactions the GA captures naturally.
  - title: Imbalanced Learning
    details: Tune class_weight as a search parameter alongside model hyperparameters. Optimize balanced_accuracy or F1 directly instead of misleading accuracy.
  - title: Comprehensive Tutorials
    details: Step-by-step walkthroughs for gradient-boosting libraries, multi-stage feature selection, and imbalanced classification — each with baseline comparisons and visualizations.
---
