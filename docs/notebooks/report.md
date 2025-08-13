# Open-Source Contribution Report

## Project Chosen and Why
- **Repository:** `rodrigo-arenas/Sklearn-genetic-opt`
- **Why:** Active ML project with recent commits. Directly related to hyperparameter tuning and feature selection using evolutionary algorithms. They welcome example and documentation contributions.

## Problem Addressed
The project added native outlier detection support in v0.12.0, but the documentation lacked a dedicated example notebook.

## What I Did
- Created a new Jupyter notebook: `Outlier_Detection_IsolationForest_GASearchCV.ipynb`
- Demonstrates unsupervised tuning of `IsolationForest` with `GASearchCV(scoring=None)`
- Includes visualization (`plot_fitness_evolution`) and optional ROC-AUC evaluation.

## How I Tested
- Ran the notebook locally end-to-end.
- Verified that best parameters were found, plot rendered, and ROC-AUC score was reasonable.

## What I Learned
- How `sklearn-genetic-opt` handles unsupervised scorers.
- Practical tips for using GA search for ML models.
- How to integrate notebooks into an open-source documentation workflow.

## Links
- Repository: https://github.com/rodrigo-arenas/Sklearn-genetic-opt
- Related issue: “Improve documentation and examples”
- Pull Request: *(add your PR link here)*
