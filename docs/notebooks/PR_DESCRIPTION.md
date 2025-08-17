Description:

### Summary
Adds a new Jupyter notebook: Outlier_Detection_IsolationForest_GASearchCV.ipynb to the documentation.

### Motivation
Recently, the project added support for unsupervised models with scoring=None, but there was no example demonstrating how to use GASearchCV for tuning outlier detection models like IsolationForest. This notebook provides a clear, end-to-end example for users.

### What’s Included
Synthetic data generation with injected outliers

Hyperparameter tuning for IsolationForest using GASearchCV (with genetic algorithms)

Visualization of fitness evolution during search

Evaluation of results via ROC-AUC, precision, recall, and F1-score

Plots demonstrating model separation of normal points and outliers

### Documentation
Notebook is placed in docs/notebooks/ and referenced in docs/index.rst so users can find it easily.

  ### Related Issue
Addresses improved documentation/examples for unsupervised model support (scoring=None).
