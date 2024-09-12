.. -*- mode: rst -*-

|Tests|_ |Codecov|_ |PythonVersion|_ |PyPi|_ |Docs|_

.. |Tests| image:: https://github.com/rodrigo-arenas/Sklearn-genetic-opt/actions/workflows/ci-tests.yml/badge.svg?branch=master
.. _Tests: https://github.com/rodrigo-arenas/Sklearn-genetic-opt/actions/workflows/ci-tests.yml

.. |Codecov| image:: https://codecov.io/gh/rodrigo-arenas/Sklearn-genetic-opt/branch/master/graphs/badge.svg?branch=master&service=github
.. _Codecov: https://codecov.io/github/rodrigo-arenas/Sklearn-genetic-opt?branch=master

.. |PythonVersion| image:: https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue
.. _PythonVersion : https://www.python.org/downloads/

.. |PyPi| image:: https://badge.fury.io/py/sklearn-genetic-opt.svg
.. _PyPi: https://badge.fury.io/py/sklearn-genetic-opt

.. |Docs| image:: https://readthedocs.org/projects/sklearn-genetic-opt/badge/?version=latest
.. _Docs: https://sklearn-genetic-opt.readthedocs.io/en/latest/?badge=latest

.. |Contributors| image:: https://contributors-img.web.app/image?repo=rodrigo-arenas/sklearn-genetic-opt
.. _Contributors: https://github.com/rodrigo-arenas/Sklearn-genetic-opt/graphs/contributors


.. image:: https://github.com/rodrigo-arenas/Sklearn-genetic-opt/blob/master/docs/logo.png?raw=true

Sklearn-genetic-opt
###################

scikit-learn models hyperparameters tuning and feature selection, using evolutionary algorithms.

This is meant to be an alternative to popular methods inside scikit-learn such as Grid Search and Randomized Grid Search
for hyperparameters tuning, and from RFE (Recursive Feature Elimination), Select From Model for feature selection.

**Table of Contents**
######################

- Sklearn-genetic-opt Overview
  - Main Features
  - Demos on Features
- Installation
  - Basic Installation
  - Full Installation with Extras
- Usage
  - Hyperparameters Tuning
  - Feature Selection
- Documentation
  - Stable
  - Latest
  - Development
- Changelog
- Important Links
- Source Code
- Contributing
- Testing


Sklearn-genetic-opt uses evolutionary algorithms from the `DEAP <https://deap.readthedocs.io/en/master/>`_  (Distributed Evolutionary Algorithms in Python) package to choose the set of hyperparameters that
optimizes (max or min) the cross-validation scores, it can be used for both regression and classification problems.

Documentation is available `here <https://sklearn-genetic-opt.readthedocs.io/>`_

Main Features:
##############

* **GASearchCV**: Main class of the package for hyperparameters tuning, holds the evolutionary cross-validation optimization routine.
* **GAFeatureSelectionCV**: Main class of the package for feature selection.
* **Algorithms**: Set of different evolutionary algorithms to use as an optimization procedure.
* **Callbacks**: Custom evaluation strategies to generate early stopping rules,
  logging (into TensorBoard, .pkl files, etc) or your custom logic.
* **Schedulers**: Adaptive methods to control learning parameters.
* **Plots**: Generate pre-defined plots to understand the optimization process.
* **MLflow**: Build-in integration with mlflow to log all the hyperparameters, cv-scores and the fitted models.

Demos on Features:
##################

Visualize the progress of your training:

.. image:: https://github.com/rodrigo-arenas/Sklearn-genetic-opt/blob/master/docs/images/progress_bar.gif?raw=true

Real-time metrics visualization and comparison across runs:

.. image:: https://github.com/rodrigo-arenas/Sklearn-genetic-opt/blob/master/docs/images/tensorboard_log.png?raw=true

Sampled distribution of hyperparameters:

.. image:: https://github.com/rodrigo-arenas/Sklearn-genetic-opt/blob/master/docs/images/density.png?raw=true

Artifacts logging:

.. image:: https://github.com/rodrigo-arenas/Sklearn-genetic-opt/blob/master/docs/images/mlflow_artifacts_4.png?raw=true


Usage:
######

Install sklearn-genetic-opt

It's advised to install sklearn-genetic using a virtual env, inside the env use::

   pip install sklearn-genetic-opt

If you want to get all the features, including plotting, tensorboard and mlflow logging capabilities,
install all the extra packages::

    pip install sklearn-genetic-opt[all]


Example: Hyperparameters Tuning
###############################

.. code-block:: python


   from sklearn_genetic import GASearchCV
   from sklearn_genetic.space import Continuous, Categorical, Integer
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split, StratifiedKFold
   from sklearn.datasets import load_digits
   from sklearn.metrics import accuracy_score

   data = load_digits()
   n_samples = len(data.images)
   X = data.images.reshape((n_samples, -1))
   y = data['target']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

   clf = RandomForestClassifier()

   # Defines the possible values to search
   param_grid = {'min_weight_fraction_leaf': Continuous(0.01, 0.5, distribution='log-uniform'),
                 'bootstrap': Categorical([True, False]),
                 'max_depth': Integer(2, 30),
                 'max_leaf_nodes': Integer(2, 35),
                 'n_estimators': Integer(100, 300)}

   # Seed solutions
   warm_start_configs = [
              {"min_weight_fraction_leaf": 0.02, "bootstrap": True, "max_depth": None, "n_estimators": 100},
              {"min_weight_fraction_leaf": 0.4, "bootstrap": True, "max_depth": 5, "n_estimators": 200},
       ]

   cv = StratifiedKFold(n_splits=3, shuffle=True)

   evolved_estimator = GASearchCV(estimator=clf,
                                  cv=cv,
                                  scoring='accuracy',
                                  population_size=20,
                                  generations=35,
                                  param_grid=param_grid,
                                  n_jobs=-1,
                                  verbose=True,
                                  use_cache=True,
                                  warm_start_configs=warm_start_configs,
                                  keep_top_k=4)

   # Train and optimize the estimator
   evolved_estimator.fit(X_train, y_train)
   # Best parameters found
   print(evolved_estimator.best_params_)
   # Use the model fitted with the best parameters
   y_predict_ga = evolved_estimator.predict(X_test)
   print(accuracy_score(y_test, y_predict_ga))

   # Saved metadata for further analysis
   print("Stats achieved in each generation: ", evolved_estimator.history)
   print("Best k solutions: ", evolved_estimator.hof)


Example: Feature Selection
##########################

.. code:: python3

    from sklearn_genetic import GAFeatureSelectionCV, ExponentialAdapter
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score
    import numpy as np

    data = load_iris()
    X, y = data["data"], data["target"]

    # Add random non-important features
    noise = np.random.uniform(5, 10, size=(X.shape[0], 5))
    X = np.hstack((X, noise))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

    clf = SVC(gamma='auto')
    mutation_scheduler = ExponentialAdapter(0.8, 0.2, 0.01)
    crossover_scheduler = ExponentialAdapter(0.2, 0.8, 0.01)

    evolved_estimator = GAFeatureSelectionCV(
        estimator=clf,
        scoring="accuracy",
        population_size=30,
        generations=20,
        mutation_probability=mutation_scheduler,
        crossover_probability=crossover_scheduler,
        n_jobs=-1)

    # Train and select the features
    evolved_estimator.fit(X_train, y_train)

    # Features selected by the algorithm
    features = evolved_estimator.support_
    print(features)

    # Predict only with the subset of selected features
    y_predict_ga = evolved_estimator.predict(X_test)
    print(accuracy_score(y_test, y_predict_ga))

    # Transform the original data to the selected features
    X_reduced = evolved_estimator.transform(X_test)

Changelog
#########

See the `changelog <https://sklearn-genetic-opt.readthedocs.io/en/latest/release_notes.html>`__
for notes on the changes of Sklearn-genetic-opt

Important links
###############

- Official source code repo: https://github.com/rodrigo-arenas/Sklearn-genetic-opt/
- Download releases: https://pypi.org/project/sklearn-genetic-opt/
- Issue tracker: https://github.com/rodrigo-arenas/Sklearn-genetic-opt/issues
- Stable documentation: https://sklearn-genetic-opt.readthedocs.io/en/stable/

Source code
###########

You can check the latest development version with the command::

   git clone https://github.com/rodrigo-arenas/Sklearn-genetic-opt.git

Install the development dependencies::
  
  pip install -r dev-requirements.txt
  
Check the latest in-development documentation: https://sklearn-genetic-opt.readthedocs.io/en/latest/

Contributing
############

Contributions are more than welcome!
There are several opportunities on the ongoing project, so please get in touch if you would like to help out.
Make sure to check the current issues and also
the `Contribution guide <https://github.com/rodrigo-arenas/Sklearn-genetic-opt/blob/master/CONTRIBUTING.md>`_.

Big thanks to the people who are helping with this project!

|Contributors|_

Testing
#######

After installation, you can launch the test suite from outside the source directory::

   pytest sklearn_genetic

