.. -*- mode: rst -*-

|Tests|_ |Codecov|_ |PythonVersion|_ |PyPi|_ |Docs|_

.. |Tests| image:: https://github.com/rodrigo-arenas/Sklearn-genetic-opt/actions/workflows/ci-tests.yml/badge.svg?branch=master
.. _Tests: https://github.com/rodrigo-arenas/Sklearn-genetic-opt/actions/workflows/ci-tests.yml

.. |Codecov| image:: https://codecov.io/gh/rodrigo-arenas/Sklearn-genetic-opt/branch/master/graphs/badge.svg?branch=master&service=github
.. _Codecov: https://codecov.io/github/rodrigo-arenas/Sklearn-genetic-opt?branch=master

.. |PythonVersion| image:: https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue
.. _PythonVersion : https://www.python.org/downloads/
.. |PyPi| image:: https://badge.fury.io/py/sklearn-genetic-opt.svg
.. _PyPi: https://badge.fury.io/py/sklearn-genetic-opt

.. |Docs| image:: https://readthedocs.org/projects/sklearn-genetic-opt/badge/?version=latest
.. _Docs: https://sklearn-genetic-opt.readthedocs.io/en/latest/?badge=latest

.. image:: https://github.com/rodrigo-arenas/Sklearn-genetic-opt/blob/master/docs/logo.png?raw=true

Sklearn-genetic-opt
###################

scikit-learn models hyperparameters tuning, using evolutionary algorithms.

This is meant to be an alternative from popular methods inside scikit-learn such as Grid Search and Randomized Grid Search.

Sklearn-genetic-opt uses evolutionary algorithms from the deap package to choose set of hyperparameters that
optimizes (max or min) the cross validation scores, it can be used for both regression and classification problems.

Documentation is available `here <https://sklearn-genetic-opt.readthedocs.io/>`_

Sampled distribution of hyperparameters:

.. image:: https://github.com/rodrigo-arenas/Sklearn-genetic-opt/blob/master/demo/images/density.png?raw=true

Optimization progress in a regression problem:

.. image:: https://github.com/rodrigo-arenas/Sklearn-genetic-opt/blob/master/demo/images/fitness.png?raw=true


Main Features:
##############

* **GASearchCV**: Principal class of the package, holds the evolutionary cross validation optimization routine
* **Algorithms**: Set of different evolutionary algorithms to use as optimization procedure
* **Callbacks**: Custom evaluation strategies to generate Early Stopping rules
* **Plots**: Generate pre-define plots to understand the optimization process

Usage:
######

Install sklearn-genetic-opt

It's advised to install sklearn-genetic using a virtual env, inside the env use::

   pip install sklearn-genetic-opt

Example
#######

.. code-block:: python

   from sklearn_genetic import GASearchCV
   from sklearn_genetic.space import Continuous, Categorical, Integer
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split, StratifiedKFold
   from sklearn.datasets import load_digits
   from sklearn.metrics import accuracy_score
   import matplotlib.pyplot as plt

   data = load_digits()
   n_samples = len(data.images)
   X = data.images.reshape((n_samples, -1))
   y = data['target']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

   clf = RandomForestClassifier()

   param_grid = {'min_weight_fraction_leaf': Continuous(0.01, 0.5, distribution='log-uniform'),
                 'bootstrap': Categorical([True, False]),
                 'max_depth': Integer(2, 30),
                 'max_leaf_nodes': Integer(2, 35),
                 'n_estimators': Integer(100, 300)}

   cv = StratifiedKFold(n_splits=3, shuffle=True)

   evolved_estimator = GASearchCV(estimator=clf,
                                  cv=cv,
                                  scoring='accuracy',
                                  population_size=10,
                                  generations=35,
                                  param_grid=param_grid,
                                  n_jobs=-1,
                                  verbose=True,
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

^^^^^^^
Results
^^^^^^^

Log controlled by verbosity

.. image:: https://github.com/rodrigo-arenas/Sklearn-genetic-opt/blob/master/demo/images/log.JPG?raw=true

Changelog
#########

See the `changelog <https://sklearn-genetic-opt.readthedocs.io/en/latest/release_notes.html>`__
for notes on the changes of Sklearn-genetic-opt

Important links
###############

- Official source code repo: https://github.com/rodrigo-arenas/Sklearn-genetic-opt/
- Download releases: https://pypi.org/project/sklearn-genetic-opt/
- Issue tracker: https://github.com/rodrigo-arenas/Sklearn-genetic-opt/issues

Source code
###########

You can check the latest development version with the command::

   git clone https://github.com/rodrigo-arenas/Sklearn-genetic-opt.git

Contributing
############

Contributions are more than welcome!
There are lots of opportunities on the on going project, so please get in touch if you would like to help out.
Also check the `Contribution guide <https://github.com/rodrigo-arenas/Sklearn-genetic-opt/blob/master/CONTRIBUTING.md>`_

Testing
#######

After installation, you can launch the test suite from outside the source directory::

   pytest sklearn_genetic



