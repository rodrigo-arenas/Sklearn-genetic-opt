sklearn-genetic-opt
===================

Evolutionary hyperparameter search and feature selection for scikit-learn.
``GASearchCV`` tunes estimator parameters. ``GAFeatureSelectionCV`` searches for
subsets of input features. Both score candidates with cross-validation and
recombine promising candidates to build the next generation.

.. image:: https://github.com/rodrigo-arenas/Sklearn-genetic-opt/blob/master/docs-vitepress/public/brand/readme-banner-1280x320.png?raw=true
   :alt: sklearn-genetic-opt
   :width: 410
   :target: https://sklearngeneticopt.rodrigo-arenas.com/stable/

Use it when a grid would be too large and you can budget several generations
of model fits. For a small discrete grid, exhaustive search is simpler.

Install
-------

With Python 3.12 or newer and scikit-learn 1.5 or newer, install the release
used in the example:

.. code-block:: bash

   python -m pip install sklearn-genetic-opt==0.15.0

The development test matrix covers CPython 3.12–3.15. On 3.15, use
scikit-learn 1.9.1 or newer. See the `installation guide`_ for
conda and optional plotting and MLflow dependencies.

Run a first search
------------------

Tune a decision tree on the built-in iris dataset. Each candidate specifies a
tree depth and a minimum leaf size. Three-fold cross-validation scores each
candidate on the training data. The test set stays separate.

.. code-block:: python

   from sklearn.datasets import load_iris
   from sklearn.model_selection import train_test_split
   from sklearn.tree import DecisionTreeClassifier
   from sklearn_genetic import EvolutionConfig, GASearchCV, RuntimeConfig
   from sklearn_genetic.space import Integer

   X, y = load_iris(return_X_y=True)
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, stratify=y, random_state=42
   )

   search = GASearchCV(
       estimator=DecisionTreeClassifier(random_state=42),
       param_grid={"max_depth": Integer(1, 6), "min_samples_leaf": Integer(1, 5)},
       scoring="accuracy",
       cv=3,
       evolution_config=EvolutionConfig(population_size=6, generations=3),
       runtime_config=RuntimeConfig(n_jobs=1, verbose=False),
       random_state=42,
   )
   search.fit(X_train, y_train)
   print(search.best_params_)
   print(search.score(X_test, y_test))

To select columns instead, follow the `feature selection tutorial`_. It shows
how to inspect ``support_`` and evaluate the selected features on held-out data.

Development optimizer catalog
-----------------------------

On ``master``, both estimators accept ``optimizer_backend``. DEAP is the default;
SciPy, Nevergrad, CatCMAwM and pymoo share the estimators' cross-validation,
results and callbacks. From a development checkout, install the desired extra
with, for example, ``python -m pip install -e ".[pymoo]"`` and pass
``optimizer_backend="pymoo"`` to either estimator. The `backend tutorial`_
shows both tasks, and the `engine comparison`_ reports quality and actual CV cost.

.. _backend tutorial: https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/tutorials/optimizer-backends
.. _engine comparison: https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/benchmarks/optimizer-catalog

Documentation and project
-------------------------

* `Stable documentation`_, `model-specific tutorials`_ and `recipes`_.
* API references for `GASearchCV`_, `GAFeatureSelectionCV`_ and `search spaces`_.
* `Search method comparisons`_, `benchmarks`_ and the `plotting gallery`_.
* `Contributing and running tests`_, `software citation`_ and `MIT license`_.

.. _installation guide: https://sklearngeneticopt.rodrigo-arenas.com/versions/0.15/guide/installation
.. _feature selection tutorial: https://sklearngeneticopt.rodrigo-arenas.com/versions/0.15/tutorials/feature-selection
.. _Stable documentation: https://sklearngeneticopt.rodrigo-arenas.com/stable/
.. _model-specific tutorials: https://sklearngeneticopt.rodrigo-arenas.com/versions/0.15/tutorials/
.. _recipes: https://sklearngeneticopt.rodrigo-arenas.com/versions/0.15/recipes/
.. _GASearchCV: https://sklearngeneticopt.rodrigo-arenas.com/versions/0.15/api/gasearchcv
.. _GAFeatureSelectionCV: https://sklearngeneticopt.rodrigo-arenas.com/versions/0.15/api/gafeatureselectioncv
.. _search spaces: https://sklearngeneticopt.rodrigo-arenas.com/versions/0.15/api/space
.. _Search method comparisons: https://sklearngeneticopt.rodrigo-arenas.com/versions/0.15/comparisons/
.. _benchmarks: https://sklearngeneticopt.rodrigo-arenas.com/versions/0.15/benchmarks/
.. _plotting gallery: https://sklearngeneticopt.rodrigo-arenas.com/versions/0.15/examples/plotting-gallery
.. _Contributing and running tests: https://github.com/rodrigo-arenas/Sklearn-genetic-opt/blob/master/CONTRIBUTING.md
.. _software citation: https://github.com/rodrigo-arenas/Sklearn-genetic-opt/blob/master/CITATION.cff
.. _MIT license: https://github.com/rodrigo-arenas/Sklearn-genetic-opt/blob/master/LICENSE
