.. -*- mode: rst -*-

|Tests|_ |Codecov|_ |PythonVersion|_ |PyPi|_ |Conda|_ |Docs|_

.. |Tests| image:: https://github.com/rodrigo-arenas/Sklearn-genetic-opt/actions/workflows/ci-tests.yml/badge.svg?branch=master
.. _Tests: https://github.com/rodrigo-arenas/Sklearn-genetic-opt/actions/workflows/ci-tests.yml

.. |Codecov| image:: https://codecov.io/gh/rodrigo-arenas/Sklearn-genetic-opt/branch/master/graphs/badge.svg?branch=master&service=github
.. _Codecov: https://codecov.io/github/rodrigo-arenas/Sklearn-genetic-opt?branch=master

.. |PythonVersion| image:: https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue
.. _PythonVersion: https://www.python.org/downloads/

.. |PyPi| image:: https://badge.fury.io/py/sklearn-genetic-opt.svg
.. _PyPi: https://badge.fury.io/py/sklearn-genetic-opt

.. |Conda| image:: https://img.shields.io/conda/vn/conda-forge/sklearn-genetic-opt.svg
.. _Conda: https://anaconda.org/conda-forge/sklearn-genetic-opt

.. |Docs| image:: https://img.shields.io/badge/docs-GitHub%20Pages-blue
.. _Docs: https://rodrigo-arenas.github.io/Sklearn-genetic-opt/

.. |Contributors| image:: https://contributors-img.web.app/image?repo=rodrigo-arenas/sklearn-genetic-opt
.. _Contributors: https://github.com/rodrigo-arenas/Sklearn-genetic-opt/graphs/contributors


.. image:: https://github.com/rodrigo-arenas/Sklearn-genetic-opt/blob/master/docs/logo.png?raw=true

sklearn-genetic-opt
###################

``sklearn-genetic-opt`` adds evolutionary optimization tools to the
scikit-learn workflow. It can tune hyperparameters with
``GASearchCV`` and select feature subsets with ``GAFeatureSelectionCV`` using
algorithms powered by `DEAP <https://deap.readthedocs.io/en/master/>`_.

The project is useful when a search space is mixed, irregular, expensive, or
not well served by an exhaustive grid. It follows familiar scikit-learn
patterns: define an estimator, define a search space, call ``fit``, inspect
``best_params_`` or ``support_``, and use the fitted object for prediction.

Documentation is available at
`rodrigo-arenas.github.io/Sklearn-genetic-opt <https://rodrigo-arenas.github.io/Sklearn-genetic-opt/>`_.

Highlights
##########

* ``GASearchCV`` for hyperparameter search across classification, regression,
  and supported outlier-detection estimators.
* ``GAFeatureSelectionCV`` for wrapper-based feature selection with
  cross-validation.
* Search spaces for integer, continuous, and categorical parameters.
* Smart initial populations with ``PopulationConfig(initializer="smart")``, including
  warm-start seeds, estimator defaults, Latin-hypercube numeric coverage,
  stratified categorical coverage, and duplicate avoidance.
* Adaptive mutation and crossover schedules.
* Optional local search, diversity control, random immigrants, and fitness
  sharing to improve exploration, avoid premature convergence, and refine good
  solutions.
* Parallel candidate evaluation with ``n_jobs`` and ``parallel_backend``.
* Evaluation caching, optimizer telemetry through ``history``, and fit-cost
  counters through ``fit_stats_``.
* Callbacks for early stopping, progress reporting, checkpoints, TensorBoard,
  and custom logic.
* Plotting helpers plus MLflow 3 logging support.

Installation
############

Install the core package with pip:

.. code-block:: bash

   pip install sklearn-genetic-opt

Or with conda from the conda-forge channel:

.. code-block:: bash

   conda install -c conda-forge sklearn-genetic-opt

Install optional plotting, MLflow, and TensorBoard integrations with pip:

.. code-block:: bash

   pip install sklearn-genetic-opt[all]

The conda package ships only the core dependencies. To use the optional
integrations in a conda environment, install the extras you need alongside it,
for example:

.. code-block:: bash

   conda install -c conda-forge sklearn-genetic-opt seaborn mlflow

Requirements
############

Core requirements:

* Python >= 3.12
* scikit-learn >= 1.9.0
* NumPy >= 2.4.6
* DEAP >= 1.4.4
* tqdm >= 4.68.3

Optional extras:

* Seaborn >= 0.13.2 for plots
* MLflow >= 3.14.0 for experiment logging
* TensorFlow >= 2.21.0 and TensorBoard >= 2.20.0,<2.21.0 for TensorBoard
  logging on Python < 3.14

Quick Start: Hyperparameter Search
##################################

.. code-block:: python

   from sklearn.datasets import load_iris
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import StratifiedKFold, train_test_split

   from sklearn_genetic import EvolutionConfig, GASearchCV, PopulationConfig, RuntimeConfig
   from sklearn_genetic.space import Categorical, Continuous, Integer

   X, y = load_iris(return_X_y=True)
   X_train, X_test, y_train, y_test = train_test_split(
       X,
       y,
       test_size=0.25,
       stratify=y,
       random_state=42,
   )

   param_grid = {
       "n_estimators": Integer(50, 200),
       "max_depth": Integer(2, 12),
       "max_features": Continuous(0.3, 1.0),
       "criterion": Categorical(["gini", "entropy", "log_loss"]),
   }

   search = GASearchCV(
       estimator=RandomForestClassifier(random_state=42),
       param_grid=param_grid,
       cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
       scoring="accuracy",
       evolution_config=EvolutionConfig(population_size=12, generations=8),
       population_config=PopulationConfig(initializer="smart"),
       runtime_config=RuntimeConfig(
           n_jobs=-1,
           parallel_backend="auto",
           use_cache=True,
           verbose=True,
       ),
   )

   search.fit(X_train, y_train)

   print(search.best_params_)
   print(search.best_score_)
   print(search.score(X_test, y_test))
   print(search.fit_stats_)

Quick Start: Feature Selection
##############################

.. code-block:: python

   import numpy as np
   from sklearn.datasets import load_iris
   from sklearn.metrics import accuracy_score
   from sklearn.model_selection import train_test_split
   from sklearn.svm import SVC

   from sklearn_genetic import (
       EvolutionConfig,
       GAFeatureSelectionCV,
       PopulationConfig,
       RuntimeConfig,
   )
   from sklearn_genetic.schedules import ExponentialAdapter

   X, y = load_iris(return_X_y=True)
   noise = np.random.default_rng(42).uniform(0, 1, size=(X.shape[0], 8))
   X = np.hstack([X, noise])

   X_train, X_test, y_train, y_test = train_test_split(
       X,
       y,
       test_size=0.25,
       stratify=y,
       random_state=42,
   )

   selector = GAFeatureSelectionCV(
       estimator=SVC(gamma="auto"),
       cv=3,
       scoring="accuracy",
       evolution_config=EvolutionConfig(
           population_size=12,
           generations=8,
           mutation_probability=ExponentialAdapter(0.8, 0.2, 0.05),
           crossover_probability=ExponentialAdapter(0.2, 0.8, 0.05),
       ),
       population_config=PopulationConfig(initializer="smart"),
       runtime_config=RuntimeConfig(n_jobs=-1, verbose=True),
   )

   selector.fit(X_train, y_train)

   print(selector.support_)
   print(accuracy_score(y_test, selector.predict(X_test)))
   print(selector.transform(X_test).shape)

Improving Search Quality
########################

The default ``PopulationConfig(initializer="smart")`` is recommended for most runs.
It improves early search coverage without reducing the number of generations
or population size.

For harder spaces, combine it with optimizer controls:

.. code-block:: python

   from sklearn_genetic import EvolutionConfig, OptimizationConfig, PopulationConfig, RuntimeConfig
   from sklearn_genetic.schedules import ExponentialAdapter, InverseAdapter

   search = GASearchCV(
       estimator=estimator,
       param_grid=param_grid,
       scoring="roc_auc",
       cv=3,
       evolution_config=EvolutionConfig(
           population_size=16,
           generations=12,
           crossover_probability=ExponentialAdapter(0.85, 0.45, 0.08),
           mutation_probability=InverseAdapter(0.18, 0.55, 0.12),
       ),
       population_config=PopulationConfig(
           initializer="smart",
           warm_start_configs=[{"C": 1.0, "class_weight": None}],
       ),
       runtime_config=RuntimeConfig(n_jobs=-1, parallel_backend="auto", use_cache=True),
       optimization_config=OptimizationConfig(
           local_search=True,
           local_search_top_k=2,
           local_search_steps=2,
           diversity_control=True,
           random_immigrants_fraction=0.15,
           fitness_sharing=True,
       ),
   )

Use:

* ``PopulationConfig.warm_start_configs`` when you already know useful candidate settings.
* ``OptimizationConfig(local_search=True)`` to refine the best candidates near the end of the run.
* ``OptimizationConfig(diversity_control=True)`` to react when the population collapses too early.
* ``OptimizationConfig(fitness_sharing=True)`` to keep multiple promising regions alive.
* ``fit_stats_`` to understand evaluation cost, cache hits, and skipped work.
* ``history`` to inspect fitness, diversity, stagnation, and optimizer
  telemetry by generation.

Parallelism
###########

``RuntimeConfig.n_jobs`` controls parallel execution:

* ``RuntimeConfig(n_jobs=1)`` runs sequentially.
* ``RuntimeConfig(n_jobs=-1)`` uses all available CPU cores.
* ``RuntimeConfig(n_jobs=k)`` uses ``k`` workers.

With ``RuntimeConfig(parallel_backend="auto")`` or
``RuntimeConfig(parallel_backend="population")``, unique candidates in the same
generation are evaluated in parallel and each candidate runs cross-validation
sequentially to avoid nested parallelism. Use
``RuntimeConfig(parallel_backend="cv")`` to evaluate candidates serially while
passing ``n_jobs`` to each candidate's cross-validation call.

Persistence and Checkpointing
#############################

Use ``ModelCheckpoint`` to write progress during long searches:

.. code-block:: python

   from sklearn_genetic.callbacks import ModelCheckpoint

   search.fit(X_train, y_train, callbacks=[ModelCheckpoint("checkpoint.pkl")])

Use ``save`` and ``load`` when you want to persist a fitted search object:

.. code-block:: python

   search.save("ga_search.pkl")
   restored = GASearchCV(estimator=estimator, param_grid=param_grid)
   restored.load("ga_search.pkl")

Benchmarks
##########

The repository includes benchmark scripts for optimizer mechanics, model
metrics, and comparisons against scikit-learn search methods:

.. code-block:: bash

   python benchmarks/benchmark_fit.py --quick
   python benchmarks/benchmark_fit.py --parallel-backends auto cv --runs 3
   python benchmarks/benchmark_search_methods.py --quick
   python benchmarks/benchmark_search_methods.py --methods gasearch randomized grid --runs 3

The reports include runtime, evaluated candidates, cross-validation effort,
cache/duplicate counts, optimizer telemetry, holdout metrics, and best
parameters.

Documentation and Examples
##########################

Useful links:

* Documentation: https://rodrigo-arenas.github.io/Sklearn-genetic-opt/
* Release notes: https://rodrigo-arenas.github.io/Sklearn-genetic-opt/versions/0.13/release-notes
* PyPI: https://pypi.org/project/sklearn-genetic-opt/
* Source code: https://github.com/rodrigo-arenas/Sklearn-genetic-opt/
* Issues: https://github.com/rodrigo-arenas/Sklearn-genetic-opt/issues

The documentation includes tutorials and executed notebooks for:

* comparing ``GASearchCV`` with sklearn search methods
* pipeline tuning and prediction
* feature selection
* multi-metric optimization
* MLflow 3 logging
* checkpointing and persistence
* advanced optimizer controls

Troubleshooting
###############

``TypeError: param_grid values must be instances of Integer, Continuous or Categorical``
    Use ``sklearn_genetic.space`` objects instead of plain lists.

    .. code-block:: python

       from sklearn_genetic.space import Integer

       param_grid = {"max_depth": Integer(2, 10)}

Missing optional dependencies
    Install the optional extras:

    .. code-block:: bash

       pip install sklearn-genetic-opt[all]

TensorFlow, TensorBoard, or MLflow dependency conflicts
    Install only the extras you need, or use a clean virtual environment.
    TensorFlow/TensorBoard support is only available on Python versions
    supported by those projects.

Contributing
############

Contributions are welcome. Please read the
`contribution guide <https://github.com/rodrigo-arenas/Sklearn-genetic-opt/blob/master/CONTRIBUTING.md>`_,
open issues for bugs or proposals, and include tests and documentation when
changing behavior.

For local development:

.. code-block:: bash

   git clone https://github.com/rodrigo-arenas/Sklearn-genetic-opt.git
   cd Sklearn-genetic-opt
   pip install -r dev-requirements.txt
   pytest sklearn_genetic

Big thanks to everyone helping improve the project.

|Contributors|_
