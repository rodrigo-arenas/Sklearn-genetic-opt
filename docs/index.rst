.. sklearn-genetic-opt documentation master file, created by
   sphinx-quickstart on Sat May 29 19:27:12 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

sklearn-genetic-opt
===================

``sklearn-genetic-opt`` adds evolutionary optimization tools to the
scikit-learn workflow. It can tune hyperparameters with
:class:`~sklearn_genetic.GASearchCV` and select feature subsets with
:class:`~sklearn_genetic.GAFeatureSelectionCV` using algorithms powered by
DEAP.

The project is useful when a search space is mixed, irregular, expensive, or
not well served by an exhaustive grid. It follows familiar scikit-learn
patterns: define an estimator, define a search space, call ``fit``, inspect
``best_params_`` or ``support_``, and use the fitted object for prediction.

Highlights
##########

* :class:`~sklearn_genetic.GASearchCV` for hyperparameter search across
  classification, regression, and supported outlier-detection estimators.
* :class:`~sklearn_genetic.GAFeatureSelectionCV` for wrapper-based feature
  selection with cross-validation.
* Search spaces for integer, continuous, and categorical parameters.
* Grouped configuration objects for readable advanced setups:
  :class:`~sklearn_genetic.EvolutionConfig`,
  :class:`~sklearn_genetic.PopulationConfig`,
  :class:`~sklearn_genetic.RuntimeConfig`, and
  :class:`~sklearn_genetic.OptimizationConfig`.
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

Install the core package:

.. code-block:: bash

   pip install sklearn-genetic-opt

Install optional plotting, MLflow, and TensorBoard integrations:

.. code-block:: bash

   pip install sklearn-genetic-opt[all]

.. |PythonMinVersion| replace:: 3.12
.. |ScikitLearnMinVersion| replace:: 1.9.0
.. |NumPyMinVersion| replace:: 2.4.6
.. |SeabornMinVersion| replace:: 0.13.2
.. |DEAPMinVersion| replace:: 1.4.4
.. |MLflowMinVersion| replace:: 3.14.0
.. |TensorflowMinVersion| replace:: 2.21.0
.. |TensorBoardMinVersion| replace:: 2.20.0
.. |tqdmMinVersion| replace:: 4.68.3

Requirements
############

Core requirements:

- Python (>= |PythonMinVersion|)
- scikit-learn (>= |ScikitLearnMinVersion|)
- NumPy (>= |NumPyMinVersion|)
- DEAP (>= |DEAPMinVersion|)
- tqdm (>= |tqdmMinVersion|)

Optional extras:

- Seaborn (>= |SeabornMinVersion|) for plots
- MLflow (>= |MLflowMinVersion|) for experiment logging
- TensorFlow (>= |TensorflowMinVersion|) and TensorBoard
  (>= |TensorBoardMinVersion|, < 2.21.0) for TensorBoard logging on Python <
  3.14

Quick Start
###########

.. code-block:: python

   from sklearn.datasets import load_iris
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import StratifiedKFold

   from sklearn_genetic import EvolutionConfig, GASearchCV, PopulationConfig, RuntimeConfig
   from sklearn_genetic.space import Categorical, Continuous, Integer

   X, y = load_iris(return_X_y=True)

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
       runtime_config=RuntimeConfig(n_jobs=-1, parallel_backend="auto", use_cache=True),
   )

   search.fit(X, y)

   print(search.best_params_)
   print(search.best_score_)
   print(search.fit_stats_)

Recommended Next Steps
######################

* Start with :doc:`tutorials/basic_usage` for the core workflow.
* Read :doc:`tutorials/advanced_optimizer_control` for local search, diversity
  control, fitness sharing, and optimizer telemetry.
* Use the notebook examples for richer, executed workflows covering pipelines,
  feature selection, MLflow 3 logging, persistence, and comparisons against
  scikit-learn search methods.

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: User Guide / Tutorials:

   tutorials/basic_usage
   tutorials/callbacks
   tutorials/custom_callback
   tutorials/adapters
   tutorials/understand_cv
   tutorials/advanced_optimizer_control
   tutorials/mlflow
   tutorials/outliers
   tutorials/reproducibility

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: Jupyter notebooks examples:

   notebooks/sklearn_comparison.ipynb
   notebooks/Pipeline_prediction.ipynb
   notebooks/Iris_feature_selection.ipynb
   notebooks/Advanced_breast_cancer_random_forest.ipynb
   notebooks/MLflow_logger.ipynb
   notebooks/Checkpointing_and_persistence.ipynb
   notebooks/Iris_multimetric.ipynb

.. toctree::
   :maxdepth: 2
   :caption: Release Notes

   release_notes

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/gasearchcv
   api/gafeatureselectioncv
   api/config
   api/callbacks
   api/schedules
   api/plots
   api/mlflow
   api/space
   api/algorithms


.. toctree::
   :maxdepth: 1
   :caption: External References:

   external_references

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

