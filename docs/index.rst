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

This example tunes a ``RandomForestClassifier`` across six hyperparameters on
the breast cancer dataset. With six mixed parameters — integers, floats, and
a categorical — this is exactly the kind of search where GA's ability to
recombine good partial solutions gives it an edge over independent random
sampling.

.. code-block:: python

   from sklearn.datasets import load_breast_cancer
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import StratifiedKFold, train_test_split
   from sklearn.metrics import roc_auc_score

   from sklearn_genetic import EvolutionConfig, GASearchCV, PopulationConfig, RuntimeConfig
   from sklearn_genetic.space import Categorical, Continuous, Integer

   X, y = load_breast_cancer(return_X_y=True)
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.25, stratify=y, random_state=42
   )

   param_grid = {
       "n_estimators": Integer(50, 250),
       "max_depth": Integer(2, 14),
       "min_samples_split": Integer(2, 12),
       "min_samples_leaf": Integer(1, 8),
       "max_features": Categorical(["sqrt", "log2", None]),
       "ccp_alpha": Continuous(0.0, 0.03),
   }

   search = GASearchCV(
       estimator=RandomForestClassifier(random_state=42),
       param_grid=param_grid,
       cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
       scoring="roc_auc",
       evolution_config=EvolutionConfig(population_size=20, generations=12),
       population_config=PopulationConfig(initializer="smart"),
       runtime_config=RuntimeConfig(n_jobs=-1, parallel_backend="auto", use_cache=True),
   )

   search.fit(X_train, y_train)

   print(search.best_params_)
   print("CV score:", round(search.best_score_, 4))

   y_prob = search.predict_proba(X_test)[:, 1]
   print("Holdout ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))

   # Evaluation cost breakdown
   print(search.fit_stats_)

Recommended Next Steps
######################

* Not sure if GA search is the right tool? Start with
  :doc:`tutorials/when_to_use` for a comparison guide and decision table.
* New to the library? :doc:`tutorials/basic_usage` walks through the full
  workflow from data loading to prediction.
* Tuning a scikit-learn ``Pipeline``? See :doc:`tutorials/pipeline_tuning` for
  the ``step__param`` naming convention and a worked regression example.
* Read :doc:`tutorials/advanced_optimizer_control` for local search, diversity
  control, fitness sharing, and optimizer telemetry when the default settings
  are not enough.

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: User Guide / Tutorials:

   tutorials/when_to_use
   tutorials/basic_usage
   tutorials/understand_cv
   tutorials/pipeline_tuning
   tutorials/callbacks
   tutorials/custom_callback
   tutorials/adapters
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
   notebooks/Plotting_gallery.ipynb

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

