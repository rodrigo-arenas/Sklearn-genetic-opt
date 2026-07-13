.. -*- mode: rst -*-

.. image:: https://github.com/rodrigo-arenas/Sklearn-genetic-opt/blob/master/docs-vitepress/public/brand/readme-banner-1280x320.png?raw=true
   :alt: sklearn-genetic-opt — hyperparameter tuning and feature selection with evolutionary algorithms
   :target: https://sklearngeneticopt.rodrigo-arenas.com/

sklearn-genetic-opt
###################

Hyperparameter tuning and feature selection for scikit-learn models using genetic algorithms.

|Tests|_ |Codecov|_ |PythonVersion|_ |PyPi| |Conda|_ |Docs|_ |Stars|_ |GoodFirstIssues|_

.. |Tests| image:: https://github.com/rodrigo-arenas/Sklearn-genetic-opt/actions/workflows/ci-tests.yml/badge.svg?branch=master
.. _Tests: https://github.com/rodrigo-arenas/Sklearn-genetic-opt/actions/workflows/ci-tests.yml

.. |Codecov| image:: https://codecov.io/gh/rodrigo-arenas/Sklearn-genetic-opt/branch/master/graphs/badge.svg?branch=master&service=github
.. _Codecov: https://codecov.io/github/rodrigo-arenas/Sklearn-genetic-opt?branch=master

.. |PythonVersion| image:: https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue
.. _PythonVersion: https://www.python.org/downloads/

.. |PyPi| image:: https://img.shields.io/pypi/v/sklearn-genetic-opt.svg
   :target: https://pypi.org/project/sklearn-genetic-opt/
   :alt: PyPI package version

.. |Conda| image:: https://img.shields.io/conda/vn/conda-forge/sklearn-genetic-opt.svg
.. _Conda: https://anaconda.org/conda-forge/sklearn-genetic-opt

.. |Docs| image:: https://img.shields.io/badge/docs-GitHub%20Pages-blue
.. _Docs: https://sklearngeneticopt.rodrigo-arenas.com/

.. |Stars| image:: https://img.shields.io/github/stars/rodrigo-arenas/Sklearn-genetic-opt?style=social
.. _Stars: https://github.com/rodrigo-arenas/Sklearn-genetic-opt

.. |GoodFirstIssues| image:: https://img.shields.io/github/issues/rodrigo-arenas/Sklearn-genetic-opt/good%20first%20issue?label=good%20first%20issues&color=7057ff
.. _GoodFirstIssues: https://github.com/rodrigo-arenas/Sklearn-genetic-opt/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22

.. |Contributors| image:: https://contributors-img.web.app/image?repo=rodrigo-arenas/sklearn-genetic-opt
.. _Contributors: https://github.com/rodrigo-arenas/Sklearn-genetic-opt/graphs/contributors


``sklearn-genetic-opt`` is a scikit-learn-compatible optimization toolkit for users
who want a smarter alternative to ``GridSearchCV`` and ``RandomizedSearchCV``.
Its genetic algorithm evaluates complete parameter configurations — finding
regions where ``learning_rate × n_estimators`` or ``C × gamma`` are jointly optimal,
something one-parameter-at-a-time approaches miss. It also provides
``GAFeatureSelectionCV``, a wrapper-based selector that searches the full space
of feature subsets simultaneously instead of eliminating features one at a time.

If ``sklearn-genetic-opt`` saves you time, consider starring the
`GitHub repository <https://github.com/rodrigo-arenas/Sklearn-genetic-opt>`_.
It helps more practitioners discover the project.


Why use sklearn-genetic-opt?
#############################

* **Drop-in scikit-learn API** — ``GASearchCV`` has the same ``fit`` / ``predict`` / ``best_params_`` interface as ``GridSearchCV``; replace it in one line.
* **Handles interacting parameters** — genetic algorithms evaluate complete configurations, naturally finding cross-parameter sweet spots that random or grid search miss.
* **Joint feature selection and tuning** — run ``GAFeatureSelectionCV`` and ``GASearchCV`` in a two-stage workflow; no separate feature-selection library needed.
* **Mixed search spaces** — ``Integer``, ``Continuous`` (uniform or log-uniform), and ``Categorical`` types in the same search.
* **Smart initialization** — Latin hypercube seeding, estimator defaults, warm-start configs, and duplicate avoidance give the first generation a head start over random initialization.
* **Early stopping callbacks** — ``ConsecutiveStopping``, ``DeltaThreshold``, and ``TimerStopping`` end the search automatically when it converges or runs out of time.
* **Adaptive schedules** — crossover and mutation rates anneal over generations, shifting from exploration to exploitation.
* **Optimization history and plots** — per-generation fitness, diversity, and telemetry stored in ``history``; built-in plots visualize the full search.
* **MLflow integration** — every evaluated candidate is automatically logged as a child run for experiment comparison.
* **Parallel execution** — ``n_jobs=-1`` parallelizes candidate or fold evaluation.


When should you use it?
########################

* Your model is expensive to train and you can only afford 50–200 total evaluations.
* Your search space has 5+ hyperparameters that interact (gradient boosting, SVM, regularized regression).
* You want feature selection and hyperparameter tuning in a single reproducible workflow.
* You want optimization history, convergence plots, callbacks, or MLflow tracking built in.
* You have known-good configurations to warm-start from (prior runs, published defaults).
* ``GridSearchCV`` is too slow and ``RandomizedSearchCV`` keeps returning similar bad results.


When should you NOT use it?
############################

* **You need a fast baseline** — start with a fixed configuration or ``RandomizedSearchCV(n_iter=20)``; it's faster and good enough to validate your pipeline.
* **Your grid is tiny** (fewer than 50 combinations) — ``GridSearchCV`` covers it exhaustively and is simpler to reason about.
* **Your model and dataset are fast** (< 1 s per fit) — the overhead of managing a population adds up relative to just running all combinations.
* **You need distributed optimization** across a cluster — use Optuna with its distributed backends.
* **You need strict Bayesian guarantees** on the exploration-exploitation trade-off — use Optuna (TPE) or scikit-optimize.


How it compares
################

.. list-table::
   :header-rows: 1
   :widths: 20 28 28 24

   * - Tool
     - Best for
     - Key limitation
     - Where sklearn-genetic-opt helps
   * - ``GridSearchCV``
     - Small, fully discrete grids; guaranteed complete coverage
     - Combinatorial explosion on 4+ params; no continuous params natively
     - Large, mixed, or continuous spaces with interacting parameters
   * - ``RandomizedSearchCV``
     - Larger budgets; simple independent parameter spaces
     - No learning from past evaluations; treats each parameter independently
     - Exploits cross-parameter interactions; adaptive schedules; early stopping
   * - Optuna
     - Sequential Bayesian (TPE) search; distributed optimization; neural architecture search
     - No native sklearn cross-validation; no built-in wrapper feature selection
     - Drop-in sklearn API; built-in ``GAFeatureSelectionCV``
   * - RFE
     - Greedy feature elimination for models with ``coef_`` or ``feature_importances_``
     - Greedy and sequential; can miss non-greedy optimal subsets
     - Evaluates all subsets simultaneously; works with any estimator
   * - ``SelectFromModel``
     - Fast embedded selection via a threshold on feature importance
     - Tied to model-specific importances; no cross-estimator comparison
     - Estimator-agnostic wrapper; combinable with hyperparameter tuning in one workflow
   * - **sklearn-genetic-opt**
     - Large or mixed spaces; joint feature + parameter search; history, plots, callbacks
     - Slower than Bayesian methods on small smooth spaces; population size needs tuning
     - —


Quick Start
###########

**Install**

.. code-block:: bash

   pip install sklearn-genetic-opt
   # With optional plotting, MLflow, and TensorBoard extras:
   # pip install sklearn-genetic-opt[all]

**Hyperparameter search**

.. code-block:: python

   from sklearn.datasets import load_breast_cancer
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import StratifiedKFold, train_test_split

   from sklearn_genetic import GASearchCV, EvolutionConfig, RuntimeConfig
   from sklearn_genetic.space import Categorical, Continuous, Integer

   X, y = load_breast_cancer(return_X_y=True)
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, stratify=y, random_state=42
   )

   search = GASearchCV(
       estimator=RandomForestClassifier(random_state=42),
       param_grid={
           "n_estimators": Integer(50, 300),
           "max_depth":    Integer(3, 20),
           "max_features": Continuous(0.1, 1.0),
           "class_weight": Categorical([None, "balanced"]),
       },
       cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
       scoring="roc_auc",
       evolution_config=EvolutionConfig(population_size=15, generations=12),
       runtime_config=RuntimeConfig(n_jobs=-1, verbose=True),
       random_state=42,
   )
   search.fit(X_train, y_train)

   print(search.best_params_)          # best hyperparameter configuration
   print(search.best_score_)           # best cross-validated ROC-AUC
   print(search.score(X_test, y_test)) # test-set score

**Use a starter preset**

.. code-block:: python

   from sklearn_genetic import random_forest_classifier_space, xgboost_classifier_space

   rf_param_grid = random_forest_classifier_space(profile="balanced")
   xgb_param_grid = xgboost_classifier_space(profile="balanced")

**Convert a RandomizedSearchCV-style space**

.. code-block:: python

   from scipy import stats

   from sklearn_genetic.space import from_sklearn_space

   param_grid = from_sklearn_space({
       "n_estimators": stats.randint(50, 300),
       "max_depth": [3, 5, 10, None],
       "max_features": stats.uniform(0.1, 0.9),
   })

**Feature selection**

.. code-block:: python

   import numpy as np
   from sklearn.datasets import load_iris
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import StratifiedKFold, train_test_split

   from sklearn_genetic import GAFeatureSelectionCV, EvolutionConfig, RuntimeConfig

   X, y = load_iris(return_X_y=True)
   # Add 8 noise features — the selector should drop them
   X = np.hstack([X, np.random.default_rng(42).uniform(0, 1, (X.shape[0], 8))])
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, stratify=y, random_state=42
   )

   selector = GAFeatureSelectionCV(
       estimator=RandomForestClassifier(n_estimators=100, random_state=42),
       cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
       scoring="accuracy",
       evolution_config=EvolutionConfig(population_size=20, generations=15),
       runtime_config=RuntimeConfig(n_jobs=-1, verbose=True),
       random_state=42,
   )
   selector.fit(X_train, y_train)

   print(selector.support_)                    # boolean mask of selected features
   print(selector.score(X_test, y_test))       # accuracy on selected features

`Read the full Getting Started guide →
<https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/guide/basic-usage>`_


What you can do
################

**Track optimization progress generation by generation.** The fitness curve
shows when the search converges so you know whether to add more generations
or stop early.

.. image:: https://github.com/rodrigo-arenas/Sklearn-genetic-opt/blob/master/docs-vitepress/public/images/plotting_gallery_fitness_evolution.png?raw=true
   :alt: Fitness evolution — best and mean CV score plotted over generations
   :width: 680

**See where the search explored and which parameter combinations scored
highest.** The scatter plot reveals the productive region of the
``learning_rate × n_estimators`` interaction — a band a one-at-a-time
sweep cannot find.

.. image:: https://github.com/rodrigo-arenas/Sklearn-genetic-opt/blob/master/docs-vitepress/public/images/tune_xgboost_interaction.png?raw=true
   :alt: Every evaluated candidate colored by CV score, learning_rate vs n_estimators
   :width: 680

**Inspect the full search in one view.** The search overview panel combines
scores, parameter distributions, diversity, and candidate decisions.

.. image:: https://github.com/rodrigo-arenas/Sklearn-genetic-opt/blob/master/docs-vitepress/public/images/basic_usage_search_overview.png?raw=true
   :alt: Search overview dashboard showing all evaluated candidates
   :width: 680

`See the full Plotting Gallery →
<https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/examples/plotting-gallery>`_


Common use cases
#################

**Hyperparameter tuning**

* `Tune RandomForestClassifier <https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/tutorials/tune-random-forest>`_ — 7-parameter joint search, classification and regression
* `Tune XGBoost <https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/tutorials/tune-xgboost>`_ — 9 interacting params, ``n_jobs=1`` oversubscription fix, interaction visualization
* `Tune LightGBM <https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/tutorials/tune-lightgbm>`_ — ``num_leaves`` / ``max_depth`` interaction, parameter scatter plots
* `Tune CatBoost <https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/tutorials/tune-catboost>`_ — ``bagging_temperature``, ``border_count``, GPU tip
* `Tune Gradient Boosting (sklearn) <https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/tutorials/tune-gradient-boosting>`_ — HistGBM vs classic GBM, speed comparison
* `Tune Logistic Regression <https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/tutorials/tune-logistic-regression>`_ — solver / penalty compatibility, multi-penalty with SAGA
* `Tune SVM (C, kernel, gamma) <https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/tutorials/tune-svm>`_ — C–gamma interaction, mandatory Pipeline + StandardScaler
* `Tune a scikit-learn Pipeline <https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/guide/pipeline-tuning>`_ — step prefix patterns, ColumnTransformer
* `Tune for imbalanced datasets <https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/tutorials/imbalanced-classification>`_ — ``class_weight`` as a search param, balanced accuracy

**Feature selection**

* `Feature selection with genetic algorithms <https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/tutorials/feature-selection>`_ — 3-stage: select, retune, validate
* `Combine feature selection + hyperparameter tuning <https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/recipes/feature-selection/select-then-tune>`_ — two-stage pipeline recipe

**Experiment tracking and tooling**

* `MLflow experiment tracking <https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/guide/mlflow>`_ — log every candidate as a child run
* `Callbacks and early stopping <https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/guide/callbacks>`_ — ConsecutiveStopping, TimerStopping, DeltaThreshold
* `Checkpointing and resume <https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/guide/reproducibility>`_ — save and continue long searches
* `Visualize optimization progress <https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/examples/plotting-gallery>`_ — full gallery of available plots

`Browse all Recipes →
<https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/recipes/>`_
`Browse all Tutorials →
<https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/tutorials/>`_

Citation
########

If ``sklearn-genetic-opt`` supports your research or a published project, please
cite the software version you used. GitHub can generate a citation from
``CITATION.cff`` through the "Cite this repository" button, and citation
managers can read the same metadata directly from the repository.

Recommended citation:

.. code-block:: text

   Arenas, R. (2026). sklearn-genetic-opt: Hyperparameter tuning and feature
   selection using genetic algorithms, built on top of scikit-learn (Version
   0.13.3) [Computer software].
   https://github.com/rodrigo-arenas/Sklearn-genetic-opt

BibTeX:

.. code-block:: bibtex

   @software{arenas_2026_sklearn_genetic_opt,
     author = {Arenas, Rodrigo},
     title = {{sklearn-genetic-opt}: Hyperparameter tuning and feature selection
       using genetic algorithms, built on top of scikit-learn},
     year = {2026},
     version = {0.13.3},
     url = {https://github.com/rodrigo-arenas/Sklearn-genetic-opt}
   }

Learning paths
###############

**New user**
  `Install <https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/guide/installation>`_
  → `Getting Started with GASearchCV <https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/guide/basic-usage>`_
  → `How Hyperparameter Optimization Works <https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/guide/how-hyperparameter-optimization-works>`_
  → `Pick a Recipe <https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/recipes/>`_

**ML practitioner**
  `When to Use Genetic Algorithm Search <https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/guide/when-to-use>`_
  → `Choosing the Right Search Space <https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/guide/choosing-search-spaces>`_
  → `Model-specific Tutorials <https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/tutorials/>`_
  → `Optuna vs sklearn-genetic-opt <https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/comparisons/optuna-vs-sklearn-genetic-opt>`_

**Contributor**
  `Contributing guide <https://github.com/rodrigo-arenas/Sklearn-genetic-opt/blob/master/CONTRIBUTING.md>`_
  → `Open issues <https://github.com/rodrigo-arenas/Sklearn-genetic-opt/issues>`_
  → `Benchmarks documentation <https://rodrigo-arenas.github.io/Sklearn-genetic-opt/versions/latest/benchmarks/>`_


Benchmarks
##########

The repository includes benchmark scripts that compare ``GASearchCV`` against
``RandomizedSearchCV``, ``GridSearchCV``, and Optuna (TPE) using the
`Bayesmark <https://github.com/uber/bayesmark>`_ experimental design — same
datasets, same search spaces, equal evaluation budget:

.. code-block:: bash

   pip install sklearn-genetic-opt[benchmark]
   python benchmarks/benchmark_bayesmark.py --quick

See the `Benchmarks documentation
<https://rodrigo-arenas.github.io/Sklearn-genetic-opt/versions/latest/benchmarks/>`_
for methodology and full results.


Installation
############

.. code-block:: bash

   # Core package
   pip install sklearn-genetic-opt

   # With plotting, MLflow, and TensorBoard:
   pip install sklearn-genetic-opt[all]

   # conda
   conda install -c conda-forge sklearn-genetic-opt

Requires Python ≥ 3.12 and scikit-learn ≥ 1.5.0.
See `Installation <https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/guide/installation>`_
for the full requirements table and optional extras.


Contributing
############

Contributions of all sizes are welcome — from fixing a typo to adding a new
tutorial or benchmark.

Good ways to start:

* **Add a Recipe** for an estimator or workflow not yet covered — `see existing Recipes <https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/recipes/>`_.
* **Write or improve a tutorial** (new models, edge cases, regression examples).
* **Test with a new estimator** from another framework and report the results.
* **Add a benchmark** comparing search methods on a real dataset.
* **Fix typing, CI, or formatting** — ``black .`` keeps the style consistent.
* **Answer questions** in open issues.
* **Share your work** — add a blog post, article, or video to the
  `Community Articles page <https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/community/articles>`_.

.. code-block:: bash

   git clone https://github.com/rodrigo-arenas/Sklearn-genetic-opt.git
   cd Sklearn-genetic-opt
   pip install -r dev-requirements.txt
   pytest sklearn_genetic

Read the `contribution guide <https://github.com/rodrigo-arenas/Sklearn-genetic-opt/blob/master/CONTRIBUTING.md>`_
before opening a pull request. If you are not sure where to start,
`open an issue <https://github.com/rodrigo-arenas/Sklearn-genetic-opt/issues>`_
and ask — small contributions are very welcome.

|Contributors|_


Links
#####

* `Documentation <https://sklearngeneticopt.rodrigo-arenas.com/>`_
* `API reference <https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/api/gasearchcv>`_
* `Tutorials <https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/tutorials/>`_
* `Recipes <https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/recipes/>`_
* `Grid vs Random vs Bayesian vs Genetic comparison <https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/comparisons/grid-search-vs-genetic-algorithms>`_
* `Common hyperparameter tuning mistakes <https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/guide/common-mistakes>`_
* `Troubleshooting <https://sklearngeneticopt.rodrigo-arenas.com/versions/latest/guide/troubleshooting>`_
* `Release notes <https://sklearngeneticopt.rodrigo-arenas.com/stable/>`_
* `PyPI <https://pypi.org/project/sklearn-genetic-opt/>`_
* `Issues <https://github.com/rodrigo-arenas/Sklearn-genetic-opt/issues>`_
