Release Notes
=============

Some notes on new features in various releases

What's new in 0.11.1
--------------------

^^^^^^^^^^
Bug Fixes:
^^^^^^^^^^

* Fixed a bug that would generate AttributeError: 'GASearchCV' object has no attribute 'creator'


What's new in 0.11.0
--------------------

^^^^^^^^^
Features:
^^^^^^^^^

* Added a parameter `use_cache`, which defaults to ``True``. When enabled, the algorithm will skip re-evaluating solutions that have already been evaluated, retrieving the performance metrics from the cache instead.
  If use_cache is set to ``False``, the algorithm will always re-evaluate solutions, even if they have been seen before, to obtain fresh performance metrics.
* Add a parameter in `GAFeatureSelectionCV` named warm_start_configs, defaults to ``None``, a list of predefined hyperparameter configurations to seed the initial population.
  Each element in the list is a dictionary where the keys are the names of the hyperparameters,
  and the values are the corresponding hyperparameter values to be used for the individual.

  Example:

    .. code-block:: python
       :linenos:

       warm_start_configs = [
              {"min_weight_fraction_leaf": 0.02, "bootstrap": True, "max_depth": None, "n_estimators": 100},
              {"min_weight_fraction_leaf": 0.4, "bootstrap": True, "max_depth": 5, "n_estimators": 200},
       ]

  The genetic algorithm will initialize part of the population with these configurations to
  warm-start the optimization process. The remaining individuals in the population will
  be initialized randomly according to the defined hyperparameter space.

  This parameter is useful when prior knowledge of good hyperparameter configurations exists,
  allowing the algorithm to focus on refining known good solutions while still exploring new
  areas of the hyperparameter space. If set to ``None``, the entire population will be initialized
  randomly.
* Introduced a **novelty search strategy** to the `GASearchCV` class. This strategy rewards solutions that are more distinct from others
  in the population by incorporating a **novelty score** into the fitness evaluation. The novelty score encourages exploration and promotes diversity,
  reducing the risk of premature convergence to local optima.

       - **Novelty Score**: Calculated based on the distance between an individual and its nearest neighbors in the population.
         Individuals with higher novelty scores are more distinct from the rest of the population.
       - **Fitness Evaluation**: The overall fitness is now a combination of the traditional performance score and the novelty score,
         allowing the algorithm to balance between exploiting known good solutions and exploring new, diverse ones.
       - **Improved Exploration**: This strategy helps explore new areas of the hyperparameter space, increasing the likelihood of discovering better solutions and avoiding local optima.

^^^^^^^^^^^^
API Changes:
^^^^^^^^^^^^

* Dropped support for python 3.8

What's new in 0.10.1
--------------------

^^^^^^^^^
Features:
^^^^^^^^^

* Install tensorflow when use ``pip install sklearn-genetic-opt[all]``

^^^^^^^^^^
Bug Fixes:
^^^^^^^^^^

* Fixed a bug that wouldn't allow to clone the GA classes when used inside a pipeline


What's new in 0.10.0
--------------------

^^^^^^^^^^^^
API Changes:
^^^^^^^^^^^^

* `GAFeatureSelectionCV` now mimics the scikit-learn FeatureSelection algorithms API instead of Grid Search, this enables
  easier implementation as a selection method that is closer to the scikit-learn API
* Improved `GAFeatureSelectionCV` candidate generation when `max_features` is set, it also ensures
  there is at least one feature selected
* `crossover_probability` and `mutation_probability` are now correctly passed to the mate and mutation
  functions inside GAFeatureSelectionCV
* Dropped support for python 3.7 and add support for python 3.10+
* Update most important packages from dev-requirements.txt to more recent versions
* Update deprecated functions in tests

^^^^^^^^^^
Bug Fixes:
^^^^^^^^^^

* Fixed the API docs of :class:`~sklearn_genetic.GAFeatureSelectionCV`, it was pointing to the wrong class

What's new in 0.9.0
-------------------

^^^^^^^^^
Features:
^^^^^^^^^

* Introducing Adaptive Schedulers to enable adaptive mutation and crossover probabilities;
  currently, supported schedulers are:

  - :class:`~sklearn_genetic.schedules.ConstantAdapter`
  - :class:`~sklearn_genetic.schedules.ExponentialAdapter`
  - :class:`~sklearn_genetic.schedules.InverseAdapter`
  - :class:`~sklearn_genetic.schedules.PotentialAdapter`


* Add `random_state` parameter (default= ``None``) in :class:`~sklearn_genetic.space.Continuous`,
  :class:`~sklearn_genetic.space.Categorical` and :class:`~sklearn_genetic.space.Integer` classes
  to leave fixed the random seed during hyperparameters sampling.
  Take into account that this only ensures that the space components are reproducible, not all the package.
  This is due to the DEAP dependency, which doesn't seem to have a native way to set the random seed.

^^^^^^^^^^^^
API Changes:
^^^^^^^^^^^^

* Changed the default values of `mutation_probability` and `crossover_probability`
  to 0.8 and 0.2, respectively.

* The `weighted_choice` function used in :class:`~sklearn_genetic.GAFeatureSelectionCV` was
  re-written to give more probability to a number of features closer to the `max_features` parameter

* Removed unused and wrong function :func:`~sklearn_genetic.plots.plot_parallel_coordinates`

^^^^^^^^^^
Bug Fixes:
^^^^^^^^^^

* Now when using the :func:`~sklearn_genetic.plots.plot_search_space` function, all the parameters get casted
  as np.float64 to avoid errors on seaborn package while plotting bool values.

What's new in 0.8.1
-------------------

^^^^^^^^^
Features:
^^^^^^^^^

* If the `max_features` parameter from :class:`~sklearn_genetic.GAFeatureSelectionCV` is set,
  the initial population is now sampled giving more probability to solutions with less than `max_features` features.


What's new in 0.8.0
-------------------

^^^^^^^^^
Features:
^^^^^^^^^

* :class:`~sklearn_genetic.GAFeatureSelectionCV` now has a parameter called `max_features`, int, default=None.
  If it's not None, it will penalize individuals with more features than max_features, putting a "soft" upper bound
  to the number of features to be selected.

* Classes :class:`~sklearn_genetic.GASearchCV` and :class:`~sklearn_genetic.GAFeatureSelectionCV`
  now support multi-metric evaluation the same way scikit-learn does,
  you will see this reflected on the `logbook` and `cv_results_` objects, where now you get results for each metric.
  As in scikit-learn, if multi-metric is used, the `refit` parameter must be a str specifying the metric to evaluate the cv-scores.
  See more in the :class:`~sklearn_genetic.GASearchCV` and :class:`~sklearn_genetic.GAFeatureSelectionCV` API documentation.

* Training gracefully stops if interrupted by some of these exceptions:
  ``KeyboardInterrupt``, ``SystemExit``, ``StopIteration``.
  When one of these exceptions is raised, the model finishes the current generation and saves the current
  best model. It only works if at least one generation has been completed.

^^^^^^^^^^^^
API Changes:
^^^^^^^^^^^^

* The following parameters changed their default values to create more extensive
  and different models with better results:

  - population_size from 10 to 50

  - generations from 40 to 80

  - mutation_probability from 0.1 to 0.2

^^^^^
Docs:
^^^^^

* A new notebook called Iris_multimetric was added to showcase the new multi-metric capabilities.

What's new in 0.7.0
-------------------

^^^^^^^^^
Features:
^^^^^^^^^

* :class:`~sklearn_genetic.GAFeatureSelectionCV` for feature selection along
  with any scikit-learn classifier or regressor. It optimizes the cv-score
  while minimizing the number of features to select.
  This class is compatible with the mlflow and tensorboard integration,
  the Callbacks and the ``plot_fitness_evolution`` function.

^^^^^^^^^^^^
API Changes:
^^^^^^^^^^^^

* The module :mod:`~sklearn_genetic.mlflow` was renamed to :class:`~sklearn_genetic.mlflow_log`
  to avoid unexpected errors on name resolutions

What's new in 0.6.1
-------------------

^^^^^^^^^
Features:
^^^^^^^^^

* Added the parameter `generations` to the :class:`~sklearn_genetic.callbacks.DeltaThreshold`.
  Now it compares the maximum and minimum values of a metric from the last generations, instead
  of just the current and previous ones. The default value is 2, so the behavior remains the same
  as in previous versions.

^^^^^^^^^^
Bug Fixes:
^^^^^^^^^^

* When a param_grid of length 1 is provided, a user warning is raised instead of an error.
  Internally it will swap the crossover operation to use the DEAP's :func:`~tools.cxSimulatedBinaryBounded`.
* When using :class:`~sklearn_genetic.space.Continuous` class with boundaries `lower` and `upper`,
  a uniform distribution  with limits `[lower, lower + upper]` was sampled, now, it's properly sampled
  using a `[lower, upper]` limits.


What's new in 0.6.0
-------------------

^^^^^^^^^
Features:
^^^^^^^^^

* Added the :class:`~sklearn_genetic.callbacks.ProgressBar` callback, it uses tqdm progress bar to shows
  how many generations are left in the training progress.
* Added the :class:`~sklearn_genetic.callbacks.TensorBoard` callback to log the
  generation metrics, watch in real time while the models are trained
  and compare different runs in your TensorBoard instance.
* Added the :class:`~sklearn_genetic.callbacks.TimerStopping` callback to stop
  the iterations after a total (threshold) fitting time has been elapsed.
* Added new parallel coordinates plot in  :func:`~sklearn_genetic.plots.plot_parallel_coordinates`.
* Now if one or more callbacks decides to stop the algorithm, it will print
  its class name to know which callbacks were responsible of the stopping.
* Added support for extra methods coming from scikit-learn's BaseSearchCV, like `cv_results_`,
  `best_index_` and `refit_time_` among others.
* Added methods `on_start` and `on_end` to :class:`~sklearn_genetic.callbacks.base.BaseCallback`.
  Now the algorithms check for the callbacks like this:

  - **on_start**: When the evolutionary algorithm is called from the GASearchCV.fit method.

  - **on_step:** When the evolutionary algorithm finishes a generation (no change here).

  - **on_end:** At the end of the last generation.

^^^^^^^^^^
Bug Fixes:
^^^^^^^^^^

* A missing statement was making that the callbacks start to get evaluated from generation 1, ignoring generation 0.
  Now this is properly handled and callbacks work from generation 0.

^^^^^^^^^^^^
API Changes:
^^^^^^^^^^^^

* The modules :mod:`~sklearn_genetic.plots` and :class:`~sklearn_genetic.mlflow.MLflowConfig`
  now requires an explicit installation of seaborn and mlflow, now those
  are optionally installed using ``pip install sklearn-genetic-opt[all].``
* The GASearchCV.logbook property now has extra information that comes from the
  scikit-learn cross_validate function.
* An optional extra parameter was added to GASearchCV, named `return_train_score`: bool, default= ``False``.
  As in scikit-learn, it controls if the `cv_results_` should have the training scores.

^^^^^
Docs:
^^^^^

* Edited all demos to be in the jupyter notebook format.
* Added embedded jupyter notebooks examples.
* The modules of the package now have a summary of their classes/functions in the docs.
* Updated the callbacks and custom callbacks tutorials to add new TensorBoard callback and
  the new methods on the base callback.


^^^^^^^^^
Internal:
^^^^^^^^^

* Now the hof uses the `self.best_params_` for the position 0, to be consistent with the
  scikit-learn API and parameters like `self.best_index_`


What's new in 0.5.0
-------------------

^^^^^^^^^
Features:
^^^^^^^^^


* Build-in integration with MLflow using the :class:`~sklearn_genetic.mlflow.MLflowConfig`
  and the new parameter `log_config` from :class:`~sklearn_genetic.GASearchCV`

* Implemented the callback :class:`~sklearn_genetic.callbacks.LogbookSaver`
  which saves the estimator.logbook object with all the fitted hyperparameters
  and their cross-validation score

* Added the parameter `estimator` to all the functions on
  the module :mod:`~sklearn_genetic.callbacks`

^^^^^
Docs:
^^^^^

* Added user guide "Integrating with MLflow"
* Update the tutorial "Custom Callbacks" for new API inheritance behavior

^^^^^^^^^
Internal:
^^^^^^^^^

* Added a base class :class:`~sklearn_genetic.callbacks.base.BaseCallback` from
  which all Callbacks must inherit from
* Now coverage report doesn't take into account the lines with # pragma: no cover
  and # noqa

What's new in 0.4.1
-------------------

^^^^^
Docs:
^^^^^

* Added user guide on "Understanding the evaluation process"
* Several guides on contributing, code of conduct
* Added important links
* Docs requirements are now independent of package requirements

^^^^^^^^^
Internal:
^^^^^^^^^

* Changed test ci from travis to Github actions

What's new in 0.4
-----------------

^^^^^^^^^
Features:
^^^^^^^^^

* Implemented callbacks module to stop the optimization process based in the
  current iteration metrics, currently implemented:
  :class:`~sklearn_genetic.callbacks.ThresholdStopping` ,
  :class:`~sklearn_genetic.callbacks.ConsecutiveStopping`
  and :class:`~sklearn_genetic.callbacks.DeltaThreshold`.
* The algorithms 'eaSimple', 'eaMuPlusLambda', 'eaMuCommaLambda'
  are now implemented in the module :mod:`~sklearn_genetic.algorithms`
  for more control over their options, rather that taking the deap.algorithms module
* Implemented the :mod:`~sklearn_genetic.plots` module and added the function
  :func:`~sklearn_genetic.plots.plot_search_space`,
  this function plots a mixed counter, scatter and histogram plots
  over all the fitted hyperparameters and their cross-validation score
* Documentation based in rst with Sphinx to host in read the docs.
  It includes public classes and functions documentation as well
  as several tutorials on how to use the package
* Added `best_params_` and `best_estimator_` properties
  after fitting GASearchCV
* Added optional parameters `refit`, `pre_dispatch` and `error_score`


^^^^^^^^^^^^
API Changes:
^^^^^^^^^^^^

* Removed support for python 3.6, changed the libraries supported
  versions to be the same as scikit-learn current version
* Several internal changes on the documentation and variables naming
  style to be compatible with Sphinx
* Removed the parameters `continuous_parameters`, `categorical_parameters` and `integer_parameters`
  replacing them with `param_grid`

What's new in 0.3
-----------------

^^^^^^^^^
Features:
^^^^^^^^^

* Added the space module to control better the data
  types and ranges of each hyperparameter, their distribution to sample random values from,
  and merge all data types in one Space class that can work with the new param_grid parameter
* Changed the `continuous_parameters`, `categorical_parameters` and `integer_parameters`
  for the `param_grid`, the first ones still work but will be removed in a next version
* Added the option to use the eaMuCommaLambda algorithm from deap
* The `mu` and `lambda_` parameters of the internal eaMuPlusLambda and eaMuCommaLambda
  now are in terms of the initial population size and not the number of generations

What's new in 0.2
-----------------

^^^^^^^^^
Features:
^^^^^^^^^

* Enabled deap's eaMuPlusLambda algorithm for the optimization process, now is the default routine
* Added a logbook and history properties to the fitted GASearchCV  to make post-fit analysis
* ``Elitism=False`` now implements a roulette selection instead of ignoring the parameter
* Added the parameter keep_top_k to control the number of solutions if the hall of fame (hof)

^^^^^^^^^^^^
API Changes:
^^^^^^^^^^^^

* Refactored the optimization algorithm to use DEAP package instead
  of a custom implementation, this causes the removal of several methods, properties and variables inside the GASearchCV class
* The parameter encoding_length has been removed, it's no longer required to the GASearchCV class
* Renamed the property of the fitted estimator from `best_params_` to `best_params`
* The verbosity now prints the deap log of the fitness function,
  it's standard deviation, max and min values from each generation
* The variable `GASearchCV._best_solutions` was removed and it's meant to be
  replaced with `GASearchCV.logbook` and `GASearchCV.history`
* Changed default parameters crossover_probability from 1 to 0.8 and generations from 50 to 40

What's new in 0.1
-----------------

^^^^^^^^^
Features:
^^^^^^^^^

* :class:`~sklearn_genetic.GASearchCV` for hyperparameters tuning
  using custom genetic algorithm for scikit-learn
  classification and regression models
* :func:`~sklearn_genetic.plots.plot_fitness_evolution` function to see the average
  fitness values over generations
