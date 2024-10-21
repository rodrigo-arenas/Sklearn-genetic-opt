import random
import time
import warnings

import numpy as np
from deap import base, creator, tools
from sklearn.base import clone
from sklearn.model_selection import cross_validate
from sklearn.base import is_classifier, is_regressor, BaseEstimator, MetaEstimatorMixin
from sklearn.feature_selection import SelectorMixin
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.metaestimators import available_if
from sklearn.feature_selection._from_model import _estimator_has
from sklearn.metrics import check_scoring
from sklearn.exceptions import NotFittedError
from sklearn.model_selection._search import BaseSearchCV
from sklearn.model_selection._split import check_cv
from sklearn.metrics._scorer import _check_multimetric_scoring

from .parameters import Algorithms, Criteria
from .space import Space
from .algorithms import algorithms_factory
from .callbacks.validations import check_callback
from .schedules.validations import check_adapter
from .utils.cv_scores import (
    create_gasearch_cv_results_,
    create_feature_selection_cv_results_,
)
from .utils.random import weighted_bool_individual
from .utils.tools import cxUniform, mutFlipBit, novelty_scorer

import pickle
import os
from .callbacks.model_checkpoint import ModelCheckpoint


class GASearchCV(BaseSearchCV):
    """
    Evolutionary optimization over hyperparameters.

    GASearchCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba", "decision_function",
    "predict_log_proba" if they are implemented in the
    estimator used.
    The parameters of the estimator used to apply these methods are optimized
    by cross-validated search over parameter settings.

    Parameters
    ----------
    estimator : estimator object, default=None
        estimator object implementing 'fit'
        The object to use to fit the data.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - CV splitter,
        - An iterable yielding (train, test) splits as arrays of indices.
        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

    param_grid : dict, default=None
        Grid with the parameters to tune, expects keys a valid name
        of hyperparameter based on the estimator selected and as values
        one of :class:`~sklearn_genetic.space.Integer` ,
        :class:`~sklearn_genetic.space.Categorical`
        :class:`~sklearn_genetic.space.Continuous` classes.
        At least two parameters are advised to be provided in order to successfully make
        an optimization routine.

    population_size : int, default=10
        Size of the initial population to sample randomly generated individuals.

    generations : int, default=40
        Number of generations or iterations to run the evolutionary algorithm.

    crossover_probability : float or a Scheduler, default=0.8
        Probability of crossover operation between two individuals.

    mutation_probability : float or a Scheduler, default=0.1
        Probability of child mutation.

    tournament_size : int, default=3
        Number of individuals to perform tournament selection.

    elitism : bool, default=True
        If True takes the *tournament_size* best solution to the next generation.

    scoring : str, callable, list, tuple or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set.
        If `scoring` represents a single score, one can use:

        - a single string;
        - a callable that returns a single value.
        If `scoring` represents multiple scores, one can use:

        - a list or tuple of unique strings;
        - a callable returning a dictionary where the keys are the metric
          names and the values are the metric scores;
        - a dictionary with metric names as keys and callables a values.

    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the cross-validation splits.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    verbose : bool, default=True
        If ``True``, shows the metrics on the optimization routine.

    keep_top_k : int, default=1
        Number of best solutions to keep in the hof object. If a callback stops the algorithm before k iterations,
        it will return only one set of parameters per iteration.

    criteria : {'max', 'min'} , default='max'
        ``max`` if a higher scoring metric is better, ``min`` otherwise.

    algorithm : {'eaMuPlusLambda', 'eaMuCommaLambda', 'eaSimple'}, default='eaMuPlusLambda'
        Evolutionary algorithm to use.
        See more details in the deap algorithms documentation.

    refit : bool, str, or callable, default=True
        Refit an estimator using the best found parameters on the whole
        dataset.
        For multiple metric evaluation, this needs to be a `str` denoting the
        scorer that would be used to find the best parameters for refitting
        the estimator at the end.
        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``GASearchCV`` instance.
        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_params_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer.
        See ``scoring`` parameter to know more about multiple metric
        evaluation.

        If ``False``, it is not possible to make predictions
        using this GASearchCV instance after fitting.

    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:
            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
            - An int, giving the exact number of total jobs that are
              spawned
            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to ``'raise'``, the error is raised.
        If a numeric value is given, FitFailedWarning is raised.

    return_train_score: bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

    log_config : :class:`~sklearn_genetic.mlflow.MLflowConfig`, default = None
        Configuration to log metrics and models to mlflow, of None,
        no mlflow logging will be performed

    use_cache: bool, default=True
        If set to true it will avoid to re-evaluating solutions that have already seen,
        otherwise it will always evaluate the solutions to get the performance metrics

    Attributes
    ----------

    logbook : :class:`DEAP.tools.Logbook`
        Contains the logs of every set of hyperparameters fitted with its average scoring metric.
    history : dict
        Dictionary of the form:
        {"gen": [],
        "fitness": [],
        "fitness_std": [],
        "fitness_max": [],
        "fitness_min": []}

         *gen* returns the index of the evaluated generations.
         Each entry on the others lists, represent the average metric in each generation.

    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.
    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score
        on the left out data. Not available if ``refit=False``.
    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.
    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.
        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).
    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.
    n_splits_ : int
        The number of cross-validation splits (folds/iterations).
    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.
        This is present only if ``refit`` is not False.
    """

    def __init__(
        self,
        estimator,
        cv=3,
        param_grid=None,
        scoring=None,
        population_size=50,
        generations=80,
        crossover_probability=0.2,
        mutation_probability=0.8,
        tournament_size=3,
        elitism=True,
        verbose=True,
        keep_top_k=1,
        criteria="max",
        algorithm="eaMuPlusLambda",
        refit=True,
        n_jobs=1,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
        log_config=None,
        use_cache=True,
        warm_start_configs=None,
    ):
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.population_size = population_size
        self.generations = generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.crossover_adapter = check_adapter(self.crossover_probability)
        self.mutation_adapter = check_adapter(self.mutation_probability)
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.verbose = verbose
        self.keep_top_k = keep_top_k
        self.criteria = criteria
        self.param_grid = param_grid
        self.algorithm = algorithm
        self.refit = refit
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score
        # self.creator = creator
        self.log_config = log_config
        self.use_cache = use_cache
        self.fitness_cache = {}
        self.warm_start_configs = warm_start_configs or []

        # Check that the estimator is compatible with scikit-learn
        if not is_classifier(self.estimator) and not is_regressor(self.estimator):
            raise ValueError(f"{self.estimator} is not a valid Sklearn classifier or regressor")

        if criteria not in Criteria.list():
            raise ValueError(f"Criteria must be one of {Criteria.list()}, got {criteria} instead")
        # Minimization is handle like an optimization problem with a change in the score sign
        elif criteria == Criteria.max.value:
            self.criteria_sign = 1.0
        elif criteria == Criteria.min.value:
            self.criteria_sign = -1.0

        # Saves the param_grid and computes some extra properties in the same object
        self.space = Space(param_grid)

        if len(self.space) == 1:  # pragma: no cover
            warnings.warn(
                "Warning, only one parameter was provided to the param_grid, the optimization routine "
                "might not have effect or it could lead to errors, it's advised to use at least 2 parameters"
            )

        super(GASearchCV, self).__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
        )

    def _register(self):
        """
        This function is the responsible for registering the DEAPs necessary methods
        and create other objects to hold the hof, logbook and stats.
        """
        self.toolbox = base.Toolbox()

        creator.create("FitnessMax", base.Fitness, weights=[self.criteria_sign, 1.0])
        creator.create("Individual", list, fitness=creator.FitnessMax)

        attributes = []
        # Assign all the parameters defined in the param_grid
        # It uses the distribution parameter to set the sampling function
        for parameter, dimension in self.space.param_grid.items():
            self.toolbox.register(f"{parameter}", dimension.sample)
            attributes.append(getattr(self.toolbox, parameter))

        IND_SIZE = 1

        self.toolbox.register(
            "individual",
            tools.initCycle,
            creator.Individual,
            tuple(attributes),
            n=IND_SIZE,
        )

        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        if len(self.space) == 1:
            sampler = list(self.space.param_grid.values())[0]
            lower, upper = sampler.lower, sampler.upper

            self.toolbox.register(
                "mate", tools.cxSimulatedBinaryBounded, low=lower, up=upper, eta=10
            )
        else:
            self.toolbox.register("mate", tools.cxTwoPoint)

        self.toolbox.register("mutate", self.mutate)
        if self.elitism:
            self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        else:
            self.toolbox.register("select", tools.selRoulette)

        self.toolbox.register("evaluate", self.evaluate)

        self._pop = self._initialize_population()
        self._hof = tools.HallOfFame(self.keep_top_k)

        self._stats = tools.Statistics(lambda ind: ind.fitness.values)
        self._stats.register("fitness", np.mean, axis=0)
        self._stats.register("fitness_std", np.std, axis=0)
        self._stats.register("fitness_max", np.max, axis=0)
        self._stats.register("fitness_min", np.min, axis=0)

        self.logbook = tools.Logbook()

    def _initialize_population(self):
        """
        Initialize the population, using warm-start configurations if provided.
        """
        population = []
        # Seed part of the population with warm-start values
        num_warm_start = min(len(self.warm_start_configs), self.population_size)

        for config in self.warm_start_configs[:num_warm_start]:
            # Sample an individual from the warm-start configuration
            individual_values = self.space.sample_warm_start(config)
            individual_values_list = list(individual_values.values())

            # Manually create the individual and assign its fitness
            individual = creator.Individual(individual_values_list)
            population.append(individual)

        # Fill the remaining population with random individuals
        num_random = self.population_size - num_warm_start
        population.extend(self.toolbox.population(n=num_random))

        return population

    def mutate(self, individual):
        """
        This function is responsible for change a randomly selected parameter from an individual
        Parameters
        ----------
        individual: Individual object
            The individual (set of hyperparameters) that is being generated

        Returns
        -------
            Mutated individual
        """

        # Randomly select one of the hyperparameters
        gen = random.randrange(0, len(self.space))
        parameter_idx = self.space.parameters[gen]
        parameter = self.space[parameter_idx]

        # Using the defined distribution from the para_grid value
        # Make a random sample of the parameter
        individual[gen] = parameter.sample()

        return [individual]

    def evaluate(self, individual):
        """
        Compute the cross-validation scores and record the logbook and mlflow (if specified)
        Parameters
        ----------
        individual: Individual object
            The individual (set of hyperparameters) that is being evaluated
        Returns
        -------
            The fitness value of the estimator candidate, corresponding to the cv-score

        """

        # Dictionary representation of the individual with key-> hyperparameter name, value -> value
        current_generation_params = {
            key: individual[n] for n, key in enumerate(self.space.parameters)
        }

        # Convert hyperparameters to a tuple to use as a key in the cache
        individual_key = tuple(sorted(current_generation_params.items()))

        # Check if the individual has already been evaluated
        if individual_key in self.fitness_cache and self.use_cache:
            # Retrieve cached result
            cached_result = self.fitness_cache[individual_key]
            # Ensure the logbook is updated even if the individual is cached
            self.logbook.record(parameters=cached_result["current_generation_params"])
            return cached_result["fitness"]

        local_estimator = clone(self.estimator)
        local_estimator.set_params(**current_generation_params)

        # Compute the cv-metrics
        cv_results = cross_validate(
            local_estimator,
            self.X_,
            self.y_,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            pre_dispatch=self.pre_dispatch,
            error_score=self.error_score,
            return_train_score=self.return_train_score,
        )

        cv_scores = cv_results[f"test_{self.refit_metric}"]
        score = np.mean(cv_scores)

        novelty_score = novelty_scorer(individual, self._pop)

        # Uses the log config to save in remote log server (e.g MLflow)
        if self.log_config is not None:
            self.log_config.create_run(
                parameters=current_generation_params,
                score=score,
                estimator=local_estimator,
            )

        # These values are used to compute cv_results_ property
        current_generation_params["score"] = score
        current_generation_params["cv_scores"] = cv_scores
        current_generation_params["fit_time"] = cv_results["fit_time"]
        current_generation_params["score_time"] = cv_results["score_time"]

        for metric in self.metrics_list:
            current_generation_params[f"test_{metric}"] = cv_results[f"test_{metric}"]

            if self.return_train_score:
                current_generation_params[f"train_{metric}"] = cv_results[f"train_{metric}"]

        index = len(self.logbook.chapters["parameters"])
        current_generation_params = {"index": index, **current_generation_params}

        # Log the hyperparameters and the cv-score
        self.logbook.record(parameters=current_generation_params)

        fitness_result = [score, novelty_score]

        if self.use_cache:
            # Store the fitness result and the current generation parameters in the cache
            self.fitness_cache[individual_key] = {
                "fitness": fitness_result,
                "current_generation_params": current_generation_params,
            }

        return fitness_result

    def fit(self, X, y, callbacks=None):
        """
        Main method of GASearchCV, starts the optimization
        procedure with the hyperparameters of the given estimator

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            The data to fit. Can be for example a list, or an array.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs), \
            default=None
            The target variable to try to predict in the case of
            supervised learning.
        callbacks: list or callable
            One or a list of the callbacks methods available in
            :class:`~sklearn_genetic.callbacks`.
            The callback is evaluated after fitting the estimators from the generation 1.
        """

        self.X_ = X
        self.y_ = y
        self._n_iterations = self.generations + 1
        self.refit_metric = "score"
        self.multimetric_ = False

        # Make sure the callbacks are valid
        self.callbacks = check_callback(callbacks)

        # Load state if a checkpoint exists
        for callback in self.callbacks:
            if isinstance(callback, ModelCheckpoint):
                if os.path.exists(callback.checkpoint_path):
                    checkpoint_data = callback.load()
                    if checkpoint_data:
                        self.__dict__.update(checkpoint_data["estimator_state"])  # noqa
                        self.logbook = checkpoint_data["logbook"]
                    break

        if callable(self.scoring):
            self.scorer_ = self.scoring
            self.metrics_list = [self.refit_metric]
        elif self.scoring is None or isinstance(self.scoring, str):
            self.scorer_ = check_scoring(self.estimator, self.scoring)
            self.metrics_list = [self.refit_metric]
        else:
            self.scorer_ = _check_multimetric_scoring(self.estimator, self.scoring)
            self._check_refit_for_multimetric(self.scorer_)
            self.refit_metric = self.refit
            self.metrics_list = self.scorer_.keys()
            self.multimetric_ = True

        # Check cv and get the n_splits
        cv_orig = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        self.n_splits_ = cv_orig.get_n_splits(X, y)

        # Set the DEAPs necessary methods
        self._register()

        # Optimization routine from the selected evolutionary algorithm
        pop, log, n_gen = self._select_algorithm(pop=self._pop, stats=self._stats, hof=self._hof)

        # Update the _n_iterations value as the algorithm could stop earlier due a callback
        self._n_iterations = n_gen

        self.cv_results_ = create_gasearch_cv_results_(
            logbook=self.logbook,
            space=self.space,
            return_train_score=self.return_train_score,
            metrics=self.metrics_list,
        )

        self.history = {
            "gen": log.select("gen"),
            "fitness": log.select("fitness"),
            "fitness_std": log.select("fitness_std"),
            "fitness_max": log.select("fitness_max"),
            "fitness_min": log.select("fitness_min"),
        }

        # Imitate the logic of scikit-learn refit parameter
        if self.refit:
            self.best_index_ = self.cv_results_[f"rank_test_{self.refit_metric}"].argmin()
            self.best_score_ = self.cv_results_[f"mean_test_{self.refit_metric}"][self.best_index_]
            self.best_params_ = self.cv_results_["params"][self.best_index_]

            self.estimator.set_params(**self.best_params_)

            refit_start_time = time.time()
            self.estimator.fit(
                self.X_,
                self.y_,
            )
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

            self.best_estimator_ = self.estimator
            self.estimator_ = self.best_estimator_

            # hof keeps the best params according to the fitness value
            # To be consistent with self.best_estimator_, if more than 1 model gets the
            # same score, it could lead to differences between hof and self.best_estimator_
            self._hof.remove(0)
            self._hof.items.insert(0, list(self.best_params_.values()))
            self._hof.keys.insert(0, self.best_score_)

        self.hof = {
            k: {key: self._hof[k][n] for n, key in enumerate(self.space.parameters)}
            for k in range(len(self._hof))
        }

        del creator.FitnessMax
        del creator.Individual

        return self

    def save(self, filepath):
        """Save the current state of the GASearchCV instance to a file."""
        try:
            checkpoint_data = {"estimator_state": self.__dict__, "logbook": None}
            if hasattr(self, "logbook"):
                checkpoint_data["logbook"] = self.logbook
            with open(filepath, "wb") as f:
                pickle.dump(checkpoint_data, f)
            print(f"GASearchCV model successfully saved to {filepath}")
        except Exception as e:
            print(f"Error saving GASearchCV: {e}")

    def load(self, filepath):
        """Load a GASearchCV instance from a file."""
        try:
            with open(filepath, "rb") as f:
                checkpoint_data = pickle.load(f)
                for key, value in checkpoint_data["estimator_state"].items():
                    setattr(self, key, value)
                self.logbook = checkpoint_data["logbook"]
            print(f"GASearchCV model successfully loaded from {filepath}")
        except Exception as e:
            print(f"Error loading GASearchCV: {e}")

    def _select_algorithm(self, pop, stats, hof):
        """
        It selects the algorithm to run from the sklearn_genetic.algorithms module
        based in the parameter self.algorithm.

        Parameters
        ----------
        pop: pop object from DEAP
        stats: stats object from DEAP
        hof: hof object from DEAP

        Returns
        -------
        pop: pop object
            The last evaluated population
        log: Logbook object
            It contains the calculated metrics {'fitness', 'fitness_std', 'fitness_max', 'fitness_min'}
            the number of generations and the number of evaluated individuals per generation
        n_gen: int
            The number of generations that the evolutionary algorithm ran
        """

        selected_algorithm = algorithms_factory.get(self.algorithm, None)
        if selected_algorithm:
            pop, log, gen = selected_algorithm(
                pop,
                self.toolbox,
                mu=self.population_size,
                lambda_=2 * self.population_size,
                cxpb=self.crossover_adapter,
                stats=stats,
                mutpb=self.mutation_adapter,
                ngen=self.generations,
                halloffame=hof,
                callbacks=self.callbacks,
                verbose=self.verbose,
                estimator=self,
            )

        else:
            raise ValueError(
                f"The algorithm {self.algorithm} is not supported, "
                f"please select one from {Algorithms.list()}"
            )

        return pop, log, gen

    def _run_search(self, evaluate_candidates):
        pass  # noqa

    @property
    def _fitted(self):
        try:
            check_is_fitted(self.estimator)
            is_fitted = True
        except Exception as e:
            is_fitted = False

        has_history = hasattr(self, "history") and bool(self.history)
        return all([is_fitted, has_history, self.refit])

    def __getitem__(self, index):
        """

        Parameters
        ----------
        index: slice required to get

        Returns
        -------
        Best solution of the iteration corresponding to the index number
        """
        if not self._fitted:
            raise NotFittedError(
                f"This GASearchCV instance is not fitted yet "
                f"or used refit=False. Call 'fit' with appropriate "
                f"arguments before using this estimator."
            )

        return {
            "gen": self.history["gen"][index],
            "fitness": self.history["fitness"][index],
            "fitness_std": self.history["fitness_std"][index],
            "fitness_max": self.history["fitness_max"][index],
            "fitness_min": self.history["fitness_min"][index],
        }

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        """
        Returns
        -------
        Iteration over the statistics found in each generation
        """
        if self.n < self._n_iterations + 1:
            result = self.__getitem__(self.n)
            self.n += 1
            return result
        else:
            raise StopIteration  # pragma: no cover

    def __len__(self):
        """
        Returns
        -------
        Number of generations fitted if .fit method has been called,
        self.generations otherwise
        """
        return self._n_iterations


class GAFeatureSelectionCV(MetaEstimatorMixin, SelectorMixin, BaseEstimator):
    """
    Evolutionary optimization for feature selection.

    GAFeatureSelectionCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba", "decision_function",
    "predict_log_proba" if they are implemented in the
    estimator used.
    The features (variables) used by the estimator are found by optimizing
    the cv-scores and by minimizing the number of features

    Parameters
    ----------
    estimator : estimator object, default=None
        estimator object implementing 'fit'
        The object to use to fit the data.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - CV splitter,
        - An iterable yielding (train, test) splits as arrays of indices.
        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

    population_size : int, default=10
        Size of the initial population to sample randomly generated individuals.

    generations : int, default=40
        Number of generations or iterations to run the evolutionary algorithm.

    crossover_probability : float or a Scheduler, default=0.2
        Probability of crossover operation between two individuals.

    mutation_probability : float or a Scheduler, default=0.8
        Probability of child mutation.

    tournament_size : int, default=3
        Number of individuals to perform tournament selection.

    elitism : bool, default=True
        If True takes the *tournament_size* best solution to the next generation.

    max_features : int, default=None
        The upper bound number of features to be selected.

    scoring : str, callable, list, tuple or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set.
        If `scoring` represents a single score, one can use:

        - a single string;
        - a callable that returns a single value.
        If `scoring` represents multiple scores, one can use:

        - a list or tuple of unique strings;
        - a callable returning a dictionary where the keys are the metric
          names and the values are the metric scores;
        - a dictionary with metric names as keys and callables a values.

    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the cross-validation splits.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    verbose : bool, default=True
        If ``True``, shows the metrics on the optimization routine.

    keep_top_k : int, default=1
        Number of best solutions to keep in the hof object. If a callback stops the algorithm before k iterations,
        it will return only one set of parameters per iteration.

    criteria : {'max', 'min'} , default='max'
        ``max`` if a higher scoring metric is better, ``min`` otherwise.

    algorithm : {'eaMuPlusLambda', 'eaMuCommaLambda', 'eaSimple'}, default='eaMuPlusLambda'
        Evolutionary algorithm to use.
        See more details in the deap algorithms documentation.

    refit : bool, str, or callable, default=True
        Refit an estimator using the best found parameters on the whole
        dataset.
        For multiple metric evaluation, this needs to be a `str` denoting the
        scorer that would be used to find the best parameters for refitting
        the estimator at the end.
        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``FeatureSelectionCV`` instance.
        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_params_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer.
        See ``scoring`` parameter to know more about multiple metric
        evaluation.

        If ``False``, it is not possible to make predictions
        using this GASearchCV instance after fitting.

    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:
            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
            - An int, giving the exact number of total jobs that are
              spawned
            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to ``'raise'``, the error is raised.
        If a numeric value is given, FitFailedWarning is raised.

    return_train_score: bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

    log_config : :class:`~sklearn_genetic.mlflow.MLflowConfig`, default = None
        Configuration to log metrics and models to mlflow, of None,
        no mlflow logging will be performed

    use_cache: bool, default=True
        If set to true it will avoid to re-evaluating solutions that have already seen,
        otherwise it will always evaluate the solutions to get the performance metrics

    Attributes
    ----------

    logbook : :class:`DEAP.tools.Logbook`
        Contains the logs of every set of hyperparameters fitted with its average scoring metric.
    history : dict
        Dictionary of the form:
        {"gen": [],
        "fitness": [],
        "fitness_std": [],
        "fitness_max": [],
        "fitness_min": []}

         *gen* returns the index of the evaluated generations.
         Each entry on the others lists, represent the average metric in each generation.

    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.
    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score
        on the left out data. Not available if ``refit=False``.
    best_features_ : list
        List of bool, each index represents one feature in the same order the data was fed.
        1 means the feature was selected, 0 means the features was discarded.
    support_ : list
        The mask of selected features.
    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.
    n_splits_ : int
        The number of cross-validation splits (folds/iterations).
    n_features_in_ : int
        Number of features seen (selected) during fit.
    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.
        This is present only if ``refit`` is not False.
    """

    def __init__(
        self,
        estimator,
        cv=3,
        scoring=None,
        population_size=50,
        generations=80,
        crossover_probability=0.2,
        mutation_probability=0.8,
        tournament_size=3,
        elitism=True,
        max_features=None,
        verbose=True,
        keep_top_k=1,
        criteria="max",
        algorithm="eaMuPlusLambda",
        refit=True,
        n_jobs=1,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
        log_config=None,
        use_cache=True,
    ):
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.population_size = population_size
        self.generations = generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.crossover_adapter = check_adapter(self.crossover_probability)
        self.mutation_adapter = check_adapter(self.mutation_probability)
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.max_features = max_features
        self.verbose = verbose
        self.keep_top_k = keep_top_k
        self.criteria = criteria
        self.algorithm = algorithm
        self.refit = refit
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score
        # self.creator = creator
        self.log_config = log_config
        self.use_cache = use_cache
        self.fitness_cache = {}

        # Check that the estimator is compatible with scikit-learn
        if not is_classifier(self.estimator) and not is_regressor(self.estimator):
            raise ValueError(f"{self.estimator} is not a valid Sklearn classifier or regressor")

        if criteria not in Criteria.list():
            raise ValueError(f"Criteria must be one of {Criteria.list()}, got {criteria} instead")
        # Minimization is handle like an optimization problem with a change in the score sign
        elif criteria == Criteria.max.value:
            self.criteria_sign = 1.0
        elif criteria == Criteria.min.value:
            self.criteria_sign = -1.0

    def _register(self):
        """
        This function is the responsible for registering the DEAPs necessary methods
        and create other objects to hold the hof, logbook and stats.
        """
        self.toolbox = base.Toolbox()

        # Criteria sign to set max or min problem
        # And -1.0 as second weight to minimize number of features
        creator.create("FitnessMax", base.Fitness, weights=[self.criteria_sign, -1.0])
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Register the array to choose the features
        # Each binary value represents if the feature is selected or not

        self.toolbox.register(
            "individual",
            weighted_bool_individual,
            creator.Individual,
            weight=self.features_proportion,
            size=self.n_features,
        )

        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("mate", cxUniform, indpb=self.crossover_adapter.current_value)
        self.toolbox.register("mutate", mutFlipBit, indpb=self.mutation_adapter.current_value)

        if self.elitism:
            self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        else:
            self.toolbox.register("select", tools.selRoulette)

        self.toolbox.register("evaluate", self.evaluate)

        self._pop = self.toolbox.population(n=self.population_size)
        self._hof = tools.HallOfFame(self.keep_top_k)

        # Stats among axis 0 to get two values:
        # One based on the score and the other in the number of features
        self._stats = tools.Statistics(ind_fitness_values)
        self._stats.register("fitness", np.mean, axis=0)
        self._stats.register("fitness_std", np.std, axis=0)
        self._stats.register("fitness_max", np.max, axis=0)
        self._stats.register("fitness_min", np.min, axis=0)

        self.logbook = tools.Logbook()

    def evaluate(self, individual):
        """
        Compute the cross-validation scores and record the logbook and mlflow (if specified)
        Parameters
        ----------
        individual: Individual object
            The individual (set of features) that is being evaluated

        Returns
        -------
        fitness: List
            Returns a list with two values.
            The first one is the corresponding to the cv-score
            The second one is the number of features selected

        """

        bool_individual = np.array(individual, dtype=bool)

        current_generation_params = {"features": bool_individual}

        local_estimator = clone(self.estimator)
        n_selected_features = np.sum(individual)

        # Convert the individual to a tuple to use as a key in the cache
        individual_key = tuple(individual)

        # Check if the individual has already been evaluated
        if individual_key in self.fitness_cache and self.use_cache:
            cached_result = self.fitness_cache[individual_key]
            # Ensure the logbook is updated even if the individual is cached
            self.logbook.record(parameters=cached_result["current_generation_features"])
            return cached_result["fitness"]

        # Compute the cv-metrics using only the selected features
        cv_results = cross_validate(
            local_estimator,
            self.X_[:, bool_individual],
            self.y_,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            pre_dispatch=self.pre_dispatch,
            error_score=self.error_score,
            return_train_score=self.return_train_score,
        )

        cv_scores = cv_results[f"test_{self.refit_metric}"]
        score = np.mean(cv_scores)

        # Uses the log config to save in remote log server (e.g MLflow)
        if self.log_config is not None:
            self.log_config.create_run(
                parameters=current_generation_params,
                score=score,
                estimator=local_estimator,
            )

        # These values are used to compute cv_results_ property
        current_generation_params["score"] = score
        current_generation_params["cv_scores"] = cv_scores
        current_generation_params["fit_time"] = cv_results["fit_time"]
        current_generation_params["score_time"] = cv_results["score_time"]

        for metric in self.metrics_list:
            current_generation_params[f"test_{metric}"] = cv_results[f"test_{metric}"]

            if self.return_train_score:
                current_generation_params[f"train_{metric}"] = cv_results[f"train_{metric}"]

        index = len(self.logbook.chapters["parameters"])
        current_generation_features = {"index": index, **current_generation_params}

        # Log the features and the cv-score
        self.logbook.record(parameters=current_generation_features)

        # Penalize individuals with more features than the max_features parameter

        if self.max_features and (
            n_selected_features > self.max_features or n_selected_features == 0
        ):
            score = -self.criteria_sign * 100000

            # Prepare the fitness result
        fitness_result = [score, n_selected_features]

        if self.use_cache:
            # Store the fitness result and the current generation features in the cache
            self.fitness_cache[individual_key] = {
                "fitness": fitness_result,
                "current_generation_features": current_generation_features,
            }

        return fitness_result

    def fit(self, X, y, callbacks=None):
        """
        Main method of GAFeatureSelectionCV, starts the optimization
        procedure with to find the best features set

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to fit. Can be for example a list, or an array.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs), \
            default=None
            The target variable to try to predict in the case of
            supervised learning.
        callbacks: list or callable
            One or a list of the callbacks methods available in
            :class:`~sklearn_genetic.callbacks`.
            The callback is evaluated after fitting the estimators from the generation 1.
        """

        self.X_, self.y_ = check_X_y(X, y)
        self.n_features = X.shape[1]
        self._n_iterations = self.generations + 1
        self.refit_metric = "score"
        self.multimetric_ = False

        self.features_proportion = None
        if self.max_features:
            self.features_proportion = self.max_features / self.n_features

        # Make sure the callbacks are valid
        self.callbacks = check_callback(callbacks)

        # Load state if a checkpoint exists
        for callback in self.callbacks:
            if isinstance(callback, ModelCheckpoint):
                if os.path.exists(callback.checkpoint_path):
                    checkpoint_data = callback.load()
                    if checkpoint_data:
                        self.__dict__.update(checkpoint_data["estimator_state"])  # noqa
                        self.logbook = checkpoint_data["logbook"]
                    break

        if callable(self.scoring):
            self.scorer_ = self.scoring
            self.metrics_list = [self.refit_metric]
        elif self.scoring is None or isinstance(self.scoring, str):
            self.scorer_ = check_scoring(self.estimator, self.scoring)
            self.metrics_list = [self.refit_metric]
        else:
            self.scorer_ = _check_multimetric_scoring(self.estimator, self.scoring)
            self._check_refit_for_multimetric(self.scorer_)
            self.refit_metric = self.refit
            self.metrics_list = self.scorer_.keys()
            self.multimetric_ = True

        # Check cv and get the n_splits
        cv_orig = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        self.n_splits_ = cv_orig.get_n_splits(X, y)

        # Set the DEAPs necessary methods
        self._register()

        # Optimization routine from the selected evolutionary algorithm
        pop, log, n_gen = self._select_algorithm(pop=self._pop, stats=self._stats, hof=self._hof)

        # Update the _n_iterations value as the algorithm could stop earlier due a callback
        self._n_iterations = n_gen

        self.best_features_ = np.array(self._hof[0], dtype=bool)
        self.support_ = self.best_features_

        self.cv_results_ = create_feature_selection_cv_results_(
            logbook=self.logbook,
            return_train_score=self.return_train_score,
            metrics=self.metrics_list,
        )

        self.history = {
            "gen": log.select("gen"),
            "fitness": log.select("fitness"),
            "fitness_std": log.select("fitness_std"),
            "fitness_max": log.select("fitness_max"),
            "fitness_min": log.select("fitness_min"),
        }

        if self.refit:
            bool_individual = np.array(self.best_features_, dtype=bool)

            refit_start_time = time.time()
            self.estimator.fit(self.X_[:, bool_individual], self.y_)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

            self.best_estimator_ = self.estimator
            self.estimator_ = self.best_estimator_

        self.hof = self._hof

        del creator.FitnessMax
        del creator.Individual

        return self

    def save(self, filepath):
        """Save the current state of the GAFeatureSelectionCV instance to a file."""
        try:
            checkpoint_data = {"estimator_state": self.__dict__, "logbook": None}
            if hasattr(self, "logbook"):
                checkpoint_data["logbook"] = self.logbook

            with open(filepath, "wb") as f:
                pickle.dump(checkpoint_data, f)
            print(f"GAFeatureSelectionCV model successfully saved to {filepath}")
        except Exception as e:
            print(f"Error saving GAFeatureSelectionCV: {e}")

    def load(self, filepath):
        """Load a GAFeatureSelectionCV instance from a file."""
        try:
            with open(filepath, "rb") as f:
                checkpoint_data = pickle.load(f)
                for key, value in checkpoint_data["estimator_state"].items():
                    setattr(self, key, value)
                self.logbook = checkpoint_data["logbook"]
            print(f"GAFeatureSelectionCV model successfully loaded from {filepath}")  # noqa
        except Exception as e:
            print(f"Error loading GAFeatureSelectionCV: {e}")

    def _select_algorithm(self, pop, stats, hof):
        """
        It selects the algorithm to run from the sklearn_genetic.algorithms module
        based in the parameter self.algorithm.

        Parameters
        ----------
        pop: pop object from DEAP
        stats: stats object from DEAP
        hof: hof object from DEAP

        Returns
        -------
        pop: pop object
            The last evaluated population
        log: Logbook object
            It contains the calculated metrics {'fitness', 'fitness_std', 'fitness_max', 'fitness_min'}
            the number of generations and the number of evaluated individuals per generation
        n_gen: int
            The number of generations that the evolutionary algorithm ran
        """

        selected_algorithm = algorithms_factory.get(self.algorithm, None)
        if selected_algorithm:
            pop, log, gen = selected_algorithm(
                pop,
                self.toolbox,
                mu=self.population_size,
                lambda_=2 * self.population_size,
                cxpb=self.crossover_adapter,
                stats=stats,
                mutpb=self.mutation_adapter,
                ngen=self.generations,
                halloffame=hof,
                callbacks=self.callbacks,
                verbose=self.verbose,
                estimator=self,
            )

        else:
            raise ValueError(
                f"The algorithm {self.algorithm} is not supported, "
                f"please select one from {Algorithms.list()}"
            )

        return pop, log, gen

    def _run_search(self, evaluate_candidates):
        pass  # noqa

    @property
    def _fitted(self):
        try:
            check_is_fitted(self.estimator)
            is_fitted = True
        except Exception as e:
            is_fitted = False

        has_history = hasattr(self, "history") and bool(self.history)
        return all([is_fitted, has_history, self.refit])

    def __getitem__(self, index):
        """

        Parameters
        ----------
        index: slice required to get

        Returns
        -------
        Best solution of the iteration corresponding to the index number
        """
        if not self._fitted:
            raise NotFittedError(
                f"This GAFeatureSelectionCV instance is not fitted yet "
                f"or used refit=False. Call 'fit' with appropriate "
                f"arguments before using this estimator."
            )

        return {
            "gen": self.history["gen"][index],
            "fitness": self.history["fitness"][index],
            "fitness_std": self.history["fitness_std"][index],
            "fitness_max": self.history["fitness_max"][index],
            "fitness_min": self.history["fitness_min"][index],
        }

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        """
        Returns
        -------
        Iteration over the statistics found in each generation
        """
        if self.n < self._n_iterations + 1:
            result = self.__getitem__(self.n)
            self.n += 1
            return result
        else:
            raise StopIteration  # pragma: no cover

    def __len__(self):
        """
        Returns
        -------
        Number of generations fitted if .fit method has been called,
        self.generations otherwise
        """
        return self._n_iterations

    def _check_refit_for_multimetric(self, scores):  # pragma: no cover
        """Check `refit` is compatible with `scores` is valid"""
        multimetric_refit_msg = (
            "For multi-metric scoring, the parameter refit must be set to a "
            "scorer key or a callable to refit an estimator with the best "
            "parameter setting on the whole data and make the best_* "
            "attributes available for that metric. If this is not needed, "
            f"refit should be set to False explicitly. {self.refit!r} was "
            "passed."
        )

        valid_refit_dict = isinstance(self.refit, str) and self.refit in scores

        if self.refit is not False and not valid_refit_dict and not callable(self.refit):
            raise ValueError(multimetric_refit_msg)

    @property
    def n_features_in_(self):  # pragma: no cover
        """Number of features seen during `fit`."""
        # For consistency with other estimators we raise a AttributeError so
        # that hasattr() fails if the estimator isn't fitted.
        if not self._fitted:
            raise AttributeError(
                "{} object has no n_features_in_ attribute.".format(self.__class__.__name__)
            )

        return self.n_features

    def _get_support_mask(self):
        if not self._fitted:
            raise NotFittedError(
                f"This GAFeatureSelectionCV instance is not fitted yet "
                f"or used refit=False. Call 'fit' with appropriate "
                f"arguments before using this estimator."
            )
        return self.best_features_

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        """Call decision_function on the estimator with the best found features.
       Only available if ``refit=True`` and the underlying estimator supports
       ``decision_function``.

       Parameters
       ----------
       X : indexable, length n_samples
           Must fulfill the input assumptions of the
           underlying estimator.

       Returns
       -------
       y_score : ndarray of shape (n_samples,) or (n_samples, n_classes) \
               or (n_samples, n_classes * (n_classes-1) / 2)
           Result of the decision function for `X` based on the estimator with
           the best found parameters.
       """
        return self.estimator.decision_function(self.transform(X))

    @available_if(_estimator_has("predict"))
    def predict(self, X):
        """Call predict on the estimator with the best found features.
        Only available if ``refit=True`` and the underlying estimator supports
        ``predict``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted labels or values for `X` based on the estimator with
            the best found parameters.
        """
        return self.estimator.predict(self.transform(X))

    @available_if(_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X):
        """Call predict_log_proba on the estimator with the best found features.
        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_log_proba``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Predicted class log-probabilities for `X` based on the estimator
            with the best found parameters. The order of the classes
            corresponds to that in the fitted attribute :term:`classes_`.
        """
        return self.estimator.predict_log_proba(self.transform(X))

    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X):
        """Call predict_proba on the estimator with the best found features.
        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_proba``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Predicted class probabilities for `X` based on the estimator with
            the best found parameters. The order of the classes corresponds
            to that in the fitted attribute :term:`classes_`.
        """
        return self.estimator.predict_proba(self.transform(X))

    @available_if(_estimator_has("score"))
    def score(self, X, y):
        """Return the score on the given data, if the estimator has been refit.
        This uses the score defined by ``scoring`` where provided, and the
        ``best_estimator_.score`` method otherwise.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : array-like of shape (n_samples, n_output) \
            or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.

        Returns
        -------
        score : float
            The score defined by ``scoring`` if provided, and the
            ``best_estimator_.score`` method otherwise.
        """
        return self.estimator.score(self.transform(X), y)


# helpers


def ind_fitness_values(ind):
    return ind.fitness.values
