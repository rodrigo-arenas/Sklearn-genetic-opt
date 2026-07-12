import random
import time
import warnings

import numpy as np
from deap import base, creator, tools
from sklearn.base import clone
from sklearn.model_selection import cross_validate
from sklearn.base import is_classifier, is_regressor, BaseEstimator, MetaEstimatorMixin

try:
    from sklearn.base import is_outlier_detector
except ImportError:
    # Fallback for older sklearn versions
    def is_outlier_detector(estimator):
        return hasattr(estimator, "fit_predict") and hasattr(estimator, "decision_function")


def _safe_estimator_check(check, estimator):
    try:
        return check(estimator)
    except AttributeError:
        return False


def _is_classifier(estimator):
    return _safe_estimator_check(is_classifier, estimator)


def _is_regressor(estimator):
    return _safe_estimator_check(is_regressor, estimator)


def _is_outlier_detector(estimator):
    return _safe_estimator_check(is_outlier_detector, estimator)


from sklearn.feature_selection import SelectorMixin
from sklearn.utils import check_X_y, check_random_state
from sklearn.utils.metaestimators import available_if
from sklearn.feature_selection._from_model import _estimator_has
from sklearn.metrics import check_scoring
from sklearn.exceptions import NotFittedError
from sklearn.model_selection._search import BaseSearchCV
from sklearn.model_selection._split import check_cv
from sklearn.metrics._scorer import _check_multimetric_scoring

from .parameters import Criteria
from .space import Categorical, Continuous, Integer, Space
from ._base import GeneticEstimatorMixin, reset_adapters as _reset_adapters
from .callbacks.validations import check_callback
from .schedules.validations import check_adapter
from .utils.cv_scores import (
    create_gasearch_cv_results_,
    create_feature_selection_cv_results_,
)
from .utils.random import weighted_bool_individual
from .utils.tools import cxUniform, mutFlipBit
from .evaluation import (
    create_fit_stats as _create_fit_stats,
    evaluate_population as _evaluate_population_batch,
    logbook_record as _logbook_record,
    record_fit_stats as _record_fit_stats,
    validate_error_score as _validate_error_score,
    validate_parallel_backend as _validate_parallel_backend,
)
from .population import (
    initialize_feature_population,
    initialize_search_population,
    validate_population_initializer as _validate_population_initializer,
)
from .optimizer_control import (
    adaptive_tournament_size,
    validate_optimizer_control as _validate_optimizer_control,
)

import os
from .callbacks.model_checkpoint import ModelCheckpoint
from .config import EvolutionConfig, OptimizationConfig, PopulationConfig, RuntimeConfig


def _seed_global_rngs(random_state):
    """Seed the global ``random`` and NumPy RNGs from a single ``random_state``.

    A genetic search draws on several sources of randomness — population
    initialization, DEAP's mutation/crossover operators (which use the global
    ``random`` module), random immigrants, and NumPy sampling. Seeding them all
    from one estimator-level ``random_state`` at the start of ``fit`` gives
    reproducible runs without callers having to seed the global RNGs by hand.

    ``random_state=None`` keeps the default non-deterministic behaviour, matching
    scikit-learn's convention.
    """
    if random_state is None:
        return
    seed = int(check_random_state(random_state).randint(0, 2**31 - 1))
    random.seed(seed)
    np.random.seed(seed)


def _resolve_config_value(config, field_name, fallback):
    if config is None:
        return fallback

    return getattr(config, field_name, fallback)


class GASearchCV(GeneticEstimatorMixin, BaseSearchCV):
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
        Size of the initial population to sample generated individuals.

    evolution_config : :class:`~sklearn_genetic.config.EvolutionConfig`, default=None
        Optional grouped configuration for core genetic algorithm controls such
        as population size, generation count, crossover, mutation, tournament
        size, elitism, hall-of-fame size, criteria, and algorithm.

    population_config : :class:`~sklearn_genetic.config.PopulationConfig`, default=None
        Optional grouped configuration for initial population behavior,
        including ``initializer`` and ``warm_start_configs``.

    runtime_config : :class:`~sklearn_genetic.config.RuntimeConfig`, default=None
        Optional grouped configuration for parallelism, caching, train-score
        collection, error handling, and verbose output.

    optimization_config : :class:`~sklearn_genetic.config.OptimizationConfig`, default=None
        Optional grouped configuration for local refinement, diversity control,
        adaptive selection, fitness sharing, and robust final selection.

    population_initializer : {'smart', 'random'}, default='smart'
        Strategy used to generate the initial population. ``'smart'`` combines
        valid warm-start configurations, valid estimator defaults, Latin
        hypercube sampling for numeric dimensions, stratified categorical
        values, and duplicate avoidance. ``'random'`` uses the previous random
        sampling behavior.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the search. When set, it seeds every
        stochastic step from a single place at ``fit`` time — population
        initialization (including the Latin hypercube sampler), mutation,
        crossover, and random immigrants — so repeated fits give identical
        results without manually seeding the global ``random`` / ``numpy`` RNGs.
        ``None`` keeps the default non-deterministic behaviour.

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
        Number of jobs to run in parallel. Candidate evaluations in each
        generation are parallelized when possible; each candidate then runs
        cross-validation sequentially to avoid nested parallelism.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    parallel_backend : {'auto', 'population', 'cv'}, default='auto'
        Controls where ``n_jobs`` parallelism is applied during ``fit``.
        ``'auto'`` and ``'population'`` evaluate unique candidates in each
        generation in parallel when possible. ``'cv'`` keeps candidate
        evaluation serial and passes ``n_jobs`` to each candidate's
        cross-validation call.

    local_search : bool, default=False
        If ``True``, run a short local refinement phase around the current
        hall-of-fame individuals after the genetic search finishes.

    local_search_top_k : int, default=1
        Number of hall-of-fame individuals used as local-search seeds.

    local_search_steps : int, default=1
        Number of neighbor candidates generated per local-search seed.

    local_search_radius : float, default=0.1
        Fraction of the search range used to sample local numeric neighbors.
        For categorical parameters, a different category is sampled.

    diversity_control : bool, default=True
        If ``True``, monitor diversity and stagnation to boost mutation,
        replace duplicate candidates, and inject random immigrants.

    adaptive_selection : bool, default=False
        If ``True``, adapt tournament size from generation telemetry. Selection
        pressure is reduced when diversity is low or the search is stagnant,
        and slightly increased when the population is improving with enough
        diversity.

    selection_pressure_min : int, default=2
        Minimum tournament size used by adaptive selection.

    selection_pressure_max : int, default=None
        Maximum tournament size used by adaptive selection. If ``None``, the
        maximum is one larger than ``tournament_size``.

    offspring_diversity_retries : int, default=0
        Number of retries used when replacing duplicate or parent-matching
        offspring with new random candidates.

    diversity_threshold : float, default=0.25
        Diversity value below which diversity control can trigger.

    diversity_stagnation_generations : int, default=5
        Number of stagnant generations after which diversity control can
        inject random immigrants.

    diversity_mutation_boost : float, default=2.0
        Multiplicative boost applied to mutation probability when diversity
        control triggers. The boosted value is capped to DEAP's valid range.

    random_immigrants_fraction : float, default=0.1
        Fraction of offspring replaced by random immigrants when diversity
        control triggers.

    fitness_sharing : bool, default=False
        If ``True``, temporarily penalize candidates in crowded niches during
        selection. Raw cross-validation scores and ``cv_results_`` are not
        modified.

    sharing_radius : float, default=0.2
        Normalized distance below which two individuals are considered part of
        the same niche for fitness sharing.

    sharing_alpha : float, default=1.0
        Shape parameter that controls how quickly sharing pressure decreases
        with distance inside ``sharing_radius``.

    final_selection : bool, default=False
        If ``True``, re-evaluate the top ``final_selection_top_k`` candidates
        after the GA finishes and select ``best_params_`` from those robust
        final scores before refitting.

    final_selection_top_k : int, default=3
        Number of top candidates from the original GA ``cv_results_`` to
        re-evaluate during final selection.

    final_selection_cv : int, cross-validation splitter or iterable, default=None
        Cross-validation strategy used for final selection. If ``None``, the
        same CV splits used during the GA are reused.

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
        than CPUs can process. This parameter can be ``None`` to dispatch all
        jobs immediately, an integer number of total jobs to spawn, or a string
        expression as a function of ``n_jobs``, such as ``'2*n_jobs'``.

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
        Dictionary with one list per generation. It includes ``gen``,
        ``fitness``, ``fitness_std``, ``fitness_best``, ``fitness_max``, ``fitness_min``,
        population diversity fields, stagnation fields, optimizer-control
        telemetry, and local-refinement telemetry.

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
    fit_stats_ : dict
        Counters collected during the last ``fit`` call. Includes evaluated
        candidates, unique candidates, cross-validation calls, cache hits,
        duplicate candidates, skipped invalid candidates, and population-level
        parallel/serial batch counts.
    """

    def __init__(
        self,
        estimator,
        cv=3,
        param_grid=None,
        scoring=None,
        population_size=50,
        generations=80,
        crossover_probability=0.8,
        mutation_probability=0.1,
        tournament_size=3,
        elitism=True,
        verbose=True,
        keep_top_k=1,
        criteria="max",
        algorithm="eaMuPlusLambda",
        refit=True,
        n_jobs=None,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
        log_config=None,
        use_cache=True,
        warm_start_configs=None,
        evolution_config=None,
        population_config=None,
        runtime_config=None,
        optimization_config=None,
        parallel_backend="auto",
        population_initializer="smart",
        random_state=None,
        local_search=False,
        local_search_top_k=1,
        local_search_steps=1,
        local_search_radius=0.1,
        diversity_control=True,
        diversity_threshold=0.25,
        diversity_stagnation_generations=5,
        diversity_mutation_boost=2.0,
        random_immigrants_fraction=0.1,
        adaptive_selection=False,
        selection_pressure_min=2,
        selection_pressure_max=None,
        offspring_diversity_retries=0,
        fitness_sharing=False,
        sharing_radius=0.2,
        sharing_alpha=1.0,
        final_selection=False,
        final_selection_top_k=3,
        final_selection_cv=None,
    ):
        legacy_warm_start_configs = warm_start_configs

        population_size = _resolve_config_value(
            evolution_config, "population_size", population_size
        )
        generations = _resolve_config_value(evolution_config, "generations", generations)
        crossover_probability = _resolve_config_value(
            evolution_config, "crossover_probability", crossover_probability
        )
        mutation_probability = _resolve_config_value(
            evolution_config, "mutation_probability", mutation_probability
        )
        tournament_size = _resolve_config_value(
            evolution_config, "tournament_size", tournament_size
        )
        elitism = _resolve_config_value(evolution_config, "elitism", elitism)
        keep_top_k = _resolve_config_value(evolution_config, "keep_top_k", keep_top_k)
        criteria = _resolve_config_value(evolution_config, "criteria", criteria)
        algorithm = _resolve_config_value(evolution_config, "algorithm", algorithm)

        population_initializer = _resolve_config_value(
            population_config, "initializer", population_initializer
        )
        warm_start_configs = _resolve_config_value(
            population_config, "warm_start_configs", warm_start_configs
        )

        n_jobs = _resolve_config_value(runtime_config, "n_jobs", n_jobs)
        pre_dispatch = _resolve_config_value(runtime_config, "pre_dispatch", pre_dispatch)
        error_score = _resolve_config_value(runtime_config, "error_score", error_score)
        return_train_score = _resolve_config_value(
            runtime_config, "return_train_score", return_train_score
        )
        use_cache = _resolve_config_value(runtime_config, "use_cache", use_cache)
        parallel_backend = _resolve_config_value(
            runtime_config, "parallel_backend", parallel_backend
        )
        verbose = _resolve_config_value(runtime_config, "verbose", verbose)

        local_search = _resolve_config_value(optimization_config, "local_search", local_search)
        local_search_top_k = _resolve_config_value(
            optimization_config, "local_search_top_k", local_search_top_k
        )
        local_search_steps = _resolve_config_value(
            optimization_config, "local_search_steps", local_search_steps
        )
        local_search_radius = _resolve_config_value(
            optimization_config, "local_search_radius", local_search_radius
        )
        diversity_control = _resolve_config_value(
            optimization_config, "diversity_control", diversity_control
        )
        diversity_threshold = _resolve_config_value(
            optimization_config, "diversity_threshold", diversity_threshold
        )
        diversity_stagnation_generations = _resolve_config_value(
            optimization_config,
            "diversity_stagnation_generations",
            diversity_stagnation_generations,
        )
        diversity_mutation_boost = _resolve_config_value(
            optimization_config, "diversity_mutation_boost", diversity_mutation_boost
        )
        random_immigrants_fraction = _resolve_config_value(
            optimization_config, "random_immigrants_fraction", random_immigrants_fraction
        )
        adaptive_selection = _resolve_config_value(
            optimization_config, "adaptive_selection", adaptive_selection
        )
        selection_pressure_min = _resolve_config_value(
            optimization_config, "selection_pressure_min", selection_pressure_min
        )
        selection_pressure_max = _resolve_config_value(
            optimization_config, "selection_pressure_max", selection_pressure_max
        )
        offspring_diversity_retries = _resolve_config_value(
            optimization_config, "offspring_diversity_retries", offspring_diversity_retries
        )
        fitness_sharing = _resolve_config_value(
            optimization_config, "fitness_sharing", fitness_sharing
        )
        sharing_radius = _resolve_config_value(
            optimization_config, "sharing_radius", sharing_radius
        )
        sharing_alpha = _resolve_config_value(optimization_config, "sharing_alpha", sharing_alpha)
        final_selection = _resolve_config_value(
            optimization_config, "final_selection", final_selection
        )
        final_selection_top_k = _resolve_config_value(
            optimization_config, "final_selection_top_k", final_selection_top_k
        )
        final_selection_cv = _resolve_config_value(
            optimization_config, "final_selection_cv", final_selection_cv
        )

        self.evolution_config = evolution_config
        self.population_config = population_config
        self.runtime_config = runtime_config
        self.optimization_config = optimization_config
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
        self.warm_start_configs = legacy_warm_start_configs
        self._warm_start_configs = warm_start_configs
        self.parallel_backend = parallel_backend
        self.population_initializer = population_initializer
        self.random_state = random_state
        self.local_search = local_search
        self.local_search_top_k = local_search_top_k
        self.local_search_steps = local_search_steps
        self.local_search_radius = local_search_radius
        self.diversity_control = diversity_control
        self.diversity_threshold = diversity_threshold
        self.diversity_stagnation_generations = diversity_stagnation_generations
        self.diversity_mutation_boost = diversity_mutation_boost
        self.random_immigrants_fraction = random_immigrants_fraction
        self.adaptive_selection = adaptive_selection
        self.selection_pressure_min = selection_pressure_min
        self.selection_pressure_max = selection_pressure_max
        self.offspring_diversity_retries = offspring_diversity_retries
        self.fitness_sharing = fitness_sharing
        self.sharing_radius = sharing_radius
        self.sharing_alpha = sharing_alpha
        self.final_selection = final_selection
        self.final_selection_top_k = final_selection_top_k
        self.final_selection_cv = final_selection_cv

        _validate_parallel_backend(self.parallel_backend)
        _validate_error_score(self.error_score)
        _validate_population_initializer(self.population_initializer)
        if self.final_selection_top_k < 1:
            raise ValueError("final_selection_top_k must be greater than or equal to 1")
        _validate_optimizer_control(
            self.local_search_top_k,
            self.local_search_steps,
            self.local_search_radius,
            self.diversity_threshold,
            self.diversity_stagnation_generations,
            self.diversity_mutation_boost,
            self.random_immigrants_fraction,
            self.sharing_radius,
            self.sharing_alpha,
            self.selection_pressure_min,
            self.selection_pressure_max,
            self.offspring_diversity_retries,
        )

        # Check that the estimator is compatible with scikit-learn
        if not (
            _is_classifier(self.estimator)
            or _is_regressor(self.estimator)
            or _is_outlier_detector(self.estimator)
        ):
            raise ValueError(
                f"{self.estimator} is not a valid Sklearn classifier, regressor, or outlier detector"
            )

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

        creator.create("FitnessMax", base.Fitness, weights=(self.criteria_sign,))
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

        if len(self.space) == 1 and hasattr(list(self.space.param_grid.values())[0], "lower"):
            sampler = list(self.space.param_grid.values())[0]
            lower, upper = sampler.lower, sampler.upper

            self.toolbox.register(
                "mate_raw", tools.cxSimulatedBinaryBounded, low=lower, up=upper, eta=10
            )
        else:
            self.toolbox.register("mate_raw", tools.cxUniform, indpb=0.5)

        self.toolbox.register("mate", self.mate)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("select", self.select)

        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("evaluate_population", self.evaluate_population)

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
        population = initialize_search_population(self, self.toolbox, creator.Individual)
        for individual in population:
            self._repair_individual(individual)
        return population

    def select(self, population, k):
        if not self.elitism:
            self._selection_pressure_ = None
            return tools.selRoulette(population, k)

        tournament_size = adaptive_tournament_size(
            self,
            getattr(self, "_last_generation_record", None),
            len(population),
        )
        self._selection_pressure_ = tournament_size
        return tools.selTournament(population, k, tournsize=tournament_size)

    def _repair_value(self, dimension, value):
        if isinstance(dimension, Integer):
            if value is None:
                return dimension.sample()

            repaired = int(round(float(value)))
            return int(np.clip(repaired, dimension.lower, dimension.upper))

        if isinstance(dimension, Continuous):
            if value is None:
                return dimension.sample()

            repaired = float(value)
            return float(np.clip(repaired, dimension.lower, dimension.upper))

        if isinstance(dimension, Categorical):
            return value if value in dimension.choices else dimension.sample()

        return value

    def _repair_individual(self, individual):
        if not hasattr(self, "space"):
            return individual

        for index, parameter in enumerate(self.space.parameters):
            individual[index] = self._repair_value(self.space[parameter], individual[index])

        return individual

    def mate(self, individual_1, individual_2):
        offspring_1, offspring_2 = self.toolbox.mate_raw(individual_1, individual_2)
        self._repair_individual(offspring_1)
        self._repair_individual(offspring_2)
        return offspring_1, offspring_2

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
        self._repair_individual(individual)

        return [individual]

    def _individual_key(self, individual):
        current_generation_params = {
            key: individual[n] for n, key in enumerate(self.space.parameters)
        }
        return tuple(sorted(current_generation_params.items()))

    def evaluate_population(self, individuals):
        for individual in individuals:
            self._repair_individual(individual)
        return _evaluate_population_batch(self, individuals, "current_generation_params")

    def _evaluate_individual(self, individual, n_jobs=None):
        self._repair_individual(individual)
        # Dictionary representation of the individual with key-> hyperparameter name, value -> value
        current_generation_params = {
            key: individual[n] for n, key in enumerate(self.space.parameters)
        }

        local_estimator = clone(self.estimator)
        local_estimator.set_params(**current_generation_params)

        # standard cross_validate for all estimator types is used
        cv_results = cross_validate(
            local_estimator,
            self.X_,
            self.y_,
            cv=self._cv_splits,
            scoring=self.scorer_,
            n_jobs=n_jobs,
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

        fitness_result = (score,)

        return fitness_result, current_generation_params, True, False

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

        # Convert hyperparameters to a tuple to use as a key in the cache
        self._repair_individual(individual)
        individual_key = self._individual_key(individual)

        # Check if the individual has already been evaluated
        if individual_key in self.fitness_cache and self.use_cache:
            # Retrieve cached result
            cached_result = self.fitness_cache[individual_key]
            # Ensure the logbook is updated even if the individual is cached
            self.logbook.record(parameters=cached_result["current_generation_params"])
            _record_fit_stats(self, evaluated=1, cache_hits=1)
            return cached_result["fitness"]

        candidate_n_jobs = self.n_jobs if self.parallel_backend == "cv" else 1
        (
            fitness_result,
            current_generation_params,
            used_cv,
            skipped_invalid,
        ) = self._evaluate_individual(
            individual,
            n_jobs=candidate_n_jobs,
        )
        current_generation_params = _logbook_record(
            self.logbook,
            "parameters",
            current_generation_params,
        )

        if self.use_cache:
            # Store the fitness result and the current generation parameters in the cache
            self.fitness_cache[individual_key] = {
                "fitness": fitness_result,
                "current_generation_params": current_generation_params,
            }

        _record_fit_stats(
            self,
            evaluated=1,
            unique=1,
            cv_calls=int(used_cv),
            skipped=int(skipped_invalid),
        )

        return fitness_result

    def _candidate_params_from_index(self, index):
        return self.cv_results_["params"][index]

    def _top_candidate_indices(self):
        ranks = np.asarray(self.cv_results_[f"rank_test_{self.refit_metric}"])
        return list(np.argsort(ranks)[: self.final_selection_top_k])

    def _final_selection_splits(self):
        if self.final_selection_cv is None:
            return self._cv_splits

        cv = check_cv(self.final_selection_cv, self.y_, classifier=_is_classifier(self.estimator))
        return list(cv.split(self.X_, self.y_, groups=getattr(self, "groups_", None)))

    def _score_final_candidate(self, params, cv_splits):
        local_estimator = clone(self.estimator)
        local_estimator.set_params(**params)

        cv_results = cross_validate(
            local_estimator,
            self.X_,
            self.y_,
            cv=cv_splits,
            scoring=self.scorer_,
            n_jobs=self.n_jobs,
            pre_dispatch=self.pre_dispatch,
            error_score=self.error_score,
            return_train_score=False,
        )
        cv_scores = cv_results[f"test_{self.refit_metric}"]
        return float(np.mean(cv_scores)), cv_scores

    def _select_final_candidate(self):
        original_best_index = int(self.cv_results_[f"rank_test_{self.refit_metric}"].argmin())
        original_best_score = float(
            self.cv_results_[f"mean_test_{self.refit_metric}"][original_best_index]
        )
        original_best_params = self._candidate_params_from_index(original_best_index)

        self.final_selection_results_ = {
            "enabled": bool(self.final_selection),
            "top_k": 1,
            "cv": self.final_selection_cv,
            "original_best_index": original_best_index,
            "original_best_score": original_best_score,
            "original_best_params": original_best_params,
            "selected_index": original_best_index,
            "selected_score": original_best_score,
            "selected_params": original_best_params,
            "changed": False,
            "candidates": [],
            "time_seconds": 0.0,
        }

        if not self.final_selection:
            return original_best_index, original_best_score, original_best_params

        started_at = time.time()
        cv_splits = self._final_selection_splits()
        candidate_results = []
        seen_params = set()

        for index in self._top_candidate_indices():
            params = self._candidate_params_from_index(index)
            params_key = tuple(sorted(params.items()))
            if params_key in seen_params:
                continue
            seen_params.add(params_key)

            score, cv_scores = self._score_final_candidate(params, cv_splits)
            candidate_results.append(
                {
                    "index": int(index),
                    "original_score": float(
                        self.cv_results_[f"mean_test_{self.refit_metric}"][index]
                    ),
                    "score": score,
                    "cv_scores": cv_scores.tolist(),
                    "params": params,
                }
            )

        if candidate_results:
            selected = max(candidate_results, key=lambda item: item["score"])
            selected_index = selected["index"]
            selected_score = selected["score"]
            selected_params = selected["params"]
        else:  # pragma: no cover
            selected_index = original_best_index
            selected_score = original_best_score
            selected_params = original_best_params

        self.final_selection_results_.update(
            {
                "top_k": self.final_selection_top_k,
                "selected_index": selected_index,
                "selected_score": selected_score,
                "selected_params": selected_params,
                "changed": selected_index != original_best_index,
                "candidates": candidate_results,
                "time_seconds": time.time() - started_at,
            }
        )

        return selected_index, selected_score, selected_params

    def fit(self, X, y=None, callbacks=None, groups=None):
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
            supervised learning. For outlier detection, y can be None.
        callbacks: list or callable
            One or a list of the callbacks methods available in
            :class:`~sklearn_genetic.callbacks`.
            The callback is evaluated after fitting the estimators from the generation 1.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test sets. Only used in conjunction with a "Group" cv
            instance such as :class:`~sklearn.model_selection.GroupKFold`.
        """

        self.X_ = X
        self.y_ = y
        self.groups_ = groups
        self._n_iterations = self.generations + 1
        self.refit_metric = "score"
        self.multimetric_ = False

        # added a handle outlier detection jussst in case where y might be None
        if _is_outlier_detector(self.estimator) and y is None:
            # and for unsupervised outlier detection, it will create dummy y for cv compatibility :)
            self.y_ = np.zeros(X.shape[0])

        # Make sure the callbacks are valid
        self.callbacks = check_callback(callbacks)

        checkpoint_loaded = False
        restored_logbook = None
        restored_fit_stats = None
        restored_generation_log = None

        # Load state if a checkpoint exists
        for callback in self.callbacks:
            if isinstance(callback, ModelCheckpoint):
                if os.path.exists(callback.checkpoint_path):
                    checkpoint_data = callback.load()
                    if checkpoint_data:
                        self.__dict__.update(checkpoint_data["estimator_state"])  # noqa
                        # Restore the fitness cache so already-evaluated
                        # candidates are reused instead of re-evaluated. Older
                        # checkpoints have no ``runtime_state``, so guard with
                        # .get() for backward compatibility.
                        runtime_state = checkpoint_data.get("runtime_state") or {}
                        cached = runtime_state.get("fitness_cache")
                        if cached is not None:
                            self.fitness_cache = cached
                        restored_fit_stats = runtime_state.get("fit_stats_")
                        # ``checkpoint_data["logbook"]`` is the per-*generation*
                        # summary log (see ModelCheckpoint.on_step), not
                        # ``self.logbook`` -- restoring it onto ``self.logbook``
                        # would silently break ``cv_results_``/``history``.
                        # ``_register()`` below also unconditionally creates a
                        # fresh ``self.logbook``, so stash the real one and put
                        # it back afterwards (see comment near the
                        # ``_register()`` call).
                        restored_logbook = runtime_state.get("candidate_logbook")
                        # The per-generation summary log saved under the
                        # legacy ``"logbook"`` key is exactly what generation
                        # numbering needs to continue from -- see
                        # ``_seed_logbook`` in ``algorithms.py``.
                        restored_generation_log = checkpoint_data.get("logbook")
                        checkpoint_loaded = True
                    break

        # Seed after the checkpoint (if any) is loaded: ``self.__dict__.update()``
        # above may have just replaced ``self.random_state`` with the value from
        # a resumed run's checkpoint, and seeding before that point would use the
        # pre-resume value instead (see #299).
        _seed_global_rngs(self.random_state)

        if not checkpoint_loaded:
            _reset_adapters(self)

        # Preserve cumulative counters across a resume instead of zeroing them,
        # so e.g. ``cache_hits``/``evaluated_candidates`` reflect the whole run.
        self.fit_stats_ = (
            restored_fit_stats if restored_fit_stats is not None else _create_fit_stats()
        )
        self._resume_generation_log = restored_generation_log

        if callable(self.scoring):
            self.scorer_ = self.scoring
            self.metrics_list = [self.refit_metric]
        elif self.scoring is None or isinstance(self.scoring, str):
            # it will handle outlier detectors that don't have a score method
            if _is_outlier_detector(self.estimator) and self.scoring is None:
                # this function creates a default scorer for outlier detection
                def default_outlier_scorer(estimator, X, y=None):
                    if hasattr(estimator, "score_samples"):
                        return np.mean(estimator.score_samples(X))
                    elif hasattr(estimator, "decision_function"):
                        return np.mean(estimator.decision_function(X))
                    else:
                        predictions = estimator.fit_predict(X)
                        return np.mean(predictions == 1)

                self.scorer_ = default_outlier_scorer
                self.metrics_list = [self.refit_metric]
            else:
                self.scorer_ = check_scoring(self.estimator, self.scoring)
                self.metrics_list = [self.refit_metric]
        else:
            self.scorer_ = _check_multimetric_scoring(self.estimator, self.scoring)
            self._check_refit_for_multimetric(self.scorer_)
            self.refit_metric = self.refit
            self.metrics_list = self.scorer_.keys()
            self.multimetric_ = True

        # Check cv and get the n_splits
        if _is_outlier_detector(self.estimator):
            # For outlier detectors, better to use KFold instead of classifier-based CV
            from sklearn.model_selection import KFold

            cv_orig = KFold(n_splits=self.cv if isinstance(self.cv, int) else 5)
            self.n_splits_ = cv_orig.get_n_splits(X, self.y_)
        else:
            cv_orig = check_cv(self.cv, self.y_, classifier=_is_classifier(self.estimator))
            self.n_splits_ = cv_orig.get_n_splits(X, self.y_, groups=self.groups_)
        self._cv_splits = list(cv_orig.split(self.X_, self.y_, groups=self.groups_))

        # Set the DEAPs necessary methods
        self._register()
        # ``_register()`` always creates a fresh, empty ``self.logbook`` (it
        # also builds the toolbox/population/hof/stats, which are ``deap``
        # objects that cannot be checkpointed and are intentionally rebuilt on
        # every ``fit`` call). Put the restored per-candidate logbook back so
        # resumed candidates are not dropped from ``cv_results_``/``history``.
        if restored_logbook is not None:
            self.logbook = restored_logbook

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
            "fitness_best": log.select("fitness_best"),
            "fitness_max": log.select("fitness_max"),
            "fitness_min": log.select("fitness_min"),
            "population_size": log.select("population_size"),
            "unique_individuals": log.select("unique_individuals"),
            "unique_individual_ratio": log.select("unique_individual_ratio"),
            "genotype_diversity": log.select("genotype_diversity"),
            "fitness_improvement": log.select("fitness_improvement"),
            "fitness_improved": log.select("fitness_improved"),
            "stagnation_generations": log.select("stagnation_generations"),
            "best_generation": log.select("best_generation"),
            "mutation_probability": log.select("mutation_probability"),
            "selection_pressure": log.select("selection_pressure"),
            "diversity_control_triggered": log.select("diversity_control_triggered"),
            "random_immigrants": log.select("random_immigrants"),
            "duplicate_replacements": log.select("duplicate_replacements"),
            "local_refinements": log.select("local_refinements"),
            "fitness_sharing_applied": log.select("fitness_sharing_applied"),
            "mean_niche_count": log.select("mean_niche_count"),
            "max_niche_count": log.select("max_niche_count"),
        }

        # Imitate the logic of scikit-learn refit parameter
        if self.refit:
            self.best_index_, self.best_score_, self.best_params_ = self._select_final_candidate()

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


class GAFeatureSelectionCV(GeneticEstimatorMixin, MetaEstimatorMixin, SelectorMixin, BaseEstimator):
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
        Size of the initial population to sample generated individuals.

    evolution_config : :class:`~sklearn_genetic.config.EvolutionConfig`, default=None
        Optional grouped configuration for core genetic algorithm controls such
        as population size, generation count, crossover, mutation, tournament
        size, elitism, hall-of-fame size, criteria, and algorithm.

    population_config : :class:`~sklearn_genetic.config.PopulationConfig`, default=None
        Optional grouped configuration for the initial feature-mask population.

    runtime_config : :class:`~sklearn_genetic.config.RuntimeConfig`, default=None
        Optional grouped configuration for parallelism, caching, train-score
        collection, error handling, and verbose output.

    optimization_config : :class:`~sklearn_genetic.config.OptimizationConfig`, default=None
        Optional grouped configuration for local refinement, diversity control,
        adaptive selection, and fitness sharing. Final-selection fields are
        ignored by :class:`~sklearn_genetic.GAFeatureSelectionCV`.

    population_initializer : {'smart', 'random'}, default='smart'
        Strategy used to generate the initial population. ``'smart'`` creates
        duplicate-aware feature masks with a spread of selected-feature counts.
        ``'random'`` uses the previous weighted random feature-mask sampling.

    local_search : bool, default=False
        If ``True``, run a short local refinement phase around the current
        hall-of-fame feature masks after the genetic search finishes.

    local_search_top_k : int, default=1
        Number of hall-of-fame feature masks used as local-search seeds.

    local_search_steps : int, default=1
        Number of neighbor feature masks generated per local-search seed.

    local_search_radius : float, default=0.1
        Fraction of features to flip when sampling a local neighbor.

    diversity_control : bool, default=True
        If ``True``, monitor diversity and stagnation to boost mutation,
        replace duplicate candidates, and inject random immigrants.

    diversity_threshold : float, default=0.25
        Diversity value below which diversity control can trigger.

    diversity_stagnation_generations : int, default=5
        Number of stagnant generations after which diversity control can
        inject random immigrants.

    diversity_mutation_boost : float, default=2.0
        Multiplicative boost applied to mutation probability when diversity
        control triggers. The boosted value is capped to DEAP's valid range.

    random_immigrants_fraction : float, default=0.1
        Fraction of offspring replaced by random immigrants when diversity
        control triggers.

    adaptive_selection : bool, default=False
        If ``True``, adapt tournament size from generation telemetry. Selection
        pressure is reduced when diversity is low or the search is stagnant,
        and slightly increased when the population is improving with enough
        diversity.

    selection_pressure_min : int, default=2
        Minimum tournament size used by adaptive selection.

    selection_pressure_max : int, default=None
        Maximum tournament size used by adaptive selection. If ``None``, the
        maximum is one larger than ``tournament_size``.

    offspring_diversity_retries : int, default=0
        Number of retries used when replacing duplicate or parent-matching
        offspring with new random feature masks.

    fitness_sharing : bool, default=False
        If ``True``, temporarily penalize candidates in crowded niches during
        selection. Raw cross-validation scores and ``cv_results_`` are not
        modified.

    sharing_radius : float, default=0.2
        Normalized distance below which two individuals are considered part of
        the same niche for fitness sharing.

    sharing_alpha : float, default=1.0
        Shape parameter that controls how quickly sharing pressure decreases
        with distance inside ``sharing_radius``.

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
        Number of jobs to run in parallel. Candidate evaluations in each
        generation are parallelized when possible; each candidate then runs
        cross-validation sequentially to avoid nested parallelism.
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
        than CPUs can process. This parameter can be ``None`` to dispatch all
        jobs immediately, an integer number of total jobs to spawn, or a string
        expression as a function of ``n_jobs``, such as ``'2*n_jobs'``.

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
        Dictionary with one list per generation. It includes ``gen``,
        ``fitness``, ``fitness_std``, ``fitness_best``, ``fitness_max``, ``fitness_min``,
        population diversity fields, stagnation fields, optimizer-control
        telemetry, and local-refinement telemetry.

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
    fit_stats_ : dict
        Counters collected during the last ``fit`` call. Includes evaluated
        candidates, unique candidates, cross-validation calls, cache hits,
        duplicate candidates, skipped invalid candidates, and population-level
        parallel/serial batch counts.
    """

    def __init__(
        self,
        estimator,
        cv=3,
        scoring=None,
        population_size=50,
        generations=80,
        crossover_probability=0.8,
        mutation_probability=0.1,
        tournament_size=3,
        elitism=True,
        max_features=None,
        verbose=True,
        keep_top_k=1,
        criteria="max",
        algorithm="eaMuPlusLambda",
        refit=True,
        n_jobs=None,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
        log_config=None,
        use_cache=True,
        evolution_config=None,
        population_config=None,
        runtime_config=None,
        optimization_config=None,
        parallel_backend="auto",
        population_initializer="smart",
        random_state=None,
        local_search=False,
        local_search_top_k=1,
        local_search_steps=1,
        local_search_radius=0.1,
        diversity_control=True,
        diversity_threshold=0.25,
        diversity_stagnation_generations=5,
        diversity_mutation_boost=2.0,
        random_immigrants_fraction=0.1,
        adaptive_selection=False,
        selection_pressure_min=2,
        selection_pressure_max=None,
        offspring_diversity_retries=0,
        fitness_sharing=False,
        sharing_radius=0.2,
        sharing_alpha=1.0,
    ):
        population_size = _resolve_config_value(
            evolution_config, "population_size", population_size
        )
        generations = _resolve_config_value(evolution_config, "generations", generations)
        crossover_probability = _resolve_config_value(
            evolution_config, "crossover_probability", crossover_probability
        )
        mutation_probability = _resolve_config_value(
            evolution_config, "mutation_probability", mutation_probability
        )
        tournament_size = _resolve_config_value(
            evolution_config, "tournament_size", tournament_size
        )
        elitism = _resolve_config_value(evolution_config, "elitism", elitism)
        keep_top_k = _resolve_config_value(evolution_config, "keep_top_k", keep_top_k)
        criteria = _resolve_config_value(evolution_config, "criteria", criteria)
        algorithm = _resolve_config_value(evolution_config, "algorithm", algorithm)

        population_initializer = _resolve_config_value(
            population_config, "initializer", population_initializer
        )

        n_jobs = _resolve_config_value(runtime_config, "n_jobs", n_jobs)
        pre_dispatch = _resolve_config_value(runtime_config, "pre_dispatch", pre_dispatch)
        error_score = _resolve_config_value(runtime_config, "error_score", error_score)
        return_train_score = _resolve_config_value(
            runtime_config, "return_train_score", return_train_score
        )
        use_cache = _resolve_config_value(runtime_config, "use_cache", use_cache)
        parallel_backend = _resolve_config_value(
            runtime_config, "parallel_backend", parallel_backend
        )
        verbose = _resolve_config_value(runtime_config, "verbose", verbose)

        local_search = _resolve_config_value(optimization_config, "local_search", local_search)
        local_search_top_k = _resolve_config_value(
            optimization_config, "local_search_top_k", local_search_top_k
        )
        local_search_steps = _resolve_config_value(
            optimization_config, "local_search_steps", local_search_steps
        )
        local_search_radius = _resolve_config_value(
            optimization_config, "local_search_radius", local_search_radius
        )
        diversity_control = _resolve_config_value(
            optimization_config, "diversity_control", diversity_control
        )
        diversity_threshold = _resolve_config_value(
            optimization_config, "diversity_threshold", diversity_threshold
        )
        diversity_stagnation_generations = _resolve_config_value(
            optimization_config,
            "diversity_stagnation_generations",
            diversity_stagnation_generations,
        )
        diversity_mutation_boost = _resolve_config_value(
            optimization_config, "diversity_mutation_boost", diversity_mutation_boost
        )
        random_immigrants_fraction = _resolve_config_value(
            optimization_config, "random_immigrants_fraction", random_immigrants_fraction
        )
        adaptive_selection = _resolve_config_value(
            optimization_config, "adaptive_selection", adaptive_selection
        )
        selection_pressure_min = _resolve_config_value(
            optimization_config, "selection_pressure_min", selection_pressure_min
        )
        selection_pressure_max = _resolve_config_value(
            optimization_config, "selection_pressure_max", selection_pressure_max
        )
        offspring_diversity_retries = _resolve_config_value(
            optimization_config, "offspring_diversity_retries", offspring_diversity_retries
        )
        fitness_sharing = _resolve_config_value(
            optimization_config, "fitness_sharing", fitness_sharing
        )
        sharing_radius = _resolve_config_value(
            optimization_config, "sharing_radius", sharing_radius
        )
        sharing_alpha = _resolve_config_value(optimization_config, "sharing_alpha", sharing_alpha)

        self.evolution_config = evolution_config
        self.population_config = population_config
        self.runtime_config = runtime_config
        self.optimization_config = optimization_config
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
        self.parallel_backend = parallel_backend
        self.population_initializer = population_initializer
        self.random_state = random_state
        self.local_search = local_search
        self.local_search_top_k = local_search_top_k
        self.local_search_steps = local_search_steps
        self.local_search_radius = local_search_radius
        self.diversity_control = diversity_control
        self.diversity_threshold = diversity_threshold
        self.diversity_stagnation_generations = diversity_stagnation_generations
        self.diversity_mutation_boost = diversity_mutation_boost
        self.random_immigrants_fraction = random_immigrants_fraction
        self.adaptive_selection = adaptive_selection
        self.selection_pressure_min = selection_pressure_min
        self.selection_pressure_max = selection_pressure_max
        self.offspring_diversity_retries = offspring_diversity_retries
        self.fitness_sharing = fitness_sharing
        self.sharing_radius = sharing_radius
        self.sharing_alpha = sharing_alpha

        _validate_parallel_backend(self.parallel_backend)
        _validate_error_score(self.error_score)
        _validate_population_initializer(self.population_initializer)
        _validate_optimizer_control(
            self.local_search_top_k,
            self.local_search_steps,
            self.local_search_radius,
            self.diversity_threshold,
            self.diversity_stagnation_generations,
            self.diversity_mutation_boost,
            self.random_immigrants_fraction,
            self.sharing_radius,
            self.sharing_alpha,
            self.selection_pressure_min,
            self.selection_pressure_max,
            self.offspring_diversity_retries,
        )

        # added new check for whether the estimator is compatible with scikit-learn
        if not (
            _is_classifier(self.estimator)
            or _is_regressor(self.estimator)
            or _is_outlier_detector(self.estimator)
        ):
            raise ValueError(
                f"{self.estimator} is not a valid Sklearn classifier, regressor, or outlier detector"
            )

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
            "individual_raw",
            weighted_bool_individual,
            creator.Individual,
            weight=self.features_proportion,
            size=self.n_features,
        )
        self.toolbox.register("individual", self._new_feature_individual)

        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("mate_raw", cxUniform, indpb=self.crossover_adapter.current_value)
        self.toolbox.register("mutate_raw", mutFlipBit, indpb=self.mutation_adapter.current_value)
        self.toolbox.register("mate", self.mate)
        self.toolbox.register("mutate", self.mutate)

        self.toolbox.register("select", self.select)

        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("evaluate_population", self.evaluate_population)

        self._pop = self._initialize_population()
        self._hof = tools.HallOfFame(self.keep_top_k)

        # Stats among axis 0 to get two values:
        # One based on the score and the other in the number of features
        self._stats = tools.Statistics(ind_fitness_values)
        self._stats.register("fitness", np.mean, axis=0)
        self._stats.register("fitness_std", np.std, axis=0)
        self._stats.register("fitness_max", np.max, axis=0)
        self._stats.register("fitness_min", np.min, axis=0)

        self.logbook = tools.Logbook()

    def _initialize_population(self):
        population = initialize_feature_population(self, self.toolbox, creator.Individual)
        for individual in population:
            self._repair_individual(individual)
        return population

    def select(self, population, k):
        if not self.elitism:
            self._selection_pressure_ = None
            return tools.selRoulette(population, k)

        tournament_size = adaptive_tournament_size(
            self,
            getattr(self, "_last_generation_record", None),
            len(population),
        )
        self._selection_pressure_ = tournament_size
        return tools.selTournament(population, k, tournsize=tournament_size)

    def _repair_individual(self, individual):
        for index, value in enumerate(individual):
            individual[index] = 1 if value else 0

        max_features = getattr(self, "max_features", None)

        if max_features and sum(individual) > max_features:
            selected = [index for index, value in enumerate(individual) if value]
            random.shuffle(selected)
            for index in selected[max_features:]:
                individual[index] = 0

        if sum(individual) == 0:
            individual[random.randrange(0, len(individual))] = 1

        return individual

    def _new_feature_individual(self):
        return self._repair_individual(self.toolbox.individual_raw())

    def mate(self, individual_1, individual_2):
        offspring_1, offspring_2 = self.toolbox.mate_raw(individual_1, individual_2)
        self._repair_individual(offspring_1)
        self._repair_individual(offspring_2)
        return offspring_1, offspring_2

    def mutate(self, individual):
        (mutated,) = self.toolbox.mutate_raw(individual)
        self._repair_individual(mutated)
        return (mutated,)

    def _individual_key(self, individual):
        return tuple(individual)

    def evaluate_population(self, individuals):
        for individual in individuals:
            self._repair_individual(individual)
        return _evaluate_population_batch(self, individuals, "current_generation_features")

    def _build_feature_evaluation_record(self, current_generation_params, cv_results):
        cv_scores = cv_results[f"test_{self.refit_metric}"]
        score = np.mean(cv_scores)

        current_generation_params["score"] = score
        current_generation_params["cv_scores"] = cv_scores
        current_generation_params["fit_time"] = cv_results["fit_time"]
        current_generation_params["score_time"] = cv_results["score_time"]

        for metric in self.metrics_list:
            current_generation_params[f"test_{metric}"] = cv_results[f"test_{metric}"]

            if self.return_train_score:
                current_generation_params[f"train_{metric}"] = cv_results[f"train_{metric}"]

        return score, current_generation_params

    def _penalized_feature_cv_results(self, score):
        cv_results = {
            "fit_time": np.zeros(self.n_splits_),
            "score_time": np.zeros(self.n_splits_),
        }

        for metric in self.metrics_list:
            cv_results[f"test_{metric}"] = np.full(self.n_splits_, score)

            if self.return_train_score:
                cv_results[f"train_{metric}"] = np.full(self.n_splits_, score)

        return cv_results

    def _evaluate_individual(self, individual, n_jobs=None):
        self._repair_individual(individual)
        bool_individual = np.array(individual, dtype=bool)

        current_generation_params = {"features": bool_individual}

        n_selected_features = np.sum(individual)

        max_features = getattr(self, "max_features", None)

        if max_features and (n_selected_features > max_features or n_selected_features == 0):
            score = -self.criteria_sign * 100000
            cv_results = self._penalized_feature_cv_results(score)
            _, current_generation_params = self._build_feature_evaluation_record(
                current_generation_params, cv_results
            )

            fitness_result = [score, n_selected_features]

            return fitness_result, current_generation_params, False, True

        local_estimator = clone(self.estimator)

        # Use standard cross_validate for all estimator types
        cv_results = cross_validate(
            local_estimator,
            self.X_[:, bool_individual],
            self.y_,
            cv=self._cv_splits,
            scoring=self.scorer_,
            n_jobs=n_jobs,
            pre_dispatch=self.pre_dispatch,
            error_score=self.error_score,
            return_train_score=self.return_train_score,
        )

        score, current_generation_params = self._build_feature_evaluation_record(
            current_generation_params, cv_results
        )

        # Uses the log config to save in remote log server (e.g MLflow)
        if self.log_config is not None:
            self.log_config.create_run(
                parameters=current_generation_params,
                score=score,
                estimator=local_estimator,
            )

        fitness_result = [score, n_selected_features]

        return fitness_result, current_generation_params, True, False

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

        # Convert the individual to a tuple to use as a key in the cache
        self._repair_individual(individual)
        individual_key = self._individual_key(individual)

        # Check if the individual has already been evaluated
        if individual_key in self.fitness_cache and self.use_cache:
            cached_result = self.fitness_cache[individual_key]
            # Ensure the logbook is updated even if the individual is cached
            self.logbook.record(parameters=cached_result["current_generation_features"])
            _record_fit_stats(self, evaluated=1, cache_hits=1)
            return cached_result["fitness"]

        candidate_n_jobs = self.n_jobs if self.parallel_backend == "cv" else 1
        (
            fitness_result,
            current_generation_params,
            used_cv,
            skipped_invalid,
        ) = self._evaluate_individual(
            individual,
            n_jobs=candidate_n_jobs,
        )
        current_generation_params = _logbook_record(
            self.logbook,
            "parameters",
            current_generation_params,
        )

        if self.use_cache:
            # Store the fitness result and the current generation features in the cache
            self.fitness_cache[individual_key] = {
                "fitness": fitness_result,
                "current_generation_features": current_generation_params,
            }

        _record_fit_stats(
            self,
            evaluated=1,
            unique=1,
            cv_calls=int(used_cv),
            skipped=int(skipped_invalid),
        )

        return fitness_result

    def fit(self, X, y=None, callbacks=None, groups=None):
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
            supervised learning. For outlier detection, y can be None.
        callbacks: list or callable
            One or a list of the callbacks methods available in
            :class:`~sklearn_genetic.callbacks`.
            The callback is evaluated after fitting the estimators from the generation 1.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test sets. Only used in conjunction with a "Group" cv
            instance such as :class:`~sklearn.model_selection.GroupKFold`.
        """

        self.X_, self.y_ = check_X_y(X, y, accept_sparse=True) if y is not None else (X, None)
        self.groups_ = groups

        # Handle outlier detection case if y is none
        if _is_outlier_detector(self.estimator) and y is None:
            self.X_ = X
            self.y_ = np.zeros(X.shape[0])

        self.n_features = X.shape[1]
        self._n_iterations = self.generations + 1
        self.refit_metric = "score"
        self.multimetric_ = False

        self.features_proportion = None
        max_features = getattr(self, "max_features", None)
        if max_features:
            self.features_proportion = max_features / self.n_features

        # Make sure the callbacks are valid
        self.callbacks = check_callback(callbacks)

        checkpoint_loaded = False
        restored_logbook = None
        restored_fit_stats = None
        restored_generation_log = None

        # Load state if a checkpoint exists
        for callback in self.callbacks:
            if isinstance(callback, ModelCheckpoint):
                if os.path.exists(callback.checkpoint_path):
                    checkpoint_data = callback.load()
                    if checkpoint_data:
                        self.__dict__.update(checkpoint_data["estimator_state"])  # noqa
                        # Restore the fitness cache so already-evaluated
                        # candidates are reused instead of re-evaluated. Older
                        # checkpoints have no ``runtime_state``, so guard with
                        # .get() for backward compatibility.
                        runtime_state = checkpoint_data.get("runtime_state") or {}
                        cached = runtime_state.get("fitness_cache")
                        if cached is not None:
                            self.fitness_cache = cached
                        restored_fit_stats = runtime_state.get("fit_stats_")
                        # ``checkpoint_data["logbook"]`` is the per-*generation*
                        # summary log (see ModelCheckpoint.on_step), not
                        # ``self.logbook`` -- restoring it onto ``self.logbook``
                        # would silently break ``cv_results_``/``history``.
                        # ``_register()`` below also unconditionally creates a
                        # fresh ``self.logbook``, so stash the real one and put
                        # it back afterwards (see comment near the
                        # ``_register()`` call).
                        restored_logbook = runtime_state.get("candidate_logbook")
                        # The per-generation summary log saved under the
                        # legacy ``"logbook"`` key is exactly what generation
                        # numbering needs to continue from -- see
                        # ``_seed_logbook`` in ``algorithms.py``.
                        restored_generation_log = checkpoint_data.get("logbook")
                        checkpoint_loaded = True
                    break

        # Seed after the checkpoint (if any) is loaded: ``self.__dict__.update()``
        # above may have just replaced ``self.random_state`` with the value from
        # a resumed run's checkpoint, and seeding before that point would use the
        # pre-resume value instead (see #299).
        _seed_global_rngs(self.random_state)

        if not checkpoint_loaded:
            _reset_adapters(self)

        # Preserve cumulative counters across a resume instead of zeroing them,
        # so e.g. ``cache_hits``/``evaluated_candidates`` reflect the whole run.
        self.fit_stats_ = (
            restored_fit_stats if restored_fit_stats is not None else _create_fit_stats()
        )
        self._resume_generation_log = restored_generation_log

        if callable(self.scoring):
            self.scorer_ = self.scoring
            self.metrics_list = [self.refit_metric]
        elif self.scoring is None or isinstance(self.scoring, str):
            # Handle outlier detectors that don't have a score method
            if _is_outlier_detector(self.estimator) and self.scoring is None:
                # this function creates a default scorer for outlier detection
                def default_outlier_scorer(estimator, X, y=None):
                    if hasattr(estimator, "score_samples"):
                        return np.mean(estimator.score_samples(X))
                    elif hasattr(estimator, "decision_function"):
                        return np.mean(estimator.decision_function(X))
                    else:
                        predictions = estimator.fit_predict(X)
                        return np.mean(predictions == 1)

                self.scorer_ = default_outlier_scorer
                self.metrics_list = [self.refit_metric]
            else:
                self.scorer_ = check_scoring(self.estimator, self.scoring)
                self.metrics_list = [self.refit_metric]
        else:
            self.scorer_ = _check_multimetric_scoring(self.estimator, self.scoring)
            self._check_refit_for_multimetric(self.scorer_)
            self.refit_metric = self.refit
            self.metrics_list = self.scorer_.keys()
            self.multimetric_ = True

        # Check cv and get the n_splits
        if _is_outlier_detector(self.estimator):
            from sklearn.model_selection import KFold

            cv_orig = KFold(n_splits=self.cv if isinstance(self.cv, int) else 5)
            self.n_splits_ = cv_orig.get_n_splits(X, self.y_)
        else:
            cv_orig = check_cv(self.cv, self.y_, classifier=_is_classifier(self.estimator))
            self.n_splits_ = cv_orig.get_n_splits(X, self.y_, groups=self.groups_)
        self._cv_splits = list(cv_orig.split(self.X_, self.y_, groups=self.groups_))

        # Set the DEAPs necessary methods
        self._register()
        # ``_register()`` always creates a fresh, empty ``self.logbook`` (it
        # also builds the toolbox/population/hof/stats, which are ``deap``
        # objects that cannot be checkpointed and are intentionally rebuilt on
        # every ``fit`` call). Put the restored per-candidate logbook back so
        # resumed candidates are not dropped from ``cv_results_``/``history``.
        if restored_logbook is not None:
            self.logbook = restored_logbook

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
            "fitness_best": log.select("fitness_best"),
            "fitness_max": log.select("fitness_max"),
            "fitness_min": log.select("fitness_min"),
            "population_size": log.select("population_size"),
            "unique_individuals": log.select("unique_individuals"),
            "unique_individual_ratio": log.select("unique_individual_ratio"),
            "genotype_diversity": log.select("genotype_diversity"),
            "fitness_improvement": log.select("fitness_improvement"),
            "fitness_improved": log.select("fitness_improved"),
            "stagnation_generations": log.select("stagnation_generations"),
            "best_generation": log.select("best_generation"),
            "mutation_probability": log.select("mutation_probability"),
            "selection_pressure": log.select("selection_pressure"),
            "diversity_control_triggered": log.select("diversity_control_triggered"),
            "random_immigrants": log.select("random_immigrants"),
            "duplicate_replacements": log.select("duplicate_replacements"),
            "local_refinements": log.select("local_refinements"),
            "fitness_sharing_applied": log.select("fitness_sharing_applied"),
            "mean_niche_count": log.select("mean_niche_count"),
            "max_niche_count": log.select("max_niche_count"),
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
