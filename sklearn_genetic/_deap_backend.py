"""DEAP-specific lifecycle and legacy result views."""

from copy import deepcopy
import copyreg
from functools import partial
from itertools import repeat
from operator import is_

from deap import base, tools

from ._telemetry import (
    POPULATION_FITNESS_FIELDS,
    _population_diversity,
    feature_population_diversity,
    population_fitness_stats,
)
from .optimizer_control import adaptive_tournament_size
from .utils._score_statistics import is_finite_score
from .utils.random import weighted_bool_individual
from .utils.tools import cxUniform, mutFlipBit
from .callbacks.early_stoppers import CVBudgetStopping, TimerStopping, _NATIVE_CV_BUDGET_START
from .evaluation import evaluate_population, evaluation_context, restore_candidate_positions


def _restore_search_elites(hall_of_fame, elites, individual_type):
    # DEAP inserts equal-fitness entries ahead of existing ties.
    for genes, fitness in reversed(elites[: hall_of_fame.maxsize]):
        individual = individual_type(genes)
        individual.fitness.values = fitness
        hall_of_fame.update([individual])


class _SearchStatistics(tools.Statistics):
    def __init__(self):
        super().__init__()
        self.fields.extend((*POPULATION_FITNESS_FIELDS, "finite_fitness_count"))

    def compile(self, population):
        scores = [member.fitness.values for member in population]
        return {**population_fitness_stats(scores), **super().compile(scores)}


def run_deap(estimator, restored_logbook, restored_elites, checkpoint_loaded, *, data_hash):
    for key in ("_candidate_rows", "_feature_elites"):
        vars(estimator).pop(key, None)
    estimator._register()
    estimator._deap_checkpoint_data_hash = data_hash
    try:
        _restore_search_elites(estimator._hof, restored_elites, estimator.toolbox._individual_type)
        if restored_logbook is not None:
            estimator.logbook = restored_logbook
            restore_candidate_positions(estimator, restored_logbook.chapters["parameters"])
        population = estimator._pop
        warm_masks = getattr(
            getattr(estimator, "population_config", None), "warm_start_masks", None
        )
        if (
            hasattr(estimator, "n_features")
            and not checkpoint_loaded
            and estimator.population_initializer == "smart"
            and (warm_masks is None or len(warm_masks) == 0)
            and any(isinstance(callback, TimerStopping) for callback in estimator.callbacks)
        ):
            densest = max(range(len(population)), key=lambda index: sum(population[index]))
            population.insert(0, population.pop(densest))
        _, log, count = estimator._select_algorithm(population, estimator._stats, estimator._hof)
        return log, count
    finally:
        del estimator._deap_checkpoint_data_hash


def legacy_logbook(records):
    book = tools.Logbook()
    book.chapters["parameters"].extend(records)
    book.extend({} for _ in records)
    return book


# Feature fitness breaks ties in favor of smaller subsets.
class _MaxFitness(base.Fitness):
    weights = (1.0, -1.0)


class _MinFitness(base.Fitness):
    weights = (-1.0, -1.0)


class _MaxParameterFitness(base.Fitness):
    weights = (1.0,)


class _MinParameterFitness(base.Fitness):
    weights = (-1.0,)


def _init_individual(self, values=()):
    self.fitness = type(self).fitness()
    list.__init__(self, values)


class _MaxParameterIndividual(list):
    __init__ = _init_individual
    fitness = _MaxParameterFitness


class _MinParameterIndividual(list):
    __init__ = _init_individual
    fitness = _MinParameterFitness


class _MaxFeatureIndividual(list):
    __init__ = _init_individual
    fitness = _MaxFitness


class _MinFeatureIndividual(list):
    __init__ = _init_individual
    fitness = _MinFitness


_FEATURE_CLASS_ATTRIBUTES = {
    cls: frozenset(vars(cls)) for cls in (_MaxFeatureIndividual, _MinFeatureIndividual)
}


class _ArchivedMask(list):
    pass


def legacy_feature_archive(elites, size, sign):
    archive = tools.HallOfFame(size)
    for genes, scores in reversed(elites):
        mask = _ArchivedMask(genes)
        mask.fitness = (_MaxFitness if sign > 0 else _MinFitness)(scores)
        archive.update([mask])
    return archive


def _clone_feature_mask(x, memo=None, *, mask_type, native_attributes):
    """Copy immutable integer genes in bulk and independently clone DEAP state."""
    if (
        memo is not None
        or type(x) is not mask_type
        or vars(x).keys() != {"fitness"}
        or type(x.fitness) is not mask_type.fitness
        or mask_type in copyreg.dispatch_table
        or mask_type.__bases__ != (list,)
        # Pickle may cache an empty slot-name list on this list subclass.
        or vars(mask_type).get("__slotnames__")
        or vars(mask_type).keys() - {"__slotnames__"} != native_attributes
        # Stream exact type checks without resuming a Python generator for every column.
        or not all(map(is_, map(type, x), repeat(int)))
    ):
        return deepcopy(x, memo)

    child = list.__new__(mask_type)
    child.fitness = deepcopy(x.fitness, {id(x): child})
    child.extend(x)
    return child


def register_parameters(self):
    self.toolbox = base.Toolbox()
    individual_type = _MaxParameterIndividual if self.criteria_sign > 0 else _MinParameterIndividual

    attributes = []
    # Assign all the parameters defined in the param_grid
    # It uses the distribution parameter to set the sampling function
    for parameter, dimension in self.space.param_grid.items():
        self.toolbox.register(f"{parameter}", dimension.sample)
        attributes.append(getattr(self.toolbox, parameter))

    self.toolbox.register(
        "individual",
        tools.initCycle,
        individual_type,
        tuple(attributes),
        n=1,
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

    self.toolbox._individual_type = individual_type
    self._pop = self._initialize_population()
    self._hof = tools.HallOfFame(self.keep_top_k, similar=self._same_individual)

    self._stats = _SearchStatistics()
    if self.space._keyed_categories:
        self._stats.population_diversity = partial(_population_diversity, space=self.space)

    self.logbook = tools.Logbook()


def register_features(self):
    self.toolbox = base.Toolbox()
    individual_type = _MaxFeatureIndividual if self.criteria_sign > 0 else _MinFeatureIndividual
    self.toolbox._individual_type = individual_type

    self.toolbox.register(
        "clone",
        _clone_feature_mask,
        mask_type=individual_type,
        native_attributes=_FEATURE_CLASS_ATTRIBUTES[individual_type],
    )

    # Register the array to choose the features
    # Each binary value represents if the feature is selected or not

    self.toolbox.register(
        "individual_raw",
        weighted_bool_individual,
        individual_type,
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

    self._stats = _SearchStatistics()
    self._stats.population_diversity = feature_population_diversity

    self.logbook = tools.Logbook()


def select_tournament(self, population, k):
    tournament_size = adaptive_tournament_size(
        self,
        getattr(self, "_last_generation_record", None),
        len(population),
    )
    self._selection_pressure_ = tournament_size
    return tools.selTournament(population, k, tournsize=tournament_size)


def initialize_population(estimator, initializer):
    from .genetic_search import _uses_native_feature_initialization

    toolbox = estimator.toolbox
    population = initializer(estimator, toolbox, toolbox._individual_type)
    if not _uses_native_feature_initialization(estimator, initializer):
        for individual in population:
            estimator._repair_individual(individual)
    elif all(
        type(callback) is CVBudgetStopping
        and getattr(callback.on_start, "__func__", None) is _NATIVE_CV_BUDGET_START
        for callback in getattr(estimator, "callbacks", ())
    ):
        # These callbacks only record the evaluation count before the first batch.
        # Other callbacks can edit the population before it is scored.
        toolbox.register(
            "_evaluate_initial_masks",
            evaluate_population,
            estimator,
            cache_record_key="current_generation_features",
        )
    return population


def select_algorithm(self, pop, stats, hof):
    from .algorithms import algorithms_factory
    from .parameters import Algorithms

    fit_stats = self.fit_stats_
    stats.register("cross_validate_calls", lambda _: fit_stats["cross_validate_calls"])
    stats.register(
        "fitness_best_observed",
        _best_observed_stat(self.logbook.chapters["parameters"], self.criteria == "min"),
    )
    selected_algorithm = algorithms_factory.get(self.algorithm, None)
    if selected_algorithm:
        population_size = int(self.population_size)
        with evaluation_context(self):
            pop, log, gen = selected_algorithm(
                pop,
                self.toolbox,
                mu=population_size,
                lambda_=2 * population_size,
                cxpb=self.crossover_adapter,
                stats=stats,
                mutpb=self.mutation_adapter,
                ngen=int(self.generations),
                halloffame=hof,
                callbacks=self.callbacks,
                verbose=self.verbose,
                estimator=self,
                resume_log=getattr(self, "_resume_generation_log", None),
            )
    else:
        raise ValueError(
            f"The algorithm {self.algorithm} is not supported, "
            f"please select one from {Algorithms.list()}"
        )

    return pop, log, gen


def _best_observed_stat(records, minimize):
    # Stats.compile runs at each generation boundary and after final refinement.
    # Candidate records also include discarded offspring and repeated evaluations.
    position = 0
    best = None

    def best_observed(_population):
        nonlocal position, best
        for index in range(position, len(records)):
            score = records[index]["score"]
            if is_finite_score(score) and (
                best is None or (score < best if minimize else score > best)
            ):
                best = score
        position = len(records)
        return float("nan") if best is None else best

    return best_observed
