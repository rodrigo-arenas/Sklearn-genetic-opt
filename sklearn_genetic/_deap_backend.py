"""DEAP-specific lifecycle and legacy result views."""

from copy import deepcopy
import copyreg
from math import isfinite
from itertools import repeat
from operator import is_

import numpy as np
from deap import base, creator, tools

from ._telemetry import feature_population_diversity
from .optimizer_control import adaptive_tournament_size
from .utils.random import weighted_bool_individual
from .utils.tools import cxUniform, mutFlipBit
from .callbacks.early_stoppers import TimerStopping
from .evaluation import evaluation_timers


def _restore_search_elites(hall_of_fame, elites):
    # DEAP inserts equal-fitness entries ahead of existing ties.
    for genes, fitness in reversed(elites[: hall_of_fame.maxsize]):
        individual = creator.Individual(genes)
        individual.fitness.values = fitness
        hall_of_fame.update([individual])


def run_deap(estimator, restored_logbook, restored_elites, checkpoint_loaded):
    for key in ("_candidate_rows", "_feature_elites"):
        vars(estimator).pop(key, None)
    estimator._register()
    try:
        _restore_search_elites(estimator._hof, restored_elites)
        if restored_logbook is not None:
            estimator.logbook = restored_logbook
        population = estimator._pop
        if (
            hasattr(estimator, "n_features")
            and not checkpoint_loaded
            and estimator.population_initializer == "smart"
            and any(isinstance(callback, TimerStopping) for callback in estimator.callbacks)
        ):
            densest = max(range(len(population)), key=lambda index: sum(population[index]))
            population.insert(0, population.pop(densest))
        _, log, count = estimator._select_algorithm(population, estimator._stats, estimator._hof)
        return log, count
    finally:
        del creator.FitnessMax
        del creator.Individual


def legacy_logbook(records):
    book = tools.Logbook()
    book.chapters["parameters"].extend(records)
    book.extend({} for _ in records)
    return book


class _MaxFitness(base.Fitness):
    weights = (1.0, -1.0)


class _MinFitness(base.Fitness):
    weights = (-1.0, -1.0)


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

    creator.create("FitnessMax", base.Fitness, weights=(self.criteria_sign,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    attributes = []
    # Assign all the parameters defined in the param_grid
    # It uses the distribution parameter to set the sampling function
    for parameter, dimension in self.space.param_grid.items():
        self.toolbox.register(f"{parameter}", dimension.sample)
        attributes.append(getattr(self.toolbox, parameter))

    self.toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
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

    self._pop = self._initialize_population()
    self._hof = tools.HallOfFame(self.keep_top_k, similar=self._same_individual)

    self._stats = tools.Statistics(lambda ind: ind.fitness.values)
    self._stats.register("fitness", np.mean, axis=0)
    self._stats.register("fitness_std", np.std, axis=0)
    self._stats.register("fitness_max", np.max, axis=0)
    self._stats.register("fitness_min", np.min, axis=0)

    self.logbook = tools.Logbook()


def register_features(self):
    self.toolbox = base.Toolbox()

    # Criteria sign to set max or min problem
    # And -1.0 as second weight to minimize number of features
    creator.create("FitnessMax", base.Fitness, weights=[self.criteria_sign, -1.0])
    creator.create("Individual", list, fitness=creator.FitnessMax)

    self.toolbox.register(
        "clone",
        _clone_feature_mask,
        mask_type=creator.Individual,
        native_attributes=frozenset(vars(creator.Individual)),
    )

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
    self._stats = tools.Statistics(lambda ind: ind.fitness.values)
    self._stats.population_diversity = feature_population_diversity
    self._stats.register("fitness", np.mean, axis=0)
    self._stats.register("fitness_std", np.std, axis=0)
    self._stats.register("fitness_max", np.max, axis=0)
    self._stats.register("fitness_min", np.min, axis=0)

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
    population = initializer(estimator, estimator.toolbox, creator.Individual)
    for individual in population:
        estimator._repair_individual(individual)
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
        with evaluation_timers(self):
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
            if isfinite(score) and (best is None or (score < best if minimize else score > best)):
                best = score
        position = len(records)
        return float("nan") if best is None else best

    return best_observed
