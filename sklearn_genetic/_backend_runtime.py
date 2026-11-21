"""Evaluate and record optimizer proposals using the shared estimator CV pipeline."""

from dataclasses import dataclass
from itertools import islice

import numpy as np

from .backends import OptimizationProblem
from ._telemetry import (
    _new_telemetry_state,
    _population_diversity,
    _print_verbose_record,
    complete_generation,
    feature_population_diversity,
)
from .population import initialize_feature_population, initialize_search_population
from .space import Space
from .utils.random import weighted_bool_individual


class _Sampler:
    def __init__(self, individual):
        self.individual = individual

    def population(self, n):
        return [self.individual() for _ in range(n)]


def _make_problem(estimator):
    space = getattr(estimator, "space", None)
    if space is None:

        def sample():
            mask = weighted_bool_individual(
                list, estimator.features_proportion, estimator.n_features
            )
            return estimator._repair_individual(mask)

        population = initialize_feature_population(estimator, _Sampler(sample), list)
    else:

        def sample():
            return [dimension.sample() for dimension in space.param_grid.values()]

        population = initialize_search_population(estimator, _Sampler(sample), list)
    for candidate in population:
        estimator._repair_individual(candidate)
    return OptimizationProblem(
        task="features" if space is None else "parameters",
        initial_population=tuple(tuple(row) for row in population),
        parameter_names=tuple(space.parameters) if space is not None else (),
        dimensions=tuple(space.param_grid.values()) if space is not None else (),
        max_features=getattr(estimator, "max_features", None),
        generations=estimator.generations,
        crossover_probability=estimator.crossover_probability,
        seed=int(np.random.randint(2**31)),
    )


@dataclass(frozen=True, eq=False)
class _Observation:
    owner: object
    genes: tuple
    fitness: tuple


class _SearchRun:
    def __init__(self, estimator, problem):
        self.estimator = estimator
        self.problem = problem
        self.owner = object()
        self.generations = []
        self.elites = []
        self.pending = 0
        self.evaluations = 0
        self.state = _new_telemetry_state()
        self.best_observed = None

    def _rank(self, observation):
        score, *secondary = observation.fitness
        return (self.estimator.criteria_sign * score, *(-x for x in secondary))

    def _archive(self, observation):
        rank = self._rank(observation)
        if len(self.elites) == self.estimator.keep_top_k and not rank > self._rank(self.elites[-1]):
            return
        key = self.estimator._individual_key(observation.genes)
        for elite in self.elites:
            if key == self.estimator._individual_key(elite.genes):
                return
        # Historical archives insert new distinct candidates ahead of equal scores.
        index = next(
            (i for i, elite in enumerate(self.elites) if self._rank(elite) <= rank),
            len(self.elites),
        )
        self.elites.insert(index, observation)
        del self.elites[self.estimator.keep_top_k :]

    def evaluate(self, candidates):
        if self.estimator is None:
            raise RuntimeError("Optimizer evaluation is closed")
        remaining = self.problem.max_evaluations - self.evaluations
        width = len(self.problem.initial_population[0])
        rows = [
            list(islice(candidate, width + 1)) for candidate in islice(candidates, remaining + 1)
        ]
        if not rows or len(rows) > remaining:
            raise ValueError("Expected a nonempty batch within the optimizer proposal budget")
        for row in rows:
            if len(row) != width:
                raise ValueError(f"Optimizer candidate must have {width} coordinates")
            if self.problem.task == "features":
                if not all(
                    isinstance(bit, (bool, int, np.bool_, np.integer)) and bit in (0, 1)
                    for bit in row
                ) or not 1 <= sum(map(int, row)) <= (self.problem.max_features or width):
                    raise ValueError(
                        "Optimizer candidate must be a nonempty binary mask within max_features"
                    )
            elif not all(
                Space.value_in_dimension(dim, value)
                for dim, value in zip(self.problem.dimensions, row)
            ):
                raise ValueError("Optimizer candidate is outside the parameter space")
        fitnesses = self.estimator.evaluate_population(rows)
        replies = []
        for row, fitness in zip(rows, fitnesses, strict=True):
            handle = _Observation(self.owner, tuple(row), tuple(fitness))
            self._archive(handle)
            score = self.estimator._candidate_rows[self.evaluations]["score"]
            if np.isfinite(score) and (
                self.best_observed is None
                or self.estimator.criteria_sign * (score - self.best_observed) > 0
            ):
                self.best_observed = score
            self.evaluations += 1
            replies.append((-self.estimator.criteria_sign * fitness[0], handle))
        self.pending += len(rows)
        return replies

    def report(self, survivors):
        if self.estimator is None:
            raise RuntimeError("Optimizer reporting is closed")
        population_size = len(self.problem.initial_population)
        survivors = list(islice(survivors, population_size + 1))
        if len(survivors) > population_size:
            raise ValueError("Optimizer report exceeds population_size")
        if not survivors or any(
            not isinstance(h, _Observation) or h.owner is not self.owner for h in survivors
        ):
            raise ValueError("Report must contain evaluation handles from this fit")
        gen = len(self.generations)
        if gen > self.problem.generations:
            raise ValueError("Too many optimizer generation reports")
        # Match the established column-wise reductions for two-objective feature masks.
        scores = np.asarray([h.fitness for h in survivors])
        diversity = (
            feature_population_diversity
            if self.problem.task == "features"
            else _population_diversity
        )
        row = diversity([h.genes for h in survivors])
        row.update(
            fitness=np.mean(scores, axis=0)[0],
            fitness_std=np.std(scores, axis=0)[0],
            fitness_max=np.max(scores, axis=0)[0],
            fitness_min=np.min(scores, axis=0)[0],
            fitness_best_observed=np.nan if self.best_observed is None else self.best_observed,
            cross_validate_calls=self.estimator.fit_stats_["cross_validate_calls"],
        )
        complete_generation(
            row, self.state, gen, self.estimator.criteria_sign, self.elites[0].fitness[0]
        )
        if self.estimator.verbose:
            _print_verbose_record(gen, self.pending, row, print_header=gen == 0)
        self.generations.append(dict(gen=gen, nevals=self.pending, **row))
        self.pending = 0


def run_optimizer(estimator, optimize):
    for name in ("toolbox", "_stats", "_pop", "_hof", "logbook", "hof", "_feature_elites"):
        vars(estimator).pop(name, None)
    estimator._candidate_rows = []
    problem = _make_problem(estimator)
    run = _SearchRun(estimator, problem)
    try:
        optimize(problem, run.evaluate, run.report)
    finally:
        run.estimator = None
    if not run.generations or run.pending:
        raise ValueError("Optimizer must report survivors after its final evaluation")
    estimator._hof = [h.genes for h in run.elites]
    if problem.task == "features":
        estimator._feature_elites = (
            [(h.genes, h.fitness) for h in run.elites],
            estimator.keep_top_k,
            estimator.criteria_sign,
        )
    return run.generations, len(run.generations)
