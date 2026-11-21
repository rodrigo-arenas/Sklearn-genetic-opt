"""Evaluate and record optimizer proposals using the shared estimator CV pipeline."""

from dataclasses import dataclass
from itertools import islice
import random
import pickle

from joblib import hash as joblib_hash

import numpy as np

from .backends import OptimizationProblem
from .callbacks.model_checkpoint import ModelCheckpoint
from .callbacks.validations import eval_callbacks
from .evaluation import evaluation_timers, _time_budget_exhausted
from ._telemetry import (
    _new_telemetry_state,
    _population_diversity,
    _print_verbose_record,
    complete_generation,
    feature_population_diversity,
)
from .population import initialize_feature_population, initialize_search_population
from .space import Continuous, Space
from .utils.random import weighted_bool_individual


class _Sampler:
    def __init__(self, individual):
        self.individual = individual

    def population(self, n):
        return [self.individual() for _ in range(n)]


def _make_problem(estimator, checkpointing):
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
        checkpoint_requested=checkpointing,
    )


@dataclass(frozen=True, eq=False)
class _Observation:
    owner: object
    genes: tuple
    fitness: tuple


class _GenerationLog(list):
    """Generation rows with the column selection used by historical callbacks."""

    def select(self, *fields):
        columns = tuple([row.get(field) for row in self] for field in fields)
        return columns[0] if len(columns) == 1 else columns


class _StopSearch(Exception):
    """Unwind a synchronous optimizer at a report or evaluation budget boundary."""


class _SearchRun:
    def __init__(self, estimator, problem):
        self.estimator = estimator
        self.problem = problem
        self.owner = object()
        self.generations = _GenerationLog()
        self.elites = []
        self.pending = 0
        self.evaluations = 0
        self.state = _new_telemetry_state()
        self.best_observed = None
        self.timed_batch = []
        self.stopped = False
        self.checkpoint_context = None
        self.resume_supported = False
        self.solver_state = None
        # Persist after observers have finished using the proposal RNGs for this step.
        self.step_callbacks = sorted(
            estimator.callbacks, key=lambda c: isinstance(c, ModelCheckpoint)
        )

    def _callbacks(self, method, record=None):
        callbacks = self.step_callbacks if method == "on_step" else self.estimator.callbacks
        return eval_callbacks(callbacks, record, self.generations, self.estimator, method)

    def snapshot(self):
        # Keep the latest resumable boundary when a deadline cuts a generation short.
        if (
            self.generations[-1].get("partial")
            and self.resume_supported
            and len(self.generations) > 1
        ):
            return None
        est = self.estimator
        return {
            "optimizer_checkpoint_version": 1,
            "backend": _backend_name(est.optimizer_backend),
            "resumable": self.resume_supported and self.solver_state is not None,
            "context": self.checkpoint_context,
            "problem": self.problem,
            "solver_state": self.solver_state,
            "owner": self.owner,
            "elites": self.elites,
            "logbook": list(self.generations),
            "candidate_records": list(est._candidate_rows),
            "fitness_cache": est.fitness_cache,
            "fit_stats": est.fit_stats_,
            "cv_splits": est._cv_splits,
            "telemetry": self.state,
            "best_observed": self.best_observed,
            "rng": (random.getstate(), np.random.get_state()),
        }

    def restore(self, snapshot):
        self.owner = snapshot["owner"]
        self.elites = snapshot["elites"]
        self.generations = _GenerationLog(snapshot["logbook"])
        self.state = snapshot["telemetry"]
        self.best_observed = snapshot["best_observed"]
        self.estimator._candidate_rows = snapshot["candidate_records"]
        self.evaluations = len(self.estimator._candidate_rows)
        self.estimator.fitness_cache = snapshot["fitness_cache"]
        self.estimator.fit_stats_ = snapshot["fit_stats"]
        if self.problem.task == "parameters":
            # Preserve the snapshot's shared identities between categories, archived
            # observations and cache keys, including estimator-valued categories.
            self.estimator.space = Space(
                dict(zip(self.problem.parameter_names, self.problem.dimensions))
            )

    def _deadline(self):
        if self.timed_batch:
            self._record(self.timed_batch, partial=True)
        self.stopped = True
        raise _StopSearch

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
        if self.stopped:
            raise _StopSearch
        if _time_budget_exhausted(self.estimator):
            self._deadline()
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
        requested = len(rows)
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
        if self.estimator._evaluation_timers:
            self.timed_batch.extend(handle for _, handle in replies)
            if len(rows) < requested:
                self._deadline()
        return replies

    def report(self, survivors, *, state=None):
        if self.estimator is None:
            raise RuntimeError("Optimizer reporting is closed")
        if self.stopped:
            raise _StopSearch
        population_size = len(self.problem.initial_population)
        survivors = list(islice(survivors, population_size + 1))
        if len(survivors) > population_size:
            raise ValueError("Optimizer report exceeds population_size")
        if not survivors or any(
            not isinstance(h, _Observation) or h.owner is not self.owner for h in survivors
        ):
            raise ValueError("Report must contain evaluation handles from this fit")
        if self.checkpoint_context is not None and state is None:
            raise ValueError(
                "A resumable optimizer must report non-None state at each checkpoint boundary"
            )
        self.solver_state = state if self.problem.checkpoint_requested else None
        self._record(survivors)

    def _record(self, survivors, partial=False):
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
        if partial:
            self.solver_state = None
            row["partial"] = True
        record = dict(gen=gen, nevals=self.pending, **row)
        self.generations.append(record)
        self.pending = 0
        self.timed_batch.clear()
        if self._callbacks("on_step", record):
            self.stopped = True
            raise _StopSearch


def _backend_name(backend):
    if isinstance(backend, str):
        return backend
    kind = backend if hasattr(backend, "__qualname__") else type(backend)
    return f"{kind.__module__}.{kind.__qualname__}"


def _config_hash(estimator):
    params = estimator.get_params(deep=False)
    if "param_grid" in params:
        # Keep coordinate order. A bound scipy.stats sampler carries the process-wide
        # NumPy RNG and is derived entirely from the Continuous constructor settings.
        params["param_grid"] = tuple(
            (
                name,
                (
                    (Continuous, dim.lower, dim.upper, dim.distribution, dim.random_state)
                    if type(dim) is Continuous
                    else dim
                ),
            )
            for name, dim in params["param_grid"].items()
        )
    try:
        return joblib_hash(params)
    except (pickle.PicklingError, TypeError, AttributeError) as exc:
        raise ValueError(
            "Exact checkpoint resume needs serializable search settings. Use importable scorers and optimizers."
        ) from exc


def checkpoint_context(estimator, optimize, checkpoint):
    if not callable(getattr(optimize, "resume", None)) or not any(
        isinstance(c, ModelCheckpoint) for c in estimator.callbacks
    ):
        return None
    if checkpoint and checkpoint["backend"] != _backend_name(estimator.optimizer_backend):
        raise ValueError("Checkpoint belongs to a different optimizer backend")
    # Fingerprint training inputs once, before sampling folds, dimensions or RNGs.
    data = joblib_hash(
        (type(estimator), estimator.X_, estimator.y_, estimator.groups_, estimator._fit_params)
    )
    config = _config_hash(estimator)
    if checkpoint:
        saved_data, initial_config, sampled_config = checkpoint["context"]
        if data != saved_data or config not in (initial_config, sampled_config):
            raise ValueError("Checkpoint data or search configuration differs from this fit")
        return checkpoint["context"]
    return (data, config, None)


def run_optimizer(estimator, optimize, checkpoint, context):
    for name in ("toolbox", "_stats", "_pop", "_hof", "logbook", "hof", "_feature_elites"):
        vars(estimator).pop(name, None)
    estimator._candidate_rows = []
    checkpointing = any(isinstance(c, ModelCheckpoint) for c in estimator.callbacks)
    resumable = callable(getattr(optimize, "resume", None))
    problem = checkpoint["problem"] if checkpoint else _make_problem(estimator, checkpointing)
    run = _SearchRun(estimator, problem)
    run.resume_supported = resumable
    if context is not None and not checkpoint:
        # Reusing the same estimator can legitimately expose advanced splitter/dimension RNGs.
        context = (*context[:2], _config_hash(estimator))
    run.checkpoint_context = context
    if checkpoint:
        run.restore(checkpoint)
    if checkpointing:
        estimator._optimizer_snapshot = run.snapshot
    try:
        with evaluation_timers(estimator):
            run._callbacks("on_start")
            try:
                if checkpoint:
                    random.setstate(checkpoint["rng"][0])
                    np.random.set_state(checkpoint["rng"][1])
                    estimator._evaluation_started = True
                    if len(run.generations) <= problem.generations:
                        optimize.resume(
                            problem, run.evaluate, run.report, checkpoint["solver_state"]
                        )
                else:
                    optimize(problem, run.evaluate, run.report)
            except _StopSearch:
                if not run.stopped:
                    raise
            finally:
                run._callbacks("on_end")
    finally:
        vars(estimator).pop("_optimizer_snapshot", None)
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
