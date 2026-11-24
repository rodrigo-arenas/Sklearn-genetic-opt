"""Evaluate and record optimizer proposals using the shared estimator CV pipeline."""

from collections import Counter
from copy import copy
from dataclasses import dataclass
from hashlib import md5
from itertools import islice, repeat
from operator import is_
import random
import pickle

from joblib import hash as joblib_hash

import numpy as np

from ._optimizer_checkpoint import (
    checkpoint_engine,
    environment,
    resume_method,
    training_data_hash,
    validate_boundary,
)
from .backends import OptimizationProblem
from .callbacks.model_checkpoint import ModelCheckpoint
from .callbacks.validations import eval_callbacks
from .config import PopulationConfig
from .evaluation import (
    evaluate_population,
    evaluation_context,
    _evaluation_limit_reached,
    restore_candidate_positions,
    start_evaluation_callbacks,
)
from .optimizer_control import finite_search_space_size
from ._telemetry import (
    _new_telemetry_state,
    _population_diversity,
    _print_verbose_record,
    complete_generation,
    feature_population_diversity,
    population_fitness_stats,
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
    from .genetic_search import _uses_native_feature_initialization

    space = getattr(estimator, "space", None)
    if space is None:
        initializer = initialize_feature_population

        def sample():
            mask = weighted_bool_individual(
                list, estimator.features_proportion, estimator.n_features
            )
            return estimator._repair_individual(mask)

    else:
        initializer = initialize_search_population

        def sample():
            return [dimension.sample() for dimension in space.param_grid.values()]

    population = initializer(estimator, _Sampler(sample), list)
    if not _uses_native_feature_initialization(estimator, initializer):
        for candidate in population:
            estimator._repair_individual(candidate)
    max_features = getattr(estimator, "max_features", None)
    return OptimizationProblem(
        task="features" if space is None else "parameters",
        initial_population=tuple(tuple(row) for row in population),
        parameter_names=tuple(space.parameters) if space is not None else (),
        dimensions=tuple(space.param_grid.values()) if space is not None else (),
        max_features=None if max_features is None else int(max_features),
        generations=int(estimator.generations),
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


def _valid_feature_mask(row, limit):
    """Return mask validity and whether every coordinate is a Python int."""
    if all(map(is_, map(type, row), repeat(int))):
        # Bundled engines yield plain integers. Use bulk built-in operations
        # for their binary values and count without casting every coordinate.
        return set(row) <= {0, 1} and 1 <= sum(row) <= limit, True
    valid = (
        all(isinstance(bit, (bool, int, np.bool_, np.integer)) and bit in (0, 1) for bit in row)
        and 1 <= sum(map(int, row)) <= limit
    )
    return valid, False


class _SearchRun:
    def __init__(self, estimator, problem, space_size):
        self.estimator = estimator
        self.problem = problem
        self.space_size = space_size
        self.exhausted = False
        estimator.search_space_exhausted_ = False
        self.owner = object()
        self.generations = _GenerationLog()
        self.elites = []
        # Parameter keys can be expensive to rebuild for every member of a large archive.
        self._elite_keys = (
            Counter() if problem.task == "parameters" and estimator.keep_top_k > 1 else None
        )
        self.pending = 0
        self.evaluations = 0
        self._initial_feature_masks = None
        self.state = _new_telemetry_state()
        self.best_observed = None
        self.limited_batch = []
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

    def _update_coverage(self):
        # This cache contains only observations from this fit or its exact resume.
        self.exhausted = self.space_size == len(self.estimator.fitness_cache)
        self.estimator.search_space_exhausted_ = self.exhausted

    def snapshot(self):
        # Keep the latest resumable boundary when a limit cuts a generation short.
        if (
            self.generations[-1].get("partial")
            and self.resume_supported
            and len(self.generations) > 1
        ):
            return None
        est = self.estimator
        return {
            "optimizer_checkpoint_version": 1,
            "outlier_scoring_uses_fitted_detector": True,
            "typed_categorical_identity": True,
            "stable_nan_categories": True,
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
        restore_candidate_positions(self.estimator, self.estimator._candidate_rows)
        if self.problem.task == "parameters":
            # Preserve the snapshot's shared identities between categories, archived
            # observations and cache keys, including estimator-valued categories.
            self.estimator.space = Space(
                dict(zip(self.problem.parameter_names, self.problem.dimensions))
            )
            if not snapshot.get("stable_nan_categories"):
                self.estimator.fitness_cache = self.estimator.space._rekey_nan_cache(
                    self.estimator.fitness_cache
                )
        if self._elite_keys is not None:
            self._elite_keys = Counter(
                self.estimator._individual_key(elite.genes) for elite in self.elites
            )
        self._update_coverage()

    def _stop_at_evaluation_limit(self):
        if self.limited_batch:
            self._record(self.limited_batch, partial=True)
        self.stopped = True
        raise _StopSearch

    def _rank(self, observation):
        score, *secondary = observation.fitness
        return (self.estimator.criteria_sign * score, *(-x for x in secondary))

    def _archive(self, observation):
        rank = self._rank(observation)
        keep = self.estimator.keep_top_k
        if len(self.elites) == keep and not rank > self._rank(self.elites[-1]):
            return
        key = self.estimator._individual_key(observation.genes)
        keys = self._elite_keys
        if keys is None:
            for elite in self.elites:
                if key == self.estimator._individual_key(elite.genes):
                    return
        elif key in keys:
            return
        # Historical archives insert new distinct candidates ahead of equal scores.
        index = next(
            (i for i, elite in enumerate(self.elites) if self._rank(elite) <= rank),
            len(self.elites),
        )
        self.elites.insert(index, observation)
        if keys is not None:
            keys[key] = 1
            for removed in self.elites[keep:]:
                removed_key = self.estimator._individual_key(removed.genes)
                keys[removed_key] -= 1
                # Older checkpoints can have equivalent keys for more than one elite.
                if not keys[removed_key]:
                    del keys[removed_key]
        del self.elites[keep:]

    def _proposal_rows(self, candidates, remaining):
        # The built-in selector already repaired these exact immutable masks.
        if (
            candidates is self._initial_feature_masks
            and candidates is not None
            and len(candidates) <= remaining
        ):
            # Truncation only needs a mutable outer list of immutable masks.
            return list(candidates), True

        width = len(self.problem.initial_population[0])
        features = self.problem.task == "features"
        rows = [
            (
                candidate
                if features and type(candidate) is tuple
                else list(islice(candidate, width + 1))
            )
            for candidate in islice(candidates, remaining + 1)
        ]
        if not rows or len(rows) > remaining:
            raise ValueError("Expected a nonempty batch within the optimizer proposal budget")
        canonical_masks = features
        validated_tuples = {}
        for index, row in enumerate(rows):
            immutable = type(row) is tuple
            if immutable and validated_tuples.get(id(row)) is row:
                continue
            if len(row) != width:
                raise ValueError(f"Optimizer candidate must have {width} coordinates")
            if features:
                valid, canonical = _valid_feature_mask(row, self.problem.max_features or width)
                if not valid:
                    raise ValueError(
                        "Optimizer candidate must be a nonempty binary mask within max_features"
                    )
                canonical_masks &= canonical
                if immutable:
                    if canonical:
                        # Keep an identity only for tuples of immutable Python integers.
                        validated_tuples[id(row)] = row
                    else:
                        rows[index] = list(row)
            elif not all(
                Space.value_in_dimension(dim, value)
                for dim, value in zip(self.problem.dimensions, row)
            ):
                raise ValueError("Optimizer candidate is outside the parameter space")
        return rows, canonical_masks

    def evaluate(self, candidates):
        from .genetic_search import _uses_native_feature_evaluation

        if self.estimator is None:
            raise RuntimeError("Optimizer evaluation is closed")
        # A resumed solver may replay its journal first. Stop when it asks the
        # live evaluator for fresh proposals, after replay has verified the log.
        if self.stopped or self.exhausted:
            self.stopped = True
            raise _StopSearch
        if _evaluation_limit_reached(self.estimator):
            self._stop_at_evaluation_limit()
        remaining = self.problem.max_evaluations - self.evaluations
        rows, canonical_masks = self._proposal_rows(candidates, remaining)
        requested = len(rows)
        if (
            canonical_masks
            and self.problem.max_features == self.estimator.max_features
            and _uses_native_feature_evaluation(self.estimator)
        ):
            # The backend validated these integer bits against the active cap.
            fitnesses = evaluate_population(self.estimator, rows, "current_generation_features")
        else:
            if self.problem.task == "features":
                # Repair and custom evaluators can edit each mask independently.
                for index, row in enumerate(rows):
                    if type(row) is tuple:
                        rows[index] = list(row)
            fitnesses = self.estimator.evaluate_population(rows)
        replies = []
        minimize = self.estimator.criteria_sign < 0
        for row, fitness in zip(rows, fitnesses, strict=True):
            handle = _Observation(self.owner, tuple(row), tuple(fitness))
            self._archive(handle)
            score = self.estimator._candidate_rows[self.evaluations]["score"]
            if np.isfinite(score) and (
                self.best_observed is None
                or (score < self.best_observed if minimize else score > self.best_observed)
            ):
                self.best_observed = score
            self.evaluations += 1
            replies.append((-self.estimator.criteria_sign * fitness[0], handle))
        self.pending += len(rows)
        if self.estimator._evaluation_timers or self.estimator._evaluation_cv_limit is not None:
            self.limited_batch.extend(handle for _, handle in replies)
            if len(rows) < requested:
                self._stop_at_evaluation_limit()
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
        population = [h.genes for h in survivors]
        row = (
            feature_population_diversity(population)
            if self.problem.task == "features"
            else _population_diversity(population, self.estimator.space)
        )
        row.update(population_fitness_stats([h.fitness for h in survivors]))
        row.update(
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
        self.limited_batch.clear()
        if not partial:
            self._update_coverage()
        if self._callbacks("on_step", record) or self.exhausted:
            self.stopped = True
            raise _StopSearch


def _backend_name(backend):
    if isinstance(backend, str):
        return backend
    kind = backend if hasattr(backend, "__qualname__") else type(backend)
    return f"{kind.__module__}.{kind.__qualname__}"


def _cv_fingerprint(cv, index_kinds, *, python_selectors):
    if type(cv) not in (list, tuple) or not cv:
        return cv
    folds = []
    for fold in cv:
        if type(fold) not in (list, tuple) or len(fold) != 2:
            return cv
        # sklearn.clone can turn a fold shared by search and finalist CV into
        # two independent copies. The ordered selectors define both folds.
        selectors = []
        for indices in fold:
            if (
                type(indices) is np.ndarray
                and indices.ndim == 1
                and indices.dtype.kind in index_kinds
            ):
                selectors.append(joblib_hash(np.ascontiguousarray(indices)))
            elif (
                python_selectors
                and type(indices) in (list, tuple)
                and all(type(index) in (int, bool) for index in indices)
            ):
                # Keep the container, scalar types and ordering of native selectors.
                contents = pickle.dumps(indices, protocol=3)
                selectors.append(
                    ("python_cv_selector", md5(contents, usedforsecurity=False).hexdigest())
                )
            else:
                return cv
        folds.append(tuple(selectors))
    return ("explicit_cv_indices", tuple(folds))


def _config_hash(estimator, *, index_kinds="biu", python_selectors=True):
    params = estimator.get_params(deep=False)
    population_config = params.get("population_config")
    if (
        isinstance(population_config, PopulationConfig)
        and population_config.warm_start_masks is None
    ):
        # Preserve hashes of older PopulationConfig instances without this field.
        population_config = copy(population_config)
        vars(population_config).pop("warm_start_masks", None)
        params["population_config"] = population_config
    if params.get("objective_metric") is None:
        # The default matches checkpoints created before this option existed.
        params.pop("objective_metric", None)
    if not params.get("preserve_dataframe", False):
        params.pop("preserve_dataframe", None)
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
        if index_kinds:
            for name in ("cv", "final_selection_cv"):
                params[name] = _cv_fingerprint(
                    params[name], index_kinds, python_selectors=python_selectors
                )
        return joblib_hash(params)
    except (pickle.PicklingError, TypeError, AttributeError) as exc:
        raise ValueError(
            "Exact checkpoint resume needs serializable search settings. Use importable scorers and optimizers."
        ) from exc


def checkpoint_context(estimator, optimize, checkpoint):
    if not callable(resume_method(optimize)) or not any(
        isinstance(c, ModelCheckpoint) for c in estimator.callbacks
    ):
        return None
    if checkpoint and checkpoint["backend"] != _backend_name(estimator.optimizer_backend):
        raise ValueError("Checkpoint belongs to a different optimizer backend")
    engine = checkpoint_engine(optimize)
    compatibility = (environment(engine),) if engine else ()
    if checkpoint and checkpoint["context"][3:] != compatibility:
        raise ValueError(
            "Optimizer checkpoint environment or adapter version differs from this fit"
        )
    if checkpoint and engine:
        validate_boundary(engine, checkpoint)
    # Fingerprint training inputs once, before sampling folds, dimensions or RNGs.
    data = training_data_hash(estimator)
    config = _config_hash(estimator)
    if checkpoint:
        saved_data, initial_config, sampled_config = checkpoint["context"][:3]
        accepted = (initial_config, sampled_config)
        # Older checkpoints normalized only NumPy selectors, or hashed all
        # fold containers together with the rest of the search settings.
        if data != saved_data or (
            config not in accepted
            and _config_hash(estimator, python_selectors=False) not in accepted
            and _config_hash(estimator, index_kinds="iu", python_selectors=False) not in accepted
            and _config_hash(estimator, index_kinds="") not in accepted
        ):
            raise ValueError("Checkpoint data or search configuration differs from this fit")
        return checkpoint["context"]
    return (data, config, None, *compatibility)


def run_optimizer(estimator, optimize, checkpoint, context):
    from .genetic_search import _uses_native_feature_evaluation

    for name in ("toolbox", "_stats", "_pop", "_hof", "logbook", "hof", "_feature_elites"):
        vars(estimator).pop(name, None)
    estimator._candidate_rows = []
    checkpointing = any(isinstance(c, ModelCheckpoint) for c in estimator.callbacks)
    resumable = callable(resume_method(optimize))
    problem = checkpoint["problem"] if checkpoint else _make_problem(estimator, checkpointing)
    # Plugins control their stopping rules. Keep the full proposal history when
    # finalist selection is enabled, since repeats may affect its tie handling.
    space_size = None
    finalists_enabled = estimator.refit and estimator.final_selection
    track_coverage = bool(checkpoint_engine(optimize)) and not finalists_enabled
    initial_use_cache = estimator.use_cache
    if track_coverage:
        space_size = finite_search_space_size(estimator, problem.max_evaluations)
    run = _SearchRun(estimator, problem, space_size)
    if not checkpoint and problem.task == "features" and _uses_native_feature_evaluation(estimator):
        run._initial_feature_masks = problem.initial_population
    run.resume_supported = resumable
    if context is not None and not checkpoint:
        # Reusing the same estimator can legitimately expose advanced splitter/dimension RNGs.
        context = (*context[:2], _config_hash(estimator), *context[3:])
    run.checkpoint_context = context
    if checkpoint:
        run.restore(checkpoint)
    if checkpointing:
        estimator._optimizer_snapshot = run.snapshot
    try:
        with evaluation_context(estimator):
            start_evaluation_callbacks(estimator.callbacks, estimator)
            if track_coverage and estimator.use_cache != initial_use_cache:
                run.space_size = finite_search_space_size(estimator, problem.max_evaluations)
                run._update_coverage()
            try:
                if checkpoint:
                    random.setstate(checkpoint["rng"][0])
                    np.random.set_state(checkpoint["rng"][1])
                    estimator._evaluation_started = True
                    if len(run.generations) <= problem.generations:
                        resume_method(optimize)(
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
