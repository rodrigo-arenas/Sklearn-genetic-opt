import logging
import pickle
from copy import copy
from dataclasses import fields
from numbers import Integral
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from .config import EvolutionConfig, OptimizationConfig, RuntimeConfig
from .schedules.validations import check_adapter

logger = logging.getLogger(__name__)
_GROUPED_PARAMETERS = {
    "evolution_config": tuple(field.name for field in fields(EvolutionConfig)),
    "runtime_config": tuple(field.name for field in fields(RuntimeConfig)),
    "optimization_config": tuple(
        field.name for field in fields(OptimizationConfig) if field.name != "final_selection_cv"
    ),
}


def reset_adapters(estimator):
    estimator.crossover_adapter.reset()
    estimator.mutation_adapter.reset()


def _short_cv_positions(ranks, top_k):
    """Select a stable prefix from integer ranks, or defer to the caller's sort."""
    dtype = ranks.dtype
    if (
        top_k is None
        or ranks.ndim != 1
        or len(ranks) < 1024
        or top_k > len(ranks) // 8
        or not isinstance(dtype, np.dtype)
        or dtype.kind not in "iu"
    ):
        return None

    limit = int(top_k)
    values = np.asarray(ranks)
    if np.all(values[:-1] <= values[1:]):
        return range(limit)

    # Sort only the rows that beat the cutoff. Take ties in evaluation order.
    cutoff = np.partition(values, limit - 1)[limit - 1]
    better = np.flatnonzero(values < cutoff)
    ordered = better[np.argsort(values[better], kind="stable")]
    tied = np.flatnonzero(values == cutoff)[: limit - len(better)]
    return np.concatenate((ordered, tied))


def generation_history(records):
    """Build the public column-wise history from generation rows, including legacy gaps."""
    fields = (
        "gen",
        "cross_validate_calls",
        "fitness",
        "fitness_std",
        "fitness_best",
        "fitness_best_observed",
        "fitness_max",
        "fitness_min",
        "finite_fitness_count",
        "population_size",
        "unique_individuals",
        "unique_individual_ratio",
        "genotype_diversity",
        "fitness_improvement",
        "fitness_improved",
        "stagnation_generations",
        "best_generation",
        "mutation_probability",
        "selection_pressure",
        "diversity_control_triggered",
        "random_immigrants",
        "duplicate_replacements",
        "local_refinements",
        "fitness_sharing_applied",
        "mean_niche_count",
        "max_niche_count",
    )
    history = {key: [] for key in fields}
    for row in records:
        if row.get("partial") and "partial" not in history:
            history["partial"] = [False] * len(history["gen"])
        for key, values in history.items():
            values.append(row.get(key, False) if key == "partial" else row.get(key))
    return history


def history_record(history, index):
    return {key: values[index] for key, values in history.items()}


class GeneticEstimatorMixin:
    # ``hof`` is a DEAP HallOfFame but is part of the public, user-facing
    # result surface, so it is deliberately NOT excluded: it is plain
    # picklable data and ``test_estimator_serialization`` asserts on it.
    _volatile_pickle_attrs = {
        "toolbox",
        "_stats",
        "_pop",
        "_hof",
        "_optimizer_snapshot",
        "_population_pool",
    }

    # Contract for the two serialization paths (see #297):
    #
    # * ``_serializable_state()`` is the FULL snapshot used by ``save``/``load``.
    #   It is restored onto an existing instance, so it may hold
    #   non-constructor attributes (``X_``, ``cv_results_``, ``best_estimator_``…).
    # * ``_checkpoint_state()`` is the CONSTRUCTOR-COMPATIBLE subset used by
    #   ``ModelCheckpoint`` to resume a search. Every key is an ``__init__``
    #   parameter, so it can be replayed as ``ClassName(**state)``, and it is a
    #   strict subset of ``_serializable_state()``.
    #
    # The serialization contract test enforces that relationship (every
    # checkpoint key is an __init__ parameter and lives in _serializable_state)
    # so the two paths cannot silently drift apart.
    _checkpoint_core_keys = (
        "estimator",
        "cv",
        "scoring",
        "criteria",
        "return_train_score",
        "error_score",
        "population_size",
        "generations",
        "crossover_probability",
        "mutation_probability",
        "algorithm",
        "random_state",
    )
    _checkpoint_optional_defaults = {
        "local_search": False,
        "local_search_top_k": 1,
        "local_search_steps": 1,
        "local_search_radius": 0.1,
        "diversity_control": False,
        "diversity_threshold": 0.1,
        "diversity_stagnation_generations": 5,
        "diversity_mutation_boost": 2.0,
        "random_immigrants_fraction": 0.1,
        "fitness_sharing": False,
        "sharing_radius": 0.2,
        "sharing_alpha": 1.0,
        "final_selection": False,
        "final_selection_top_k": 3,
        "final_selection_cv": None,
    }

    def set_params(self, **params):
        """Set estimator parameters using the constructor's grouped precedence."""
        if "population_config" in params or "population_initializer" in params:
            config = params.get("population_config", self.population_config)
            # Match the constructor so sklearn.clone sees the same initializer.
            if hasattr(config, "initializer"):
                params["population_initializer"] = config.initializer
        for group, names in _GROUPED_PARAMETERS.items():
            if group in params or any(name in params for name in names):
                config = params.get(group, getattr(self, group))
                params.update(
                    (name, getattr(config, name)) for name in names if hasattr(config, name)
                )
        update_final_cv = "optimization_config" in params or "final_selection_cv" in params
        if update_final_cv:
            direct_cv = params.get("final_selection_cv", self._direct_finalist_cv())
            config = params.get("optimization_config", self.optimization_config)
            params["final_selection_cv"] = (
                getattr(config, "final_selection_cv", None) if direct_cv is None else direct_cv
            )
        result = super().set_params(**params)
        if update_final_cv:
            self._explicit_finalist_cv = direct_cv
        return result

    def _direct_finalist_cv(self):
        if hasattr(self, "_explicit_finalist_cv"):
            return self._explicit_finalist_cv
        # Older saves kept the effective CV, sharing the same object for a grouped value.
        grouped = getattr(self.optimization_config, "final_selection_cv", None)
        return None if self.final_selection_cv is grouped else self.final_selection_cv

    def __sklearn_clone__(self):
        groups = {
            name: config
            for name in _GROUPED_PARAMETERS
            if (config := getattr(self, name)) is not None
        }
        if not groups:
            return super().__sklearn_clone__()
        # Clone the direct settings normally. Then restore the group's shared
        # values so separately copied settings cannot disagree by identity.
        source = copy(self)
        for name in groups:
            setattr(source, name, None)
        source.final_selection_cv = self._direct_finalist_cv()
        result = super(GeneticEstimatorMixin, source).__sklearn_clone__()
        result.set_params(**clone(groups, safe=False))
        if "evolution_config" in groups:
            result._configure_adapters()
        return result

    def _configure_adapters(self):
        self.crossover_adapter = check_adapter(self.crossover_probability)
        self.mutation_adapter = check_adapter(self.mutation_probability)

    def __getstate__(self):
        """Exclude unpicklable DEAP internals from the serialized state."""
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in self._volatile_pickle_attrs
        }

    def __setstate__(self, state):
        """Restore instance state, leaving DEAP attrs unset (rebuilt on fit)."""
        state = dict(state)
        self._clear_fit_details()
        for key in self._volatile_pickle_attrs | {
            "X_",
            "y_",
            "groups_",
            "_fit_params",
            "_cv_splits",
            "_explicit_finalist_cv",
        }:
            self.__dict__.pop(key, None)
        self.optimizer_backend = "deap"
        self.objective_metric = None
        self.__dict__.update(state)

    def _clear_fit_details(self):
        for name in (
            "fit_counts_",
            "fit_stats_",
            "search_time_",
            "search_space_exhausted_",
            "feature_names_in_",
            "_preserved_dataframe_",
            "history",
            "cv_results_",
            "scorer_",
            "multimetric_",
            "n_splits_",
            "logbook",
            "hof",
            "_candidate_rows",
            "_candidate_parameter_conflicts",
            "_feature_elites",
            "best_estimator_",
            "estimator_",
            "refit_time_",
            "best_params_",
            "best_features_",
            "support_",
            "best_index_",
            "best_score_",
            "final_selection_results_",
        ):
            self.__dict__.pop(name, None)

    def _serializable_state(self):
        """Backward-compatible accessor for save/load internals."""
        return self.__getstate__()

    def _checkpoint_state(self):
        """Constructor-compatible subset of the state used to resume a search.

        Include the task-specific search space from ``param_grid`` or
        ``max_features`` when the estimator defines it.
        """
        state = {key: getattr(self, key) for key in self._checkpoint_core_keys}
        if hasattr(self, "param_grid"):
            state["param_grid"] = self.param_grid
        if hasattr(self, "max_features"):
            state["max_features"] = self.max_features
            state["population_initializer"] = self.population_initializer
            if self.preserve_dataframe:
                state["preserve_dataframe"] = self.preserve_dataframe
            population_config = getattr(self, "population_config", None)
            if getattr(population_config, "warm_start_masks", None) is not None:
                state["population_config"] = population_config
        if self.objective_metric is not None:
            state["objective_metric"] = self.objective_metric
        for key, default in self._checkpoint_optional_defaults.items():
            state[key] = getattr(self, key, default)
        return state

    def save(self, filepath):
        """Save the current state of the estimator instance to a file.

        Uses ``__getstate__`` to exclude unpicklable DEAP internals.
        The saved file is a pickled ``dict`` with keys ``estimator_state``
        and ``logbook`` for backward compatibility with ``load()``.
        """
        class_name = self.__class__.__name__
        checkpoint_data = {"estimator_state": self.__getstate__(), "logbook": None}
        if "_candidate_rows" not in vars(self) and hasattr(self, "logbook"):
            checkpoint_data["logbook"] = self.logbook

        try:
            serialized_checkpoint = pickle.dumps(checkpoint_data)
        except (pickle.PicklingError, TypeError, AttributeError) as e:
            logger.error("Error serializing %s: %s", class_name, e)
            raise

        try:
            with open(filepath, "wb") as f:
                f.write(serialized_checkpoint)
        except OSError as e:
            logger.error("Error saving %s to %s: %s", class_name, filepath, e)
            raise

        logger.info("%s model successfully saved to %s", class_name, filepath)

    def load(self, filepath):
        """Load an estimator instance from a file.

        Restores state via ``__setstate__``.  Accepts both the current format
        (pickled ``dict`` with ``estimator_state`` key) and a raw pickled
        instance produced by ``pickle.dumps(self)``.
        """
        class_name = self.__class__.__name__
        try:
            with open(filepath, "rb") as f:
                checkpoint_data = pickle.load(f)
        except FileNotFoundError:
            logger.error("Error loading %s: file not found: %s", class_name, filepath)
            raise
        except (pickle.UnpicklingError, EOFError, ValueError) as e:
            logger.error("Error loading %s: corrupted checkpoint: %s", class_name, e)
            raise
        except OSError as e:
            logger.error("Error loading %s from %s: %s", class_name, filepath, e)
            raise

        try:
            if isinstance(checkpoint_data, dict) and "estimator_state" in checkpoint_data:
                state = checkpoint_data["estimator_state"]
                if "_candidate_rows" not in state:
                    state = {**state, "logbook": checkpoint_data["logbook"]}
            else:
                # Raw instance written with ``pickle.dump(self, f)``; its DEAP
                # internals were already stripped by ``__getstate__``.
                state = checkpoint_data.__dict__
            self.__setstate__(state)
        except KeyError as e:
            logger.error("Error loading %s: missing key in checkpoint: %s", class_name, e)
            raise
        except AttributeError as e:
            logger.error("Error loading %s: invalid checkpoint payload: %s", class_name, e)
            raise

        logger.info("%s model successfully loaded from %s", class_name, filepath)

    def _validate_backend(self):
        from .backends import resolve_optimizer

        return resolve_optimizer(self)

    def _prepare_checkpoint(self, optimizer):
        from .callbacks.model_checkpoint import (
            deap_checkpoint_data,
            load_search_checkpoint,
            restore_legacy_outlier_target,
        )

        checkpoint = load_search_checkpoint(self.callbacks, optimizer)
        restore_legacy_outlier_target(self, checkpoint)
        if optimizer is None:
            return checkpoint, deap_checkpoint_data(self, checkpoint)
        from ._backend_runtime import checkpoint_context

        return checkpoint, checkpoint_context(self, optimizer, checkpoint)

    def _run_optimizer(self, optimizer, runtime, checkpoint, context):
        previous_calls = self.fit_stats_.get("cross_validate_calls", 0)
        started_at = perf_counter()
        try:
            if optimizer is not None:
                from ._backend_runtime import run_optimizer

                if checkpoint:
                    previous_calls = checkpoint["fit_stats"]["cross_validate_calls"]
                result = run_optimizer(self, optimizer, checkpoint, context)
            else:
                from ._deap_backend import run_deap

                result = run_deap(
                    self,
                    runtime.get("candidate_logbook"),
                    runtime.get("elites", ()),
                    bool(checkpoint),
                    data_hash=context,
                )
        finally:
            self.search_time_ = perf_counter() - started_at

        calls = self.fit_stats_["cross_validate_calls"] - previous_calls
        self.fit_counts_ = {
            "search_cv_calls": calls,
            "search_fold_fits": calls * len(self._cv_splits),
            "finalist_cv_calls": 0,
            "finalist_fold_fits": 0,
            "finalist_reused_folds": 0,
            "refit_fits": 0,
        }
        return result

    def __getattr__(self, name):
        # Historical result objects are materialized only when explicitly requested.
        if name == "logbook" and "_candidate_rows" in vars(self):
            from ._deap_backend import legacy_logbook

            self.logbook = legacy_logbook(self._candidate_rows)
            self._candidate_rows = self.logbook.chapters["parameters"]
            return self.logbook
        if name == "hof" and "_feature_elites" in vars(self):
            from ._deap_backend import legacy_feature_archive

            self.hof = legacy_feature_archive(*self._feature_elites)
            return self.hof
        raise AttributeError(name)

    def _select_algorithm(self, pop, stats, hof):
        from ._deap_backend import select_algorithm

        return select_algorithm(self, pop, stats, hof)

    def cv_results_dataframe(self, *, preserve_index=False, columns=None, top_k=None):
        """Return the candidate CV results as a pandas DataFrame.

        Rows are sorted by ascending test rank for the selected scoring metric
        (the `refit` or `objective_metric` in a multi-metric search). Ties retain their order
        in `cv_results_`. For feature selection, `params` contains the feature
        mask for each candidate. The method leaves `cv_results_` unchanged and
        also works after a fit with `refit=False` or when every candidate
        failed and the search recorded its CV results.

        Parameters
        ----------
        preserve_index : bool, default=False
            Use row positions in `cv_results_` as labels in the
            `candidate_index` index. Use `.loc[best_index_]` to select
            the selected row when that attribute is available.
        columns : str or iterable of str, default=None
            Return only these columns, in the requested order. A string
            selects one column. By default, return all columns. Sorting
            uses the selected test rank even when that column is omitted.
            Unknown column names raise `KeyError`.
        top_k : positive int or None, default=None
            Return at most this many rows in ranked order. Ties retain their
            order in `cv_results_`, including at the cutoff. By default,
            return all rows.
        """
        check_is_fitted(self, "cv_results_")
        if top_k is not None and (
            isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k <= 0
        ):
            raise ValueError("top_k must be a positive integer or None")
        rank_column = f"rank_test_{self.refit_metric}"
        data = self.cv_results_
        include_mask_params = "features" in data
        if columns is not None:
            columns = [columns] if isinstance(columns, str) else list(columns)
            include_mask_params = include_mask_params and "params" in columns
            data = {
                name: data[name]
                for name in (*columns, rank_column)
                if not (include_mask_params and name == "params")
            }
        results = pd.DataFrame(data, copy=top_k is None)
        masks = self.cv_results_["features"] if include_mask_params else None
        if top_k is not None:
            ranks = results[rank_column]
            positions = _short_cv_positions(ranks, top_k)
            if positions is None:
                positions = (
                    ranks.reset_index(drop=True).sort_values(kind="stable").index[: int(top_k)]
                )
            results = results.take(positions)
            if masks is not None:
                masks = [masks[i] for i in positions]
        if masks is not None:
            results["params"] = [{"features": list(mask)} for mask in masks]
        if preserve_index:
            results.index.name = "candidate_index"
        if top_k is None:
            results = results.sort_values(
                rank_column, kind="stable", ignore_index=not preserve_index
            )
        elif not preserve_index:
            results.index = pd.RangeIndex(len(results))
        return results if columns is None else results.loc[:, columns]

    def _run_search(self, evaluate_candidates):
        pass  # noqa

    def __sklearn_is_fitted__(self):
        state = vars(self)
        if "cv_results_" not in state:
            return False
        if self.refit:
            return self._fitted
        # Older search-only results may have no best_index_ or failure flag.
        return "best_index_" in state or not all(self.cv_results_.get("all_fits_failed", (False,)))

    @property
    def _fitted(self):
        if not hasattr(self, "best_estimator_"):
            return False
        try:
            check_is_fitted(self.best_estimator_)
            is_fitted = True
        except NotFittedError:
            is_fitted = False

        has_history = hasattr(self, "history") and bool(self.history)
        return all([is_fitted, has_history, self.refit])

    def __getitem__(self, index):
        check_is_fitted(self, "history")
        return history_record(self.history, index)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n >= len(self):
            raise StopIteration
        result = self[self.n]
        self.n += 1
        return result

    def __len__(self):
        check_is_fitted(self, "history")
        return len(self.history["gen"])
