import logging
import pickle

import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


def reset_adapters(estimator):
    estimator.crossover_adapter.reset()
    estimator.mutation_adapter.reset()


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
    _volatile_pickle_attrs = {"toolbox", "_stats", "_pop", "_hof", "_optimizer_snapshot"}

    # Contract for the two serialization paths (see #297):
    #
    # * ``_serializable_state()`` is the FULL snapshot used by ``save``/``load``.
    #   It is restored with ``setattr`` onto an existing instance, so it may hold
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

    def __getstate__(self):
        """Exclude unpicklable DEAP internals from the serialized state."""
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in self._volatile_pickle_attrs
        }

    def __setstate__(self, state):
        """Restore instance state, leaving DEAP attrs unset (rebuilt on fit)."""
        self.optimizer_backend = "deap"
        for key in ("_candidate_rows", "_feature_elites", "logbook", "hof"):
            self.__dict__.pop(key, None)
        self.__dict__.update(state)

    def _serializable_state(self):
        """Backward-compatible accessor for save/load internals."""
        return self.__getstate__()

    def _checkpoint_state(self):
        """Constructor-compatible subset of the state used to resume a search.

        ``param_grid`` is only included when the estimator defines it
        (GAFeatureSelectionCV does not), so this works for both estimators and a
        feature selector never raises ``AttributeError`` while checkpointing.
        """
        state = {key: getattr(self, key) for key in self._checkpoint_core_keys}
        if hasattr(self, "param_grid"):
            state["param_grid"] = self.param_grid
        for key, default in self._checkpoint_optional_defaults.items():
            state[key] = getattr(self, key, default)
        if state["final_selection"] == "reuse_folds":
            state["error_score"] = self.error_score
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
                self.__setstate__(checkpoint_data["estimator_state"])
                if "_candidate_rows" not in vars(self):
                    self.logbook = checkpoint_data["logbook"]
            else:
                # Raw instance written with ``pickle.dump(self, f)``; its DEAP
                # internals were already stripped by ``__getstate__``.
                self.__setstate__(checkpoint_data.__dict__)
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
        from .callbacks.model_checkpoint import load_search_checkpoint

        checkpoint = load_search_checkpoint(self.callbacks, optimizer)
        if optimizer is None:
            return checkpoint, None
        from ._backend_runtime import checkpoint_context

        return checkpoint, checkpoint_context(self, optimizer, checkpoint)

    def _run_optimizer(self, optimizer, runtime, checkpoint, context):
        if optimizer is not None:
            from ._backend_runtime import run_optimizer

            return run_optimizer(self, optimizer, checkpoint, context)

        from ._deap_backend import run_deap

        return run_deap(
            self, runtime.get("candidate_logbook"), runtime.get("elites", ()), bool(checkpoint)
        )

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

    def cv_results_dataframe(self):
        """Return the candidate CV results as a pandas DataFrame.

        Rows are sorted by ascending test rank for the selected scoring metric
        (the `refit` metric in a multi-metric search). Ties retain their order
        in `cv_results_`. For feature selection, `params` contains the feature
        mask for each candidate. The method leaves `cv_results_` unchanged and
        also works after a single-metric fit with `refit=False`.
        """
        check_is_fitted(self, "cv_results_")
        results = pd.DataFrame(self.cv_results_)
        if "features" in results:
            results["params"] = [{"features": list(mask)} for mask in results["features"]]
        return results.sort_values(
            f"rank_test_{self.refit_metric}", kind="stable", ignore_index=True
        )

    def _run_search(self, evaluate_candidates):
        pass  # noqa

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
        if not self._fitted:
            raise NotFittedError(
                f"This {self.__class__.__name__} instance is not fitted yet "
                f"or used refit=False. Call 'fit' with appropriate "
                f"arguments before using this estimator."
            )

        return history_record(self.history, index)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self.history["gen"]):
            result = self.__getitem__(self.n)
            self.n += 1
            return result
        else:
            raise StopIteration  # pragma: no cover

    def __len__(self):
        return self._n_iterations
