import logging
import pickle
import random
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from sklearn import get_config
from sklearn.utils.validation import _num_samples

from .._optimizer_checkpoint import _is_nonnegative_count, resume_method, training_data_hash
from ..evaluation import create_fit_stats
from .base import BaseCallback
from .early_stoppers import _validate_positive_integer

logger = logging.getLogger(__name__)

_OPTIMIZER_RESUME_FIELDS = {
    "backend",
    "context",
    "problem",
    "solver_state",
    "owner",
    "elites",
    "logbook",
    "candidate_records",
    "fitness_cache",
    "fit_stats",
    "cv_splits",
    "telemetry",
    "best_observed",
    "rng",
}


def _search_count(estimator, field):
    return getattr(estimator, "fit_stats_", {}).get(field, 0)


class ModelCheckpoint(BaseCallback):
    """Save generation checkpoints and report the latest fit's resumable CV work.

    With ``every_n_generations=N``, save at generation indices 0, N, 2*N, and
    so on. A fit can finish after the last scheduled write.
    DEAP saves the updated final generation again when local refinement runs.
    """

    def __init__(self, checkpoint_path, *, every_n_generations=1, **dump_options):
        _validate_positive_integer(every_n_generations, "every_n_generations")
        self.checkpoint_path = checkpoint_path
        self.every_n_generations = int(every_n_generations)
        self.dump_options = dump_options

    def on_start(self, estimator=None):
        self._starting_cv_calls = _search_count(estimator, "cross_validate_calls")
        self._starting_refinements = _search_count(estimator, "local_refinement_candidates")
        self.resumable_cv_calls_ = 0
        self.unsaved_cv_calls_ = 0

    def on_step(self, record=None, logbook=None, estimator=None):
        interval = getattr(self, "every_n_generations", 1)
        if interval == 1 or logbook[-1]["gen"] % interval == 0:
            self._save(logbook, estimator)

    def _save(self, logbook, estimator):
        temporary_path = None
        try:
            snapshot = getattr(estimator, "_optimizer_snapshot", None)
            if snapshot is None:
                checkpoint_data = self._legacy_snapshot(estimator, logbook)
            else:
                checkpoint_data = snapshot()
                if checkpoint_data is None:
                    return
            checkpoint_data["enable_metadata_routing"] = bool(
                get_config()["enable_metadata_routing"]
            )
            checkpoint_data["preserves_missing_outlier_target"] = not getattr(
                estimator, "_uses_legacy_outlier_target", False
            )
            destination = Path(self.checkpoint_path)
            with NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
                temporary_path = Path(stream.name)
                try:
                    pickle.dump(checkpoint_data, stream)
                except (ValueError, RecursionError) as error:
                    raise pickle.PicklingError(f"{type(error).__name__}: {error}") from error
            temporary_path.replace(destination)
            temporary_path = None
        except (OSError, pickle.PicklingError, AttributeError, TypeError) as e:
            logger.warning("Error saving checkpoint to %s: %s", self.checkpoint_path, e)
            return
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError as error:
                    logger.warning(
                        "Error removing temporary checkpoint %s: %s", temporary_path, error
                    )

        if hasattr(self, "_starting_cv_calls"):
            self.resumable_cv_calls_ = (
                _search_count(estimator, "cross_validate_calls") - self._starting_cv_calls
                if checkpoint_data.get("resumable", True)
                else 0
            )
        logger.info("Checkpoint saved to %s", self.checkpoint_path)

    def on_end(self, logbook=None, estimator=None):
        if hasattr(self, "_starting_cv_calls"):
            if (
                logbook
                and getattr(estimator, "_optimizer_snapshot", None) is None
                and _search_count(estimator, "local_refinement_candidates")
                > self._starting_refinements
            ):
                # DEAP updates the final log record after the last on_step.
                self._save(logbook, estimator)
            fresh_calls = _search_count(estimator, "cross_validate_calls") - self._starting_cv_calls
            self.unsaved_cv_calls_ = fresh_calls - self.resumable_cv_calls_

    @staticmethod
    def _legacy_snapshot(estimator, logbook):
        estimator_state = estimator._checkpoint_state()
        # Keep resume data apart from the constructor arguments. The candidate
        # log backs cv_results_, while logbook contains generation summaries.
        # on_step serializes this payload synchronously, so the pickler can
        # preserve objects shared between the candidate log and fitness cache.
        runtime_state = {
            "elites": [
                (list(individual), individual.fitness.values)
                for individual in getattr(estimator, "_hof", ())
            ],
            "rng_state": (random.getstate(), np.random.get_state()),
            "fitness_cache": getattr(estimator, "fitness_cache", {}),
            "fit_stats_": getattr(estimator, "fit_stats_", None),
            "candidate_logbook": getattr(estimator, "logbook", None),
            "copy_cv_indices": getattr(estimator, "_copy_cv_indices", False),
        }
        if getattr(estimator, "_save_outlier_cv_splits", False):
            runtime_state["outlier_cv_splits"] = estimator._cv_splits
        data_hash = getattr(estimator, "_deap_checkpoint_data_hash", None)
        if data_hash is not None:
            runtime_state["training_data_hash"] = data_hash
            if estimator.cv is not None and not isinstance(estimator.cv, Integral):
                runtime_state["search_cv_splits"] = estimator._cv_splits
        if getattr(estimator, "multimetric_", False):
            runtime_state["refit_metric"] = estimator.refit_metric
        crossover_adapter = getattr(estimator, "crossover_adapter", None)
        mutation_adapter = getattr(estimator, "mutation_adapter", None)
        if crossover_adapter is not None and mutation_adapter is not None:
            runtime_state["adapter_state"] = {
                "crossover": crossover_adapter.state_dict(),
                "mutation": mutation_adapter.state_dict(),
            }
        return {
            "estimator_state": estimator_state,
            "logbook": logbook,
            "runtime_state": runtime_state,
            "outlier_scoring_uses_fitted_detector": True,
            "typed_categorical_identity": True,
            "stable_nan_categories": True,
        }

    def load(self):
        """Load the model state from the checkpoint file."""
        try:
            with open(self.checkpoint_path, "rb") as f:
                checkpoint_data = pickle.load(f)
        except FileNotFoundError:
            logger.error("Error loading checkpoint: file not found: %s", self.checkpoint_path)
            raise
        except (pickle.UnpicklingError, EOFError, ValueError) as e:
            logger.error("Error loading checkpoint: corrupted file: %s", e)
            raise
        except OSError as e:
            logger.error("Error loading checkpoint from %s: %s", self.checkpoint_path, e)
            raise

        return checkpoint_data


def restore_legacy_outlier_target(estimator, checkpoint):
    """Keep the target used by saved outlier CV scores on continuation."""
    estimator._uses_legacy_outlier_target = False
    if (
        checkpoint is None
        or checkpoint.get("preserves_missing_outlier_target")
        or estimator.y_ is not None
    ):
        return

    from ..genetic_search import _is_outlier_detector

    if _is_outlier_detector(estimator.estimator):
        estimator.y_ = np.zeros(_num_samples(estimator.X_))
        estimator._uses_legacy_outlier_target = True


def deap_checkpoint_data(estimator, checkpoint):
    """Fingerprint fresh inputs or verify them against a DEAP checkpoint."""
    if not any(isinstance(callback, ModelCheckpoint) for callback in estimator.callbacks):
        return None
    saved = None
    if checkpoint is not None:
        saved = (checkpoint.get("runtime_state") or {}).get("training_data_hash")
        # Resuming a historical snapshot cannot establish where its scores came from.
        if saved is None:
            return None
    try:
        current = training_data_hash(estimator)
    except (pickle.PicklingError, TypeError, AttributeError) as exc:
        if saved is None:
            return None
        raise ValueError(
            "Cannot verify the training data for this checkpoint. Use a new checkpoint path."
        ) from exc
    if saved is not None and current != saved:
        raise ValueError(
            "Checkpoint training data differs from this fit. Use a new checkpoint path."
        )
    return current


def _last_generation_record(logbook):
    if isinstance(logbook, (list, tuple)) and logbook and isinstance(logbook[-1], Mapping):
        return logbook[-1]
    return {}


def _validate_counter(value, path, field):
    if not _is_nonnegative_count(value):
        raise ValueError(
            f"Invalid search checkpoint at {path}: {field} must be a nonnegative integer. "
            "Use a new checkpoint path to start a search."
        )


def _validate_saved_cv_calls(stats, last, path, stats_field):
    key = "cross_validate_calls"
    if key not in stats:
        return
    count = stats[key]
    field = f"{stats_field}.{key}"
    _validate_counter(count, path, field)
    if key in last:
        reported = last[key]
        if not _is_nonnegative_count(reported) or int(reported) != int(count):
            raise ValueError(
                f"Invalid search checkpoint at {path}: {field} does not match "
                f"the last generation's {key}. Use a new checkpoint path to start a search."
            )


def _restore_deap_fit_stats(data, path):
    runtime = data.get("runtime_state")
    if runtime is None:
        runtime = {}
    if not isinstance(runtime, Mapping):
        raise ValueError(f"Invalid search checkpoint at {path}: runtime_state must be a mapping.")
    saved_stats = runtime.get("fit_stats_")
    if saved_stats is None:
        saved_stats = {}
    if not isinstance(saved_stats, Mapping):
        raise ValueError(
            f"Invalid search checkpoint at {path}: runtime_state.fit_stats_ must be a mapping."
        )

    stats = create_fit_stats()
    stats.update(saved_stats)
    last = _last_generation_record(data.get("logbook"))
    key = "cross_validate_calls"
    if key not in saved_stats and key in last:
        _validate_saved_cv_calls(last, {}, path, "logbook[-1]")
        stats[key] = int(last[key])
    else:
        _validate_saved_cv_calls(saved_stats, last, path, "runtime_state.fit_stats_")
    return {**data, "runtime_state": {**runtime, "fit_stats_": stats}}


def _validate_optimizer_resume_fields(data, path):
    missing = _OPTIMIZER_RESUME_FIELDS.difference(data)
    stats = data.get("fit_stats")
    required_counters = create_fit_stats()
    if isinstance(stats, Mapping):
        missing.update(f"fit_stats.{name}" for name in required_counters if name not in stats)
    if missing:
        raise ValueError(
            f"Invalid search checkpoint at {path}: missing resume fields "
            f"{', '.join(sorted(missing))}. Use a new checkpoint path to start a search."
        )
    if not isinstance(stats, Mapping):
        raise ValueError(f"Invalid search checkpoint at {path}: fit_stats must be a mapping.")
    _validate_saved_cv_calls(stats, _last_generation_record(data["logbook"]), path, "fit_stats")
    for name in required_counters:
        if name != "cross_validate_calls":
            _validate_counter(stats[name], path, f"fit_stats.{name}")


def load_search_checkpoint(callbacks, optimizer):
    """Read the first configured checkpoint, preserving legacy DEAP file handling."""
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            path = Path(callback.checkpoint_path)
            if not path.exists():
                continue
            data = callback.load()
            if not isinstance(data, Mapping) or not data:
                raise ValueError(
                    f"Invalid search checkpoint at {path}: expected a nonempty snapshot mapping. "
                    "Use a new checkpoint path to start a search."
                )
            if "optimizer_checkpoint_version" in data:
                if optimizer is None:
                    raise ValueError("A plugin checkpoint cannot resume a DEAP search")
                if data["optimizer_checkpoint_version"] != 1:
                    raise ValueError("Unsupported optimizer checkpoint version")
                resumable = data.get("resumable")
                if not isinstance(resumable, bool):
                    raise ValueError(
                        f"Invalid search checkpoint at {path}: missing or invalid resumable flag."
                    )
                if not resumable or not callable(resume_method(optimizer)):
                    raise ValueError(
                        "This progress snapshot does not support exact resume. Use a new checkpoint path to start a search."
                    )
                _validate_optimizer_resume_fields(data, path)
            else:
                if not isinstance(data.get("estimator_state"), Mapping):
                    raise ValueError(
                        f"Invalid search checkpoint at {path}: missing or invalid estimator state. "
                        "Use a new checkpoint path to start a search."
                    )
                if optimizer is not None:
                    raise ValueError(
                        "A historical DEAP checkpoint requires optimizer_backend='deap'"
                    )
                data = _restore_deap_fit_stats(data, path)
            if "enable_metadata_routing" in data:
                saved_routing = data["enable_metadata_routing"]
                routing = bool(get_config()["enable_metadata_routing"])
                if saved_routing is not routing:
                    raise ValueError(
                        f"Cannot resume search checkpoint at {path} with "
                        f"enable_metadata_routing={routing!r}. The checkpoint used "
                        f"enable_metadata_routing={saved_routing!r}. Restore that setting "
                        "or use a new checkpoint path."
                    )
            return data
    return None
