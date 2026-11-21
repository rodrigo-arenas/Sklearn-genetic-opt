import logging
import pickle
import random
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .._optimizer_checkpoint import resume_method
from .base import BaseCallback
from copy import deepcopy

logger = logging.getLogger(__name__)


class ModelCheckpoint(BaseCallback):
    def __init__(self, checkpoint_path, **dump_options):
        self.checkpoint_path = checkpoint_path
        self.dump_options = dump_options

    def on_step(self, record=None, logbook=None, estimator=None):
        temporary_path = None
        try:
            snapshot = getattr(estimator, "_optimizer_snapshot", None)
            if snapshot is None:
                checkpoint_data = self._legacy_snapshot(estimator, logbook)
            else:
                checkpoint_data = snapshot()
                if checkpoint_data is None:
                    return
            destination = Path(self.checkpoint_path)
            with NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
                temporary_path = Path(stream.name)
                pickle.dump(checkpoint_data, stream)
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

        logger.info("Checkpoint saved to %s", self.checkpoint_path)

    @staticmethod
    def _legacy_snapshot(estimator, logbook):
        estimator_state = estimator._checkpoint_state()
        # Runtime state is kept separate from ``estimator_state`` so the
        # latter stays constructor-compatible (it is consumed as
        # ``GASearchCV(**estimator_state)``). Restoring the fitness cache on
        # resume lets already-evaluated candidates be reused (see fit()).
        # ``fit_stats_`` is restored too so counters (cache hits, evaluated
        # candidates, ...) accumulate across a resume instead of resetting.
        #
        # ``candidate_logbook`` is ``estimator.logbook`` itself -- the
        # per-candidate log ("Contains the logs of every set of
        # hyperparameters fitted with its average scoring metric", see the
        # class docstring) that backs ``cv_results_``/``history``. It is
        # NOT the same object as the ``logbook`` argument this method
        # receives, which is the per-*generation* summary log the
        # algorithms module builds for its own bookkeeping and callbacks;
        # saving only that one under the legacy ``"logbook"`` key (kept
        # below for backward compatibility) silently dropped every
        # already-evaluated candidate on resume.
        runtime_state = {
            "elites": [
                (list(individual), individual.fitness.values)
                for individual in getattr(estimator, "_hof", ())
            ],
            "rng_state": (random.getstate(), np.random.get_state()),
            "fitness_cache": getattr(estimator, "fitness_cache", {}),
            "fit_stats_": deepcopy(getattr(estimator, "fit_stats_", None)),
            "candidate_logbook": deepcopy(getattr(estimator, "logbook", None)),
        }
        crossover_adapter = getattr(estimator, "crossover_adapter", None)
        mutation_adapter = getattr(estimator, "mutation_adapter", None)
        if crossover_adapter is not None and mutation_adapter is not None:
            runtime_state["adapter_state"] = {
                "crossover": crossover_adapter.state_dict(),
                "mutation": mutation_adapter.state_dict(),
            }
        return {
            "estimator_state": estimator_state,
            "logbook": deepcopy(logbook),
            "runtime_state": runtime_state,
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


def load_search_checkpoint(callbacks, optimizer):
    """Read the first configured checkpoint, preserving legacy DEAP file handling."""
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            if not Path(callback.checkpoint_path).exists():
                continue
            data = callback.load()
            if not data:
                break
            if "optimizer_checkpoint_version" in data:
                if optimizer is None:
                    raise ValueError("A plugin checkpoint cannot resume a DEAP search")
                if data["optimizer_checkpoint_version"] != 1:
                    raise ValueError("Unsupported optimizer checkpoint version")
                if not data["resumable"] or not callable(resume_method(optimizer)):
                    raise ValueError(
                        "This progress snapshot does not support exact resume. Use a new checkpoint path to start a search."
                    )
            elif optimizer is not None:
                raise ValueError("A historical DEAP checkpoint requires optimizer_backend='deap'")
            return data
    return None
