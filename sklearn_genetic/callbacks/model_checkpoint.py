import logging
import pickle

from .base import BaseCallback
from .loggers import LogbookSaver
from copy import deepcopy

logger = logging.getLogger(__name__)


class ModelCheckpoint(BaseCallback):
    def __init__(self, checkpoint_path, **dump_options):
        self.checkpoint_path = checkpoint_path
        self.dump_options = dump_options

    def on_step(self, record=None, logbook=None, estimator=None):
        if logbook is not None and len(logbook) > 0:
            logbook_saver = LogbookSaver(self.checkpoint_path, **self.dump_options)  # noqa
            logbook_saver.on_step(record, logbook, estimator)

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
            "fitness_cache": getattr(estimator, "fitness_cache", {}),
            "fit_stats_": deepcopy(getattr(estimator, "fit_stats_", None)),
            "candidate_logbook": deepcopy(getattr(estimator, "logbook", None)),
        }
        checkpoint_data = {
            "estimator_state": estimator_state,
            "logbook": deepcopy(logbook),
            "runtime_state": runtime_state,
        }
        try:
            with open(self.checkpoint_path, "wb") as f:
                pickle.dump(checkpoint_data, f)
        except (OSError, pickle.PicklingError) as e:
            logger.warning("Error saving checkpoint to %s: %s", self.checkpoint_path, e)
            return

        logger.info("Checkpoint saved to %s", self.checkpoint_path)

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
