import pickle
from .base import BaseCallback
from .loggers import LogbookSaver
from copy import deepcopy

# Single source of truth for the constructor-compatible state persisted in a
# checkpoint, so the key set cannot silently drift or duplicate.
#
# Core keys exist on both GASearchCV and GAFeatureSelectionCV and are read
# directly. Optional keys map to the default used when the estimator does not
# define them.
_CHECKPOINT_CORE_KEYS = (
    "estimator",
    "cv",
    "scoring",
    "population_size",
    "generations",
    "crossover_probability",
    "mutation_probability",
    "algorithm",
)
_CHECKPOINT_OPTIONAL_DEFAULTS = {
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
}


def _build_estimator_state(estimator):
    """Build the constructor-compatible checkpoint state for an estimator.

    Works for both GASearchCV and GAFeatureSelectionCV: ``param_grid`` is only
    included when the estimator actually defines it (GAFeatureSelectionCV does
    not), so checkpointing a feature selector no longer raises AttributeError.
    """
    state = {key: getattr(estimator, key) for key in _CHECKPOINT_CORE_KEYS}
    if hasattr(estimator, "param_grid"):
        state["param_grid"] = estimator.param_grid
    for key, default in _CHECKPOINT_OPTIONAL_DEFAULTS.items():
        state[key] = getattr(estimator, key, default)
    return state


class ModelCheckpoint(BaseCallback):
    def __init__(self, checkpoint_path, **dump_options):
        self.checkpoint_path = checkpoint_path
        self.dump_options = dump_options

    def on_step(self, record=None, logbook=None, estimator=None):
        try:
            if logbook is not None and len(logbook) > 0:
                logbook_saver = LogbookSaver(self.checkpoint_path, **self.dump_options)  # noqa
                logbook_saver.on_step(record, logbook, estimator)

            estimator_state = _build_estimator_state(estimator)
            # Runtime state is kept separate from ``estimator_state`` so the
            # latter stays constructor-compatible (it is consumed as
            # ``GASearchCV(**estimator_state)``). Restoring the fitness cache on
            # resume lets already-evaluated candidates be reused (see fit()).
            runtime_state = {"fitness_cache": getattr(estimator, "fitness_cache", {})}
            checkpoint_data = {
                "estimator_state": estimator_state,
                "logbook": deepcopy(logbook),
                "runtime_state": runtime_state,
            }
            with open(self.checkpoint_path, "wb") as f:
                pickle.dump(checkpoint_data, f)
                print(f"Checkpoint saved to {self.checkpoint_path}")

        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    def load(self):
        """Load the model state from the checkpoint file."""
        try:
            with open(self.checkpoint_path, "rb") as f:
                checkpoint_data = pickle.load(f)
                return checkpoint_data
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None
