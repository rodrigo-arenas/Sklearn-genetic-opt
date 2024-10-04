import pickle
from .base import BaseCallback
from .loggers import LogbookSaver
from copy import deepcopy


class ModelCheckpoint(BaseCallback):
    def __init__(self, checkpoint_path, **dump_options):
        self.checkpoint_path = checkpoint_path
        self.dump_options = dump_options

    def on_step(self, record=None, logbook=None, estimator=None):
        try:
            if logbook is not None and len(logbook) > 0:
                logbook_saver = LogbookSaver(self.checkpoint_path, **self.dump_options)  # noqa
                logbook_saver.on_step(record, logbook, estimator)

            estimator_state = {
                "estimator": estimator.estimator,
                "cv": estimator.cv,
                "scoring": estimator.scoring,
                "population_size": estimator.population_size,
                "generations": estimator.generations,
                "crossover_probability": estimator.crossover_probability,
                "mutation_probability": estimator.mutation_probability,
                "param_grid": estimator.param_grid,
                "algorithm": estimator.algorithm,
                "param_grid": estimator.param_grid,
            }
            checkpoint_data = {"estimator_state": estimator_state, "logbook": deepcopy(logbook)}
            with open(self.checkpoint_path, "wb") as f:
                pickle.dump(checkpoint_data, f)
                print(f"Checkpoint save in {self.checkpoint_path}")

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
