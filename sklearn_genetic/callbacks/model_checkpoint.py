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
                "local_search": getattr(estimator, "local_search", False),
                "local_search_top_k": getattr(estimator, "local_search_top_k", 1),
                "local_search_steps": getattr(estimator, "local_search_steps", 1),
                "local_search_radius": getattr(estimator, "local_search_radius", 0.1),
                "diversity_control": getattr(estimator, "diversity_control", False),
                "diversity_threshold": getattr(estimator, "diversity_threshold", 0.1),
                "diversity_stagnation_generations": getattr(
                    estimator, "diversity_stagnation_generations", 5
                ),
                "diversity_mutation_boost": getattr(estimator, "diversity_mutation_boost", 2.0),
                "random_immigrants_fraction": getattr(estimator, "random_immigrants_fraction", 0.1),
                "fitness_sharing": getattr(estimator, "fitness_sharing", False),
                "sharing_radius": getattr(estimator, "sharing_radius", 0.2),
                "sharing_alpha": getattr(estimator, "sharing_alpha", 1.0),
            }
            checkpoint_data = {"estimator_state": estimator_state, "logbook": deepcopy(logbook)}
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
