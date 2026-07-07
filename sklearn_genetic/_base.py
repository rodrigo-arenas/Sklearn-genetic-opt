import pickle

from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from .algorithms import algorithms_factory
from .parameters import Algorithms


def reset_adapters(estimator):
    estimator.crossover_adapter.reset()
    estimator.mutation_adapter.reset()


def history_record(history, index):
    return {key: values[index] for key, values in history.items()}


class GeneticEstimatorMixin:
    _volatile_pickle_attrs = {"toolbox", "_stats", "_pop", "_hof", "hof"}

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
    }

    def _serializable_state(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in self._volatile_pickle_attrs
        }

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
        return state

    def save(self, filepath):
        """Save the current state of the estimator instance to a file."""
        class_name = self.__class__.__name__
        try:
            checkpoint_data = {"estimator_state": self._serializable_state(), "logbook": None}
            if hasattr(self, "logbook"):
                checkpoint_data["logbook"] = self.logbook

            serialized_checkpoint = pickle.dumps(checkpoint_data)
            with open(filepath, "wb") as f:
                f.write(serialized_checkpoint)
            print(f"{class_name} model successfully saved to {filepath}")
        except Exception as e:
            print(f"Error saving {class_name}: {e}")

    def load(self, filepath):
        """Load an estimator instance from a file."""
        class_name = self.__class__.__name__
        try:
            with open(filepath, "rb") as f:
                checkpoint_data = pickle.load(f)
                for key, value in checkpoint_data["estimator_state"].items():
                    setattr(self, key, value)
                self.logbook = checkpoint_data["logbook"]
            print(f"{class_name} model successfully loaded from {filepath}")
        except Exception as e:
            print(f"Error loading {class_name}: {e}")

    def _select_algorithm(self, pop, stats, hof):
        selected_algorithm = algorithms_factory.get(self.algorithm, None)
        if selected_algorithm:
            pop, log, gen = selected_algorithm(
                pop,
                self.toolbox,
                mu=self.population_size,
                lambda_=2 * self.population_size,
                cxpb=self.crossover_adapter,
                stats=stats,
                mutpb=self.mutation_adapter,
                ngen=self.generations,
                halloffame=hof,
                callbacks=self.callbacks,
                verbose=self.verbose,
                estimator=self,
            )
        else:
            raise ValueError(
                f"The algorithm {self.algorithm} is not supported, "
                f"please select one from {Algorithms.list()}"
            )

        return pop, log, gen

    def _run_search(self, evaluate_candidates):
        pass  # noqa

    @property
    def _fitted(self):
        try:
            check_is_fitted(self.estimator)
            is_fitted = True
        except Exception:
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
        if self.n < self._n_iterations + 1:
            result = self.__getitem__(self.n)
            self.n += 1
            return result
        else:
            raise StopIteration  # pragma: no cover

    def __len__(self):
        return self._n_iterations
