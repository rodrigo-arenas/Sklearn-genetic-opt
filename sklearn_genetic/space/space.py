import numpy as np
from scipy import stats
import random

from .space_parameters import (
    IntegerDistributions,
    ContinuousDistributions,
    CategoricalDistributions,
)
from .base import BaseDimension


class Integer(BaseDimension):
    """class for hyperparameters search space of integer values"""

    def __init__(
        self,
        lower: int = None,
        upper: int = None,
        distribution: str = "uniform",
        random_state=None,
    ):
        """
        Parameters
        ----------
        lower : int, default=None
            Lower bound of the possible values of the hyperparameter.

        upper : int, default=None
            Upper bound of the possible values of the hyperparameter.

        distribution : str, default="uniform"
            Distribution to sample initial population and mutation values, currently only supports 'uniform'.
        random_state : int or None, RandomState instance, default=None
            Pseudo random number generator state used for random dimension sampling.
        """

        if not isinstance(lower, int):
            raise ValueError("lower bound must be an integer")

        if not isinstance(upper, int):
            raise ValueError("upper bound must be an integer")

        if lower > upper:
            raise ValueError("The upper bound can not be smaller that the lower bound")

        if distribution not in IntegerDistributions.list():
            raise ValueError(
                f"distribution must be one of {IntegerDistributions.list()}, got {distribution} instead"
            )

        self.lower = lower
        self.upper = upper
        self.distribution = distribution
        self.random_state = random_state
        self.rng = None if not self.random_state else np.random.default_rng(self.random_state)

        if self.distribution == IntegerDistributions.uniform.value:
            self.rvs = stats.randint.rvs

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(lower={self.lower}, upper={self.upper}, "
            f"distribution={self.distribution!r})"
        )

    def sample(self):
        """Sample a random value from the assigned distribution"""

        return self.rvs(self.lower, self.upper + 1, random_state=self.rng)


class Continuous(BaseDimension):
    """class for hyperparameters search space of real values"""

    def __init__(
        self,
        lower: float = None,
        upper: float = None,
        distribution: str = "uniform",
        random_state=None,
    ):
        """
        Parameters
        ----------
        lower : int, default=None
            Lower bound of the possible values of the hyperparameter.

        upper : int, default=None
            Upper bound of the possible values of the hyperparameter.

        distribution : {'uniform', 'log-uniform'}, default='uniform'
            Distribution to sample initial population and mutation values.
        random_state : int or None, RandomState instance, default=None
            Pseudo random number generator state used for random dimension sampling.
        """

        if not isinstance(lower, (int, float)):
            raise ValueError("lower bound must be an integer or float")

        if not isinstance(upper, (int, float)):
            raise ValueError("upper bound must be an integer or float")

        if lower > upper:
            raise ValueError("The upper bound can not be smaller that the lower bound")

        if distribution not in ContinuousDistributions.list():
            raise ValueError(
                f"distribution must be one of {ContinuousDistributions.list()}, got {distribution} instead"
            )

        self.lower = lower
        self.upper = upper
        self.distribution = distribution
        self.shifted_upper = self.upper
        self.random_state = random_state
        self.rng = None if not self.random_state else np.random.default_rng(self.random_state)

        if self.distribution == ContinuousDistributions.uniform.value:
            self.rvs = stats.uniform.rvs
            self.shifted_upper = self.upper - self.lower
        elif self.distribution == ContinuousDistributions.log_uniform.value:
            self.rvs = stats.loguniform.rvs

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(lower={self.lower}, upper={self.upper}, "
            f"distribution={self.distribution!r})"
        )

    def sample(self):
        """Sample a random value from the assigned distribution"""

        return self.rvs(self.lower, self.shifted_upper, random_state=self.rng)


class Categorical(BaseDimension):
    """class for hyperparameters search space of categorical values"""

    def __init__(
        self,
        choices: list = None,
        priors: list = None,
        distribution: str = "choice",
        random_state=None,
    ):
        """
        Parameters
        ----------
        choices: list, default=None
            List with all the possible values of the hyperparameter.

        priors: int, default=None
            List with the probability of sampling each element of the "choices", if not set gives equals probability.

        distribution: str, default='choice'
            Distribution to sample initial population and mutation values, currently only supports "choice".
        random_state : int or None, RandomState instance, default=None
            Pseudo random number generator state used for random dimension sampling.
        """

        if not choices or not isinstance(choices, list):
            raise ValueError("choices must be a non empty list")

        if priors is None:
            self.priors = priors
        elif sum(priors) != 1:
            raise ValueError(
                f"The sum of the probabilities in the priors must be one, got {sum(priors)} instead"
            )
        elif not len(priors) == len(choices):
            raise ValueError("priors and choices must have same size")
        else:
            self.priors = priors

        if distribution not in CategoricalDistributions.list():
            raise ValueError(
                f"distribution must be one of {CategoricalDistributions.list()}, got {distribution} instead"
            )

        self.choices = choices
        self.distribution = distribution
        self.random_state = random_state
        random.seed(random_state)
        self.rng = None if not self.random_state else np.random.default_rng(self.random_state)

        if self.distribution == CategoricalDistributions.choice.value:
            self.rvs = self.rng.choice if self.rng else random.choice

    def __repr__(self):
        if self.priors is None:
            return f"{self.__class__.__name__}(choices={self.choices!r})"

        return f"{self.__class__.__name__}(choices={self.choices!r}, priors={self.priors!r})"

    def sample(self):
        """Sample a random value from the assigned distribution"""

        return self.rvs(self.choices)


def check_space(param_grid: dict = None):
    """Validate that ``param_grid`` is a non-empty mapping of space objects.

    Parameters
    ----------
    param_grid: dict, default=None
        Dictionary of the form {"hyperparameter_name": :obj:`~sklearn_genetic.space`}

    Raises
    ------
    ValueError
        If ``param_grid`` is empty, or if any value is not an instance of
        :class:`~sklearn_genetic.space.Integer`,
        :class:`~sklearn_genetic.space.Categorical` or
        :class:`~sklearn_genetic.space.Continuous`. The message names the
        offending key and the type that was passed instead.
    """
    if not param_grid:
        raise ValueError("param_grid can not be empty")

    # Make sure that each of the param_grid values are defined using one of the available Space objects
    for key, value in param_grid.items():
        if not isinstance(value, BaseDimension):
            raise ValueError(
                f"Invalid param_grid entry for '{key}': expected a space object "
                f"(Continuous, Integer, or Categorical), got "
                f"{type(value).__name__} instead.\n"
                f"Example:\n"
                f"    from sklearn_genetic.space import Categorical\n"
                f'    param_grid = {{"{key}": Categorical([...])}}'
            )


class Space(object):
    """Search space for all the models hyperparameters"""

    def __init__(self, param_grid: dict = None):
        """
        Parameters
        ----------
        param_grid: dict, default=None
            Grid with the parameters to tune, expects keys a valid name
            of hyperparameter based on the estimator selected and as values
            one of :class:`~sklearn_genetic.space.Integer` ,
            :class:`~sklearn_genetic.space.Categorical`
            :class:`~sklearn_genetic.space.Continuous` classes
        """
        check_space(param_grid)

        self.param_grid = param_grid

    def sample_warm_start(self, warm_start_values: dict):
        """
        Sample a predefined configuration (warm-start) or fill in random values if missing.

        Parameters
        ----------
        warm_start_values: dict
            Predefined configuration values for hyperparameters.

        Returns
        -------
        A dictionary containing sampled values for each hyperparameter.

        Raises
        ------
        ValueError
            If ``warm_start_values`` is not a dict, contains a hyperparameter
            name that is not in the search space (e.g. a misspelled key), or
            assigns a value that falls outside its dimension. Missing keys are
            allowed and filled by sampling.
        """
        self._validate_warm_start_config(warm_start_values)

        sampled_params = {}
        for param, dimension in self.param_grid.items():
            if param in warm_start_values:
                sampled_params[param] = warm_start_values[param]
            else:
                sampled_params[param] = dimension.sample()
        return sampled_params

    def _validate_warm_start_config(self, warm_start_values):
        """Validate a single warm-start config against the search space."""
        if not isinstance(warm_start_values, dict):
            raise ValueError(
                "Each warm_start_configs entry must be a dict mapping "
                "hyperparameter names to values, got "
                f"{type(warm_start_values).__name__} instead."
            )

        unknown = [name for name in warm_start_values if name not in self.param_grid]
        if unknown:
            raise ValueError(
                f"warm_start_configs contains unknown hyperparameter(s) "
                f"{sorted(unknown)} that are not in the search space. Valid "
                f"names are {sorted(self.param_grid)}. (Check for typos, e.g. "
                f"'max_depths' instead of 'max_depth'.)"
            )

        # Provided values must fall inside their dimension. Missing keys are
        # intentionally allowed (they are filled by sampling).
        for name, value in warm_start_values.items():
            dimension = self.param_grid[name]
            if not self.value_in_dimension(dimension, value):
                raise ValueError(
                    f"warm_start_configs value {value!r} for '{name}' is outside "
                    f"its search space {dimension!r}."
                )

    @staticmethod
    def value_in_dimension(dimension, value) -> bool:
        """Return ``True`` if ``value`` is a valid sample for ``dimension``.

        Single source of truth shared by warm-start validation and population
        initialization (``population._is_dimension_value_valid``) so the two
        cannot drift. NumPy scalars are accepted alongside Python numbers.
        """
        if isinstance(dimension, Integer):
            return (
                isinstance(value, (int, np.integer)) and dimension.lower <= value <= dimension.upper
            )
        if isinstance(dimension, Continuous):
            return (
                isinstance(value, (int, float, np.integer, np.floating))
                and dimension.lower <= value <= dimension.upper
            )
        if isinstance(dimension, Categorical):
            return value in dimension.choices
        return False

    @property
    def dimensions(self):
        """

        Returns
        -------
        The number of hyperparameters defined in the param_grid

        """
        return len(self.param_grid)

    @property
    def parameters(self):
        """

        Returns
        -------
        A list with all the names of the hyperparametes in the param_Grid
        """
        return list(self.param_grid.keys())

    def __len__(self):
        return self.dimensions

    def __getitem__(self, index):
        return self.param_grid[index]
