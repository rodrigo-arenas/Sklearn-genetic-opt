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

    def sample(self):
        """Sample a random value from the assigned distribution"""

        return self.rvs(self.choices)


def check_space(param_grid: dict = None):
    """

    Parameters
    ----------
    param_grid: dict, default=None
        Dictionary with the for {"hyperparameter_name": :obj:`~sklearn_genetic.space`}

    Returns
    -------
        Raises a Value Error if the dictionary does not have valid space instances
    """
    if not param_grid:
        raise ValueError(f"param_grid can not be empty")

    # Make sure that each of the param_grid values are defined using one of the available Space objects
    for key, value in param_grid.items():
        if not isinstance(value, BaseDimension):
            raise ValueError(
                f"{key} must be a valid instance of Integer, Categorical or Continuous classes"
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
        """
        sampled_params = {}
        for param, dimension in self.param_grid.items():
            if param in warm_start_values:
                sampled_params[param] = warm_start_values[param]
            else:
                sampled_params[param] = dimension.sample()
        return sampled_params

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
