from dataclasses import dataclass
from functools import cached_property
from math import inf, isfinite, nextafter
import random
from struct import pack
from sys import float_info
from typing import Any, List, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator
from scipy import stats
from sklearn.utils import check_random_state

from .space_parameters import (
    IntegerDistributions,
    ContinuousDistributions,
    CategoricalDistributions,
)
from .base import BaseDimension
from ..evaluation import candidate_parameters
from ..utils.tools import _hashable_key

_NAN_CATEGORY_TYPES = frozenset((float, np.float16, np.float32, np.float64))


@dataclass(frozen=True)
class _FloatingNaN:
    kind: type
    bits: bytes


def _is_floating_nan(value):
    return type(value) in _NAN_CATEGORY_TYPES and value != value


def _nan_category_key(value):
    if not _is_floating_nan(value):
        return None
    kind = type(value)
    return _FloatingNaN(kind, pack(">d", value) if kind is float else value.tobytes())


def _definite_equality(left, right):
    result = left == right
    try:
        return bool(result)
    except (TypeError, ValueError):
        # Some comparisons report an unknown or ambiguous truth value.
        return False


def _same_config(left, right, *, strict_types=False):
    """Compare estimator constructor trees without requiring picklable classes."""
    if left is right:
        return True
    if strict_types and type(left) is not type(right):
        return False
    if isinstance(left, BaseEstimator):
        return type(left) is type(right) and _same_config(
            left.get_params(deep=False), right.get_params(deep=False), strict_types=strict_types
        )
    if isinstance(left, dict):
        if type(left) is not type(right):
            return False
        if tuple(left) != tuple(right):
            return False
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(
                _same_config(value, right[key], strict_types=strict_types)
                for key, value in left.items()
            )
        )
    if isinstance(left, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_same_config(a, b, strict_types=strict_types) for a, b in zip(left, right))
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.shape == right.shape
            and left.dtype == right.dtype
        ):
            return False
        if left.dtype.kind == "O":
            # NumPy delegates object equality to Python, which can conflate
            # types or raise on nested arrays. Use the same rules as containers.
            return all(
                _same_config(a, b, strict_types=strict_types) for a, b in zip(left.flat, right.flat)
            )
        return np.array_equal(
            left, right, equal_nan=left.dtype.kind in "fc" and right.dtype.kind in "fc"
        )
    if isinstance(left, np.random.RandomState):
        return isinstance(right, np.random.RandomState) and _same_config(
            left.get_state(), right.get_state()
        )
    if isinstance(left, np.random.Generator):
        return isinstance(right, np.random.Generator) and _same_config(
            left.bit_generator.state, right.bit_generator.state
        )
    return _definite_equality(left, right)


def _category_types(value):
    """Describe types inside containers whose values may compare equal in Python."""
    if isinstance(value, dict):
        return type(value), tuple(
            (_category_types(key), _category_types(item)) for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return type(value), tuple(map(_category_types, value))
    if isinstance(value, np.ndarray) and value.dtype.hasobject:
        return type(value), _category_types(value.tolist())
    if isinstance(value, (set, frozenset)):
        return type(value), frozenset(
            (_hashable_key(item), _category_types(item)) for item in value
        )
    return type(value)


class Integer(BaseDimension):
    """class for hyperparameters search space of integer values"""

    def __init__(
        self,
        lower: Optional[int] = None,
        upper: Optional[int] = None,
        distribution: str = "uniform",
        random_state: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
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
        self.rng = (
            np.random.default_rng(self.random_state) if self.random_state is not None else None
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(lower={self.lower}, upper={self.upper}, "
            f"distribution={self.distribution!r})"
        )

    def sample(self) -> int:
        """Sample a random value from the assigned distribution"""

        if self.lower == self.upper:
            return int(self.lower)
        # Match SciPy's int64 draws even on platforms where randint defaults to int32.
        if self.rng is None:
            return int(np.random.randint(self.lower, self.upper + 1, dtype=np.int64))
        return int(self.rng.integers(self.lower, self.upper + 1, dtype=np.int64))

    def _round_and_clip(self, value):
        if type(value) is int and type(self.lower) is int and type(self.upper) is int:
            if value < self.lower:
                value = self.lower
            return self.upper if value > self.upper else value

        if isinstance(value, (int, np.integer)):
            rounded = int(value)
        else:
            rounded = round(float(value))
        return int(min(self.upper, max(self.lower, rounded)))


class Continuous(BaseDimension):
    """class for hyperparameters search space of real values"""

    def __init__(
        self,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        distribution: str = "uniform",
        random_state: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
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

        if distribution not in ContinuousDistributions.list():
            raise ValueError(
                f"distribution must be one of {ContinuousDistributions.list()}, got {distribution} instead"
            )

        self.lower = lower
        self.upper = upper
        self.distribution = distribution
        self._validate_bounds()
        self.random_state = random_state
        self.rng = (
            np.random.default_rng(self.random_state) if self.random_state is not None else None
        )

        if self.distribution == ContinuousDistributions.uniform.value:
            self.rvs = stats.uniform.rvs
        elif self.distribution == ContinuousDistributions.log_uniform.value:
            self.rvs = stats.loguniform.rvs

    def _validate_bounds(self):
        lower, upper = self.lower, self.upper
        for name, bound in (("lower", lower), ("upper", upper)):
            if not isinstance(bound, (int, float)):
                raise ValueError(f"{name} bound must be an integer or float")
            if not -float_info.max <= bound <= float_info.max:
                if lower == upper:
                    raise ValueError(
                        "A fixed continuous bound must be a finite, exactly representable float"
                    )
                raise ValueError(f"{name} bound must be within the finite float range")
        if lower > upper:
            raise ValueError("The upper bound can not be smaller that the lower bound")
        if lower == upper and float(lower) != lower:
            raise ValueError(
                "A fixed continuous bound must be a finite, exactly representable float"
            )
        sample_lower, sample_upper = self._sampling_bounds()
        if sample_lower > sample_upper:
            raise ValueError("A continuous interval must contain a representable float")
        if self.distribution == "uniform" and not isfinite(sample_upper - sample_lower):
            raise ValueError("A uniform continuous interval must have a finite span")
        if self.distribution == "log-uniform" and lower <= 0:
            raise ValueError(
                f"lower bound must be greater than zero for the log-uniform distribution, got {lower} instead"
            )

    def _sampling_bounds(self):
        lower, upper = float(self.lower), float(self.upper)
        # An integer endpoint can lie between two representable floats.
        if lower < self.lower:
            lower = nextafter(lower, inf)
        if upper > self.upper:
            upper = nextafter(upper, -inf)
        return lower, upper

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(lower={self.lower}, upper={self.upper}, "
            f"distribution={self.distribution!r})"
        )

    def _clip_scalar(self, value):
        if self.lower < value < self.upper:
            return float(value)
        # Keep NumPy's boundary semantics, including signed zero.
        lower, upper = self._sampling_bounds()
        return float(np.clip(value, lower, upper))

    def sample(self) -> float:
        """Sample a random value from the assigned distribution"""

        lower, upper = self._sampling_bounds()
        if lower == upper:
            return lower
        rng = self.rng if self.rng is not None else check_random_state(None)
        if self.distribution == "uniform":
            sample = self.rvs(lower, upper - lower, random_state=rng)
        elif np.log(lower) == np.log(upper):
            # A logarithmic transform can lose the width of a valid interval.
            sample = stats.uniform.rvs(lower, upper - lower, random_state=rng)
        else:
            sample = self.rvs(lower, upper, random_state=rng)
        return sample if lower <= sample <= upper else self._clip_scalar(sample)


class Categorical(BaseDimension):
    """class for hyperparameters search space of categorical values"""

    def __init__(
        self,
        choices: Optional[List[Any]] = None,
        priors: Optional[List[float]] = None,
        distribution: str = "choice",
        random_state: Optional[int] = None,
    ) -> None:
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
        elif any(p < 0 for p in priors):
            raise ValueError(f"priors must be non-negative, got {priors} instead")
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
        self.rng = (
            np.random.default_rng(self.random_state) if self.random_state is not None else None
        )

    def __repr__(self) -> str:
        if self.priors is None:
            return f"{self.__class__.__name__}(choices={self.choices!r})"

        return f"{self.__class__.__name__}(choices={self.choices!r}, priors={self.priors!r})"

    def _choice_index(self, value):
        """Resolve cloned estimator and container values to their original category."""
        if type(value) in _NAN_CATEGORY_TYPES and value != value:
            nan_key = _nan_category_key(value)
            # Python's pickle creates a new float object for each NaN occurrence.
            return next(
                (
                    index
                    for index, choice in enumerate(self.choices)
                    if nan_key == _nan_category_key(choice)
                ),
                None,
            )
        for index, choice in enumerate(self.choices):
            if choice is value:
                return index
        if isinstance(value, BaseEstimator):
            return self._estimator_choice_index(value)
        compatible = None
        for index, choice in enumerate(self.choices):
            if _same_config(value, choice, strict_types=True) and (
                _category_types(value) == _category_types(choice)
            ):
                return index
            if compatible is None and _same_config(value, choice):
                compatible = index
        return compatible

    def _estimator_choice_index(self, estimator):
        # Reuse shallow constructor parameters during this lookup. Read them
        # again on later calls so changes to choices are visible.
        kind = type(estimator)
        params = None
        compatible = None
        for index, choice in enumerate(self.choices):
            if type(choice) is not kind:
                continue
            if params is None:
                params = estimator.get_params(deep=False)
            choice_params = choice.get_params(deep=False)
            if _same_config(params, choice_params, strict_types=True) and (
                _category_types(estimator) == _category_types(choice)
            ):
                return index
            if compatible is None and _same_config(params, choice_params):
                compatible = index
        return compatible

    def sample(self) -> Any:
        """Sample a random value from the assigned distribution"""

        n_choices = len(self.choices)
        if self.rng is not None:
            index = self.rng.choice(n_choices, p=self.priors)
        elif self.priors is None:
            index = random.randrange(n_choices)
        else:
            index = random.choices(range(n_choices), weights=self.priors, k=1)[0]
        # Index into the original list so the exact Python object (and its
        # type) is returned, instead of a value numpy has coerced to one of
        # its own scalar types (e.g. np.True_, np.int64) -- see #324.
        return self.choices[int(index)]


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
        if isinstance(value, Continuous):
            try:
                value._validate_bounds()
            except ValueError as error:
                raise ValueError(f"Invalid param_grid entry for '{key}': {error}") from error


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

    @cached_property
    def _typed_categories(self):
        # Keep archived cache keys for dimensions whose legacy keys are unambiguous.
        typed = set()
        for name, dimension in self.param_grid.items():
            if not isinstance(dimension, Categorical):
                continue
            observed = {}
            for choice in dimension.choices:
                key = _hashable_key(choice)
                previous = observed.setdefault(key, choice)
                if previous is not choice and _category_types(previous) != _category_types(choice):
                    typed.add(name)
                    break
        return typed

    def _category_key(self, name, value):
        if name in self._nan_categories:
            nan_key = _nan_category_key(value)
            if nan_key is not None:
                return nan_key
        key = _hashable_key(value)
        return (key, _category_types(value)) if name in self._typed_categories else key

    @cached_property
    def _nan_categories(self):
        return {
            name
            for name, dimension in self.param_grid.items()
            if isinstance(dimension, Categorical) and any(map(_is_floating_nan, dimension.choices))
        }

    @cached_property
    def _keyed_categories(self):
        return self._typed_categories | self._nan_categories

    def _candidate_key(self, values):
        params = {}
        for (name, dimension), value in zip(self.param_grid.items(), values, strict=True):
            if isinstance(dimension, Categorical):
                choice_index = dimension._choice_index(value)
                if choice_index is not None:
                    value = dimension.choices[choice_index]
                if name in self._keyed_categories:
                    value = self._category_key(name, value)
            params[name] = value
        return _hashable_key(tuple(sorted(params.items())))

    def _population_key(self, values):
        if self._keyed_categories:
            return self._candidate_key(values)
        return _hashable_key(tuple(values))

    def _keyed_population_values(self, values):
        return tuple(
            self._category_key(name, value) if name in self._keyed_categories else value
            for name, value in zip(self.param_grid, values)
        )

    def _rekey_nan_cache(self, cache):
        if not cache or not self._nan_categories:
            return cache
        # Historical snapshots used non-reflexive floating-point values as keys.
        # Retain the first recorded score when old aliases now resolve to one key.
        restored = {}
        for result in cache.values():
            params = candidate_parameters(result["current_generation_params"])
            restored.setdefault(
                self._candidate_key(params[name] for name in self.param_grid), result
            )
        return restored

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
    def _repair_value(dimension, value):
        if isinstance(dimension, Integer):
            if value is None:
                return dimension.sample()
            return dimension._round_and_clip(value)
        if isinstance(dimension, Continuous):
            if value is None:
                return dimension.sample()
            return dimension._clip_scalar(float(value))
        if isinstance(dimension, Categorical):
            index = dimension._choice_index(value)
            return dimension.choices[index] if index is not None else dimension.sample()
        return value

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
            return dimension._choice_index(value) is not None
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
