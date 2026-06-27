from collections.abc import Mapping

import numpy as np

from .base import BaseDimension
from .space import Categorical, Continuous, Integer


def from_sklearn_space(param_distributions):
    """Convert sklearn/scipy-style parameter distributions to genetic search spaces.

    Parameters
    ----------
    param_distributions : dict
        Dictionary using the style accepted by ``RandomizedSearchCV``:
        list-like values for categorical choices and scipy frozen distributions
        for sampled numeric values.

    Returns
    -------
    dict
        Dictionary whose values are ``Integer``, ``Continuous``, or ``Categorical``
        dimensions.
    """
    if not isinstance(param_distributions, Mapping) or not param_distributions:
        raise ValueError("param_distributions must be a non-empty mapping")

    return {
        parameter: _convert_dimension(parameter, distribution)
        for parameter, distribution in param_distributions.items()
    }


def _convert_dimension(parameter, distribution):
    if isinstance(distribution, BaseDimension):
        return distribution

    if _is_list_like(distribution):
        choices = list(distribution)
        if len(choices) == 0:
            raise ValueError(f"{parameter} has no categorical choices")
        return Categorical(choices)

    scipy_dimension = _convert_scipy_distribution(parameter, distribution)
    if scipy_dimension is not None:
        return scipy_dimension

    raise ValueError(
        f"{parameter} must be a list-like value, scipy frozen distribution, "
        "or sklearn_genetic space dimension"
    )


def _is_list_like(value):
    return isinstance(value, (list, tuple, set, range, np.ndarray))


def _convert_scipy_distribution(parameter, distribution):
    scipy_dist = getattr(distribution, "dist", None)
    if scipy_dist is None:
        return None

    name = getattr(scipy_dist, "name", None)
    args = getattr(distribution, "args", ())
    kwds = getattr(distribution, "kwds", {})

    if name == "randint":
        low, high = _bounds_from_args(parameter, args, kwds, ("low", "high"))
        return Integer(int(low), int(high) - 1)

    if name == "uniform":
        loc = kwds.get("loc", args[0] if len(args) >= 1 else 0)
        scale = kwds.get("scale", args[1] if len(args) >= 2 else 1)
        return Continuous(float(loc), float(loc) + float(scale))

    if name in {"loguniform", "reciprocal"}:
        lower, upper = _bounds_from_args(parameter, args, kwds, ("a", "b"))
        return Continuous(float(lower), float(upper), distribution="log-uniform")

    raise ValueError(
        f"{parameter} uses scipy.stats.{name}, which can not be converted "
        f"automatically. Supported scipy distributions are randint, uniform, "
        f"loguniform, and reciprocal. For anything else, define the dimension "
        f"manually, e.g. Continuous(low, high) or "
        f"Continuous(low, high, distribution='log-uniform') for '{parameter}'."
    )


def _bounds_from_args(parameter, args, kwds, names):
    if all(name in kwds for name in names):
        return kwds[names[0]], kwds[names[1]]

    if len(args) >= 2:
        return args[0], args[1]

    raise ValueError(f"{parameter} distribution must define {names[0]} and {names[1]} bounds")
