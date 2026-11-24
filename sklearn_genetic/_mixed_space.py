"""Translate external optimizer coordinates to the estimator's original values."""

import zlib

import numpy as np

from .space import Categorical, Continuous, Integer


class MixedSpace:
    """Keep constants out of solvers and represent categories by their positions."""

    def __init__(self, problem):
        self.problem = problem
        self.axes = []
        self.positions = []
        self._real_intervals = {}
        if problem.task == "features":
            self.axes = [("bit", 1)] * len(problem.initial_population[0])
            return
        for index, (name, dim) in enumerate(zip(problem.parameter_names, problem.dimensions)):
            if isinstance(dim, Categorical) and dim.priors is None:
                axis = ("choice", len(dim.choices) - 1)
            elif isinstance(dim, Integer) and dim.upper - dim.lower < 2**52:
                axis = ("int", dim.upper - dim.lower)
            elif isinstance(dim, Continuous) and np.isfinite(float(dim.upper) - float(dim.lower)):
                axis = ("real", int(dim.upper != dim.lower))
            else:
                raise ValueError(
                    f"Unsupported dimension {name!r}: use unweighted categories, "
                    "integer spans below 2**52, or finite real intervals"
                )
            if axis[1]:
                self.positions.append(index)
                self.axes.append(axis)
                if type(dim) is Continuous:
                    self._real_intervals[index] = real_coordinate_interval(dim)

    def encode(self, row):
        if self.problem.task == "features":
            return list(map(int, row))
        result = []
        for i in self.positions:
            dim, value = self.problem.dimensions[i], row[i]
            if isinstance(dim, Categorical):
                result.append(dim._choice_index(value))
            elif isinstance(dim, Integer):
                result.append(int(value) - dim.lower)
            else:
                result.append(encode_real(dim, value, interval=self._real_intervals.get(i)))
        return result

    def decode(self, values):
        if self.problem.task == "features":
            bits = np.array(values, dtype=np.uint8)
            count = int(bits.sum())
            cap = self.problem.max_features or len(bits)
            if not count or count > cap:
                self._repair_mask(bits, count, cap)
            return bits.tolist()
        row = list(self.problem.initial_population[0])
        for i, value in zip(self.positions, values, strict=True):
            dim = self.problem.dimensions[i]
            if isinstance(dim, Categorical):
                row[i] = dim.choices[int(value)]
            elif isinstance(dim, Integer):
                row[i] = dim.lower + int(value)
            else:
                row[i] = decode_real(dim, value, interval=self._real_intervals.get(i))
        return row

    def decode_masks(self, values):
        bits = np.array(values, dtype=np.uint8)
        counts = bits.sum(axis=1)
        cap = self.problem.max_features or bits.shape[1]
        for index in np.flatnonzero((counts == 0) | (counts > cap)):
            self._repair_mask(bits[index], int(counts[index]), cap)
        return bits

    def _repair_mask(self, bits, count, cap):
        seed = zlib.crc32(bits.tobytes(), self.problem.seed)
        rng = np.random.default_rng(seed)
        if not count:
            bits[rng.integers(len(bits))] = 1
        else:
            bits[rng.choice(np.flatnonzero(bits), count - cap, replace=False)] = 0


def encode_real(dim, value, *, interval=None):
    lower, upper, log_lower, span = real_coordinate_interval(dim) if interval is None else interval
    if lower == upper:
        return 0.5
    if log_lower is not None:
        return (np.log(float(value)) - log_lower) / span
    return (value - lower) / span


def real_coordinate_interval(dim):
    """Return the representable bounds, optional log origin and coordinate span."""
    lower, upper = dim._sampling_bounds()
    if lower != upper and dim.distribution == "log-uniform":
        log_lower, log_upper = np.log(lower), np.log(upper)
        if log_lower != log_upper:
            return lower, upper, log_lower, log_upper - log_lower
    return lower, upper, None, upper - lower


def decode_real(dim, value, *, interval=None):
    lower, upper, log_lower, span = real_coordinate_interval(dim) if interval is None else interval
    if lower == upper:
        return lower
    if log_lower is not None:
        return dim._clip_scalar(np.exp(log_lower + value * span))
    return dim._clip_scalar(lower + value * span)
