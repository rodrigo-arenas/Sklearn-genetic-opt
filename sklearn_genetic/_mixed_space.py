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
                result.append(encode_real(dim, value))
        return result

    def decode(self, values):
        if self.problem.task == "features":
            bits = np.array(values, dtype=np.uint8)
            count = int(bits.sum())
            cap = self.problem.max_features or len(bits)
            if not count or count > cap:
                seed = zlib.crc32(bits.tobytes(), self.problem.seed)
                rng = np.random.default_rng(seed)
                if not count:
                    bits[rng.integers(len(bits))] = 1
                else:
                    bits[rng.choice(np.flatnonzero(bits), count - cap, replace=False)] = 0
            return bits.tolist()
        row = list(self.problem.initial_population[0])
        for i, value in zip(self.positions, values, strict=True):
            dim = self.problem.dimensions[i]
            if isinstance(dim, Categorical):
                row[i] = dim.choices[int(value)]
            elif isinstance(dim, Integer):
                row[i] = dim.lower + int(value)
            else:
                row[i] = decode_real(dim, value)
        return row


def encode_real(dim, value):
    if dim.lower == dim.upper:
        return 0.5
    if dim.distribution == "log-uniform" and np.log(dim.upper) != np.log(dim.lower):
        return (np.log(value) - np.log(dim.lower)) / (np.log(dim.upper) - np.log(dim.lower))
    return (value - dim.lower) / (dim.upper - dim.lower)


def decode_real(dim, value):
    if dim.lower == dim.upper:
        return float(dim.lower)
    if dim.distribution == "log-uniform" and np.log(dim.upper) != np.log(dim.lower):
        result = np.exp(np.log(dim.lower) + value * (np.log(dim.upper) - np.log(dim.lower)))
    else:
        result = dim.lower + value * (dim.upper - dim.lower)
    return float(np.clip(result, dim.lower, dim.upper))
