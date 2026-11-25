"""Stable reductions for extreme finite cross-validation and fitness scores."""

from functools import cache
from math import isfinite
from statistics import mean

import numpy as np


def is_finite_score(value):
    """Check finiteness without narrowing NumPy extended scores."""
    if isinstance(value, np.longdouble):
        return bool(np.isfinite(value))
    return value is not None and isfinite(value)


@cache
def score_scale_limits(dtype):
    limits = np.finfo(dtype)
    # Squaring adjacent differences or summing squared deviations can leave
    # the dtype's range even when the standard deviation fits in that dtype.
    return limits.max, np.sqrt(limits.tiny) / limits.eps, np.sqrt(limits.max) / 2


def _restore_scaled_mean(scaled, exponent):
    return np.ldexp(np.clip(np.mean(scaled), scaled.min(), scaled.max()), exponent)


def score_mean(scores, *, scale=None):
    """Keep the mean of finite folds usable when their sum can overflow."""
    scores = np.asarray(scores)
    if scores.dtype.kind == "f" and scores.size > 1:
        largest, _, _ = score_scale_limits(scores.dtype)
        limit = largest / scores.size
        if scale is None:
            # A short vector can skip allocating an absolute-value array.
            # A NaN skipped by max still propagates through the mean.
            if scores.ndim == 1 and scores.size <= 16 and max(map(abs, scores.tolist())) < limit:
                return scores.mean()
            scale = np.max(np.abs(scores))
        if scale >= limit and np.isfinite(scale):
            if scores.dtype.itemsize <= 8:
                # Exact accumulation preserves cancellation as well as the range.
                with np.errstate(under="ignore"):
                    return scores.dtype.type(mean(map(float, scores)))
            _, exponent = np.frexp(scale)
            with np.errstate(under="ignore"):
                return _restore_scaled_mean(np.ldexp(scores, -exponent), exponent)
    return scores.mean()


def scaled_score_moments(values, scale):
    # Power-of-two scaling preserves neighboring scores. Center before
    # reducing variance to keep narrow spreads at large offsets.
    mantissa, exponent = np.frexp(scale)
    with np.errstate(under="ignore"):
        scaled = np.ldexp(values, -exponent)
        average = score_mean(values, scale=scale)
        deviation = np.ldexp(min(np.std(scaled - scaled[0]), mantissa), exponent)
    return average, deviation
