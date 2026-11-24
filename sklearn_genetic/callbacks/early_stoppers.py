from datetime import UTC, datetime
from fractions import Fraction
from math import inf, isfinite
from numbers import Integral, Real
from operator import ge, index, le
from time import monotonic

import numpy as np

from .validations import check_stats
from .base import BaseCallback


def _validate_numeric(value, parameter_name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{parameter_name} must be a numeric value")
    # Comparing the value with itself detects NaN without casting large integers to float.
    if value != value:
        raise ValueError(f"{parameter_name} received NaN. Supply a defined numeric value.")


def _validate_positive_integer(value, parameter_name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{parameter_name} must be a positive integer")

    if value <= 0:
        raise ValueError(f"{parameter_name} must be greater than 0")


def _validate_positive_numeric(value, parameter_name):
    _validate_numeric(value, parameter_name)

    if value <= 0:
        raise ValueError(f"{parameter_name} must be greater than 0")


def _at_least_as_good(metric, estimator):
    minimize = metric != "fitness_std" and getattr(estimator, "criteria", "max") == "min"
    return le if minimize else ge


def _finite_score(value):
    return value is not None and isfinite(value)


def _as_fraction(value):
    return (
        Fraction(int(value)) if isinstance(value, Integral) else Fraction(*value.as_integer_ratio())
    )


class ThresholdStopping(BaseCallback):
    """
    Stop when a finite metric reaches the target in the search direction.
    For fitness_std, stop when the dispersion reaches or exceeds the target.
    """

    def __init__(self, threshold, metric="fitness"):
        """
        Parameters
        ----------
        threshold: float
            Target for the selected generation metric.
        metric: str, default="fitness"
            History metric to watch, such as "fitness_best_observed".
        """

        check_stats(metric)
        _validate_numeric(threshold, "threshold")

        self.threshold = threshold
        self.metric = metric

    def on_step(self, record=None, logbook=None, estimator=None):
        stat = None
        if record is not None:
            stat = record.get(self.metric)
        elif logbook is not None:
            # Get the last metric value
            stat = logbook[-1].get(self.metric)

        else:
            raise ValueError("At least one of record or logbook parameters must be provided")

        if _finite_score(stat) and _at_least_as_good(self.metric, estimator)(stat, self.threshold):
            print(f"INFO: {self.__class__.__name__} callback met its criteria")
            return True
        return False


class ConsecutiveStopping(BaseCallback):
    """
    Stop when none of the last N values is worse than the current value.
    Score metrics follow the search criterion. For fitness_std, larger values
    retain their historical meaning of improvement.
    The current value and every value in the comparison window must be finite.
    """

    def __init__(self, generations, metric="fitness"):
        """
        Parameters
        ----------
        generations: int
            Positive number of previous generations to compare with the current generation.
        metric: str, default="fitness"
            History metric to watch, such as "fitness_best_observed".
        """

        check_stats(metric)
        _validate_positive_integer(generations, "generations")

        self.generations = generations
        self.metric = metric

    def on_step(self, record=None, logbook=None, estimator=None):
        if logbook is not None:
            window = index(self.generations)
            if len(logbook) <= window:
                return False

            if record is not None:
                current_stat = record.get(self.metric)
            else:
                current_stat = logbook[-1].get(self.metric)

            # Compare the current metric with the previous window.
            stats = [row.get(self.metric) for row in logbook[-window - 1 : -1]]
            if not _finite_score(current_stat) or not all(map(_finite_score, stats)):
                return False

            at_least_as_good = _at_least_as_good(self.metric, estimator)
            if all(at_least_as_good(stat, current_stat) for stat in stats):
                print(f"INFO: {self.__class__.__name__} callback met its criteria")
                return True
            return False

        else:
            raise ValueError("logbook parameter must be provided")


class DeltaThreshold(BaseCallback):
    """
    Stop when the range across the last N generations is within the threshold
    and every value in that window is finite.
    """

    def __init__(self, threshold, generations=2, metric: str = "fitness"):
        """
        Parameters
        ----------
        threshold: float
            Threshold to compare the differences between cross-validation scores.
        generations: int, default=2
            Number of generations to compare, includes the current generation.
        metric: str, default="fitness"
            History metric to watch, such as "fitness_best_observed".
        """

        check_stats(metric)
        _validate_numeric(threshold, "threshold")
        _validate_positive_integer(generations, "generations")

        self.threshold = threshold
        self.generations = generations
        self.metric = metric

    def on_step(self, record=None, logbook=None, estimator=None):
        if logbook is not None:
            window = index(self.generations)
            if len(logbook) < window:
                return False

            stats = [row.get(self.metric) for row in logbook[-window:]]
            if not all(map(_finite_score, stats)):
                return False

            low, high = min(stats), max(stats)
            low = int(low) if isinstance(low, Integral) else low
            high = int(high) if isinstance(high, Integral) else high
            threshold = self.threshold
            with np.errstate(over="ignore"):
                diff = high - low
            if diff == inf and 0 <= threshold < inf:
                # An overflowing floating-point range can still fit the threshold.
                diff = _as_fraction(high) - _as_fraction(low)
                threshold = _as_fraction(threshold)

            if diff <= threshold:
                print(f"INFO: {self.__class__.__name__} callback met its criteria")
                return True
            return False

        else:
            raise ValueError("logbook parameter must be provided")


class CVBudgetStopping(BaseCallback):
    """Limit fresh candidate cross-validation calls during one fit.

    Each candidate scored with cross-validation counts once, regardless of
    the number of folds. Cache hits and scores restored from a checkpoint do
    not count. Finalist validation and the full-data refit have their own
    counts and run after the search.

    Parameters
    ----------
    max_cv_calls : int
        Positive limit on new candidate cross-validation calls.
    """

    def __init__(self, max_cv_calls):
        _validate_positive_integer(max_cv_calls, "max_cv_calls")
        self.max_cv_calls = int(max_cv_calls)

    def on_start(self, estimator=None):
        self._starting_calls = estimator.fit_stats_["cross_validate_calls"]

    def on_step(self, record=None, logbook=None, estimator=None):
        return (
            estimator.fit_stats_["cross_validate_calls"] - self._starting_calls >= self.max_cv_calls
        )


_NATIVE_CV_BUDGET_START = CVBudgetStopping.on_start


class TimerStopping(BaseCallback):
    """
    Stop starting new candidate batches when the elapsed search time expires.
    A monotonic clock measures elapsed time across system clock adjustments.
    An in-flight CV batch and final selection/refit can finish after the limit.
    A search without usable checkpoint results completes an initial batch.
    """

    def __init__(self, total_seconds):
        """
        Parameters
        ----------
        total_seconds: float
            Positive elapsed time limit in seconds for candidate search.
        """
        _validate_positive_numeric(total_seconds, "total_seconds")

        self.initial_training_time = None
        self.total_seconds = total_seconds

    def on_start(self, estimator=None):
        self._started_at = monotonic()
        self.initial_training_time = datetime.now(UTC).replace(tzinfo=None)

    def _time_exceeded(self):
        return monotonic() - self._started_at >= self.total_seconds

    def on_step(self, record=None, logbook=None, estimator=None):
        if self._time_exceeded():
            print(f"INFO: {self.__class__.__name__} callback met its criteria")
            return True
        return False
