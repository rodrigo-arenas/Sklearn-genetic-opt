from collections.abc import Callable
from joblib import dump
from copy import deepcopy
import logging

from .parameters import Metrics


def check_stats(metric):
    if metric not in Metrics.list():
        raise ValueError(
            f"metric must be one of {Metrics.list()}, but got {metric} instead"
        )


def check_callback(callback):
    """
    Check if callback is a callable or a list of callables.
    """
    if callback is not None:
        if isinstance(callback, Callable):
            return [callback]

        elif isinstance(callback, list) and all(
            [isinstance(c, Callable) for c in callback]
        ):
            return callback

        else:
            raise ValueError(
                "callback should be either a callable or a list of callables."
            )
    else:
        return []


def eval_callbacks(callbacks, record, logbook, estimator):
    """Evaluate list of callbacks on result.
    Parameters
    ----------
    callbacks : list of callables
        Callbacks to evaluate.
    record : logbook record
    logbook:
            Current stream logbook with the stats required
    estimator: :class:`~sklearn_genetic.GASearchCV`, default = None
        Estimator that is being optimized

    Returns
    -------
    decision : bool
        Decision of the callbacks whether or not to keep optimizing
    """
    stop = False
    if callbacks:
        for c in callbacks:
            decision = c(record, logbook, estimator)
            if decision is not None:
                stop = stop or decision

    return stop


class ThresholdStopping:
    """
    Stop the optimization if the metric from
    cross validation score is greater or equals than the define threshold
    """

    def __init__(self, threshold, metric="fitness"):
        """
        Parameters
        ----------
        threshold: float, default=None
            Threshold to compare against the current cross validation average score and determine if
            the optimization process must stop
        metric: {'fitness', 'fitness_std', 'fitness_max', 'fitness_min'}, default ='fitness'
            Name of the metric inside 'record' logged in each iteration
        """

        check_stats(metric)

        self.threshold = threshold
        self.metric = metric

    def on_step(self, record, logbook, estimator):
        """
        Parameters
        ----------
        record: dict: default=None
            A logbook record
        logbook:
            Current stream logbook with the stats required
        estimator:
            :class:`~sklearn_genetic.GASearchCV` Estimator that is being optimized

        Returns
        -------
        decision: bool
            True if the optimization algorithm must stop, false otherwise
        """
        if record is not None:
            return record[self.metric] >= self.threshold
        elif logbook is not None:
            # Get the last metric value
            stat = logbook.select(self.metric)[-1]
            return stat >= self.threshold
        else:
            raise ValueError(
                "At least one of record or logbook parameters must be provided"
            )

    def __call__(self, record=None, logbook=None, estimator=None):
        return self.on_step(record, logbook, estimator)


class ConsecutiveStopping:
    """
    Stop the optimization if the current metric value is no greater that at least one metric from the last N generations
    """

    def __init__(self, generations, metric="fitness"):
        """
        Parameters
        ----------
        generations: int, default=None
            Number of current generations to compare against current generation
        metric: {'fitness', 'fitness_std', 'fitness_max', 'fitness_min'}, default ='fitness'
            Name of the metric inside 'record' logged in each iteration
        """

        check_stats(metric)

        self.generations = generations
        self.metric = metric

    def on_step(self, record=None, logbook=None, estimator=None):
        """
        Parameters
        ----------
        record: dict: default=None
            A logbook record
        logbook:
            Current stream logbook with the stats required
        estimator:
            :class:`~sklearn_genetic.GASearchCV` Estimator that is being optimized

        Returns
        -------
        decision: bool
            True if the optimization algorithm must stop, false otherwise
        """
        if logbook is not None:
            if len(logbook) <= self.generations:
                return False

            if record is not None:
                current_stat = record[self.metric]
            else:
                current_stat = logbook.select(self.metric)[-1]

            # Compare the current metric with the last |generations| metrics
            stats = logbook.select(self.metric)[(-self.generations - 1) : -1]
            return all(stat >= current_stat for stat in stats)
        else:
            raise ValueError("logbook parameter must be provided")

    def __call__(self, record=None, logbook=None, estimator=None):
        return self.on_step(record, logbook, estimator)


class DeltaThreshold:
    """
    Stop the optimization if the absolute difference between the current and last metric less or equals than a threshold
    """

    def __init__(self, threshold, metric: str = "fitness"):
        """
        Parameters
        ----------
        threshold: float, default=None
            Threshold to compare the differences between cross validation scores
        metric: {'fitness', 'fitness_std', 'fitness_max', 'fitness_min'}, default ='fitness'
            Name of the metric inside 'record' logged in each iteration
        """

        check_stats(metric)

        self.threshold = threshold
        self.metric = metric

    def on_step(self, record=None, logbook=None, estimator=None):
        """
        Parameters
        ----------
        record: dict: default=None
            A logbook record
        logbook:
            Current stream logbook with the stats required
        estimator:
            :class:`~sklearn_genetic.GASearchCV` Estimator that is being optimized

        Returns
        -------
        decision: bool
            True if the optimization algorithm must stop, false otherwise
        """
        if logbook is not None:
            if len(logbook) <= 1:
                return False

            if record is not None:
                current_stat = record[self.metric]
            else:
                current_stat = logbook.select(self.metric)[-1]

            # Compare the current metric with the last |generations| metrics
            previous_stat = logbook.select(self.metric)[-2]

            return abs(current_stat - previous_stat) <= self.threshold
        else:
            raise ValueError("logbook parameter must be provided")

    def __call__(self, record=None, logbook=None, estimator=None):
        return self.on_step(record, logbook, estimator)


class LogbookSaver:
    """
    Saves the estimator.logbook parameter chapter object in a local file system
    """

    def __init__(self, checkpoint_path, **dump_options):
        """
        Parameters
        ----------
        checkpoint_path: str
            Location where checkpoint will be saved to
        dump_options, str
            Valid kwargs from joblib :class:`~joblib.dump`
        """

        self.checkpoint_path = checkpoint_path
        self.dump_options = dump_options

    def on_step(self, record=None, logbook=None, estimator=None):
        """
        Parameters
        ----------
        record: dict: default=None
            A logbook record
        logbook:
            Current stream logbook with the stats required
        estimator:
            :class:`~sklearn_genetic.GASearchCV` Estimator that is being optimized

        Returns
        -------
        decision: False
            Always returns False as this class doesn't take decisions over the optimization
        """
        try:
            dump_logbook = deepcopy(estimator.logbook.chapters["parameters"])
            dump(dump_logbook, self.checkpoint_path, **self.dump_options)
        except Exception as e:
            logging.error("Could not save the Logbook in the checkpoint")

        return False

    def __call__(self, record=None, logbook=None, estimator=None):
        return self.on_step(record, logbook, estimator)
