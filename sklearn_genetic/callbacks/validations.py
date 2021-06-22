from ..parameters import Metrics
from .base import BaseCallback


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
        if isinstance(callback, BaseCallback):
            return [callback]

        elif isinstance(callback, list) and all(
            [isinstance(c, BaseCallback) for c in callback]
        ):
            return callback

        else:
            raise ValueError(
                "callback should be either a class or a list of classes with inheritance from "
                "callbacks.base.BaseCallback"
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
