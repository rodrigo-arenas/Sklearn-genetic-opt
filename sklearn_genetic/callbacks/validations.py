from typing import Callable

from ..parameters import Metrics


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
