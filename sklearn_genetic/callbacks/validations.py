from ..parameters import Metrics, CallbackMethods
from .base import BaseCallback


def check_stats(metric):
    if metric not in Metrics.list():
        raise ValueError(f"metric must be one of {Metrics.list()}, but got {metric} instead")


def check_callback(callback):
    """
    Check if callback is a callable or a list of callables.
    """
    if callback is not None:
        if isinstance(callback, BaseCallback):
            return [callback]

        elif isinstance(callback, list) and all([isinstance(c, BaseCallback) for c in callback]):
            return callback

        else:
            raise ValueError(
                "callback should be either a class or a list of classes with inheritance from "
                "callbacks.base.BaseCallback"
            )
    else:
        return []


def eval_callbacks(callbacks, record, logbook, estimator, method):
    """Evaluate list of callbacks on result.
    Parameters
    ----------
    callbacks : list of callables
        Callbacks to evaluate.
    record : logbook record
    logbook: logbook object
            Current stream logbook with the stats required
    estimator: :class:`~sklearn_genetic.GASearchCV`, default = None
        Estimator that is being optimized
    method: {'on_start', 'on_step', 'on_end'}
        The method to be called from the callback

    Returns
    -------
    decision : bool
        Decision of the callbacks whether or not to keep optimizing
    """

    if method not in CallbackMethods.list():
        raise ValueError(
            f"The callback method must be one of {CallbackMethods.list()}, but got {method} instead"
        )

    stop = False
    decision = None

    if callbacks:
        for callback in callbacks:
            callback_method = getattr(callback, method)

            if method == CallbackMethods.on_start.value:
                decision = callback_method(estimator)

            elif method == CallbackMethods.on_step.value:
                decision = callback_method(record, logbook, estimator)

            elif method == CallbackMethods.on_end.value:
                decision = callback_method(logbook, estimator)

            if decision is not None:
                stop = stop or decision

    return stop
