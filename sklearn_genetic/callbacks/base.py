from abc import ABC, abstractmethod


class BaseCallback(ABC):
    """
    Base Callback from which all Callbacks must inherit from
    """

    @abstractmethod
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

        pass  # pragma: no cover

    @abstractmethod
    def __call__(self, record=None, logbook=None, estimator=None):
        pass  # pragma: no cover
