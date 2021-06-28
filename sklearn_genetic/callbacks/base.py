from abc import ABC


class BaseCallback(ABC):
    """
    Base Callback from which all Callbacks must inherit from
    """

    def on_start(self, estimator=None):
        """
        Take actions at the start of the training

        Parameters
        ----------
        estimator:
            :class:`~sklearn_genetic.GASearchCV` Estimator that is being optimized


        """
        pass  # pragma: no cover

    def on_step(self, record=None, logbook=None, estimator=None):
        """
        Take actions after fitting each generation.

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
        decision: bool, default=False
            If ``True``, the optimization process is stopped, else, if continues to the next generation.
        """

        return False

    def on_end(self, logbook=None, estimator=None):
        """
        Take actions at the end of the training

        Parameters
        ----------
        logbook:
            Current stream logbook with the stats required
        estimator:
            :class:`~sklearn_genetic.GASearchCV` Estimator that is being optimized


        """
        pass  # pragma: no cover

    def __call__(self, record=None, logbook=None, estimator=None):
        return self.on_step(record, logbook, estimator)
