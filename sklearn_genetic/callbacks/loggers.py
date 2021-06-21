import logging
from copy import deepcopy
from joblib import dump


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
