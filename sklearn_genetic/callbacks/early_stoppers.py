from datetime import datetime

from .validations import check_stats
from .base import BaseCallback


class ThresholdStopping(BaseCallback):
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

    def on_step(self, record=None, logbook=None, estimator=None):
        stat = None
        if record is not None:
            stat = record[self.metric]
        elif logbook is not None:
            # Get the last metric value
            stat = logbook.select(self.metric)[-1]

        else:
            raise ValueError(
                "At least one of record or logbook parameters must be provided"
            )

        if stat is not None and stat >= self.threshold:
            print(f"INFO: {self.__class__.__name__} callback met its criteria")
            return True
        return False


class ConsecutiveStopping(BaseCallback):
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
        if logbook is not None:
            if len(logbook) <= self.generations:
                return False

            if record is not None:
                current_stat = record[self.metric]
            else:
                current_stat = logbook.select(self.metric)[-1]

            # Compare the current metric with the last |generations| metrics
            stats = logbook.select(self.metric)[(-self.generations - 1): -1]

            if all(stat >= current_stat for stat in stats):
                print(f"INFO: {self.__class__.__name__} callback met its criteria")
                return True
            return False

        else:
            raise ValueError("logbook parameter must be provided")


class DeltaThreshold(BaseCallback):
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
        if logbook is not None:
            if len(logbook) <= 1:
                return False

            if record is not None:
                current_stat = record[self.metric]
            else:
                current_stat = logbook.select(self.metric)[-1]

            # Compare the current metric with the last |generations| metrics
            previous_stat = logbook.select(self.metric)[-2]

            if abs(current_stat - previous_stat) <= self.threshold:
                print(f"INFO: {self.__class__.__name__} callback met its criteria")
                return True
            return False

        else:
            raise ValueError("logbook parameter must be provided")


class TimerStopping(BaseCallback):
    """
    Stops the optimization process if a limit training time has been elapsed.
    This time is checked after each generation fit
    """

    def __init__(self, total_seconds):
        """
        Parameters
        ----------
        total_seconds: int
            Total time in seconds that the estimator is allowed to fit
        """
        self.initial_training_time = None
        self.total_seconds = total_seconds

    def on_start(self, estimator=None):
        self.initial_training_time = datetime.utcnow()

    def on_step(self, record=None, logbook=None, estimator=None):
        current_time = datetime.utcnow()
        difference = current_time - self.initial_training_time
        difference_seconds = difference.total_seconds()

        if difference_seconds >= self.total_seconds:
            print(f"INFO: {self.__class__.__name__} callback met its criteria")
            return True
        return False
