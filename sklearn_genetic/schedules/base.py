from abc import ABC, abstractmethod


class BaseAdapter(ABC):
    """
    Base class for all the adapters

    Parameters
    ----------
        initial_value : float,
            Initial value to be adapted
        end_value : float,
            The final (asymptotic) value that the initial_value can take
        adaptive_rate : float,
            Controls how fast the initial_value approaches the end_value
        kwargs : dict,
            Possible extra parameters, None for now

    Attributes
    ----------
        current_step : int,
            The current number of iterations that the adapter has run

        current_value : float,
            The transformed initial_value after current_steps changes
    """

    def __init__(self, initial_value, end_value, adaptive_rate, **kwargs):
        self.initial_value = initial_value
        self.end_value = end_value
        self.adaptive_rate = adaptive_rate
        self.current_value = self.initial_value
        self.current_step = 0

    @abstractmethod
    def step(self):
        """
        Run one iteration of the transformation
        """
        raise NotImplementedError("Scheduler must override step()")  # pragma: no cover
