import math
from .base import BaseAdapter


class ConstantAdapter(BaseAdapter):
    """
    This adapter keep the current value equals to the initial_value
    it's mainly used to have an uniform interface when defining a
    parameter as constant vs as an adapter

    Parameters
    ----------
        initial_value : float,
            Initial value to be adapted
        end_value : float,
            The final (asymptotic) value that the initial_value can take
        adaptive_rate : float,
            Controls how fast the initial_value approaches the end_value

    Attributes
    ----------
        current_step : int,
            The current number of iterations that the adapter has run

        current_value : float,
            Same as the initial_value
    """

    def __init__(self, initial_value, end_value, adaptive_rate):
        super().__init__(initial_value, end_value, adaptive_rate)

    def step(self):
        self.current_step += 1
        return self.current_value


class ExponentialAdapter(BaseAdapter):
    """
    Adapts the initial value towards the end value using an exponential "decay" function

    Parameters
    ----------
    initial_value : float,
        Initial value to be adapted
    end_value : float,
        The final (asymptotic) value that the initial_value can take
    adaptive_rate : float,
        Controls how fast the initial_value approaches the end_value

    Attributes
    ----------
    current_step : int,
        The current number of iterations that the adapter has run

    current_value : float,
        The transformed initial_value after current_steps changes
    """

    def __init__(self, initial_value, end_value, adaptive_rate):
        super().__init__(initial_value, end_value, adaptive_rate)

    def step(self):
        self.current_value = (
            (self.initial_value - self.end_value)
            * math.exp(-self.adaptive_rate * self.current_step)
        ) + self.end_value
        self.current_step += 1

        return self.current_value


class InverseAdapter(BaseAdapter):
    """
    Adapts the initial value towards the end value using a "decay" function of the form 1/x

    Parameters
    ----------
    initial_value : float,
        Initial value to be adapted
    end_value : float,
        The final (asymptotic) value that the initial_value can take
    adaptive_rate : float,
        Controls how fast the initial_value approaches the end_value

    Attributes
    ----------
    current_step : int,
        The current number of iterations that the adapter has run

    current_value : float,
        The transformed initial_value after current_steps changes
    """

    def __init__(self, initial_value, end_value, adaptive_rate):
        super().__init__(initial_value, end_value, adaptive_rate)

    def step(self):
        self.current_value = self.end_value + (
            (self.initial_value - self.end_value) / (1 + self.adaptive_rate * self.current_step)
        )
        self.current_step += 1

        return self.current_value


class PotentialAdapter(BaseAdapter):
    """
    Adapts the initial value towards the end value using a potential "decay" function

    Parameters
    ----------
    initial_value : float,
        Initial value to be adapted
    end_value : float,
        The final (asymptotic) value that the initial_value can take
    adaptive_rate : float,
        Controls how fast the initial_value approaches the end_value

    Attributes
    ----------
    current_step : int,
        The current number of iterations that the adapter has run

    current_value : float,
        The transformed initial_value after current_steps changes
    """

    def __init__(self, initial_value, end_value, adaptive_rate):
        super().__init__(initial_value, end_value, adaptive_rate)

    def step(self):
        self.current_value = self.end_value + (self.initial_value - self.end_value) * math.pow(
            (1 - self.adaptive_rate), self.current_step
        )
        self.current_step += 1

        return self.current_value
