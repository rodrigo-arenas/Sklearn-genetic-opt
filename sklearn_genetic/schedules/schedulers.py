import math
from .base import BaseAdapter


class ExponentialAdapter(BaseAdapter):
    """
    Adapts the initial value towards the end value using an exponential "decay" function
    """
    def __init__(self, initial_value, end_value, adaptive_rate):
        super().__init__(initial_value, end_value, adaptive_rate)

    def step(self, *args, **kwargs):
        self.current_value = ((self.initial_value - self.end_value) * math.exp(
            -self.adaptive_rate * self.current_step)) + self.end_value
        self.current_step += 1

        return self.current_value


class InverseAdapter(BaseAdapter):
    """
        Adapts the initial value towards the end value using a "decay" function of the for 1/x
        """
    def __init__(self, initial_value, end_value, adaptive_rate):
        super().__init__(initial_value, end_value, adaptive_rate)

    def step(self, *args, **kwargs):
        self.current_value = self.end_value + ((self.initial_value - self.end_value) / (1 + self.adaptive_rate * self.current_step))
        self.current_step += 1

        return self.current_value


class PotentialAdapter(BaseAdapter):
    """
        Adapts the initial value towards the end value using an potential "decay" function
        """
    def __init__(self, initial_value, end_value, adaptive_rate):
        super().__init__(initial_value, end_value, adaptive_rate)

    def step(self, *args, **kwargs):
        self.current_value = self.end_value + (self.initial_value - self.end_value)*math.pow(self.adaptive_rate, self.current_step)
        self.current_step += 1

        return self.current_value
