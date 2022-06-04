import math
from .base import BaseAdapter


class ExponentialAdapter(BaseAdapter):
    def __init__(self, initial_value, end_value, decay_rate):
        super().__init__(initial_value, end_value, decay_rate)

    def step(self, *args, **kwargs):
        self.current_value = ((self.initial_value - self.end_value) * math.exp(
            -self.decay_rate * self.current_step)) + self.end_value
        self.current_step += 1

        return self.current_value


class InverseAdapter(BaseAdapter):
    def __init__(self, initial_value, end_value, decay_rate):
        super().__init__(initial_value, end_value, decay_rate)

    def step(self, *args, **kwargs):
        self.current_value = self.end_value + ((self.initial_value - self.end_value) / (1 + self.decay_rate * self.current_step))
        self.current_step += 1

        return self.current_value
