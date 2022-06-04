import math
from .base import BaseScheduler


class ExponentialDecay(BaseScheduler):
    def __init__(self, initial_value, decay_rate, min_value=None):
        super().__init__(initial_value, min_value, decay_rate)

    def step(self, *args, **kwargs):
        self.current_value = self.initial_value * math.exp(-self.decay_rate * self.current_step)
        self.current_step += 1
        if self.min_value:
            return max(self.current_value, self.min_value)

        return self.current_value


class InverseDecay(BaseScheduler):
    def __init__(self, initial_value, decay_rate, min_value=None):
        super().__init__(initial_value, min_value, decay_rate)

    def step(self, *args, **kwargs):
        self.current_value = self.initial_value / (1 + self.decay_rate * self.current_step)
        self.current_step += 1
        if self.min_value:
            return max(self.current_value, self.min_value)

        return self.current_value
