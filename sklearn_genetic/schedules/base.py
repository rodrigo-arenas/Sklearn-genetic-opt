from abc import ABC


class BaseAdapter(ABC):
    def __init__(self, initial_value, end_value, decay_rate, **kwargs):
        self.initial_value = initial_value
        self.end_value = end_value
        self.decay_rate = decay_rate
        self.current_value = self.initial_value
        self.current_step = 0

    def step(self, *args, **kwargs):
        raise NotImplementedError("Scheduler must override step()")
