import enum

"""
This module contains all the possible random distributions names
that can be set in each of the Space variables
"""


class ExtendedEnum(enum.Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class IntegerDistributions(ExtendedEnum):
    uniform = "uniform"


class ContinuousDistributions(ExtendedEnum):
    uniform = "uniform"
    log_uniform = "log-uniform"


class CategoricalDistributions(ExtendedEnum):
    choice = "choice"
