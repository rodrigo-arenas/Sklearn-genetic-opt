import enum


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
