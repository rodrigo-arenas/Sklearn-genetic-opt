import enum


class ExtendedEnum(enum.Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class Algorithms(ExtendedEnum):
    eaSimple = "eaSimple"
    eaMuPlusLambda = "eaMuPlusLambda"
    eaMuCommaLambda = "eaMuCommaLambda"


class Criteria(ExtendedEnum):
    max = "max"
    min = "min"


class Metrics(ExtendedEnum):
    fitness = "fitness"
    fitness_std = "fitness_std"
    fitness_max = "fitness_max"
    fitness_min = "fitness_min"


class CallbackMethods(ExtendedEnum):
    on_start = "on_start"
    on_step = "on_step"
    on_end = "on_end"
