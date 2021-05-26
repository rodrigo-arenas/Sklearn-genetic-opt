import enum


class ExtendedEnum(enum.Enum):

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class Algorithms(ExtendedEnum):
    eaSimple = 'eaSimple'
    eaMuPlusLambda = 'eaMuPlusLambda'


class Criteria(ExtendedEnum):
    max = 'max'
    min = 'min'
