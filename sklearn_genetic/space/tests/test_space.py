import pytest

from ..space import Categorical, Integer, Continuous


def test_sample_variables():
    my_categorical = Categorical(choices=['car', 'byc', 'house'], priors=[0.2, 0.1, 0.7])
    for _ in range(20):
        assert my_categorical.sample() in ['car', 'byc', 'house']

    my_continuous = Continuous(lower=0.01, upper=0.5, distribution='log-uniform')
    for _ in range(100):
        assert my_continuous.sample() <= 0.5
        assert my_continuous.sample() >= 0

    my_continuous = Continuous(lower=0.0, upper=0.5, distribution='uniform')
    for _ in range(100):
        assert my_continuous.sample() <= 0.5
        assert my_continuous.sample() >= 0

    my_integer = Integer(lower=5, upper=20, distribution='uniform')
    for _ in range(100):
        assert my_integer.sample() >= 5
        assert my_integer.sample() <= 20


def test_wrong_boundaries():
    with pytest.raises(Exception) as excinfo:
        my_continuous = Continuous(lower=10, upper=0.5)

    assert str(excinfo.value) == "The upper bound can not be smaller that the lower bound"

    with pytest.raises(Exception) as excinfo:
        my_integer = Integer(lower=10, upper=2)

    assert str(excinfo.value) == "The upper bound can not be smaller that the lower bound"


def test_wrong_distributions():
    with pytest.raises(Exception) as excinfo:
        my_continuous = Continuous(lower=2, upper=10, distribution='normal')
    assert str(excinfo.value) == "distribution must be one of ['uniform', 'log-uniform'], got normal instead"

    with pytest.raises(Exception) as excinfo:
        my_categorical = Categorical([True, False], distribution='sample')
    assert str(excinfo.value) == "distribution must be one of ['choice'], got sample instead"

    with pytest.raises(Exception) as excinfo:
        my_integer = Integer(lower=2, upper=10, distribution='log-uniform')
    assert str(excinfo.value) == "distribution must be one of ['uniform'], got log-uniform instead"


def test_categorical_bad_parameters():

    with pytest.raises(Exception) as excinfo:
        my_categorical = Categorical(priors=[0.1, 0.9])
    assert str(excinfo.value) == "choices can not be empty"

    with pytest.raises(Exception) as excinfo:
        my_categorical = Categorical(choices=[True, False], priors=[0.1, 0.8])
    assert str(excinfo.value) == "The sum of the probabilities in the priors must be one, got 0.9 instead"

    with pytest.raises(Exception) as excinfo:
        my_categorical = Categorical([True], priors=[0.1, 0.9])
    assert str(excinfo.value) == "priors and choices must have same size"
