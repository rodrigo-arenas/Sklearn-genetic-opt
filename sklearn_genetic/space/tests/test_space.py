import pytest

from ..space import Categorical, Integer, Continuous, Space
from ..base import BaseDimension


def test_sample_variables():
    my_categorical = Categorical(
        choices=["car", "byc", "house"], priors=[0.2, 0.1, 0.7]
    )
    for _ in range(20):
        assert my_categorical.sample() in ["car", "byc", "house"]

    my_continuous = Continuous(lower=0.01, upper=0.5, distribution="log-uniform")
    for _ in range(100):
        assert my_continuous.sample() <= 0.5
        assert my_continuous.sample() >= 0

    my_continuous = Continuous(lower=0.0, upper=0.5, distribution="uniform")
    for _ in range(100):
        assert my_continuous.sample() <= 0.5
        assert my_continuous.sample() >= 0

    my_integer = Integer(lower=5, upper=20, distribution="uniform")
    for _ in range(100):
        assert my_integer.sample() >= 5
        assert my_integer.sample() <= 20


def test_wrong_boundaries():
    with pytest.raises(Exception) as excinfo:
        my_continuous = Continuous(lower=10, upper=0.5)

    assert (
        str(excinfo.value) == "The upper bound can not be smaller that the lower bound"
    )

    with pytest.raises(Exception) as excinfo:
        my_integer = Integer(lower=10, upper=2)

    assert (
        str(excinfo.value) == "The upper bound can not be smaller that the lower bound"
    )


def test_wrong_distributions():
    with pytest.raises(Exception) as excinfo:
        my_continuous = Continuous(lower=2, upper=10, distribution="normal")
    assert (
        str(excinfo.value)
        == "distribution must be one of ['uniform', 'log-uniform'], got normal instead"
    )

    with pytest.raises(Exception) as excinfo:
        my_categorical = Categorical([True, False], distribution="sample")
    assert (
        str(excinfo.value)
        == "distribution must be one of ['choice'], got sample instead"
    )

    with pytest.raises(Exception) as excinfo:
        my_integer = Integer(lower=2, upper=10, distribution="log-uniform")
    assert (
        str(excinfo.value)
        == "distribution must be one of ['uniform'], got log-uniform instead"
    )


def test_categorical_bad_parameters():

    with pytest.raises(Exception) as excinfo:
        my_categorical = Categorical(priors=[0.1, 0.9])
    assert str(excinfo.value) == "choices must be a non empty list"

    with pytest.raises(Exception) as excinfo:
        my_categorical = Categorical(choices=[True, False], priors=[0.1, 0.8])
    assert (
        str(excinfo.value)
        == "The sum of the probabilities in the priors must be one, got 0.9 instead"
    )

    with pytest.raises(Exception) as excinfo:
        my_categorical = Categorical([True], priors=[0.1, 0.9])
    assert str(excinfo.value) == "priors and choices must have same size"


def test_check_space_fail():
    with pytest.raises(Exception) as excinfo:
        my_space = Space()
    assert str(excinfo.value) == "param_grid can not be empty"

    param_grid = {
        "min_weight_fraction_leaf": Continuous(
            lower=0.001, upper=0.5, distribution="log-uniform"
        ),
        "max_leaf_nodes": Integer(lower=2, upper=35),
        "criterion": Categorical(choices=["gini", "entropy"]),
        "max_depth": range(10, 20),
    }

    with pytest.raises(Exception) as excinfo:
        my_space = Space(param_grid)
    assert (
        str(excinfo.value)
        == "max_depth must be a valid instance of Integer, Categorical or Continuous classes"
    )


def test_bad_data_types():

    with pytest.raises(Exception) as excinfo:
        Categorical((True, False))
    assert str(excinfo.value) == "choices must be a non empty list"

    with pytest.raises(Exception) as excinfo:
        Integer(5.4, 10)
    assert str(excinfo.value) == "lower bound must be an integer"

    with pytest.raises(Exception) as excinfo:
        Integer(5, 10.4)
    assert str(excinfo.value) == "upper bound must be an integer"

    with pytest.raises(Exception) as excinfo:
        Continuous([1], 10)
    assert str(excinfo.value) == "lower bound must be an integer or float"

    with pytest.raises(Exception) as excinfo:
        Continuous(5, [10.4])
    assert str(excinfo.value) == "upper bound must be an integer or float"


def test_wrong_dimension():
    with pytest.raises(Exception) as excinfo:

        class FakeDimension(BaseDimension):
            def __init__(self):
                pass

        FakeDimension().sample()

    assert (
        str(excinfo.value)
        == "The sample method must be defined according each data type handler"
    )
