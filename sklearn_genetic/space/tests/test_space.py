import pytest

from ..space import Categorical, Integer, Continuous, Space
from ..base import BaseDimension


@pytest.mark.parametrize(
    "data_object, parameters",
    [
        (Continuous, {"lower": 0.01, "upper": 0.5, "distribution": "log-uniform"}),
        (Continuous, {"lower": 0.0, "upper": 0.5, "distribution": "uniform"}),
        (Integer, {"lower": 5, "upper": 20, "distribution": "uniform"}),
    ],
)
def test_sample_variables(data_object, parameters):
    my_categorical = Categorical(choices=["car", "byc", "house"], priors=[0.2, 0.1, 0.7])
    for _ in range(20):
        assert my_categorical.sample() in ["car", "byc", "house"]

    my_variable = data_object(**parameters)
    for _ in range(100):
        assert my_variable.sample() <= parameters["upper"]
        assert my_variable.sample() >= parameters["lower"]


@pytest.mark.parametrize(
    "data_object, parameters, message",
    [
        (
            Continuous,
            {"lower": 10, "upper": 0.5},
            "The upper bound can not be smaller that the lower bound",
        ),
        (
            Integer,
            {"lower": 10, "upper": 2},
            "The upper bound can not be smaller that the lower bound",
        ),
    ],
)
def test_wrong_boundaries(data_object, parameters, message):
    with pytest.raises(Exception) as excinfo:
        data_object(**parameters)
    assert str(excinfo.value) == message


@pytest.mark.parametrize(
    "data_object, parameters, message",
    [
        (
            Continuous,
            {"lower": 10, "upper": 50, "distribution": "normal"},
            "distribution must be one of ['uniform', 'log-uniform'], got normal instead",
        ),
        (
            Categorical,
            {"choices": [True, False], "distribution": "sample"},
            "distribution must be one of ['choice'], got sample instead",
        ),
        (
            Integer,
            {"lower": 2, "upper": 10, "distribution": "log-uniform"},
            "distribution must be one of ['uniform'], got log-uniform instead",
        ),
    ],
)
def test_wrong_distributions(data_object, parameters, message):
    with pytest.raises(Exception) as excinfo:
        data_object(**parameters)
    assert str(excinfo.value) == message


@pytest.mark.parametrize(
    "data_object, parameters, message",
    [
        (Categorical, {"priors": [0.1, 0.9]}, "choices must be a non empty list"),
        (
            Categorical,
            {"choices": [True, False], "priors": [0.1, 0.8]},
            "The sum of the probabilities in the priors must be one, got 0.9 instead",
        ),
        (
            Categorical,
            {"choices": [True], "priors": [0.1, 0.9]},
            "priors and choices must have same size",
        ),
    ],
)
def test_categorical_bad_parameters(data_object, parameters, message):
    with pytest.raises(Exception) as excinfo:
        data_object(**parameters)
    assert str(excinfo.value) == message


def test_check_space_fail():
    with pytest.raises(Exception) as excinfo:
        my_space = Space()
    assert str(excinfo.value) == "param_grid can not be empty"

    param_grid = {
        "min_weight_fraction_leaf": Continuous(lower=0.001, upper=0.5, distribution="log-uniform"),
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


@pytest.mark.parametrize(
    "data_object, parameters, message",
    [
        (Categorical, (True, False), "choices must be a non empty list"),
        (Integer, (5.4, 10), "lower bound must be an integer"),
        (Integer, (5, 10.4), "upper bound must be an integer"),
        (Continuous, ([1], 10), "lower bound must be an integer or float"),
        (Continuous, (5, [10.4]), "upper bound must be an integer or float"),
    ],
)
def test_bad_data_types(data_object, parameters, message):
    with pytest.raises(Exception) as excinfo:
        data_object(*parameters)
    assert str(excinfo.value) == message


def test_wrong_dimension():
    possible_messages = [
        "Can't instantiate abstract class FakeDimension with abstract methods sample",
        "Can't instantiate abstract class FakeDimension with abstract method sample",
    ]
    with pytest.raises(Exception) as excinfo:

        class FakeDimension(BaseDimension):
            def __init__(self):
                pass

        FakeDimension().sample()

    assert any([str(excinfo.value) == i for i in possible_messages])
