import pytest
import numpy as np
from scipy import stats

from ..space import Categorical, Integer, Continuous, Space
from ..converters import from_sklearn_space
from ..base import BaseDimension


@pytest.mark.parametrize(
    "space_object, expected_repr",
    [
        (
            Continuous(0.01, 1.0, distribution="log-uniform"),
            "Continuous(lower=0.01, upper=1.0, distribution='log-uniform')",
        ),
        (Integer(10, 200), "Integer(lower=10, upper=200, distribution='uniform')"),
        (
            Categorical(["rbf", "linear", "poly"]),
            "Categorical(choices=['rbf', 'linear', 'poly'])",
        ),
        (
            Categorical(["a", "b"], priors=[0.8, 0.2]),
            "Categorical(choices=['a', 'b'], priors=[0.8, 0.2])",
        ),
    ],
)
def test_space_repr(space_object, expected_repr):
    assert repr(space_object) == expected_repr


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

    with pytest.raises(ValueError) as excinfo:
        my_space = Space(param_grid)
    message = str(excinfo.value)
    # The improved message names the offending key, the type that was passed,
    # and shows a corrective example.
    assert "Invalid param_grid entry for 'max_depth'" in message
    assert "got range instead" in message
    assert "Categorical" in message


def test_check_space_invalid_type_message():
    """A non-space value yields a clear, actionable error (issue #210)."""
    param_grid = {"kernel": ["rbf", "linear"]}  # should be Categorical([...])

    with pytest.raises(ValueError, match=r"Invalid param_grid entry for 'kernel'"):
        Space(param_grid)

    with pytest.raises(ValueError) as excinfo:
        Space(param_grid)
    message = str(excinfo.value)
    assert "expected a space object (Continuous, Integer, or Categorical)" in message
    assert "got list instead" in message
    assert 'param_grid = {"kernel": Categorical([...])}' in message


def _warm_start_space():
    return Space(
        {
            "max_depth": Integer(2, 20),
            "criterion": Categorical(["gini", "entropy"]),
        }
    )


def test_sample_warm_start_accepts_valid_config():
    space = _warm_start_space()
    sampled = space.sample_warm_start({"max_depth": 5, "criterion": "gini"})
    assert sampled["max_depth"] == 5
    assert sampled["criterion"] == "gini"


def test_sample_warm_start_rejects_unknown_key():
    """A misspelled hyperparameter name is reported clearly (issue #220)."""
    space = _warm_start_space()
    with pytest.raises(ValueError) as excinfo:
        space.sample_warm_start({"max_depths": 5})  # typo: should be max_depth
    message = str(excinfo.value)
    assert "max_depths" in message
    assert "not in the search space" in message
    assert "max_depth" in message  # the valid name is suggested


def test_sample_warm_start_rejects_value_outside_space():
    """A value outside its dimension is reported clearly (issue #220)."""
    space = _warm_start_space()
    with pytest.raises(ValueError, match="outside its search space"):
        space.sample_warm_start({"max_depth": 99})
    with pytest.raises(ValueError, match="outside its search space"):
        space.sample_warm_start({"criterion": "log_loss"})


def test_sample_warm_start_rejects_non_dict_config():
    space = _warm_start_space()
    with pytest.raises(ValueError, match="must be a dict"):
        space.sample_warm_start([("max_depth", 5)])


def test_sample_warm_start_accepts_numpy_scalars():
    """NumPy scalars are valid values, not rejected as out-of-space (#220)."""
    space = Space({"max_depth": Integer(2, 20), "lr": Continuous(0.0, 1.0)})
    sampled = space.sample_warm_start({"max_depth": np.int64(5), "lr": np.float64(0.5)})
    assert sampled["max_depth"] == 5
    assert sampled["lr"] == 0.5


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
    with pytest.raises(Exception) as excinfo:

        class FakeDimension(BaseDimension):
            def __init__(self):
                pass

        FakeDimension().sample()

    message = str(excinfo.value)
    assert "Can't instantiate abstract class FakeDimension" in message
    assert "sample" in message


def test_from_sklearn_space_converts_lists_and_scipy_distributions():
    native_dimension = Integer(1, 3)
    converted = from_sklearn_space(
        {
            "criterion": ["gini", "entropy"],
            "max_depth": stats.randint(2, 12),
            "learning_rate": stats.uniform(0.01, 0.29),
            "alpha": stats.loguniform(1e-5, 1e-1),
            "already_native": native_dimension,
        }
    )

    assert isinstance(converted["criterion"], Categorical)
    assert converted["criterion"].choices == ["gini", "entropy"]
    assert isinstance(converted["max_depth"], Integer)
    assert converted["max_depth"].lower == 2
    assert converted["max_depth"].upper == 11
    assert isinstance(converted["learning_rate"], Continuous)
    assert converted["learning_rate"].lower == pytest.approx(0.01)
    assert converted["learning_rate"].upper == pytest.approx(0.3)
    assert converted["learning_rate"].distribution == "uniform"
    assert isinstance(converted["alpha"], Continuous)
    assert converted["alpha"].distribution == "log-uniform"
    assert converted["already_native"].lower == 1
    assert converted["already_native"] is native_dimension


def test_from_sklearn_space_converts_keyword_scipy_distributions():
    converted = from_sklearn_space(
        {
            "max_depth": stats.randint(low=3, high=9),
            "subsample": stats.uniform(loc=0.25, scale=0.5),
            "reg_alpha": stats.reciprocal(a=1e-6, b=1e-2),
        }
    )

    assert converted["max_depth"].lower == 3
    assert converted["max_depth"].upper == 8
    assert converted["subsample"].lower == pytest.approx(0.25)
    assert converted["subsample"].upper == pytest.approx(0.75)
    assert converted["reg_alpha"].lower == pytest.approx(1e-6)
    assert converted["reg_alpha"].upper == pytest.approx(1e-2)
    assert converted["reg_alpha"].distribution == "log-uniform"


def test_from_sklearn_space_converts_range_and_numpy_array_to_categorical():
    converted = from_sklearn_space(
        {
            "depth_options": range(2, 5),
            "activation": ["relu", "tanh"],
            "batch_size": np.array([32, 64, 128]),
        }
    )

    assert converted["depth_options"].choices == [2, 3, 4]
    assert converted["activation"].choices == ["relu", "tanh"]
    assert converted["batch_size"].choices == [32, 64, 128]


def test_from_sklearn_space_rejects_unsupported_distributions():
    with pytest.raises(ValueError) as excinfo:
        from_sklearn_space({"alpha": stats.expon()})

    assert "scipy.stats.expon" in str(excinfo.value)


def test_from_sklearn_space_rejects_empty_or_ambiguous_values():
    with pytest.raises(ValueError, match="non-empty mapping"):
        from_sklearn_space({})

    with pytest.raises(ValueError, match="has no categorical choices"):
        from_sklearn_space({"criterion": []})

    with pytest.raises(ValueError, match="must be a list-like value"):
        from_sklearn_space({"max_depth": 3})


def test_from_sklearn_space_rejects_broken_frozen_distribution_bounds(monkeypatch):
    broken_distribution = stats.randint(1, 5)
    monkeypatch.setattr(broken_distribution, "args", ())
    monkeypatch.setattr(broken_distribution, "kwds", {})

    with pytest.raises(ValueError, match="must define low and high bounds"):
        from_sklearn_space({"max_depth": broken_distribution})
