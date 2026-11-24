from collections import Counter
from typing import Any

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
        (
            Continuous,
            {"lower": 0, "upper": 1, "distribution": "log-uniform"},
            "lower bound must be greater than zero for the log-uniform distribution, got 0 instead",
        ),
        (
            Continuous,
            {"lower": -1.5, "upper": 1, "distribution": "log-uniform"},
            "lower bound must be greater than zero for the log-uniform distribution, got -1.5 instead",
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
        (
            Categorical,
            {"choices": ["a", "b"], "priors": [1.5, -0.5]},
            "priors must be non-negative, got [1.5, -0.5] instead",
        ),
    ],
)
def test_categorical_bad_parameters(data_object, parameters, message):
    with pytest.raises(Exception) as excinfo:
        data_object(**parameters)
    assert str(excinfo.value) == message


def test_categorical_random_state_is_deterministic():
    """Same random_state yields the same sample sequence (#296).

    This exercises the seeded ``rng.choice`` path, which the existing sampling
    tests (using no ``random_state``) do not cover.
    """
    choices = ["a", "b", "c", "d"]
    first = Categorical(choices, random_state=42)
    second = Categorical(choices, random_state=42)

    seq_first = [first.sample() for _ in range(15)]
    seq_second = [second.sample() for _ in range(15)]

    assert seq_first == seq_second
    # Every draw is a valid choice.
    assert all(value in choices for value in seq_first)


def test_categorical_sampling_with_priors_stays_in_choices():
    """A priored categorical only ever samples declared choices (#296)."""
    choices = ["x", "y", "z"]
    dimension = Categorical(choices, priors=[0.2, 0.3, 0.5], random_state=7)

    assert dimension.priors == [0.2, 0.3, 0.5]
    assert all(dimension.sample() in choices for _ in range(50))


def test_categorical_priors_skew_sampling_with_numpy_rng():
    """Priors must actually bias sampling, not just be stored (#314).

    Uses the seeded ``rng.choice`` path (random_state is given). With a heavy
    98/2 skew over 4000 draws, an unweighted 50/50 draw is statistically
    impossible to land in the assertion window below.
    """
    dimension = Categorical(["a", "b"], priors=[0.98, 0.02], random_state=1)

    counts = Counter(dimension.sample() for _ in range(4000))

    assert counts["a"] > 3500
    assert counts["b"] < 500


def test_categorical_priors_skew_sampling_with_stdlib_random():
    """Same as above, but exercising the unseeded ``random.choices`` path (#314)."""
    dimension = Categorical(["a", "b"], priors=[0.98, 0.02])

    counts = Counter(dimension.sample() for _ in range(4000))

    assert counts["a"] > 3500
    assert counts["b"] < 500


def test_categorical_random_state_zero_uses_numpy_generator():
    """random_state=0 must not be treated as falsy and silently ignored (#314)."""
    dimension = Categorical(["a", "b"], random_state=0)

    assert dimension.rng is not None


@pytest.mark.parametrize(
    "choices, priors, random_state",
    [
        ([True, False], None, None),
        ([True, False], None, 1),
        ([1, 2, 3], None, None),
        ([1, 2, 3], None, 1),
        (["a", "b", "c"], [0.5, 0.3, 0.2], None),
        (["a", "b", "c"], [0.5, 0.3, 0.2], 1),
    ],
)
def test_categorical_sample_returns_native_python_types(choices, priors, random_state):
    """sample() must return the exact stored object, not a numpy scalar (#324).

    A seeded Categorical previously sampled via ``self.rng.choice(self.choices, ...)``,
    which numpy coerces to one of its own scalar types (e.g. ``np.True_``,
    ``np.int64``) -- unlike the unseeded ``random``-module path, which returns the
    original Python object untouched. The return type should not depend on
    whether a seed was given.
    """
    dimension = Categorical(choices, priors=priors, random_state=random_state)
    expected_type = type(choices[0])

    for _ in range(20):
        value = dimension.sample()
        assert value in choices
        assert type(value) is expected_type, f"expected {expected_type}, got {type(value)}"


@pytest.mark.parametrize(
    "data_object, parameters",
    [
        (Integer, {"lower": 1, "upper": 1000}),
        (Continuous, {"lower": 0.0, "upper": 1.0}),
    ],
)
def test_integer_and_continuous_random_state_zero_uses_numpy_generator(data_object, parameters):
    """random_state=0 must not be treated as falsy and silently ignored (#322)."""
    dimension = data_object(random_state=0, **parameters)

    assert dimension.rng is not None


@pytest.mark.parametrize(
    "data_object, parameters",
    [
        (Integer, {"lower": 1, "upper": 1000}),
        (Continuous, {"lower": 0.0, "upper": 1.0}),
    ],
)
def test_integer_and_continuous_random_state_zero_is_deterministic(data_object, parameters):
    """Same random_state=0 yields the same sample sequence (#322)."""
    first = data_object(random_state=0, **parameters)
    second = data_object(random_state=0, **parameters)

    seq_first = [first.sample() for _ in range(5)]
    seq_second = [second.sample() for _ in range(5)]

    assert seq_first == seq_second


def test_space_classes_have_complete_type_annotations():
    """random_state and sample() carry annotations on every space class (#209)."""
    import inspect

    for cls, sample_return in [(Integer, int), (Continuous, float), (Categorical, Any)]:
        init_params = inspect.signature(cls.__init__).parameters
        assert init_params["random_state"].annotation is not inspect.Parameter.empty
        sample_sig = inspect.signature(cls.sample)
        assert sample_sig.return_annotation is sample_return

    # The abstract base also annotates its sample() return type.
    assert inspect.signature(BaseDimension.sample).return_annotation is Any


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


def test_estimator_category_lookup_reuses_constructor_params(monkeypatch):
    from sklearn.base import clone
    from sklearn.linear_model import Ridge

    choices = [None, Ridge(alpha=0.5), Ridge(alpha=1), Ridge(alpha=1.0)]
    dimension = Categorical(choices)
    value = clone(choices[-1])
    original_get_params = Ridge.get_params
    reads = Counter()

    def counted_params(estimator, deep=True):
        reads[id(estimator)] += 1
        return original_get_params(estimator, deep=deep)

    monkeypatch.setattr(Ridge, "get_params", counted_params)
    expected_reads = Counter(id(model) for model in [value, *choices[1:]])

    assert dimension._choice_index(value) == 3
    assert reads == expected_reads

    # A later lookup must see changes and retain the first compatible choice
    # when there is no strict match.
    choices[-1].set_params(alpha=2.0)
    reads.clear()
    assert dimension._choice_index(value) == 2
    assert reads == expected_reads


def test_candidate_keys_use_canonical_names_and_require_one_value_per_dimension():
    grid = {"last": Integer(1, 8), "first": Integer(1, 8)}
    space = Space(grid)
    reordered = Space(dict(reversed(list(grid.items()))))

    key = space._candidate_key(iter([7, np.int64(2)]))
    assert key == (("first", 2), ("last", 7))
    assert type(key[0][1]) is np.int64
    assert key == reordered._candidate_key(iter([np.int64(2), 7]))
    for values in ([7], [7, 2, 3]):
        with pytest.raises(ValueError, match="zip"):
            space._candidate_key(iter(values))


def test_nan_categorical_keys_preserve_scalar_types_and_bits_after_pickle():
    import pickle
    from struct import unpack

    different_payload = unpack(">d", bytes.fromhex("7ff8000000000001"))[0]
    choices = [
        float("nan"),
        float("-nan"),
        different_payload,
        np.float16("nan"),
        np.float32("nan"),
        np.float64("nan"),
    ]
    space = Space({"missing": Categorical(choices + [float("nan")])})
    restored = pickle.loads(pickle.dumps(space))

    keys = [space._candidate_key([choice]) for choice in choices]
    assert len(set(keys)) == len(choices)
    for index, choice in enumerate(choices):
        assert restored["missing"]._choice_index(choice) == index
        assert restored._candidate_key([choice]) == keys[index]
    assert restored._population_key([float("nan")]) == keys[0]
    warm = restored.sample_warm_start({"missing": float("nan")})
    assert type(warm["missing"]) is float
    unlisted = unpack(">d", bytes.fromhex("7ff8000000000002"))[0]
    with pytest.raises(ValueError, match="outside its search space"):
        restored.sample_warm_start({"missing": unlisted})


def test_sample_warm_start_rejects_unknown_key():
    """A misspelled hyperparameter name is reported clearly (issue #220)."""
    space = _warm_start_space()
    with pytest.raises(ValueError) as excinfo:
        space.sample_warm_start({"max_depths": 5})  # typo: should be max_depth
    message = str(excinfo.value)
    assert "max_depths" in message
    assert "not in the search space" in message
    assert "max_depth" in message  # the valid name is suggested


def test_sample_warm_start_rejects_non_dict_config():
    space = _warm_start_space()
    with pytest.raises(ValueError, match="must be a dict"):
        space.sample_warm_start([("max_depth", 5)])


def test_sample_warm_start_rejects_value_outside_space():
    """A value outside its dimension is reported clearly (issue #220)."""
    space = _warm_start_space()  # max_depth Integer(2, 20), criterion gini/entropy
    with pytest.raises(ValueError, match="outside its search space"):
        space.sample_warm_start({"max_depth": 999})
    with pytest.raises(ValueError, match="outside its search space"):
        space.sample_warm_start({"criterion": "log_loss"})


def test_sample_warm_start_allows_missing_keys():
    """Missing keys are intentionally allowed and filled by sampling."""
    space = _warm_start_space()
    sampled = space.sample_warm_start({"max_depth": 5})  # criterion omitted
    assert sampled["max_depth"] == 5
    assert sampled["criterion"] in ["gini", "entropy"]


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

    message = str(excinfo.value)
    assert "alpha uses scipy.stats.expon" in message
    assert (
        "Supported scipy distributions are randint, uniform, loguniform, and reciprocal" in message
    )
    assert "define the search space manually" in message
    assert "Continuous(lower, upper)" in message
    # Guidance for users who actually wanted a small set of choices (#260).
    assert "Categorical([...])" in message
    assert "choices" in message


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


def test_log_uniform_accepts_positive_lower_bound():
    dimension = Continuous(1e-4, 1.0, distribution="log-uniform", random_state=0)
    for _ in range(20):
        assert 1e-4 <= dimension.sample() <= 1.0


def test_seeded_fixed_log_dimension_samples_its_only_value():
    assert Continuous(0.01, 0.01, distribution="log-uniform", random_state=0).sample() == 0.01


@pytest.mark.parametrize("bound", [2**80 + 1, -(2**80 + 1)])
def test_fixed_integer_samples_exact_python_integer(bound):
    value = Integer(bound, bound).sample()
    assert type(value) is int
    assert value == bound


@pytest.mark.parametrize(
    "lower, upper, value, expected",
    [
        (-3, 5, -4, -3),
        (-3, 5, 5, 5),
        (-3, 5, 6, 5),
        (2**100, 2**100 + 10, 2**100 + 1, 2**100 + 1),
        (0, 2**64 - 1, np.uint64(2**64 - 1), 2**64 - 1),
        (0, 3, False, 0),
        (False, True, 2, 1),
        (-5, 5, np.float64(-2.5), -2),
        (-5, 5, 3.5, 4),
    ],
)
def test_integer_repair_preserves_bounds_rounding_and_python_type(lower, upper, value, expected):
    repaired = Integer(lower, upper)._round_and_clip(value)
    assert type(repaired) is int
    assert repaired == expected


@pytest.mark.parametrize("bound", [np.inf, -np.inf, 2**53 + 1, 10**400])
def test_fixed_continuous_bound_requires_an_exact_finite_float(bound):
    with pytest.raises(ValueError, match="finite, exactly representable"):
        Continuous(bound, bound)


@pytest.mark.parametrize(
    "bounds, message",
    [
        ((np.nan, 1.0), "lower bound"),
        ((0.0, np.nan), "upper bound"),
        ((-np.inf, 1.0), "lower bound"),
        ((0.0, np.inf), "upper bound"),
        ((0, 10**400), "upper bound"),
        ((2**54 + 1, 2**54 + 3), "representable float"),
        ((-1.5e308, 1.5e308), "finite span"),
    ],
)
def test_invalid_continuous_endpoints_are_rejected(bounds, message):
    with pytest.raises(ValueError, match=message):
        Continuous(*bounds)


@pytest.mark.parametrize(
    "bounds, expected",
    [((2**53, 2**53 + 1), 2**53), ((2**53 + 1, 2**53 + 2), 2**53 + 2)],
)
@pytest.mark.parametrize("distribution", ["uniform", "log-uniform"])
def test_nonrepresentable_integer_bound_keeps_the_float_inside(bounds, expected, distribution):
    dimension = Continuous(*bounds, distribution=distribution, random_state=19)
    assert (dimension.lower, dimension.upper) == bounds
    assert {dimension.sample() for _ in range(8)} == {float(expected)}
    assert dimension._clip_scalar(float("-inf")) == expected
    assert dimension._clip_scalar(float("inf")) == expected


def test_deserialized_continuous_bounds_are_checked_when_building_space():
    dimension = Continuous(0.0, 1.0)
    dimension.upper = np.inf
    with pytest.raises(ValueError, match="upper bound"):
        Space({"alpha": dimension})


@pytest.mark.parametrize("random_state", [None, 19])
def test_log_uniform_sampling_covers_bounds_when_logarithms_tie(random_state, monkeypatch):
    rng = np.random.RandomState(19)
    monkeypatch.setattr("sklearn_genetic.space.space.check_random_state", lambda _: rng)
    lower = 1e308
    upper = np.nextafter(lower, np.inf)
    assert np.log(lower) == np.log(upper)
    dimension = Continuous(lower, upper, distribution="log-uniform", random_state=random_state)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        assert {dimension.sample() for _ in range(64)} == {lower, upper}


@pytest.mark.parametrize("bounds", [(-1.0, 1.0), (-0.0, 1.0), (0.0, 0.0), (-2e30, 2e30)])
@pytest.mark.parametrize("dtype", [float, np.float32, np.float64])
def test_continuous_scalar_clipping_matches_numpy(bounds, dtype):
    dimension = Continuous(*bounds)
    for sample in (-np.inf, -2.0, -1.0, -0.0, 0.0, 0.5, 1.0, 2.0, np.inf, np.nan):
        value = dtype(sample)
        expected = float(np.clip(value, *bounds))
        actual = dimension._clip_scalar(value)
        np.testing.assert_equal(actual, expected)
        if expected == 0:
            assert np.signbit(actual) == np.signbit(expected)


@pytest.mark.parametrize("weights", [np.array([1.0, 2.0]), np.array([1.0, np.nan])])
@pytest.mark.parametrize("random_state", [0, np.random.RandomState(3), np.random.default_rng(3)])
def test_warm_start_accepts_cloned_nested_estimator_configuration(random_state, weights):
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    class LocalTransformer(FunctionTransformer):
        pass

    candidate = Pipeline(
        [
            (
                "transform",
                LocalTransformer(kw_args={"weights": weights, "random_state": random_state}),
            )
        ]
    )
    space = Space({"model": Categorical([candidate, "passthrough"])})
    copied = clone(candidate)
    assert space.sample_warm_start({"model": copied})["model"] is copied
    copied.set_params(
        transform__kw_args={"weights": np.array([1.0, 3.0]), "random_state": random_state}
    )
    with pytest.raises(ValueError, match="outside"):
        space.sample_warm_start({"model": copied})
