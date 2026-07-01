import pytest
from ..schedulers import (
    BaseAdapter,
    ConstantAdapter,
    ExponentialAdapter,
    InverseAdapter,
    PotentialAdapter,
)
from ..validations import check_adapter


def test_base_scheduler_attributes():
    assert hasattr(BaseAdapter, "step")


def test_wrong_scheduler_methods():
    class DummyAdapter(BaseAdapter):
        def __init__(self, initial_value, end_value, adaptive_rate, **kwargs):
            super().__init__(initial_value, end_value, adaptive_rate, **kwargs)

        def run(self):
            pass

    with pytest.raises(Exception) as excinfo:
        dummy_adapter = DummyAdapter(0.1, 0.2, 0.5)
        dummy_adapter.run()

    message = str(excinfo.value)
    assert "Can't instantiate abstract class DummyAdapter" in message
    assert "step" in message


def test_check_adapter():
    regular_adapters = [
        ExponentialAdapter(0.1, 0.5, 0.01),
        InverseAdapter(0.1, 0.5, 0.01),
        PotentialAdapter(0.1, 0.5, 0.01),
    ]
    for adapter in regular_adapters:
        assert check_adapter(adapter) == adapter

    constant_adapter = check_adapter(0.6)

    assert isinstance(constant_adapter, ConstantAdapter)
    for _ in range(10):
        assert constant_adapter.step() == 0.6

    with pytest.raises(ValueError) as excinfo:
        check_adapter(True)

    message = str(excinfo.value)
    # The received type is reported so misconfigurations are easy to debug ...
    assert "bool" in message
    # ... and the expected adapter base class is still mentioned.
    assert "schedulers.base.BaseAdapter" in message


class _NotAnAdapter:
    """A plain object that does not inherit from ``BaseAdapter``."""


@pytest.mark.parametrize(
    "invalid_adapter, expected_type_name",
    [
        (True, "bool"),
        ("constant", "str"),
        ([0.5], "list"),
        (None, "NoneType"),
        (_NotAnAdapter(), "_NotAnAdapter"),
    ],
)
def test_check_adapter_invalid_object_error_message(invalid_adapter, expected_type_name):
    # 1. Invalid objects raise a ValueError.
    with pytest.raises(ValueError) as excinfo:
        check_adapter(invalid_adapter)

    message = str(excinfo.value)
    # 2. The message names the received type so the mistake is obvious.
    assert expected_type_name in message
    # 3. The message still points users to the expected adapter base class.
    assert "schedulers.base.BaseAdapter" in message


def test_scheduler_reset():
    scheduler = ExponentialAdapter(initial_value=0.8, end_value=0.2, adaptive_rate=0.1)

    scheduler.step()
    scheduler.step()

    assert scheduler.current_step == 2
    assert scheduler.current_value != scheduler.initial_value

    assert scheduler.reset() is scheduler
    assert scheduler.current_step == 0
    assert scheduler.current_value == scheduler.initial_value


@pytest.mark.parametrize(
    "method, params",
    [
        (
            ExponentialAdapter,
            {"initial_value": 0.6, "end_value": 0.4, "adaptive_rate": 0.01},
        ),
        (
            ExponentialAdapter,
            {"initial_value": 0.6, "end_value": 0.4, "adaptive_rate": 0},
        ),
        (
            InverseAdapter,
            {"initial_value": 0.6, "end_value": 0.4, "adaptive_rate": 0.01},
        ),
        (InverseAdapter, {"initial_value": 0.6, "end_value": 0.4, "adaptive_rate": 0}),
        (
            PotentialAdapter,
            {"initial_value": 0.6, "end_value": 0.4, "adaptive_rate": 0.01},
        ),
        (
            PotentialAdapter,
            {"initial_value": 0.6, "end_value": 0.4, "adaptive_rate": 0},
        ),
    ],
)
def test_scheduler_decay_limits(method, params):
    scheduler = method(**params)
    for _ in range(100):
        scheduler.step()

    assert scheduler.current_value >= params["end_value"]
    assert scheduler.current_value <= params["initial_value"]


@pytest.mark.parametrize(
    "method, params",
    [
        (
            ExponentialAdapter,
            {"initial_value": 0.2, "end_value": 0.5, "adaptive_rate": 0.01},
        ),
        (
            ExponentialAdapter,
            {"initial_value": 0.2, "end_value": 0.5, "adaptive_rate": 0},
        ),
        (
            InverseAdapter,
            {"initial_value": 0.2, "end_value": 0.5, "adaptive_rate": 0.01},
        ),
        (InverseAdapter, {"initial_value": 0.2, "end_value": 0.5, "adaptive_rate": 0}),
        (
            PotentialAdapter,
            {"initial_value": 0.2, "end_value": 0.5, "adaptive_rate": 0.01},
        ),
        (
            PotentialAdapter,
            {"initial_value": 0.2, "end_value": 0.5, "adaptive_rate": 0},
        ),
    ],
)
def test_scheduler_ascent_limits(method, params):
    scheduler = method(**params)
    for _ in range(100):
        scheduler.step()

    assert scheduler.current_value <= params["end_value"]
    assert scheduler.current_value >= params["initial_value"]
