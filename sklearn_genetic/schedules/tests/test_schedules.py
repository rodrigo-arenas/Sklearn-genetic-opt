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
    possible_messages = [
        "Can't instantiate abstract class DummyAdapter with abstract methods step",
        "Can't instantiate abstract class DummyAdapter with abstract method step",
    ]

    class DummyAdapter(BaseAdapter):
        def __init__(self, initial_value, end_value, adaptive_rate, **kwargs):
            super().__init__(initial_value, end_value, adaptive_rate, **kwargs)

        def run(self):
            pass

    with pytest.raises(Exception) as excinfo:
        dummy_adapter = DummyAdapter(0.1, 0.2, 0.5)
        dummy_adapter.run()

    assert any([str(excinfo.value) == i for i in possible_messages])


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

    with pytest.raises(Exception) as excinfo:
        check_adapter(True)
    assert (
        str(excinfo.value)
        == "adapter should be either a class with inheritance from schedulers.base.BaseAdapter "
        "or a real number."
    )


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
