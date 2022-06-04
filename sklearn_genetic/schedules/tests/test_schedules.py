import pytest
from ..schedulers import (
    BaseAdapter,
    ExponentialAdapter,
    InverseAdapter,
    PotentialAdapter,
)


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
