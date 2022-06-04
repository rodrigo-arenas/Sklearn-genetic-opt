import pytest
from ..schedulers import ExponentialAdapter, InverseAdapter


@pytest.mark.parametrize(
    "method, params",
    [
        (
                ExponentialAdapter,
                {"initial_value": 0.6, "decay_rate": 0.01, "end_value": 0.4},
        ),
        (ExponentialAdapter, {"initial_value": 0.6, "end_value": 0.4, "decay_rate": 0}),
        (InverseAdapter, {"initial_value": 0.6, "end_value": 0.4, "decay_rate": 0.01}),
        (InverseAdapter, {"initial_value": 0.6, "end_value": 0.4, "decay_rate": 0}),
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
                {"initial_value": 0.2, "decay_rate": 0.01, "end_value": 0.5},
        ),
        (ExponentialAdapter, {"initial_value": 0.2, "end_value": 0.5, "decay_rate": 0}),
        (InverseAdapter, {"initial_value": 0.2, "end_value": 0.5, "decay_rate": 0.01}),
        (InverseAdapter, {"initial_value": 0.2, "end_value": 0.5, "decay_rate": 0}),
    ],
)
def test_scheduler_ascent_limits(method, params):
    scheduler = method(**params)
    for _ in range(100):
        scheduler.step()

    assert scheduler.current_value <= params["end_value"]
    assert scheduler.current_value >= params["initial_value"]
