import pytest
from ..schedulers import ExponentialDecay, InverseDecay


@pytest.mark.parametrize(
    "method, params",
    [
        (ExponentialDecay, {"initial_value": 0.6, "decay_rate": 0.01}),
        (
            ExponentialDecay,
            {"initial_value": 0.6, "decay_rate": 0.01, "min_value": 0.4},
        ),
        (ExponentialDecay, {"initial_value": 0.6, "decay_rate": 0}),
        (InverseDecay, {"initial_value": 0.6, "decay_rate": 0.01, "min_value": 0.4}),
        (InverseDecay, {"initial_value": 0.6, "decay_rate": 0.01}),
        (InverseDecay, {"initial_value": 0.6, "decay_rate": 0}),
    ],
)
def test_scheduler_limits(method, params):
    scheduler = method(**params)
    for _ in range(100):
        scheduler.step()

    assert scheduler.current_value >= params.get("min_value", 0)
    assert scheduler.current_value <= params.get("initial_value", 1)
