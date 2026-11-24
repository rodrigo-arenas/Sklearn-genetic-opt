import random

import pytest

from ..utils.tools import check_bool_individual, cxUniform


def test_bool_array():
    assert sum(check_bool_individual([1, 1, 1, 0, 1, 0, 1, 0])) == 5
    assert sum(check_bool_individual([0, 0, 0, 0, 0, 0])) == 1


@pytest.mark.parametrize("probability", [0.0, 0.2, 0.8, 1.0])
def test_feature_uniform_crossover_matches_deap_on_wide_masks(probability):
    from deap import tools

    first = [0, 1, 0, 0] * 513
    second = [0, 0, 1, 0] * 514
    state = random.getstate()
    try:
        random.seed(4451)
        expected = tuple(
            map(
                check_bool_individual,
                tools.cxUniform(first.copy(), second.copy(), probability),
            )
        )
        expected_state = random.getstate()
        random.seed(4451)
        crossed = cxUniform(first, second, probability)
        assert crossed[0] is first and crossed[1] is second
        assert crossed == expected
        assert random.getstate() == expected_state
    finally:
        random.setstate(state)
