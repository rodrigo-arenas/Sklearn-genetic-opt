from ..utils.tools import check_bool_individual


def test_bool_array():
    assert sum(check_bool_individual([1, 1, 1, 0, 1, 0, 1, 0])) == 5
    assert sum(check_bool_individual([0, 0, 0, 0, 0, 0])) == 1
