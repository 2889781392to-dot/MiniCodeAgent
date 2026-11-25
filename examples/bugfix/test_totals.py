from totals import total_even


def test_empty():
    assert total_even([]) == 0


def test_ignores_odd_values():
    assert total_even([2, 3, 4]) == 6
