import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from helicon.lib import util


class TestUtil(object):
    def test_parse_param_str(self):
        assert util.parse_param_str("a=b:c=d,e") == (None, {"a": "b", "c": "d,e"})
        assert util.parse_param_str("opt:a=b:c=d,e") == ("opt", {"a": "b", "c": "d,e"})
        assert util.parse_param_str('a=1:b=2.0:c=True:d="hello"') == (
            None,
            {"a": 1, "b": 2.0, "c": 1, "d": "hello"},
        )

    def test_validate_param_dict(self):
        param_ref = {"a": 1, "b": 2.0, "c": "hello"}
        param = {"a": "2", "b": 3.0}
        final_param, changed, unsupported = util.validate_param_dict(param, param_ref)
        assert final_param == {"a": 2, "b": 3.0, "c": "hello"}
        assert changed == {"a": 2, "b": 3.0}
        assert unsupported == {}

        param = {"a": "2", "d": "world"}
        final_param, changed, unsupported = util.validate_param_dict(param, param_ref)
        assert final_param == {"a": 2, "b": 2.0, "c": "hello"}
        assert changed == {"a": 2}
        assert unsupported == {"d": "world"}

    def test_set_angle_range(self):
        assert util.set_angle_range(200, range=[-180, 180]) == pytest.approx(-160)
        assert util.set_angle_range(-200, range=[-180, 180]) == pytest.approx(160)
        assert util.set_angle_range(20, range=[-180, 180]) == pytest.approx(20)

    def test_set_to_periodic_range(self):
        assert util.set_to_periodic_range(200, min=-180, max=180) == pytest.approx(-160)
        assert util.set_to_periodic_range(-200, min=-180, max=180) == pytest.approx(160)
        assert util.set_to_periodic_range(20, min=-180, max=180) == pytest.approx(20)

    def test_flatten(self):
        assert util.flatten([1, [2, 3], [4, [5, 6]]]) == [1, 2, 3, 4, 5, 6]
        assert util.flatten((1, (2, 3), (4, (5, 6)))) == (1, 2, 3, 4, 5, 6)

    def test_unique(self):
        assert util.unique([1, 2, 2, 3, 1, 4]) == [1, 2, 3, 4]

    def test_order_by_unique_counts(self):
        labels = np.array([1, 1, 2, 2, 2, 3, 3, 3, 3, -1])
        ordered_labels = util.order_by_unique_counts(labels)
        assert [int(x) for x in ordered_labels] == [2, 2, 1, 1, 1, 0, 0, 0, 0, -1]

    def test_split_array(self):
        arr = [1, 2, 3, 4, 5]
        group1, group2 = util.split_array(arr)
        assert sum(arr[i] for i in group1) == 7
        assert sum(arr[i] for i in group2) == 8

    def test_assign_to_groups(self):
        numbers = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
        group_size = 3
        result = util.assign_to_groups(numbers, group_size)
        assert result == {1: 1, 2: 1, 3: 2, 4: 3}

    def test_timer(self):
        with util.Timer(verbose=0) as t:
            pass
        assert t.interval > 0

    def test_cache(self):
        mock_func = MagicMock()
        mock_func.return_value = 42

        cached_func = util.cache(expires_after=1)(mock_func)

        # Call the function twice
        result1 = cached_func()
        result2 = cached_func()

        # Check that the mock function was only called once
        mock_func.assert_called_once()
        assert result1 == 42
        assert result2 == 42
