"""The conftest hook that runs the browser (pytest-playwright) tests last."""

import conftest


class _Item:
    def __init__(self, name, fixturenames):
        self.name = name
        self.fixturenames = fixturenames


class TestBrowserTestsRunLast:
    def test_uses_playwright(self):
        assert conftest._uses_playwright(_Item("a", ["page", "tmp_path"]))
        assert conftest._uses_playwright(_Item("b", ["browser"]))
        assert not conftest._uses_playwright(_Item("c", ["tmp_path"]))
        assert not conftest._uses_playwright(object())

    def test_moves_browser_tests_to_the_end_keeping_order(self):
        items = [
            _Item("p1", ["page"]),
            _Item("a", []),
            _Item("p2", ["context"]),
            _Item("b", ["tmp_path"]),
            _Item("p3", ["page", "tmp_path"]),
        ]
        conftest.pytest_collection_modifyitems(None, items)
        assert [i.name for i in items] == ["a", "b", "p1", "p2", "p3"]

    def test_no_browser_tests_leaves_order_alone(self):
        items = [_Item("b", []), _Item("a", [])]
        conftest.pytest_collection_modifyitems(None, items)
        assert [i.name for i in items] == ["b", "a"]
