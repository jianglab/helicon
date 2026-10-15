"""The list of slow tests names tests that are still there."""

from pathlib import Path

import conftest

TESTS = Path(__file__).parent


def test_every_listed_test_is_in_its_file():
    ids = conftest._slow_ids()
    assert ids, "slow_tests.txt is empty or missing"
    stale = []
    for nodeid in sorted(ids):
        path, *names = nodeid.split("::")
        source = (TESTS.parent / path).read_text()
        if not all(
            f"def {name}" in source or f"class {name}" in source for name in names
        ):
            stale.append(nodeid)
    assert (
        not stale
    ), f"not in the test files any more (run update_slow_tests.py): {stale}"


def test_the_markers_are_declared(pytestconfig):
    declared = "\n".join(pytestconfig.getini("markers"))
    assert "slow:" in declared and "browser:" in declared


def test_the_listed_tests_are_marked():
    # the hook itself: a test id of the list is marked, another is not
    class Item:
        def __init__(self, nodeid, fixtures=()):
            self.nodeid, self.fixturenames, self.marks = nodeid, fixtures, []

        def add_marker(self, mark):
            self.marks.append(mark.name)

    first = sorted(conftest._slow_ids())[0]
    items = [
        Item(first + "[case]"),
        Item("tests/test_x.py::test_y"),
        Item("tests/test_x.py::test_p", ("page",)),
    ]
    conftest.pytest_collection_modifyitems(None, items)
    assert items[0].marks == ["slow"]
    assert items[1].marks == []
    assert items[2].marks == ["browser"]
