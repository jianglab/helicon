"""Looking for code in a source file without being tied to how it is laid out.

Some tests check that a piece of wiring is there (a call made with a given first argument, a
test in a given order) by looking for text in a source file. Where the text spans lines, its
indentation and line breaks are the formatter's, and a harmless re-indent broke the test. Here
all runs of white space count as one space, so the text is found wherever it is laid out.
"""

import re


def squash(text: str) -> str:
    """The text with every run of white space as one space."""
    return re.sub(r"\s+", " ", text)


def contains(source: str, needle: str) -> bool:
    """Whether ``needle`` is in ``source``, whatever the line breaks and indentation."""
    return squash(needle) in squash(source)


def count(source: str, needle: str) -> int:
    """How many times ``needle`` is in ``source``, whatever the layout."""
    return squash(source).count(squash(needle))


def file_source(path) -> str:
    """The source of a file, with that of the file split off from it (``<name>_ui.py`` next to
    ``<name>_tab.py``) *before* it, as the page came before the server in the tab it was split from.
    """
    from pathlib import Path

    path = Path(path)
    stem = path.stem[:-4] if path.stem.endswith("_tab") else path.stem
    parts = []
    sibling = path.with_name(stem + "_ui.py")
    if sibling.is_file() and sibling != path:
        parts.append(sibling.read_text(encoding="utf-8"))
    parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def tab_source(obj) -> str:
    """The source of a module with what was split off from it (see :func:`file_source`), or, for a
    function or a class, ``inspect.getsource``.

    A tab of the web apps keeps its page in ``<name>_ui.py`` next to it (so the file of the tab is
    not 4,000 lines); the tests that look for what is wired in the page look in both.
    """
    import inspect

    if not inspect.ismodule(obj):
        return inspect.getsource(obj)
    return file_source(obj.__file__)
