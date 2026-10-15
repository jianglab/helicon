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
