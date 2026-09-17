"""The numba cache fallback.

A cache is an optimisation. Losing it should cost startup time, not the
ability to start at all -- which is what happened when a full disk left numba
with nowhere to write: every module-scope ``@njit(cache=True)`` raised during
import and the CLI could not run.
"""

import logging

import pytest

from helicon.lib.numba_compat import cached_jit


def _jit_that_cannot_cache(calls):
    def jit(**kwargs):
        calls.append(kwargs)
        if kwargs.get("cache"):
            raise RuntimeError(
                "cannot cache function 'f': no locator available for file 'x.py'"
            )
        return lambda func: func

    return jit


def _jit_that_caches(calls):
    def jit(**kwargs):
        calls.append(kwargs)
        return lambda func: func

    return jit


def test_caching_is_requested_when_it_works():
    calls = []
    cached_jit(_jit_that_caches(calls), nogil=True)(lambda x: x)
    assert calls == [dict(cache=True, nogil=True)]


def test_falls_back_to_no_caching_rather_than_raising():
    calls = []
    f = cached_jit(_jit_that_cannot_cache(calls), nogil=True)(lambda x: x * 2)
    assert [c["cache"] for c in calls] == [True, False]
    assert f(21) == 42


def test_other_keywords_survive_the_fallback():
    calls = []
    cached_jit(
        _jit_that_cannot_cache(calls), nopython=True, nogil=True, parallel=False
    )(lambda: None)
    for call in calls:
        assert call["nopython"] and call["nogil"] and call["parallel"] is False


def test_a_real_compilation_error_still_raises():
    """Only the cache failure is tolerated. A function numba genuinely cannot
    compile must still fail loudly, or the fallback would hide real bugs."""

    def jit(**kwargs):
        def decorate(func):
            raise TypeError("cannot determine Numba type")

        return decorate

    with pytest.raises(TypeError):
        cached_jit(jit)(lambda: None)


def test_the_fallback_is_reported(caplog):
    import helicon.lib.numba_compat as compat

    compat._warned = False
    with caplog.at_level(logging.WARNING):
        cached_jit(_jit_that_cannot_cache([]))(lambda: None)
    assert "cache" in caplog.text.lower()
    assert "NUMBA_CACHE_DIR" in caplog.text


def test_it_is_reported_once_not_once_per_function(caplog):
    """All such functions fail together for the same reason; repeating the
    message per function would bury the cause."""
    import helicon.lib.numba_compat as compat

    compat._warned = False
    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            cached_jit(_jit_that_cannot_cache([]))(lambda: None)
    assert caplog.text.lower().count("numba cannot write") == 1
