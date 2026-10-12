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


class TestLazyJit:
    """numba is imported, and a function compiled, on its first call: not with helicon."""

    def test_importing_helicon_does_not_import_the_heavy_packages(self):
        import subprocess
        import sys

        code = (
            "import sys, helicon; "
            "print(','.join(m for m in ('numba', 'curvepy', 'h5py') if m in sys.modules))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == ""

    def test_the_first_call_compiles_with_numba(self):
        pytest.importorskip("numba")
        import numpy as np

        from helicon.lib.numba_compat import lazy_jit

        @lazy_jit("njit")
        def total(a):
            s = 0.0
            for x in a:
                s += x
            return s

        assert total(np.arange(5.0)) == 10.0
        assert total.py_func(np.arange(5.0)) == 10.0
        import numba

        # what the wrapper calls now is numba's dispatcher, not the Python function
        held = [
            c.cell_contents
            for c in total.__closure__
            if isinstance(c.cell_contents, list)
        ]
        assert isinstance(held[0][0], numba.core.registry.CPUDispatcher)

    def test_prange_becomes_numba_prange_when_compiled(self):
        pytest.importorskip("numba")
        import numba

        from helicon.lib import transforms

        assert transforms.prange in (range, numba.prange)
        import numpy as np

        data = np.zeros((4, 4, 4), dtype=np.float32)
        data[2, 2, 2] = 1.0
        transforms.apply_helical_symmetry(data, 1.0, 30.0, 1.0)
        assert transforms.prange is numba.prange

    def test_without_numba_it_runs_as_python(self, monkeypatch, caplog):
        import builtins

        from helicon.lib import numba_compat

        real_import = builtins.__import__

        def no_numba(name, *args, **kwargs):
            if name == "numba":
                raise ImportError("no numba here")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_numba)

        @numba_compat.lazy_jit("njit")
        def double(x):
            return 2 * x

        with caplog.at_level(logging.WARNING, logger=numba_compat.__name__):
            assert double(21) == 42
        assert "numba is not installed" in caplog.text
