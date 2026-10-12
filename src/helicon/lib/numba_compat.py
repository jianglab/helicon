"""Numba decorators that survive an unusable on-disk cache.

Numba decides *where* a function's compiled cache will live at decoration
time, not at call time. It tries a fixed list of locations -- ``NUMBA_CACHE_DIR``
if set, then the source tree's ``__pycache__``, then the user-wide cache
directory -- and accepts the first it can both create and write a file into.
If none qualifies it raises ``RuntimeError: cannot cache function ... no
locator available for file ...``.

For a decorator at module scope that error happens during import, so a machine
that merely cannot *write a cache* cannot start the program at all:

    RuntimeError: cannot cache function '_fsc_shell_reduce': no locator
    available for file '.../helicon/lib/analysis.py'

which is what a full disk, a read-only checkout, or a cache directory with
broken ownership looks like from the outside. That is the wrong failure. The
cache is an optimisation -- without it numba recompiles on each run, costing
seconds of startup -- so losing it should cost those seconds, not the session.

Hence :func:`cached_jit`, which asks for caching and quietly settles for none.
It deliberately catches only ``RuntimeError``: a genuinely broken function
should still fail loudly at decoration, and does.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_warned = False


def cached_jit(jit, **kwargs):
    """Apply *jit* with ``cache=True``, falling back to ``cache=False``.

    ``jit`` is the numba decorator factory to use -- ``njit``, or ``jit`` with
    ``nopython=True`` passed through ``kwargs``. Every other keyword is handed
    to it unchanged.

    The fallback is reported once per process rather than once per function:
    there are several such functions and they all fail together, for the same
    reason, so repeating it would only bury the cause.
    """

    def decorate(func):
        global _warned
        try:
            return jit(cache=True, **kwargs)(func)
        except RuntimeError as exc:
            if not _warned:
                _warned = True
                logger.warning(
                    "numba cannot write its compilation cache (%s); continuing "
                    "without it, which costs a few seconds of startup on every "
                    "run. Check free disk space, that the checkout is writable, "
                    "and the permissions on the user-wide numba cache directory "
                    "-- or set NUMBA_CACHE_DIR to somewhere writable.",
                    exc,
                )
            return jit(cache=False, **kwargs)(func)

    return decorate


def has_numba():
    """Whether numba can be imported, without importing it (which takes a second or more)."""
    import importlib.util

    return importlib.util.find_spec("numba") is not None


def set_num_threads(n):
    """``numba.set_num_threads``, importing numba only now; nothing without numba."""
    try:
        import numba
    except ImportError:
        return
    numba.set_num_threads(max(1, min(int(n), int(numba.config.NUMBA_NUM_THREADS))))


def lazy_jit(kind="njit", **kwargs):
    """Compile a function with numba on its first call, not when its module is imported.

    Importing numba takes a second or more, and compiling a decorated function at
    import would make every ``import helicon`` pay for numba whether or not the
    function is ever called. The decorated function is compiled with
    :func:`cached_jit` the first time it is called. A module global ``prange``
    that is the builtin ``range`` (the fallback for running without numba) is
    ``numba.prange`` in the compiled copy of the function; the module's own
    ``prange`` is left alone. Without numba it runs as plain Python, with a
    warning the first time.

    What is returned is a Python function that compiles and calls the real one,
    not a numba dispatcher: it can be called from Python, but not from inside
    another numba-compiled function, which cannot type it. A function that other
    compiled functions call must be decorated with :func:`cached_jit` directly.

    Parameters
    ----------
    kind : str, optional
        The numba decorator: ``"njit"`` or ``"jit"`` (with ``nopython=True`` in
        ``kwargs``). Defaults to ``"njit"``.
    **kwargs
        Passed to the decorator, besides ``cache``.
    """
    import functools
    import threading

    def decorate(func):
        compiled = []
        lock = threading.Lock()

        @functools.wraps(func)
        def wrapper(*args, **kw):
            if not compiled:
                with lock:
                    if not compiled:
                        compiled.append(_compile(func, kind, kwargs))
            return compiled[0](*args, **kw)

        wrapper.py_func = func
        return wrapper

    return decorate


def _compile(func, kind, kwargs):
    try:
        import numba
    except ImportError:
        logger.warning(
            "numba is not installed: %s runs as plain Python, much more slowly. "
            "Install numba ('pip install numba') to speed it up.",
            func.__qualname__,
        )
        return func
    if func.__globals__.get("prange") is range:
        # numba reads globals when it compiles: a copy of the function whose
        # globals say numba.prange, so that the module's own name is not changed
        import types

        copy = types.FunctionType(
            func.__code__,
            {**func.__globals__, "prange": numba.prange},
            func.__name__,
            func.__defaults__,
            func.__closure__,
        )
        copy.__kwdefaults__ = func.__kwdefaults__
        copy.__qualname__ = func.__qualname__
        copy.__module__ = func.__module__
        func = copy
    return cached_jit(getattr(numba, kind), **kwargs)(func)
