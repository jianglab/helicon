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
