"""Faster start of the tabs: a cheaper way for Shiny to name its reactive values.

Shiny (1.6) names every ``reactive.value`` it creates without a ``name=`` after the
variable it is assigned to, for its OpenTelemetry logs and debugging. It finds
the assignment with ``inspect.stack()``, which builds the source context of every
frame on the call stack: in the deep stack of a running server that took about
160 ms for each value, and HILL creates 49 when it starts (8 of the 12 seconds of
its server start). This replaces that with a walk up the frames that reads only
the one source line it needs, and finds the same names.
"""

from __future__ import annotations

import logging
import re
import sys

logger = logging.getLogger(__name__)

# the same patterns as Shiny's: `name = [reactive.]value(` and `self.name = ...`,
# a type annotation allowed before the `=`
_ASSIGNED = re.compile(r"^(\w+)\s*(?::[^=]+)?\s*=\s*(?:reactive\.)?[Vv]alue\s*[\[\(]")
_ATTRIBUTE = re.compile(
    r"^\w+\.(\w+)\s*(?::[^=]+)?\s*=\s*(?:reactive\.)?[Vv]alue\s*[\[\(]"
)
_BARE = re.compile(r"\b[Vv]alue\b")

_applied = False
# Shiny's own, kept so that it can be compared with or restored
_original = None


def _fast_try_infer_name(self):
    """``Value._try_infer_name`` without ``inspect.stack()``: the name, or None."""
    import linecache

    from shiny.reactive._utils import is_user_code_frame

    try:
        frame = sys._getframe(2)  # [0] this, [1] Value.__init__, [2] the caller
        while frame is not None:
            filename = frame.f_code.co_filename
            if not is_user_code_frame(filename):
                frame = frame.f_back
                continue
            line = linecache.getline(filename, frame.f_lineno).strip()
            if not line:
                return None
            match = _ASSIGNED.match(line) or _ATTRIBUTE.match(line)
            if match:
                return match.group(1)
            if _BARE.search(line):
                # a multiline assignment: the target is on a line before
                return self._try_infer_name_from_preceding_lines(
                    filename, frame.f_lineno
                )
            frame = frame.f_back
    except Exception:
        return None
    return None


def apply():
    """Use the fast name inference in Shiny's reactive values; once, and only where it fits.

    Returns
    -------
    bool
        Whether it is in use. Nothing is changed when this Shiny has no such method,
        or not the helpers it needs.
    """
    global _applied
    if _applied:
        return True
    try:
        from shiny.reactive import _reactives, _utils
    except ImportError:
        return False
    value = getattr(_reactives, "Value", None)
    if not (
        value is not None
        and hasattr(value, "_try_infer_name")
        and hasattr(value, "_try_infer_name_from_preceding_lines")
        and hasattr(_utils, "is_user_code_frame")
    ):
        logger.debug("shiny has changed: its reactive values are named its own way")
        return False
    global _original
    if _original is None:
        _original = value._try_infer_name
    value._try_infer_name = _fast_try_infer_name
    _applied = True
    return True
