"""Alive-window trackers for the display command.

Each display category (napari viewer, gallery, plot, text, images2star,
proc3d) keeps a ``_DisplayTracker`` so that "reuse the most recently used
alive window" works consistently across every button. The mode-name to
tracker mapping in ``_TRACKER_FOR`` drives the file-browser dispatch.
"""

from __future__ import annotations

import time


class _DisplayTracker:
    """Track alive windows of a single display category.

    Each tracked window carries a ``time.monotonic()`` timestamp that
    records when it was created or last activated (brought to front).
    ``active()`` returns the window with the most recent timestamp.

    Timeline of approaches tried (for the record):

    1. ``owns_fn`` + ``app.focusChanged`` → ``on_focus()``
       Problem: gallery windows display non-focusable image widgets, so
       clicking them never changes the application *focus widget* — the
       file browser's QTreeView keeps focus.  ``on_focus`` never matched.

    2. ``changeEvent`` with ``QEvent.Type.WindowActivate``
       Problem: this event type is never delivered on macOS.

    3. ``changeEvent`` with ``QEvent.Type.ActivationChange`` + ``isActiveWindow()``
       Works on all platforms.  When a window is brought to front Qt sends
       ``ActivationChange``, and ``isActiveWindow()`` tells us whether
       this window is now the active one.
    """

    def __init__(self, is_alive):
        self._windows: dict = {}  # window → monotonic timestamp
        self._is_alive = is_alive

    def alive(self) -> list:
        """Prune dead windows and return the living ones (newest first)."""
        self._windows = {w: t for w, t in self._windows.items() if self._is_alive(w)}
        return sorted(self._windows, key=self._windows.get, reverse=True)

    def active(self):
        """Return the most-recent alive window, or ``None``."""
        alive = self.alive()
        return alive[0] if alive else None

    def register(self, window):
        """Track a new window with current timestamp."""
        self._windows[window] = time.monotonic()

    def on_activate(self, window):
        """Update the timestamp when *window* is activated (brought to front)."""
        if window in self._windows:
            self._windows[window] = time.monotonic()

    def on_close(self, window):
        """Remove a closed window from tracking."""
        self._windows.pop(window, None)


# ---------------------------------------------------------------------------
# Alive-check helpers used by the trackers
# ---------------------------------------------------------------------------


def _is_alive_viewer(v):
    try:
        w = v.window._qt_window
        return w is not None and w.isVisible()
    except Exception:
        return False


def _is_alive_widget(w):
    try:
        return w is not None and w.isVisible()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Per-category trackers (module-level so gallery_backends can reference them)
# ---------------------------------------------------------------------------

_napari = _DisplayTracker(_is_alive_viewer)
_gallery = _DisplayTracker(_is_alive_widget)
_plot = _DisplayTracker(_is_alive_widget)
_text = _DisplayTracker(_is_alive_widget)
_images2star = _DisplayTracker(_is_alive_widget)
_proc3d = _DisplayTracker(_is_alive_widget)

_NAPARI_MODES = {"slice", "volume", "3dplot", "stats", "html"}
_GALLERY_MODES = {"gallery", "optimiser", "2dclasses", "orthogonal"}
_TEXT_MODES = {"text"}
_PLOT_MODES = {"fsc"}
_IMAGES2STAR_MODES = {"images2star"}
_PROC3D_MODES = {"proc3d"}


def _quit_all_windows():
    """Close every tracked window and the file browser, then quit."""
    from PySide6.QtWidgets import QApplication

    for tracker in (_napari, _gallery, _text, _plot, _images2star, _proc3d):
        for w in list(tracker.alive()):
            try:
                w.close()
            except Exception:
                pass
    for w in QApplication.topLevelWidgets():
        try:
            w.close()
        except Exception:
            pass


def _install_window_shortcuts(window):
    """Install Ctrl+W (close) and Ctrl+Q (quit) on *window*."""
    from PySide6.QtGui import QAction, QShortcut, QKeySequence

    close_sc = QShortcut(QKeySequence("Ctrl+W"), window)
    close_sc.activated.connect(window.close)

    # The file browser already binds Ctrl+Q to its File -> Quit action.
    # Installing a second Ctrl+Q handler in the same window makes the
    # shortcut ambiguous: Qt emits an "Ambiguous shortcut overload" warning
    # and neither binding fires reliably. Skip the shortcut when the window
    # already provides a matching action.
    quit_key = QKeySequence(QKeySequence.StandardKey.Quit)
    if not any(a.shortcut() == quit_key for a in window.findChildren(QAction)):
        quit_sc = QShortcut(quit_key, window)
        quit_sc.activated.connect(_quit_all_windows)


_TRACKER_FOR: dict[str, _DisplayTracker] = {}
for _m in _NAPARI_MODES:
    _TRACKER_FOR[_m] = _napari
for _m in _GALLERY_MODES:
    _TRACKER_FOR[_m] = _gallery
for _m in _TEXT_MODES:
    _TRACKER_FOR[_m] = _text
for _m in _PLOT_MODES:
    _TRACKER_FOR[_m] = _plot
for _m in _IMAGES2STAR_MODES:
    _TRACKER_FOR[_m] = _images2star
for _m in _IMAGES2STAR_MODES:
    _TRACKER_FOR[_m] = _images2star
for _m in _PROC3D_MODES:
    _TRACKER_FOR[_m] = _proc3d
