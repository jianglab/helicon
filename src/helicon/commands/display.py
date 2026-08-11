#!/usr/bin/env python

"""A file browser for viewing image, map, star, bild, eps, pdf, html, and text files"""

from __future__ import annotations

import sys
import time
import warnings

# Suppress harmless mrcfile divide-by-zero warnings when reading headers.
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in divide",
    category=RuntimeWarning,
    module="mrcfile",
)

# On macOS, set the process name before NSApplication is initialized
# (triggered by PySide6 import below). NSApplication caches the app name
# from getprogname() at first init; after that it can't be changed.
if sys.platform == "darwin":
    try:
        import ctypes
        import ctypes.util

        ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True).setprogname(
            b"Helicon"
        )
    except Exception:
        pass

import argparse
import colorsys
import os
import re
from pathlib import Path

import helicon
from helicon.lib.exceptions import HeliconDependencyError

try:
    from helicon.lib.file_browser import FolderBrowserWidget
except ImportError:
    FolderBrowserWidget = None

# Keep the module importable without the Qt stack (display.py degrades to a
# HeliconDependencyError in main() when PySide6 is missing).  The FSC plot
# window is only ever constructed from Qt code paths.
try:
    from PySide6.QtWidgets import QLayout, QMainWindow, QWidget
    from PySide6.QtCore import QObject, QPoint, QRect, QSize
except ImportError:  # pragma: no cover - only without the Qt stack
    QLayout = None
    QMainWindow = None
    QWidget = None
    QObject = None
    QPoint = None
    QRect = None
    QSize = None

_FscPlotBase = QMainWindow if QMainWindow is not None else object
_FlowLayoutBase = QLayout if QLayout is not None else object
_FlowContainerBase = QWidget if QWidget is not None else object


def _prepare_process_identity() -> None:
    """Set the process name used by Qt and desktop environments."""
    if sys.argv:
        sys.argv[0] = "Helicon"


def _set_application_identity(app) -> None:
    """Set the user-facing application name used by Qt menus and windows."""
    _prepare_process_identity()
    app.setApplicationName("Helicon")
    app.setApplicationDisplayName("Helicon")
    _set_macos_app_identity("Helicon")


def _set_macos_app_identity(name: str = "Helicon") -> None:
    """Set process name and application menu title via macOS Cocoa APIs."""
    if sys.platform != "darwin":
        return
    try:
        import ctypes
        import ctypes.util

        ctypes.cdll.LoadLibrary(ctypes.util.find_library("Foundation"))
        ctypes.cdll.LoadLibrary(ctypes.util.find_library("AppKit"))
        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))

        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]

        msg_send = objc.objc_msgSend
        msg_send.restype = ctypes.c_void_p

        ns_proc_info = objc.objc_getClass(b"NSProcessInfo")
        proc_info_sel = objc.sel_registerName(b"processInfo")
        set_proc_name_sel = objc.sel_registerName(b"setProcessName:")
        ns_string_cls = objc.objc_getClass(b"NSString")
        str_sel = objc.sel_registerName(b"stringWithUTF8String:")

        msg_send.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]
        ns_name = msg_send(ns_string_cls, str_sel, name.encode("utf-8"))

        msg_send.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        proc = msg_send(ns_proc_info, proc_info_sel)
        if proc:
            msg_send.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            msg_send(proc, set_proc_name_sel, ns_name)

        ns_app_cls = objc.objc_getClass(b"NSApplication")
        shared_app_sel = objc.sel_registerName(b"sharedApplication")
        msg_send.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        ns_app = msg_send(ns_app_cls, shared_app_sel)
        if ns_app:
            main_menu_sel = objc.sel_registerName(b"mainMenu")
            main_menu = msg_send(ns_app, main_menu_sel)
            if main_menu:
                item_at_idx_sel = objc.sel_registerName(b"itemAtIndex:")
                msg_send.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
                item0 = msg_send(main_menu, item_at_idx_sel, 0)
                if item0:
                    set_title_sel = objc.sel_registerName(b"setTitle:")
                    submenu_sel = objc.sel_registerName(b"submenu")
                    msg_send.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                    sub = msg_send(item0, submenu_sel)
                    msg_send.argtypes = [
                        ctypes.c_void_p,
                        ctypes.c_void_p,
                        ctypes.c_void_p,
                    ]
                    if sub:
                        msg_send(sub, set_title_sel, ns_name)
                    msg_send(item0, set_title_sel, ns_name)
    except Exception:
        pass


def _qt_argv() -> list[str]:
    """Return argv with a stable executable name for Qt desktop identity."""
    return ["Helicon", *sys.argv[1:]]


def _xcb_platform_available() -> bool:
    """Return True when Qt's xcb platform plugin can actually be loaded.

    Since Qt 6.5 the xcb plugin refuses to load when ``libxcb-cursor0``
    is missing, which is common on minimal WSL images.  Probing the
    plugin (load, then unload) detects that before ``QApplication`` is
    created, so the app can fall back to the Wayland platform instead of
    aborting at startup.
    """
    try:
        from PySide6.QtCore import QLibraryInfo, QPluginLoader
        from pathlib import Path

        platforms = (
            Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)) / "platforms"
        )
        for name in ("libqxcb.so", "qxcb.dll"):
            candidate = platforms / name
            if not candidate.is_file():
                continue
            loader = QPluginLoader(str(candidate))
            if loader.load():
                loader.unload()
                return True
        return False
    except Exception:
        return False


def _force_x11_platform_under_wslg() -> None:
    """Prefer the X11 (xcb) Qt platform when running under WSLg.

    WSLg's native Wayland protocol cannot carry window icons, so Qt
    windows launched under WSLg Wayland always show the fallback Tux in
    the Windows taskbar.  WSLg provides an X server (Xwayland), where
    window icons are delivered via ``_NET_WM_ICON``.  Switching to xcb
    therefore lets the Helicon icon reach the Windows taskbar.

    The switch is only made when the xcb platform plugin is loadable
    (``_xcb_platform_available``); otherwise the app keeps the working
    Wayland platform rather than aborting at startup and warns about the
    missing system dependency.  An explicitly set ``QT_QPA_PLATFORM``
    (e.g. ``offscreen`` for headless use) is always respected, as is a
    missing X server.
    """
    if "QT_QPA_PLATFORM" in os.environ:
        return
    is_wsl = os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP")
    if not is_wsl or not os.environ.get("WAYLAND_DISPLAY"):
        return
    if not os.environ.get("DISPLAY"):
        return
    if not _xcb_platform_available():
        print(
            "[helicon] WSLg detected, but the xcb Qt platform plugin cannot "
            "load (missing system library libxcb-cursor0?). Falling back to "
            "Wayland, so window icons will not appear in the Windows "
            "taskbar. Install it with: sudo apt install libxcb-cursor0",
            file=sys.stderr,
        )
        return
    os.environ["QT_QPA_PLATFORM"] = "xcb"


def _macos_ns_app():
    """Return the shared NSApplication pointer via safe ctypes dispatch."""
    if sys.platform != "darwin":
        return None
    import ctypes
    import ctypes.util

    ctypes.cdll.LoadLibrary(ctypes.util.find_library("Foundation"))
    ctypes.cdll.LoadLibrary(ctypes.util.find_library("AppKit"))
    objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    objc.objc_msgSend.restype = ctypes.c_void_p
    objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    return objc.objc_msgSend(
        objc.objc_getClass(b"NSApplication"),
        objc.sel_registerName(b"sharedApplication"),
    )


def _macos_msg(receiver, selector, *args, restype=None, argtypes=()):
    """Send an Objective-C message with a freshly typed ``objc_msgSend``.

    ``objc_msgSend`` is variadic; reusing the ctypes closure across calls with
    mismatched signatures poisons the libffi state and can segfault. Each call
    here re-declares ``restype``/``argtypes`` immediately before dispatch.

    ``restype`` defaults to ``ctypes.c_void_p``; the default is applied inside
    the function so this module can import on non-macOS platforms where
    ``ctypes`` is not imported at module level.
    """
    if sys.platform != "darwin":
        return None
    import ctypes
    import ctypes.util

    if restype is None:
        restype = ctypes.c_void_p
    objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
    objc.objc_msgSend.restype = restype
    objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, *argtypes]
    return objc.objc_msgSend(receiver, selector, *args)


def _macos_sel(name: str):
    """Register and return an Objective-C selector."""
    if sys.platform != "darwin":
        return None
    import ctypes
    import ctypes.util

    objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    return objc.sel_registerName(name.encode("utf-8"))


def _macos_class(name: str):
    """Return the Objective-C class object for *name*."""
    if sys.platform != "darwin":
        return None
    import ctypes
    import ctypes.util

    objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    return objc.objc_getClass(name.encode("utf-8"))


def _macos_menu_item_count(ns_app) -> int:
    """Return the number of top-level items in NSApp's main menu, or -1."""
    if not ns_app:
        return -1
    import ctypes

    main_menu = _macos_msg(ns_app, _macos_sel("mainMenu"))
    if not main_menu:
        return -1
    count = _macos_msg(
        main_menu,
        _macos_sel("numberOfItems"),
        restype=ctypes.c_long,
        argtypes=[],
    )
    return int(count or 0)


def _macos_frontmost_pid() -> int:
    """Return the PID of the frontmost application, or -1 on failure."""
    import ctypes

    workspace = _macos_msg(
        _macos_class("NSWorkspace"),
        _macos_sel("sharedWorkspace"),
    )
    if not workspace:
        return -1
    front = _macos_msg(workspace, _macos_sel("frontmostApplication"))
    if not front:
        return -1
    return int(
        _macos_msg(
            front,
            _macos_sel("processIdentifier"),
            restype=ctypes.c_int,
            argtypes=[],
        )
    )


def _macos_native_windows() -> list:
    """Return the native NSWindow pointers of Qt's visible top-level windows."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        return []
    windows = []
    for widget in app.topLevelWidgets():
        if widget.isVisible() and widget.winId():
            ns_window = _macos_msg(int(widget.winId()), _macos_sel("window"))
            if ns_window:
                windows.append(ns_window)
    return windows


def _macos_activate_and_front() -> None:
    """Activate the app and make its visible windows key and front natively.

    ``activateWithOptions:`` is the modern replacement for the deprecated
    ``activateIgnoringOtherApps:``; the flags 1|2 = ActivateAllWindows |
    ActivateIgnoreOtherApps. ``makeKeyAndOrderFront:`` on the NSWindow is the
    native equivalent of the click that makes the window key and the app's
    menu bar visible. Qt's ``activateWindow()`` can silently no-op while the
    app is starting up, so drive the native windows directly as well.
    """
    import ctypes

    ns_app = _macos_ns_app()
    if not ns_app:
        return
    # NSApplicationActivationPolicyRegular = 0. NSInteger is a long on arm64;
    # passing c_int here silently truncates and breaks the call.
    _macos_msg(
        ns_app,
        _macos_sel("setActivationPolicy:"),
        ctypes.c_long(0),
        restype=ctypes.c_bool,
        argtypes=[ctypes.c_long],
    )
    current = _macos_msg(
        _macos_class("NSRunningApplication"),
        _macos_sel("currentApplication"),
    )
    if current:
        _macos_msg(
            current,
            _macos_sel("activateWithOptions:"),
            ctypes.c_ulong(3),
            restype=None,
            argtypes=[ctypes.c_ulong],
        )
    for ns_window in _macos_native_windows():
        _macos_msg(
            ns_window,
            _macos_sel("makeKeyAndOrderFront:"),
            None,  # sender: nil
            restype=None,
            argtypes=[ctypes.c_void_p],
        )
    # Qt-level nudge so the cocoa plugin installs/syncs the native menu bar.
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is not None:
        for window in app.topLevelWidgets():
            if window.isVisible():
                window.activateWindow()
                window.raise_()
        app.processEvents()


def _macos_resign_active() -> None:
    """Resign active with a regular activation policy.

    NSApplicationActivationPolicyRegular = 0. NSInteger is a long on arm64;
    passing c_int here silently truncates and breaks the call.
    """
    import ctypes

    ns_app = _macos_ns_app()
    if not ns_app:
        return
    _macos_msg(
        ns_app,
        _macos_sel("setActivationPolicy:"),
        ctypes.c_long(0),
        restype=ctypes.c_bool,
        argtypes=[ctypes.c_long],
    )
    _macos_msg(ns_app, _macos_sel("deactivate"), restype=None, argtypes=[])


def _macos_activate_pid(pid: int) -> None:
    """Activate another running application by process id.

    Used to reproduce the "click another app" half of the manual focus cycle:
    the window server only swaps the displayed menu bar through a real
    resign/activate transition, and activating the app that was frontmost at
    launch makes that transition explicit.
    """
    if not pid or pid == os.getpid():
        return
    import ctypes

    running = _macos_msg(
        _macos_class("NSRunningApplication"),
        _macos_sel("runningApplicationWithProcessIdentifier:"),
        ctypes.c_int(int(pid)),
        argtypes=[ctypes.c_int],
    )
    if running:
        _macos_msg(
            running,
            _macos_sel("activateWithOptions:"),
            ctypes.c_ulong(2),  # ActivateIgnoreOtherApps
            restype=None,
            argtypes=[ctypes.c_ulong],
        )


def _force_macos_menu_realization(full_cycle: bool = True) -> None:
    """Reproduce the click-away-and-back focus cycle that realizes the menu.

    On macOS a QMainWindow's ``File``/``View`` menus are a native NSMenu that
    Qt installs into the application's shared top-of-screen menu bar. Launched
    from a terminal the app is often never granted a real activation cycle, so
    the menu bar stays blank until the user clicks another app and back.

    This mirrors that cycle with real timing: resign active now, then on a
    later event-loop turn re-activate and make the windows key and front —
    exactly the two-step transition a manual click produces. Pass
    ``full_cycle=False`` for follow-up pings that only re-activate and re-key
    the window without another resign (which would flicker).
    """
    if sys.platform != "darwin":
        return
    try:
        ns_app = _macos_ns_app()
        if not ns_app:
            return
        if full_cycle:
            # Resign active first, exactly like switching to another app.
            _macos_resign_active()
            # Re-activate on a separate event-loop turn so the window server
            # observes a genuine resign -> re-activate transition (a same-turn
            # deactivate+activate is coalesced and never shows the menu).
            from PySide6.QtCore import QTimer

            QTimer.singleShot(120, _macos_activate_and_front)
        else:
            _macos_activate_and_front()
    except Exception:
        pass


def _load_napari():
    """Import napari only when a napari-backed display is requested."""
    import napari

    _patch_napari_value_bug()
    _patch_napari_icon()
    return napari


def _display_theme_stylesheet() -> str:
    """Return the shared Qt stylesheet for button-launched display windows."""
    from helicon.lib.file_browser import (
        _THEME_COLORS,
        _resolved_theme,
        _saved_theme,
    )

    colors = _THEME_COLORS[_resolved_theme(_saved_theme())]
    return f"""
        QWidget {{
            background-color: {colors["window"]};
            color: {colors["text"]};
        }}
        QLineEdit, QTextEdit, QPlainTextEdit, QComboBox {{
            background-color: {colors["input"]};
            color: {colors["text"]};
            border: 1px solid {colors["border"]};
            border-radius: 3px;
            padding: 3px;
        }}
        QToolButton, QPushButton {{
            background-color: {colors["input"]};
            color: {colors["text"]};
            border: 1px solid {colors["border"]};
            border-radius: 3px;
            padding: 3px 8px;
        }}
        QToolButton:hover, QPushButton:hover {{
            background-color: {colors["accent"]};
            color: #ffffff;
        }}
        QToolButton:pressed, QPushButton:pressed {{
            background-color: {colors["pressed"]};
            border: 1px solid {colors["accent_border"]};
        }}
        QComboBox QAbstractItemView {{
            background-color: {colors["input"]};
            color: {colors["text"]};
            selection-background-color: {colors["accent"]};
        }}
    """


def _display_theme_palette():
    """Build a Qt palette matching the persisted display theme."""
    from PySide6.QtGui import QPalette, QColor
    from helicon.lib.file_browser import (
        _THEME_COLORS,
        _resolved_theme,
        _saved_theme,
    )

    colors = _THEME_COLORS[_resolved_theme(_saved_theme())]
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(colors["window"]))
    palette.setColor(QPalette.ColorRole.Base, QColor(colors["input"]))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(colors["window"]))
    palette.setColor(QPalette.ColorRole.Text, QColor(colors["text"]))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(colors["text"]))
    palette.setColor(QPalette.ColorRole.Button, QColor(colors["input"]))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(colors["text"]))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(colors["accent"]))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    return palette


def _display_plot_theme_colors() -> dict[str, str]:
    """Return colors for plot widgets in the resolved display theme."""
    from helicon.lib.file_browser import _resolved_theme, _saved_theme

    if _resolved_theme(_saved_theme()) == "Light":
        return {
            "background": "#ffffff",
            "foreground": "#202020",
            "grid": "#909090",
            "crosshair": "#707070",
            "tooltip_background": "rgba(255,255,255,220)",
            "tooltip_foreground": "#202020",
        }
    return {
        "background": "#202020",
        "foreground": "#dcdcdc",
        "grid": "#a0a0a0",
        "crosshair": "#969696",
        "tooltip_background": "rgba(0,0,0,180)",
        "tooltip_foreground": "#dcdcdc",
    }


def _refresh_display_theme_windows() -> None:
    """Reapply the saved theme to already-open auxiliary display windows."""
    if QLayout is None:
        return
    from PySide6.QtWidgets import QApplication

    stylesheet = _display_theme_stylesheet()
    palette = _display_theme_palette()
    for window in QApplication.topLevelWidgets():
        if window.property("helicon_theme_window"):
            window.setStyleSheet(stylesheet)
            window.setPalette(palette)
            for child in window.findChildren(QWidget):
                child.setStyleSheet(stylesheet)
                child.setPalette(palette)
            apply_theme = getattr(window, "_apply_display_theme", None)
            if apply_theme is not None:
                apply_theme()
            try:
                from helicon.lib.gallery_widget import _apply_gallery_theme

                _apply_gallery_theme(window)
            except Exception:
                pass
    _refresh_napari_theme()


class _FlowLayout(_FlowLayoutBase):
    """Minimal flow layout that wraps child widgets to the available width.

    Used for the iteration checkbox strip so that jobs with hundreds of
    refinement iterations keep the window a normal size: the checkboxes wrap
    onto multiple rows inside a scroll area instead of stretching the window
    horizontally.  Height-for-width reporting lets the surrounding scroll
    area show a vertical scrollbar once the wrapped content grows past the
    capped strip height.
    """

    def __init__(self, parent=None, margin=0, spacing=6):
        super().__init__(parent)
        self.setContentsMargins(margin, margin, margin, margin)
        self._spacing = spacing
        self._item_list = []

    def addItem(self, item):
        self._item_list.append(item)

    def count(self):
        return len(self._item_list)

    def itemAt(self, index):
        if 0 <= index < len(self._item_list):
            return self._item_list[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._item_list):
            return self._item_list.pop(index)
        return None

    def expandingDirections(self):
        from PySide6.QtCore import Qt

        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._item_list:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QSize(
            margins.left() + margins.right(), margins.top() + margins.bottom()
        )
        return size

    def _do_layout(self, rect, test_only):
        from PySide6.QtCore import Qt

        margins = self.contentsMargins()
        effective = rect.adjusted(
            margins.left(), margins.top(), -margins.right(), -margins.bottom()
        )
        x = effective.x()
        y = effective.y()
        line_height = 0
        line_items = []
        lines = []
        for item in self._item_list:
            widget = item.widget()
            if widget is not None and widget.isHidden():
                continue
            hint = item.sizeHint()
            space_x = self._spacing
            space_y = self._spacing
            next_x = x + hint.width() + space_x
            if next_x - space_x > effective.right() and line_height > 0:
                lines.append((y, line_height, line_items))
                x = effective.x()
                y = y + line_height + space_y
                next_x = x + hint.width() + space_x
                line_height = 0
                line_items = []
            line_items.append((item, x, hint))
            x = next_x
            line_height = max(line_height, hint.height())
        if line_items:
            lines.append((y, line_height, line_items))

        if not test_only:
            for line_y, height, items in lines:
                for item, item_x, hint in items:
                    item_y = line_y + (height - hint.height()) // 2
                    item.setGeometry(QRect(QPoint(item_x, item_y), hint))

        if not lines:
            return margins.top() + margins.bottom()
        last_y, last_height, _items = lines[-1]
        return last_y + last_height - rect.y() + margins.bottom()


class _FlowContainer(_FlowContainerBase):
    """QWidget whose size hint reflects the wrapped height of its flow layout.

    The scroll area uses this size hint to decide whether the wrapped
    checkbox rows overflow the capped strip height and need a scrollbar.
    """

    def __init__(self, flow=None):
        super().__init__()
        self._flow = flow

    def sizeHint(self):
        from PySide6.QtCore import QSize

        if self._flow is None:
            return QSize(400, 40)
        width = self.width() if self.width() > 0 else 400
        return QSize(width, self._flow.heightForWidth(width))

    def minimumSizeHint(self):
        # Keep the scroll strip narrow no matter how many checkboxes it
        # holds.  If this reported the full wrapped size, the scroll area's
        # minimum width would push the whole window wide on real displays
        # and the checkboxes would sit on a single row with empty space.
        # The scroll area (not the window) absorbs the overflow instead.
        from PySide6.QtCore import QSize

        return QSize(200, 50)


class _WebAppState:
    """One Helicon Lab shiny server launched from the file browser.

    Parameters
    ----------
    proc : subprocess.Popen
        The running app process returned by ``launch_shiny_app``.
    token : str
        Per-launch token embedded in the app URL; control endpoints use
        it to scope navigation to this instance.
    """

    def __init__(self, proc, token: str):
        self.proc = proc
        self.token = token
        self.base_url = None  # set by the url callback once the port is known
        self.last_used = time.monotonic()

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def touch(self) -> None:
        self.last_used = time.monotonic()


# Coexisting web-app servers for this display process (same model as
# ``_DisplayTracker`` for napari/gallery).  ``New`` unchecked → navigate
# the most recently used alive instance; ``New`` checked → spawn another
# that coexists.  ``_terminate_web_apps`` reaps all on display exit.
_WEB_APP_INSTANCES: list = []


def _web_app_alive() -> list:
    """Prune dead servers and return living ones, newest-used first."""
    # Mutate in place so importers that bound the list name stay in sync.
    _WEB_APP_INSTANCES[:] = [s for s in _WEB_APP_INSTANCES if s.is_alive()]
    return sorted(_WEB_APP_INSTANCES, key=lambda s: s.last_used, reverse=True)


def _web_app_active():
    """Most recently used alive web-app server, or ``None``."""
    alive = _web_app_alive()
    return alive[0] if alive else None


def _terminate_web_apps() -> None:
    """Terminate every tracked web app process (file browser is exiting).

    Sends SIGTERM first, waits briefly, then SIGKILL for any survivors so
    a hung shiny child cannot outlive a graceful display exit.
    """
    for state in list(_WEB_APP_INSTANCES):
        if state.is_alive():
            try:
                state.proc.terminate()
            except Exception:
                pass
    deadline = time.monotonic() + 2.0
    for state in list(_WEB_APP_INSTANCES):
        if not state.is_alive():
            continue
        remaining = deadline - time.monotonic()
        try:
            state.proc.wait(timeout=max(remaining, 0.05))
        except Exception:
            try:
                state.proc.kill()
            except Exception:
                pass
            try:
                state.proc.wait(timeout=1.0)
            except Exception:
                pass
    _WEB_APP_INSTANCES.clear()


def _navigate_web_app(state: _WebAppState, query_params: dict) -> bool:
    """POST ``query_params`` to ``state``; return True if the tab accepted."""
    import json
    import urllib.request

    if state.base_url is None or not state.is_alive():
        return False
    try:
        req = urllib.request.Request(
            f"{state.base_url}helicon/navigate?token={state.token}",
            data=json.dumps({"query_params": query_params}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode())
        if body.get("ok") and body.get("alive"):
            state.touch()
            return True
    except Exception:
        pass
    return False


def _spawn_web_app(query_params: dict) -> _WebAppState:
    """Start a new shiny server, open one browser tab, and track it."""
    import secrets

    from helicon.lib.shiny import launch_shiny_app

    token = secrets.token_hex(8)
    params = dict(query_params)
    params["helicon_token"] = token
    new_state = _WebAppState(None, token)

    def _on_url(url):
        # ``launch_shiny_app`` skips its own ``_open_browser`` when a
        # ``url_callback`` is supplied, so the fresh-launch path must open
        # the browser itself; subsequent clicks navigate this instance.
        from helicon.lib.shiny import _open_browser

        new_state.base_url = url.split("?", 1)[0]
        _open_browser(url)

    proc = launch_shiny_app(
        "helicon.webApps.app:app",
        block=False,
        query_params=params,
        url_callback=_on_url,
    )
    new_state.proc = proc
    new_state.touch()
    _WEB_APP_INSTANCES.append(new_state)
    return new_state


def _launch_or_reuse_web_app(
    key: str, query_params: dict, *, new_window: bool = False
) -> None:
    """Open ``query_params`` in Helicon Lab, matching other display buttons.

    ``new_window=False`` (default, ``New`` unchecked): navigate the most
    recently used alive shiny instance via ``/helicon/navigate``.  If none
    is usable, spawn one.

    ``new_window=True`` (``New`` checked): always spawn a new shiny server
    and browser tab; existing instances keep running for side-by-side
    comparison.  Later clicks with ``New`` unchecked target the instance
    most recently launched or successfully navigated (same “most recent
    focus/interaction” rule as napari/gallery).

    Parameters
    ----------
    key : str
        Logical tool name (unused for routing; all tools share instances).
    query_params : dict
        Shiny bookmark query params produced by ``_make_bookmark_query``.
    new_window : bool, optional
        If True, spawn a coexisting instance. Defaults to False.
    """
    query_params = dict(query_params)
    query_params["helicon_theme"] = _get_display_theme()

    if not new_window:
        for state in _web_app_alive():
            if _navigate_web_app(state, query_params):
                return

    _spawn_web_app(query_params)


def _get_display_theme() -> str:
    """Return the persisted file-browser/web-app theme."""
    from helicon.lib.file_browser import _saved_theme

    return _saved_theme()


def _napari_display_theme() -> str:
    """Map the saved Helicon theme to napari's theme names."""
    theme = _get_display_theme()
    return {"Dark": "dark", "Light": "light", "System": "system"}[theme]


def _napari_canvas_background() -> str:
    """Return the napari canvas background for the saved display theme."""
    from helicon.lib.file_browser import _resolved_theme

    if _resolved_theme(_get_display_theme()) == "Light":
        return "#ffffff"
    return "#202020"


def _refresh_napari_theme() -> None:
    """Apply the saved theme to all currently open napari viewers."""
    try:
        theme = _napari_display_theme()
        background = _napari_canvas_background()
        for viewer in _napari.alive():
            viewer.theme = theme
            viewer.background_color = background
    except Exception:
        # napari is optional and may not be initialized yet.
        pass


def _get_qsettings():
    from PySide6.QtCore import QSettings

    return QSettings("helicon", "display")


def _is_wsl():
    """Return True if running inside Windows Subsystem for Linux."""
    import platform

    if platform.system() != "Linux":
        return False
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def _restore_geometry(dock, viewer):
    """Restore saved window sizes and reposition after the compositor places them."""
    from PySide6.QtCore import QTimer

    def _apply(attempt=0):
        settings = _get_qsettings()

        # Restore the viewer geometry if a viewer exists.
        if viewer is not None:
            try:
                qt_win = viewer.window._qt_window
            except AttributeError:
                qt_win = None
            if qt_win is not None and not qt_win.isVisible() and attempt < 10:
                QTimer.singleShot(50, lambda: _apply(attempt + 1))
                return
            if qt_win is not None:
                viewer_ba = settings.value("viewer_ba")
                if viewer_ba is not None:
                    try:
                        qt_win.restoreGeometry(viewer_ba)
                    except AttributeError:
                        pass

        # Restore the dock (browser) geometry independently.
        dock_ba = settings.value("dock_ba")
        if dock_ba is not None:
            try:
                dock.restoreGeometry(dock_ba)
            except (AttributeError, TypeError):
                pass
        elif viewer is not None:
            _position_default(dock, viewer)
        dock.show()

    QTimer.singleShot(0, _apply)


def _position_default(dock, viewer):
    try:
        napari_rect = viewer.window._qt_window.frameGeometry()
        dock.adjustSize()
        dock_rect = dock.frameGeometry()
        w = max(dock_rect.width(), 300)
        x = napari_rect.x() - w - 10
        y = napari_rect.y()
        h = napari_rect.height()
        dock.setGeometry(x, y, w, h)
    except AttributeError:
        pass


def _install_dock_save_hook(dock):
    """Install an event filter to save dock geometry when it closes independently."""
    from PySide6.QtCore import QEvent, QObject

    class _DockCloseFilter(QObject):
        """Saves only dock geometry when the dock closes independently."""

        def __init__(self, dock, parent=None):
            super().__init__(parent)
            self._dock = dock

        def eventFilter(self, obj, event):
            if event.type() == QEvent.Close:
                settings = _get_qsettings()
                settings.setValue("dock_ba", self._dock.saveGeometry())
                save_cols = getattr(self._dock, "_save_col_widths", None)
                if callable(save_cols):
                    save_cols()
            return False

    dock_flt = _DockCloseFilter(dock, parent=dock)
    dock.installEventFilter(dock_flt)
    return dock_flt


def _install_viewer_save_hook(dock, viewer):
    """Install an event filter to save both viewer and dock geometry on viewer close."""
    from PySide6.QtCore import QEvent, QObject

    class _ViewerCloseFilter(QObject):
        """Saves both viewer and dock geometry when the viewer window closes."""

        def __init__(self, dock, viewer, parent=None):
            super().__init__(parent)
            self._dock = dock
            self._viewer = viewer

        def eventFilter(self, obj, event):
            if event.type() == QEvent.Close:
                _save_geometry(self._dock, self._viewer)
            return False

    flt = _ViewerCloseFilter(dock, viewer, parent=viewer.window._qt_window)
    viewer.window._qt_window.installEventFilter(flt)
    return flt


def _save_geometry(dock, viewer):
    settings = _get_qsettings()
    qt_win = viewer.window._qt_window

    cached_ba = getattr(viewer, "_display_only_ba", None)
    viewer_ba = cached_ba if cached_ba is not None else qt_win.saveGeometry()
    settings.setValue("viewer_ba", viewer_ba)

    settings.setValue("dock_ba", dock.saveGeometry())

    save_cols = getattr(dock, "_save_col_widths", None)
    if callable(save_cols):
        save_cols()


# MRC-format image extensions. RELION/CTFFIND write CTF power spectra as MRC
# maps with a ``.ctf`` suffix (e.g. ``*_PS.ctf``).
_MRC_EXTENSIONS = {".mrc", ".mrcs", ".map", ".ctf"}

# Star files that describe pipelines/optimisation rather than image data;
# opened as text, not as image/volume stacks.
_METADATA_STAR_SUFFIXES = (
    "pipeline.star",
    "optimiser.star",
    "model.star",
    "sampling.star",
    "job.star",
    "extractpick.star",
    "frameimage.star",
    "autopick.star",
)


class _LazyStarStack:
    """Lazy array that reads individual images from star file references on demand."""

    def __init__(self, entries: list[tuple[int, str, float]], shape: tuple, dtype):
        self._entries = entries
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self._cache: dict[int, object] = {}

    def __getitem__(self, key):
        import numpy as np

        if isinstance(key, int):
            key = key % self.shape[0]
            if key not in self._cache:
                self._cache[key] = self._read(key)
            return self._cache[key]

        if isinstance(key, slice):
            indices = range(*key.indices(self.shape[0]))
            return np.stack([self._read(i) for i in indices])

        if isinstance(key, tuple):
            if len(key) == 1:
                return self[key[0]]
            return np.stack(
                [self._read(i)[key[1:]] for i in range(*key[0].indices(self.shape[0]))]
            )

        raise TypeError(f"unsupported key type: {type(key)}")

    def _read(self, idx: int):
        import mrcfile

        frame_idx_0based, mrc_path, apix = self._entries[idx]
        with mrcfile.open(mrc_path, permissive=True) as mrc:
            data = mrc.data
            if data.ndim == 2:
                return data
            return data[frame_idx_0based]

    @property
    def nbytes(self):
        return self.shape[0] * self.shape[1] * self.shape[2] * 4


class _SliceDirectionWidget:
    """Replaces axis labels with a Z/Y/X dropdown at the right end of each slider.

    For 3D volumes displayed as 2D slices the dropdown lets the user swap
    which spatial axis (Z/Y/X) the visible slider navigates through. For a
    *true* image stack (a list of independent 2D frames — ``.mrcs`` particle
    stacks, ``_data.star`` reference stacks, multi-page PDFs) the axis
    selector is meaningless because axis 0 is a frame index, not a spatial
    axis. ``set_stack_mode(True)`` hides every combo for that case.
    """

    _combos: list = []
    # True while the active layer is an image stack (non-spatial axis 0):
    # the axis-direction combos are hidden so the Z/Y/X selector does not
    # appear for stacks where it would be nonsensical.
    _stack_mode: bool = False

    def __init__(self, viewer):
        self._viewer = viewer

    @classmethod
    def set_stack_mode(cls, is_stack: bool) -> None:
        """Show or hide all axis-direction combos.

        Called by ``_open_file`` immediately before ``viewer.add_image(...)`` so
        that newly-built sliders pick up the right state and any pre-existing
        combos (reused across files) are updated in place.
        """
        cls._stack_mode = bool(is_stack)
        alive = []
        for combo in cls._combos:
            try:
                if cls._stack_mode:
                    combo.hide()
                    combo.setFixedWidth(0)
                else:
                    combo.setFixedWidth(60)
                    combo.show()
                alive.append(combo)
            except RuntimeError:
                continue
        cls._combos = alive

    def inject(self):
        from PySide6.QtWidgets import QComboBox

        from napari._qt.widgets.qt_dims import QtDimSliderWidget

        # Start fresh: combos from a previous viewer are tied to C++ widgets
        # that no longer exist, so never carry them into a new viewer.
        _SliceDirectionWidget._combos = []

        _orig_init = QtDimSliderWidget.__init__

        def _patched_init(self_slider, parent, axis):
            _orig_init(self_slider, parent, axis)

            self_slider.axis_label.hide()
            self_slider.axis_label.setFixedWidth(0)
            self_slider.totslice_label.hide()
            self_slider.totslice_label.setFixedWidth(0)
            sep = self_slider.findChild(type(self_slider).__mro__[0])
            from PySide6.QtWidgets import QFrame

            for child in self_slider.findChildren(QFrame):
                child.hide()
                child.setFixedWidth(0)

            combo = QComboBox(self_slider)
            combo.addItems(["Z", "Y", "X"])
            combo.setFixedSize(60, 22)
            combo.setStyleSheet(_display_theme_stylesheet())

            # Suppress the popup entirely — cycle through items on click instead
            combo.showPopup = lambda: None

            def _mousePressEvent(event):
                idx = (combo.currentIndex() + 1) % combo.count()
                combo.blockSignals(True)
                combo.setCurrentIndex(idx)
                combo.blockSignals(False)
                combo.currentIndexChanged.emit(idx)

            combo.mousePressEvent = _mousePressEvent

            if _SliceDirectionWidget._stack_mode:
                combo.hide()
                combo.setFixedWidth(0)

            _SliceDirectionWidget._combos.append(combo)

            def _on_change(idx):
                alive = []
                for c in _SliceDirectionWidget._combos:
                    if c is combo:
                        alive.append(c)
                        continue
                    # A combo from a previous viewer has a destroyed C++ object
                    # and raises RuntimeError on access; skip and drop it.
                    try:
                        c.blockSignals(True)
                        c.setCurrentIndex(idx)
                        c.blockSignals(False)
                        alive.append(c)
                    except RuntimeError:
                        continue
                # Drop combos whose C++ object has been destroyed so the
                # list never accumulates stale references across viewers.
                _SliceDirectionWidget._combos = alive
                ndim = self_slider.dims.ndim
                if ndim >= 3:
                    orders = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
                    step = [int(self_slider.dims.nsteps[d]) // 2 for d in range(ndim)]
                    self_slider.dims.current_step = tuple(step)
                    self_slider.dims.order = orders[idx]
                    self_slider.qt_dims._update_slider()
                    self_slider.qt_dims._update_range()

            combo.currentIndexChanged.connect(_on_change)
            self_slider.layout().addWidget(combo)

        if not getattr(QtDimSliderWidget, "_patched", False):
            QtDimSliderWidget.__init__ = _patched_init
            QtDimSliderWidget._patched = True


def _add_welcome_shortcut(viewer):
    """Add custom shortcut to the napari welcome screen."""
    try:
        from napari._qt.widgets.qt_welcome import QtWelcomeWidget, QtShortcutLabel
    except ImportError:
        return

    def _inject():
        qt_window = viewer.window._qt_window
        welcome_widgets = qt_window.findChildren(QtWelcomeWidget)
        if not welcome_widgets:
            return

        welcome = welcome_widgets[0]
        from PySide6.QtWidgets import QFormLayout

        def find_form_layout(widget):
            for child in widget.children():
                if isinstance(child, QFormLayout):
                    return child
                result = find_form_layout(child)
                if result:
                    return result
            return None

        form_layout = find_form_layout(welcome)
        if form_layout is None:
            return

        shortcut_label = QtShortcutLabel("Mid mouse button click")
        description_label = QtShortcutLabel("Toggle left side control panel")

        form_layout.addRow(shortcut_label, description_label)

    from PySide6.QtCore import QTimer

    QTimer.singleShot(0, _inject)


def _auto_contrast(data):
    """Compute contrast limits using robust statistics.

    Black point = max(median - 3*MAD, 1st percentile)
    White point = min(median + 3*MAD, 99th percentile)

    Parameters
    ----------
    data : numpy.ndarray
        Image data.

    Returns
    -------
    tuple[float, float]
        (black_point, white_point) contrast limits.
    """
    import numpy as np

    flat = data.ravel()
    med = float(np.median(flat))
    mad = float(np.median(np.abs(flat - med))) * 1.4826
    p1 = float(np.percentile(flat, 1))
    p99 = float(np.percentile(flat, 99))

    black = max(med - 3 * mad, p1)
    white = min(med + 3 * mad, p99)

    if black >= white:
        black = float(flat.min())
        white = float(flat.max())
        if black == white:
            white = black + 1.0

    return black, white


def _enable_continuous_auto_contrast(layer, viewer) -> None:
    """Enable continuous auto-contrast and sync napari's UI toggle.

    napari only binds the 'continuous' button's ``toggled`` signal to
    ``layer._keep_auto_contrast`` (one direction). Setting the attribute
    programmatically therefore never lights up the button, leaving the UI
    out of sync with the actual (continuous) behavior. We set the attribute
    directly so the contrast behavior is guaranteed even without a GUI, then
    check the button as a virtual click so the 'continuous' label highlights.
    The UI sync is best-effort: skipped when no Qt controls exist (headless
    runs or mocked tests).
    """
    layer._keep_auto_contrast = True
    try:
        controls = viewer.window._qt_viewer.controls.widgets[layer]
        auto_btn = controls._contrast_limits_control.auto_scale_bar._auto_btn
        auto_btn.setChecked(True)
    except (AttributeError, KeyError):
        pass


def _save_qimage(qimage, parent=None, default_name=None):
    """Open a Save-As dialog and write a QImage to the chosen file.

    Supported formats: PNG, TIFF, PDF, SVG.  The dialog filter list
    determines the format from the chosen extension.

    Parameters
    ----------
    qimage : PySide6.QtGui.QImage
        The image to save.
    parent : QWidget, optional
        Parent widget for the dialog (centre-on-parent).
    default_name : str, optional
        Suggested base filename (without extension).  The dialog pre-fills
        it with a ``.png`` suffix, replacing any existing suffix on the
        caller's source name.
    """
    from PySide6.QtWidgets import QFileDialog
    from PySide6.QtGui import QPixmap

    if isinstance(qimage, QPixmap):
        qimage = qimage.toImage()

    if default_name and isinstance(default_name, (str, bytes)):
        stem = (
            Path(default_name).stem if "." in Path(default_name).name else default_name
        )
        suggested = stem + ".png"
    else:
        suggested = ""

    filt = "Images (*.png *.tiff *.tif);;PDF (*.pdf);;SVG (*.svg)"
    dlg = QFileDialog(parent, "Save Image")
    dlg.setAcceptMode(QFileDialog.AcceptSave)
    dlg.setNameFilter(filt)
    # selectFile() reliably pre-fills the file name on native dialogs
    # (macOS/Windows); passing it only via the static getSaveFileName()
    # "directory" argument is ignored by the native save dialog.
    if suggested:
        dlg.selectFile(suggested)
    if not dlg.exec():
        return
    selected = dlg.selectedFiles()
    if not selected:
        return
    path = selected[0]
    ext = path.rsplit(".", 1)[-1].lower()
    if ext in ("pdf", "svg"):
        _render_qimage_vector(qimage, path, ext)
    else:
        qimage.save(path)


def _render_qimage_vector(qimage, path, fmt):
    """Render a QImage to a vector file (PDF or SVG).

    Parameters
    ----------
    qimage : PySide6.QtGui.QImage
        Source image.
    path : str
        Destination file path.
    fmt : str
        ``"pdf"`` or ``"svg"``.
    """
    from PySide6.QtGui import QPainter
    from PySide6.QtCore import QMarginsF

    w, h = qimage.width(), qimage.height()
    if fmt == "pdf":
        from PySide6.QtGui import QPdfWriter, QPageSize
        from PySide6.QtCore import QSizeF

        writer = QPdfWriter(path)
        writer.setResolution(72)
        writer.setPageSize(
            QPageSize(QSizeF(w, h), QPageSize.Unit.Point),
        )
        writer.setPageMargins(QMarginsF(0, 0, 0, 0))
        painter = QPainter(writer)
    else:
        from PySide6.QtSvg import QSvgGenerator
        from PySide6.QtCore import QSize, QRectF

        gen = QSvgGenerator()
        gen.setFileName(path)
        gen.setSize(QSize(w, h))
        gen.setViewBox(QRectF(0, 0, w, h))
        painter = QPainter(gen)

    from PySide6.QtCore import QRect

    painter.drawImage(QRect(0, 0, w, h), qimage)
    painter.end()


def _install_viewer_save_menu(viewer):
    """Register a right-click save menu on the napari canvas.

    napari's vispy canvas consumes mouse events at the vispy level before Qt
    event filters ever see them.  The camera's ``viewbox_mouse_event`` is
    connected as a **bound method** in ``EventEmitter._callbacks``, so
    patching the class or instance method *after* connection has no effect —
    the original function pointer is frozen at connect time.

    The fix is to **disconnect** the camera's original callbacks from the
    viewbox's mouse events and **connect** new filtered wrappers that:

    * Block right-button events from reaching the camera (preventing zoom).
    * Show a "Save Canvas As…" context menu on right-click press.
    * Forward all other events to the original camera handler unchanged.
    """
    from unittest.mock import MagicMock

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QMenu

    try:
        if isinstance(viewer.window, MagicMock):
            return
        qv = viewer.window._qt_viewer
        canvas = getattr(qv, "canvas", None)
        if canvas is None:
            return

        view = getattr(canvas, "view", None)
        if view is None:
            return
        camera = getattr(view, "camera", None)
        if camera is None:
            return

        original_handler = getattr(camera, "viewbox_mouse_event", None)
        if original_handler is None:
            return

        _mouse_names = ("mouse_press", "mouse_move", "mouse_release")
        for name in _mouse_names:
            emitter = getattr(view.events, name, None)
            if emitter is not None:
                try:
                    emitter.disconnect(original_handler)
                except Exception:
                    pass

        def _show_save_menu():
            gpos = canvas.native.cursor().pos()
            menu = QMenu()
            menu.addAction("Save Canvas As…")
            menu.triggered.connect(
                lambda action: (
                    _save_viewport(viewer)
                    if "Save" in (action.text() if action else "")
                    else None
                )
            )
            menu.exec(gpos)

        def _filtered_mouse_event(event):
            etype = getattr(event, "type", None)

            if etype == "mouse_press" and getattr(event, "button", None) == 2:
                QTimer.singleShot(0, _show_save_menu)

            original_handler(event)

        for name in _mouse_names:
            emitter = getattr(view.events, name, None)
            if emitter is not None:
                emitter.connect(_filtered_mouse_event)

    except Exception:
        pass


def _crop_to_content(arr):
    """Crop an RGB/RGBA screenshot to the tight bounding box of the data.

    napari's ``screenshot(canvas_only=True)`` still contains the uniformly
    filled canvas background (letterboxing when the data does not fill the
    canvas).  The background is assumed to be a single solid colour, sampled
    from the top-left corner; every pixel that differs from it is content,
    and the result is cropped to that region's bounding box.

    Parameters
    ----------
    arr : numpy.ndarray
        ``H x W x (3 or 4)`` uint8 image.

    Returns
    -------
    numpy.ndarray
        Cropped image, or ``arr`` unchanged if it has no colour channel or
        no non-background pixels.
    """
    import numpy as np

    if arr.ndim != 3 or arr.shape[2] < 3:
        return arr
    h, w = arr.shape[:2]
    bg = arr[0, 0, :3].astype(int)
    rgb = arr[:, :, :3].astype(int)
    # Allow a small tolerance for anti-aliased edges against the background.
    diff = np.any(np.abs(rgb - bg) > 8, axis=2)
    if not diff.any():
        return arr
    rows = np.where(diff.any(axis=1))[0]
    cols = np.where(diff.any(axis=0))[0]
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    return arr[y0:y1, x0:x1]


def _viewer_source_name(viewer) -> str | None:
    """Best-effort source file name for a napari viewer's current content.

    napari records the path a layer was loaded from in ``layer.source.path``
    (image/points/etc.).  Returns the first such path's name, or ``None`` if
    no layer carries a source path (e.g. generated/empty viewers).
    """
    try:
        for layer in viewer.layers:
            src = getattr(layer, "source", None)
            path = getattr(src, "path", None)
            if path:
                return Path(path).name
    except Exception:
        pass
    return None


def _save_viewport(viewer, parent=None):
    """Capture the napari canvas and save it via the file dialog.

    Parameters
    ----------
    viewer : napari.Viewer
        The viewer whose canvas to capture.
    parent : QWidget, optional
        Parent for the save dialog.
    """
    import numpy as np
    from PySide6.QtGui import QImage

    try:
        arr = np.ascontiguousarray(viewer.screenshot(canvas_only=True, flash=False))
    except Exception:
        return
    # Drop the empty canvas padding around the data (mirrors the gallery
    # save, which crops to the drawn thumbnail bounding box).
    arr = _crop_to_content(arr)
    # _crop_to_content returns a (possibly non-contiguous) view; QImage needs
    # contiguous memory, so copy before wrapping.
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    qimg = QImage(arr.data, w, h, w * 4, QImage.Format_RGBA8888)
    qimg.ndarray = arr
    # Prefer the name recorded by _open_file (covers openers that add layers
    # directly); fall back to napari's own layer.source.path for files opened
    # through viewer.open().
    source = getattr(viewer, "_source_name", None) or _viewer_source_name(viewer)
    _save_qimage(qimg, parent, default_name=source)


def _launch_chimerax(path: str) -> None:
    """Open ``path`` in an external ChimeraX process.

    ChimeraX is launched detached so it runs independently of the helicon
    process. A clear message is printed if ChimeraX cannot be found.
    """
    import subprocess

    from helicon.lib.file_browser import _find_chimerax

    exe = _find_chimerax()
    if exe is None:
        print(
            "[helicon] ChimeraX not found. Install it from "
            "https://www.cgl.ucsf.edu/chimerax/ or put it on your PATH."
        )
        return
    try:
        subprocess.Popen([exe, path])
        print(f"[helicon] launched ChimeraX with {path}")
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[helicon] failed to launch ChimeraX: {exc}")


def _make_bookmark_query(tab_name: str, params: dict) -> dict:
    """Build a query_params dict that produces a Shiny bookmark URL.

    Returns a dict suitable for ``launch_shiny_app(query_params=...)`` that
    produces: ``?_inputs_&helicon_tab="TabName"&_values_&p=...``
    """
    import json

    return {
        "_inputs_": "",
        "helicon_tab": f'"{tab_name}"',
        "_values_": "",
        "p": json.dumps(params),
    }


def _launch_denovo3d(path: str, *, new_window: bool = False) -> None:
    """Open a .mrcs file in the Helicon Lab Denovo3D tab via bookmark URL."""
    from pathlib import Path

    from PySide6.QtWidgets import QMessageBox

    try:
        _launch_or_reuse_web_app(
            "Denovo3D",
            _make_bookmark_query(
                "Denovo3D",
                {
                    "input_mode_images": "url",
                    "url_images": str(Path(path).resolve()),
                },
            ),
            new_window=new_window,
        )
    except Exception as exc:
        QMessageBox.critical(
            None,
            "Denovo3D Launch Error",
            f"Failed to launch Denovo3D:\n{exc}",
        )


def _launch_whereismyclass(path: str, *, new_window: bool = False) -> None:
    """Open a star/cs file in the WhereIsMyClass tab via bookmark URL."""
    from pathlib import Path

    from PySide6.QtWidgets import QMessageBox

    try:
        _launch_or_reuse_web_app(
            "WhereIsMyClass",
            _make_bookmark_query(
                "WhereIsMyClass",
                {
                    "input_mode": "url",
                    "url_star": str(Path(path).resolve()),
                },
            ),
            new_window=new_window,
        )
    except Exception as exc:
        QMessageBox.critical(
            None,
            "WhereIsMyClass Launch Error",
            f"Failed to launch WhereIsMyClass:\n{exc}",
        )


def _launch_helicalprojection(path: str, *, new_window: bool = False) -> None:
    """Open a file in the HelicalProjection tab via bookmark URL."""
    from pathlib import Path

    from PySide6.QtWidgets import QMessageBox

    try:
        _launch_or_reuse_web_app(
            "HelicalProjection",
            _make_bookmark_query(
                "HelicalProjection",
                {
                    "mode_images": "url",
                    "url_images": str(Path(path).resolve()),
                },
            ),
            new_window=new_window,
        )
    except Exception as exc:
        QMessageBox.critical(
            None,
            "HelicalProjection Launch Error",
            f"Failed to launch HelicalProjection:\n{exc}",
        )


def _launch_helicalpitch(path: str, *, new_window: bool = False) -> None:
    """Open a file in the HelicalPitch tab via bookmark URL.

    Derives the companion file (star↔mrcs) from the given path when it
    exists on disk, so both params and class images are loaded together.
    """
    import re
    from pathlib import Path

    from PySide6.QtWidgets import QMessageBox

    try:
        file_path = Path(path).resolve()
        suffix = file_path.suffix.lower()

        bookmark = {
            "mode_params": "url",
            "mode_classes": "url",
        }

        if suffix in (".star", ".cs"):
            bookmark["url_params"] = str(file_path)
            iter_match = re.search(r"run_it(\d+)", file_path.name)
            if iter_match:
                mrcs_file = (
                    file_path.parent / f"run_it{iter_match.group(1)}_classes.mrcs"
                )
                if mrcs_file.exists():
                    bookmark["url_classes"] = str(mrcs_file)
        else:
            bookmark["url_classes"] = str(file_path)
            iter_match = re.search(r"run_it(\d+)", file_path.name)
            if iter_match:
                star_file = file_path.parent / f"run_it{iter_match.group(1)}_data.star"
                if star_file.exists():
                    bookmark["url_params"] = str(star_file)

        _launch_or_reuse_web_app(
            "HelicalPitch",
            _make_bookmark_query("HelicalPitch", bookmark),
            new_window=new_window,
        )
    except Exception as exc:
        QMessageBox.critical(
            None,
            "HelicalPitch Launch Error",
            f"Failed to launch HelicalPitch:\n{exc}",
        )


def _launch_hill(path: str, *, new_window: bool = False) -> None:
    """Open a file in the HILL tab of the consolidated Helicon Lab web app.

    Launches the unified Shiny app and selects the HILL tab with the given file.
    """
    from PySide6.QtWidgets import QMessageBox

    try:
        params = _make_bookmark_query("HILL", {})
        params["input_mode"] = "2"
        params["img_file_url"] = path
        _launch_or_reuse_web_app("HILL", params, new_window=new_window)
    except Exception as exc:
        QMessageBox.critical(
            None,
            "HILL Launch Error",
            f"Failed to launch HILL:\n{exc}",
        )


def _launch_hi3d(path: str, *, new_window: bool = False) -> None:
    """Open a file in the HI3D tab of the consolidated Helicon Lab web app.

    Launches the unified Shiny app and selects the HI3D tab with the given file.
    """
    from PySide6.QtWidgets import QMessageBox

    try:
        params = _make_bookmark_query("HI3D", {})
        params["img_file_url"] = path
        _launch_or_reuse_web_app("HI3D", params, new_window=new_window)
    except Exception as exc:
        QMessageBox.critical(
            None,
            "HI3D Launch Error",
            f"Failed to launch HI3D:\n{exc}",
        )


def _launch_truefsc(path: str, parent=None) -> None:
    """Compute True FSC from the two half-maps referenced by a model.star file."""
    import logging
    import os
    import re
    import tempfile
    from pathlib import Path

    from PySide6.QtCore import QThread, Signal
    from PySide6.QtWidgets import QDialog, QLabel, QPushButton, QTextEdit, QVBoxLayout

    model_path = Path(path)
    model_dir = model_path.parent
    model_name = model_path.name

    match = re.match(r"(run_it\d+)_half(\d)_", model_name)
    if match:
        prefix = match.group(1)
        map1 = model_dir / f"{prefix}_half1_class001.mrc"
        map2 = model_dir / f"{prefix}_half2_class001.mrc"
    elif model_name == "run_model.star":
        map1 = model_dir / "run_half1_class001_unfil.mrc"
        map2 = model_dir / "run_half2_class001_unfil.mrc"
    else:
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.warning(
            None,
            "trueFSC Error",
            f"Cannot determine half-maps from:\n{model_name}",
        )
        return

    if not map1.exists() or not map2.exists():
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.warning(
            None,
            "trueFSC Error",
            f"Half-maps not found:\n{map1}\n{map2}",
        )
        return

    if os.access(model_dir, os.W_OK):
        output_dir = model_dir
    else:
        output_dir = Path(tempfile.mkdtemp(prefix="helicon_truefsc_"))

    plot_file = output_dir / "trueFSC.pdf"

    from helicon.commands.trueFSC import compute_truefsc

    class LogHandler(logging.Handler):
        def __init__(self, signal):
            super().__init__()
            self._signal = signal

        def emit(self, record):
            msg = self.format(record)
            self._signal.emit(msg)

    class Worker(QThread):
        line_received = Signal(str)
        finished = Signal(object)
        error = Signal(str)

        def __init__(self):
            super().__init__()

        def run(self):
            try:
                self.line_received.emit(f"Map 1: {map1}")
                self.line_received.emit(f"Map 2: {map2}")
                self.line_received.emit(f"Output: {plot_file}")
                self.line_received.emit("")

                handler = LogHandler(self.line_received)
                handler.setFormatter(logging.Formatter("%(message)s"))
                logger = logging.getLogger("helicon.commands.trueFSC")
                logger.addHandler(handler)
                logger.setLevel(logging.DEBUG)
                try:
                    result = compute_truefsc(
                        str(map1),
                        str(map2),
                        str(plot_file),
                    )
                finally:
                    logger.removeHandler(handler)
                self.finished.emit(result)
            except Exception as e:
                self.error.emit(str(e))

    class ProgressDialog(QDialog):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setStyleSheet(_display_theme_stylesheet())
            self.setWindowTitle("trueFSC")
            self.setMinimumSize(500, 300)
            layout = QVBoxLayout(self)

            self.label = QLabel("Running trueFSC...")
            layout.addWidget(self.label)

            self.text_edit = QTextEdit()
            self.text_edit.setReadOnly(True)
            layout.addWidget(self.text_edit)

            self.close_btn = QPushButton("Close")
            self.close_btn.setEnabled(False)
            self.close_btn.clicked.connect(self.accept)
            layout.addWidget(self.close_btn)

        def append_line(self, line):
            self.text_edit.append(line)
            scrollbar = self.text_edit.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

        def set_result(self, result):
            if result and result.get("plot_file"):
                self.label.setText(
                    f"True FSC completed - Resolution: {result['resolution']:.2f} A"
                )
                viewer = _napari.active()
                if viewer is None:
                    viewer = _create_napari_viewer()
                _open_file(viewer, str(result["plot_file"]), mode="slice")
            else:
                self.label.setText("trueFSC completed")
            if output_dir != model_dir:
                self.text_edit.append(f"\nResults saved to: {output_dir}")
            self.close_btn.setEnabled(True)

        def set_error(self, error_msg):
            self.label.setText("trueFSC failed")
            self.text_edit.append(f"\nError: {error_msg}")
            self.close_btn.setEnabled(True)

    dialog = ProgressDialog(parent)
    worker = Worker()
    worker.line_received.connect(dialog.append_line)
    worker.finished.connect(dialog.set_result)
    worker.error.connect(dialog.set_error)
    worker.start()
    dialog.exec()


class _Images2StarActivationFilter(QObject):
    """Forward panel focus changes to the images2star window tracker.

    Mirrors how the gallery/text/FSC windows report activation so that
    ``tracker.active()`` always points at the most recently focused panel
    (reuse target when the ``New`` checkbox is off).
    """

    def __init__(self, window, tracker):
        super().__init__(window)
        self._window = window
        self._tracker = tracker
        window.installEventFilter(self)

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent

        if event.type() == QEvent.Type.ActivationChange and obj.isActiveWindow():
            self._tracker.on_activate(self._window)
        return False


def _open_images2star_tools(
    path: str, parent=None, reuse_window=None, tracker=None
) -> None:
    """Open the Images2Star tools panel (preview + save) for a dataset.

    Follows the gallery/text/FSC lifecycle: with ``reuse_window`` the panel
    reloads the new file in place; otherwise a fresh non-modal panel is shown
    (the file browser stays usable) and registered with ``tracker`` so later
    clicks reuse it unless the ``New`` checkbox is checked.
    """
    from pathlib import Path

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QMessageBox

    from helicon.lib.images2star_widget import Images2StarDialog

    try:
        path = str(Path(path).resolve())
        if (
            reuse_window is not None
            and _is_alive_widget(reuse_window)
            and isinstance(reuse_window, Images2StarDialog)
        ):
            reuse_window.load_path(path)
            reuse_window.show()
            reuse_window.raise_()
            reuse_window.activateWindow()
            return

        dialog = Images2StarDialog(path, parent=parent)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.setModal(False)
        if tracker is not None:
            tracker.register(dialog)
            dialog.destroyed.connect(lambda *_: tracker.on_close(dialog))
            _Images2StarActivationFilter(dialog, tracker)
        # Offset the panel at least half of its own width from the parent
        # so the two windows sit side by side instead of fully overlapping
        # the file browser (the dialog already resized itself in __init__).
        if parent:
            parent_geo = parent.geometry()
            offset_x = max(dialog.width(), parent_geo.width()) // 2
            dialog.move(parent_geo.x() + offset_x, parent_geo.y() + 40)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
    except Exception as exc:
        QMessageBox.critical(
            None,
            "Images2Star Error",
            f"Failed to open Images2Star tools:\n{exc}",
        )


def _open_helical_angle_stats_plot(
    result: dict,
) -> None:
    """Display a generated helical-angle variance plot in napari."""
    plot_file = result.get("plot_file")
    if not plot_file:
        raise ValueError("helical-angle variance calculation returned no plot")

    viewer = _napari.active()
    if viewer is None:
        viewer = _create_napari_viewer()
    _open_file(viewer, str(plot_file), mode="slice")


def _helical_angle_stats_paths(path: str) -> tuple[Path, Path, Path]:
    """Return the output directory, derived STAR path, and plot path."""
    import tempfile

    input_path = Path(path)
    input_dir = input_path.parent
    if os.access(input_dir, os.W_OK):
        output_dir = input_dir
    else:
        output_dir = Path(tempfile.mkdtemp(prefix="helicon_angle_stats_"))
    output_star = output_dir / f"{input_path.stem}.helical_angle_variance.star"
    plot_file = output_dir / f"{input_path.stem}.helical_angle_variance.pdf"
    return output_dir, output_star, plot_file


def _launch_helical_angle_stats(
    path: str,
    parent=None,
) -> None:
    """Compute and display Class3D/Refine3D helical-angle variance statistics."""
    from PySide6.QtCore import QThread, Signal
    from PySide6.QtWidgets import QDialog, QLabel, QPushButton, QTextEdit, QVBoxLayout

    input_path = Path(path)
    input_dir = input_path.parent
    output_dir, output_star, plot_file = _helical_angle_stats_paths(path)

    from helicon.plugins.images2star.estimatehelicalanglevariance import (
        estimate_helical_angle_variance_from_star,
    )

    class Worker(QThread):
        line_received = Signal(str)
        finished = Signal(object)
        error = Signal(str)

        def run(self):
            try:
                self.line_received.emit(f"Input: {input_path}")
                self.line_received.emit(f"Output STAR: {output_star}")
                self.line_received.emit(f"Plot: {plot_file}")
                self.line_received.emit("")
                result = estimate_helical_angle_variance_from_star(
                    str(input_path),
                    str(output_star),
                    str(plot_file),
                )
                self.finished.emit(result)
            except Exception as exc:
                self.error.emit(str(exc))

    class ProgressDialog(QDialog):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setStyleSheet(_display_theme_stylesheet())
            self.setWindowTitle("Helical-angle statistics")
            self.setMinimumSize(550, 300)
            layout = QVBoxLayout(self)

            self.label = QLabel("Calculating helical-angle statistics...")
            layout.addWidget(self.label)

            self.text_edit = QTextEdit()
            self.text_edit.setReadOnly(True)
            layout.addWidget(self.text_edit)

            self.close_btn = QPushButton("Close")
            self.close_btn.setEnabled(False)
            self.close_btn.clicked.connect(self.accept)
            layout.addWidget(self.close_btn)

        def append_line(self, line):
            self.text_edit.append(line)
            scrollbar = self.text_edit.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

        def set_result(self, result):
            try:
                _open_helical_angle_stats_plot(result)
            except Exception as exc:
                self.set_error(str(exc))
                return
            self.label.setText("Helical-angle statistics completed")
            if output_dir != input_dir:
                self.text_edit.append(f"\nResults saved to: {output_dir}")
            self.close_btn.setEnabled(True)

        def set_error(self, error_msg):
            self.label.setText("Helical-angle statistics failed")
            self.text_edit.append(f"\nError: {error_msg}")
            self.close_btn.setEnabled(True)

    dialog = ProgressDialog(parent)
    worker = Worker()
    worker.line_received.connect(dialog.append_line)
    worker.finished.connect(dialog.set_result)
    worker.error.connect(dialog.set_error)
    worker.start()
    dialog.exec()


def _hide_layer_panels(viewer) -> None:
    """Hide the left-side layer list and layer controls dock widgets.

    The display command shows the floating folder browser as the primary
    navigation surface, so the napari layer panel is hidden by default in
    both the main and any new display window. A middle-click on the canvas
    toggles it back.
    """
    try:
        from unittest.mock import MagicMock

        if isinstance(viewer.window, MagicMock):
            return
        qv = viewer.window._qt_viewer
        qv.dockLayerList.hide()
        qv.dockLayerControls.hide()
    except Exception:
        pass


def _install_panel_toggle(viewer) -> None:
    """Install a middle-click handler that toggles the layer panel.

    Middle-clicking the canvas shows/hides the left-side layer list and
    layer controls. The display area keeps its size and screen position:
    the window grows leftward when the panel opens and shrinks from the
    left when it closes, so the canvas is never resized.

    Three sources of unwanted zoom had to be defeated:

    * A Qt event filter consumes ``MouseButtonPress`` (triggers panel toggle),
      ``MouseButtonRelease`` (clears the shared ``_middle_held`` flag), and
      **``MouseButtonDblClick``** (napari binds ALL double-clicks to
      ``double_click_to_zoom`` which multiplies ``viewer.camera.zoom * 2`` —
      this was the root cause of the progressive zoom on rapid middle clicks).

    * A wrap around the camera's ``viewbox_mouse_event`` suppresses camera
      pan/zoom when the ``button`` or ``buttons`` field contains a middle value.
      VisPy and Qt number the middle button differently (3 vs 4), so all
      of ``(2, 3, 4)`` are checked.  The viewbox stores the connection as
      ``(camera, "viewbox_mouse_event")`` — a weak ref + method name — so
      patching the instance attribute (not the class) is honoured on each emit.

    * The ``_middle_held`` flag bridges two layers: the Qt event filter sets it
      on middle press; the camera wrap reads it to suppress ``mouse_wheel``
      events that have ``buttons=[]`` (no button reported).  These micro-scroll
      events are generated by the physical scroll wheel mechanism as it
      settles — they carry neither the middle button flag nor arrive through the
      Qt event filter, so only the flag can suppress them.

    napari swaps the camera instance when the 2D/3D display mode changes, so
    the wrap is re-applied whenever ``dims.ndisplay`` changes.

    References
    ----------
    VisPy button map (``_qt.py``): 1=left, 2=right, 3=middle
    Qt enum values: 1=left, 2=right, 4=middle
    napari double-click zoom: ``_viewer_mouse_bindings.py:double_click_to_zoom``
    """
    try:
        from unittest.mock import MagicMock

        from PySide6.QtCore import QEvent, QObject, Qt

        if isinstance(viewer.window, MagicMock):
            return
        qv = viewer.window._qt_viewer
        layer_list = qv.dockLayerList
        layer_controls = qv.dockLayerControls

        # Shared state: the event filter sets this on press/release, and the
        # camera wrap reads it to suppress micro-scroll events that arrive as
        # a side-effect of pressing the physical scroll wheel.
        _middle_held = [False]

        class _MiddleClickFilter(QObject):
            def __init__(
                self, layer_list, layer_controls, qt_viewer, viewer, parent=None
            ):
                super().__init__(parent)
                self._layer_list = layer_list
                self._layer_controls = layer_controls
                self._qt_viewer = qt_viewer
                self._viewer = viewer
                self._saved_dock_w = 0
                self._toggling = False
                self._pending_timer = None
                self._panel_shown = False

            def _toggle_panel(self):
                if self._toggling:
                    return
                if self._pending_timer is not None:
                    self._pending_timer.stop()
                    self._pending_timer = None
                self._toggling = True
                try:
                    self._do_toggle()
                finally:
                    self._toggling = False

            def _apply_geometry(self, handle, win, x, y, w, h):
                if handle:
                    handle.setGeometry(x, y, w, h)
                else:
                    win.setGeometry(x, y, w, h)

            def _cache_display_rect(self, handle, win):
                self._viewer._display_only_ba = win.saveGeometry()

            def _do_toggle(self):
                win = self._qt_viewer.window()
                if win is None:
                    return
                from PySide6.QtCore import QTimer
                from PySide6.QtWidgets import QStyle

                handle = win.windowHandle()
                style = self._layer_list.style()
                grip = max(
                    style.pixelMetric(QStyle.PixelMetric.PM_DockWidgetSeparatorExtent),
                    style.pixelMetric(QStyle.PixelMetric.PM_DockWidgetHandleExtent),
                    0,
                )
                hiding = self._layer_list.isVisible()
                base = handle.geometry() if handle else win.geometry()
                if hiding:
                    dock_w = self._layer_list.width() + grip
                    self._saved_dock_w = dock_w
                    self._layer_list.hide()
                    self._layer_controls.hide()
                    self._panel_shown = False
                    new_x = base.x() + dock_w
                    new_w = max(base.width() - dock_w, 1)
                    self._apply_geometry(
                        handle, win, new_x, base.y(), new_w, base.height()
                    )
                    self._cache_display_rect(handle, win)
                else:
                    self._cache_display_rect(handle, win)
                    if self._saved_dock_w > 0:
                        dock_w = self._saved_dock_w
                        new_x = base.x() - dock_w
                        new_w = base.width() + dock_w
                        self._apply_geometry(
                            handle, win, new_x, base.y(), new_w, base.height()
                        )
                        self._layer_list.show()
                        self._layer_controls.show()
                        self._panel_shown = True
                    else:
                        # Docks were hidden since cold start and have never
                        # been laid out, so width() is not yet valid. Resize the
                        # window first using a sizeHint estimate so the docks
                        # land in the new left-hand space (no canvas-overlap
                        # flash), then correct on the next turn once Qt has laid
                        # out the freshly-shown docks and width() is real.
                        est = self._layer_list.sizeHint().width() + grip
                        self._saved_dock_w = est
                        new_x = base.x() - est
                        new_w = base.width() + est
                        self._apply_geometry(
                            handle, win, new_x, base.y(), new_w, base.height()
                        )
                        self._layer_list.show()
                        self._layer_controls.show()
                        self._panel_shown = True
                        self._pending_timer = QTimer()
                        self._pending_timer.setSingleShot(True)
                        self._pending_timer.timeout.connect(
                            lambda: self._finish_first_show(handle, win, grip, est)
                        )
                        self._pending_timer.start(0)

            def _finish_first_show(self, handle, win, grip, est):
                self._pending_timer = None
                dock_w = self._layer_list.width() + grip
                self._saved_dock_w = dock_w
                # The window was already shifted left by ``est`` in the
                # current turn; correct only the remaining difference.
                delta = dock_w - est
                if delta == 0:
                    return
                base = handle.geometry() if handle else win.geometry()
                new_x = base.x() - delta
                new_w = base.width() + delta
                self._apply_geometry(handle, win, new_x, base.y(), new_w, base.height())

            def eventFilter(self, obj, event):
                etype = event.type()
                if etype in (
                    QEvent.MouseButtonPress,
                    QEvent.MouseButtonRelease,
                    QEvent.MouseButtonDblClick,
                ):
                    if event.button() == Qt.MouseButton.MiddleButton:
                        if etype == QEvent.MouseButtonPress:
                            _middle_held[0] = True
                            self._toggle_panel()
                            return True
                        elif etype == QEvent.MouseButtonRelease:
                            _middle_held[0] = False
                        elif etype == QEvent.MouseButtonDblClick:
                            return True
                return False

        canvas_native = getattr(getattr(qv, "canvas", None), "native", None)
        if isinstance(canvas_native, QObject):
            mf = _MiddleClickFilter(
                layer_list,
                layer_controls,
                qv,
                viewer,
                parent=canvas_native,
            )
            canvas_native.installEventFilter(mf)
        else:
            mf = _MiddleClickFilter(layer_list, layer_controls, qv, viewer, parent=qv)
            qv.installEventFilter(mf)
        viewer._panel_toggle = mf

        view = getattr(getattr(qv, "canvas", None), "view", None)
        if view is None:
            return

        # VisPy reports middle as 3 (its own enum), Qt reports it as 4.
        # VisPy right button is 2 — do NOT include it here.
        middle_values = (3, 4)

        def _wrap_camera() -> None:
            camera = getattr(view, "camera", None)
            if camera is None or getattr(camera, "_panel_toggle_wrapped", False):
                return
            original = camera.viewbox_mouse_event

            def _wrapped_viewbox_mouse_event(event) -> None:
                btn = getattr(event, "button", None)
                btns = getattr(event, "buttons", None) or []
                is_middle = (
                    btn in middle_values
                    or any(b in middle_values for b in btns)
                    or (
                        getattr(event, "type", None) == "mouse_wheel"
                        and _middle_held[0]
                    )
                )
                if is_middle:
                    try:
                        event.handled = True
                    except Exception:
                        pass
                    return
                original(event)

            camera.viewbox_mouse_event = _wrapped_viewbox_mouse_event
            camera._panel_toggle_wrapped = True

        _wrap_camera()
        try:
            viewer.dims.events.ndisplay.connect(lambda *a, **k: _wrap_camera())
        except Exception:
            pass
    except Exception:
        pass


def _create_napari_viewer(title="Helicon display"):
    """Create a new napari viewer with standard helicon customizations.

    Sets up the ``_SliceDirectionWidget`` and hides the layer panels so
    the viewer matches the default helicon look.  Raises
    ``HeliconDependencyError`` if napari or OpenGL is unavailable.
    """
    from unittest.mock import MagicMock

    napari = _load_napari()

    try:
        new_viewer = napari.Viewer(title=title)
    except Exception as exc:
        raise HeliconDependencyError(
            f"Failed to create the napari viewer: {exc}\n"
            "This can happen when no OpenGL-accelerated display is "
            "available.\nTry setting QT_QPA_PLATFORM=offscreen or "
            "updating your GPU drivers."
        ) from exc
    try:
        new_viewer.theme = _napari_display_theme()
        new_viewer.background_color = _napari_canvas_background()
    except Exception:
        pass
    _napari.register(new_viewer)
    _hide_layer_panels(new_viewer)
    try:
        if not isinstance(new_viewer.window, MagicMock):
            _SliceDirectionWidget(new_viewer).inject()
    except Exception:
        pass
    return new_viewer


def _is_text_file(path: str) -> bool:
    """Check if a file is a text file by attempting UTF-8 decode.

    Called only after known-type suffix checks have failed, so this is the
    fallback that distinguishes pure-text files from unknown binary.
    """
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
        if b"\x00" in chunk:
            return False
        chunk.decode("utf-8")
        return True
    except (UnicodeDecodeError, OSError):
        return False


def _open_text_window(path, reuse_window=None):
    """Open a text file in a standalone text window."""
    from pathlib import Path

    try:
        from PySide6.QtWidgets import QMainWindow, QTextEdit
        from PySide6.QtGui import QFont, QShortcut, QKeySequence
    except ImportError:
        return

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return

    if reuse_window is not None:
        try:
            if reuse_window.isVisible():
                if not reuse_window._maybe_save_before_replace(Path(path).name):
                    return reuse_window
                reuse_window._source_path = path
                reuse_window._text_edit.setPlainText(content)
                reuse_window._text_edit.document().setModified(False)
                reuse_window.setWindowTitle(f"Helicon - {Path(path).name}")
                reuse_window.show()
                reuse_window.raise_()
                return reuse_window
        except Exception:
            pass

    class _TextWindow(QMainWindow):
        def __init__(self, parent=None):
            super().__init__(parent)
            from PySide6.QtWidgets import (
                QWidget,
                QVBoxLayout,
                QHBoxLayout,
                QLineEdit,
                QToolButton,
            )
            from PySide6.QtCore import Qt

            self._source_path = path
            self.setProperty("helicon_theme_window", True)
            self.setStyleSheet(_display_theme_stylesheet())
            self.setPalette(_display_theme_palette())
            self.setWindowTitle(f"Helicon - {Path(path).name}")
            self.resize(700, 500)

            central = QWidget()
            layout = QVBoxLayout(central)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

            te = QTextEdit(self)
            # Editable: the buffer can be edited and saved to a *new* file via
            # Save As…; the original file on disk is never overwritten.
            te.setReadOnly(False)
            te.setFont(QFont("Courier New", 12))
            te.setLineWrapMode(QTextEdit.WidgetWidth)
            te.setStyleSheet(_display_theme_stylesheet())
            self._text_edit = te
            layout.addWidget(te, 1)

            action_bar = QWidget()
            action_bar.setStyleSheet(_display_theme_stylesheet())
            action_layout = QHBoxLayout(action_bar)
            action_layout.setContentsMargins(6, 4, 6, 4)
            action_layout.setSpacing(6)

            save_btn = QToolButton()
            save_btn.setText("Save As…")
            save_btn.setToolTip("Save the edited text to a new file (Ctrl+Shift+S)")
            save_btn.setStyleSheet(_display_theme_stylesheet())
            save_btn.clicked.connect(self._save_as)
            action_layout.addWidget(save_btn)
            action_layout.addStretch(1)
            layout.addWidget(action_bar)

            find_bar = QWidget()
            find_bar.setStyleSheet(_display_theme_stylesheet())
            find_layout = QHBoxLayout(find_bar)
            find_layout.setContentsMargins(6, 4, 6, 4)
            find_layout.setSpacing(6)

            find_input = QLineEdit()
            find_input.setPlaceholderText("Find…")
            find_input.setStyleSheet(_display_theme_stylesheet())
            find_input.returnPressed.connect(self._find_next)
            self._find_input = find_input

            close_btn = QToolButton()
            close_btn.setText("✕")
            close_btn.setToolTip("Close find bar")
            close_btn.clicked.connect(lambda: self._toggle_find_bar(False))
            close_btn.setStyleSheet(_display_theme_stylesheet())

            find_layout.addWidget(find_input, 1)
            find_layout.addWidget(close_btn)
            self._find_bar = find_bar
            self._find_bar_visible = False
            find_bar.hide()
            layout.addWidget(find_bar)

            self.setCentralWidget(central)

            find_sc = QShortcut(QKeySequence.StandardKey.Find, self)
            find_sc.activated.connect(lambda: self._toggle_find_bar(True))

            esc_sc = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
            esc_sc.activated.connect(self._close_find_bar)

            wrap_sc = QShortcut(QKeySequence("Ctrl+Shift+W"), self)
            wrap_sc.activated.connect(self._toggle_wrap)

            save_as_sc = QShortcut(QKeySequence("Ctrl+Shift+S"), self)
            save_as_sc.activated.connect(self._save_as)

            _install_window_shortcuts(self)

            self._text_edit.document().modificationChanged.connect(self._on_modified)

        def _on_modified(self, modified: bool) -> None:
            title = f"Helicon - {Path(self._source_path).name}"
            if modified:
                title += " *"
            self.setWindowTitle(title)

        def _save_as(self) -> None:
            from PySide6.QtWidgets import QFileDialog, QMessageBox

            source = Path(self._source_path)
            default = str(source.with_name(f"{source.stem}_edited{source.suffix}"))
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save As…",
                default,
                "All Files (*)",
            )
            if not filename:
                return
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(self._text_edit.toPlainText())
            except OSError as exc:
                QMessageBox.warning(
                    self, "Save failed", f"Could not write {filename}:\n{exc}"
                )
                return
            self._source_path = filename
            self._text_edit.document().setModified(False)
            self.statusBar().showMessage(f"Saved to {filename}", 4000)

        def _maybe_save_before_replace(self, new_name: str) -> bool:
            """Return True if it is safe to replace the buffer with *new_name*."""
            if not self._text_edit.document().isModified():
                return True
            from PySide6.QtWidgets import QMessageBox

            return self._confirm_unsaved(
                f"Save changes to {Path(self._source_path).name} "
                f"before opening {new_name}?"
            )

        def _maybe_save_before_close(self) -> bool:
            """Return True if it is safe to close the window."""
            if not self._text_edit.document().isModified():
                return True
            return self._confirm_unsaved(
                f"Save changes to {Path(self._source_path).name} before closing?"
            )

        def _confirm_unsaved(self, message: str) -> bool:
            """Ask Save / Discard / Cancel for unsaved changes.

            Returns True when the buffer may be discarded (saved or discarded
            explicitly); False when the user cancelled.
            """
            from PySide6.QtWidgets import QMessageBox

            buttons = (
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel
            )
            ret = QMessageBox.question(
                self,
                "Unsaved changes",
                message,
                buttons,
                QMessageBox.StandardButton.Save,
            )
            if ret == QMessageBox.StandardButton.Save:
                self._save_as()
                return not self._text_edit.document().isModified()
            return ret == QMessageBox.StandardButton.Discard

        def _toggle_find_bar(self, show: bool) -> None:
            if show:
                self._find_bar.show()
                self._find_input.setFocus()
                self._find_input.selectAll()
                self._find_bar_visible = True
            else:
                self._find_bar.hide()
                self._find_bar_visible = False
                self._text_edit.setFocus()

        def _close_find_bar(self) -> None:
            if self._find_bar_visible:
                self._toggle_find_bar(False)

        def _find_next(self) -> None:
            text = self._find_input.text()
            if not text:
                return
            from PySide6.QtGui import QTextCursor, QTextDocument

            found = self._text_edit.find(
                text, QTextDocument.FindFlag.FindCaseSensitively
            )
            if not found:
                cursor = self._text_edit.textCursor()
                cursor.movePosition(QTextCursor.MoveOperation.Start)
                self._text_edit.setTextCursor(cursor)
                self._text_edit.find(text, QTextDocument.FindFlag.FindCaseSensitively)

        def _toggle_wrap(self) -> None:
            current = self._text_edit.lineWrapMode()
            if current == QTextEdit.NoWrap:
                self._text_edit.setLineWrapMode(QTextEdit.WidgetWidth)
                self.statusBar().showMessage("Word wrap: ON", 2000)
            else:
                self._text_edit.setLineWrapMode(QTextEdit.NoWrap)
                self.statusBar().showMessage("Word wrap: OFF", 2000)

        def closeEvent(self, event):
            if self._text_edit.document().isModified():
                if not self._maybe_save_before_close():
                    event.ignore()
                    return
            _text.on_close(self)
            super().closeEvent(event)

        def changeEvent(self, event):
            from PySide6.QtCore import QEvent

            if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
                _text.on_activate(self)
            super().changeEvent(event)

    win = _TextWindow()
    win._text_edit.setPlainText(content)
    _text.register(win)
    win.show()
    return win


def _open_html(viewer, path: str) -> None:
    import webbrowser

    webbrowser.open(f"file://{path}")


def _open_pdf(viewer, path: str) -> None:
    """Open a PDF file, rendering pages as images in napari."""
    from pathlib import Path

    try:
        from PySide6.QtPdf import QPdfDocument
        from PySide6.QtCore import QSize
        from PySide6.QtGui import QImage
    except ImportError:
        return
    import numpy as np

    doc = QPdfDocument()
    doc.load(path)
    n_pages = doc.pageCount()
    if n_pages == 0:
        return

    dpi = 150
    pages = []
    max_w, max_h = 0, 0
    for i in range(n_pages):
        pt_size = doc.pagePointSize(i)
        w_px = int(pt_size.width() * dpi / 72)
        h_px = int(pt_size.height() * dpi / 72)
        max_w = max(max_w, w_px)
        max_h = max(max_h, h_px)

    for i in range(n_pages):
        pt_size = doc.pagePointSize(i)
        w_px = int(pt_size.width() * dpi / 72)
        h_px = int(pt_size.height() * dpi / 72)
        scale = min(max_w / w_px, max_h / h_px)
        rw = int(w_px * scale)
        rh = int(h_px * scale)
        img = doc.render(i, QSize(rw, rh))
        img = img.convertToFormat(QImage.Format.Format_ARGB32)
        ptr = img.bits()
        arr = np.frombuffer(bytes(ptr), dtype=np.uint8).reshape(
            img.height(), img.width(), 4
        )
        rgb = arr[:, :, 2::-1].astype(np.float32)
        alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
        composite = alpha * rgb + (1.0 - alpha) * 255.0
        canvas = np.full((max_h, max_w, 3), 255.0, dtype=np.float32)
        y0 = (max_h - rh) // 2
        x0 = (max_w - rw) // 2
        canvas[y0 : y0 + rh, x0 : x0 + rw] = composite
        pages.append(canvas)

    data = np.stack(pages) if len(pages) > 1 else pages[0]
    name = Path(path).name
    contrast = _auto_contrast(data)
    # Multi-page PDF is a stack of 2D pages, not a 3D volume: hide the
    # axis selector so it does not appear for the page-frame axis.
    _SliceDirectionWidget.set_stack_mode(len(pages) > 1)
    layer = viewer.add_image(
        data,
        name=name,
        contrast_limits=contrast,
        interpolation2d="linear",
        interpolation3d="linear",
    )
    _enable_continuous_auto_contrast(layer, viewer)
    layer.contrast_limits_range = (float(data.min()), float(data.max()))
    _reset_view(viewer)
    _hide_layer_panels(viewer)
    if len(pages) > 1:
        step = list(viewer.dims.current_step)
        step[0] = 0
        viewer.dims.current_step = step


def _open_eps(viewer, path: str) -> None:
    """Open an EPS (PostScript) file by rasterizing it to a PNG.

    Qt has no PostScript interpreter, so EPS cannot be read by QPdfDocument
    or QImageReader. We rasterize with Ghostscript (``gs``) when available,
    falling back to Quick Look (``qlmanage``) on macOS, then display the
    image reusing the same white-background compositing and contrast logic
    as the PDF viewer.
    """
    import numpy as np
    from PySide6.QtGui import QImage

    out_path = _rasterize_eps(path)
    if out_path is None:
        print(
            "[helicon] Ghostscript (gs) is required to display EPS files but "
            "was not found on your PATH. Install it with "
            "'conda install -c conda-forge ghostscript' or "
            "'brew install ghostscript'."
        )
        return

    try:
        img = QImage(out_path)
        if img.isNull():
            print(f"[helicon] failed to render EPS: {path}")
            return
        img = img.convertToFormat(QImage.Format.Format_ARGB32)
        ptr = img.bits()
        arr = np.frombuffer(bytes(ptr), dtype=np.uint8).reshape(
            img.height(), img.width(), 4
        )
        rgb = arr[:, :, 2::-1].astype(np.float32)  # BGRA -> RGB float
        alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
        # Composite onto white background
        composite = alpha * rgb + (1.0 - alpha) * 255.0

        name = Path(path).name
        contrast = _auto_contrast(composite)
        layer = viewer.add_image(
            composite,
            name=name,
            contrast_limits=contrast,
            interpolation2d="linear",
            interpolation3d="linear",
        )
        _enable_continuous_auto_contrast(layer, viewer)
        layer.contrast_limits_range = (
            float(composite.min()),
            float(composite.max()),
        )
        _reset_view(viewer)
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass
        # qlmanage writes into its own temp dir; drop the empty dir as well.
        try:
            os.rmdir(Path(out_path).parent)
        except OSError:
            pass


def _rasterize_eps(path: str) -> str | None:
    """Rasterize an EPS file to a PNG and return its path, or None on failure.

    Tries Ghostscript first (all platforms), then Quick Look (``qlmanage``)
    on macOS. The caller owns the returned temporary file and is expected
    to remove it.
    """
    import shutil
    import subprocess
    import sys
    import tempfile

    gs = shutil.which("gs") or shutil.which("ghostscript")
    if gs is not None:
        tmp_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_png.close()
        try:
            subprocess.run(
                [
                    gs,
                    "-dQUIET",
                    "-dNOPAUSE",
                    "-dBATCH",
                    "-dSAFER",
                    "-dEPSCrop",
                    "-sDEVICE=png16m",
                    "-r150",
                    f"-sOutputFile={tmp_png.name}",
                    path,
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return tmp_png.name
        except (subprocess.CalledProcessError, OSError):
            try:
                os.remove(tmp_png.name)
            except OSError:
                pass

    if sys.platform == "darwin":
        qlmanage = shutil.which("qlmanage")
        if qlmanage is not None:
            out_dir = tempfile.mkdtemp(prefix="helicon_eps_")
            try:
                result = subprocess.run(
                    [qlmanage, "-t", "-s", "2048", "-o", out_dir, path],
                    capture_output=True,
                    timeout=60,
                )
            except (subprocess.SubprocessError, OSError):
                result = None
            candidate = Path(out_dir) / f"{Path(path).name}.png"
            if result is not None and result.returncode == 0 and candidate.is_file():
                return str(candidate)
            try:
                os.rmdir(out_dir)
            except OSError:
                pass

    return None


def _capped_cylinder_mesh(p1, p2, radius, segments=24):
    """Vertices + triangle indices for a capped cylinder between ``p1`` and
    ``p2`` (ChimeraX BILD semantics: capped by default).

    Ported from ChimeraX ``shape.cylinder_geometry``: the side is a tube and
    each flat end is closed with a triangle fan so the rod looks solid.
    """
    import numpy as np

    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    axis = p2 - p1
    length = float(np.linalg.norm(axis))
    if length == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    a = axis / length
    helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(a, helper)
    u /= np.linalg.norm(u)
    v = np.cross(a, u)

    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    ring = radius * (np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v)
    bot = p1 + ring
    top = p2 + ring
    verts = np.vstack([bot, top, p1, p2])
    n_bot, n_top = 2 * segments, segments
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([i, j, n_top + j])
        faces.append([i, n_top + j, n_top + i])
        faces.append([n_bot, j, i])
        faces.append([n_bot + 1, n_top + i, n_top + j])
    return verts, np.array(faces, int)


def _sphere_mesh(center, radius, rings=16, segments=24):
    """Vertices + triangle indices for a UV sphere (always closed)."""
    import numpy as np

    center = np.asarray(center, float)
    verts = [center]
    for i in range(1, rings):
        phi = np.pi * i / rings
        r = radius * np.sin(phi)
        y = radius * np.cos(phi)
        theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
        for t in theta:
            verts.append(center + np.array([r * np.cos(t), y, r * np.sin(t)]))
    verts.append(center + np.array([0.0, radius, 0.0]))
    verts = np.array(verts)
    n = len(verts)
    pole_s, pole_n = 0, n - 1
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([pole_s, j + 1, i + 1])
    for k in range(rings - 2):
        base = 1 + k * segments
        nxt = base + segments
        for i in range(segments):
            j = (i + 1) % segments
            faces.append([base + i, base + j, nxt + j])
            faces.append([base + i, nxt + j, nxt + i])
    last_base = 1 + (rings - 2) * segments
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([pole_n, last_base + j, last_base + i])
    return verts, np.array(faces, int)


def _open_bild(viewer, path: str) -> None:
    from pathlib import Path
    import numpy as np

    # Each primitive becomes a capped 3D surface (ChimeraX semantics: cylinders
    # are solid/capped unless the ``open`` keyword is given). We render every
    # primitive as its own surface layer so per-object colors are preserved.
    meshes = []  # list of (vertices, faces, color)
    current_color = [1.0, 1.0, 1.0]

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(".color"):
                parts = line.split()
                current_color = [float(parts[1]), float(parts[2]), float(parts[3])]
            elif line.startswith(".cylinder"):
                parts = line.split()
                x1, y1, z1 = float(parts[1]), float(parts[2]), float(parts[3])
                x2, y2, z2 = float(parts[4]), float(parts[5]), float(parts[6])
                r = float(parts[7])
                # ``open`` keyword (8th field) leaves the cylinder uncapped.
                capped = not (len(parts) > 8 and parts[8] == "open")
                verts, faces = _capped_cylinder_mesh((x1, y1, z1), (x2, y2, z2), r)
                if not capped:
                    # Drop the two end-cap fans (last 2 * segments triangles);
                    # approximate by rebuilding an uncapped tube.
                    n = len(verts)
                    segs = (n - 4) // 2
                    side_faces = faces[: 2 * segs]
                    verts = verts[: 2 * segs]
                    faces = side_faces
                meshes.append((verts, faces, current_color))
            elif line.startswith(".sphere"):
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                r = float(parts[4])
                verts, faces = _sphere_mesh((x, y, z), r)
                meshes.append((verts, faces, current_color))

    if not meshes:
        return

    # Merge every primitive into a single surface layer. A BILD file can hold
    # thousands of objects (e.g. an angular-distribution plot); one layer per
    # object would create thousands of layers and stall napari's 3D view.
    all_verts = []
    all_faces = []
    all_colors = []
    offset = 0
    for verts, faces, color in meshes:
        if len(verts) == 0 or len(faces) == 0:
            continue
        all_verts.append(verts)
        all_faces.append(faces + offset)
        all_colors.append(np.tile(np.asarray(color, float), (len(verts), 1)))
        offset += len(verts)

    if not all_verts:
        return

    vertices = np.vstack(all_verts)
    faces = np.vstack(all_faces)
    vertex_colors = np.vstack(all_colors)
    name = Path(path).name
    viewer.add_surface(
        (vertices, faces),
        name=name,
        vertex_colors=vertex_colors,
        shading="smooth",
    )
    viewer.dims.ndisplay = 3
    _reset_view(viewer)


def _patch_napari_value_bug() -> None:
    """Patch napari 0.8.0 bugs that crash the UI in 3D display mode.

    Two independent napari 0.8.0 bugs are triggered by switching a volume
    layer to 3D (``viewer.dims.ndisplay = 3``):

    1. ``ScalarFieldBase._get_value_3d`` computes a 2D ``slice_shape`` but
       indexes it with a 3D ``dims_displayed`` (``slice_shape`` is
       ``(256, 256)`` while ``level0_shape`` is ``(256, 256, 256)``),
       raising ``ValueError`` from ``_get_value_ray``. The ``StatusChecker``
       thread calls ``get_value`` on every mouse move, so this crashes the
       UI thread.

    2. ``Image._update_thumbnail`` receives a 1D thumbnail (the napari
       projection collapses the 2D slice to 1D) while ``zoom_factor`` is
       2D, so ``scipy.ndimage.zoom`` raises ``RuntimeError``. This fires
       during the re-slice that ``ndisplay = 3`` triggers.

    In both cases the failure is non-essential (cursor readout / layer
    thumbnail), so we patch the class methods once and degrade gracefully
    instead of crashing.

    Per-instance attribute patching did not take effect because the
    ``StatusChecker`` thread resolves the class method rather than the
    monkey-patched instance attribute, so we patch the classes directly.
    """
    try:
        from napari.layers._scalar_field.scalar_field import ScalarFieldBase
    except Exception:
        ScalarFieldBase = None

    try:
        from napari.layers.image.image import Image as _NapariImage
    except Exception:
        _NapariImage = None

    if ScalarFieldBase is not None and not getattr(
        ScalarFieldBase, "_helicon_value_patched", False
    ):
        _orig_get_value_3d = ScalarFieldBase._get_value_3d

        def _safe_get_value_3d(
            self, start_point=None, end_point=None, dims_displayed=None
        ):
            try:
                return _orig_get_value_3d(
                    self,
                    start_point=start_point,
                    end_point=end_point,
                    dims_displayed=dims_displayed,
                )
            except (ValueError, IndexError, RuntimeError):
                return None

        ScalarFieldBase._get_value_3d = _safe_get_value_3d
        ScalarFieldBase._helicon_value_patched = True

    if _NapariImage is not None and not getattr(
        _NapariImage, "_helicon_thumb_patched", False
    ):
        _orig_update_thumbnail = _NapariImage._update_thumbnail

        def _safe_update_thumbnail(self):
            try:
                return _orig_update_thumbnail(self)
            except (ValueError, IndexError, RuntimeError):
                return None

        _NapariImage._update_thumbnail = _safe_update_thumbnail
        _NapariImage._helicon_thumb_patched = True


def _patch_napari_icon() -> None:
    """Point napari's window icon at the Helicon icon.

    napari's ``_QtMainWindow`` calls ``QApplication.setWindowIcon`` with the
    napari logo both when a viewer is created and whenever the theme changes
    (``_update_logo``). On macOS that call also swaps the Dock icon away from
    Helicon. ``_QtMainWindow._get_window_icon`` intentionally reads a
    ``_window_icon`` attribute when present (napari's documented extension
    point for custom window icons), so we install the Helicon icon there:
    every napari ``setWindowIcon`` call then installs the Helicon logo, and
    no napari code path can replace it afterwards.

    napari renders window icons through ``QSvgRenderer``, so the resource is
    an SVG embedding a downscaled copy of the Helicon PNG.
    """
    try:
        from napari._qt.qt_main_window import (
            _QtMainWindow as _NapariMainWindow,
        )
    except Exception:
        try:
            # napari < 0.5 exported the (then public) QtMainWindow name.
            from napari._qt.qt_main_window import (
                QtMainWindow as _NapariMainWindow,
            )
        except Exception:
            return
    if getattr(_NapariMainWindow, "_helicon_icon_patched", False):
        return
    svg_path = Path(__file__).parent.parent / "resources" / "icon.svg"
    if not svg_path.is_file():
        return
    try:
        orig_init = _NapariMainWindow.__init__

        def _init_with_helicon_icon(self, *args, **kwargs):
            self._window_icon = str(svg_path)
            return orig_init(self, *args, **kwargs)

        _NapariMainWindow.__init__ = _init_with_helicon_icon
        _NapariMainWindow._helicon_icon_patched = True
    except Exception:
        pass


def _is_metadata_star(path: str) -> bool:
    """Star files that describe pipelines/optimisation rather than image data.

    These should be opened as text, not as an image/volume stack.
    """
    from pathlib import Path

    name = Path(path).name.lower()
    return any(name.endswith(suffix) for suffix in _METADATA_STAR_SUFFIXES)


def _is_optimiser_star(path: str) -> bool:
    """Return True for RELION optimiser.star files.

    These contain references to MRC files whose center slices can be
    displayed in a gallery view.
    """
    from pathlib import Path

    return Path(path).name.lower().endswith("optimiser.star")


def _parse_optimiser_star(optimiser_path: str) -> list[str] | None:
    """Parse a RELION optimiser.star file and extract referenced MRC file paths.

    The optimiser.star file references model.star files, which in turn
    contain the actual MRC file paths in the ``_rlnReferenceImage`` column.

    Parameters
    ----------
    optimiser_path : str
        Path to the optimiser.star file.

    Returns
    -------
    list of str or None
        List of resolved MRC file paths, or None if parsing fails.
    """
    from pathlib import Path

    star_dir = Path(optimiser_path).parent
    model_star_paths: list[str] = []

    try:
        with open(optimiser_path) as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue

                if s.startswith("_rlnModelStarFile"):
                    parts = s.split()
                    if len(parts) >= 2:
                        model_rel = parts[-1]
                        for ancestor in [star_dir] + list(star_dir.parents):
                            candidate = ancestor / model_rel
                            if candidate.is_file():
                                model_star_paths.append(str(candidate))
                                break
    except Exception:
        return None

    if not model_star_paths:
        return None

    mrc_paths: list[str] = []
    for model_path in model_star_paths:
        result = _parse_model_star(model_path)
        if result:
            for p in result:
                if p not in mrc_paths:
                    mrc_paths.append(p)

    return mrc_paths if mrc_paths else None


def _parse_model_star(model_path: str) -> list[str] | None:
    """Extract referenced MRC file paths from a RELION model.star.

    Reads the ``data_model_classes`` section and resolves the
    ``_rlnReferenceImage`` column to absolute MRC paths.
    """
    from pathlib import Path

    model_dir = Path(model_path).parent
    in_loop = False
    in_data_model_classes = False
    col_names: list[str] = []
    ref_image_col_idx = -1
    mrc_paths: list[str] = []

    try:
        with open(model_path) as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue

                if s.startswith("data_"):
                    in_data_model_classes = "model_classes" in s
                    in_loop = False
                    col_names = []
                    ref_image_col_idx = -1
                    continue

                if s == "loop_":
                    in_loop = True
                    col_names = []
                    ref_image_col_idx = -1
                    continue

                if in_loop and s.startswith("_"):
                    col_names.append(s.split()[0])
                    if "referenceimage" in s.lower():
                        ref_image_col_idx = len(col_names) - 1
                    continue

                if not in_data_model_classes or ref_image_col_idx < 0:
                    continue

                if not in_loop:
                    continue

                parts = s.split()
                if ref_image_col_idx >= len(parts):
                    continue

                mrc_rel = parts[ref_image_col_idx]

                resolved = None
                for ancestor in [model_dir] + list(model_dir.parents):
                    candidate = ancestor / mrc_rel
                    if candidate.is_file():
                        resolved = str(candidate)
                        break

                if resolved and resolved not in mrc_paths:
                    mrc_paths.append(resolved)
    except Exception:
        return None

    return mrc_paths if mrc_paths else None


def _parse_class2d_model_star(
    model_path: str,
) -> tuple[list[tuple[str, int]], list[float]] | None:
    """Extract MRC references and class distributions from a RELION model.star.

    Reads the ``data_model_classes`` section and resolves
    ``_rlnReferenceImage`` (which may be ``idx@path.mrcs`` for Class2D)
    and ``_rlnClassDistribution`` (abundance) columns.

    Returns
    -------
    tuple of (entries, distributions) or None
        * ``entries``: list of ``(mrc_path, frame_idx)`` tuples.
        * ``distributions``: list of class distribution values (0-1).
    """
    from pathlib import Path

    model_dir = Path(model_path).parent
    in_loop = False
    in_data_model_classes = False
    col_names: list[str] = []
    ref_image_col_idx = -1
    class_dist_col_idx = -1
    entries: list[tuple[str, int]] = []
    distributions: list[float] = []

    try:
        with open(model_path) as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue

                if s.startswith("data_"):
                    in_data_model_classes = "model_classes" in s
                    in_loop = False
                    col_names = []
                    ref_image_col_idx = -1
                    class_dist_col_idx = -1
                    continue

                if s == "loop_":
                    in_loop = True
                    col_names = []
                    ref_image_col_idx = -1
                    class_dist_col_idx = -1
                    continue

                if in_loop and s.startswith("_"):
                    col_name = s.split()[0]
                    col_names.append(col_name)
                    if "referenceimage" in col_name.lower():
                        ref_image_col_idx = len(col_names) - 1
                    elif "classdistribution" in col_name.lower():
                        class_dist_col_idx = len(col_names) - 1
                    continue

                if not in_data_model_classes:
                    continue

                if not in_loop:
                    continue

                parts = s.split()
                if ref_image_col_idx < 0 or ref_image_col_idx >= len(parts):
                    continue

                ref_raw = parts[ref_image_col_idx]
                frame_idx = 0
                if "@" in ref_raw:
                    idx_str, file_part = ref_raw.split("@", 1)
                    frame_idx = int(idx_str) - 1
                else:
                    file_part = ref_raw

                resolved = None
                for ancestor in [model_dir] + list(model_dir.parents):
                    candidate = ancestor / file_part
                    if candidate.is_file():
                        resolved = str(candidate)
                        break

                if not resolved:
                    continue

                dist = 0.0
                if class_dist_col_idx >= 0 and class_dist_col_idx < len(parts):
                    try:
                        dist = float(parts[class_dist_col_idx])
                    except ValueError:
                        dist = 0.0

                entries.append((resolved, frame_idx))
                distributions.append(dist)
    except Exception:
        return None

    if not entries:
        return None

    return entries, distributions


def _parse_star_image_refs(
    star_path: str,
) -> tuple[list[tuple[int, str, float]], tuple, float, int] | None:
    """Parse a .star file line-by-line and build lazy image-stack entries.

    Extracts only the ImageName/MicrographName column instead of loading the
    entire file into a pandas DataFrame (which blocks for minutes on large
    *data.star files with millions of particles). Resolved MRC paths are
    cached because most particles reference frames from the same file.

    Parameters
    ----------
    star_path : str
        Path to the .star file.

    Returns
    -------
    tuple of (entries, first_shape, first_apix, n_skipped) or None
        * ``entries``: list of ``(frame_idx_0based, mrc_path, 0.0)`` tuples.
        * ``first_shape``: ``(nx, ny)`` or ``(nx, ny, nz)`` of the first image.
        * ``first_apix``: pixel size in Angstroms (fallback 1.0).
        * ``n_skipped``: number of data lines whose binary images could not
          be found on disk.
        Returns None if no image references could be resolved.
    """
    from pathlib import Path

    import mrcfile

    star_dir = Path(star_path).parent

    col_names: list[str] = []
    image_col_idx = -1
    in_loop = False
    in_data = False
    entries: list[tuple[int, str, float]] = []
    first_shape: tuple | None = None
    first_apix = 1.0
    n_skipped = 0
    resolved_cache: dict[str, str | None] = {}

    def _resolve(img_rel: str) -> str | None:
        if img_rel in resolved_cache:
            return resolved_cache[img_rel]
        resolved = None
        for ancestor in [star_dir] + list(star_dir.parents):
            candidate = ancestor / img_rel
            if candidate.is_file():
                resolved = str(candidate)
                break
        resolved_cache[img_rel] = resolved
        return resolved

    try:
        with open(star_path) as f:
            for line in f:
                raw = line.rstrip("\n\r")
                s = raw.strip()
                if not s or s.startswith("#"):
                    continue

                if s.startswith("data_") or s == "loop_":
                    in_loop = s == "loop_"
                    in_data = False
                    if in_loop:
                        col_names = []
                        image_col_idx = -1
                    continue

                if in_loop and s.startswith("_"):
                    col_names.append(s.split()[0])
                    if image_col_idx < 0:
                        cl = s.lower()
                        if "imagename" in cl or "micrographname" in cl:
                            image_col_idx = len(col_names) - 1
                    continue

                if image_col_idx < 0:
                    continue

                if not in_data:
                    in_data = True

                parts = raw.split()
                if image_col_idx >= len(parts):
                    continue
                image_ref = parts[image_col_idx]

                if "@" in image_ref:
                    idx_str, img_rel = image_ref.split("@", 1)
                else:
                    idx_str, img_rel = "1", image_ref

                try:
                    frame_idx = int(idx_str) - 1
                except ValueError:
                    continue

                resolved_path = _resolve(img_rel)
                if resolved_path is None:
                    n_skipped += 1
                    continue

                entries.append((frame_idx, resolved_path, 0.0))

                if first_shape is None:
                    with mrcfile.open(resolved_path, header_only=True) as mrc:
                        nx = int(mrc.header.nx)
                        ny = int(mrc.header.ny)
                        nz = int(mrc.header.nz)
                        first_shape = (
                            (nx, ny)
                            if nz == 1 or Path(resolved_path).suffix.lower() == ".mrcs"
                            else (nx, ny, nz)
                        )
                        first_apix = float(mrc.voxel_size.x)
                        if first_apix <= 0:
                            first_apix = 1.0
    except Exception:
        return None

    if not entries or first_shape is None:
        if n_skipped:
            return [], first_shape or (0, 0), first_apix, n_skipped
        return None

    return entries, first_shape, first_apix, n_skipped


def _set_ndisplay(viewer, value: int) -> None:
    """Set the viewer's 2D/3D display dimension, tolerating empty dims.

    Setting ``dims.ndisplay`` before any layer exists (or on a mock
    viewer in tests) is harmless; guard so a transient state never
    raises and aborts the open.
    """
    try:
        viewer.dims.ndisplay = value
    except (AttributeError, RuntimeError, ValueError):
        pass


def _reset_view(viewer) -> None:
    """Recenter/zoom the camera to fit the current layers ("home" view).

    Called after a new file's layer is added so the incoming file is
    framed correctly rather than left at the previous file's camera pose.
    Guarded so a mock viewer in tests (or a transient state) never
    raises and aborts the open.
    """
    try:
        viewer.reset_view()
    except (AttributeError, RuntimeError, ValueError):
        pass


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

_NAPARI_MODES = {"slice", "volume", "3dplot", "stats", "html"}
_GALLERY_MODES = {"gallery", "optimiser", "2dclasses", "orthogonal"}
_TEXT_MODES = {"text"}
_PLOT_MODES = {"fsc"}
_IMAGES2STAR_MODES = {"images2star"}


def _quit_all_windows():
    """Close every tracked window and the file browser, then quit."""
    from PySide6.QtWidgets import QApplication

    for tracker in (_napari, _gallery, _text, _plot, _images2star):
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


def _wrap_gallery_with_panel(gallery: "ImageGalleryWidget") -> "QWidget":
    """Wrap an ImageGalleryWidget with a left-side _ControlPanel sibling.

    The panel is prepended to the left.  Toggling it grows the parent
    window leftward by ``_ControlPanel.PANEL_WIDTH`` so the gallery
    widget keeps both its width and its screen position unchanged.
    """
    from PySide6.QtWidgets import QHBoxLayout, QSizePolicy, QWidget

    from helicon.lib.gallery_widget import _ControlPanel

    wc = _ControlPanel.PANEL_WIDTH

    panel = _ControlPanel()
    panel.hide()
    panel.setFixedWidth(wc)
    panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)

    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    layout.addWidget(panel)
    layout.addWidget(gallery, 1)

    def _on_toggle():
        win = container.window()
        visible = not panel.isVisible()
        # Operate on the underlying QWindow's geometry, which is the FRAME
        # rect (title bar + borders included) and is not subject to the
        # client-vs-frame conversion that QWidget.setGeometry does.  This
        # keeps the frame's top-left y pinned across toggles on macOS,
        # Linux, and Windows/WSL -- anchoring the client rect would let the
        # title-bar height accumulate into y on every toggle.
        handle = win.windowHandle()
        if handle is None:
            # Fall back to QWidget geometry (no platform window yet).
            if visible:
                panel.show()
                win.setGeometry(win.x() - wc, win.y(), win.width() + wc, win.height())
            else:
                panel.hide()
                win.setGeometry(
                    win.x() + wc, win.y(), max(win.width() - wc, 1), win.height()
                )
            return
        fg = handle.geometry()
        if visible:
            panel.show()
            handle.setGeometry(fg.x() - wc, fg.y(), fg.width() + wc, fg.height())
        else:
            panel.hide()
            handle.setGeometry(
                fg.x() + wc, fg.y(), max(fg.width() - wc, 1), fg.height()
            )

    gallery.panel_toggle_requested.connect(_on_toggle)

    def _on_brightness(val):
        gallery.set_brightness(val / 100.0)
        panel._brightness_val.setText(f"{gallery._brightness:.2f}")
        _refresh_histogram()

    def _on_contrast(val):
        gallery.set_contrast(val / 100.0)
        panel._contrast_val.setText(f"{gallery._contrast:.2f}")
        _refresh_histogram()

    def _on_gamma(val):
        gallery.set_gamma(val / 100.0)
        panel._gamma_val.setText(f"{gallery._gamma:.2f}")
        _refresh_histogram()

    def _on_log_transform(checked):
        gallery.set_log_transform(checked)
        _refresh_histogram()

    def _refresh_histogram():
        if gallery.has_data() and panel._histogram_chk.isChecked():
            panel._histogram_widget.update_histogram(
                gallery._read_fn,
                gallery._n,
                gallery._brightness,
                gallery._contrast,
                gallery._gamma,
                gallery._log_transform,
            )

    def _on_histogram_toggled(checked):
        panel._histogram_widget.setVisible(checked)
        _refresh_histogram()

    def _on_autocontrast():
        panel._brightness_slider.setValue(0)
        panel._contrast_slider.setValue(100)
        panel._gamma_slider.setValue(100)

    def _on_scope_selected(checked):
        if checked:
            gallery.set_adjust_scope("selected")

    def _on_scope_all(checked):
        if checked:
            gallery.set_adjust_scope("all")

    panel._brightness_slider.valueChanged.connect(_on_brightness)
    panel._contrast_slider.valueChanged.connect(_on_contrast)
    panel._gamma_slider.valueChanged.connect(_on_gamma)
    panel._auto_btn.clicked.connect(_on_autocontrast)
    panel._radio_selected.toggled.connect(_on_scope_selected)
    panel._radio_all.toggled.connect(_on_scope_all)
    panel.log_changed.connect(_on_log_transform)
    panel.histogram_changed.connect(_on_histogram_toggled)

    def _on_show_labels(checked):
        gallery.set_show_labels(checked)

    panel.show_labels_changed.connect(_on_show_labels)

    if hasattr(gallery, "view_changed"):
        gallery.view_changed.connect(_refresh_histogram)

    _refresh_histogram()

    return container


def _open_gallery(
    read_fn, n, img_w, img_h, apix, name, reuse_window=None, tracker=None
) -> None:
    """Show the lazy thumbnail grid for a stack in a standalone window.

    Parameters
    ----------
    read_fn : callable
        ``read_fn(i) -> numpy.ndarray`` returning frame ``i`` lazily.
    n : int
        Number of frames in the stack.
    img_w : int
        Native frame width.
    img_h : int
        Native frame height.
    apix : float
        Pixel size (A/px) for the slice view scale.
    name : str
        Display name (usually the file name).
    tracker : _DisplayTracker, optional
        Tracker to register new windows with for lifecycle management.
    """
    from helicon.lib.gallery_backends import StackGallery

    gallery = StackGallery(
        star_path=name,
        read_fn=read_fn,
        n=n,
        img_w=img_w,
        img_h=img_h,
        apix=apix,
    )
    return gallery.open(reuse_window=reuse_window, tracker=tracker)


def _model_fsc_curves(data, *, use_ssnr_map: bool = False):
    """Extract resolution curves from parsed RELION model STAR data.

    Class3D does not use gold-standard half-set refinement, so its
    ``rlnGoldStandardFsc`` values are zero.  For those jobs, convert
    ``rlnSsnrMap`` to the FSC-equivalent MAP weight
    ``W_MAP = SSNR_MAP / (1 + SSNR_MAP)``.
    """
    import numpy as np

    all_curves = []
    class_sections = [
        (int(key.rsplit("_", 1)[-1]), value)
        for key, value in data.items()
        if re.match(r"^(?:data_)?model_class_\d+$", str(key), re.IGNORECASE)
    ]
    for class_num, df in sorted(class_sections):

        sf_col = None
        ang_col = None
        gold_standard_fsc_col = None
        fallback_fsc_col = None
        ssnr_map_col = None
        for col in df.columns:
            cl = col.lower().lstrip("_")
            if "goldstandardfsc" in cl:
                gold_standard_fsc_col = col
            elif "fouriershellcorrelation" in cl and "phase" not in cl:
                fallback_fsc_col = col
            if cl == "rlnssnrmap":
                ssnr_map_col = col
            if cl == "rlnresolution":
                sf_col = col
            if cl == "rlnangstromresolution":
                ang_col = col

        curve_col = (
            ssnr_map_col if use_ssnr_map else gold_standard_fsc_col or fallback_fsc_col
        )
        if sf_col is None or curve_col is None:
            continue

        spatial_freq = np.asarray(df[sf_col], dtype=np.float64)
        curve = np.asarray(df[curve_col], dtype=np.float64)
        if use_ssnr_map:
            curve = curve / (1.0 + curve)

        order = np.argsort(spatial_freq)
        spatial_freq = spatial_freq[order]
        curve = curve[order]

        angstrom = None
        if ang_col is not None:
            angstrom = np.asarray(df[ang_col], dtype=np.float64)[order]

        all_curves.append((spatial_freq, curve, f"Class {class_num}", angstrom))

    return all_curves


_ITERATION_FILE_RE = re.compile(r"(?:^|_)it(\d+)(?:_(.*))?_model\.star$", re.IGNORECASE)


def _iteration_label(path: str | Path) -> str:
    """Human-readable label for a model.star iteration file.

    Numbered RELION snapshots are labelled with the plain iteration number
    (e.g. ``run_it025_model.star`` becomes ``25``) so a row of many checkboxes
    stays compact; files without an iteration marker keep their file stem
    (e.g. ``run_model.star`` becomes ``run_model``).
    """
    path = Path(path)
    match = _ITERATION_FILE_RE.search(path.name)
    if match:
        return str(int(match.group(1)))
    if path.name.lower() == "run_model.star":
        return "final"
    return path.stem


def _iteration_model_files(star_path: str | Path) -> list[tuple[Path, int | None]]:
    """Return refinement-iteration model.star files next to ``star_path``.

    RELION writes one ``run_itNNN_model.star`` per refinement iteration, so the
    "all iterations" set for an FSC plot is every ``*_model.star`` file in the
    same job directory.  Files are sorted by iteration number, and the selected
    file is always included even when its name carries no iteration marker
    (e.g. the final ``run_model.star``).

    Parameters
    ----------
    star_path : str or Path
        The model.star file selected in the file browser.

    Returns
    -------
    list of (Path, int or None)
        ``(path, iteration_number)`` pairs ordered by iteration number;
        ``iteration_number`` is ``None`` for files without an ``_itNNN_``
        marker.
    """
    path = Path(star_path)
    numbered: dict[int, list[tuple[Path, str | None]]] = {}
    unnumbered: list[Path] = []
    candidates = sorted(path.parent.glob("*model.star"))
    if path not in candidates:
        candidates.append(path)
    for candidate in candidates:
        match = _ITERATION_FILE_RE.search(candidate.name)
        if match:
            numbered.setdefault(int(match.group(1)), []).append(
                (candidate, match.group(2))
            )
        elif candidate.name.lower() == "run_model.star":
            # RELION's final Refine3D model is not tagged with an iteration
            # number, but it belongs at the end of the iteration list.
            unnumbered.append(candidate)

    selected_match = _ITERATION_FILE_RE.search(path.name)
    preferred_variant = selected_match.group(2) if selected_match else None
    matches: list[tuple[Path, int | None]] = []
    for iteration_number in sorted(numbered):
        candidates_for_iteration = numbered[iteration_number]
        selected_candidate = next(
            (
                candidate
                for candidate, _variant in candidates_for_iteration
                if candidate == path
            ),
            None,
        )
        if selected_candidate is not None:
            chosen = selected_candidate
        else:
            chosen = next(
                (
                    candidate
                    for candidate, variant in candidates_for_iteration
                    if variant == preferred_variant
                ),
                candidates_for_iteration[0][0],
            )
        matches.append((chosen, iteration_number))

    for candidate in unnumbered:
        matches.append((candidate, None))
    if not any(candidate == path for candidate, _num in matches):
        matches.append((path, None))
    return matches


def _distinct_curve_colors(count: int) -> list[tuple[int, int, int]]:
    """Generate a color sequence with strong separation between neighbors."""
    if count <= 0:
        return []

    # Start from the existing blue hue, then distribute candidate colors
    # evenly around the hue wheel.  Greedily choosing the candidate farthest
    # from the previous one gives an alternating sequence around the wheel,
    # which keeps adjacent curves as distinct as possible.
    base_hue = 210.0 / 360.0
    candidate_hues = [(base_hue + index / count) % 1.0 for index in range(count)]
    remaining = set(range(1, count))
    order = [0]

    def hue_distance(first: float, second: float) -> float:
        distance = abs(first - second)
        return min(distance, 1.0 - distance)

    while remaining:
        previous_hue = candidate_hues[order[-1]]
        next_index = max(
            remaining,
            key=lambda index: hue_distance(previous_hue, candidate_hues[index]),
        )
        order.append(next_index)
        remaining.remove(next_index)

    return [
        tuple(
            round(channel * 255)
            for channel in colorsys.hsv_to_rgb(candidate_hues[index], 0.85, 0.9)
        )
        for index in order
    ]


def _read_fsc_curves(star_path: str | Path, *, use_ssnr_map: bool):
    """Read FSC curves from a model.star file, or ``None`` on any failure.

    Returns the same tuple format as :func:`_model_fsc_curves`, or ``None`` if
    the file cannot be read or contains no usable resolution/FSC columns.
    Used for sibling iteration files, where a missing/broken file should be
    skipped rather than shown as an error.

    Parameters
    ----------
    star_path : str or Path
        Path of the model.star file to read.
    use_ssnr_map : bool
        Convert ``rlnSsnrMap`` to the FSC-equivalent MAP weight (Class3D).

    Returns
    -------
    list or None
        Curves as returned by ``_model_fsc_curves``, or ``None`` on failure.
    """
    try:
        import starfile

        data = starfile.read(str(star_path), always_dict=True)
    except Exception:
        return None
    if not any(
        re.match(r"^(?:data_)?model_class_\d+$", str(key), re.IGNORECASE)
        for key in data
    ):
        return None
    curves = _model_fsc_curves(data, use_ssnr_map=use_ssnr_map)
    return curves or None


class _FscPlotWindow(_FscPlotBase):
    """FSC plot window with an iteration checkbox bar above the plot.

    One checkbox per refinement-iteration model.star file lets the user
    interactively toggle which iterations are overlaid, with "all" / "none"
    shortcut buttons appended to the end of the checkbox strip.  Each
    displayed curve receives a distinct color, with neighboring curves
    chosen to be as visually separated as possible; iterations also use
    distinct line styles.  Class3D jobs plot the SSNR-derived W_MAP curves,
    show one checkbox per class (all checked by default), and only plot the
    checked classes; Refine3D (and other) jobs plot the gold-standard FSC.

    Parameters
    ----------
    iterations : list of tuple(Path, int or None)
        ``(path, iteration_number)`` pairs as returned by
        ``_iteration_model_files``, restricted to files with readable curves.
    curves_by_path : dict
        Mapping ``Path -> list of curves`` in the format returned by
        ``_model_fsc_curves``.
    default_path : str or Path
        Iteration checked by default (the file selected in the browser).
    is_class3d : bool
        Whether the job is a Class3D refinement (W_MAP instead of FSC).
    parent : QWidget, optional
        Parent widget for the window.
    """

    def __init__(
        self,
        iterations: list[tuple[Path, int | None]],
        curves_by_path: dict,
        default_path: str | Path,
        is_class3d: bool,
        parent=None,
    ):
        super().__init__(parent)
        self.setProperty("helicon_theme_window", True)
        self.setStyleSheet(_display_theme_stylesheet())
        self.setPalette(_display_theme_palette())
        self._is_class3d = bool(is_class3d)
        self._paths: list[Path] = []
        self._curves_by_path: dict = {}
        self._checkboxes = []
        self._class_checkboxes = []
        self._curve_items = []
        self._plot = None
        self._legend = None
        self._threshold_line = None
        self._threshold_label = None
        self._crosshair_lines = []
        self._coord_text = None
        self._bar_layout = None
        self._class_bar_layout = None
        self._class_row = None
        self._build_ui()
        self.set_data(iterations, curves_by_path, default_path)
        self.resize(900, 600)

    def _build_ui(self) -> None:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QCheckBox,
            QFrame,
            QGridLayout,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QScrollArea,
            QVBoxLayout,
            QWidget,
        )

        import pyqtgraph as pg

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        # Control bar: one checkbox per iteration, plus "all" / "none"
        # shortcut buttons appended to the end of the checkbox strip.  The
        # checkboxes live in a wrapping, scrollable strip (set_data() adds
        # them) so jobs with hundreds of iterations never stretch the
        # window; a second row holds one checkbox per class (Class3D jobs
        # only).
        bar = QWidget()
        bar_layout = QVBoxLayout(bar)
        bar_layout.setContentsMargins(8, 4, 8, 4)
        bar_layout.setSpacing(2)

        iter_row = QWidget()
        self._bar_layout = QGridLayout(iter_row)
        self._bar_layout.setContentsMargins(0, 0, 0, 0)
        self._bar_layout.setHorizontalSpacing(8)
        self._bar_layout.setVerticalSpacing(0)
        sample_checkbox = QCheckBox("it000")
        row_height = sample_checkbox.sizeHint().height()
        # A little extra breathing room between neighbouring checkboxes:
        # roughly a quarter of one label character on top of the flow's
        # base spacing.
        label_space = max(
            1, round(sample_checkbox.fontMetrics().horizontalAdvance("0") * 0.25)
        )
        checkbox_spacing = 6 + label_space

        iter_label = QLabel("Iterations:")
        iter_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        iter_label.setFixedHeight(row_height)
        self._bar_layout.addWidget(iter_label, 0, 0, Qt.AlignmentFlag.AlignTop)

        iter_scroll = QScrollArea()
        iter_scroll.setWidgetResizable(True)
        iter_scroll.setFrameShape(QFrame.Shape.NoFrame)
        iter_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        iter_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._iter_flow = _FlowLayout(spacing=checkbox_spacing)
        iter_container = _FlowContainer(self._iter_flow)
        iter_container.setLayout(self._iter_flow)
        iter_scroll.setWidget(iter_container)
        self._iter_scroll = iter_scroll
        self._bar_layout.addWidget(iter_scroll, 0, 1, Qt.AlignmentFlag.AlignTop)
        self._bar_layout.setColumnStretch(1, 1)

        select_all_btn = QPushButton("all")
        button_height = max(row_height, select_all_btn.sizeHint().height())
        button_width = max(
            select_all_btn.sizeHint().width(),
            QPushButton("none").sizeHint().width(),
        )
        iter_scroll.setMaximumHeight(button_height * 2 + checkbox_spacing)
        select_all_btn.setFixedHeight(button_height)
        select_all_btn.setFixedWidth(button_width)
        select_all_btn.setStyleSheet("QPushButton { padding-bottom: 2px; }")
        select_all_btn.setToolTip("Plot every iteration")
        select_all_btn.clicked.connect(self._select_all)
        unselect_all_btn = QPushButton("none")
        unselect_all_btn.setFixedHeight(button_height)
        unselect_all_btn.setFixedWidth(button_width)
        unselect_all_btn.setStyleSheet("QPushButton { padding-bottom: 2px; }")
        unselect_all_btn.setToolTip("Plot no iterations")
        unselect_all_btn.clicked.connect(self._unselect_all)
        self._iter_select_all_btn = select_all_btn
        self._iter_unselect_all_btn = unselect_all_btn
        iter_button_group = QWidget()
        iter_button_layout = QHBoxLayout(iter_button_group)
        iter_button_layout.setContentsMargins(0, 0, 0, 0)
        iter_button_layout.setSpacing(6)
        iter_button_layout.addWidget(select_all_btn)
        iter_button_layout.addWidget(unselect_all_btn)
        self._iter_button_group = iter_button_group
        self._iter_flow.addWidget(iter_button_group)
        bar_layout.addWidget(iter_row)

        class_row = QWidget()
        self._class_bar_layout = QGridLayout(class_row)
        self._class_bar_layout.setContentsMargins(0, 0, 0, 0)
        self._class_bar_layout.setHorizontalSpacing(8)
        self._class_bar_layout.setVerticalSpacing(0)
        class_label = QLabel("Classes:")
        self._class_label = class_label
        class_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        class_label.setFixedHeight(row_height)
        self._class_bar_layout.addWidget(class_label, 0, 0, Qt.AlignmentFlag.AlignTop)
        class_scroll = QScrollArea()
        class_scroll.setWidgetResizable(True)
        class_scroll.setFrameShape(QFrame.Shape.NoFrame)
        class_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        class_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._class_flow = _FlowLayout(spacing=checkbox_spacing)
        class_container = _FlowContainer(self._class_flow)
        class_container.setLayout(self._class_flow)
        class_scroll.setWidget(class_container)
        class_scroll.setMaximumHeight(row_height * 2 + 10)
        self._class_scroll = class_scroll
        self._class_bar_layout.addWidget(class_scroll, 0, 1, Qt.AlignmentFlag.AlignTop)
        self._class_bar_layout.setColumnStretch(1, 1)
        class_select_all_btn = QPushButton("all")
        class_select_all_btn.setFixedHeight(button_height)
        class_select_all_btn.setFixedWidth(button_width)
        class_select_all_btn.setStyleSheet("QPushButton { padding-bottom: 2px; }")
        class_select_all_btn.setToolTip("Check every class")
        class_select_all_btn.clicked.connect(self._select_all_classes)
        class_unselect_all_btn = QPushButton("none")
        class_unselect_all_btn.setFixedHeight(button_height)
        class_unselect_all_btn.setFixedWidth(button_width)
        class_unselect_all_btn.setStyleSheet("QPushButton { padding-bottom: 2px; }")
        class_unselect_all_btn.setToolTip("Uncheck every class")
        class_unselect_all_btn.clicked.connect(self._unselect_all_classes)
        class_button_group = QWidget()
        class_button_layout = QHBoxLayout(class_button_group)
        class_button_layout.setContentsMargins(0, 0, 0, 0)
        class_button_layout.setSpacing(6)
        class_button_layout.addWidget(class_select_all_btn)
        class_button_layout.addWidget(class_unselect_all_btn)
        self._class_select_all_btn = class_select_all_btn
        self._class_unselect_all_btn = class_unselect_all_btn
        self._class_button_group = class_button_group
        class_select_all_btn.setVisible(False)
        class_unselect_all_btn.setVisible(False)
        class_button_group.setVisible(False)
        class_row.setHidden(True)
        self._class_row = class_row
        bar_layout.addWidget(class_row)

        layout.addWidget(bar)

        pg.setConfigOptions(antialias=True)
        plot_widget = pg.PlotWidget()
        layout.addWidget(plot_widget)
        self.plot_widget = plot_widget

        plot = plot_widget.getPlotItem()
        self._plot = plot
        plot.getAxis("bottom").enableAutoSIPrefix(False)
        plot.setLabel("bottom", "Resolution", units="1/Å")
        curve_label = "W_MAP" if self._is_class3d else "FSC"
        plot.setLabel("left", curve_label)
        plot.setYRange(0, 1.05)
        self._legend = plot.addLegend()
        plot.showGrid(x=True, y=True, alpha=0.3)

        top_axis = plot.getAxis("top")
        top_axis.enableAutoSIPrefix(False)
        top_axis.setLabel("Resolution", units="Å")

        def _angstrom_tickStrings(values, scale, spacing):
            return [f"{1.0 / v:.1f}" if v > 0 else "" for v in values]

        top_axis.tickStrings = _angstrom_tickStrings
        top_axis.show()

        threshold_pen = pg.mkPen(
            color=(220, 50, 50), width=1, style=Qt.PenStyle.DashLine
        )
        self._threshold_line = pg.InfiniteLine(pos=0.143, angle=0, pen=threshold_pen)
        plot.addItem(self._threshold_line)
        threshold_label = pg.TextItem("0.143", color=(220, 50, 50), anchor=(0, 0))
        self._threshold_label = threshold_label
        threshold_label.setZValue(5)
        plot.addItem(threshold_label, ignoreBounds=True)

        def _position_threshold_label():
            vr = plot.vb.viewRect()
            threshold_label.setPos(vr.left() + vr.width() * 0.005, 0.143)

        _position_threshold_label()
        plot.vb.sigResized.connect(_position_threshold_label)

        crosshair_pen = pg.mkPen(
            color=(150, 150, 150, 160), width=1, style=Qt.PenStyle.DashLine
        )
        vline = pg.InfiniteLine(angle=90, pen=crosshair_pen, movable=False)
        hline = pg.InfiniteLine(angle=0, pen=crosshair_pen, movable=False)
        self._crosshair_lines = [vline, hline]
        plot.addItem(vline, ignoreBounds=True)
        plot.addItem(hline, ignoreBounds=True)
        coord_text = pg.TextItem(anchor=(0, 1), color=(220, 220, 220))
        self._coord_text = coord_text
        coord_text.setZValue(20)
        plot.addItem(coord_text, ignoreBounds=True)

        def _on_mouse_moved(evt):
            pos = plot.vb.mapSceneToView(evt if hasattr(evt, "x") else evt[0])
            x, y = pos.x(), pos.y()
            if not plot.vb.viewRect().contains(pos):
                coord_text.setVisible(False)
                return
            vline.setValue(x)
            hline.setValue(y)
            vline.setVisible(True)
            hline.setVisible(True)
            ang = f"{1.0 / x:.2f}" if x > 0 else "∞"
            coord_text.setHtml(
                f'<span style="color: {self._plot_theme_colors["tooltip_foreground"]};'
                f' background-color: {self._plot_theme_colors["tooltip_background"]};'
                f' padding: 2px;">'
                f"{x:.4f} 1/Å  ({ang} Å)<br>{curve_label} = {y:.4f}</span>"
            )
            coord_text.setVisible(True)
            vr = plot.vb.viewRect()
            dx = vr.width() * 0.02
            dy = vr.height() * 0.02
            tx = x + dx
            ty = y + dy
            if tx + vr.width() * 0.15 > vr.right():
                tx = x - vr.width() * 0.15 - dx
            coord_text.setPos(tx, ty)

        plot_widget.scene().sigMouseMoved.connect(_on_mouse_moved)

        self.setCentralWidget(central)
        self._apply_display_theme()

    def _apply_display_theme(self) -> None:
        """Apply the saved theme to the pyqtgraph canvas and decorations."""
        if self._plot is None:
            return

        from PySide6.QtCore import Qt

        import pyqtgraph as pg

        colors = _display_plot_theme_colors()
        self._plot_theme_colors = colors
        self.plot_widget.setBackground(colors["background"])

        for axis_name in ("left", "bottom", "top", "right"):
            axis = self._plot.getAxis(axis_name)
            axis.setPen(colors["foreground"])
            axis.setTextPen(colors["foreground"])
            axis.setTickPen(colors["foreground"])

        curve_label = "W_MAP" if self._is_class3d else "FSC"
        self._plot.setLabel(
            "bottom", "Resolution", units="1/Å", color=colors["foreground"]
        )
        self._plot.setLabel("left", curve_label, color=colors["foreground"])
        self._plot.getAxis("top").setLabel(
            "Resolution", units="Å", color=colors["foreground"]
        )
        self._plot.showGrid(x=True, y=True, alpha=0.3)

        if self._threshold_line is not None:
            self._threshold_line.setPen(
                pg.mkPen(color=(220, 50, 50), width=1, style=Qt.PenStyle.DashLine)
            )
        if self._threshold_label is not None:
            self._threshold_label.setColor((220, 50, 50))
        for line in self._crosshair_lines:
            line.setPen(
                pg.mkPen(
                    color=colors["crosshair"],
                    width=1,
                    style=Qt.PenStyle.DashLine,
                )
            )
        if self._coord_text is not None:
            self._coord_text.setColor(colors["tooltip_foreground"])

        if self._legend is not None:
            for _sample, label in self._legend.items:
                label.setText(label.text, color=colors["foreground"])

    def set_data(
        self,
        iterations: list[tuple[Path, int | None]],
        curves_by_path: dict,
        default_path: str | Path,
        is_class3d: bool | None = None,
    ) -> None:
        """Replace the iteration set and replot, keeping the same window."""
        from PySide6.QtWidgets import QCheckBox

        if is_class3d is not None:
            self._is_class3d = bool(is_class3d)
            curve_label = "W_MAP" if self._is_class3d else "FSC"
            self._plot.setLabel("left", curve_label)

        # Keep the "all"/"none" buttons at the end of the checkbox strip;
        # they are re-appended after the checkboxes below.
        self._iter_flow.removeWidget(self._iter_button_group)

        for checkbox in self._checkboxes:
            try:
                checkbox.toggled.disconnect(self._rebuild_curves)
            except RuntimeError:
                pass
            self._iter_flow.removeWidget(checkbox)
            checkbox.deleteLater()
        self._checkboxes.clear()

        self._paths = [path for path, _num in iterations]
        self._curves_by_path = dict(curves_by_path)

        for path, _num in iterations:
            checkbox = QCheckBox(_iteration_label(path))
            checkbox.setChecked(False)
            self._iter_flow.addWidget(checkbox)
            self._checkboxes.append(checkbox)

        default = Path(default_path)
        default_index = 0
        for index, path in enumerate(self._paths):
            if path == default:
                default_index = index
                break
        if self._checkboxes:
            self._checkboxes[default_index].setChecked(True)

        for checkbox in self._checkboxes:
            checkbox.toggled.connect(self._rebuild_curves)

        self._iter_flow.addWidget(self._iter_button_group)

        self._rebuild_class_checkboxes()
        self._rebuild_curves()

    def _rebuild_class_checkboxes(self) -> None:
        """Replace the class checkboxes, keeping the same window."""
        from PySide6.QtWidgets import QCheckBox

        # The "all"/"none" buttons trail the class checkboxes while the
        # strip is shown; drop them so the rebuilt checkboxes come first.
        self._class_flow.removeWidget(self._class_button_group)

        for checkbox in self._class_checkboxes:
            try:
                checkbox.toggled.disconnect(self._rebuild_curves)
            except RuntimeError:
                pass
            self._class_flow.removeWidget(checkbox)
            checkbox.deleteLater()
        self._class_checkboxes.clear()

        class_count = max(
            (len(curves) for curves in self._curves_by_path.values()), default=0
        )
        show_classes = self._is_class3d and class_count > 1
        self._class_row.setHidden(not show_classes)
        show_class_buttons = show_classes and class_count > 3
        self._class_select_all_btn.setVisible(show_class_buttons)
        self._class_unselect_all_btn.setVisible(show_class_buttons)
        self._class_button_group.setVisible(show_class_buttons)
        if not show_classes:
            return

        for index in range(class_count):
            checkbox = QCheckBox(str(index + 1))
            checkbox.setChecked(True)
            self._class_flow.addWidget(checkbox)
            self._class_checkboxes.append(checkbox)

        for checkbox in self._class_checkboxes:
            checkbox.toggled.connect(self._rebuild_curves)

        if show_class_buttons:
            self._class_flow.addWidget(self._class_button_group)

    def _checked_paths(self) -> list[Path]:
        return [
            path
            for path, checkbox in zip(self._paths, self._checkboxes)
            if checkbox.isChecked()
        ]

    def _select_all(self) -> None:
        if not self._checkboxes:
            return
        for checkbox in self._checkboxes:
            checkbox.blockSignals(True)
            checkbox.setChecked(True)
        for checkbox in self._checkboxes:
            checkbox.blockSignals(False)
        self._rebuild_curves()

    def _unselect_all(self) -> None:
        if not self._checkboxes:
            return
        for checkbox in self._checkboxes:
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
        for checkbox in self._checkboxes:
            checkbox.blockSignals(False)
        self._rebuild_curves()

    def _select_all_classes(self) -> None:
        if not self._class_checkboxes:
            return
        for checkbox in self._class_checkboxes:
            checkbox.blockSignals(True)
            checkbox.setChecked(True)
        for checkbox in self._class_checkboxes:
            checkbox.blockSignals(False)
        self._rebuild_curves()

    def _unselect_all_classes(self) -> None:
        if not self._class_checkboxes:
            return
        for checkbox in self._class_checkboxes:
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
        for checkbox in self._class_checkboxes:
            checkbox.blockSignals(False)
        self._rebuild_curves()

    def _checked_class_indices(self) -> list[int] | None:
        """Indices of the checked classes, or None when class filtering is off."""
        if not self._class_checkboxes:
            return None
        return [
            index
            for index, checkbox in enumerate(self._class_checkboxes)
            if checkbox.isChecked()
        ]

    def _rebuild_curves(self) -> None:
        """Clear and re-draw the FSC curves for the checked iterations."""
        from PySide6.QtCore import Qt

        import pyqtgraph as pg

        plot = self._plot
        for item in self._curve_items:
            plot.removeItem(item)
        self._curve_items.clear()

        selected = self._checked_paths()
        checked_classes = self._checked_class_indices()
        curves_by_path = {
            path: (curves if self._is_class3d else curves[:1])
            for path, curves in self._curves_by_path.items()
        }
        curve_count = sum(
            sum(
                checked_classes is None or class_index in checked_classes
                for class_index in range(len(curves_by_path[path]))
            )
            for path in selected
        )
        colors = _distinct_curve_colors(curve_count)
        line_styles = [
            Qt.PenStyle.SolidLine,
            Qt.PenStyle.DashLine,
            Qt.PenStyle.DotLine,
            Qt.PenStyle.DashDotLine,
            Qt.PenStyle.DashDotDotLine,
        ]
        multi = len(self._paths) > 1
        color_index = 0
        for it_index, path in enumerate(self._paths):
            if path not in selected:
                continue
            curves = curves_by_path[path]
            style = line_styles[it_index % len(line_styles)]
            it_label = _iteration_label(path)
            for class_index, (spatial_freq, fsc, label, _angstrom) in enumerate(curves):
                if checked_classes is not None and class_index not in checked_classes:
                    continue
                color = colors[color_index]
                color_index += 1
                pen = pg.mkPen(color=color, width=2, style=style)
                if not self._is_class3d:
                    curve_name = it_label
                else:
                    curve_name = f"{it_label} · {label}" if multi else label
                self._curve_items.append(
                    plot.plot(spatial_freq, fsc, pen=pen, name=curve_name)
                )
        self._apply_display_theme()

    def closeEvent(self, event):
        _plot.on_close(self)
        super().closeEvent(event)

    def changeEvent(self, event):
        from PySide6.QtCore import QEvent

        if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
            _plot.on_activate(self)
        super().changeEvent(event)


def _open_fsc_plot(star_path: str, reuse_window=None) -> None:
    """Display the FSC curves from RELION model.star files using pyqtgraph.

    Reads ``data_model_class_N`` sections and plots resolution vs. FSC for
    each class.  Class3D jobs use the FSC-equivalent MAP weight calculated
    from ``rlnSsnrMap``; other jobs use the gold-standard FSC column.  A
    horizontal line at 0.143 marks the FSC threshold.

    When the job directory contains more than one ``*_model.star`` file (one
    per refinement iteration), a checkbox bar at the top of the plot lets the
    user interactively choose which iterations to overlay, with the file
    selected in the browser checked by default and "all" / "none" shortcut
    buttons at the end of the checkbox strip.  Class3D jobs get an extra row
    of checkboxes, one per class (all checked by default), that filter which
    classes are drawn.
    """
    try:
        import pyqtgraph  # noqa: F401 - availability probe
        from PySide6.QtWidgets import QMessageBox
    except ImportError:
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.warning(
            None,
            "Missing dependency",
            "pyqtgraph is required to display FSC curves.\n"
            "Install it with: pip install pyqtgraph",
        )
        return

    try:
        import starfile

        data = starfile.read(star_path, always_dict=True)
    except Exception:
        QMessageBox.warning(
            None,
            "Error",
            f"Failed to read {Path(star_path).name}.\n"
            "Make sure it is a valid RELION model.star file.",
        )
        return

    if not any(
        re.match(r"^(?:data_)?model_class_\d+$", str(key), re.IGNORECASE)
        for key in data
    ):
        QMessageBox.information(
            None,
            "No FSC data",
            f"No data_model_class_N sections found in {Path(star_path).name}.",
        )
        return

    is_class3d = any(
        part.lower().startswith("class3d") for part in Path(star_path).parts
    )
    all_curves = _model_fsc_curves(data, use_ssnr_map=is_class3d)

    if not all_curves:
        curve_column = "rlnSsnrMap" if is_class3d else "FSC"
        QMessageBox.warning(
            None,
            "Column error",
            f"Could not find resolution/{curve_column} columns in any class section.",
        )
        return

    # Read curves for every iteration file in the job directory.  The file
    # selected in the browser is already parsed; sibling files that cannot be
    # read or have no usable FSC columns are skipped silently.
    iterations = _iteration_model_files(star_path)
    primary = Path(star_path)
    curves_by_path = {}
    for path, _num in iterations:
        curves = (
            all_curves
            if path == primary
            else _read_fsc_curves(str(path), use_ssnr_map=is_class3d)
        )
        if curves:
            curves_by_path[path] = curves

    plot_iterations = [
        (path, num) for path, num in iterations if path in curves_by_path
    ]
    if not plot_iterations:
        QMessageBox.warning(
            None,
            "Column error",
            "Could not find resolution/FSC columns in any selected file.",
        )
        return

    name = Path(star_path).name
    window_title = (
        name
        if len(plot_iterations) == 1
        else f"{name} ({len(plot_iterations)} iterations)"
    )

    if (
        reuse_window is not None
        and _is_alive_widget(reuse_window)
        and isinstance(reuse_window, _FscPlotWindow)
    ):
        reuse_window.set_data(
            plot_iterations,
            curves_by_path,
            default_path=primary,
            is_class3d=is_class3d,
        )
        reuse_window.setWindowTitle(f"FSC — {window_title}")
        reuse_window.show()
        reuse_window.raise_()
        return

    win = _FscPlotWindow(
        plot_iterations,
        curves_by_path,
        default_path=primary,
        is_class3d=is_class3d,
    )
    win.setWindowTitle(f"FSC — {window_title}")
    _plot.register(win)
    _install_window_shortcuts(win)
    win.show()


def _open_xyz_slice_gallery(
    star_path: str, reuse_window=None, tracker=None
) -> "QMainWindow | None":
    """Display center slices (Z, Y, X) of MRC files referenced by a star file.

    Delegates to ``Class3dGallery`` (with abundance labels) or
    ``Refine3dGallery`` (without) based on the path.
    """
    from helicon.lib.gallery_backends import Class3dGallery, Refine3dGallery

    name = Path(star_path).name
    is_refine = any(p.startswith("Refine3D") for p in Path(star_path).parts)
    gallery = Refine3dGallery(star_path) if is_refine else Class3dGallery(star_path)
    return gallery.open(reuse_window=reuse_window, tracker=tracker)


def _open_orthogonal_viewer(
    mrc_path: str, reuse_window=None, tracker=None
) -> "QMainWindow | None":
    """Open an interactive orthogonal slice viewer for a 3D MRC/MAP file."""
    from helicon.lib.gallery_backends import OrthogonalGallery

    gallery = OrthogonalGallery(mrc_path)
    return gallery.open(reuse_window=reuse_window, tracker=tracker)


def _find_model_star_from_optimiser(optimiser_path: str) -> str | None:
    """Find the model.star referenced by an optimiser.star file."""
    from pathlib import Path

    star_dir = Path(optimiser_path).parent

    try:
        with open(optimiser_path) as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if s.startswith("_rlnModelStarFile"):
                    parts = s.split()
                    if len(parts) >= 2:
                        model_rel = parts[-1]
                        for ancestor in [star_dir] + list(star_dir.parents):
                            candidate = ancestor / model_rel
                            if candidate.is_file():
                                return str(candidate)
    except Exception:
        pass
    return None


def _open_2d_classes_gallery(
    star_path: str, reuse_window=None, tracker=None
) -> "QMainWindow | None":
    """Display 2D class averages from a Class2D model.star.

    Shows one MRC per class (``_rlnReferenceImage``) with abundance labels
    (``_rlnClassDistribution``). Sort-by-abundance and reverse-sort controls
    are provided in the control panel.
    """
    from helicon.lib.gallery_backends import Class2dGallery

    gallery = Class2dGallery(star_path)
    return gallery.open(reuse_window=reuse_window, tracker=tracker)


def _open_frame_in_slice_view(viewer, read_fn, idx, img_w, img_h, apix, name) -> None:
    """Open a single gallery frame in the normal 2D slice view.

    Reuses the same ``add_image`` call used by the slice branch so behaviour
    is identical: the clicked frame is shown and the camera fits it. Any
    prior gallery/slice layer is removed first to avoid accumulation.

    Parameters
    ----------
    viewer : napari.Viewer
        Target viewer.
    read_fn : callable
        ``read_fn(i) -> numpy.ndarray`` returning frame ``i``.
    idx : int
        Global frame index to display.
    img_w : int
        Native frame width.
    img_h : int
        Native frame height.
    apix : float
        Pixel size (A/px).
    name : str
        Display name.
    """
    try:
        for layer in list(viewer.layers):
            viewer.layers.remove(layer)
    except (AttributeError, RuntimeError):
        pass

    frame = read_fn(idx)
    contrast = _auto_contrast(frame)
    layer = viewer.add_image(
        frame,
        name=name,
        scale=(apix, apix),
        contrast_limits=contrast,
        interpolation2d="linear",
        interpolation3d="linear",
    )
    _enable_continuous_auto_contrast(layer, viewer)
    _SliceDirectionWidget.set_stack_mode(True)
    viewer.dims.ndisplay = 2
    _reset_view(viewer)
    # Reveal the window so a freshly-created (hidden) viewer actually shows
    # the clicked frame; a no-op when the window is already visible.
    try:
        viewer.window._qt_window.show()
        viewer.window._qt_window.raise_()
    except Exception:
        pass


def _open_file(viewer, path: str, mode: str | None = None, reuse_gallery=None) -> None:
    from pathlib import Path

    if viewer is not None:
        # Reset the viewer to the mode expected for THIS file before any
        # layer is added. Otherwise a stale ndisplay (e.g. a 3D volume
        # view left over from the previous file) persists and produces a
        # wrong/empty view for the incoming file. Drop the previous
        # layers so reset_view() below fits only the new file.
        try:
            old_layers = list(viewer.layers)
        except (TypeError, AttributeError):
            old_layers = []
        for layer in old_layers:
            try:
                viewer.layers.remove(layer)
            except Exception:
                pass
        # Reset to the volume default; specific branches below set True when the
        # incoming layer is a true image stack (axis 0 is a frame index, not a
        # spatial axis) so the Z/Y/X axis selector is hidden for those files.
        _SliceDirectionWidget.set_stack_mode(False)
        if mode == "volume":
            _set_ndisplay(viewer, 3)
        else:
            # 2D slice / image stack / general / text / pdf / eps -> 2D.
            _set_ndisplay(viewer, 2)
    else:
        _SliceDirectionWidget.set_stack_mode(False)

    if viewer is not None:
        viewer.title = f"Helicon - {Path(path).name}"
        # Remember the opened file name so the Save-As dialog can pre-fill it
        # (mirrors ImageGalleryWidget._source_name).  napari only populates
        # layer.source.path when a file is loaded via viewer.open(); most
        # helicon openers add layers directly, so we record it here instead.
        viewer._source_name = Path(path).name

    # When a display mode is forced (button bar), the *.mrc / *.map volume
    # branch must show as a 3D volume (ndisplay=3) or a 2D slice stack
    # (ndisplay=2) instead of the auto-chosen default.
    force_ndisplay = None
    if mode == "volume":
        force_ndisplay = 3
    elif mode == "slice":
        force_ndisplay = 2

    ext = Path(path).suffix.lower()

    if ext == ".star" and mode == "optimiser":
        _open_xyz_slice_gallery(path, reuse_window=reuse_gallery, tracker=_gallery)
        return

    if ext == ".star" and mode == "2dclasses":
        _open_2d_classes_gallery(path, reuse_window=reuse_gallery, tracker=_gallery)
        return

    # "text" mode: always open any star file as text, regardless of type.
    if ext == ".star" and mode == "text":
        _open_text_window(path, reuse_window=reuse_gallery)
        return

    if ext == ".star" and _is_metadata_star(path):
        _open_text_window(path, reuse_window=reuse_gallery)
        return

    if ext == ".star":
        import mrcfile

        result = _parse_star_image_refs(path)
        if result is None:
            return
        entries, first_shape, first_apix, n_skipped = result

        if n_skipped:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(
                None,
                "Missing images",
                f"{n_skipped} image(s) referenced in {Path(path).name} "
                "could not be found on disk and were skipped.",
            )
        if not entries:
            return

        n = len(entries)
        with mrcfile.open(entries[0][1], permissive=True) as mrc:
            frame = mrc.data[entries[0][0]] if mrc.data.ndim >= 3 else mrc.data
            dtype = frame.dtype
            # Contrast limits from the first frame only. Passing explicit
            # contrast_limits stops napari from scanning the entire stack to
            # compute a global range (which would force every referenced
            # image to load). Continuous auto-contrast below keeps each
            # navigated slice contrasted independently, so only the visible
            # frame is read -- the lazy behaviour we want.
            contrast = _auto_contrast(frame)

        stack_shape = (n,) + first_shape
        lazy = _LazyStarStack(entries, stack_shape, dtype)

        name = Path(path).name
        if mode == "gallery":
            _open_gallery(
                read_fn=lazy.__getitem__,
                n=n,
                img_w=first_shape[0],
                img_h=first_shape[1],
                apix=first_apix,
                name=name,
                reuse_window=reuse_gallery,
                tracker=_gallery,
            )
            return

        # This is a true image stack (frame index axis 0), not a 3D volume:
        # hide the Z/Y/X axis selector.
        _SliceDirectionWidget.set_stack_mode(True)
        layer = viewer.add_image(
            lazy,
            name=name,
            scale=(1.0,) + (first_apix,) * len(first_shape),
            contrast_limits=contrast,
            interpolation2d="linear",
            interpolation3d="linear",
        )
        _enable_continuous_auto_contrast(layer, viewer)
        viewer.dims.ndisplay = 2
        # napari defaults the shown slice to the middle frame; start at frame 0.
        step = list(viewer.dims.current_step)
        step[0] = 0
        viewer.dims.current_step = step
        _reset_view(viewer)
        return

    if ext == ".bild":
        _open_bild(viewer, path)
        return

    if ext == ".pdf":
        _open_pdf(viewer, path)
        return

    if ext == ".eps":
        _open_eps(viewer, path)
        return

    if ext in (".html", ".htm"):
        _open_html(viewer, path)
        return

    if _is_text_file(path):
        _open_text_window(path, reuse_window=reuse_gallery)
        return

    if ext in _MRC_EXTENSIONS:
        import struct
        import numpy as np
        import dask.array as da
        from dask import delayed

        # Parse the MRC header directly from raw bytes so we never open
        # the file through mrcfile (which mmap's the *data* section and
        # causes the OS to page the entire file into RSS).
        _MODE_DTYPE = {
            0: np.int8,
            1: np.int16,
            2: np.float32,
            3: np.complex64,
            4: np.int16,
            6: np.uint16,
            12: np.float16,
            13: np.uint8,
            14: np.int32,
            15: np.float64,
        }
        try:
            with open(path, "rb") as _fh:
                _raw_hdr = _fh.read(1024)
            _nx, _ny, _nz = struct.unpack_from("<3i", _raw_hdr, 0)
            _mode = struct.unpack_from("<i", _raw_hdr, 12)[0]
            _nsymbt = struct.unpack_from("<i", _raw_hdr, 92)[0]
            _header_offset = 1024 + _nsymbt
            # cella (floats at offsets 40, 44, 48) / dimensions → pixel size
            _cella_x = struct.unpack_from("<f", _raw_hdr, 40)[0]
            _apix = float(_cella_x) / _nx if _nx else 1.0
        except Exception:
            _nx = _ny = _nz = 0
            _mode = 2
            _apix = 1.0
            _header_offset = 1024
        if _apix <= 0:
            try:
                import mrcfile as _mrcfile_fallback

                with _mrcfile_fallback.open(path, permissive=True) as _m:
                    _apix = float(_m.voxel_size.x)
            except Exception:
                _apix = 1.0
        if _apix <= 0:
            _apix = 1.0

        _dtype = _MODE_DTYPE.get(_mode, np.float32)

        _data_mmap = np.memmap(
            path, dtype=_dtype, mode="r", offset=_header_offset, shape=(_nz, _ny, _nx)
        )

        def _read_raw(i):
            return np.array(_data_mmap[i])

        name = Path(path).name

        # Build a lazily-read 3D array (axis 0 = Z / frame). Planes are read
        # from disk individually, so a 2D view never loads the whole file.
        if _nz >= 1 and _nx > 0 and _ny > 0:
            _vol = da.stack(
                [
                    da.from_delayed(
                        delayed(_read_raw)(i), shape=(_ny, _nx), dtype=_dtype
                    )
                    for i in range(_nz)
                ]
            )
        else:
            _vol = None

        # Gallery: read one plane at a time (lazy) — no full-volume load.
        # Must be checked before the single-plane branch so that a 1-image
        # .mrcs opened via the gallery button still reaches the gallery.
        if mode == "gallery":
            n = int(_vol.shape[0])
            img_h, img_w = int(_vol.shape[1]), int(_vol.shape[2])

            def _gal_read(i):
                return np.asarray(_vol[i])

            _open_gallery(
                read_fn=_gal_read,
                n=n,
                img_w=img_w,
                img_h=img_h,
                apix=_apix,
                name=name,
                reuse_window=reuse_gallery,
                tracker=_gallery,
            )
            return

        # Single 2D plane: one eager read is fine and keeps the 2D path simple.
        if _vol is None or (_vol.ndim == 3 and _vol.shape[0] == 1):
            if viewer is None:
                return
            _single = _read_raw(0) if _vol is not None else None
            if _single is None:
                with mrcfile.open(path, permissive=True) as m:
                    _single = np.asarray(m.data)
            if _single.ndim >= 3:
                _single = _single[0]
            _contrast = _auto_contrast(_single)
            _SliceDirectionWidget.set_stack_mode(False)
            layer = viewer.add_image(
                _single,
                name=name,
                scale=(_apix, _apix),
                contrast_limits=_contrast,
                interpolation2d="linear",
                interpolation3d="linear",
            )
            _enable_continuous_auto_contrast(layer, viewer)
            _lo, _hi = float(_single.min()), float(_single.max())
            if _lo >= _hi:
                _hi = _lo + 1.0
            layer.contrast_limits_range = (_lo, _hi)
            _reset_view(viewer)
            return

        if mode == "orthogonal" and _nz > 1:
            _open_orthogonal_viewer(path, reuse_window=reuse_gallery, tracker=_gallery)
            return

        # Slice / volume view: show the lazy 3D array; napari loads one plane
        # at a time as the user scrolls.
        _SliceDirectionWidget.set_stack_mode(ext == ".mrcs")
        _mid = int(_vol.shape[0]) // 2
        _sample = np.asarray(_vol[_mid])
        contrast = _auto_contrast(_sample)
        layer = viewer.add_image(
            _vol,
            name=name,
            scale=(_apix, _apix, _apix),
            contrast_limits=contrast,
            interpolation2d="linear",
            interpolation3d="linear",
        )
        _enable_continuous_auto_contrast(layer, viewer)
        _lo, _hi = float(_sample.min()), float(_sample.max())
        if _lo >= _hi:
            _hi = _lo + 1.0
        layer.contrast_limits_range = (_lo, _hi)

        if len(viewer.dims.current_step) > 0:
            step = list(viewer.dims.current_step)
            if ext == ".mrcs":
                step[0] = 0
                viewer.dims.current_step = tuple(step)
                viewer.dims.ndisplay = 2
            else:
                step[0] = _mid
                viewer.dims.current_step = tuple(step)
                if force_ndisplay is not None:
                    viewer.dims.ndisplay = force_ndisplay
                else:
                    dims = _vol.shape
                    viewer.dims.ndisplay = (
                        3 if dims[0] * dims[1] * dims[2] < 512**3 else 2
                    )
        _reset_view(viewer)
        return
    else:
        n_before = len(viewer.layers)
        viewer.open(path)
        for layer in viewer.layers[n_before:]:
            _enable_continuous_auto_contrast(layer, viewer)
        _reset_view(viewer)


def _run_standalone() -> None:
    """Open a single file in a fresh napari viewer (separate process).

    Used by the "New display window" feature. Running the second viewer in
    its own process gives it a clean OpenGL context and avoids the VisPy/
    Qt-Wayland segfault that occurs when a second napari canvas is created
    inside an already-running viewer's event loop (fine on macOS, fatal
    under WSLg Wayland).

    Expects ``sys.argv[1]`` = file path and optional ``sys.argv[2]`` = mode.
    """
    _prepare_process_identity()
    _force_x11_platform_under_wslg()
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(_qt_argv())
    _set_application_identity(app)
    napari = _load_napari()

    path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None

    viewer = _create_napari_viewer(title=f"Helicon - {Path(path).name}")
    _install_panel_toggle(viewer)
    try:
        _open_file(viewer, path, mode=mode)
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[helicon] failed to display {path}: {exc}")
    napari.run()


def main(args: argparse.Namespace) -> None:
    """Launch napari with a floating folder browser.

    Opens a napari viewer window alongside a floating folder browser
    window. Double-click any image or volume file in the browser to
    display it in napari. Window positions and sizes are remembered
    between launches.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    """
    if FolderBrowserWidget is None:
        raise HeliconDependencyError(
            "Qt widgets are required for the display command. "
            "Install PySide6: pip install PySide6"
        )

    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import QWidget, QApplication
    from PySide6.QtGui import QShortcut, QKeySequence

    _prepare_process_identity()
    _force_x11_platform_under_wslg()
    app = QApplication.instance() or QApplication(_qt_argv())
    _set_application_identity(app)

    start_dir = args.folder if args.folder else os.getcwd()

    if app is not None:
        _icon_path = Path(__file__).parent.parent / "resources" / "icon.png"
        if _icon_path.is_file():
            from PySide6.QtGui import QIcon

            app.setWindowIcon(QIcon(str(_icon_path)))

    def _track_viewer(v):
        from PySide6.QtCore import QEvent, QObject

        class _CloseFilter(QObject):
            def __init__(self, viewer, parent=None):
                super().__init__(parent)
                self._viewer = viewer

            def eventFilter(self, obj, event):
                if event.type() == QEvent.Close:
                    _napari.on_close(self._viewer)
                elif event.type() == QEvent.ActivationChange and obj.isActiveWindow():
                    _napari.on_activate(self._viewer)
                return False

        try:
            qt_window = v.window._qt_window
            if qt_window is not None and isinstance(qt_window, QObject):
                qt_window.destroyed.connect(lambda *_: _napari.on_close(v))
                flt = _CloseFilter(v, parent=qt_window)
                qt_window.installEventFilter(flt)
        except Exception:
            pass

    def _ensure_napari_viewer():
        """Return the active napari viewer, creating one on first use."""
        from unittest.mock import MagicMock

        v = _napari.active()
        if v is not None:
            return v
        new_viewer = _create_napari_viewer()
        _track_viewer(new_viewer)
        _add_welcome_shortcut(new_viewer)
        _install_panel_toggle(new_viewer)
        try:
            if not isinstance(new_viewer.window, MagicMock):
                _install_viewer_save_menu(new_viewer)
        except Exception:
            pass
        _install_viewer_save_hook(widget, new_viewer)
        return new_viewer

    def _show_napari_viewer():
        """Ensure a napari viewer exists and reveal it."""
        v = _ensure_napari_viewer()
        try:
            v.window._qt_window.show()
            v.window._qt_window.raise_()
        except Exception:
            pass

    def _spawn_viewer_and_open(path, mode=None):
        """Open ``path`` in a new napari viewer.

        On macOS multiple in-process napari viewers coexist safely, so the
        new window is opened in-process and tracked by the ``_napari``
        tracker.  On every other platform (notably Linux/Wayland), creating
        a second in-process napari viewer segfaults, so the new window is
        spawned in a separate process.
        """
        if sys.platform == "darwin":
            try:
                new_viewer = _create_napari_viewer(title=f"Helicon - {Path(path).name}")
            except Exception as exc:  # pragma: no cover - environment dependent
                print(f"[helicon] failed to open new display window: {exc}")
                return
            _track_viewer(new_viewer)
            try:
                if not isinstance(new_viewer.window, MagicMock):
                    _install_viewer_save_menu(new_viewer)
            except Exception:
                pass
            try:
                _open_file(new_viewer, path, mode=mode)
            except Exception as exc:  # pragma: no cover - environment dependent
                print(f"[helicon] failed to display {path}: {exc}")
            return

        import subprocess

        try:
            subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "from helicon.commands.display import _run_standalone; "
                    "_run_standalone()",
                    path,
                    mode or "",
                ],
                start_new_session=True,
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"[helicon] failed to open new display window: {exc}")

    def _categorize_file(path):
        """Return (tracker, primary_mode) for a file based on its type."""
        modes = widget._display_modes_for(path)
        if not modes:
            return _napari, "slice"
        # For optimiser/model.star files, prefer gallery over text
        if modes[0] == "text":
            for m in modes:
                if m in _GALLERY_MODES:
                    return _TRACKER_FOR[m], m
        return _TRACKER_FOR.get(modes[0], _napari), modes[0]

    def _on_file_selected(path):
        tracker, mode = _categorize_file(path)
        if tracker is _napari:
            _show_napari_viewer()
            _open_file(_napari.active(), path)
        elif tracker is _gallery:
            reuse = tracker.active()
            _open_file(None, path, mode=mode, reuse_gallery=reuse)
        elif tracker is _text:
            _open_text_window(path, reuse_window=tracker.active())
        elif tracker is _plot:
            _open_fsc_plot(path, reuse_window=tracker.active())
        elif tracker is _images2star:
            _open_images2star_tools(
                path,
                parent=widget,
                reuse_window=tracker.active(),
                tracker=tracker,
            )

    def _on_file_selected_new_window(path):
        tracker, mode = _categorize_file(path)
        if tracker is _napari:
            _spawn_viewer_and_open(path)
        else:
            _on_display_requested(path, mode, new_window=True)

    def _on_display_requested(path, mode, new_window):
        if mode == "chimerax":
            _launch_chimerax(path)
            return
        if mode == "denovo3D":
            _launch_denovo3d(path, new_window=new_window)
            return
        if mode == "whereIsMyClass":
            _launch_whereismyclass(path, new_window=new_window)
            return
        if mode == "helicalProjection":
            _launch_helicalprojection(path, new_window=new_window)
            return
        if mode == "helicalPitch":
            _launch_helicalpitch(path, new_window=new_window)
            return
        if mode == "hill":
            _launch_hill(path, new_window=new_window)
            return
        if mode == "hi3d":
            _launch_hi3d(path, new_window=new_window)
            return
        if mode == "trueFSC":
            _launch_truefsc(path, parent=widget)
            return
        if mode == "stats":
            _launch_helical_angle_stats(
                path,
                parent=widget,
            )
            return
        tracker = _TRACKER_FOR.get(mode)
        if tracker is None:
            return
        if tracker is _napari:
            if (
                new_window
                and _napari.active() is not None
                and _is_alive_viewer(_napari.active())
            ):
                _spawn_viewer_and_open(path, mode=mode)
            else:
                _show_napari_viewer()
                try:
                    _open_file(_napari.active(), path, mode=mode)
                except Exception as exc:
                    import logging

                    logging.getLogger("helicon").warning(
                        "failed to display %s in napari: %s", path, exc
                    )
        else:
            reuse = None if new_window else tracker.active()
            if tracker is _gallery:
                _open_file(None, path, mode=mode, reuse_gallery=reuse)
            elif tracker is _text:
                _open_text_window(path, reuse_window=reuse)
            elif tracker is _plot:
                _open_fsc_plot(path, reuse_window=reuse)
            elif tracker is _images2star:
                _open_images2star_tools(
                    path,
                    parent=widget,
                    reuse_window=reuse,
                    tracker=tracker,
                )

    widget = FolderBrowserWidget(start_dir=start_dir)
    widget.file_selected.connect(_on_file_selected)
    widget.file_selected_new_window.connect(_on_file_selected_new_window)
    widget.display_requested.connect(_on_display_requested)
    widget.setWindowFlags(Qt.WindowType.Window)
    widget.setWindowTitle("Helicon - Files")
    widget.show()

    # On macOS the application-name menu item is only realized once the
    # QMainWindow's native menu bar is installed by the event loop. Re-apply
    # the identity after the window is shown so the top-left menu reads
    # "Helicon" instead of the inherited "python3.14". Also nudge the app to
    # become active so macOS renders the (initially blank) native menu bar.
    if sys.platform == "darwin":
        QTimer.singleShot(0, lambda: _set_macos_app_identity("Helicon"))

        menu_state = {
            # App that was frontmost at launch (typically the terminal) —
            # used to reproduce the "click another app" half of the cycle.
            "alternate_pid": _macos_frontmost_pid(),
            "refreshed": False,
        }

        def _realize_macos_menu(attempt: int, full_cycle: bool) -> None:
            """Reproduce the click-away-and-back cycle until the menu exists.

            Launched from a terminal the app is often never granted a real
            activation cycle, so the native File/View menu bar stays blank
            until the user clicks another app and back. Repeat the resign/
            re-activate cycle until NSApp's main menu actually contains the
            File/View items and Helicon is the frontmost application (that is
            what makes macOS display its menu bar), or give up after a short
            cap. The first two attempts do the full cycle; later ones only
            re-activate to avoid flicker in case Qt merely needed a late
            window-activation.
            """
            _force_macos_menu_realization(full_cycle=full_cycle)
            # A full cycle resigns now and re-activates 120 ms later, so its
            # result cannot be judged until after that delayed activation.
            delay = 280 if full_cycle else 30
            QTimer.singleShot(delay, lambda: _realize_macos_menu_check(attempt))

        def _realize_macos_menu_check(attempt: int) -> None:
            """Evaluate the last activation attempt; retry if still needed."""
            count = _macos_menu_item_count(_macos_ns_app())
            frontmost = _macos_frontmost_pid()
            if count >= 2 and frontmost == os.getpid():
                if not menu_state["refreshed"]:
                    # The menu is fully installed and we are frontmost, but the
                    # window server may still be showing the menu bar from
                    # before File/View were installed (or the previous app's).
                    # Repeat the user's manual action: resign, briefly bring
                    # the launch-time frontmost app forward, then come back —
                    # now that the menu exists, the re-activation re-reads it.
                    menu_state["refreshed"] = True
                    _macos_menu_refresh(attempt)
                    return
                # The activation cycle rebuilds the native menu bar, which can
                # reset the application-menu title (observed as "Apple", Qt's
                # placeholder). Re-apply the "Helicon" identity now that the
                # menu exists so the top-left menu reads Helicon.
                _set_macos_app_identity("Helicon")
                return
            if attempt >= 12:
                return
            QTimer.singleShot(
                250,
                lambda: _realize_macos_menu(attempt + 1, attempt < 3),
            )

        def _macos_menu_refresh(attempt: int) -> None:
            """Resign, activate the launch-time frontmost app, then come back."""
            _macos_resign_active()
            alternate = menu_state.get("alternate_pid") or 0
            if alternate and alternate != os.getpid():
                QTimer.singleShot(90, lambda: _macos_activate_pid(alternate))
            QTimer.singleShot(240, _macos_activate_and_front)
            QTimer.singleShot(420, lambda: _realize_macos_menu_check(attempt))

        QTimer.singleShot(150, lambda: _realize_macos_menu(1, True))

    try:
        from unittest.mock import MagicMock

        if not isinstance(widget, MagicMock):
            _install_window_shortcuts(widget)
    except Exception:
        pass

    _install_dock_save_hook(widget)
    _restore_geometry(widget, None)

    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is not None:
        app.aboutToQuit.connect(_terminate_web_apps)

    # Also terminate web app servers if the process dies without a clean
    # Qt shutdown (SIGTERM/SIGINT/SIGHUP, exit() in an except block).
    # Handlers clean up then restore default disposition and re-raise.
    # Linux children also get PR_SET_PDEATHSIG from launch_shiny_app so
    # they die if this process is SIGKILL'd and never runs these handlers.
    import atexit
    import signal

    atexit.register(_terminate_web_apps)
    _exit_signals = [signal.SIGTERM, signal.SIGINT]
    if hasattr(signal, "SIGHUP"):
        _exit_signals.append(signal.SIGHUP)
    for signum in _exit_signals:

        def _handle_signal(sig=signum):
            _terminate_web_apps()
            signal.signal(sig, signal.SIG_DFL)
            os.kill(os.getpid(), sig)

        try:
            signal.signal(signum, _handle_signal)
        except (ValueError, OSError):
            pass

    app.exec()


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add CLI arguments for the display command.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to attach arguments to.

    Returns
    -------
    argparse.ArgumentParser
        The parser with arguments added.
    """
    parser.add_argument(
        "folder",
        nargs="?",
        type=str,
        metavar="<folder>",
        default=None,
        help="folder to browse (default: current directory)",
    )

    return parser
