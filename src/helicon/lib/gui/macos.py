"""macOS/desktop application identity and window-activation helpers.

These functions are used by the ``helicon display`` command and its
standalone child processes to set the Qt/macOS application name, choose a
working Qt platform under WSLg, and realize the native menu bar on macOS.
"""

from __future__ import annotations

import os
import sys


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

    Whenever Qt ends up on xcb (forced here or already set), PyOpenGL is
    pushed to its GLX backend to match: when ``WAYLAND_DISPLAY`` is set
    PyOpenGL defaults to EGL, but under WSLg Qt renders through the
    Xwayland server (xcb/GLX), and that EGL/GLX mismatch makes napari's
    vispy canvas fail with "Attempt to retrieve context when no valid
    context".  ``PYOPENGL_PLATFORM`` is set with ``os.environ.setdefault``
    so an explicit user value always wins, and it must be set before
    napari/PyOpenGL is imported (PyOpenGL caches its backend on first
    import).
    """
    if "QT_QPA_PLATFORM" in os.environ:
        if os.environ["QT_QPA_PLATFORM"].split(":")[0] == "xcb":
            os.environ.setdefault("PYOPENGL_PLATFORM", "glx")
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
    os.environ.setdefault("PYOPENGL_PLATFORM", "glx")


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
