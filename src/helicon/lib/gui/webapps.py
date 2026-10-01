"""Shiny web-app instance lifecycle for the file browser.

``_WEB_APP_INSTANCES`` tracks every ``helicon.webApps.app`` server spawned
from the display app; reuse navigates the most-recently-used alive instance
while ``New``-checked clicks spawn coexisting servers.
``_terminate_web_apps`` reaps everything on display exit.
"""

from __future__ import annotations

import time

from .theme import _get_display_theme


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


def _make_bookmark_query(tab_name: str, params: dict) -> dict:
    """Build the query params of a bookmark URL that opens a tab.

    Parameters
    ----------
    tab_name : str
        The tab's name in the navbar.
    params : dict
        The tab's parameters, by the short keys of its ``BOOKMARK_DEFAULTS``.

    Returns
    -------
    dict
        Query key -> value for ``launch_shiny_app(query_params=...)``, e.g.
        ``?tab=HelicalPitch&url_params=/path/run_it020_data.star``; see
        ``helicon.webApps.bookmark``.
    """
    from helicon.webApps import bookmark

    return bookmark.query(tab_name, params)


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


def _class2d_bookmark(path: str) -> dict:
    """Bookmark inputs of a Class2D job for HelicalPitch and AbInitio3D.

    Both tabs take the Class2D parameters and the class averages. The
    companion of the given file is filled in when it exists on disk (see
    ``helicon.webApps.class2d_files.companion``): the RELION
    ``run_itNNN_data.star`` and ``run_itNNN_classes.mrcs`` of the same
    iteration, or a cryoSPARC job's particles and class averages.

    Parameters
    ----------
    path : str
        A Class2D parameter file (.star, .cs) or class-average stack.

    Returns
    -------
    dict
        The tab's bookmark values, with ``url_params`` and ``url_classes``
        set for the files that were found.
    """
    from pathlib import Path

    from helicon.webApps import class2d_files

    file_path = str(Path(path).absolute())
    other = class2d_files.companion(file_path)
    bookmark = {"mode_params": "url", "mode_classes": "url"}
    if class2d_files.is_params_file(file_path):
        bookmark["url_params"] = file_path
        if other:
            bookmark["url_classes"] = other
    else:
        bookmark["url_classes"] = file_path
        if other:
            bookmark["url_params"] = other
    return bookmark


def _launch_helicalpitch(path: str, *, new_window: bool = False) -> None:
    """Open a file in the HelicalPitch tab via bookmark URL.

    Derives the companion file (star↔mrcs) from the given path when it
    exists on disk, so both params and class images are loaded together.
    """
    from PySide6.QtWidgets import QMessageBox

    try:
        _launch_or_reuse_web_app(
            "HelicalPitch",
            _make_bookmark_query("HelicalPitch", _class2d_bookmark(path)),
            new_window=new_window,
        )
    except Exception as exc:
        QMessageBox.critical(
            None,
            "HelicalPitch Launch Error",
            f"Failed to launch HelicalPitch:\n{exc}",
        )


def _launch_abinitio3d(path: str, *, new_window: bool = False) -> None:
    """Open a Class2D job in the AbInitio3D tab via bookmark URL.

    The companion file (star↔mrcs, or a cryoSPARC class-average .mrc's
    particles dataset) is found the way HelicalPitch finds it, so both the
    parameters and the class averages are loaded together.
    """
    from PySide6.QtWidgets import QMessageBox

    try:
        _launch_or_reuse_web_app(
            "AbInitio3D",
            _make_bookmark_query("AbInitio3D", _class2d_bookmark(path)),
            new_window=new_window,
        )
    except Exception as exc:
        QMessageBox.critical(
            None,
            "AbInitio3D Launch Error",
            f"Failed to launch AbInitio3D:\n{exc}",
        )


def _launch_hill(path: str, *, new_window: bool = False) -> None:
    """Open a file in the HILL tab of the consolidated Helicon Lab web app.

    Launches the unified Shiny app and selects the HILL tab with the given file.
    """
    from PySide6.QtWidgets import QMessageBox

    try:
        from pathlib import Path

        params = _make_bookmark_query(
            "HILL", {"input_mode": "2", "url": str(Path(path).resolve())}
        )
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
        from pathlib import Path

        params = _make_bookmark_query(
            "HI3D", {"input_mode": "url", "url": str(Path(path).resolve())}
        )
        _launch_or_reuse_web_app("HI3D", params, new_window=new_window)
    except Exception as exc:
        QMessageBox.critical(
            None,
            "HI3D Launch Error",
            f"Failed to launch HI3D:\n{exc}",
        )
