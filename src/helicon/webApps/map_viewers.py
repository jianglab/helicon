"""Open a map of a web app in napari or ChimeraX on this computer.

The web apps that make 3D maps (Denovo3D, AbInitio3D) can show them in the viewers of
the file browser: napari's 3D volume view (``helicon display``) and ChimeraX. Both are
desktop programs, so a viewer is started only when the server runs on the same
computer as the browser (the rule of the Home tab for its desktop programs): never on
a hosting service, never for a remote visitor. The map is written to a file in the
cache folder and the viewer started detached, as the file browser does.

The map can also be shown in the browser, in Mol* (the viewer of EMDB, PDBe and RCSB),
on a local run and on a hosted site. It is opened in a new tab only when the button
is clicked: nothing of Mol* is loaded before, and the page of the new tab loads it
from a CDN. The tab is opened by the click itself (a tab the server opens later is
blocked as a pop-up) and sent to the page once the map is ready. The map is served
to that session alone, from a route whose address holds the session's id.
"""

from __future__ import annotations

import logging
import sys
import secrets
import time
from pathlib import Path

import numpy as np
from shiny import reactive, ui

logger = logging.getLogger(__name__)

# maps written for viewing, and how long they are kept
_FOLDER = "viewed_maps"
_KEEP_SECONDS = 24 * 3600


def napari_available() -> bool:
    """Whether napari (and so the file browser's display) is installed."""
    import helicon

    return bool(helicon.has_napari())


def chimerax_path() -> str | None:
    """The ChimeraX executable, or None."""
    from helicon.lib.gui.file_browser import _find_chimerax

    return _find_chimerax()


def can_launch(session) -> bool:
    """Whether a viewer may be started for the visitor of ``session``.

    Only when the browser is on the server's own computer: not on a hosting service,
    not for a visitor from elsewhere (decided from the connection's address).
    """
    from . import deployment
    from .tabs import home_tab

    if deployment.is_cloud():
        return False
    if not home_tab.client_is_local(session):
        return False
    try:
        return session.clientdata.url_hostname() in home_tab._LOCAL_HOSTS
    except Exception:
        return False


def write_map(volume, apix, name="map") -> Path:
    """Write ``volume`` as an MRC file in the cache folder, and tidy up old ones.

    Parameters
    ----------
    volume : numpy.ndarray
        The map (z, y, x).
    apix : float
        Voxel size, in A.
    name : str, optional
        Start of the file name.

    Returns
    -------
    pathlib.Path
    """
    import helicon
    import mrcfile

    folder = Path(helicon.cache_dir) / _FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    now = time.time()
    for old in folder.glob("*.mrc"):
        try:
            if now - old.stat().st_mtime > _KEEP_SECONDS:
                old.unlink()
        except OSError:
            pass
    # a random part as well: two maps written in the same millisecond would
    # otherwise get one name, and the second would overwrite the first
    path = (
        folder / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}_{int(now * 1000) % 1000:03d}"
        f"_{secrets.token_hex(3)}.mrc"
    )
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(np.asarray(volume, dtype=np.float32))
        mrc.voxel_size = float(apix)
    return path


def napari_command(path) -> list[str]:
    """The command that opens ``path`` as a 3D volume in a new napari window
    (the file browser's Volume view, in a process of its own)."""
    return [
        sys.executable,
        "-c",
        "from helicon.commands.display import _run_standalone; _run_standalone()",
        str(path),
        "volume",
    ]


def chimerax_command(path) -> list[str] | None:
    """The command that opens ``path`` in ChimeraX, or None without ChimeraX."""
    exe = chimerax_path()
    return None if exe is None else [exe, str(path)]


def _spawn(command, log_name):
    """Start ``command`` detached, its output in a log in the cache folder."""
    import helicon
    from helicon.lib.terminal import _spawn_detached

    log = Path(helicon.cache_dir) / "logs" / f"{log_name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(
            f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(command[:1])} =====\n"
        )
    return _spawn_detached(command, log_path=log, return_process=True)


def launch(viewer, path) -> str | None:
    """Open ``path`` in ``viewer`` (``"napari"`` or ``"chimerax"``).

    Returns
    -------
    str or None
        None when it was started; otherwise what to tell the user.
    """
    if viewer == "napari":
        if not napari_available():
            return 'napari is not installed (pip install "helicon[gui]").'
        command = napari_command(path)
    elif viewer == "chimerax":
        command = chimerax_command(path)
        if command is None:
            return (
                "ChimeraX was not found. Install it from "
                "https://www.cgl.ucsf.edu/chimerax/ or put it on your PATH."
            )
    else:
        return f"unknown viewer {viewer!r}"
    proc = _spawn(command, f"map_viewer_{viewer}")
    if proc is None:
        logger.error("failed to start %s for %s", viewer, path)
        return f"{viewer} could not be started."
    return None


# Mol* is loaded by the page of the new tab, from this version
MOLSTAR_VERSION = "5.13.0"
_MOLSTAR_CDN = f"https://cdn.jsdelivr.net/npm/molstar@{MOLSTAR_VERSION}/build/viewer/"

# The click opens the tab itself (a pop-up the server opened later would be blocked);
# the server then sends the tab to the page of the map.
_OPEN_TAB_JS = (
    "window.__heliconMapTab = window.open('about:blank', '_blank');"
    "if (window.__heliconMapTab) { window.__heliconMapTab.document.write("
    "'<p style=\"font-family: sans-serif; margin: 2em;\">Preparing the map...</p>'); }"
)
_TAB_HANDLER_JS = """
if (window.Shiny && !window.__heliconMapHandler) {
  window.__heliconMapHandler = true;
  Shiny.addCustomMessageHandler('helicon_open_map', function (msg) {
    var page = new URL(msg.page, document.baseURI);
    page.searchParams.set('map', new URL(msg.map, document.baseURI).href);
    page.searchParams.set('name', msg.name || 'map');
    var tab = window.__heliconMapTab;
    window.__heliconMapTab = null;
    if (msg.error) {
      if (tab && !tab.closed) { tab.close(); }
      return;
    }
    if (tab && !tab.closed) { tab.location.href = page.href; }
    else { window.open(page.href, '_blank'); }
  });
}
"""


def mrc_bytes(volume, apix) -> bytes:
    """``volume`` as the bytes of an MRC file."""
    import tempfile

    import mrcfile

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "map.mrc"
        with mrcfile.new(str(path)) as mrc:
            mrc.set_data(np.asarray(volume, dtype=np.float32))
            mrc.voxel_size = float(apix)
        return path.read_bytes()


def molstar_page() -> str:
    """The page of the new tab: Mol*, loaded from the CDN, showing the map.

    The map's address and name come from the page's own query string.
    """
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Map - Mol*</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="{_MOLSTAR_CDN}molstar.css">
<style>html, body, #app {{ margin: 0; height: 100%; width: 100%; }}
#msg {{ font-family: sans-serif; margin: 2em; }}</style></head>
<body><div id="app"></div><p id="msg">Loading Mol* ...</p>
<script src="{_MOLSTAR_CDN}molstar.js"></script>
<script>
(function () {{
  var q = new URLSearchParams(location.search);
  var url = q.get('map');
  var msg = document.getElementById('msg');
  if (!url) {{ msg.textContent = 'No map was given.'; return; }}
  document.title = (q.get('name') || 'map') + ' - Mol*';
  molstar.Viewer.create('app', {{
    layoutIsExpanded: false, layoutShowControls: true, layoutShowRemoteState: false,
    layoutShowSequence: false, layoutShowLog: false, layoutShowLeftPanel: false,
    viewportShowExpand: true, collapseLeftPanel: true
  }}).then(function (viewer) {{
    msg.remove();
    // the isosurface at 2 standard deviations above the mean; Mol* has the sliders
    return viewer.loadVolumeFromUrl(
      {{ url: url, format: 'ccp4', isBinary: true }},
      [{{ type: 'relative', value: 2, color: 0x4f8ec9, alpha: 0.9 }}]);
  }}).catch(function (e) {{
    msg.textContent = 'The map could not be shown: ' + e;
    document.body.appendChild(msg);
  }});
}})();
</script></body></html>"""


def web_button(prefix, button_class="btn-default"):
    """The button that opens the map in Mol*, in a new browser tab."""
    return ui.tooltip(
        ui.span(
            ui.input_action_button(
                f"{prefix}_web",
                "View in browser",
                class_=button_class,
                onclick=_OPEN_TAB_JS,
            ),
            ui.tags.script(_TAB_HANDLER_JS),
            style="display: flex; align-items: stretch;",
        ),
        "Open this map in Mol* (the viewer of EMDB and PDB), in a new browser tab. "
        "Nothing of it is loaded until you click.",
    )


def web_routes(state):
    """The two handlers of a session's routes: the map, and the page that shows it.

    Parameters
    ----------
    state : dict
        ``state["map"]`` holds the bytes of the map to serve, or None.
    """
    from starlette.responses import PlainTextResponse, Response

    def serve_map(request):
        data = state["map"]
        if data is None:
            return PlainTextResponse("There is no map to show yet.", status_code=404)
        return Response(
            data,
            media_type="application/octet-stream",
            headers={"Cache-Control": "no-store"},
        )

    def serve_page(request):
        return Response(
            molstar_page(),
            media_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

    return serve_map, serve_page


def register_web(input, session, prefix, get_map, name):
    """Serve the map to Mol* in a new tab when the button is clicked.

    Parameters
    ----------
    input, session
        The module's.
    prefix : str
        The button's id is ``<prefix>_web``.
    get_map : callable
        Returns ``(volume, apix)`` of the map shown now, or None.
    name : str
        Name shown in the tab.

    Returns
    -------
    dict
        The routes' state, ``{"map": bytes or None}``; for tests.
    """
    state = {"map": None}
    serve_map, serve_page = web_routes(state)

    map_url = session.dynamic_route(f"{prefix}_map_file", serve_map)
    page_url = session.dynamic_route(f"{prefix}_map_page", serve_page)
    session.on_ended(lambda: state.update(map=None))

    @reactive.effect
    @reactive.event(input[f"{prefix}_web"])
    async def _open():
        async def tell(**kw):
            await session.send_custom_message(
                "helicon_open_map", dict(page=page_url, map=map_url, name=name, **kw)
            )

        try:
            found = get_map()
            if found is None:
                ui.notification_show("There is no map to show yet.", duration=5)
                await tell(error=True)
                return
            volume, apix = found
            state["map"] = mrc_bytes(volume, apix)
        except Exception as e:
            logger.error("could not prepare the map for the browser: %s", e)
            ui.notification_show(f"The map could not be prepared: {e}", type="error")
            await tell(error=True)
            return
        await tell()

    state["effect"] = _open
    return state


_LABEL = {"napari": "napari", "chimerax": "ChimeraX"}
# what the buttons say; napari is the program behind "View in 3D"
_BUTTON = {"napari": "View in 3D", "chimerax": "View in ChimeraX"}
_TIP = {
    "napari": "Open this map in napari's 3D view on this computer",
    "chimerax": "Open this map in ChimeraX on this computer",
}


def viewer_buttons(session, prefix, button_class="btn-default"):
    """The buttons that open the map in napari ("View in 3D") and in ChimeraX.

    An empty list where a viewer cannot be started (a hosting service, a visitor
    from another computer). The ids are ``<prefix>_napari`` and
    ``<prefix>_chimerax``; a button for a program that is not installed is
    disabled, with the reason as its tooltip. ``button_class`` is the Bootstrap
    class of the buttons: that of the download button next to them.
    """
    if not can_launch(session):
        return []
    buttons = []
    for viewer in ("napari", "chimerax"):
        missing = (
            None
            if (napari_available() if viewer == "napari" else chimerax_path())
            else (
                'napari is not installed (pip install "helicon[gui]")'
                if viewer == "napari"
                else "ChimeraX was not found; install it or put it on your PATH"
            )
        )
        button = ui.input_action_button(
            f"{prefix}_{viewer}",
            _BUTTON[viewer],
            class_=button_class,
            disabled=missing is not None or None,
        )
        # the span stretches to the row's height, and the button to the span's, so
        # that the buttons are as tall as the download button next to them
        buttons.append(
            ui.tooltip(
                ui.span(button, style="display: flex; align-items: stretch;"),
                missing or _TIP[viewer],
            )
        )
    return buttons


def download_link(session, output_id, label, button_class="btn-primary"):
    """The link of a download output, drawn as a button of the given style.

    ``render.download`` draws only the plain Shiny button. Its handler is
    registered by decorating it, whether it is a module-level output or one made
    inside a ``render.ui``; this draws the link to it with another look, as
    Denovo3D's download button always did.

    Parameters
    ----------
    session
        The module's session.
    output_id : str
        Name of the handler (the name of its function).
    label
        What the button says.
    button_class : str, optional
        Bootstrap class of the button. Defaults to the blue ``"btn-primary"``.
    """
    from htmltools import tags

    return tags.a(
        label,
        id=session.ns(output_id),
        class_=f"btn {button_class} shiny-download-link",
        href="",
        target="_blank",
    )


def button_row(
    download,
    session,
    prefix,
    justify="flex-start",
    style="",
    button_class="btn-default",
):
    """The download button, "View in browser" (Mol*) and, where they may start, the
    napari and ChimeraX buttons, in one row.

    Parameters
    ----------
    download
        The download button.
    session
        The module's session.
    prefix : str
        As for :func:`viewer_buttons`.
    justify : str, optional
        CSS ``justify-content`` of the row. Defaults to ``"flex-start"``.
    style : str, optional
        More CSS for the row.
    button_class : str, optional
        The Bootstrap class of the three buttons, to look like ``download``:
        ``"btn-primary"`` beside a blue one. Defaults to ``"btn-default"``, the
        class of Shiny's download buttons.
    """
    return ui.div(
        download,
        web_button(prefix, button_class),
        *viewer_buttons(session, prefix, button_class),
        style=(
            "display: flex; flex-wrap: wrap; align-items: stretch; gap: 8px; "
            f"justify-content: {justify}; {style}"
        ),
    )


def register(input, session, prefix, get_map, name):
    """Start the viewers when their buttons are clicked.

    Parameters
    ----------
    input, session
        The module's.
    prefix : str
        As for :func:`viewer_buttons`.
    get_map : callable
        Returns ``(volume, apix)`` of the map shown now, or None.
    name : str
        Start of the map's file name.
    """

    def make(viewer):
        @reactive.effect
        @reactive.event(input[f"{prefix}_{viewer}"])
        def _open():
            # the buttons are drawn only where a viewer may start; a client that
            # sends the click anyway is refused here as well
            if not can_launch(session):
                return
            found = get_map()
            if found is None:
                ui.notification_show("There is no map to show yet.", duration=5)
                return
            volume, apix = found
            try:
                path = write_map(volume, apix, name)
            except Exception as e:
                logger.error("could not write the map for viewing: %s", e)
                ui.notification_show(f"The map could not be written: {e}", type="error")
                return
            problem = launch(viewer, path)
            if problem:
                ui.notification_show(problem, type="warning", duration=12)
            else:
                ui.notification_show(
                    f"Opening the map in {_LABEL[viewer]} (a new window).", duration=6
                )

        return _open

    effects = [make("napari"), make("chimerax")]
    effects.append(register_web(input, session, prefix, get_map, name)["effect"])
    return effects
