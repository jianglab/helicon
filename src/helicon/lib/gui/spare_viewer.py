"""A napari window process that is started ahead of the request for a new window.

A new display window runs in a process of its own (a second napari canvas inside a running
viewer's event loop crashes on some platforms). That process had to import napari first,
about ten seconds, before it could make its window. A spare process does this import while
the file browser is in use and then waits for its file on its standard input; the file
browser gives it the file and the window appears at once. The first new window of a session
is still made the long way; after it the next spare is started.

The spare ends when the file browser does (its standard input closes), so none is left
behind. ``HELICON_NO_SPARE_VIEWER=1`` switches the spares off.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading

_COMMAND_FOR = "from helicon.commands.display import _run_standalone; _run_standalone()"
WAIT_FLAG = "--wait"

_spare: subprocess.Popen | None = None
_lock = threading.Lock()


def enabled() -> bool:
    """Whether spare processes are used (``HELICON_NO_SPARE_VIEWER`` switches them off)."""
    return os.environ.get("HELICON_NO_SPARE_VIEWER", "") in ("", "0")


def _start(*arguments: str, stdin=None) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", _COMMAND_FOR, *arguments],
        stdin=stdin,
        start_new_session=True,
    )


def start_spare() -> bool:
    """Start a spare process if there is none alive. Returns whether one is waiting."""
    global _spare
    if not enabled():
        return False
    with _lock:
        if _spare is not None and _spare.poll() is None:
            return True
        try:
            _spare = _start(WAIT_FLAG, stdin=subprocess.PIPE)
        except OSError:
            _spare = None
            return False
        return True


def open_in_new_process(path, mode=None) -> None:
    """Show ``path`` in a new napari window, using the spare process when there is one.

    Without a live spare, the window's process is started with the file, as before.
    Either way, a new spare is started for the next window.

    Parameters
    ----------
    path : str or Path
        The file to show.
    mode : str, optional
        The display mode (for example ``"volume"``); the default one when omitted.
    """
    global _spare
    with _lock:
        spare, _spare = _spare, None
    sent = False
    if spare is not None and spare.poll() is None:
        try:
            spare.stdin.write(
                (json.dumps({"path": str(path), "mode": mode or ""}) + "\n").encode()
            )
            spare.stdin.flush()
            spare.stdin.close()
            sent = True
        except (OSError, ValueError):
            sent = False
    if not sent:
        _start(str(path), mode or "")


def read_job(stream) -> tuple[str, str | None] | None:
    """Read the file and mode a spare process is to show; None when the stream just ends.

    Parameters
    ----------
    stream : file object
        The standard input of the spare process: one line of JSON.

    Returns
    -------
    tuple of (str, str or None), or None
        The path and the mode (None for the default mode), or None if the file browser
        went away without sending a file.
    """
    line = stream.readline()
    if not line:
        return None
    job = json.loads(line)
    return job["path"], (job.get("mode") or None)
