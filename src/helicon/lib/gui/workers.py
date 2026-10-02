"""Keep background ``QThread`` workers alive past the window that started them.

A tool dialog (proc3d, images2star) owns its load/save workers as Qt
children. With ``WA_DeleteOnClose`` the dialog, and with it every child, is
destroyed as soon as it closes, so a worker that is still running would be
destroyed mid-run, which makes Qt abort with "QThread: Destroyed while
thread is still running". :func:`release_threads` detaches such workers from
their window and parks them here until they finish.
"""

from __future__ import annotations

from PySide6.QtCore import QThread, SignalInstance
from PySide6.QtWidgets import QApplication

# Running workers whose window has gone away. Holding the Python wrapper here
# keeps the C++ thread object alive until the thread finishes.
_PARKED: set = set()
_QUIT_HOOKED = False


def parked_threads() -> set:
    """Return the set of workers still running after their window closed.

    Returns
    -------
    set
        The parked ``QThread`` objects (live view, do not modify).
    """
    return _PARKED


def _disconnect_signals(thread: QThread) -> None:
    """Disconnect every custom signal of ``thread`` from its receivers."""
    for cls in type(thread).__mro__:
        if cls is QThread:
            break
        for name in vars(cls):
            try:
                signal = getattr(thread, name)
            except Exception:
                continue
            if isinstance(signal, SignalInstance):
                try:
                    signal.disconnect()
                except (RuntimeError, TypeError):
                    pass  # nothing connected


def _forget(thread: QThread) -> None:
    """Drop a finished parked worker and schedule its deletion."""
    if thread in _PARKED:
        _PARKED.discard(thread)
        try:
            thread.deleteLater()
        except RuntimeError:
            pass  # already deleted


def _wait_for_parked() -> None:
    """Give parked workers a last chance to finish when the app quits."""
    for thread in list(_PARKED):
        try:
            thread.wait(5000)
        except RuntimeError:
            pass
    _PARKED.clear()


def release_threads(threads) -> None:
    """Detach workers from a closing window so none is destroyed while running.

    Finished workers are left alone (they go away with their parent). A
    worker that is still running is asked to stop, its result signals are
    disconnected (the window is gone), it is re-parented to the
    ``QApplication`` and kept in a module-level set until its ``finished``
    signal fires, after which it is deleted.

    Parameters
    ----------
    threads : iterable of QThread
        The window's worker threads.
    """
    global _QUIT_HOOKED
    app = QApplication.instance()
    for thread in list(threads):
        try:
            running = thread.isRunning()
        except RuntimeError:
            continue  # C++ object already deleted
        if not running:
            continue
        thread.requestInterruption()
        _disconnect_signals(thread)
        thread.setParent(app)
        _PARKED.add(thread)
        thread.finished.connect(lambda t=thread: _forget(t))
        if thread.isFinished():
            # Finished between the checks above and the connect.
            _forget(thread)
    if _PARKED and app is not None and not _QUIT_HOOKED:
        app.aboutToQuit.connect(_wait_for_parked)
        _QUIT_HOOKED = True
