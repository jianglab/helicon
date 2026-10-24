"""Scatter plot (pyqtgraph) of the beam shifts of the exposures of an EPU session.

EPU writes one XML file per exposure (``FoilHole_*_Data_*.xml``) with the beam shift
used for it; the shifts of a session fall into clusters, one per hole of the pattern
collected around each stage position. The plot shows them all, for the folder of
the XML files and every folder below it.
"""

from __future__ import annotations

from pathlib import Path

try:
    from PySide6.QtWidgets import QMainWindow
except ImportError:  # pragma: no cover - only without the Qt stack
    QMainWindow = None

_BeamShiftPlotBase = QMainWindow if QMainWindow is not None else object

from .theme import (
    _display_plot_theme_colors,
    _display_theme_palette,
    _display_theme_stylesheet,
)
from .trackers import _install_window_shortcuts, _is_alive_widget, _plot

# the colour of the points (the accent of the browser), drawn half transparent so that
# a cluster of many exposures is darker than a stray one
_POINT_COLOR = (74, 143, 214)


def read_folder_beamshifts(folder: str | Path) -> dict:
    """Find and read the EPU exposure XML files of a folder and the folders below it.

    Parameters
    ----------
    folder : str or Path
        The folder of the XML files.

    Returns
    -------
    dict
        ``folder``, ``files`` (the files with a beam shift), ``beamshifts``
        (``(n, 2)`` array) and ``failed`` (the files without one).
    """
    from helicon.lib.epu import find_epu_xml_files, read_epu_beamshifts

    files, beamshifts, failed = read_epu_beamshifts(find_epu_xml_files(folder))
    return dict(folder=str(folder), files=files, beamshifts=beamshifts, failed=failed)


class _BeamShiftPlotWindow(_BeamShiftPlotBase):
    """A window with the beam shifts of EPU exposures as a scatter plot.

    A point shows the name of its XML file and its beam shift when hovered.

    Parameters
    ----------
    result : dict
        What :func:`read_folder_beamshifts` read.
    parent : QWidget, optional
        Parent widget for the window.
    """

    def __init__(self, result: dict, parent=None):
        super().__init__(parent)
        self.setProperty("helicon_theme_window", True)
        self.setStyleSheet(_display_theme_stylesheet())
        self.setPalette(_display_theme_palette())
        self._plot = None
        self._scatter = None
        self._result: dict = {}
        self._build_ui()
        self.set_data(result)
        self.resize(700, 700)

    def _build_ui(self) -> None:
        import pyqtgraph as pg

        self.plot_widget = pg.PlotWidget()
        self._plot = self.plot_widget.getPlotItem()
        # the shifts in x and y are in the same units: a circle of holes stays round
        self._plot.setAspectLocked(True)
        self._scatter = pg.ScatterPlotItem(
            size=6,
            pen=None,
            brush=pg.mkBrush(*_POINT_COLOR, 140),
            hoverable=True,
            hoverSize=9,
            hoverBrush=pg.mkBrush(220, 50, 50),
            tip=self._tip,
        )
        self._plot.addItem(self._scatter)
        self.setCentralWidget(self.plot_widget)
        self._apply_display_theme()

    @staticmethod
    def _tip(x, y, data) -> str:
        return f"{data}\nx = {x:.6g}\ny = {y:.6g}"

    def set_data(self, result: dict) -> None:
        """Show the beam shifts of another folder, keeping the same window."""
        self._result = result
        xy = result["beamshifts"]
        self._scatter.setData(
            x=xy[:, 0], y=xy[:, 1], data=[Path(f).name for f in result["files"]]
        )
        self._plot.enableAutoRange()
        name = Path(result["folder"]).name or result["folder"]
        self.setWindowTitle(f"Beam shift — {name} ({len(xy)} exposures)")
        message = f"{len(xy)} exposures in {result['folder']}"
        if result["failed"]:
            message += f"; {len(result['failed'])} XML files without a beam shift"
        self.statusBar().showMessage(message)

    def _apply_display_theme(self) -> None:
        """Apply the saved theme to the pyqtgraph canvas and decorations."""
        if self._plot is None:
            return
        colors = _display_plot_theme_colors()
        self.plot_widget.setBackground(colors["background"])
        for axis_name in ("left", "bottom"):
            axis = self._plot.getAxis(axis_name)
            axis.setPen(colors["foreground"])
            axis.setTextPen(colors["foreground"])
            axis.setTickPen(colors["foreground"])
        self._plot.setLabel("bottom", "Beam shift X", color=colors["foreground"])
        self._plot.setLabel("left", "Beam shift Y", color=colors["foreground"])
        self._plot.showGrid(x=True, y=True, alpha=0.3)

    def closeEvent(self, event):
        _plot.on_close(self)
        super().closeEvent(event)

    def changeEvent(self, event):
        from PySide6.QtCore import QEvent

        if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
            _plot.on_activate(self)
        super().changeEvent(event)


def _open_beamshift_plot(result: dict, reuse_window=None):
    """Show what :func:`read_folder_beamshifts` read, in a window of its own or in
    ``reuse_window`` if that is a beam-shift window still open.

    Returns
    -------
    _BeamShiftPlotWindow or None
        The window, or None if there was nothing to show (a message said why).
    """
    from PySide6.QtWidgets import QMessageBox

    try:
        import pyqtgraph  # noqa: F401 - availability probe
    except ImportError:
        QMessageBox.warning(
            None,
            "Missing dependency",
            "pyqtgraph is required to plot the beam shifts.\n"
            "Install it with: pip install pyqtgraph",
        )
        return None
    if not len(result["beamshifts"]):
        QMessageBox.information(
            None,
            "No beam shifts",
            f"No EPU exposure XML file with a beam shift was found in {result['folder']}.",
        )
        return None
    if (
        reuse_window is not None
        and _is_alive_widget(reuse_window)
        and isinstance(reuse_window, _BeamShiftPlotWindow)
    ):
        reuse_window.set_data(result)
        reuse_window.show()
        reuse_window.raise_()
        return reuse_window
    win = _BeamShiftPlotWindow(result)
    _plot.register(win)
    _install_window_shortcuts(win)
    win.show()
    win.raise_()
    return win
