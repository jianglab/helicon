"""The micrographs of a RELION table of particles or of a picking job, each with its picks on it.

A window with the list of the micrographs of a particle star file (``particles.star`` of an
extraction or a selection, ``run_data.star`` of a classification or a refinement...) or of the
listing of a picking job (``manualpick.star``, ``autopick.star``) and, for the one selected, the
micrograph with the positions on it, drawn as they were picked: a circle as large as the particle
at each single particle, a band along each filament picked by its start and end, a dot at each
helical segment with one colour to each filament (see :data:`job_outcome.PICK_MODES`). The table
is read once, in the background; each micrograph is read when it is selected, also in the
background, and the last few are kept.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QSizePolicy,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

__all__ = ["MicrographViewer"]

# the longest side of the picture of a micrograph in the window
PICTURE_PIXELS = 1024
# the pictures of micrographs kept, to go back and forth quickly
KEPT_PICTURES = 12

# the threads that read in the background: a window that is closed meanwhile must not take them down
_READING: set = set()


class _TableWorker(QThread):
    """Reads the positions of a table or of a picking job; ``done(found)`` at the end, as
    :func:`job_outcome.coordinates_by_micrograph`, ``{}`` if it failed."""

    done = Signal(dict)

    def __init__(self, star: str):
        super().__init__()
        self._star = star

    def run(self) -> None:
        from .job_outcome import coordinates_by_micrograph

        try:
            found = coordinates_by_micrograph(self._star)
        except Exception:
            found = {}
        self.done.emit(found)


class _PictureWorker(QThread):
    """Reads a micrograph; ``done(micrograph, (picture, pixel size))`` at the end, the picture
    being ``(grey levels, step)`` or None if the micrograph is not found."""

    done = Signal(str, object)

    def __init__(self, star: str, mic: str):
        super().__init__()
        self._star, self._mic = star, mic

    def run(self) -> None:
        from .job_outcome import micrograph_of_table, micrograph_pixel_size

        picture = micrograph_of_table(self._mic, self._star, PICTURE_PIXELS)
        apix = micrograph_pixel_size(self._mic, self._star) if picture else None
        self.done.emit(self._mic, (picture, apix))


def _start(worker: QThread) -> None:
    _READING.add(worker)
    worker.finished.connect(lambda w=worker: _READING.discard(w))
    worker.start()


class _Item(QTreeWidgetItem):
    """A micrograph in the list: the counts are sorted as numbers."""

    def __lt__(self, other):
        column = self.treeWidget().sortColumn() if self.treeWidget() else 0
        if column > 0:
            return (self.data(column, Qt.ItemDataRole.UserRole) or 0) < (
                other.data(column, Qt.ItemDataRole.UserRole) or 0
            )
        return self.text(0).lower() < other.text(0).lower()


class _Picture(QLabel):
    """A picture scaled to the room it has, keeping its proportions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMinimumSize(200, 200)
        self._pixmap: QPixmap | None = None

    def set_array(self, array) -> None:
        from .browser_parts import _pixmap_of

        self._pixmap = None if array is None else _pixmap_of(array, array.shape[1])
        self._fit()

    def set_message(self, text: str) -> None:
        self._pixmap = None
        self.clear()
        self.setText(text)

    def _fit(self) -> None:
        if self._pixmap is None:
            self.clear()
            return
        self.setPixmap(
            self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()


class MicrographViewer(QMainWindow):
    """The micrographs of a table of particles or of a picking job: pick one in the list to see
    its particles on it; ``select`` is the micrograph shown first."""

    def __init__(self, star: str | Path, parent=None, select: str | None = None):
        super().__init__(parent)
        self.star = str(star)
        self._first = select
        self.setWindowTitle(f"Helicon - Micrographs of {Path(star).name}")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._particles: dict[str, dict] = {}
        self._mode = "points"
        self._diameter: float | None = None
        self._pictures: OrderedDict[str, tuple | None] = OrderedDict()
        self._current: str | None = None
        self._reading: str | None = None  # the micrograph being read
        self._muted = "#8f8f8f"

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        column = QVBoxLayout(left)
        column.setContentsMargins(6, 6, 6, 6)
        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Filter micrographs")
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._apply_filter)
        column.addWidget(self._filter)
        self._list = QTreeWidget()
        self._list.setRootIsDecorated(False)
        self._list.setUniformRowHeights(True)
        self._list.setAlternatingRowColors(True)
        self._list.setHeaderLabels(["Micrograph", "Particles"])
        self._list.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._list.header().setStretchLastSection(False)
        self._list.currentItemChanged.connect(self._on_current)
        column.addWidget(self._list, 1)
        self._summary = QLabel("Reading the table…")
        self._summary.setWordWrap(True)
        column.addWidget(self._summary)
        splitter.addWidget(left)

        right = QWidget()
        column = QVBoxLayout(right)
        column.setContentsMargins(6, 6, 6, 6)
        bar = QHBoxLayout()
        self._name = QLabel("")
        font = self._name.font()
        font.setBold(True)
        self._name.setFont(font)
        self._name.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        bar.addWidget(self._name, 1)
        self._show_dots = QCheckBox("Show particles")
        self._show_dots.setChecked(True)
        self._show_dots.toggled.connect(self._redraw)
        bar.addWidget(self._show_dots)
        column.addLayout(bar)
        self._picture = _Picture()
        column.addWidget(self._picture, 1)
        self._caption = QLabel("")
        self._caption.setWordWrap(True)
        column.addWidget(self._caption)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 760])
        self.setCentralWidget(splitter)
        self.resize(1100, 760)

        worker = _TableWorker(self.star)
        worker.done.connect(self._on_table)
        _start(worker)

    def set_theme(self, colors: dict[str, str]) -> None:
        self._muted = colors.get("muted", self._muted)
        self._summary.setStyleSheet(f"color: {self._muted};")
        self._caption.setStyleSheet(f"color: {self._muted};")
        if "alt" in colors:
            self._picture.setStyleSheet(
                f"background: {colors['alt']}; border-radius: 4px;"
            )

    # ── the list ───────────────────────────────────────────────────────────

    def _on_table(self, found: dict) -> None:
        particles = found.get("micrographs") or {}
        self._particles = particles
        if not particles:
            self._summary.setText("No particles with positions in this table")
            self._picture.set_message("No particles with positions in this table")
            return
        self._mode = found["mode"]
        self._diameter = found.get("diameter")
        self._show_dots.setText(
            "Show picks" if found.get("picked") else "Show particles"
        )
        labels = ["Micrograph"] + _COLUMNS[self._mode]
        self._list.setHeaderLabels(labels)
        for i in range(1, len(labels)):
            self._list.header().setSectionResizeMode(
                i, QHeaderView.ResizeMode.ResizeToContents
            )
        items = []
        total = 0
        for mic, p in particles.items():
            counts = _counts(p, self._mode)
            total += counts[0]
            item = _Item([Path(mic).name])
            item.setData(0, Qt.ItemDataRole.UserRole, mic)
            item.setToolTip(0, mic)
            for i, n in enumerate(counts, 1):
                item.setText(i, f"{n:,}")
                item.setData(i, Qt.ItemDataRole.UserRole, n)
                item.setTextAlignment(i, Qt.AlignmentFlag.AlignRight)
            items.append(item)
        self._list.addTopLevelItems(items)
        self._list.setSortingEnabled(True)
        self._list.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        kind = _COLUMNS[self._mode][0].lower()
        summary = f"{total:,} {kind} in {_plural(len(particles), 'micrograph')}"
        if found.get("picked"):
            summary += f"\nPicked as {_PICKED[self._mode]}"
            if self._diameter:
                summary += f", {self._diameter:g} Å across"
        self._summary.setText(summary)
        self._list.setCurrentItem(self._list.topLevelItem(0))
        if self._first is not None:
            self.select(self._first)
        self._list.setFocus()

    def _apply_filter(self, text: str) -> None:
        text = text.strip().lower()
        for i in range(self._list.topLevelItemCount()):
            item = self._list.topLevelItem(i)
            mic = item.data(0, Qt.ItemDataRole.UserRole) or ""
            item.setHidden(bool(text) and text not in mic.lower())

    def micrographs(self) -> list[str]:
        """The micrographs in the list, in the order shown."""
        return [
            self._list.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole)
            for i in range(self._list.topLevelItemCount())
        ]

    def select(self, mic: str) -> None:
        """Show a micrograph of the table."""
        for i in range(self._list.topLevelItemCount()):
            item = self._list.topLevelItem(i)
            if item.data(0, Qt.ItemDataRole.UserRole) == mic:
                self._list.setCurrentItem(item)
                return

    # ── the picture ────────────────────────────────────────────────────────

    def _on_current(self, item, _previous=None) -> None:
        if item is None:
            return
        mic = item.data(0, Qt.ItemDataRole.UserRole)
        self._current = mic
        self._name.setText(Path(mic).name)
        self._name.setToolTip(mic)
        if mic in self._pictures:
            self._pictures.move_to_end(mic)
            self._redraw()
            return
        self._caption.setText("")
        self._picture.set_message("Reading the micrograph…")
        if self._reading is None:
            self._read(mic)

    def _read(self, mic: str) -> None:
        # one read at a time: going quickly down the list reads only where it stops
        self._reading = mic
        worker = _PictureWorker(self.star, mic)
        worker.done.connect(self._on_picture)
        _start(worker)

    def _on_picture(self, mic: str, read: tuple) -> None:
        self._reading = None
        self._pictures[mic] = read
        while len(self._pictures) > KEPT_PICTURES:
            self._pictures.popitem(last=False)
        if self._current is not None and self._current not in self._pictures:
            self._read(self._current)  # another micrograph was chosen meanwhile
        elif mic == self._current:
            self._redraw()

    def _redraw(self) -> None:
        from .job_outcome import draw_coordinates, particle_groups, plain_ground

        mic = self._current
        if mic is None or mic not in self._pictures or mic not in self._particles:
            return
        particles = self._particles[mic]
        picture, apix = self._pictures[mic]
        note = ""
        if picture is None:
            picture = plain_ground(particle_groups(particles), PICTURE_PIXELS)
            note = "\nMicrograph not found: the positions alone"
        grey, step = picture
        if self._show_dots.isChecked():
            # the size of a particle, or the width of a filament, in pixels of the micrograph
            radius = self._diameter / 2 / apix if self._diameter and apix else None
            image = draw_coordinates(grey, step, particles, self._mode, radius)
        else:
            image = grey
        self._picture.set_array(image)
        self._caption.setText(_caption(particles, self._mode) + note)

    def wait(self, ms: int = 10000) -> None:
        """Wait for the reads in progress (for tests)."""
        for worker in list(_READING):
            worker.wait(ms)


# the counts in the list of the micrographs, by the mode of the picks
_COLUMNS = {
    "points": ["Particles"],
    "particles": ["Particles"],
    "startend": ["Filaments"],
    "segments": ["Segments", "Filaments"],
}
# how the picks of a picking job were made, in the summary under the list
_PICKED = {
    "particles": "single particles",
    "startend": "filaments, by their start and end",
    "segments": "helical segments",
}


def _counts(particles: dict, mode: str) -> list[int]:
    """The counts of a micrograph in the list: the columns of :data:`_COLUMNS`."""
    n = len(particles["x"])
    tube = particles.get("tube")
    filaments = len(set(tube.tolist())) if tube is not None else 0
    if mode == "startend":
        return [filaments]
    if mode == "segments":
        return [n, filaments]
    return [n]


def _plural(n: int, word: str) -> str:
    return f"{n:,} {word}{'s' * (n != 1)}"


def _caption(particles: dict, mode: str = "points") -> str:
    counts = _counts(particles, mode)
    if mode == "startend":
        return _plural(counts[0], "filament")
    if mode == "segments":
        return f"{_plural(counts[0], 'segment')} in {_plural(counts[1], 'filament')}"
    return _plural(counts[0], "particle")
