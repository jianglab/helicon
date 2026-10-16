"""What a RELION or CryoSPARC job made, as pictures: the result that you would look at first.

The preview of a job folder shows the outcome of the job, not a list of its files (they are in the
list of the window): for a 2D classification the class averages of the last iteration, for a refinement
the central section of the map, its FSC curve and the distribution of the views, for a post-processing
the sharpened map and the corrected FSC with the resolution, for a CTF estimation the histograms of the
defocus and of the resolution of the fits, for a CTF refinement the change of the defocus and the fitted
aberrations, for a motion correction the histogram of the total motion and the trajectories of a few
micrographs, for a polishing the contribution of each movie frame at some resolutions, and for the jobs
that import, pick, extract or select particles the numbers (and, for an extraction, a few of the
particles). :func:`build_outcome` finds the files of the job (the highest iteration, the final map),
reads what is needed from them, and returns the pictures as arrays. It reads files, some of them large,
and is meant to run in a thread; it needs no window, but the plots are drawn with Qt (``QImage`` and
``QPainter``), which a thread may do.

A job of a kind that is not known, or one that has not made its files yet, gives ``None`` (the preview
then shows the facts of the job and its links, and no list of files).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["build_outcome", "fsc_resolution", "view_distribution_image"]

# the number of particles read for the distribution of the views: the first ones are enough
VIEW_PARTICLES = 300_000
# the particles whose defocus before and after a CTF refinement are compared
DEFOCUS_PARTICLES = 100_000
# the tiles of a montage of class averages or maps
MAX_TILES = 24
# the particles shown as examples of an extraction
EXAMPLES = 12
# the micrographs whose motion is drawn as examples
MOTIONS = 6
# the axis of the histogram of the resolution of the CTF fits: ends where this share of the
# micrographs is beyond, or at this resolution (Å)
CTF_RESOLUTION_RANGE = (0.05, 10.0)
# a table of more bytes than this is read in blocks spread over it to count filaments: a column of
# a ``.cs`` file is spread over the whole file (the rows are stored one after the other)
EXACT_BYTES = 1 << 30
# the days that the outcome of a job is kept in the cache
OUTCOME_DAYS = 30
# the most trajectory files of a CryoSPARC motion correction that are read
MAX_TRAJECTORIES = 600
BLOCK_ROWS = 50_000
SAMPLE_BLOCKS = 5
# the most coordinate files of a picking job that are read: the others are counted from these
MAX_COORDINATE_FILES = 3000
TILE = 112


# ── small readers ────────────────────────────────────────────────────────────


def _last_numbered(folder: Path, pattern: str) -> Path | None:
    """The file of ``folder`` that matches ``pattern`` (one group, a number) with the highest number."""
    best, found = -1, None
    rx = re.compile(pattern)
    try:
        for entry in folder.iterdir():
            m = rx.fullmatch(entry.name)
            if m and int(m.group(1)) > best:
                best, found = int(m.group(1)), entry
    except OSError:
        return None
    return found


def _star_loop(path: Path, block: str) -> dict[str, list[str]]:
    """The columns of the loop of a star file's data block, as lists of text; ``{}`` if not found.

    For the tables of a model or a post-processing file (hundreds of rows); the table of the
    particles of a job has its own reader.
    """
    columns: list[str] = []
    rows: list[list[str]] = []
    state = "find"
    try:
        with open(path, errors="replace") as f:
            for line in f:
                line = line.strip()
                if state == "find":
                    if line == f"data_{block}":
                        state = "block"
                elif state == "block":
                    if line.startswith("data_"):
                        break
                    if line == "loop_":
                        state = "columns"
                elif state == "columns":
                    if line.startswith("_"):
                        columns.append(line.split()[0].lstrip("_"))
                    elif line and not line.startswith("#"):
                        state = "rows"
                        rows.append(line.split())
                elif state == "rows":
                    if not line:
                        break
                    if line.startswith("data_"):
                        break
                    if not line.startswith("#"):
                        rows.append(line.split())
    except OSError:
        return {}
    return {name: [r[i] for r in rows if len(r) > i] for i, name in enumerate(columns)}


def _star_value(path: Path, block: str, name: str) -> str | None:
    """The value of a ``_rlnName value`` line of a (non-loop) data block."""
    state = False
    try:
        with open(path, errors="replace") as f:
            for line in f:
                line = line.strip()
                if line == f"data_{block}":
                    state = True
                elif state and line.startswith("data_"):
                    return None
                elif state and line.startswith(f"_{name} "):
                    return line.split()[1]
    except OSError:
        return None
    return None


def _floats(values: list[str]):
    import numpy as np

    return np.array([float(v) for v in values], dtype=float)


def _central_slice(path: Path):
    """The central section of a map and its pixel size, reading no more than that section."""
    import mrcfile
    import numpy as np

    with mrcfile.mmap(path, mode="r", permissive=True) as mrc:
        data = mrc.data
        if data is None:
            return None, 0.0
        apix = float(mrc.voxel_size.x)
        if data.ndim == 3:
            return np.asarray(data[data.shape[0] // 2], dtype=np.float32), apix
        return np.asarray(data, dtype=np.float32), apix


def _image(array) -> "np.ndarray":  # noqa: F821
    """A section as a grey-level picture, stretched between its 1st and 99th percentiles."""
    from .browser_parts import thumbnail_array

    return thumbnail_array(array, max_size=300)


def _panel(title: str, image, caption: str = "") -> dict:
    return dict(title=title, image=image, caption=caption)


def _plot_panel(title: str, draw, caption: str = "") -> dict | None:
    """A panel with a plot, or None when the plot cannot be drawn (no Qt application, say):
    the other pictures of the job are still shown."""
    try:
        return _panel(title, draw(), caption)
    except Exception:
        logger.debug("cannot draw the plot %r", title, exc_info=True)
        return None


# ── plots, drawn with Qt ─────────────────────────────────────────────────────

_PLOT_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]


def fsc_resolution(spatial_frequency, fsc, level: float = 0.143) -> float | None:
    """The resolution (Å) at which a curve falls below ``level`` for the first time, or None.

    Parameters
    ----------
    spatial_frequency : array-like
        In 1/Å, increasing.
    fsc : array-like
        The correlation at each frequency.
    level : float, optional
        0.143 for the gold standard, 0.5 for a comparison with a model.
    """
    import numpy as np

    x = np.asarray(spatial_frequency, dtype=float)
    y = np.asarray(fsc, dtype=float)
    below = np.flatnonzero((y < level) & (x > 0))
    if len(below) == 0:
        return None
    i = below[0]
    if i == 0:
        return None
    x0, x1, y0, y1 = x[i - 1], x[i], y[i - 1], y[i]
    t = (y0 - level) / (y0 - y1) if y0 != y1 else 0.0
    frequency = x0 + t * (x1 - x0)
    return 1.0 / frequency if frequency > 0 else None


def _require_application() -> None:
    """Qt aborts the process when text is drawn without an application: say so instead."""
    from PySide6.QtGui import QGuiApplication

    if QGuiApplication.instance() is None:
        raise RuntimeError("plots are drawn with Qt, which needs an application")


def _qimage_to_array(image) -> "np.ndarray":  # noqa: F821
    """A QImage as an ``(h, w, 3)`` array (through the bytes of the image, which are safe to copy)."""
    import numpy as np
    from PySide6.QtGui import QImage

    image = image.convertToFormat(QImage.Format.Format_RGB888)
    w, h, step = image.width(), image.height(), image.bytesPerLine()
    raw = bytes(image.constBits())
    return (
        np.frombuffer(raw, dtype=np.uint8)
        .reshape(h, step)[:, : w * 3]
        .reshape(h, w, 3)
        .copy()
    )


def _plot_fsc(curves, marks, title: str = "FSC") -> "np.ndarray":  # noqa: F821
    """Draw FSC curves: ``curves`` is ``[(label, spatial frequency, fsc), ...]``; ``marks`` are
    the levels with a thin line (0.143, 0.5)."""
    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen

    _require_application()
    width, height = 460, 300
    image = QImage(width, height, QImage.Format.Format_RGB888)
    image.fill(QColor("white"))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    left, right, top, bottom = 46, 12, 26, 44
    plot = QRectF(left, top, width - left - right, height - top - bottom)
    xmax = max((float(c[1].max()) for c in curves), default=1.0) or 1.0
    font = QFont()
    font.setPointSizeF(8.5)
    painter.setFont(font)
    painter.setPen(QPen(QColor("#222222")))
    painter.drawText(QRectF(0, 3, width, 18), Qt.AlignmentFlag.AlignCenter, title)

    def at(x, y):
        return QPointF(
            plot.left() + x / xmax * plot.width(), plot.bottom() - y * plot.height()
        )

    grid = QPen(QColor("#dddddd"))
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        painter.setPen(grid)
        painter.drawLine(at(0, tick), at(xmax, tick))
        painter.setPen(QPen(QColor("#444444")))
        painter.drawText(
            QRectF(0, at(0, tick).y() - 8, left - 5, 16),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            f"{tick:g}",
        )
    # the x axis in 1/Å, labelled with the resolutions in Å at a few round values
    last_label = -1000.0
    for resolution in (100, 30, 10, 6, 4, 3, 2.5, 2, 1.5):
        x = 1.0 / resolution
        if x <= xmax:
            painter.setPen(grid)
            painter.drawLine(at(x, 0), at(x, 1))
            if at(x, 0).x() - last_label < 46:
                continue  # no room for another label
            last_label = at(x, 0).x()
            painter.setPen(QPen(QColor("#444444")))
            painter.drawText(
                QRectF(last_label - 22, plot.bottom() + 3, 44, 16),
                Qt.AlignmentFlag.AlignCenter,
                f"{resolution:g} Å",
            )
    for level in marks:
        pen = QPen(QColor("#888888"), 1)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(at(0, level), at(xmax, level))
    for i, (label, x, y) in enumerate(curves):
        path = QPainterPath()
        for j in range(len(x)):
            point = at(float(x[j]), max(0.0, min(1.0, float(y[j]))))
            path.moveTo(point) if j == 0 else path.lineTo(point)
        painter.setPen(QPen(QColor(_PLOT_COLORS[i % len(_PLOT_COLORS)]), 1.8))
        painter.drawPath(path)
    # the legend, in the corner
    for i, (label, _x, _y) in enumerate(curves):
        y = plot.top() + 6 + 15 * i
        painter.setPen(QPen(QColor(_PLOT_COLORS[i % len(_PLOT_COLORS)]), 2))
        painter.drawLine(
            QPointF(plot.right() - 130, y + 7), QPointF(plot.right() - 112, y + 7)
        )
        painter.setPen(QPen(QColor("#222222")))
        painter.drawText(
            QRectF(plot.right() - 108, y, 104, 14),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            label,
        )
    painter.setPen(QPen(QColor("#222222")))
    painter.drawRect(plot)
    painter.end()
    return _qimage_to_array(image)


def _tick_step(span: float, fewest: int = 3) -> float:
    """The step between the major ticks of an axis: 1, 2 or 5 times a power of ten (so whole
    numbers or halves for the ranges that are usual), the largest one that gives at least
    ``fewest`` intervals, but not below 0.5 for a range of 1 or more."""
    import numpy as np

    if span <= 0:
        return 1.0
    power = 10.0 ** np.floor(np.log10(span / fewest))
    step = power / 2.0
    for factor in (5.0, 2.0, 1.0):
        if span / (factor * power) >= fewest:
            step = factor * power
            break
    # a range of one unit or more has whole numbers or halves as ticks, not 0.2 or 0.1
    return float(max(step, 0.5)) if span >= 1.0 else float(step)


def _histogram_limits(values, upper: tuple[float, float] | None = None):
    """The range of a histogram and the step of its ticks: ``(low, high, step)``.

    The range runs from the 0.5th to the 99.5th percentile, so that a few outliers do not squeeze
    the rest into one bar, and out to the ticks. With ``upper=(fraction, cap)`` the top of the
    range is the value that ``fraction`` of the values are above, but not more than ``cap``: for
    a resolution, where the micrographs of low resolution are of no interest.
    """
    import numpy as np

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    low, high = np.percentile(values, [0.5, 99.5])
    if upper is not None:
        high = min(float(upper[1]), float(np.percentile(values, 100 * (1 - upper[0]))))
        low = min(low, high)
    if high - low <= 1e-9 * max(
        abs(high), abs(low), 1.0
    ):  # all the values are the same
        centre = float(np.median(values))
        low, high = centre - 0.5, centre + 0.5
    step = _tick_step(high - low)
    low = float(np.floor(low / step) * step)
    top = float(np.ceil(high / step) * step)
    if upper is not None and top > upper[1]:
        top = float(upper[1])  # the cap is a multiple of the step
    return low, top, step


def _plot_histogram(
    values,
    xlabel: str,
    title: str,
    mark: float | None = None,
    upper: tuple[float, float] | None = None,
) -> "np.ndarray":  # noqa: F821
    """Draw the histogram of ``values``; ``mark`` is a value (the median) with a vertical line.

    The axis has whole numbers or halves as ticks, over the range of :func:`_histogram_limits`
    (``upper`` as there).
    """
    import numpy as np
    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen

    _require_application()
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("nothing to draw")
    low, high, step = _histogram_limits(values, upper)
    inside = values[(values >= low) & (values <= high)]
    # bins that end at the ticks: a few to each interval between them
    intervals = max(1, round((high - low) / step))
    bins = intervals * max(1, round(min(40, max(8, np.sqrt(len(inside)))) / intervals))
    counts, edges = np.histogram(inside, bins=bins, range=(low, high))
    top = int(counts.max()) or 1

    width, height = 460, 300
    image = QImage(width, height, QImage.Format.Format_RGB888)
    image.fill(QColor("white"))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    left, right, margin_top, bottom = 46, 12, 26, 44
    plot = QRectF(left, margin_top, width - left - right, height - margin_top - bottom)
    font = QFont()
    font.setPointSizeF(8.5)
    painter.setFont(font)
    painter.setPen(QPen(QColor("#222222")))
    painter.drawText(QRectF(0, 3, width, 18), Qt.AlignmentFlag.AlignCenter, title)

    def at(x, y):
        return QPointF(
            plot.left() + (x - low) / (high - low) * plot.width(),
            plot.bottom() - y / top * plot.height(),
        )

    grid, text = QPen(QColor("#dddddd")), QPen(QColor("#444444"))
    for tick in (0.0, 0.5, 1.0):
        painter.setPen(grid)
        painter.drawLine(at(low, tick * top), at(high, tick * top))
        painter.setPen(text)
        painter.drawText(
            QRectF(0, at(low, tick * top).y() - 8, left - 5, 16),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            f"{tick * top:.0f}",
        )
    painter.setPen(QPen(QColor("#1f77b4").darker(130), 1))
    painter.setBrush(QColor("#1f77b4"))
    for count, a, b in zip(counts, edges[:-1], edges[1:]):
        if count:
            painter.drawRect(QRectF(at(a, count), at(b, 0)))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    decimals = 0 if step >= 1 else int(np.ceil(-np.log10(step)))
    for i in range(intervals + 1):
        x = low + step * i
        painter.setPen(grid)
        painter.drawLine(at(x, 0), at(x, top))
        painter.setPen(text)
        painter.drawText(
            QRectF(at(x, 0).x() - 30, plot.bottom() + 3, 60, 16),
            Qt.AlignmentFlag.AlignCenter,
            f"{x:.{decimals}f}".rstrip("0").rstrip(".") if decimals else f"{x:.0f}",
        )
    painter.drawText(
        QRectF(0, plot.bottom() + 22, width, 16),
        Qt.AlignmentFlag.AlignCenter,
        xlabel,
    )
    if mark is not None and low <= mark <= high:
        pen = QPen(QColor("#d62728"), 1.5)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(at(mark, 0), at(mark, top))
    painter.setPen(QPen(QColor("#222222")))
    painter.drawRect(plot)
    painter.end()
    return _qimage_to_array(image)


def _plot_trajectories(paths, title: str = "Motion") -> "np.ndarray":  # noqa: F821
    """Draw the motion of micrographs: ``paths`` is ``[(label, x, y), ...]`` with the shifts of the
    frames in Å, each path starting at the origin (a dot) and ending at its last frame.
    """
    import numpy as np
    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen

    _require_application()
    size = 340
    image = QImage(size, size, QImage.Format.Format_RGB888)
    image.fill(QColor("white"))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    left, right, top, bottom = 40, 12, 26, 30
    plot = QRectF(left, top, size - left - right, size - top - bottom)
    reach = max(
        (max(np.abs(x).max(), np.abs(y).max()) for _l, x, y in paths), default=1.0
    )
    reach = float(reach) * 1.1 or 1.0
    font = QFont()
    font.setPointSizeF(8.5)
    painter.setFont(font)
    painter.setPen(QPen(QColor("#222222")))
    painter.drawText(QRectF(0, 3, size, 18), Qt.AlignmentFlag.AlignCenter, title)

    def at(x, y):  # equal scales in x and y, the origin in the middle
        return QPointF(
            plot.center().x() + x / reach * plot.width() / 2,
            plot.center().y() + y / reach * plot.height() / 2,
        )

    grid, text = QPen(QColor("#dddddd")), QPen(QColor("#444444"))
    step = 10 ** np.floor(np.log10(reach))
    step *= 5 if reach / step >= 5 else (2 if reach / step >= 2.4 else 1)
    for k in range(-int(reach / step), int(reach / step) + 1):
        value = k * step
        painter.setPen(grid)
        painter.drawLine(at(value, -reach), at(value, reach))
        painter.drawLine(at(-reach, value), at(reach, value))
        painter.setPen(text)
        painter.drawText(
            QRectF(at(value, 0).x() - 20, plot.bottom() + 3, 40, 14),
            Qt.AlignmentFlag.AlignCenter,
            f"{value:g}",
        )
        painter.drawText(
            QRectF(0, at(0, value).y() - 8, left - 5, 16),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            f"{value:g}",
        )
    painter.setPen(QPen(QColor("#222222")))
    painter.drawRect(plot)
    painter.setClipRect(plot)
    for i, (label, x, y) in enumerate(paths):
        color = QColor(_PLOT_COLORS[i % len(_PLOT_COLORS)])
        path = QPainterPath()
        for j in range(len(x)):
            point = at(float(x[j]), float(y[j]))
            path.moveTo(point) if j == 0 else path.lineTo(point)
        painter.setPen(QPen(color, 1.6))
        painter.drawPath(path)
        painter.setBrush(color)
        painter.drawEllipse(at(float(x[0]), float(y[0])), 2.5, 2.5)
        painter.drawEllipse(at(float(x[-1]), float(y[-1])), 1.5, 1.5)
        painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setClipping(False)
    for i, (label, _x, _y) in enumerate(paths):
        yy = plot.top() + 5 + 14 * i
        painter.setPen(QPen(QColor(_PLOT_COLORS[i % len(_PLOT_COLORS)]), 2))
        painter.drawLine(
            QPointF(plot.right() - 80, yy + 7), QPointF(plot.right() - 64, yy + 7)
        )
        painter.setPen(QPen(QColor("#222222")))
        painter.drawText(
            QRectF(plot.right() - 60, yy, 58, 14),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            label,
        )
    painter.drawText(
        QRectF(0, size - 16, size, 14),
        Qt.AlignmentFlag.AlignCenter,
        "shift in x and y (Å)",
    )
    painter.end()
    return _qimage_to_array(image)


def _plot_lines(curves, xlabel: str, title: str) -> "np.ndarray":  # noqa: F821
    """Draw lines: ``curves`` is ``[(label, x, y), ...]``; the y axis starts at 0."""
    import numpy as np
    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen

    _require_application()
    width, height = 460, 300
    image = QImage(width, height, QImage.Format.Format_RGB888)
    image.fill(QColor("white"))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    left, right, top, bottom = 46, 12, 26, 44
    plot = QRectF(left, top, width - left - right, height - top - bottom)
    xmin = min(float(np.min(x)) for _l, x, _y in curves)
    xmax = max(float(np.max(x)) for _l, x, _y in curves)
    ymax = max(float(np.max(y)) for _l, _x, y in curves) * 1.05 or 1.0
    font = QFont()
    font.setPointSizeF(8.5)
    painter.setFont(font)
    painter.setPen(QPen(QColor("#222222")))
    painter.drawText(QRectF(0, 3, width, 18), Qt.AlignmentFlag.AlignCenter, title)

    def at(x, y):
        return QPointF(
            plot.left() + (x - xmin) / ((xmax - xmin) or 1.0) * plot.width(),
            plot.bottom() - y / ymax * plot.height(),
        )

    grid, text = QPen(QColor("#dddddd")), QPen(QColor("#444444"))
    for k in range(5):
        value = ymax * k / 4
        painter.setPen(grid)
        painter.drawLine(at(xmin, value), at(xmax, value))
        painter.setPen(text)
        painter.drawText(
            QRectF(0, at(xmin, value).y() - 8, left - 5, 16),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            f"{value:.2g}",
        )
        x = xmin + (xmax - xmin) * k / 4
        painter.drawText(
            QRectF(at(x, 0).x() - 25, plot.bottom() + 3, 50, 16),
            Qt.AlignmentFlag.AlignCenter,
            f"{x:.3g}",
        )
    painter.drawText(
        QRectF(0, plot.bottom() + 22, width, 16), Qt.AlignmentFlag.AlignCenter, xlabel
    )
    for i, (label, x, y) in enumerate(curves):
        color = QColor(_PLOT_COLORS[i % len(_PLOT_COLORS)])
        path = QPainterPath()
        for j in range(len(x)):
            point = at(float(x[j]), float(y[j]))
            path.moveTo(point) if j == 0 else path.lineTo(point)
        painter.setPen(QPen(color, 1.8))
        painter.drawPath(path)
        yy = plot.top() + 6 + 15 * i
        painter.drawLine(
            QPointF(plot.right() - 84, yy + 7), QPointF(plot.right() - 66, yy + 7)
        )
        painter.setPen(QPen(QColor("#222222")))
        painter.drawText(
            QRectF(plot.right() - 62, yy, 60, 14),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            label,
        )
    painter.setPen(QPen(QColor("#222222")))
    painter.drawRect(plot)
    painter.end()
    return _qimage_to_array(image)


_VIRIDIS = [
    (68, 1, 84),
    (71, 44, 122),
    (59, 81, 139),
    (44, 113, 142),
    (33, 144, 141),
    (39, 173, 129),
    (92, 200, 99),
    (170, 220, 50),
    (253, 231, 37),
]


def view_distribution_image(rot, tilt) -> "np.ndarray":  # noqa: F821
    """The distribution of the views: the number of particles at each rot and tilt, as a picture.

    Rot (0 to 360°) runs along x and tilt (0 to 180°) down; the colour is the count (the scale runs
    from nothing to the most). Spherical symmetry is not looked for: a helical filament, seen from
    all round its axis, is a line at a tilt of 90°.

    Parameters
    ----------
    rot, tilt : array-like
        The angles of the particles, in degrees.

    Returns
    -------
    numpy.ndarray
        ``(h, w, 3)``, uint8.
    """
    import numpy as np
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen

    _require_application()
    rot = np.mod(np.asarray(rot, dtype=float) + 180.0, 360.0) - 180.0  # -180 to 180
    tilt = np.asarray(tilt, dtype=float)
    counts, _, _ = np.histogram2d(
        tilt, rot, bins=(45, 90), range=[[0, 180], [-180, 180]]
    )
    top = counts.max() or 1.0
    scaled = counts / top
    lut = np.array(_VIRIDIS, dtype=float)
    position = scaled * (len(lut) - 1)
    low = np.clip(position.astype(int), 0, len(lut) - 2)
    frac = (position - low)[..., None]
    rgb = (lut[low] * (1 - frac) + lut[low + 1] * frac).astype(np.uint8)
    rgb[counts == 0] = (245, 245, 245)

    width, height = 460, 300
    left, right, up, down = 46, 12, 24, 40
    image = QImage(width, height, QImage.Format.Format_RGB888)
    image.fill(QColor("white"))
    painter = QPainter(image)
    plot = QRectF(left, up, width - left - right, height - up - down)
    heat = QImage(
        rgb.tobytes(),
        rgb.shape[1],
        rgb.shape[0],
        rgb.shape[1] * 3,
        QImage.Format.Format_RGB888,
    )
    painter.drawImage(plot, heat)
    font = QFont()
    font.setPointSizeF(8.5)
    painter.setFont(font)
    painter.setPen(QPen(QColor("#222222")))
    painter.drawText(
        QRectF(0, 3, width, 16), Qt.AlignmentFlag.AlignCenter, "Views (rot, tilt)"
    )
    for tick in (-180, -90, 0, 90, 180):
        x = plot.left() + (tick + 180) / 360 * plot.width()
        painter.drawText(
            QRectF(x - 22, plot.bottom() + 3, 44, 16),
            Qt.AlignmentFlag.AlignCenter,
            f"{tick}°",
        )
    for tick in (0, 45, 90, 135, 180):
        y = plot.top() + tick / 180 * plot.height()
        painter.drawText(
            QRectF(0, y - 8, left - 5, 16),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            f"{tick}°",
        )
    painter.drawText(
        QRectF(plot.left(), plot.bottom() + 20, plot.width(), 16),
        Qt.AlignmentFlag.AlignCenter,
        f"rot (the colour is the number of particles in a bin, up to {int(top)})",
    )
    painter.drawRect(plot)
    painter.end()
    return _qimage_to_array(image)


def _particle_columns(path: Path, wanted: list[str], cap: int = VIEW_PARTICLES):
    """The columns ``wanted`` of the particle table of a RELION star file, for the first ``cap``
    particles, as arrays of text; None if the table or a column is not there.

    The file is read line by line in Python, not with ``numpy.loadtxt``: that holds the interpreter
    for the whole of a large table (half a minute for 300,000 particles of a file of 160 MB), and
    the window that is waiting for this in another thread stops answering. A loop in Python gives
    the other threads their turn every few milliseconds.
    """
    names: list[str] = []
    index: list[int] = []
    columns: list[list[str]] = []
    in_particles = False
    try:
        with open(path, errors="replace", buffering=1 << 20) as f:
            for line in f:
                if columns:  # in the rows
                    if len(columns[0]) >= cap:
                        break
                    parts = line.split()
                    if not parts:
                        continue
                    if parts[0].startswith("#"):
                        continue
                    if parts[0].startswith("data_"):
                        break
                    try:
                        for column, i in zip(columns, index):
                            column.append(parts[i])
                    except IndexError:
                        continue
                    continue
                text = line.strip()
                if text.startswith("data_"):
                    in_particles = text in ("data_particles", "data_images", "data_")
                    names = []
                elif in_particles and text.startswith("_"):
                    names.append(text.split()[0].lstrip("_"))
                elif in_particles and names and text and not text.startswith("#"):
                    if any(name not in names for name in wanted):
                        return None
                    index = [names.index(name) for name in wanted]
                    columns = [[] for _ in wanted]
                    parts = text.split()
                    try:
                        for column, i in zip(columns, index):
                            column.append(parts[i])
                    except IndexError:
                        return None
    except OSError:
        return None
    if not columns or not columns[0]:
        return None
    import numpy as np

    return {name: np.array(column) for name, column in zip(wanted, columns)}


def _particle_angles(path: Path, cap: int = VIEW_PARTICLES):
    """``(rot, tilt)`` of the first ``cap`` particles of a RELION data star file, or None."""
    columns = _particle_columns(path, ["rlnAngleRot", "rlnAngleTilt"], cap)
    if columns is None:
        return None
    return (
        columns["rlnAngleRot"].astype(float),
        columns["rlnAngleTilt"].astype(float),
    )


# ── RELION ───────────────────────────────────────────────────────────────────


def _class_distribution(model_star: Path) -> list[tuple[str, float]]:
    """``[(reference image, class distribution), ...]`` from a model star file."""
    table = _star_loop(model_star, "model_classes")
    if "rlnReferenceImage" not in table or "rlnClassDistribution" not in table:
        return []
    return list(
        zip(
            table["rlnReferenceImage"],
            (float(v) for v in table["rlnClassDistribution"]),
        )
    )


def _relion_class2d(folder: Path) -> dict | None:
    import mrcfile
    import numpy as np

    from .browser_parts import _montage

    stack = _last_numbered(folder, r"run_it(\d+)_classes\.mrcs")
    if stack is None:
        return None
    iteration = int(re.search(r"it(\d+)", stack.name).group(1))
    model = folder / f"run_it{iteration:03d}_model.star"
    shares: dict[int, float] = {}
    for reference, share in _class_distribution(model):
        try:
            shares[int(reference.split("@")[0]) - 1] = share
        except ValueError:
            continue
    with mrcfile.mmap(stack, mode="r", permissive=True) as mrc:
        data = mrc.data
        if data is None or data.ndim != 3:
            return None
        n = data.shape[0]
        order = sorted(range(n), key=lambda i: -shares.get(i, 0.0))[:MAX_TILES]
        frames = [np.asarray(data[i], dtype=np.float32) for i in order]
    image = _montage(frames, tile=TILE, columns=4 if len(frames) <= 12 else 6)
    caption = f"{n} classes at iteration {iteration}"
    if shares:
        top = ", ".join(f"{i + 1} ({shares[i] * 100:.1f}%)" for i in order[:6])
        caption += f"; the most populated first: class {top} …"
    return dict(
        title=f"Class averages, iteration {iteration}",
        panels=[_panel("Class averages", image, caption)],
    )


def _relion_classes_3d(folder: Path, what: str) -> dict | None:
    from .browser_parts import _montage

    last = _last_numbered(folder, r"run_it(\d+)_class001\.mrc")
    if last is None:
        return None
    iteration = int(re.search(r"it(\d+)", last.name).group(1))
    maps = sorted(folder.glob(f"run_it{iteration:03d}_class[0-9][0-9][0-9].mrc"))
    shares = {}
    for reference, share in _class_distribution(
        folder / f"run_it{iteration:03d}_model.star"
    ):
        m = re.search(r"class(\d+)", reference)
        if m:
            shares[int(m.group(1))] = share
    frames, labels = [], []
    for path in maps[:MAX_TILES]:
        section, _ = _central_slice(path)
        if section is not None:
            frames.append(section)
            number = int(re.search(r"class(\d+)", path.name).group(1))
            labels.append(
                f"{number} ({shares[number] * 100:.1f}%)"
                if number in shares
                else str(number)
            )
    if not frames:
        return None
    image = _montage(frames, tile=TILE + 16, columns=min(len(frames), 3))
    caption = (
        f"central sections of {len(frames)} class(es) at iteration {iteration}: "
        + ", ".join(labels)
    )
    return dict(
        title=f"{what}, iteration {iteration}",
        panels=[_panel("Central sections of the classes", image, caption)],
    )


def _relion_initial_model(folder: Path) -> dict | None:
    final = folder / "initial_model.mrc"
    if final.is_file():
        section, apix = _central_slice(final)
        if section is not None:
            return dict(
                title="Initial model",
                panels=[
                    _panel("Central section", _image(section), f"{apix:.2f} Å/pixel")
                ],
            )
    return _relion_classes_3d(folder, "Initial model")


def _views_caption(shown: int, total: int | None = None) -> str:
    """The caption of the views: how many particles are in the picture, and that they are the first
    ones only when there are more (``total`` is the number of particles when it is known).
    """
    if total is not None and total > shown:
        return f"the first {shown:,} of {total:,} particles"
    if total is None and shown >= VIEW_PARTICLES:
        return f"the first {shown:,} particles"
    return f"{shown:,} particles"


def _relion_refine3d(folder: Path) -> dict | None:
    panels, facts = [], []
    final = folder / "run_class001.mrc"
    if not final.is_file():
        last = _last_numbered(folder, r"run_it(\d+)_class001\.mrc")
        final = last if last is not None else final
    if final.is_file():
        section, apix = _central_slice(final)
        if section is not None:
            panels.append(
                _panel(
                    "Map, central section",
                    _image(section),
                    f"{final.name}, {apix:.2f} Å/pixel",
                )
            )
    model = folder / "run_model.star"
    if not model.is_file():
        last = _last_numbered(folder, r"run_it(\d+)_half1_model\.star")
        model = last if last is not None else model
    table = _star_loop(model, "model_class_1") if model.is_file() else {}
    if "rlnResolution" in table and "rlnGoldStandardFsc" in table:
        x, y = _floats(table["rlnResolution"]), _floats(table["rlnGoldStandardFsc"])
        resolution = fsc_resolution(x, y, 0.143)
        caption = (
            f"gold-standard FSC reaches 0.143 at {resolution:.2f} Å"
            if resolution
            else "gold-standard FSC"
        )
        if resolution:
            facts.append(("Resolution", f"{resolution:.2f} Å (FSC 0.143)"))
        panels.append(
            _panel(
                "FSC",
                _plot_fsc([("gold standard", x, y)], (0.143, 0.5), "FSC"),
                caption,
            )
        )
    data = folder / "run_data.star"
    if not data.is_file():
        last = _last_numbered(folder, r"run_it(\d+)_data\.star")
        data = last if last is not None else data
    angles = _particle_angles(data) if data.is_file() else None
    if angles is not None and len(angles[0]):
        panel = _plot_panel(
            "Views",
            lambda: view_distribution_image(*angles),
            _views_caption(len(angles[0])),
        )
        if panel:
            panels.append(panel)
    if not panels:
        return None
    return dict(title="Refinement", panels=panels, facts=facts)


def _relion_postprocess(folder: Path) -> dict | None:
    panels, facts = [], []
    star = folder / "postprocess.star"
    final = (
        _star_value(star, "general", "rlnFinalResolution") if star.is_file() else None
    )
    bfactor = (
        _star_value(star, "general", "rlnBfactorUsedForSharpening")
        if star.is_file()
        else None
    )
    if final:
        facts.append(("Resolution", f"{float(final):.2f} Å (FSC 0.143)"))
    if bfactor:
        facts.append(("Sharpening B-factor", f"{float(bfactor):.1f} Å²"))
    mrc = folder / "postprocess.mrc"
    if mrc.is_file():
        section, apix = _central_slice(mrc)
        if section is not None:
            panels.append(
                _panel(
                    "Sharpened map, central section",
                    _image(section),
                    f"{apix:.2f} Å/pixel",
                )
            )
    table = _star_loop(star, "fsc") if star.is_file() else {}
    if "rlnResolution" in table:
        x = _floats(table["rlnResolution"])
        curves = []
        for label, column in (
            ("corrected", "rlnFourierShellCorrelationCorrected"),
            ("unmasked", "rlnFourierShellCorrelationUnmaskedMaps"),
            ("masked", "rlnFourierShellCorrelationMaskedMaps"),
        ):
            if column in table:
                curves.append((label, x, _floats(table[column])))
        if curves:
            caption = (
                f"corrected FSC reaches 0.143 at {float(final):.2f} Å"
                if final
                else "FSC"
            )
            panels.append(
                _panel("FSC", _plot_fsc(curves, (0.143, 0.5), "FSC"), caption)
            )
    if not panels:
        return None
    return dict(title="Post-processing", panels=panels, facts=facts)


def _ctf_outcome(
    defocus, resolution, count: int, source: str, defocus_unit: str = "µm"
) -> dict | None:
    """The two histograms of a CTF estimation: the defocus and the resolution of the CTF fit.

    Parameters
    ----------
    defocus : array_like or None
        The defocus of each micrograph in µm (the mean of the two axes).
    resolution : array_like or None
        The resolution in Å to which the CTF fit reaches, per micrograph.
    count : int
        The number of micrographs.
    source : str
        The file the numbers come from, for the caption.
    """
    import numpy as np

    panels, facts = [], []
    if count:
        facts.append(("Micrographs", f"{count:,}"))
    if defocus is not None and len(defocus):
        median = float(np.median(defocus))
        facts.append(
            (
                "Defocus",
                f"median {median:.2f} µm ({np.min(defocus):.2f} to {np.max(defocus):.2f})",
            )
        )
        panel = _plot_panel(
            "Defocus",
            lambda: _plot_histogram(defocus, "defocus (µm)", "Defocus", median),
            f"{source}; the dashed line is the median, {median:.2f} µm",
        )
        if panel:
            panels.append(panel)
    if resolution is not None and len(resolution):
        median = float(np.median(resolution))
        facts.append(("Max resolution of the fit", f"median {median:.2f} Å"))
        # the micrographs of low resolution (the largest numbers) would use up the axis: it ends where
        # 5% of the micrographs are beyond, or at 10 Å
        limit = _histogram_limits(resolution, CTF_RESOLUTION_RANGE)[1]
        beyond = int((np.asarray(resolution) > limit).sum())
        note = f"; {beyond:,} beyond {limit:g} Å are not shown" if beyond else ""
        panel = _plot_panel(
            "Maximum resolution of the CTF fit",
            lambda: _plot_histogram(
                resolution,
                "resolution of the CTF fit (Å)",
                "CTF fit resolution",
                median,
                CTF_RESOLUTION_RANGE,
            ),
            f"the dashed line is the median, {median:.2f} Å; smaller is better{note}",
        )
        if panel:
            panels.append(panel)
    if not panels:
        return None
    return dict(title="CTF estimation", panels=panels, facts=facts)


def _relion_ctffind(folder: Path) -> dict | None:
    import numpy as np

    star = folder / "micrographs_ctf.star"
    if not star.is_file():
        return None
    table = {}
    for block in ("micrographs", ""):
        table = _star_loop(star, block)
        if "rlnDefocusU" in table:
            break
    if "rlnDefocusU" not in table:
        return None
    u = _floats(table["rlnDefocusU"])
    v = _floats(table["rlnDefocusV"]) if "rlnDefocusV" in table else u
    n = min(len(u), len(v))
    defocus = (u[:n] + v[:n]) / 2.0 / 1e4  # Å to µm
    resolution = None
    if "rlnCtfMaxResolution" in table:
        resolution = _floats(table["rlnCtfMaxResolution"])
        resolution = resolution[np.isfinite(resolution) & (resolution > 0)]
    return _ctf_outcome(defocus, resolution, n, star.name)


def _defocus_change(folder: Path):
    """The change of the defocus of each particle (Å) made by a CTF refinement, with the name of the
    file it is compared with; None when the input of the job or its particles are not found.

    The input particles are the ``fn_data`` of ``job.star``; the particles are matched by
    their image name, so that a job that dropped particles still compares the right ones.
    """
    import numpy as np

    refined = folder / "particles_ctf_refine.star"
    if not refined.is_file():
        return None
    given = None
    job_star = folder / "job.star"
    try:
        for line in job_star.read_text(errors="replace").splitlines():
            if line.split()[:1] == ["fn_data"]:
                given = line.split()[1]
    except (OSError, IndexError):
        return None
    if not given:
        return None
    before = folder.parent.parent / given
    if not before.is_file():
        return None
    wanted = ["rlnImageName", "rlnDefocusU", "rlnDefocusV"]
    new = _particle_columns(refined, wanted, DEFOCUS_PARTICLES)
    old = _particle_columns(before, wanted, DEFOCUS_PARTICLES)
    if new is None or old is None:
        return None
    mean_old = (old["rlnDefocusU"].astype(float) + old["rlnDefocusV"].astype(float)) / 2
    if np.array_equal(new["rlnImageName"], old["rlnImageName"]):
        index = np.arange(len(mean_old))
    else:
        where = {name: k for k, name in enumerate(old["rlnImageName"])}
        index = np.array([where.get(name, -1) for name in new["rlnImageName"]])
    keep = index >= 0
    mean_new = (new["rlnDefocusU"].astype(float) + new["rlnDefocusV"].astype(float)) / 2
    if not keep.any():
        return None
    return (
        mean_new[keep] - mean_old[index[keep]]
    ), before.parent.name + "/" + before.name


def _relion_ctfrefine(folder: Path) -> dict | None:
    import numpy as np

    panels, facts = [], []
    change = _defocus_change(folder)
    if change is not None and np.any(change[0] != 0):
        delta, source = change
        median = float(np.median(delta))
        facts.append(
            (
                "Defocus change",
                f"median {median:+.1f} Å, typically ±{np.percentile(np.abs(delta), 68):.1f} Å",
            )
        )
        panel = _plot_panel(
            "Change of the defocus",
            lambda: _plot_histogram(
                delta, "refined - before (Å)", "Change of the defocus", median
            ),
            f"{len(delta):,} particles compared with {source}; the dashed line is the median",
        )
        if panel:
            panels.append(panel)
    # the aberrations that were fitted: a phase map, per optics group (the first one is shown)
    maps = [
        ("Beam tilt, fitted", "beamtilt_delta-phase_iter-fit_*.mrc"),
        ("Higher-order aberrations, fitted", "aberr_delta-phase_iter-fit_*.mrc"),
        ("Magnification, x, fitted", "mag_disp_x_fit_*.mrc"),
        ("Magnification, y, fitted", "mag_disp_y_fit_*.mrc"),
    ]
    for title, pattern in maps:
        path = next(iter(sorted(folder.glob(pattern))), None)
        if path is None:
            continue
        section, _apix = _central_slice(path)
        if section is not None:
            panels.append(_panel(title, _image(section), path.name))
    if not panels:
        return None
    return dict(title="CTF refinement", panels=panels, facts=facts)


def _scan_table(
    path: Path, blocks: tuple[str, ...], columns: list[str], keep: int = 64
):
    """Count the rows of a loop of a star file and look at two columns of it, in one pass.

    For a table with a million particles: it reads the lines and keeps no more than the distinct
    values of the first column and at most ``keep`` rows, evenly spread over the table.

    Parameters
    ----------
    path : pathlib.Path
        The star file.
    blocks : tuple of str
        The names of the data block that may hold the table (``"particles"``, ``""``...).
    columns : list of str
        The columns to look at; the distinct values of the first one are collected.
    keep : int
        The most rows kept in ``samples``: the stride between the kept rows doubles as the table
        grows, so that they stay spread over all of it.

    Returns
    -------
    dict or None
        ``{"rows": int, "distinct": set, "samples": [tuple of text per column, ...]}``, or None
        when the table or a column is not there.
    """
    names: list[str] = []
    index: list[int] = []
    rows, distinct, samples, stride = 0, set(), [], 1
    state = "find"
    try:
        with open(path, errors="replace") as f:
            for line in f:
                text = line.strip()
                if state == "find":
                    if text.startswith("data_") and text[5:] in blocks:
                        state = "block"
                elif state == "block":
                    if text.startswith("data_"):
                        state = "find"
                    elif text == "loop_":
                        names = []
                        state = "columns"
                elif state == "columns":
                    if text.startswith("_"):
                        names.append(text.split()[0].lstrip("_"))
                    elif text and not text.startswith("#"):
                        if any(c not in names for c in columns):
                            return None
                        index = [names.index(c) for c in columns]
                        state = "rows"
                if state == "rows":
                    if not text or text.startswith("data_"):
                        break
                    if text.startswith("#"):
                        continue
                    parts = text.split()
                    distinct.add(parts[index[0]])
                    if rows % stride == 0:
                        samples.append(tuple(parts[i] for i in index))
                        if len(samples) >= 2 * keep:
                            samples = samples[::2]
                            stride *= 2
                    rows += 1
    except (OSError, IndexError):
        return None
    if not rows:
        return None
    return dict(rows=rows, distinct=distinct, samples=samples)


def _image_of_particle(spec: str, roots: list[Path]):
    """The image of one particle from a RELION ``000012@path/to/stack.mrcs`` name, or None."""
    import mrcfile
    import numpy as np

    number, _, name = spec.partition("@")
    if not name or not number.isdigit() or int(number) < 1:
        return None
    for root in roots:
        path = root / name
        if path.is_file():
            with mrcfile.mmap(path, mode="r", permissive=True) as mrc:
                data = mrc.data
                if data is None:
                    return None
                if data.ndim == 2:
                    return np.asarray(data, dtype=np.float32)
                if int(number) <= data.shape[0]:
                    return np.asarray(data[int(number) - 1], dtype=np.float32)
            return None
    return None


def _extraction_outcome(
    particles: int, micrographs: int, examples: list, source: str
) -> dict | None:
    """The numbers of an extraction and a few of the particles, as a montage.

    Parameters
    ----------
    particles, micrographs : int
        How many particles, and how many micrographs they were extracted from.
    examples : list of numpy.ndarray
        The images of a few particles spread over the job.
    source : str
        The table the numbers come from, for the caption.
    """
    facts = [("Particles", f"{particles:,}")]
    if micrographs:
        facts.append(("Micrographs", f"{micrographs:,}"))
        facts.append(
            ("Particles per micrograph", f"{particles / micrographs:.1f} on average")
        )
    panels = []
    if examples:
        from .browser_parts import _montage

        panels.append(
            _panel(
                "Examples",
                _montage(examples, tile=TILE, columns=4 if len(examples) <= 8 else 6),
                f"{len(examples)} of the {particles:,} particles, evenly spread over {source}",
            )
        )
    return dict(title="Particle extraction", panels=panels, facts=facts)


def _relion_extract(folder: Path) -> dict | None:
    star = folder / "particles.star"
    scan = _scan_table(
        star, ("particles", "images", ""), ["rlnMicrographName", "rlnImageName"]
    )
    if scan is None:
        return None
    # the stacks are named from the folder of the project: the job is two folders below it
    roots = [folder.parent.parent, folder]
    samples = [image for _mic, image in scan["samples"]]
    if len(samples) > EXAMPLES:
        step = len(samples) / EXAMPLES
        samples = [samples[int(i * step)] for i in range(EXAMPLES)]
    images = []
    for spec in samples:
        try:
            image = _image_of_particle(spec, roots)
        except Exception:
            image = None
        if image is not None:
            images.append(image)
    return _extraction_outcome(scan["rows"], len(scan["distinct"]), images, star.name)


def _relion_import(folder: Path) -> dict | None:
    """The number of movies, micrographs or particles that a RELION import made."""
    for name, blocks, column, what in (
        ("movies.star", ("movies",), "rlnMicrographMovieName", "Movies imported"),
        (
            "micrographs.star",
            ("micrographs",),
            "rlnMicrographName",
            "Micrographs imported",
        ),
        (
            "particles.star",
            ("particles", "images", ""),
            "rlnImageName",
            "Particles imported",
        ),
    ):
        scan = _scan_table(folder / name, blocks, [column], keep=1)
        if scan is not None:
            return dict(title="Import", panels=[], facts=[(what, f"{scan['rows']:,}")])
    return None


def _picks_outcome(
    what: str, count: int, micrographs: int, approximate: bool = False, extra=()
) -> dict:
    """The numbers of a picking job: how many were picked, in how many micrographs, and on average.

    Parameters
    ----------
    what : str
        What was picked, plural (``"Particles"``, ``"Filaments"``).
    count, micrographs : int
        The number picked and the number of micrographs they were picked in.
    approximate : bool
        True when ``count`` was extrapolated from some of the micrographs.
    extra : sequence of tuple
        More ``(label, value)`` facts, shown after the numbers.
    """
    mark = "about " if approximate else ""
    facts = [(what, f"{mark}{count:,}")]
    if micrographs:
        facts.append(("Micrographs", f"{micrographs:,}"))
        facts.append(
            (f"{what} per micrograph", f"{count / micrographs:.1f} on average")
        )
    facts.extend(extra)
    if approximate:
        facts.append(("", "estimated; the exact number is being counted"))
    return dict(title="Picking", panels=[], facts=facts, approximate=approximate)


def _job_option(folder: Path, name: str) -> str | None:
    """The value of an option in the ``job.star`` of a RELION job."""
    try:
        for line in (folder / "job.star").read_text(errors="replace").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == name:
                return " ".join(parts[1:])
    except OSError:
        pass
    return None


def _relion_picks(folder: Path, star_name: str, helical_option: str) -> dict | None:
    """Picks of a RELION ManualPick or AutoPick job: the number of picks in the coordinate file
    of each micrograph (read from all of them up to ``MAX_COORDINATE_FILES``)."""
    listing = _star_loop(folder / star_name, "coordinate_files")
    files = listing.get("rlnMicrographCoordinates")
    if not files:
        return None
    total = len(files)
    chosen = files
    if total > MAX_COORDINATE_FILES:
        step = total / MAX_COORDINATE_FILES
        chosen = [files[int(i * step)] for i in range(MAX_COORDINATE_FILES)]
    roots = [folder.parent.parent, folder]
    rows, found = 0, 0
    for name in chosen:
        for root in roots:
            path = root / name
            if path.is_file():
                scan = _scan_table(path, ("",), ["rlnCoordinateX"], keep=1)
                rows += scan["rows"] if scan else 0
                found += 1
                break
    if not found:
        return None  # the coordinate files are not here: nothing to count
    approximate = len(chosen) < total
    if approximate:
        rows = round(rows * total / found)
    helical = _job_option(folder, helical_option) == "Yes"
    if helical and star_name == "manualpick.star":
        # a filament is picked by its start and its end: two coordinates
        return _picks_outcome("Filaments", rows // 2, total, approximate)
    return _picks_outcome(
        "Helical segments" if helical else "Particles", rows, total, approximate
    )


def _relion_manualpick(folder: Path) -> dict | None:
    return _relion_picks(folder, "manualpick.star", "do_startend")


def _relion_autopick(folder: Path) -> dict | None:
    return _relion_picks(folder, "autopick.star", "do_pick_helical_segments")


def _frame_shifts(path: Path):
    """The shift of each frame of a micrograph (Å), from the ``global_shift`` table of the star file
    that RELION's motion correction writes for it, as ``(x, y)`` arrays; None when it is not there.
    """
    table = _star_loop(path, "global_shift")
    if "rlnMicrographShiftX" not in table or "rlnMicrographShiftY" not in table:
        return None
    x, y = _floats(table["rlnMicrographShiftX"]), _floats(table["rlnMicrographShiftY"])
    n = min(len(x), len(y))
    return (x[:n], y[:n]) if n > 1 else None


def _relion_motioncorr(folder: Path) -> dict | None:
    import numpy as np

    star = folder / "corrected_micrographs.star"
    table = _star_loop(star, "micrographs")
    if "rlnAccumMotionTotal" not in table:
        return None
    total = _floats(table["rlnAccumMotionTotal"])
    if not len(total):
        return None
    median = float(np.median(total))
    facts = [
        ("Micrographs", f"{len(total):,}"),
        (
            "Total motion",
            f"median {median:.1f} Å ({np.min(total):.1f} to {np.max(total):.1f})",
        ),
    ]
    panels = []
    panel = _plot_panel(
        "Total motion",
        lambda: _plot_histogram(
            total, "length of the motion trajectory (Å)", "Total motion", median
        ),
        f"{len(total):,} micrographs; the dashed line is the median, {median:.1f} Å",
    )
    if panel:
        panels.append(panel)
    # a few trajectories, from the smallest total motion to the largest
    metadata = table.get("rlnMicrographMetadata", [])
    paths = []
    if len(metadata) == len(total):
        order = np.argsort(total)
        chosen = [
            order[int(round(q * (len(order) - 1)))] for q in np.linspace(0, 1, MOTIONS)
        ]
        for k in dict.fromkeys(chosen):
            for root in (folder.parent.parent, folder):
                shifts = (
                    _frame_shifts(root / metadata[k])
                    if (root / metadata[k]).is_file()
                    else None
                )
                if shifts is not None:
                    paths.append((f"{total[k]:.1f} Å", *shifts))
                    break
    if paths:
        panel = _plot_panel(
            "Examples of motion",
            lambda: _plot_trajectories(paths, "Motion of some micrographs"),
            f"{len(paths)} micrographs, from the least to the most motion; "
            "the large dot is the first frame",
        )
        if panel:
            panels.append(panel)
    return dict(title="Motion correction", panels=panels, facts=facts)


def frame_contributions(frames, bfactor, intercept, resolutions):
    """How much each movie frame contributes to the polished particles at some resolutions.

    The weight of a frame at the spatial frequency ``s`` is ``exp(intercept + bfactor * s**2 / 4)``
    (the scale factor and the B-factor that Bayesian polishing fits for the frame); the weights of
    the frames are divided by their sum, at each resolution.

    Parameters
    ----------
    frames : array_like
        The frame numbers.
    bfactor, intercept : array_like
        ``rlnBfactorUsedForSharpening`` (Å², negative for a frame that is damaged) and
        ``rlnFittedInterceptGuinierPlot`` (the logarithm of the scale) of each frame.
    resolutions : sequence of float
        The resolutions in Å.

    Returns
    -------
    list of tuple
        ``[(resolution, frames, share of each frame), ...]``.
    """
    import numpy as np

    frames = np.asarray(frames, dtype=float)
    bfactor = np.asarray(bfactor, dtype=float)
    intercept = np.asarray(intercept, dtype=float)
    curves = []
    for resolution in resolutions:
        weight = np.exp(intercept + bfactor / (4.0 * resolution**2))
        total = float(weight.sum())
        curves.append((resolution, frames, weight / total if total > 0 else weight))
    return curves


def _contribution_resolutions(nyquist: float = 0.0) -> list[float]:
    """The resolutions (Å) at which the contribution of the frames is drawn: some of 10 to 1.5 Å,
    none finer than the Nyquist resolution (``nyquist``, Å) of the images when it is known.
    """
    found = [r for r in (10.0, 5.0, 3.0, 2.0, 1.5) if r >= nyquist]
    return found if len(found) >= 2 else [10.0, 5.0]


def _frame_panel(curves, what: str) -> dict | None:
    """The panel with the contribution of each movie frame: ``curves`` is ``[(label, frames,
    share), ...]``."""
    return _plot_panel(
        "Contribution of the frames",
        lambda: _plot_lines(
            curves, "movie frame", "Contribution of each frame, at some resolutions"
        ),
        f"{len(curves[0][1])} frames; {what}",
    )


def _relion_polish(folder: Path) -> dict | None:
    table = _star_loop(folder / "bfactors.star", "perframe_bfactors")
    needed = (
        "rlnMovieFrameNumber",
        "rlnBfactorUsedForSharpening",
        "rlnFittedInterceptGuinierPlot",
    )
    if any(name not in table for name in needed):
        return None
    frames, bfactor, intercept = (_floats(table[name]) for name in needed)
    if len(frames) < 2:
        return None
    # no finer than the Nyquist resolution of the particles, when their pixel size is known
    nyquist = 0.0
    optics = _star_loop(folder / "shiny.star", "optics")
    for name in ("rlnImagePixelSize", "rlnMicrographPixelSize"):
        if optics.get(name):
            nyquist = 2.0 * float(optics[name][0])
            break
    curves = [
        (f"{r:g} Å", x, y)
        for r, x, y in frame_contributions(
            frames, bfactor, intercept, _contribution_resolutions(nyquist)
        )
    ]
    panel = _frame_panel(
        curves,
        "the share of each frame in the polished particles at the resolutions shown, from the "
        "fitted scale and B-factor of the frame",
    )
    if panel is None:
        return None
    return dict(
        title="Bayesian polishing",
        panels=[panel],
        facts=[("Frames", f"{len(frames)}")],
    )


def _share(selected: int, total: int | None) -> str:
    """``12,000 of 70,000 (17.1%)``, or just ``12,000`` when the total is not known."""
    if not total:
        return f"{selected:,}"
    return f"{selected:,} of {total:,} ({selected / total * 100:.1f}%)"


def _selection_outcome(
    kind: str,
    classes: tuple[int, int | None] | None,
    particles: tuple[int, int | None],
    panels=(),
) -> dict:
    """The numbers of a selection: the classes and the particles that were kept, of all of them.

    Parameters
    ----------
    kind : str
        ``"2D"``, ``"3D"`` or ``""`` (the classes of a selection of particles, say).
    classes : tuple or None
        ``(selected, all)`` classes; ``all`` may be None.
    particles : tuple
        ``(selected, all)`` particles; ``all`` may be None.
    panels : sequence of dict
        Pictures to show.
    """
    facts = []
    if classes is not None:
        label = f"{kind} classes selected" if kind else "Classes selected"
        facts.append(
            (
                label,
                (
                    f"{classes[0]:,} of {classes[1]:,}"
                    if classes[1]
                    else f"{classes[0]:,}"
                ),
            )
        )
    facts.append(("Particles selected", _share(*particles)))
    return dict(title="Selection", panels=list(panels), facts=facts)


def _relion_select(folder: Path) -> dict | None:
    """What a RELION Select job kept: classes and particles, of those of the classification."""
    import numpy as np

    particles = folder / "particles.star"
    blocks = ("particles", "images", "")
    scan = _scan_table(particles, blocks, ["rlnClassNumber"], keep=1)
    if scan is None:
        scan = _scan_table(particles, blocks, ["rlnImageName"], keep=1)
        if scan is None:
            return None
        scan["distinct"] = (
            set()
        )  # no class numbers: the classes cannot be counted from it
    roots = [folder.parent.parent, folder]

    def option(name: str) -> Path | None:
        value = (_job_option(folder, name) or "").strip().strip('"')
        for root in roots:
            if value and (root / value).is_file():
                return root / value
        return None

    kind, all_classes, all_particles = "", None, None
    optimiser = option("fn_model")
    if optimiser is not None:
        kind = (
            "2D"
            if "Class2D" in str(optimiser)
            else "3D" if "Class3D" in str(optimiser) else ""
        )
        stem = optimiser.name.replace("_optimiser.star", "")
        model = _star_loop(optimiser.with_name(f"{stem}_model.star"), "model_classes")
        all_classes = len(model.get("rlnReferenceImage", [])) or None
        data = _scan_table(
            optimiser.with_name(f"{stem}_data.star"), blocks, ["rlnClassNumber"], keep=1
        )
        all_particles = data["rows"] if data else None
    else:
        given = option("fn_data")
        if given is not None:
            data = _scan_table(given, blocks, ["rlnImageName"], keep=1)
            all_particles = data["rows"] if data else None
    classes = None
    averages = _scan_table(
        folder / "class_averages.star", ("",), ["rlnReferenceImage"], keep=1
    )
    if averages is not None:
        classes = (averages["rows"], all_classes)
    elif scan["distinct"]:
        classes = (len(scan["distinct"]), all_classes)
    panels = []
    stack = folder / "class_averages.mrcs"
    if kind == "2D" and stack.is_file():
        try:
            import mrcfile

            from .browser_parts import _montage

            with mrcfile.mmap(stack, mode="r", permissive=True) as mrc:
                data = mrc.data
                if data is not None and data.ndim == 3:
                    frames = [
                        np.asarray(data[i], dtype=np.float32)
                        for i in range(min(len(data), MAX_TILES))
                    ]
                    panels.append(
                        _panel(
                            "Selected class averages",
                            _montage(
                                frames, tile=TILE, columns=4 if len(frames) <= 12 else 6
                            ),
                            f"{len(data)} selected classes",
                        )
                    )
        except Exception:
            logger.debug("cannot read %s", stack, exc_info=True)
    return _selection_outcome(kind, classes, (scan["rows"], all_particles), panels)


def _relion_reconstruct(folder: Path) -> dict | None:
    """The central section of the map of a 3D reconstruction (the whole map, or a half)."""
    path = folder / "run_class001.mrc"
    if not path.is_file():
        path = next(iter(sorted(folder.glob("run_class001*.mrc"))), path)
    if not path.is_file():
        return None
    section, apix = _central_slice(path)
    if section is None:
        return None
    return dict(
        title="Reconstruction",
        panels=[
            _panel(
                "Map, central section",
                _image(section),
                f"{path.name}, {apix:.2f} Å/pixel",
            )
        ],
    )


def _relion_localres(folder: Path) -> dict | None:
    """The local resolution: a central section of the map and the histogram of its values."""
    import mrcfile
    import numpy as np

    path = folder / "relion_locres.mrc"
    if not path.is_file():
        return None
    with mrcfile.mmap(path, mode="r", permissive=True) as mrc:
        data = mrc.data
        if data is None or data.ndim != 3:
            return None
        section = np.asarray(data[data.shape[0] // 2], dtype=np.float32)
        values = np.asarray(data, dtype=np.float32).ravel()
        apix = float(mrc.voxel_size.x)
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values):
        return None
    median = float(np.median(values))
    facts = [
        (
            "Local resolution",
            f"median {median:.2f} Å ({values.min():.2f} to {values.max():.2f})",
        )
    ]
    panels = [
        _panel(
            "Local resolution, central section",
            _image(section),
            f"{path.name}, {apix:.2f} Å/pixel; darker is better resolved",
        )
    ]
    panel = _plot_panel(
        "Histogram of the local resolution",
        lambda: _plot_histogram(
            values, "local resolution (Å)", "Local resolution", median
        ),
        f"the dashed line is the median, {median:.2f} Å",
    )
    if panel:
        panels.append(panel)
    return dict(title="Local resolution", panels=panels, facts=facts)


def _relion_modelangelo(folder: Path) -> dict | None:
    """The size of the model that ModelAngelo built: chains, residues, atoms."""
    models = [
        p for p in sorted(folder.glob("*.cif")) if not p.name.endswith("_raw.cif")
    ]
    if not models:
        return None
    columns: list[str] = []
    atoms, residues, chains = 0, set(), set()
    with open(models[0], errors="replace") as f:
        for line in f:
            if line.startswith("_atom_site."):
                columns.append(line.strip()[len("_atom_site.") :])
            elif line.startswith(("ATOM", "HETATM")) and columns:
                parts = line.split()
                try:
                    chain = parts[columns.index("label_asym_id")]
                    residue = parts[columns.index("label_seq_id")]
                except (ValueError, IndexError):
                    continue
                atoms += 1
                chains.add(chain)
                residues.add((chain, residue))
    if not atoms:
        return None
    return dict(
        title="ModelAngelo",
        panels=[],
        facts=[
            ("Model", models[0].name),
            ("Chains", f"{len(chains)}"),
            ("Residues", f"{len(residues):,}"),
            ("Atoms", f"{atoms:,}"),
        ],
    )


def _relion_mask(folder: Path) -> dict | None:
    mask = folder / "mask.mrc"
    if not mask.is_file():
        return None
    section, apix = _central_slice(mask)
    if section is None:
        return None
    return dict(
        title="Mask",
        panels=[
            _panel("Mask, central section", _image(section), f"{apix:.2f} Å/pixel")
        ],
    )


_RELION = {
    "Class2D": _relion_class2d,
    "Class3D": lambda folder: _relion_classes_3d(folder, "3D classes"),
    "InitialModel": _relion_initial_model,
    "Refine3D": _relion_refine3d,
    "PostProcess": _relion_postprocess,
    "MaskCreate": _relion_mask,
    "Reconstruct3D": _relion_reconstruct,
    "LocalRes": _relion_localres,
    "ModelAngelo": _relion_modelangelo,
    "CtfFind": _relion_ctffind,
    "CtfRefine": _relion_ctfrefine,
    "MotionCorr": _relion_motioncorr,
    "Polish": _relion_polish,
    "Extract": _relion_extract,
    "ManualPick": _relion_manualpick,
    "AutoPick": _relion_autopick,
    "Import": _relion_import,
    "Select": _relion_select,
}


# ── CryoSPARC ────────────────────────────────────────────────────────────────


def _cs_class2d(folder: Path, uid: str) -> dict | None:
    import mrcfile
    import numpy as np

    from .browser_parts import _montage

    last = _last_numbered(folder, rf"{re.escape(uid)}_(\d+)_class_averages\.mrc")
    if last is None:
        return None
    iteration = int(re.search(r"_(\d+)_class_averages", last.name).group(1))
    counts: dict[int, int] = {}
    particles = folder / f"{uid}_{iteration:03d}_particles.cs"
    if particles.is_file():
        try:
            table = np.load(particles, allow_pickle=True)
            if "alignments2D/class" in (table.dtype.names or ()):
                classes, number = np.unique(
                    table["alignments2D/class"], return_counts=True
                )
                counts = dict(zip(classes.tolist(), number.tolist()))
        except Exception:
            counts = {}
    with mrcfile.mmap(last, mode="r", permissive=True) as mrc:
        data = mrc.data
        if data is None or data.ndim != 3:
            return None
        n = data.shape[0]
        order = sorted(range(n), key=lambda i: -counts.get(i, 0))[:MAX_TILES]
        frames = [np.asarray(data[i], dtype=np.float32) for i in order]
    image = _montage(frames, tile=TILE, columns=4 if len(frames) <= 12 else 6)
    caption = f"{n} classes at iteration {iteration}"
    if counts:
        total = sum(counts.values()) or 1
        top = ", ".join(
            f"{i + 1} ({counts.get(i, 0) / total * 100:.1f}%)" for i in order[:6]
        )
        caption += f"; the most populated first: class {top} …"
    return dict(
        title=f"Class averages, iteration {iteration}",
        panels=[_panel("Class averages", image, caption)],
    )


def _cs_table(path: Path):
    """A ``.cs`` file as a structured array (not read into memory when it can be mapped), or None."""
    import numpy as np

    try:
        return np.load(path, mmap_mode="r")
    except ValueError:  # the table holds Python objects: it cannot be mapped
        pass
    except Exception:
        return None
    try:
        return np.load(path, allow_pickle=True)
    except Exception:
        return None


def _cs_output(folder: Path, uid: str, *names: str) -> Path | None:
    """The first of the output tables of a CryoSPARC job that exists, by name: ``name``,
    ``{uid}_name``, or the numbered ``{uid}_NNN_name`` of an iteration (the last one).
    """
    for name in names:
        for candidate in (folder / name, folder / f"{uid}_{name}"):
            if candidate.is_file():
                return candidate
        last = _last_numbered(
            folder, rf"{re.escape(uid)}_(\d+)_{re.escape(Path(name).stem)}\.cs"
        )
        if last is not None:
            return last
    return None


def _cs_particle_file(folder: Path, uid: str) -> Path | None:
    """The table of the particles a job made (CryoSPARC names it for what the job did)."""
    return _cs_output(
        folder, uid, "particles.cs", "extracted_particles.cs", "picked_particles.cs"
    )


def _cs_column(tables, name: str):
    """The column ``name`` from the first of the tables that has it. A job writes only the fields it
    made in its own table and leaves the others to the ``passthrough`` table (same rows).
    """
    for table in tables:
        if table is not None and name in (table.dtype.names or ()):
            return table[name]
    return None


def _cs_passthrough(folder: Path, uid: str, what: str = "particles"):
    path = folder / f"{uid}_passthrough_{what}.cs"
    return _cs_table(path) if path.is_file() else None


def _cs_text(value) -> str:
    """A string from a ``.cs`` cell (bytes), without the ``>`` that marks a path in the project."""
    text = value.decode(errors="replace") if isinstance(value, bytes) else str(value)
    return text.lstrip(">")


def _cs_rows(folder: Path, uid: str, *names: str) -> int | None:
    """The number of rows of the first of these tables (the micrographs of a job, say)."""
    path = _cs_output(folder, uid, *names)
    table = _cs_table(path) if path is not None else None
    return len(table) if table is not None else None


def _cs_distinct(
    columns, count: int, row_bytes: int, exact: bool = False
) -> tuple[int, bool]:
    """How many distinct values there are in the columns taken together (a pair of uids, say), and
    whether that is exact.

    Only the columns are read, but they are spread over the whole file (the rows are stored one
    after the other, 300 bytes or so each): a table of more than ``EXACT_BYTES`` is read from the
    file system in full, about a minute for a few GB. Such a table is counted from a few blocks
    spread over it, scaled to all the rows (the rows of a filament follow each other, so only the
    filaments cut at the ends of a block are counted twice), unless ``exact`` asks for all of it,
    which is read in chunks. The outcome of the job is kept in the cache (:func:`build_outcome`), so
    that the exact number is not counted again.

    Parameters
    ----------
    columns : list of array_like
        The columns, mapped from the file.
    count : int
        The number of rows.
    row_bytes : int
        The bytes of a row of the file (what reading a column costs).
    exact : bool
        Count all the rows, however long it takes.

    Returns
    -------
    tuple of (int, bool)
        The number, and whether it is exact.
    """
    import numpy as np

    def distinct(rows) -> np.ndarray:
        stacked = np.stack([np.asarray(c[rows]).astype(np.uint64) for c in columns])
        return np.unique(stacked, axis=1)

    if count * row_bytes <= EXACT_BYTES:
        return int(distinct(slice(0, count)).shape[1]), True
    if exact:
        chunk = max(1, EXACT_BYTES // max(row_bytes, 1))
        parts = [
            distinct(slice(start, min(start + chunk, count)))
            for start in range(0, count, chunk)
        ]
        found = int(np.unique(np.concatenate(parts, axis=1), axis=1).shape[1])
        return found, True
    block = BLOCK_ROWS
    starts = np.linspace(0, count - block, SAMPLE_BLOCKS).astype(int)
    found = sum(distinct(slice(int(a), int(a) + block)).shape[1] for a in starts)
    return round(found / (block * len(starts)) * count), False


def _cs_volume(folder: Path, uid: str) -> dict | None:
    import numpy as np

    panels = []
    path = folder / f"{uid}_volume_map.mrc"
    if not path.is_file():
        path = _last_numbered(folder, rf"{re.escape(uid)}_(\d+)_volume_map\.mrc")
    if path is not None and path.is_file():
        section, apix = _central_slice(path)
        if section is not None:
            panels.append(
                _panel(
                    "Map, central section",
                    _image(section),
                    f"{path.name}, {apix:.2f} Å/pixel",
                )
            )
    particles = _cs_output(folder, uid, "particles.cs")
    table = _cs_table(particles) if particles is not None else None
    if table is not None and "alignments3D/pose" in (table.dtype.names or ()):
        try:
            from scipy.spatial.transform import Rotation

            poses = np.asarray(table["alignments3D/pose"][:VIEW_PARTICLES], dtype=float)
            euler = Rotation.from_rotvec(poses).as_euler("ZYZ", degrees=True)
            panel = _plot_panel(
                "Views",
                lambda: view_distribution_image(euler[:, 0], euler[:, 1]),
                _views_caption(len(euler), len(table)),
            )
            if panel:
                panels.append(panel)
        except Exception:
            logger.debug("no view distribution for %s", folder, exc_info=True)
    if not panels:
        return None
    return dict(title="Reconstruction", panels=panels)


def _cs_class3d(folder: Path, uid: str) -> dict | None:
    """The central section of the map of each class of a 3D classification, with its share of the
    particles (from the table of the particles of the class)."""
    from .browser_parts import _montage

    pattern = re.compile(rf"{re.escape(uid)}_class_(\d+)_(\d+|final)_volume\.mrc")
    found = [
        (m, p)
        for p in folder.glob(f"{uid}_class_*_volume.mrc")
        if (m := pattern.fullmatch(p.name))
    ]
    if not found:
        return None
    iterations = {m.group(2) for m, _p in found}
    iteration = "final" if "final" in iterations else max(iterations, key=int)
    maps = sorted((int(m.group(1)), p) for m, p in found if m.group(2) == iteration)[
        :MAX_TILES
    ]
    sizes = {}
    for number, _path in maps:
        table = _cs_table(folder / f"{uid}_class_{number:02d}_{iteration}_particles.cs")
        if table is not None:
            sizes[number] = len(table)
    total = sum(sizes.values())
    frames, labels = [], []
    for number, path in maps:
        section, _apix = _central_slice(path)
        if section is None:
            continue
        frames.append(section)
        share = (
            f" ({sizes[number] / total * 100:.1f}%)"
            if total and number in sizes
            else ""
        )
        labels.append(f"{number + 1}{share}")
    if not frames:
        return None
    image = _montage(frames, tile=TILE + 16, columns=min(len(frames), 3))
    caption = f"central sections of {len(frames)} class(es): " + ", ".join(labels)
    facts = [("Particles", f"{total:,}")] if total else []
    return dict(
        title=(
            "3D classes"
            if iteration == "final"
            else f"3D classes, iteration {int(iteration)}"
        ),
        panels=[_panel("Central sections of the classes", image, caption)],
        facts=facts,
    )


def _cs_ctf(folder: Path, uid: str) -> dict | None:
    """The CTF fits of a CryoSPARC CTF estimation, from the exposures table it wrote."""
    import numpy as np

    path = _cs_output(folder, uid, "exposures_ctf_estimated.cs", "exposures.cs")
    table = _cs_table(path) if path is not None else None
    if table is None or "ctf/df1_A" not in (table.dtype.names or ()):
        return None
    df1 = np.asarray(table["ctf/df1_A"], dtype=float)
    df2 = _cs_column([table], "ctf/df2_A")
    df2 = np.asarray(df2, dtype=float) if df2 is not None else df1
    defocus = (df1 + df2) / 2.0 / 1e4
    resolution = _cs_column([table], "ctf/ctf_fit_to_A")
    if resolution is not None:
        resolution = np.asarray(resolution, dtype=float)
        resolution = resolution[np.isfinite(resolution) & (resolution > 0)]
    return _ctf_outcome(defocus, resolution, len(table), path.name)


def _cs_motion(folder: Path, uid: str) -> dict | None:
    """The motion of a CryoSPARC motion correction: the length of the rigid trajectory of each
    micrograph and the trajectories of a few. The trajectories are ``.npy`` files named by the table
    (shifts in pixels of ``rigid_motion/psize_A``)."""
    import numpy as np

    path = _cs_output(folder, uid, "micrographs_rigid_aligned.cs")
    table = _cs_table(path) if path is not None else None
    if table is None or "rigid_motion/path" not in (table.dtype.names or ()):
        return None
    count = len(table)
    wanted = min(count, MAX_TRAJECTORIES)
    rows = list(dict.fromkeys(int(i * count / wanted) for i in range(wanted)))
    psize = table["rigid_motion/psize_A"]
    names = table["rigid_motion/path"]
    index = table["rigid_motion/idx"]

    def read(k: int):
        for root in (folder.parent, folder):
            trajectory = root / _cs_text(names[k])
            if trajectory.is_file():
                try:
                    xy = np.asarray(np.load(trajectory)[int(index[k])], dtype=float)
                except Exception:
                    return None
                return xy * float(psize[k]) if xy.ndim == 2 and len(xy) > 1 else None
        return None

    from concurrent.futures import ThreadPoolExecutor

    # the files are small and far away (a network file system): many at a time
    with ThreadPoolExecutor(16) as pool:
        trajectories = list(pool.map(read, rows))
    totals, shifts = [], []
    for xy in trajectories:
        if xy is not None:
            totals.append(float(np.hypot(*np.diff(xy, axis=0).T).sum()))
            shifts.append((xy[:, 0] - xy[0, 0], xy[:, 1] - xy[0, 1]))
    if not totals:
        return None
    totals_array = np.asarray(totals)
    median = float(np.median(totals_array))
    facts = [
        ("Micrographs", f"{count:,}"),
        (
            "Total motion",
            f"median {median:.1f} Å ({totals_array.min():.1f} to {totals_array.max():.1f})",
        ),
    ]
    panels = []
    panel = _plot_panel(
        "Total motion",
        lambda: _plot_histogram(
            totals_array, "length of the motion trajectory (Å)", "Total motion", median
        ),
        f"{len(totals):,} of {count:,} micrographs; the dashed line is the median, {median:.1f} Å",
    )
    if panel:
        panels.append(panel)
    order = np.argsort(totals_array)
    chosen = dict.fromkeys(
        int(order[int(round(q * (len(order) - 1)))]) for q in np.linspace(0, 1, MOTIONS)
    )
    paths = [(f"{totals[k]:.1f} Å", *shifts[k]) for k in chosen]
    panel = _plot_panel(
        "Examples of motion",
        lambda: _plot_trajectories(paths, "Motion of some micrographs"),
        f"{len(paths)} micrographs, from the least to the most motion; the large dot is the first frame",
    )
    if panel:
        panels.append(panel)
    return dict(title="Motion correction", panels=panels, facts=facts)


def _cs_picks(folder: Path, uid: str, exact: bool = False) -> dict | None:
    """The picks of a CryoSPARC picking job: particles, or filaments and their segments.

    The filaments of a table of many GB are an estimate (``outcome["approximate"]``) unless
    ``exact`` is set, which reads all of it.
    """
    path = _cs_particle_file(folder, uid)
    table = _cs_table(path) if path is not None else None
    if table is None or not len(table):
        return None
    count = len(table)
    tables = [table, _cs_passthrough(folder, uid)]
    micrograph = _cs_column(tables, "location/micrograph_uid")
    micrographs = _cs_rows(folder, uid, "picked_micrographs.cs", "micrographs.cs") or 0
    row_bytes = sum(t.dtype.itemsize for t in tables if t is not None)
    if micrograph is not None and count * row_bytes <= EXACT_BYTES:
        micrographs, _ = _cs_distinct([micrograph], count, row_bytes)
    filament = _cs_column(tables, "filament/filament_uid")
    if filament is not None and micrograph is not None:
        filaments, known = _cs_distinct([micrograph, filament], count, row_bytes, exact)
        return _picks_outcome(
            "Filaments",
            filaments,
            micrographs,
            approximate=not known,
            extra=[("Segments", f"{count:,}")],
        )
    return _picks_outcome("Particles", count, micrographs)


def _cs_extract(folder: Path, uid: str) -> dict | None:
    """The numbers of a CryoSPARC extraction, and a few of the particles."""
    import mrcfile
    import numpy as np

    path = _cs_particle_file(folder, uid)
    table = _cs_table(path) if path is not None else None
    if table is None or not len(table):
        return None
    count = len(table)
    micrographs = _cs_rows(folder, uid, "picked_micrographs.cs", "micrographs.cs") or 0
    row_bytes = table.dtype.itemsize
    if (
        "location/micrograph_uid" in (table.dtype.names or ())
        and count * row_bytes <= EXACT_BYTES
    ):
        micrographs, _ = _cs_distinct(
            [table["location/micrograph_uid"]], count, row_bytes
        )
    images = []
    if "blob/path" in (table.dtype.names or ()) and "blob/idx" in table.dtype.names:
        for row in np.linspace(0, count - 1, min(EXAMPLES, count)).astype(int):
            name = _cs_text(table["blob/path"][row])
            for root in (folder.parent, folder):
                stack = root / name
                if not stack.is_file():
                    continue
                try:
                    with mrcfile.mmap(stack, mode="r", permissive=True) as mrc:
                        data = mrc.data
                        index = int(table["blob/idx"][row])
                        if (
                            data is not None
                            and data.ndim == 3
                            and index < data.shape[0]
                        ):
                            images.append(np.asarray(data[index], dtype=np.float32))
                        elif data is not None and data.ndim == 2:
                            images.append(np.asarray(data, dtype=np.float32))
                except Exception:
                    logger.debug("cannot read %s", stack, exc_info=True)
                break
    return _extraction_outcome(count, micrographs, images, path.name)


def _cs_import(folder: Path, uid: str) -> dict | None:
    """The number of movies, micrographs or particles that a CryoSPARC import made."""
    imported = sorted(folder.glob("imported_*.cs")) or sorted(folder.glob("*.cs"))
    for path in imported:
        table = _cs_table(path)
        names = (table.dtype.names or ()) if table is not None else ()
        for field, what in (
            ("movie_blob/path", "Movies imported"),
            ("micrograph_blob/path", "Micrographs imported"),
            ("blob/path", "Particles imported"),
        ):
            if field in names and len(table):
                return dict(
                    title="Import", panels=[], facts=[(what, f"{len(table):,}")]
                )
    return None


def _cs_select(folder: Path, uid: str) -> dict | None:
    """What a CryoSPARC 2D or 3D class selection kept, of all the particles it was given: the
    classes (the templates the job selected and excluded) and the particles."""
    import numpy as np

    kept = _cs_output(folder, uid, "particles_selected.cs")
    dropped = _cs_output(folder, uid, "particles_excluded.cs")
    table = _cs_table(kept) if kept is not None else None
    if table is None:
        return None
    other = _cs_table(dropped) if dropped is not None else None
    names = table.dtype.names or ()
    kind = "3D" if "alignments3D/class" in names else "2D"
    classes = None
    chosen = _cs_rows(folder, uid, "templates_selected.cs")
    left = _cs_rows(folder, uid, "templates_excluded.cs")
    if chosen is not None:
        classes = (chosen, chosen + left if left is not None else None)
    elif f"alignments{kind}/class" in names:
        field = f"alignments{kind}/class"
        picked = set(np.unique(np.asarray(table[field])).tolist())
        every = set(picked)
        if other is not None and field in (other.dtype.names or ()):
            every |= set(np.unique(np.asarray(other[field])).tolist())
        classes = (len(picked), len(every))
    total = len(table) + len(other) if other is not None else None
    panels = []
    stack = folder / "templates_selected.mrc"
    if stack.is_file():
        try:
            import mrcfile

            from .browser_parts import _montage

            with mrcfile.mmap(stack, mode="r", permissive=True) as mrc:
                data = mrc.data
                if data is not None and data.ndim == 3:
                    frames = [
                        np.asarray(data[i], dtype=np.float32)
                        for i in range(min(len(data), MAX_TILES))
                    ]
                    panels.append(
                        _panel(
                            "Selected class averages",
                            _montage(
                                frames, tile=TILE, columns=4 if len(frames) <= 12 else 6
                            ),
                            f"{len(data)} selected classes",
                        )
                    )
        except Exception:
            logger.debug("cannot read %s", stack, exc_info=True)
    return _selection_outcome(kind, classes, (len(table), total), panels)


def _cs_refmotion(folder: Path, uid: str) -> dict | None:
    """The outcome of a CryoSPARC reference-based motion correction: how much each movie frame
    contributes at some resolutions (the empirical dose weights it fitted), and the numbers.
    """
    import numpy as np

    weights_file = folder / "refm_empirical_dw.npy"
    if not weights_file.is_file():
        return None
    weights = np.load(weights_file)
    if weights.ndim != 2 or weights.shape[0] < 2 or weights.shape[1] < 2:
        return None
    frames, bins = weights.shape
    psize = 0.0
    table = _cs_output(folder, uid, "hyperparams.cs")
    hyper = _cs_table(table) if table is not None else None
    if hyper is not None and "refmotion_doseweights/psize_A" in (
        hyper.dtype.names or ()
    ):
        psize = float(hyper["refmotion_doseweights/psize_A"][0])
    if psize <= 0:
        return None
    # the columns are the spatial frequencies from 0 to the Nyquist frequency of the weights
    nyquist = 2.0 * psize
    curves = []
    for resolution in _contribution_resolutions(nyquist):
        position = (nyquist / resolution) * (bins - 1)
        column = np.array(
            [np.interp(position, np.arange(bins), weights[f]) for f in range(frames)]
        )
        total = float(column.sum())
        curves.append(
            (
                f"{resolution:g} Å",
                np.arange(frames),
                column / total if total > 0 else column,
            )
        )
    panel = _frame_panel(
        curves,
        "the share of each frame in the particles at the resolutions shown (the empirical dose weights)",
    )
    if panel is None:
        return None
    facts = [("Frames", f"{frames}")]
    movies = _cs_rows(folder, uid, "movies.cs")
    failed = _cs_rows(folder, uid, "movies_failed.cs")
    if movies is not None:
        facts.append(
            ("Movies", f"{movies:,}" + (f" ({failed:,} failed)" if failed else ""))
        )
    particles = _cs_rows(folder, uid, "particles_0.cs", "particles.cs")
    if particles is not None:
        facts.append(("Particles", f"{particles:,}"))
    return dict(title="Reference-based motion correction", panels=[panel], facts=facts)


def _cs_sharpen(folder: Path, uid: str) -> dict | None:
    """The central section of the sharpened map of a CryoSPARC sharpening job."""
    path = folder / f"{uid}_map_sharp.mrc"
    if not path.is_file():
        return None
    section, apix = _central_slice(path)
    if section is None:
        return None
    return dict(
        title="Sharpened map",
        panels=[
            _panel(
                "Sharpened map, central section",
                _image(section),
                f"{path.name}, {apix:.2f} Å/pixel",
            )
        ],
    )


def _cs_curate(folder: Path, uid: str) -> dict | None:
    """The exposures that a curation accepted, of all of them."""
    accepted = next(iter(sorted(folder.glob("*exposures_accepted.cs"))), None)
    rejected = next(iter(sorted(folder.glob("*exposures_rejected.cs"))), None)
    table = _cs_table(accepted) if accepted is not None else None
    if table is None:
        return None
    other = _cs_table(rejected) if rejected is not None else None
    return dict(
        title="Curation of exposures",
        panels=[],
        facts=[
            (
                "Exposures accepted",
                _share(
                    len(table), len(table) + len(other) if other is not None else None
                ),
            )
        ],
    )


def _cs_subsets(folder: Path, uid: str) -> dict | None:
    """The sets a particle subsetting made, and how many particles each has."""
    sets = sorted(folder.glob(f"{uid}_particles_set_*.cs"))
    sizes = [len(t) for t in (_cs_table(path) for path in sets) if t is not None]
    if not sizes:
        return None
    facts = [("Subsets", f"{len(sizes)}"), ("Particles", f"{sum(sizes):,} in all")]
    if len(set(sizes)) == 1:
        facts[1] = ("Particles", f"{sizes[0]:,} in each, {sum(sizes):,} in all")
    return dict(title="Particle subsets", panels=[], facts=facts)


def _cs_downsample(folder: Path, uid: str) -> dict | None:
    """The particles of a downsampling: their number, box size and pixel size."""
    path = _cs_output(folder, uid, "downsampled_particles.cs")
    table = _cs_table(path) if path is not None else None
    if table is None or not len(table):
        return None
    facts = [("Particles", f"{len(table):,}")]
    names = table.dtype.names or ()
    if "blob/shape" in names and "blob/psize_A" in names:
        box = int(table["blob/shape"][0][0])
        facts.append(
            ("Box", f"{box} pixels of {float(table['blob/psize_A'][0]):.3f} Å")
        )
    return dict(title="Downsampled particles", panels=[], facts=facts)


def _cs_ctfrefine(folder: Path, uid: str) -> dict | None:
    """What a CryoSPARC CTF refinement made: the particles (the fitted values are in their table)."""
    path = _cs_particle_file(folder, uid)
    table = _cs_table(path) if path is not None else None
    if table is None or not len(table):
        return None
    return dict(
        title="CTF refinement", panels=[], facts=[("Particles", f"{len(table):,}")]
    )


def _cs_builder(job_type: str):
    if job_type == "class_2D_new" or job_type.startswith("class_2D"):
        return _cs_class2d
    if job_type.startswith("extract"):
        return _cs_extract
    if job_type == "reference_motion_correction":
        return _cs_refmotion
    if "motion_correction" in job_type:
        return _cs_motion
    if job_type == "sharpen":
        return _cs_sharpen
    if job_type.startswith("curate_exposures"):
        return _cs_curate
    if job_type == "subset_particles":
        return _cs_subsets
    if job_type == "downsample_particles":
        return _cs_downsample
    if job_type.startswith("ctf_refine"):
        return _cs_ctfrefine
    if job_type.startswith(("class_3D", "hetero_refine", "homo_abinit")):
        return _cs_class3d
    if job_type.startswith("select"):
        return _cs_select
    if job_type.startswith("import"):
        return _cs_import
    if "pick" in job_type or "tracer" in job_type:
        return _cs_picks
    if "ctf_estimation" in job_type or job_type.startswith("ctffind"):
        return _cs_ctf
    if any(word in job_type for word in ("refine", "abinit", "reconstruct")):
        return _cs_volume
    return None


# ── the entry ────────────────────────────────────────────────────────────────


def _signature(folder: Path) -> tuple:
    """``(name, size, time)`` of what is in a folder: what the cache of its outcome is keyed by, so
    that a job that is running, or was run again, is read again."""
    import os

    try:
        with os.scandir(folder) as entries:
            found = []
            for entry in entries:
                info = entry.stat()
                found.append((entry.name, info.st_size, info.st_mtime_ns))
        return tuple(sorted(found))
    except OSError:
        return ()


def _build(folder: str, kind: str, job_type: str | None, exact: bool) -> dict | None:
    """The outcome of a job, made from its files (:func:`build_outcome` without the cache)."""
    folder = Path(folder)
    try:
        if kind == "relion":
            builder = _RELION.get(folder.parent.name)
            outcome = builder(folder) if builder else None
        else:
            builder = _cs_builder(job_type or "")
            if builder is _cs_picks:
                outcome = builder(folder, folder.name, exact=exact)
            else:
                outcome = builder(folder, folder.name) if builder else None
    except (
        Exception
    ):  # a file that cannot be read is no result, not an error of the browser
        logger.debug("no outcome for %s", folder, exc_info=True)
        return None
    if outcome is not None:
        outcome.setdefault("facts", [])
    return outcome


# the function that keeps the outcomes, one for each cache folder (it can change with the
# environment)
_KEEPERS: dict[str, object] = {}


def _keeper():
    """The cached version of :func:`_build`, in the cache folder of Helicon (``job_outcome``)."""
    import datetime

    from ..cache import cache, setup_cache_dir

    folder = str(setup_cache_dir(create=False) / "job_outcome")
    if folder not in _KEEPERS:
        _KEEPERS[folder] = cache(
            cache_dir=folder,
            expires_after=datetime.timedelta(days=OUTCOME_DAYS),
        )(_build_signed)
    return _KEEPERS[folder]


def _build_signed(
    folder: str, kind: str, job_type: str | None, exact: bool, signature: tuple
) -> dict | None:
    """:func:`_build` with the signature of the folder among its arguments: part of the key of
    the cache, not of the work."""
    return _build(folder, kind, job_type, exact)


def build_outcome(
    folder: str | Path,
    kind: str,
    job_type: str | None = None,
    exact: bool = False,
    use_cache: bool = True,
) -> dict | None:
    """What a job made, as pictures, or None.

    What takes long to make is kept in the cache of Helicon (``helicon.cache``) by the files of the
    job: it is made again when one of them is changed.

    Parameters
    ----------
    folder : str or pathlib.Path
        The job folder.
    kind : str
        ``"relion"`` or ``"cryosparc"``.
    job_type : str, optional
        The CryoSPARC job type (``class_2D_new``); a RELION job's type is the name of the folder
        above it (``Class2D``).
    exact : bool
        Count what takes long to count (the filaments of a table of many GB) exactly.
    use_cache : bool
        Look in the cache, and keep the outcome in it. Defaults to True.

    Returns
    -------
    dict or None
        ``{"title": str, "panels": [{"title", "image", "caption"}, ...], "facts": [(label,
        value), ...]}``; the images are ``(h, w)`` or ``(h, w, 3)`` uint8 arrays. When a number
        in it is an estimate, ``"refine"`` is a function without arguments that makes the same
        outcome with the exact number (slowly): to run in the background.
    """
    folder = Path(folder)
    if not use_cache:
        outcome = _build(str(folder), kind, job_type, exact)
    else:
        signature = _signature(folder)
        keeper = _keeper()
        # the exact answer, when it was made before, is better than an estimate
        if not exact and keeper.is_cached(str(folder), kind, job_type, True, signature):
            exact = True
        try:
            outcome = keeper(str(folder), kind, job_type, exact, signature)
        except Exception:
            logger.debug("the cache of %s failed", folder, exc_info=True)
            outcome = _build(str(folder), kind, job_type, exact)
    if outcome is not None and outcome.pop("approximate", False):
        outcome["refine"] = lambda: build_outcome(folder, kind, job_type, True)
    return outcome
