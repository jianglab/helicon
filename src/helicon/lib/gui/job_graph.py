"""The jobs of a RELION or CryoSPARC project as a graph that you can click, zoom and pan.

A node is a job; an arrow goes from a job to the jobs that use its files (RELION: from
``default_pipeline.star``; CryoSPARC: the ``parents`` of ``job.json``). The nodes are in rows,
each job below the jobs it takes its files from, coloured by how the job ended.

The graph of a large project is slow to read, lay out and draw, so none of that is done where the
window could notice: :class:`JobGraphView` reads the project and lays it out in a thread
(:func:`build_job_graph`), then adds the nodes to the scene a few at a time between the events of the
window, and drops what is not wanted any more when another project or a refresh arrives. The labels are
left out when the graph is zoomed out too far to read them. The parts that need no Qt
(:func:`build_job_graph`, :func:`layout_layers`) can be used and tested without a window.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
    QLabel,
    QMainWindow,
    QToolBar,
    QToolButton,
)

__all__ = [
    "JobGraph",
    "JobGraphView",
    "JobGraphWindow",
    "JobNode",
    "build_job_graph",
    "layout_layers",
]

# the size of a node and the room between nodes, in scene units: a chain of jobs, which is what a
# project mostly is, runs down, so that it is as wide as the few jobs side by side
NODE_W, NODE_H = 168.0, 38.0
STEP_X, STEP_Y = 192.0, 72.0
# a long graph is cut into bands side by side, to use the width of a wide window: the room between
# bands, and the fewest layers of a band
GAP_X = 80.0
MIN_BAND_LAYERS = 6

# the colours of the job under the pointer, of the jobs it takes its files from and of the jobs
# that use its files (none is a colour of a state)
HOVER_COLORS = dict(job="#d6409f", parents="#8e6cd8", children="#17b3c6")

STATUS_COLORS = {
    "success": "#4caf50",
    "failure": "#e5534b",
    "aborted": "#e5534b",
    "running": "#e0a93b",
    "queued": "#5aa0d8",
    "pending": "#8f8f8f",
    "unknown": "#8f8f8f",
}
# the state words of default_pipeline.star, in the terms of the status of the file browser
_RELION_STATES = {
    "succeeded": "success",
    "failed": "failure",
    "aborted": "aborted",
    "running": "running",
    "scheduled": "queued",
}


@dataclass(frozen=True)
class JobNode:
    """A job of the graph."""

    key: str  # RELION: ``Class2D/job010``; CryoSPARC: ``J7``
    label: str  # what is written in the node
    type_name: str  # ``2D Classification``, ``relion.class2d``
    status: str  # a key of STATUS_COLORS
    folder: str | None  # the folder of the job, None when it is gone
    title: str = ""  # the alias (RELION) or title (CryoSPARC) of the job


@dataclass
class JobGraph:
    """The jobs of a project, how they are connected, and where each is drawn."""

    kind: str  # "relion" or "cryosparc"
    root: str  # the project folder
    nodes: list[JobNode] = field(default_factory=list)
    edges: list[tuple[int, int]] = field(default_factory=list)  # (from, to), indices
    positions: list[tuple[float, float]] = field(default_factory=list)  # top left
    places: list[tuple[int, int]] = field(default_factory=list)  # (layer, row) of each
    bands: int = 1  # how many bands side by side the layers are cut into
    _neighbours: tuple | None = field(default=None, repr=False, compare=False)

    def index_of_folder(self, folder: str | Path | None) -> int | None:
        """The node whose folder is ``folder`` (or contains it), or None."""
        if not folder:
            return None
        text = str(folder).rstrip("/")
        for i, node in enumerate(self.nodes):
            if node.folder and (
                text == node.folder or text.startswith(node.folder + "/")
            ):
                return i
        return None

    def neighbours(self) -> tuple[list[list[int]], list[list[int]]]:
        """``(parents, children)``: for each node, the nodes before it and the nodes after it."""
        if self._neighbours is None:
            parents = [[] for _ in self.nodes]
            children = [[] for _ in self.nodes]
            for a, b in self.edges:
                children[a].append(b)
                parents[b].append(a)
            self._neighbours = (parents, children)
        return self._neighbours

    def laid_out(self, aspect: float | None = None) -> "JobGraph":
        """The same graph with the nodes placed for a window of this width over height.

        The graph runs down; when it is much longer than wide it is cut into bands side by side,
        as many as make the whole about ``aspect`` wide over high (``None``: one band).
        """
        positions, bands = pack_bands(self.places, aspect)
        return replace(
            self, positions=positions, bands=bands, _neighbours=self._neighbours
        )


# ── the graph, without Qt ────────────────────────────────────────────────────


def layout_layers(n: int, edges: list[tuple[int, int]], sweeps: int = 4):
    """Put the nodes of a directed graph in columns, each node after the nodes it follows.

    The column of a node is the length of the longest chain of nodes before it. Inside a column
    the nodes are ordered to keep the arrows short and few of them crossing (the average place of
    the neighbours in the next column, a few sweeps over the columns). A cycle, which a project
    should not have, does not stop it: the nodes of a cycle go in the column after the others.

    Parameters
    ----------
    n : int
        The number of nodes, numbered ``0 ... n - 1``.
    edges : list of tuple of int
        ``(from, to)``.
    sweeps : int, optional
        How many times the order is improved; each costs about the size of the graph.

    Returns
    -------
    list of tuple of int
        ``(column, row)`` of each node; the rows of a column are ``0, 1, 2, ...``.
    """
    children = [[] for _ in range(n)]
    parents = [[] for _ in range(n)]
    indegree = [0] * n
    for a, b in edges:
        if a != b and 0 <= a < n and 0 <= b < n:
            children[a].append(b)
            parents[b].append(a)
            indegree[b] += 1
    column = [0] * n
    ready = [v for v in range(n) if indegree[v] == 0]
    done = 0
    remaining = indegree[:]
    while ready:
        v = ready.pop()
        done += 1
        for w in children[v]:
            column[w] = max(column[w], column[v] + 1)
            remaining[w] -= 1
            if remaining[w] == 0:
                ready.append(w)
    if done < n:  # a cycle: its nodes after everything that was placed
        last = max(column, default=0) + 1
        for v in range(n):
            if remaining[v] > 0:
                column[v] = last
    n_columns = max(column, default=-1) + 1
    layers = [[] for _ in range(n_columns)]
    for v in range(n):
        layers[column[v]].append(v)  # in the order of the jobs to begin with
    row = [0.0] * n
    for layer in layers:
        for i, v in enumerate(layer):
            row[v] = float(i)

    def settle(layer, neighbours):
        keyed = []
        for v in layer:
            near = neighbours[v]
            keyed.append(
                (sum(row[w] for w in near) / len(near) if near else row[v], row[v], v)
            )
        keyed.sort()
        for i, (_, _, v) in enumerate(keyed):
            row[v] = float(i)
        layer[:] = [v for _, _, v in keyed]

    for _ in range(sweeps):
        for layer in layers[1:]:
            settle(layer, parents)
        for layer in reversed(layers[:-1]):
            settle(layer, children)
    return [(column[v], int(row[v])) for v in range(n)]


def _relion_graph(root: Path) -> JobGraph:
    from .browser_parts import _read_pipeline

    pipeline = _read_pipeline(root)
    names = sorted(
        pipeline["processes"],
        key=lambda p: (int(m.group(1)) if (m := re.search(r"job(\d+)", p)) else 0),
    )
    graph = JobGraph("relion", str(root))
    index = {}
    for name in names:
        key = name.rstrip("/")
        alias = pipeline["processes"].get(name, "None").rstrip("/")
        title = "" if alias in ("", "None") else alias.rsplit("/", 1)[-1]
        kind, state = pipeline["details"].get(name, ("", ""))
        folder = root / key
        index[name] = len(graph.nodes)
        graph.nodes.append(
            JobNode(
                key=key,
                label=(
                    key.split("/", 1)[-1]
                    if title == ""
                    else f"{key.split('/', 1)[-1]}  {title}"
                ),
                type_name=key.split("/", 1)[0],
                status=_RELION_STATES.get(state.lower(), "unknown"),
                folder=str(folder) if folder.is_dir() else None,
                title=title,
            )
        )
    writer = {
        node: process
        for process, nodes in pipeline["outputs"].items()
        for node in nodes
    }
    seen = set()
    for process, nodes in pipeline["inputs"].items():
        if process not in index:
            continue
        for node in nodes:
            other = writer.get(node)
            if other in index and other != process and (other, process) not in seen:
                seen.add((other, process))
                graph.edges.append((index[other], index[process]))
    return graph


def _cryosparc_graph(root: Path) -> JobGraph:
    from .browser_parts import _cryosparc_jobs

    jobs = sorted(
        (number, name, job)
        for entries in _cryosparc_jobs(root).values()
        for number, name, job in entries
    )
    graph = JobGraph("cryosparc", str(root))
    index = {}
    for _number, name, job in jobs:
        index[job["uid"]] = len(graph.nodes)
        index.setdefault(name, len(graph.nodes))
        graph.nodes.append(
            JobNode(
                key=name,
                label=f"{name}  {job['title']}" if job["title"] else name,
                type_name=job["type_name"],
                status=job["status"],
                folder=str(root / name),
                title=job["title"],
            )
        )
    seen = set()
    for _number, name, job in jobs:
        for parent in job["parents"]:
            a, b = index.get(parent), index.get(name)
            if a is not None and b is not None and a != b and (a, b) not in seen:
                seen.add((a, b))
                graph.edges.append((a, b))
        for child in job["children"]:
            a, b = index.get(name), index.get(child)
            if a is not None and b is not None and a != b and (a, b) not in seen:
                seen.add((a, b))
                graph.edges.append((a, b))
    return graph


def pack_bands(
    places: list[tuple[int, int]], aspect: float | None = None
) -> tuple[list[tuple[float, float]], int]:
    """Where to draw nodes that are in layers (``(layer, row)``): down, in bands side by side.

    The layers run down and the nodes of a layer are side by side, centred on the widest layer. A
    graph that is much longer than wide would leave most of a wide window empty: from
    ``aspect`` (width over height of the window) the layers are cut into the number of bands that
    makes the whole about as wide over high, each band starting at the top.

    Returns
    -------
    positions : list of tuple of float
        The top left of each node, scene units.
    bands : int
        The number of bands.
    """
    if not places:
        return [], 1
    widths: dict[int, int] = {}
    for c, r in places:
        widths[c] = max(widths.get(c, 0), r + 1)
    widest = max(widths.values())
    n_layers = max(widths) + 1
    width = widest * STEP_X
    bands = 1
    if aspect and aspect > 0 and n_layers >= 2 * MIN_BAND_LAYERS:
        wanted = math.sqrt(n_layers * STEP_Y * aspect / (width + GAP_X))
        bands = max(1, min(int(round(wanted)), n_layers // MIN_BAND_LAYERS))
    per = math.ceil(n_layers / bands)
    bands = math.ceil(n_layers / per)
    positions = [
        (
            (c // per) * (width + GAP_X) + (r + (widest - widths[c]) / 2.0) * STEP_X,
            (c % per) * STEP_Y,
        )
        for c, r in places
    ]
    return positions, bands


def build_job_graph(
    kind: str, root: str | Path, aspect: float | None = None
) -> JobGraph:
    """Read the jobs of a project and lay them out. Safe to call from a thread.

    Parameters
    ----------
    kind : str
        ``"relion"`` or ``"cryosparc"``.
    root : str or pathlib.Path
        The project folder.
    aspect : float, optional
        Width over height of the window the graph is for (see :func:`pack_bands`); one band
        when omitted.

    Returns
    -------
    JobGraph
        With the position of each node (top left, scene units; the row of a job at
        ``STEP_Y`` times its depth in its band).
    """
    root = Path(root)
    graph = _relion_graph(root) if kind == "relion" else _cryosparc_graph(root)
    graph.places = layout_layers(len(graph.nodes), graph.edges)
    graph.positions, graph.bands = pack_bands(graph.places, aspect)
    return graph


# ── the view ─────────────────────────────────────────────────────────────────


# the threads that are running: a view that is closed while one reads must not take it with it
_RUNNING: set = set()


class _Worker(QThread):
    """Reads and lays out a project off the main thread; ``ready(serial, graph)`` at the end."""

    ready = Signal(int, object)

    def __init__(self, serial: int, kind: str, root: str, aspect: float | None = None):
        super().__init__()
        self._serial, self._kind, self._root = serial, kind, root
        self._aspect = aspect

    def run(self) -> None:
        try:
            graph = build_job_graph(self._kind, self._root, self._aspect)
        except Exception:  # an unreadable project is an empty graph, not a crash
            graph = JobGraph(self._kind, self._root)
        self.ready.emit(self._serial, graph)


class _NodeItem(QGraphicsItem):
    """A job: a rounded box coloured by its state, with its type and label."""

    def __init__(self, index: int, node: JobNode, view: "JobGraphView"):
        super().__init__()
        self.index = index
        self.node = node
        self._view = view
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        tip = f"{node.key}: {node.type_name}"
        if node.title:
            tip += f"\n{node.title}"
        self.setToolTip(f"{tip}\n{node.status}")

    def boundingRect(self) -> QRectF:
        return QRectF(-3, -3, NODE_W + 6, NODE_H + 6)

    def hoverEnterEvent(self, event) -> None:
        self._view._set_hover(self.index)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self._view._set_hover(None)
        super().hoverLeaveEvent(event)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        colors = self._view._colors
        role = self._view.role_of(self.index)
        if role == "other":
            painter.setOpacity(
                0.35
            )  # something is under the pointer: the rest steps back
        scale = option.levelOfDetailFromTransform(painter.worldTransform())
        base = QColor(STATUS_COLORS.get(self.node.status, STATUS_COLORS["unknown"]))
        fill = QColor(colors["alt"])
        fill = QColor(
            int(fill.red() * 0.6 + base.red() * 0.4),
            int(fill.green() * 0.6 + base.green() * 0.4),
            int(fill.blue() * 0.6 + base.blue() * 0.4),
        )
        current = self.index == self._view._current
        painter.setBrush(QBrush(fill))
        if role in HOVER_COLORS:
            painter.setPen(QPen(QColor(HOVER_COLORS[role]), 3.5))
        else:
            painter.setPen(
                QPen(
                    QColor(colors["accent"]) if current else base,
                    3 if current else 1.4,
                )
            )
        painter.drawRoundedRect(QRectF(0, 0, NODE_W, NODE_H), 6, 6)
        if scale < 0.3:
            return  # too small to read: the box and its colour are what shows
        painter.setPen(QColor(colors["text"]))
        small = QFont(painter.font())
        small.setPointSizeF(7.5)
        painter.setFont(small)
        painter.setPen(QColor(colors["muted"]))
        painter.drawText(
            QRectF(8, 3, NODE_W - 16, 14),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self.node.type_name,
        )
        big = QFont(small)
        big.setPointSizeF(9.0)
        big.setBold(True)
        painter.setFont(big)
        painter.setPen(QColor(colors["text"]))
        metrics = painter.fontMetrics()
        painter.drawText(
            QRectF(8, 18, NODE_W - 16, 17),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            metrics.elidedText(
                self.node.label, Qt.TextElideMode.ElideRight, int(NODE_W - 16)
            ),
        )


class JobGraphView(QGraphicsView):
    """The graph of a project: wheel to scroll, Ctrl+wheel to zoom, drag to pan, click a job to go to
    it; the job under the pointer, its parents and its children are marked in three colours.

    Signals
    -------
    job_clicked(str)
        A job was clicked; the value is its folder.
    graph_ready()
        The project was read and laid out (the nodes are drawn a few at a time after this).
    """

    job_clicked = Signal(str)
    graph_ready = Signal()

    SLICE_S = (
        0.012  # how long the main thread draws nodes before it lets the window answer
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setViewportUpdateMode(
            QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate
        )
        self._colors = dict(
            alt="#2b2b2b",
            text="#e6e6e6",
            muted="#9a9a9a",
            accent="#4a8fd6",
            border="#555555",
        )
        self._serial = 0
        self._worker: _Worker | None = None
        self._pending: tuple[str, str] | None = None
        self._graph: JobGraph | None = None
        self._items: list[_NodeItem] = []
        self._current: int | None = None
        self._current_folder: str | None = None
        self._hover: int | None = None
        self._hover_items: list = []
        # the window cuts a long graph into bands to fill its width; the preview does not
        self.auto_bands = False
        self._relayout = QTimer(self)
        self._relayout.setSingleShot(True)
        self._relayout.setInterval(350)
        self._relayout.timeout.connect(self._maybe_relayout)
        self._press = QPointF()
        self._fitted = (
            True  # until the user zooms: a resize keeps the whole graph in view
        )
        self._note = QLabel("", self)
        self._note.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._note.hide()
        self.setMinimumHeight(120)

    # -- what is shown ----------------------------------------------------

    def set_theme(self, colors: dict[str, str]) -> None:
        self._colors.update({k: v for k, v in colors.items() if k in self._colors})
        self.setBackgroundBrush(
            QBrush(QColor(colors.get("window", self._colors["alt"])))
        )
        self._note.setStyleSheet(f"color: {self._colors['muted']};")
        self.viewport().update()

    @property
    def graph(self) -> JobGraph | None:
        return self._graph

    def load(self, kind: str, root: str | Path) -> None:
        """Read and draw a project, in the background; what is shown now stays until it is ready."""
        self._serial += 1
        self._show_note("Reading the jobs…")
        self._pending = (kind, str(root))
        if self._worker is None or not self._worker.isRunning():
            self._start()

    def _aspect(self) -> float | None:
        """Width over height of the view when it cuts the graph into bands, else None."""
        if not self.auto_bands:
            return None
        size = self.viewport().size()
        return size.width() / size.height() if size.height() > 0 else None

    def _start(self) -> None:
        if self._pending is None:
            return
        (kind, root), self._pending = self._pending, None
        worker = _Worker(self._serial, kind, root, self._aspect())
        _RUNNING.add(worker)
        worker.ready.connect(self._on_ready)
        worker.finished.connect(self._on_finished)
        worker.finished.connect(lambda w=worker: _RUNNING.discard(w))
        self._worker = worker
        worker.start()

    def _on_finished(self) -> None:
        if self._pending is not None:
            self._start()

    def _on_ready(self, serial: int, graph: JobGraph) -> None:
        if serial != self._serial:
            return  # another project, or a refresh, was asked for meanwhile
        self.set_graph(graph)
        self.graph_ready.emit()

    def set_graph(self, graph: JobGraph) -> None:
        """Draw a graph that was built already (the nodes a few at a time)."""
        self._serial += 1
        serial = self._serial
        if self.auto_bands and graph.places:
            graph = graph.laid_out(
                self._aspect()
            )  # a copy: the graph may be shown elsewhere too
        self._graph = graph
        self._scene.clear()
        self._items = []
        self._hover, self._hover_items = None, []
        self._current = graph.index_of_folder(self._current_folder)
        if not graph.nodes:
            self._show_note("No jobs in this project yet")
            return
        self._note.hide()
        self._scene.addItem(self._edge_item(graph))
        xs = [x for x, _ in graph.positions]
        ys = [y for _, y in graph.positions]
        self._scene.setSceneRect(
            QRectF(
                min(xs) - 30,
                min(ys) - 30,
                max(xs) - min(xs) + NODE_W + 60,
                max(ys) - min(ys) + NODE_H + 60,
            )
        )
        self.overview()
        QTimer.singleShot(0, self, lambda: self._draw_nodes(serial, 0))

    def _edge_item(self, graph: JobGraph) -> QGraphicsPathItem:
        """All the arrows as one item (and the ones that go on in the next band as a dashed one):
        a graph of thousands of jobs has as many arrows."""
        path, across = QPainterPath(), QPainterPath()
        for a, b in graph.edges:
            x1 = graph.positions[a][0] + NODE_W / 2
            y1 = graph.positions[a][1] + NODE_H
            x2, y2 = graph.positions[b][0] + NODE_W / 2, graph.positions[b][1]
            target = across if y2 < y1 else path
            target.moveTo(x1, y1)
            bend = max(abs(y2 - y1) / 2, 14.0) if y2 >= y1 else 46.0
            target.cubicTo(x1, y1 + bend, x2, y2 - bend, x2, y2)
        pen = QPen(QColor(self._colors["muted"]), 1.2)
        pen.setCosmetic(True)
        item = QGraphicsPathItem(path)
        item.setPen(pen)
        item.setZValue(-2)
        item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        if not across.isEmpty():
            dashed = QPen(pen)
            dashed.setStyle(Qt.PenStyle.DashLine)
            other = QGraphicsPathItem(across, item)  # shares the fate of the other
            other.setPen(dashed)
            other.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        return item

    # -- what is under the pointer ------------------------------------------

    def role_of(self, index: int) -> str:
        """What a job is to the job under the pointer: ``job``, ``parents`` (a job it takes its
        files from), ``children`` (a job that uses its files) or ``other``; ``none`` while
        nothing is under the pointer."""
        if self._hover is None or self._graph is None:
            return "none"
        if index == self._hover:
            return "job"
        parents, children = self._graph.neighbours()
        if index in parents[self._hover]:
            return "parents"
        if index in children[self._hover]:
            return "children"
        return "other"

    def _set_hover(self, index: int | None) -> None:
        """Highlight a job, its parents and its children, each in a colour of its own."""
        if index == self._hover or self._graph is None:
            return
        for item in self._hover_items:
            self._scene.removeItem(item)
        self._hover_items = []
        self._hover = index
        if index is not None:
            graph = self._graph
            parents, children = graph.neighbours()
            for group, others in (
                ("parents", parents[index]),
                ("children", children[index]),
            ):
                path = QPainterPath()
                for other in others:
                    a, b = (other, index) if group == "parents" else (index, other)
                    x1 = graph.positions[a][0] + NODE_W / 2
                    y1 = graph.positions[a][1] + NODE_H
                    x2, y2 = graph.positions[b][0] + NODE_W / 2, graph.positions[b][1]
                    bend = max(abs(y2 - y1) / 2, 14.0) if y2 >= y1 else 46.0
                    path.moveTo(x1, y1)
                    path.cubicTo(x1, y1 + bend, x2, y2 - bend, x2, y2)
                item = QGraphicsPathItem(path)
                pen = QPen(QColor(HOVER_COLORS[group]), 3.0)
                pen.setCosmetic(True)
                item.setPen(pen)
                item.setZValue(-1)
                item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                self._scene.addItem(item)
                self._hover_items.append(item)
        self.viewport().update()

    def _draw_nodes(self, serial: int, start: int) -> None:
        graph = self._graph
        if serial != self._serial or graph is None:
            return  # another graph is being drawn
        begin = time.perf_counter()
        i = start
        while i < len(graph.nodes):
            item = _NodeItem(i, graph.nodes[i], self)
            item.setPos(*graph.positions[i])
            self._scene.addItem(item)
            self._items.append(item)
            i += 1
            if i % 20 == 0 and time.perf_counter() - begin > self.SLICE_S:
                break
        if i < len(graph.nodes):
            QTimer.singleShot(0, self, lambda: self._draw_nodes(serial, i))
        elif self._current is not None:
            self._reveal_current()

    def _show_note(self, text: str) -> None:
        self._note.setText(text)
        self._note.setGeometry(self.viewport().rect())
        self._note.show()
        self._note.raise_()

    def drawn(self) -> int:
        """How many nodes are in the scene so far."""
        return len(self._items)

    def set_current(self, folder: str | Path | None) -> None:
        """Mark the job of ``folder`` (the folder the browser is in), and bring it into view."""
        self._current_folder = str(folder) if folder else None
        if self._graph is None:
            return
        self._current = self._graph.index_of_folder(self._current_folder)
        self.viewport().update()
        self._reveal_current()

    def _reveal_current(self) -> None:
        if self._current is not None and self._current < len(self._items):
            if not self._fitted:
                self.ensureVisible(self._items[self._current], 60, 60)

    # -- zoom, pan, click ---------------------------------------------------

    # a graph that would be drawn smaller than this, to fit it all in, is shown at this size
    # instead, on its newest job (or the one the browser is in): the boxes are readable
    READABLE = 0.45

    def overview(self) -> None:
        """The first view of a graph: all of it if that can be read, else the latest jobs."""
        rect = self._scene.sceneRect()
        if rect.isEmpty() or self._graph is None:
            return
        view = self.viewport().rect()
        fit = min(view.width() / rect.width(), view.height() / rect.height())
        if fit >= self.READABLE or not self._graph.nodes:
            self.fit()
            return
        self._fitted = False
        # the width of the graph in the window, but not less than readable
        scale = max(min(view.width() / rect.width(), 1.0), self.READABLE)
        self.resetTransform()
        self.scale(scale, scale)
        newest = self._current
        if newest is None:
            newest = max(
                range(len(self._graph.nodes)), key=lambda i: self._graph.positions[i][1]
            )
        x, y = self._graph.positions[newest]
        self.centerOn(x + NODE_W / 2, y + NODE_H / 2)

    def fit(self) -> None:
        """Show the whole graph."""
        if self._scene.sceneRect().isEmpty():
            return
        self._fitted = True
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def zoom(self, factor: float) -> None:
        """Zoom by ``factor`` about the centre of the view (limits: 3 % to 400 %)."""
        current = self.transform().m11()
        factor = max(0.03 / current, min(4.0 / current, factor))
        self._fitted = False
        self.scale(factor, factor)

    def scale_factor(self) -> float:
        return self.transform().m11()

    def wheelEvent(self, event) -> None:
        """The wheel scrolls the graph up and down (a graph that runs down is read like a page);
        with the Shift key it scrolls sideways, and with the Ctrl key it zooms."""
        delta = event.angleDelta().y() or event.angleDelta().x()
        if not delta:
            return
        modifiers = event.modifiers()
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            self.zoom(1.15 ** (delta / 120.0))
        else:
            bar = (
                self.horizontalScrollBar()
                if modifiers & Qt.KeyboardModifier.ShiftModifier
                else self.verticalScrollBar()
            )
            self._fitted = False
            bar.setValue(bar.value() - int(delta * 0.9))
        event.accept()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._note.setGeometry(self.viewport().rect())
        if self._fitted:
            self.fit()
        if self.auto_bands and self._graph is not None and self._graph.places:
            self._relayout.start()  # when the resizing has stopped

    def _maybe_relayout(self) -> None:
        """After the window was resized: cut the graph into another number of bands, if the
        shape of the window now wants it."""
        graph = self._graph
        if graph is None or not graph.places:
            return
        _, bands = pack_bands(graph.places, self._aspect())
        if bands != graph.bands:
            self.set_graph(graph)

    def mousePressEvent(self, event) -> None:
        self._press = event.position()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if (event.position() - self._press).manhattanLength() > 4:
            return  # a drag, to pan
        item = self.itemAt(event.position().toPoint())
        if isinstance(item, _NodeItem) and item.node.folder:
            self._current_folder = item.node.folder
            self._current = item.index
            self.viewport().update()
            self.job_clicked.emit(item.node.folder)

    def mouseDoubleClickEvent(self, event) -> None:
        if self.itemAt(event.position().toPoint()) is None:
            self.fit()  # a double click on the background shows it all again
        else:
            super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom(1.25)
        elif key == Qt.Key.Key_Minus:
            self.zoom(0.8)
        elif key in (Qt.Key.Key_0, Qt.Key.Key_F):
            self.fit()
        else:
            super().keyPressEvent(event)

    def wait(self, ms: int = 5000) -> None:
        """Wait for the read in progress (for tests and when closing)."""
        if self._worker is not None:
            self._worker.wait(ms)


class JobGraphWindow(QMainWindow):
    """The graph in a window of its own, for a large project: zoom and pan as much as needed,
    while the file browser stays where it is. It follows the browser, and a click goes there.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Helicon - Jobs")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.view = JobGraphView(self)
        self.view.auto_bands = True
        self.setCentralWidget(self.view)
        bar = QToolBar(self)
        bar.setMovable(False)
        bar.setIconSize(QSize(16, 16))
        for text, tip, action in (
            ("+", "Zoom in (+)", lambda: self.view.zoom(1.25)),
            ("−", "Zoom out (−)", lambda: self.view.zoom(0.8)),
            (
                "Fit",
                "Show all the jobs (F, or double-click the background)",
                self.view.fit,
            ),
        ):
            button = QToolButton()
            button.setText(text)
            button.setToolTip(tip)
            button.clicked.connect(action)
            bar.addWidget(button)
        legend = QLabel()
        legend.setTextFormat(Qt.TextFormat.RichText)
        legend.setText(
            "&nbsp;&nbsp; wheel: scroll &nbsp; Ctrl+wheel: zoom &nbsp; drag: pan &nbsp; click: go to the job"
            " &nbsp;&nbsp; pointer on a job: "
            f"<span style='color:{HOVER_COLORS['job']}'>■ the job</span> "
            f"<span style='color:{HOVER_COLORS['parents']}'>■ its inputs</span> "
            f"<span style='color:{HOVER_COLORS['children']}'>■ used by</span>"
        )
        bar.addWidget(legend)
        self._legend = legend
        self.addToolBar(bar)
        self.resize(900, 600)

    def set_theme(self, colors: dict[str, str]) -> None:
        self.view.set_theme(colors)
        self._legend.setStyleSheet(f"color: {colors.get('muted', '#8f8f8f')};")
