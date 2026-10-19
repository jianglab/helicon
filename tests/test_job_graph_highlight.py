"""The graph of the jobs: the highlight of a job and its ancestors, and saving it as a PDF."""

import re
import time

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtGui import QContextMenuEvent
from PySide6.QtWidgets import QApplication

from helicon.lib.gui import job_graph as jg
from _settings import own_settings as _own_settings
from _projects import make_pipeline_project


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def pipeline_project(tmp_path):
    return make_pipeline_project(tmp_path)


@pytest.fixture(autouse=True)
def own_settings(tmp_path, monkeypatch):
    _own_settings(monkeypatch, tmp_path / "settings.ini")


def _pump(app, seconds=10.0, until=None):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        if until is not None and until():
            return True
        time.sleep(0.005)
    return until() if until is not None else True


def _chain(n, extra=()):
    """Jobs 0 -> 1 -> ... -> n-1, and the arrows ``extra``."""
    nodes = [jg.JobNode(f"J{i}", f"J{i}", "type", "success", None) for i in range(n)]
    edges = [(i, i + 1) for i in range(n - 1)] + list(extra)
    graph = jg.JobGraph("cryosparc", "/p/demo", nodes, edges)
    graph.places = jg.layout_layers(n, edges)
    graph.positions, graph.bands = jg.pack_bands(graph.places, None)
    return graph


def _media_box(path):
    data = path.read_bytes()
    assert data.startswith(b"%PDF")
    box = re.search(
        rb"/MediaBox\s*\[\s*([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)", data
    )
    assert box, "no page"
    x0, y0, x1, y1 = (float(v) for v in box.groups())
    return x1 - x0, y1 - y0


class TestAncestors:
    def test_the_ancestors_are_all_the_jobs_it_depends_on(self):
        # 0 -> 1 -> 3, 0 -> 2 -> 3, 3 -> 4, and 5 on its own
        nodes = [jg.JobNode(f"J{i}", f"J{i}", "t", "success", None) for i in range(6)]
        graph = jg.JobGraph(
            "cryosparc", "/p", nodes, [(0, 1), (1, 3), (0, 2), (2, 3), (3, 4)]
        )
        assert graph.ancestors(3) == {0, 1, 2}
        assert graph.ancestors(4) == {0, 1, 2, 3}
        assert graph.ancestors(0) == set() and graph.ancestors(5) == set()

    def test_a_cycle_does_not_make_a_job_its_own_ancestor_or_loop_for_ever(self):
        nodes = [jg.JobNode(f"J{i}", f"J{i}", "t", "success", None) for i in range(3)]
        graph = jg.JobGraph("cryosparc", "/p", nodes, [(0, 1), (1, 2), (2, 0)])
        assert graph.ancestors(2) == {0, 1}

    def test_the_subgraph_has_the_arrows_between_its_jobs_and_a_layout_of_its_own(self):
        graph = _chain(6, extra=[(1, 4)])
        sub = graph.subgraph({1, 2, 4})
        assert [n.key for n in sub.nodes] == ["J1", "J2", "J4"]
        assert sorted(sub.edges) == [(0, 1), (0, 2)]  # J1->J2 and J1->J4
        assert len(sub.positions) == 3 and len(sub.places) == 3
        assert sub.nodes[0].key == "J1" and sub.positions[0][1] == 0  # first, on top

    def test_the_subgraph_leaves_the_graph_alone(self):
        graph = _chain(5)
        graph.subgraph({0, 1})
        assert len(graph.nodes) == 5 and len(graph.edges) == 4


class TestHighlight:
    @staticmethod
    def _view(qapp, project):
        view = jg.JobGraphView()
        view.resize(500, 600)
        view.show()
        view.set_graph(jg.build_job_graph("relion", project))
        assert _pump(qapp, until=lambda: view.drawn() == len(view.graph.nodes))
        return view

    @staticmethod
    def _index(view, key):
        return next(i.index for i in view._items if i.node.key == key)

    def test_a_job_and_its_ancestors_are_marked_and_the_rest_steps_back(
        self, qapp, pipeline_project
    ):
        view = self._view(qapp, pipeline_project)
        view.highlight_ancestors(self._index(view, "Select/job012"))
        roles = {i.node.key: view.role_of(i.index) for i in view._items}
        assert roles == {
            "Select/job012": "pinjob",
            "Class2D/job010": "pin",
            "Extract/job005": "pin",
            "Import/job001": "pin",
            "Class2D/job011": "other",
        }
        assert len(view.highlighted()) == 4
        assert len(view._pinned_items) == 1  # the arrows between them

    def test_the_pointer_shows_its_own_roles_over_the_highlight_and_gives_them_back(
        self, qapp, pipeline_project
    ):
        view = self._view(qapp, pipeline_project)
        view.highlight_ancestors(self._index(view, "Select/job012"))
        view._set_hover(self._index(view, "Class2D/job010"))
        assert view.role_of(self._index(view, "Class2D/job010")) == "job"
        view._set_hover(None)
        assert view.role_of(self._index(view, "Class2D/job010")) == "pin"

    def test_clearing_it_gives_back_the_plain_graph(self, qapp, pipeline_project):
        view = self._view(qapp, pipeline_project)
        view.highlight_ancestors(self._index(view, "Select/job012"))
        view.clear_highlight()
        assert {view.role_of(i) for i in range(5)} == {"none"}
        assert view.highlighted() == [] and view._pinned_items == []

    def test_the_two_colours_are_not_those_of_a_state_or_of_the_pointer(self):
        colours = {c.lower() for c in jg.PIN_COLORS.values()}
        assert len(colours) == 2
        assert not colours & {c.lower() for c in jg.STATUS_COLORS.values()}
        assert jg.PIN_COLORS["pin"].lower() not in {
            c.lower() for c in jg.HOVER_COLORS.values()
        }

    def test_the_highlight_stays_when_the_graph_is_laid_out_again(
        self, qapp, pipeline_project
    ):
        view = self._view(qapp, pipeline_project)
        view.highlight_ancestors(self._index(view, "Select/job012"))
        view.set_graph(jg.build_job_graph("relion", pipeline_project))
        assert _pump(qapp, until=lambda: view.drawn() == 5)
        assert len(view.highlighted()) == 4
        keys = {view.graph.nodes[i].key for i in view.highlighted()}
        assert keys == {
            "Select/job012",
            "Class2D/job010",
            "Extract/job005",
            "Import/job001",
        }

    def test_a_graph_without_the_job_has_no_highlight_but_remembers_it(
        self, qapp, pipeline_project
    ):
        view = self._view(qapp, pipeline_project)
        view.highlight_ancestors(self._index(view, "Select/job012"))
        view.set_graph(_chain(3))
        assert view.highlighted() == []


class TestTheMenu:
    @staticmethod
    def _view(qapp, project):
        view = jg.JobGraphView()
        view.resize(500, 600)
        view.show()
        view.set_graph(jg.build_job_graph("relion", project))
        assert _pump(qapp, until=lambda: view.drawn() == len(view.graph.nodes))
        return view

    @staticmethod
    def _event_on(view, key):
        item = next(i for i in view._items if i.node.key == key)
        point = view.mapFromScene(item.scenePos() + item.boundingRect().center())
        return QContextMenuEvent(
            QContextMenuEvent.Reason.Mouse, point, view.mapToGlobal(point)
        )

    @staticmethod
    def _choose(view, monkeypatch, wanted):
        shown = []

        def run(menu, pos):
            texts = [a.text() for a in menu.actions() if a.text()]
            shown.append(texts)
            return next(
                (a for a in menu.actions() if a.text().startswith(wanted)), None
            )

        monkeypatch.setattr(view, "_run_menu", run)
        return shown

    def test_a_job_offers_the_highlight_and_the_entire_graph(
        self, qapp, pipeline_project, monkeypatch
    ):
        view = self._view(qapp, pipeline_project)
        shown = self._choose(view, monkeypatch, "(nothing)")
        view.contextMenuEvent(self._event_on(view, "Select/job012"))
        assert shown == [
            [
                "Highlight job012 and its ancestors",
                "Save the entire graph as PDF…",
            ]
        ]

    def test_choosing_highlight_highlights_the_job_that_was_clicked(
        self, qapp, pipeline_project, monkeypatch
    ):
        view = self._view(qapp, pipeline_project)
        self._choose(view, monkeypatch, "Highlight")
        view.contextMenuEvent(self._event_on(view, "Class2D/job010"))
        names = {view.graph.nodes[i].key for i in view.highlighted()}
        assert names == {"Class2D/job010", "Extract/job005", "Import/job001"}

    def test_with_a_highlight_the_menu_can_clear_it_and_save_it(
        self, qapp, pipeline_project, monkeypatch
    ):
        view = self._view(qapp, pipeline_project)
        view.highlight_ancestors(
            next(i.index for i in view._items if i.node.key == "Select/job012")
        )
        shown = self._choose(view, monkeypatch, "Clear")
        view.contextMenuEvent(self._event_on(view, "Import/job001"))
        assert shown[0][0].startswith("Highlight job001")
        assert "Clear the highlight" in shown[0]
        assert any(
            t.startswith("Save the highlighted graph (job012 and its ancestors)")
            for t in shown[0]
        )
        assert view.highlighted() == []

    def test_the_background_offers_the_entire_graph_only(
        self, qapp, pipeline_project, monkeypatch, tmp_path
    ):
        view = self._view(qapp, pipeline_project)
        shown = self._choose(view, monkeypatch, "Save the entire")
        monkeypatch.setattr(
            view, "_ask_pdf_path", lambda name: str(tmp_path / "all.pdf")
        )
        corner = QPoint(2, 2)
        view.contextMenuEvent(
            QContextMenuEvent(
                QContextMenuEvent.Reason.Mouse, corner, view.mapToGlobal(corner)
            )
        )
        assert shown == [["Save the entire graph as PDF…"]]
        assert (tmp_path / "all.pdf").is_file()

    def test_a_dialog_that_is_cancelled_writes_nothing(
        self, qapp, pipeline_project, monkeypatch, tmp_path
    ):
        view = self._view(qapp, pipeline_project)
        monkeypatch.setattr(view, "_ask_pdf_path", lambda name: "")
        assert view.save_pdf(highlighted=False) is None
        assert list(tmp_path.glob("*.pdf")) == []


class TestPdf:
    def test_the_entire_graph_is_a_pdf_page_that_holds_it(self, qapp, tmp_path):
        graph = _chain(12)
        path = jg.render_graph_pdf(graph, tmp_path / "all.pdf", title="demo: 12 jobs")
        width, height = _media_box(path)
        assert width > jg.NODE_W and height > 12 * jg.STEP_Y * 0.9  # a chain runs down
        assert path.stat().st_size > 1000

    def test_a_long_title_widens_the_page_of_a_narrow_graph(self, qapp, tmp_path):
        short = jg.render_graph_pdf(_chain(3), tmp_path / "s.pdf", title="a")
        title = (
            "a job with a long name and the 120 jobs that it descends from, all of them"
        )
        long = jg.render_graph_pdf(_chain(3), tmp_path / "l.pdf", title=title)
        assert _media_box(long)[0] > _media_box(short)[0] + 100

    def test_the_file_is_vector_text_not_a_picture(self, qapp, tmp_path):
        path = jg.render_graph_pdf(_chain(3), tmp_path / "t.pdf", title="demo")
        data = path.read_bytes()
        assert b"/Font" in data and not re.search(rb"/Subtype\s*/Image", data)

    def test_the_highlighted_graph_is_a_smaller_page_than_the_entire_graph(
        self, qapp, tmp_path
    ):
        graph = _chain(20, extra=[(0, 10)])
        everything = jg.render_graph_pdf(graph, tmp_path / "a.pdf")
        sub = graph.subgraph(graph.ancestors(5) | {5})
        part = jg.render_graph_pdf(
            sub,
            tmp_path / "b.pdf",
            mark=sub.nodes.index(next(n for n in sub.nodes if n.key == "J5")),
        )
        assert _media_box(part)[1] < _media_box(everything)[1]

    def test_a_graph_of_thousands_of_jobs_is_a_page_that_a_reader_opens(
        self, qapp, tmp_path
    ):
        graph = _chain(2500)
        path = jg.render_graph_pdf(graph, tmp_path / "big.pdf")
        width, height = _media_box(path)
        assert max(width, height) <= jg.PDF_MAX_POINTS + 1

    def test_a_long_graph_is_cut_into_bands_for_a_page_of_a_given_shape(
        self, qapp, tmp_path
    ):
        graph = _chain(120)
        path = jg.render_graph_pdf(graph, tmp_path / "bands.pdf", aspect=1.4)
        width, height = _media_box(path)
        assert 0.5 < width / height < 3.0

    def test_no_jobs_is_an_error_and_an_unwritable_place_too(self, qapp, tmp_path):
        with pytest.raises(ValueError):
            jg.render_graph_pdf(jg.JobGraph("relion", "/p"), tmp_path / "x.pdf")
        with pytest.raises(OSError):
            jg.render_graph_pdf(_chain(2), tmp_path / "missing_folder" / "x.pdf")

    def test_the_view_saves_the_highlighted_graph_and_says_so(
        self, qapp, pipeline_project, tmp_path
    ):
        view = TestHighlight._view(qapp, pipeline_project)
        saved = []
        view.pdf_saved.connect(saved.append)
        assert (
            view.save_pdf(highlighted=True, path=str(tmp_path / "h.pdf")) is None
        )  # nothing highlighted
        view.highlight_ancestors(TestHighlight._index(view, "Select/job012"))
        out = view.save_pdf(highlighted=True, path=str(tmp_path / "h.pdf"))
        assert out == str(tmp_path / "h.pdf") and saved == [out]
        whole = view.save_pdf(highlighted=False, path=str(tmp_path / "w.pdf"))
        assert (
            _media_box(tmp_path / "h.pdf")[1] < _media_box(tmp_path / "w.pdf")[1]
            or whole
        )

    def test_the_name_asked_for_is_the_job_for_a_highlight_and_the_project_for_the_rest(
        self, qapp, pipeline_project, monkeypatch, tmp_path
    ):
        view = TestHighlight._view(qapp, pipeline_project)
        names = []
        monkeypatch.setattr(
            view, "_ask_pdf_path", lambda name: names.append(name) or ""
        )
        view.highlight_ancestors(TestHighlight._index(view, "Select/job012"))
        view.save_pdf(highlighted=True)
        view.save_pdf(highlighted=False)
        assert names == [
            "Select_job012_ancestors.pdf",
            f"{pipeline_project.name}_jobs.pdf",
        ]
