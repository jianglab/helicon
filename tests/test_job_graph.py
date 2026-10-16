"""The graph of the jobs of a project: its layout, its reading in a thread, and the view."""

import time

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from helicon.lib.gui import browser_parts as bp
from helicon.lib.gui import file_browser as fb
from helicon.lib.gui import job_graph as jg
from _settings import own_settings as _own_settings
from _projects import make_cs_project, make_pipeline_project


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def pipeline_project(tmp_path):
    return make_pipeline_project(tmp_path)


@pytest.fixture
def cs_project(tmp_path):
    return make_cs_project(tmp_path)


@pytest.fixture(autouse=True)
def own_settings(tmp_path, monkeypatch):
    _own_settings(monkeypatch, tmp_path / "settings.ini")


def _pump(app, seconds=0.2, until=None):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        if until is not None and until():
            return True
        time.sleep(0.005)
    return until() if until is not None else True


class TestLayout:
    def test_a_job_is_below_the_jobs_it_follows(self):
        places = jg.layout_layers(4, [(0, 1), (1, 2), (0, 2), (2, 3)])
        assert [c for c, _ in places] == [0, 1, 2, 3]

    def test_the_column_is_the_longest_chain_before_a_job(self):
        # 0 -> 1 -> 3 and 0 -> 3: 3 is after the longer chain
        places = jg.layout_layers(4, [(0, 1), (1, 3), (0, 3), (0, 2)])
        assert places[3][0] == 2 and places[2][0] == 1

    def test_no_two_jobs_share_a_place(self):
        edges = [(0, i) for i in range(1, 8)] + [(i, 8) for i in range(1, 8)]
        places = jg.layout_layers(9, edges)
        assert len(set(places)) == 9
        assert sorted(r for c, r in places if c == 1) == list(range(7))

    def test_jobs_without_arrows_are_in_the_first_row(self):
        assert jg.layout_layers(3, []) == [(0, 0), (0, 1), (0, 2)]

    def test_nothing_gives_nothing(self):
        assert jg.layout_layers(0, []) == []

    def test_a_cycle_does_not_stop_it(self):
        places = jg.layout_layers(4, [(0, 1), (1, 2), (2, 1), (2, 3)])
        assert len(places) == 4 and len(set(places)) == 4
        assert places[0][0] == 0

    def test_a_loop_on_a_job_and_arrows_to_nowhere_are_ignored(self):
        assert jg.layout_layers(2, [(0, 0), (0, 5), (-1, 1), (0, 1)]) == [
            (0, 0),
            (1, 0),
        ]

    def test_the_order_keeps_parallel_chains_parallel(self):
        # two chains that start in the order a, b and end in the order b, a would cross if the
        # order were that of the numbers: the sweeps untangle them
        # 0 -> 3 ; 1 -> 2 ; (jobs 2, 3 follow)
        places = jg.layout_layers(4, [(0, 3), (1, 2)])
        top = {0: places[0][1], 1: places[1][1]}
        bottom = {0: places[3][1], 1: places[2][1]}
        assert (top[0] < top[1]) == (bottom[0] < bottom[1])

    def test_a_large_graph_is_laid_out_in_a_moment(self):
        import random

        random.seed(3)
        n = 4000
        edges = [(i, random.randrange(i + 1, min(n, i + 30))) for i in range(n - 1)]
        start = time.perf_counter()
        places = jg.layout_layers(n, edges)
        assert time.perf_counter() - start < 2.0
        assert len(set(places)) == n


class TestReadingAProject:
    def test_relion_jobs_are_nodes_and_files_are_arrows(self, pipeline_project):
        graph = jg.build_job_graph("relion", pipeline_project)
        keys = [n.key for n in graph.nodes]
        assert keys == [
            "Import/job001",
            "Extract/job005",
            "Class2D/job010",
            "Class2D/job011",
            "Select/job012",
        ]
        named = {(keys[a], keys[b]) for a, b in graph.edges}
        assert named == {
            ("Import/job001", "Extract/job005"),
            ("Extract/job005", "Class2D/job010"),
            ("Extract/job005", "Class2D/job011"),
            ("Class2D/job010", "Select/job012"),
            ("Extract/job005", "Select/job012"),
        }

    def test_a_job_shows_its_type_alias_and_how_it_ended(self, pipeline_project):
        by_key = {
            n.key: n for n in jg.build_job_graph("relion", pipeline_project).nodes
        }
        node = by_key["Class2D/job010"]
        assert node.type_name == "Class2D" and node.title == "looks"
        assert node.label == "job010  looks" and node.status == "success"
        assert by_key["Class2D/job011"].status == "failure"
        assert (
            by_key["Class2D/job011"].title == ""
            and by_key["Class2D/job011"].label == "job011"
        )
        assert node.folder == str(pipeline_project / "Class2D/job010")

    def test_a_job_that_is_deleted_has_no_folder_and_is_still_there(
        self, pipeline_project
    ):
        import shutil

        shutil.rmtree(pipeline_project / "Extract/job005")
        by_key = {
            n.key: n for n in jg.build_job_graph("relion", pipeline_project).nodes
        }
        assert by_key["Extract/job005"].folder is None

    def test_every_job_is_below_the_jobs_it_follows(self, pipeline_project):
        graph = jg.build_job_graph("relion", pipeline_project)
        assert graph.edges
        for a, b in graph.edges:
            assert graph.positions[b][1] > graph.positions[a][1]

    def test_cryosparc_jobs_and_their_parents(self, cs_project):
        graph = jg.build_job_graph("cryosparc", cs_project)
        keys = [n.key for n in graph.nodes]
        assert keys[:3] == ["J5", "J6", "J7"] and "J99" not in keys
        named = {(keys[a], keys[b]) for a, b in graph.edges}
        assert named == {("J5", "J6"), ("J6", "J7"), ("J6", "J12")}
        by_key = {n.key: n for n in graph.nodes}
        assert (
            by_key["J7"].label == "J7  first look" and by_key["J7"].status == "success"
        )
        assert by_key["J7"].type_name == "2D Classification"
        assert by_key["J12"].status == "failure" and by_key["J14"].status == "pending"

    def test_the_children_of_the_file_count_when_the_parents_do_not_say(
        self, cs_project
    ):
        import json

        record = json.loads((cs_project / "J13" / "job.json").read_text())
        record["children"] = ["J14", "J404"]  # J404 does not exist
        (cs_project / "J13" / "job.json").write_text(json.dumps(record))
        bp.invalidate_project_cache()
        graph = jg.build_job_graph("cryosparc", cs_project)
        keys = [n.key for n in graph.nodes]
        assert ("J13", "J14") in {(keys[a], keys[b]) for a, b in graph.edges}

    def test_a_folder_that_is_not_a_project_is_an_empty_graph(self, tmp_path):
        assert jg.build_job_graph("relion", tmp_path).nodes == []
        assert jg.build_job_graph("cryosparc", tmp_path).nodes == []

    def test_the_node_of_a_folder_or_a_file_in_it(self, pipeline_project):
        graph = jg.build_job_graph("relion", pipeline_project)
        job = pipeline_project / "Class2D/job010"
        i = graph.index_of_folder(job)
        assert graph.nodes[i].key == "Class2D/job010"
        assert graph.index_of_folder(job / "run_it020_data.star") == i
        assert graph.index_of_folder(pipeline_project) is None
        assert graph.index_of_folder(None) is None


class TestBands:
    """A graph that runs down, cut into bands side by side to fill a wide window."""

    @staticmethod
    def _chain(n):
        return [(i, 0) for i in range(n)]

    def test_without_an_aspect_it_is_one_band_running_down(self):
        positions, bands = jg.pack_bands(self._chain(60))
        assert bands == 1
        assert {x for x, _ in positions} == {0.0}
        assert [y for _, y in positions] == [i * jg.STEP_Y for i in range(60)]

    def test_a_small_graph_is_one_band_in_any_window(self):
        assert jg.pack_bands(self._chain(8), aspect=3.0)[1] == 1

    def test_a_long_graph_fills_a_wide_window(self):
        positions, bands = jg.pack_bands(self._chain(120), aspect=2.0)
        assert bands >= 2
        width = max(x for x, _ in positions) + jg.NODE_W
        height = max(y for _, y in positions) + jg.NODE_H
        # about as wide over high as the window (within a factor of two)
        assert 1.0 <= width / height <= 4.0

    def test_a_very_tall_window_keeps_it_in_one_band(self):
        # one band of a chain is 0.02 wide over high: no more than that is wanted
        assert jg.pack_bands(self._chain(120), aspect=0.02)[1] == 1

    def test_each_band_starts_at_the_top_and_no_two_nodes_share_a_place(self):
        places = self._chain(100) + [(i, 1) for i in range(0, 100, 7)]
        positions, bands = jg.pack_bands(places, aspect=2.5)
        assert bands >= 2
        assert min(y for _, y in positions) == 0.0
        assert len(set(positions)) == len(positions)
        # the layers keep their order inside a band, and the bands their order along x
        per = -(-100 // bands)
        for (layer, _), (x, y) in zip(places, positions):
            assert y == (layer % per) * jg.STEP_Y
            assert x >= (layer // per) * 0

    def test_a_band_has_at_least_a_few_layers(self):
        positions, bands = jg.pack_bands(self._chain(20), aspect=50.0)
        assert bands <= 20 // jg.MIN_BAND_LAYERS

    def test_the_same_graph_is_laid_out_again_for_another_window(
        self, pipeline_project
    ):
        graph = jg.build_job_graph("relion", pipeline_project)
        again = graph.laid_out(3.0)
        assert again.nodes == graph.nodes and again.edges == graph.edges
        assert again is not graph  # the first is not changed: it may be shown elsewhere

    def test_the_window_cuts_a_long_graph_and_the_preview_does_not(self, qapp):
        graph = jg.JobGraph("relion", "/x")
        n = 80
        graph.nodes = [
            jg.JobNode(f"Class2D/job{i:03d}", f"job{i:03d}", "Class2D", "success", None)
            for i in range(n)
        ]
        graph.edges = [(i, i + 1) for i in range(n - 1)]
        graph.places = [(i, 0) for i in range(n)]
        graph.positions, graph.bands = jg.pack_bands(graph.places)
        window = jg.JobGraphWindow()
        window.resize(1400, 600)
        window.show()
        window.view.set_graph(graph)
        assert window.view.graph.bands >= 2
        preview = jg.JobGraphView()
        preview.resize(300, 260)
        preview.set_graph(graph)
        assert preview.graph.bands == 1
        window.close()

    def test_the_arrows_that_go_on_in_the_next_band_are_dashed(self, qapp):
        graph = jg.JobGraph("relion", "/x")
        n = 40
        graph.nodes = [
            jg.JobNode(f"Class2D/job{i:03d}", f"job{i:03d}", "Class2D", "success", None)
            for i in range(n)
        ]
        graph.edges = [(i, i + 1) for i in range(n - 1)]
        graph.places = [(i, 0) for i in range(n)]
        view = jg.JobGraphView()
        view.auto_bands = True
        view.resize(1500, 400)
        view.set_graph(graph)
        assert view.graph.bands >= 2
        edges = next(
            i for i in view._scene.items() if hasattr(i, "path") and i.zValue() == -2
        )
        assert any(isinstance(c, type(edges)) for c in edges.childItems())


class TestHover:
    @staticmethod
    def _view(qapp, project):
        view = jg.JobGraphView()
        view.resize(500, 600)
        view.show()
        view.set_graph(jg.build_job_graph("relion", project))
        assert _pump(qapp, 10, until=lambda: view.drawn() == len(view.graph.nodes))
        return view

    def _index(self, view, key):
        return next(i.index for i in view._items if i.node.key == key)

    def test_nothing_is_marked_until_the_pointer_is_on_a_job(
        self, qapp, pipeline_project
    ):
        view = self._view(qapp, pipeline_project)
        assert {view.role_of(i) for i in range(5)} == {"none"}

    def test_the_job_its_parents_and_its_children_each_have_a_role(
        self, qapp, pipeline_project
    ):
        view = self._view(qapp, pipeline_project)
        view._set_hover(self._index(view, "Class2D/job010"))
        roles = {i.node.key: view.role_of(i.index) for i in view._items}
        assert roles == {
            "Class2D/job010": "job",
            "Extract/job005": "parents",
            "Select/job012": "children",
            "Import/job001": "other",
            "Class2D/job011": "other",
        }

    def test_the_three_roles_have_three_colours_and_none_is_a_state_colour(self):
        colours = set(jg.HOVER_COLORS.values())
        assert len(colours) == 3
        assert not colours & {c.lower() for c in jg.STATUS_COLORS.values()}

    def test_the_arrows_of_the_job_are_drawn_in_the_colours_of_its_neighbours(
        self, qapp, pipeline_project
    ):
        view = self._view(qapp, pipeline_project)
        view._set_hover(self._index(view, "Extract/job005"))
        colours = {item.pen().color().name().lower() for item in view._hover_items}
        assert colours == {c.lower() for k, c in jg.HOVER_COLORS.items() if k != "job"}
        # one arrow in from the import, three out
        assert len(view._hover_items) == 2
        view._set_hover(None)
        assert view._hover_items == [] and view.role_of(0) == "none"

    def test_moving_the_pointer_on_to_a_job_marks_it(self, qapp, pipeline_project):
        from PySide6.QtCore import QEvent
        from PySide6.QtWidgets import QGraphicsSceneHoverEvent

        view = self._view(qapp, pipeline_project)
        item = next(i for i in view._items if i.node.key == "Select/job012")
        item.hoverEnterEvent(
            QGraphicsSceneHoverEvent(QEvent.Type.GraphicsSceneHoverEnter)
        )
        assert view.role_of(item.index) == "job"
        item.hoverLeaveEvent(
            QGraphicsSceneHoverEvent(QEvent.Type.GraphicsSceneHoverLeave)
        )
        assert view.role_of(item.index) == "none"

    def test_a_new_graph_forgets_the_hover(self, qapp, pipeline_project):
        view = self._view(qapp, pipeline_project)
        view._set_hover(0)
        view.set_graph(view.graph)
        assert view._hover is None and view._hover_items == []

    def test_painting_with_a_job_marked_does_not_fail(self, qapp, pipeline_project):
        from PySide6.QtGui import QImage, QPainter

        view = self._view(qapp, pipeline_project)
        view._set_hover(self._index(view, "Extract/job005"))
        image = QImage(300, 300, QImage.Format.Format_ARGB32)
        painter = QPainter(image)
        view.render(painter)
        painter.end()
        assert not image.isNull()


class TestTheView:
    def test_the_project_is_read_in_the_background_and_then_drawn(
        self, qapp, pipeline_project
    ):
        view = jg.JobGraphView()
        view.resize(500, 400)
        view.show()
        ready = []
        view.graph_ready.connect(lambda: ready.append(True))
        view.load("relion", pipeline_project)
        # load() has returned and nothing is there yet: the window was not kept waiting
        assert view.graph is None and view.drawn() == 0
        assert _pump(qapp, 20, until=lambda: ready)
        assert _pump(qapp, 5, until=lambda: view.drawn() == 5)
        view.wait()

    def test_the_nodes_are_drawn_a_few_at_a_time(self, qapp, pipeline_project):
        view = jg.JobGraphView()
        view.SLICE_S = 0.0  # a slice of twenty nodes at the least
        graph = jg.build_job_graph("relion", pipeline_project)
        big = jg.JobGraph("relion", graph.root)
        for i in range(70):
            big.nodes.append(graph.nodes[i % len(graph.nodes)])
        big.positions = [(30.0 * (i % 10), 70.0 * (i // 10)) for i in range(70)]
        view.set_graph(big)
        assert view.drawn() == 0  # set_graph returns before the first node
        seen = []
        end = time.monotonic() + 10
        while view.drawn() < 70 and time.monotonic() < end:
            qapp.processEvents()
            seen.append(view.drawn())
        assert view.drawn() == 70
        assert (
            len(set(seen)) >= 3
        )  # in several slices, with the window answering between them

    def test_an_answer_that_is_no_longer_wanted_is_dropped(
        self, qapp, pipeline_project, cs_project
    ):
        view = jg.JobGraphView()
        view.resize(400, 300)
        view.load("relion", pipeline_project)
        view.load("cryosparc", cs_project)  # asked before the first one arrived
        assert _pump(
            qapp,
            20,
            until=lambda: view.graph is not None and view.graph.kind == "cryosparc",
        )
        _pump(qapp, 0.5)
        assert view.graph.kind == "cryosparc"
        view.wait()

    def test_a_project_without_jobs_says_so(self, qapp, tmp_path):
        view = jg.JobGraphView()
        view.resize(400, 300)
        view.show()
        view.load("relion", tmp_path)
        assert _pump(qapp, 20, until=lambda: view.graph is not None)
        assert "No jobs" in view._note.text() and not view._note.isHidden()
        view.wait()

    @staticmethod
    def _drawn_view(qapp, project, kind="relion"):
        view = jg.JobGraphView()
        view.resize(500, 600)
        view.show()
        view.set_graph(jg.build_job_graph(kind, project))
        assert _pump(qapp, 10, until=lambda: view.drawn() == len(view.graph.nodes))
        return view

    def test_a_click_on_a_job_asks_for_its_folder(self, qapp, pipeline_project):
        view = self._drawn_view(qapp, pipeline_project)
        clicked = []
        view.job_clicked.connect(clicked.append)
        item = next(i for i in view._items if i.node.key == "Class2D/job010")
        point = view.mapFromScene(item.sceneBoundingRect().center())
        QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=point)
        assert clicked == [str(pipeline_project / "Class2D/job010")]

    def test_a_drag_pans_and_does_not_click(self, qapp, pipeline_project):
        view = self._drawn_view(qapp, pipeline_project)
        clicked = []
        view.job_clicked.connect(clicked.append)
        item = view._items[0]
        start = view.mapFromScene(item.sceneBoundingRect().center())
        QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(view.viewport(), start + QPoint(40, 30))
        QTest.mouseRelease(
            view.viewport(), Qt.MouseButton.LeftButton, pos=start + QPoint(40, 30)
        )
        assert clicked == []

    def test_a_job_whose_folder_is_gone_is_not_a_link(self, qapp, pipeline_project):
        import shutil

        shutil.rmtree(pipeline_project / "Extract/job005")
        view = self._drawn_view(qapp, pipeline_project)
        clicked = []
        view.job_clicked.connect(clicked.append)
        item = next(i for i in view._items if i.node.key == "Extract/job005")
        point = view.mapFromScene(item.sceneBoundingRect().center())
        QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=point)
        assert clicked == []

    @staticmethod
    def _wheel(view, delta, modifiers=Qt.KeyboardModifier.NoModifier):
        event = QWheelEvent(
            QPointF(100, 100),
            QPointF(100, 100),
            QPoint(0, 0),
            QPoint(0, delta),
            Qt.MouseButton.NoButton,
            modifiers,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        view.wheelEvent(event)

    def test_the_wheel_scrolls_up_and_down_and_does_not_zoom(self, qapp):
        view = self._tall_view(qapp)
        before = view.scale_factor()
        bar = view.verticalScrollBar()
        start = bar.value()
        self._wheel(view, -120)  # a notch towards you: down the graph
        assert bar.value() > start
        self._wheel(view, 120)
        assert bar.value() == start
        assert view.scale_factor() == before

    def test_shift_and_the_wheel_scroll_sideways(self, qapp):
        view = self._tall_view(qapp, wide=True)
        bar = view.horizontalScrollBar()
        start = bar.value()
        self._wheel(view, -120, Qt.KeyboardModifier.ShiftModifier)
        assert bar.value() > start

    def test_ctrl_and_the_wheel_zoom_within_limits(self, qapp, pipeline_project):
        view = self._drawn_view(qapp, pipeline_project)
        before = view.scale_factor()
        ctrl = Qt.KeyboardModifier.ControlModifier
        self._wheel(view, 120, ctrl)
        assert view.scale_factor() > before
        for _ in range(60):
            self._wheel(view, 120, ctrl)
        assert view.scale_factor() <= 4.0 + 1e-9
        for _ in range(200):
            self._wheel(view, -120, ctrl)
        assert view.scale_factor() >= 0.03 - 1e-9

    @staticmethod
    def _tall_view(qapp, wide=False):
        graph = jg.JobGraph("relion", "/x")
        n = 40
        for i in range(n):
            graph.nodes.append(
                jg.JobNode(
                    f"Class2D/job{i:03d}", f"job{i:03d}", "Class2D", "success", None
                )
            )
        graph.edges = [(i, i + 1) for i in range(n - 1)]
        graph.places = [(i, 0) for i in range(n)]
        graph.positions, graph.bands = jg.pack_bands(graph.places)
        view = jg.JobGraphView()
        view.resize(400, 300)
        view.show()
        view.set_graph(graph)
        view.resetTransform()
        view.scale(1.0, 1.0)
        _pump(qapp, 1.0)
        if wide:
            view.scale(3.0, 3.0)
        view.verticalScrollBar().setValue(0)
        view.horizontalScrollBar().setValue(0)
        return view

    def test_fit_shows_all_the_jobs(self, qapp, pipeline_project):
        view = self._drawn_view(qapp, pipeline_project)
        view.zoom(3.0)
        view.fit()
        visible = view.mapToScene(view.viewport().rect()).boundingRect()
        for item in view._items:
            assert visible.contains(item.sceneBoundingRect().center())

    def test_a_large_graph_opens_at_a_readable_size_on_the_newest_job(self, qapp):
        graph = jg.JobGraph("relion", "/x")
        n = 60
        for i in range(n):
            graph.nodes.append(
                jg.JobNode(
                    f"Class2D/job{i:03d}", f"job{i:03d}", "Class2D", "success", None
                )
            )
        graph.edges = [(i, i + 1) for i in range(n - 1)]
        graph.positions = [(0.0, i * jg.STEP_Y) for i in range(n)]
        view = jg.JobGraphView()
        view.resize(400, 300)
        view.show()
        view.set_graph(graph)
        assert view.scale_factor() >= jg.JobGraphView.READABLE - 1e-9
        center = view.mapToScene(view.viewport().rect().center())
        assert center.y() > (n - 6) * jg.STEP_Y  # the last jobs, not the first

    def test_the_current_job_is_marked_and_brought_into_view(
        self, qapp, pipeline_project
    ):
        view = self._drawn_view(qapp, pipeline_project)
        view.set_current(pipeline_project / "Select/job012" / "x.star")
        assert view._items[view._current].node.key == "Select/job012"
        view.set_current(None)
        assert view._current is None

    def test_painting_at_any_size_does_not_fail(self, qapp, pipeline_project):
        from PySide6.QtGui import QImage, QPainter

        view = self._drawn_view(qapp, pipeline_project)
        for factor in (1.0, 0.1, 6.0):
            view.zoom(factor)
            image = QImage(300, 300, QImage.Format.Format_ARGB32)
            painter = QPainter(image)
            view.render(painter)
            painter.end()
        # the labels are left out when too small to read: the boxes are still drawn
        assert not image.isNull()


class TestInThePreview:
    def test_a_project_folder_shows_its_jobs_in_place_of_the_folders(
        self, qapp, pipeline_project
    ):
        pane = bp.PreviewPane()
        pane.resize(320, 700)
        pane.show()
        pane.show_info(bp.build_preview(pipeline_project))
        assert pane._graph_box is not None and not pane._graph_box.isHidden()
        assert pane._text.isHidden()  # no list of the folders
        assert _pump(qapp, 20, until=lambda: pane._graph_view.graph is not None)
        assert len(pane._graph_view.graph.nodes) == 5
        pane._graph_view.wait()
        pane.clear()
        assert pane._graph_box.isHidden()

    def test_the_graph_takes_the_height_of_the_pane_that_is_left(
        self, qapp, pipeline_project
    ):
        pane = bp.PreviewPane()
        pane.resize(320, 900)
        pane.show()
        pane.show_info(bp.build_preview(pipeline_project))
        _pump(qapp, 1.0)
        taller = pane._graph_view.height()
        # the title, the picture of nothing, the facts and the header of the graph use the rest
        assert taller > 900 - 330
        pane.resize(320, 500)
        _pump(qapp, 0.5)
        assert pane._graph_view.height() < taller  # it follows the size of the pane
        assert pane._graph_view.height() >= 180
        pane._graph_view.wait()

    def test_other_folders_show_the_list_and_no_graph(self, qapp, pipeline_project):
        pane = bp.PreviewPane()
        pane.show_info(bp.build_preview(pipeline_project / "Class2D"))
        assert pane._graph_box is None or pane._graph_box.isHidden()
        assert not pane._text.isHidden() or pane._text.toPlainText() != ""

    def test_the_graph_module_is_loaded_only_when_a_project_is_shown(
        self, qapp, tmp_path
    ):
        pane = bp.PreviewPane()
        pane.show_info(bp.build_preview(tmp_path))
        assert pane._graph_view is None

    def test_a_click_in_the_preview_asks_for_the_job(self, qapp, pipeline_project):
        pane = bp.PreviewPane()
        pane.resize(320, 700)
        pane.show()
        asked = []
        pane.path_requested.connect(asked.append)
        pane.show_info(bp.build_preview(pipeline_project))
        assert _pump(qapp, 20, until=lambda: pane._graph_view.drawn() == 5)
        pane._graph_view.job_clicked.emit(str(pipeline_project / "Select/job012"))
        assert asked == [str(pipeline_project / "Select/job012")]
        pane._graph_view.wait()

    def test_the_window_shows_the_same_jobs_and_a_click_there_goes_too(
        self, qapp, pipeline_project
    ):
        pane = bp.PreviewPane()
        pane.resize(320, 700)
        pane.show()
        asked = []
        pane.path_requested.connect(asked.append)
        assert pane._graph_window is None
        pane.show_info(bp.build_preview(pipeline_project))
        assert _pump(qapp, 20, until=lambda: pane._graph_view.graph is not None)
        pane.open_graph_window()
        window = pane._graph_window
        assert window.isVisible()
        # not read again: the same jobs, laid out again for the window
        assert window.view.graph.nodes == pane._graph_view.graph.nodes
        assert _pump(qapp, 10, until=lambda: window.view.drawn() == 5)
        window.view.job_clicked.emit(str(pipeline_project / "Extract/job005"))
        assert asked == [str(pipeline_project / "Extract/job005")]
        pane.open_graph_window()
        assert pane._graph_window is window  # one window
        window.close()
        pane._graph_view.wait()

    def test_the_window_follows_the_folder_of_the_browser(self, qapp, pipeline_project):
        pane = bp.PreviewPane()
        pane.show()
        pane.show_info(bp.build_preview(pipeline_project))
        assert _pump(qapp, 20, until=lambda: pane._graph_view.graph is not None)
        pane.open_graph_window()
        window = pane._graph_window
        assert _pump(qapp, 10, until=lambda: window.view.drawn() == 5)
        pane.set_current(str(pipeline_project / "Class2D/job011"))
        assert window.view._items[window.view._current].node.key == "Class2D/job011"
        window.close()
        pane._graph_view.wait()

    def test_the_window_has_no_button_without_a_project(self, qapp, tmp_path):
        pane = bp.PreviewPane()
        pane.show_info(bp.build_preview(tmp_path))
        pane.open_graph_window()  # nothing to show: no window
        assert pane._graph_window is None


class TestInTheBrowser:
    def test_a_click_on_a_job_goes_to_it_and_the_window_follows(
        self, qapp, pipeline_project
    ):
        window = fb.FolderBrowserWidget(start_dir=str(pipeline_project))
        window.resize(1200, 700)
        window.show()
        try:
            assert _pump(
                qapp, 20, until=lambda: window._preview._graph_view is not None
            )
            assert _pump(
                qapp, 20, until=lambda: window._preview._graph_view.graph is not None
            )
            window._preview.open_graph_window()
            graph_window = window._preview._graph_window
            assert _pump(qapp, 10, until=lambda: graph_window.view.drawn() == 5)
            target = pipeline_project / "Class2D/job010"
            graph_window.view.job_clicked.emit(str(target))
            assert window._model._root_path == str(target)
            _pump(qapp, 1.0)
            view = graph_window.view
            assert view._items[view._current].node.key == "Class2D/job010"
            graph_window.close()
        finally:
            window._preview.wait()
            if window._preview._graph_view is not None:
                window._preview._graph_view.wait()
            window.close()
