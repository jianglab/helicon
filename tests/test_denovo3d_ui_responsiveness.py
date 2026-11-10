"""Tests for Denovo3D single-threaded UI responsiveness, score_only scan, and Plotly HTML size."""

from __future__ import annotations

import asyncio

import numpy as np

from helicon.lib import system as helicon_system
from helicon.webApps.tabs import denovo3d_bookmark, denovo3d_tab, denovo3d_ui


def _disable_real_threads(monkeypatch):
    """Simulate a single-threaded runtime (Pyodide) across helicon and TaskPool."""
    monkeypatch.setattr(helicon_system, "has_real_threads", lambda: False)
    monkeypatch.setattr(denovo3d_tab.helicon, "has_real_threads", lambda: False)


class _ValueBox:
    """Minimal reactive.value stand-in recording every ``.set()`` call."""

    def __init__(self, initial=None):
        self.value = initial
        self.history = []

    def __call__(self):
        return self.value

    def set(self, val):
        self.value = val
        self.history.append(val)


class _FakeProgress:
    """Session-free stand-in for ``shiny.ui.Progress`` recording updates."""

    instances: list[_FakeProgress] = []

    def __init__(self, min=0, max=1, **_kwargs):
        self.min = min
        self.max = max
        self.updates = []
        _FakeProgress.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def set(self, value=None, message=None, detail=None):
        self.updates.append((value, message, detail))


def _make_task(
    ti,
    ntasks,
    twist,
    rise=4.75,
    *,
    image_index="1",
    model="gauss",
    return_3d=False,
    image=None,
):
    """Construct a positional task tuple matching ``denovo3d_pipeline.process_one_task``."""
    if image is None:
        image = np.ones((8, 16), dtype=np.float32)
    ny, nx = image.shape
    apix = 5.0
    return (
        ti,
        ntasks,
        image,
        "Selected image",
        image_index,
        float(twist),
        float(rise),
        (float(rise), float(rise)),
        1,
        0,
        (0, 0),
        0,
        0,
        0,
        0,
        apix,
        "",
        -1,
        0,
        0,
        -1,
        -1,
        -1,
        -1,
        nx * apix,
        ny * apix,
        0.0,
        3 * float(rise),
        -1,
        "linear",
        0,
        bool(return_3d),
        "cosine",
        {"model": model, "l1_ratio": 0.5},
        2,
        1,
    )


def _make_task_result(task, score, *, full_projections=False):
    """Build a synthetic ``process_one_task`` result tuple for *task*."""
    data = task[2]
    image_file = task[3]
    image_index = task[4]
    twist = float(task[5])
    rise = float(task[6])
    csym = int(task[8])
    return_3d = bool(task[-5])
    x_proj = np.full((8, 20), score, dtype=np.float32)
    if full_projections:
        y_proj = np.full((8, 20), score, dtype=np.float32)
        z_sec = np.full((8, 8), score, dtype=np.float32)
        vol = np.ones((4, 8, 8), dtype=np.float32)
        return_data = (
            x_proj,
            y_proj,
            z_sec,
            (vol, None, None) if return_3d else None,
            8,
            8,
            16,
            4,
            vol if return_3d else None,
        )
    else:
        return_data = (
            x_proj,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    params = (
        data,
        image_file,
        image_index,
        5.0,
        5.0,
        twist,
        rise,
        csym,
        0,
        0,
        0,
    )
    return (float(score), return_data, params)


class TestSingleThreadedResponsiveness:
    def test_scan_yields_to_event_loop_between_tasks_and_updates_progress(
        self, monkeypatch
    ):
        _FakeProgress.instances.clear()
        monkeypatch.setattr(denovo3d_tab.ui, "Progress", _FakeProgress)
        _disable_real_threads(monkeypatch)

        events = []

        def fake_process_one_task(*task):
            twist = task[5]
            score_only = bool(task[-3].get("score_only", False))
            events.append(("task", round(twist, 2), score_only))
            return _make_task_result(
                task, score=1.0 - abs(twist - 1.2), full_projections=not score_only
            )

        monkeypatch.setattr(
            denovo3d_tab.denovo3d_pipeline, "process_one_task", fake_process_one_task
        )

        twists = [1.0, 1.1, 1.2, 1.3]
        tasks = [_make_task(i, len(twists), tw) for i, tw in enumerate(twists)]
        raw_rv = _ValueBox([])
        ranked_rv = _ValueBox([])
        scan_rv = _ValueBox([])
        abort_flag = [False]

        async def run():
            done = False

            async def observer():
                while not done:
                    events.append(("tick", None, None))
                    await asyncio.sleep(0)

            obs = asyncio.create_task(observer())
            try:
                await denovo3d_tab._run_reconstruction_task(
                    tasks,
                    cpu=1,
                    abort_ref=abort_flag,
                    reconstruction_results_raw=raw_rv,
                    reconstruction_results=ranked_rv,
                    scan_results=scan_rv,
                    top_n=2,
                    lr_algorithm="gauss",
                    rec_algorithm="elasticnet",
                    display_model="elasticnet",
                    is_scan=True,
                )
            finally:
                done = True
                await obs

        asyncio.run(run())

        # Every consecutive pair of synchronous tasks must have an event-loop
        # tick between them rather than executing back-to-back inside list comp.
        task_indices = [i for i, ev in enumerate(events) if ev[0] == "task"]
        assert len(task_indices) == 4 + 2  # 4 scan tasks + 2 redraw tasks
        for a, b in zip(task_indices, task_indices[1:]):
            between = [ev[0] for ev in events[a + 1 : b]]
            assert "tick" in between

        scan_progress = _FakeProgress.instances[0]
        completed_values = [u[0] for u in scan_progress.updates if u[0] is not None]
        assert completed_values == [1, 2, 3, 4]
        assert len(ranked_rv.history) >= 4

    def test_scan_stops_immediately_on_abort_flag_without_running_remaining_tasks(
        self, monkeypatch
    ):
        _FakeProgress.instances.clear()
        monkeypatch.setattr(denovo3d_tab.ui, "Progress", _FakeProgress)
        _disable_real_threads(monkeypatch)

        executed = []
        abort_flag = [False]

        def fake_process_one_task(*task):
            executed.append((round(task[5], 2), bool(task[-3].get("score_only"))))
            return _make_task_result(task, score=0.5, full_projections=False)

        monkeypatch.setattr(
            denovo3d_tab.denovo3d_pipeline, "process_one_task", fake_process_one_task
        )

        twists = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
        tasks = [_make_task(i, len(twists), tw) for i, tw in enumerate(twists)]

        async def run():
            done = False

            async def click_stop_after_two():
                while not done and len(executed) < 2:
                    await asyncio.sleep(0)
                if len(executed) >= 2:
                    abort_flag[0] = True

            stopper = asyncio.create_task(click_stop_after_two())
            try:
                await denovo3d_tab._run_reconstruction_task(
                    tasks,
                    cpu=1,
                    abort_ref=abort_flag,
                    reconstruction_results_raw=_ValueBox([]),
                    reconstruction_results=_ValueBox([]),
                    scan_results=_ValueBox([]),
                    top_n=3,
                    lr_algorithm="gauss",
                    rec_algorithm="elasticnet",
                    display_model="elasticnet",
                    is_scan=True,
                )
            finally:
                done = True
                await stopper

        asyncio.run(run())

        # Stopped right after the 2nd task: remaining 4 scan tasks and all
        # Phase 2 redraw tasks were skipped.
        assert executed == [(0.8, True), (0.9, True)]

    def test_redraw_with_yields_between_tasks_and_stops_on_abort_flag(
        self, monkeypatch
    ):
        _FakeProgress.instances.clear()
        monkeypatch.setattr(denovo3d_tab.ui, "Progress", _FakeProgress)
        _disable_real_threads(monkeypatch)

        executed = []
        abort_flag = [False]

        def fake_process_one_task(*task):
            executed.append((round(task[5], 2), task[-3]))
            return _make_task_result(task, score=0.9, full_projections=True)

        monkeypatch.setattr(
            denovo3d_tab.denovo3d_pipeline, "process_one_task", fake_process_one_task
        )

        twists = [1.0, 1.1, 1.2, 1.3, 1.4]
        tasks = [_make_task(i, len(twists), tw) for i, tw in enumerate(twists)]
        results = [
            _make_task_result(t, score=float(i), full_projections=False)
            for i, t in enumerate(tasks)
        ]
        ranked = sorted(results, key=lambda r: r[0], reverse=True)
        log = denovo3d_tab._denovo3d_logger()

        async def run():
            done = False

            async def click_stop_after_two():
                while not done and len(executed) < 2:
                    await asyncio.sleep(0)
                if len(executed) >= 2:
                    abort_flag[0] = True

            stopper = asyncio.create_task(click_stop_after_two())
            try:
                return await denovo3d_tab._redraw_with(
                    tasks,
                    results,
                    ranked,
                    top_n=5,
                    display_model="elasticnet",
                    cpu=1,
                    abort_ref=abort_flag,
                    log=log,
                )
            finally:
                done = True
                await stopper

        updated = asyncio.run(run())
        assert len(executed) == 2
        assert all("score_only" not in algo for _, algo in executed)
        # Exactly 2 entries had their z-sections filled before abort.
        filled = [r for r in updated if r[1][2] is not None]
        assert len(filled) == 2


class TestScoreOnlyScanAndTopNRedraw:
    def test_multi_pair_scan_handles_none_z_sections_and_redraws_when_solvers_match(
        self, monkeypatch
    ):
        _FakeProgress.instances.clear()
        monkeypatch.setattr(denovo3d_tab.ui, "Progress", _FakeProgress)
        _disable_real_threads(monkeypatch)

        calls = []

        def fake_process_one_task(*task):
            algo = dict(task[-3])
            score_only = bool(algo.get("score_only", False)) and not bool(task[-5])
            calls.append((round(task[5], 2), algo))
            # True peak at twist=1.2; redraw score deliberately differs to prove
            # Phase 1 ranking scores are preserved.
            score = (1.0 - abs(task[5] - 1.2)) if score_only else 0.01
            return _make_task_result(task, score=score, full_projections=not score_only)

        monkeypatch.setattr(
            denovo3d_tab.denovo3d_pipeline, "process_one_task", fake_process_one_task
        )

        twists = [1.0, 1.1, 1.2, 1.3, 1.4]
        tasks = [
            _make_task(i, len(twists), tw, model="gauss") for i, tw in enumerate(twists)
        ]
        raw_rv = _ValueBox([])
        ranked_rv = _ValueBox([])
        scan_rv = _ValueBox([])

        intermediate_galleries = []
        intermediate_figures = []

        orig_set = ranked_rv.set

        def tracking_set(val):
            orig_set(val)
            # Simulate both reactive consumers firing on every update of
            # reconstruction_results: projections gallery + live scores plot.
            labels, imgs = denovo3d_tab._collect_projection_gallery(
                ranked_rv(), raw_rv(), top_n=2
            )
            intermediate_galleries.append((labels, imgs))
            fig = denovo3d_ui._scores_figure(ranked_rv())
            if fig is not None:
                intermediate_figures.append(fig)

        ranked_rv.set = tracking_set

        # Pass display_model=None with search_algo == rec_algo == "gauss" to
        # verify score_only scan STILL triggers Phase 2 redraw for top_n.
        asyncio.run(
            denovo3d_tab._run_reconstruction_task(
                tasks,
                cpu=1,
                abort_ref=[False],
                reconstruction_results_raw=raw_rv,
                reconstruction_results=ranked_rv,
                scan_results=scan_rv,
                display_model=None,
                top_n=2,
                lr_algorithm="gauss",
                rec_algorithm="gauss",
                is_scan=True,
            )
        )

        # Phase 1: 5 calls with score_only=True; Phase 2: 2 calls without score_only.
        assert len(calls) == 7
        phase1 = calls[:5]
        phase2 = calls[5:]
        assert all(algo.get("score_only") is True for _, algo in phase1)
        assert all("score_only" not in algo for _, algo in phase2)
        assert all(algo["model"] == "gauss" for _, algo in phase2)
        assert {tw for tw, _ in phase2} == {1.2, 1.1} or {tw for tw, _ in phase2} == {
            1.2,
            1.3,
        }

        # Intermediate updates during score_only scan safely skipped None
        # z-sections without raising, while live score figures built cleanly.
        assert intermediate_galleries[0] == ([], [])
        assert len(intermediate_figures) >= 4

        # Final gallery has full Input / X / Z images for top_n=2 (6 images).
        final_labels, final_images = intermediate_galleries[-1]
        assert len(final_labels) == 6
        assert len(final_images) == 6
        # Search scores were preserved rather than overwritten by redraw's 0.01.
        assert ranked_rv()[0][0] > 0.99

        # Asking for more than top_n (including un-redrawn score_only entries)
        # skips the None z-section entries cleanly instead of crashing.
        wider_labels, wider_images = denovo3d_tab._collect_projection_gallery(
            ranked_rv(), raw_rv(), top_n=5
        )
        assert len(wider_labels) == 6
        assert len(wider_images) == 6

    def test_single_pair_reconstruction_runs_once_without_score_only(self, monkeypatch):
        _FakeProgress.instances.clear()
        monkeypatch.setattr(denovo3d_tab.ui, "Progress", _FakeProgress)
        _disable_real_threads(monkeypatch)

        calls = []

        def fake_process_one_task(*task):
            calls.append(dict(task[-3]))
            return _make_task_result(task, score=0.95, full_projections=True)

        monkeypatch.setattr(
            denovo3d_tab.denovo3d_pipeline, "process_one_task", fake_process_one_task
        )

        tasks = [_make_task(0, 1, 1.2, model="elasticnet", return_3d=True)]
        raw_rv = _ValueBox([])
        ranked_rv = _ValueBox([])
        scan_rv = _ValueBox([])

        asyncio.run(
            denovo3d_tab._run_reconstruction_task(
                tasks,
                cpu=1,
                abort_ref=[False],
                reconstruction_results_raw=raw_rv,
                reconstruction_results=ranked_rv,
                scan_results=scan_rv,
                display_model=None,
                top_n=10,
                lr_algorithm="gauss",
                rec_algorithm="elasticnet",
                is_scan=False,
            )
        )

        assert len(calls) == 1
        assert "score_only" not in calls[0]
        assert ranked_rv()[0][1][3] is not None

    def test_top_n_default_matches_bookmark(self):
        assert denovo3d_bookmark.BOOKMARK_DEFAULTS["top_n"] == (
            "dn_top_n_results",
            10,
        )

    def test_end_to_end_pipeline_score_only_scan_and_top_n_redraw(self, monkeypatch):
        """Exercise real ``denovo3d_pipeline.process_one_task`` with ``score_only``."""
        _FakeProgress.instances.clear()
        monkeypatch.setattr(denovo3d_tab.ui, "Progress", _FakeProgress)
        _disable_real_threads(monkeypatch)

        rng = np.random.default_rng(7)
        image = rng.normal(size=(16, 32)).astype(np.float32)
        twists = [-30.0, -25.0, -20.0]
        tasks = [
            _make_task(i, len(twists), tw, model="gauss", image=image)
            for i, tw in enumerate(twists)
        ]
        raw_rv = _ValueBox([])
        ranked_rv = _ValueBox([])
        scan_rv = _ValueBox([])

        asyncio.run(
            denovo3d_tab._run_reconstruction_task(
                tasks,
                cpu=1,
                abort_ref=[False],
                reconstruction_results_raw=raw_rv,
                reconstruction_results=ranked_rv,
                scan_results=scan_rv,
                display_model=None,
                top_n=1,
                lr_algorithm="gauss",
                rec_algorithm="gauss",
                is_scan=True,
            )
        )

        ranked = ranked_rv()
        assert len(ranked) == 3
        # Top-1 candidate was redrawn with full projections; remaining 2 kept
        # score_only return_data (z-sections None).
        assert ranked[0][1][2] is not None
        assert ranked[1][1][2] is None
        assert ranked[2][1][2] is None

        labels, images = denovo3d_tab._collect_projection_gallery(
            ranked, raw_rv(), top_n=3
        )
        assert len(labels) == 3
        assert len(images) == 3


class TestDenovo3DCompactPlotlyHtml:
    def test_fig_to_html_is_compact_without_inline_plotlyjs(self):
        res = [
            (0.7, None, (None, "", "1", 5.0, 5.0, 1.0, 4.75, 1, 0, 0, 0)),
            (0.9, None, (None, "", "1", 5.0, 5.0, 1.2, 4.75, 1, 0, 0, 0)),
            (0.8, None, (None, "", "1", 5.0, 5.0, 1.4, 4.75, 1, 0, 0, 0)),
        ]
        fig = denovo3d_ui._scores_figure(res)
        html = str(
            denovo3d_ui._fig_to_html(fig, click_input="denovo3d-dn_pick_candidate")
        )
        assert len(html) < 50_000
        assert "Plotly.newPlot" in html
        assert "Shiny.setInputValue('denovo3d-dn_pick_candidate'" in html
