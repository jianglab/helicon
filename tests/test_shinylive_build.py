"""The browser-safe tabs actually run with no server once exported.

``shinylive export`` turns a tab into a static site where Shiny runs inside
Pyodide in the visitor's own browser. This builds the HelicalLattice bundle
(tools/shinylive/build.py), serves it from a plain file server exactly as a
static host would, and drives it with Playwright the same way the server
based web-app tests drive ``helicon webApps`` -- proof that a change to the
tab still exports and still reacts, not just that the build command exits 0.

``shinylive`` is a build-time tool, not a ``helicon`` dependency (see
tools/shinylive/README.md), and the first run downloads its web assets
(~100 MB) from GitHub -- this test is marked ``browser`` (it uses the ``page``
fixture) and is also ``slow``, so it is already outside the everyday and CI
fast runs; add it to ``tests/slow_tests.txt`` if ``update_slow_tests.py``
does not pick it up on its own.
"""

import shutil
import sys
import threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import pytest
from playwright.sync_api import Page

pytest.importorskip("shinylive")
# the package alone is not enough: the export runs its command-line tool
if shutil.which("shinylive") is None:
    pytest.skip("the shinylive command is not on PATH", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).parents[1] / "tools" / "shinylive"))
import build as shinylive_build  # noqa: E402

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def helical_lattice_site(tmp_path_factory):
    """Export HelicalLattice and serve it; yield the site's base URL."""
    out_dir = shinylive_build.build_bundle(
        shinylive_build.BUNDLES["helical_lattice"],
        out_dir=tmp_path_factory.mktemp("helical_lattice_site"),
        staging_dir=tmp_path_factory.mktemp("helical_lattice_staging"),
    )

    class _Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(out_dir), **kwargs)

        def log_message(self, *args):  # quiet: pytest captures stdout anyway
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/"
    finally:
        server.shutdown()
        thread.join()


def _serve(out_dir):
    """Serve *out_dir* from a plain file server; yield its base URL."""

    class _Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(out_dir), **kwargs)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/"
    finally:
        server.shutdown()
        thread.join()


@pytest.fixture(scope="module")
def helical_projection_site(tmp_path_factory):
    """Export HelicalProjection (mode="package": the real helicon wheel) and
    serve it; yield the site's base URL."""
    wheel = shinylive_build.build_wheel(out_dir=tmp_path_factory.mktemp("wheel"))
    out_dir = shinylive_build.build_bundle(
        shinylive_build.BUNDLES["helical_projection"],
        out_dir=tmp_path_factory.mktemp("helical_projection_site"),
        staging_dir=tmp_path_factory.mktemp("helical_projection_staging"),
        wheel=wheel,
    )
    yield from _serve(out_dir)


@pytest.fixture(scope="module")
def hi3d_site(tmp_path_factory):
    """Export HI3D (mode="package", bokeh) and serve it; yield its base URL."""
    wheel = shinylive_build.build_wheel(out_dir=tmp_path_factory.mktemp("wheel"))
    out_dir = shinylive_build.build_bundle(
        shinylive_build.BUNDLES["hi3d"],
        out_dir=tmp_path_factory.mktemp("hi3d_site"),
        staging_dir=tmp_path_factory.mktemp("hi3d_staging"),
        wheel=wheel,
    )
    yield from _serve(out_dir)


@pytest.fixture(scope="module")
def where_is_my_class_site(tmp_path_factory):
    """Export whereIsMyClass (mode="package", plotly+shinywidgets) and serve
    it; yield its base URL."""
    wheel = shinylive_build.build_wheel(out_dir=tmp_path_factory.mktemp("wheel"))
    out_dir = shinylive_build.build_bundle(
        shinylive_build.BUNDLES["where_is_my_class"],
        out_dir=tmp_path_factory.mktemp("where_is_my_class_site"),
        staging_dir=tmp_path_factory.mktemp("where_is_my_class_staging"),
        wheel=wheel,
    )
    yield from _serve(out_dir)


class TestHelicalLatticeExport:
    def test_builds_a_static_site(self, helical_lattice_site):
        # the fixture itself raises (CalledProcessError) if export fails;
        # reaching here means shinylive export exited 0
        assert helical_lattice_site.startswith("http://127.0.0.1:")

    def test_runs_with_no_server_and_reacts_to_input(
        self, page: Page, helical_lattice_site
    ):
        # shinylive renders the exported app inside an iframe of its own (a
        # fresh, randomly-named path each load, for the cross-origin
        # isolation Pyodide's worker needs) -- reach the app's own DOM
        # through it rather than the outer page.
        page.goto(helical_lattice_site)
        app = page.frame_locator("iframe.app-frame")
        # Pyodide boot + package install, not a page load: generous timeout
        twist_input = app.locator("input#helical_lattice_tab-twist")
        twist_input.wait_for(timeout=60_000)
        twist_input.click()
        twist_input.fill("")
        twist_input.type("-29")
        page.keyboard.press("Enter")
        app.locator("text=twist=-29.00").first.wait_for(timeout=15_000)

    def test_no_console_errors(self, page: Page, helical_lattice_site):
        errors = []
        page.on(
            "console", lambda m: errors.append(m.text) if m.type == "error" else None
        )
        page.goto(helical_lattice_site)
        app = page.frame_locator("iframe.app-frame")
        app.locator("input#helical_lattice_tab-twist").wait_for(timeout=60_000)
        assert errors == []


class TestHelicalProjectionExport:
    """mode="package": the real helicon wheel, built and micropip-installed,
    not a vendored copy -- the harder of the two bundling modes, and the one
    that found the quirks tools/shinylive/README.md documents (a transitive
    dependency of a URL-installed wheel is not force-loaded; a staged
    requirement is not readable, only a served one)."""

    def test_builds_a_static_site(self, helical_projection_site):
        assert helical_projection_site.startswith("http://127.0.0.1:")

    def test_runs_with_no_server(self, page: Page, helical_projection_site):
        # a heavier install than HelicalLattice's (scipy, scikit-image,
        # mrcfile, ...): a generous timeout, same reasoning. "text=" would
        # also match the (hidden) <title>, so this looks for a button instead
        page.goto(helical_projection_site)
        app = page.frame_locator("iframe.app-frame")
        app.get_by_role("button", name="Compare projections").wait_for(timeout=120_000)

    def test_switching_to_upload_mode_shows_a_file_input(
        self, page: Page, helical_projection_site
    ):
        page.goto(helical_projection_site)
        app = page.frame_locator("iframe.app-frame")
        app.get_by_role("button", name="Compare projections").wait_for(timeout=120_000)
        # "upload" also names the 3D-maps panel's own radio; the 2D-images
        # one (the active tab by default) is the first input of that value.
        # A native element.click() via evaluate, not Playwright's synthetic
        # pointer click (even with .check()/force=True): against this
        # Pyodide-rendered radio, the synthetic click intermittently leaves
        # the input unchecked (reproduced directly, ~3/4 of the time)
        # while a real DOM click() is reliably picked up by React's handler.
        radio = app.locator('input[type="radio"][value="upload"]').first
        radio.wait_for(timeout=20_000)
        radio.evaluate("el => el.click()")
        app.locator("input[type=file]").first.wait_for(timeout=20_000)


class TestHI3DExport:
    """mode="package" with a bokeh (not plotly/shinywidgets) rendering path --
    the real blocker this one found was unrelated to bokeh itself:
    helicon.lib.cache's background pruning sweep span a raw
    threading.Thread(...).start(), which raises under Pyodide's single
    thread; fixed by routing it through system.has_real_threads()."""

    def test_builds_a_static_site(self, hi3d_site):
        assert hi3d_site.startswith("http://127.0.0.1:")

    def test_runs_with_no_server(self, page: Page, hi3d_site):
        # "Refine twist/rise" only renders after a map is loaded and its ACF
        # computed; "Change EMDB ID" is the default input mode's own button,
        # present from the first render
        page.goto(hi3d_site)
        app = page.frame_locator("iframe.app-frame")
        app.get_by_role("button", name="Change EMDB ID").wait_for(timeout=120_000)

    def test_no_tracebacks(self, page: Page, hi3d_site):
        # a preload-time traceback (import side effects, a crashed background
        # task) shows up as a console error, not a failed Python exception
        # Playwright could catch directly. Not "no console errors": every
        # mode="package" bundle that pulls in pandas also has Pyodide try (and
        # fail) to fetch the optional pyarrow/pyodide-unix-timezones packages,
        # which is benign noise unrelated to anything helicon does.
        errors = []
        page.on(
            "console", lambda m: errors.append(m.text) if m.type == "error" else None
        )
        page.goto(hi3d_site)
        app = page.frame_locator("iframe.app-frame")
        app.get_by_role("button", name="Change EMDB ID").wait_for(timeout=120_000)
        page.wait_for_timeout(5_000)
        tracebacks = [e for e in errors if "Traceback" in e]
        assert tracebacks == []


@pytest.fixture(scope="module")
def hill_site(tmp_path_factory):
    """Export HILL (mode="package", bokeh rendered through shinywidgets) and
    serve it; yield its base URL."""
    wheel = shinylive_build.build_wheel(out_dir=tmp_path_factory.mktemp("wheel"))
    out_dir = shinylive_build.build_bundle(
        shinylive_build.BUNDLES["hill"],
        out_dir=tmp_path_factory.mktemp("hill_site"),
        staging_dir=tmp_path_factory.mktemp("hill_staging"),
        wheel=wheel,
    )
    yield from _serve(out_dir)


@pytest.fixture(scope="module")
def helical_pitch_site(tmp_path_factory):
    """Export HelicalPitch and serve it; yield its base URL."""
    wheel = shinylive_build.build_wheel(out_dir=tmp_path_factory.mktemp("wheel"))
    out_dir = shinylive_build.build_bundle(
        shinylive_build.BUNDLES["helical_pitch"],
        out_dir=tmp_path_factory.mktemp("helical_pitch_site"),
        staging_dir=tmp_path_factory.mktemp("helical_pitch_staging"),
        wheel=wheel,
    )
    yield from _serve(out_dir)


@pytest.fixture(scope="module")
def abinitio3d_site(tmp_path_factory):
    """Export abinitio3D and serve it; yield its base URL."""
    wheel = shinylive_build.build_wheel(out_dir=tmp_path_factory.mktemp("wheel"))
    out_dir = shinylive_build.build_bundle(
        shinylive_build.BUNDLES["abinitio3d"],
        out_dir=tmp_path_factory.mktemp("abinitio3d_site"),
        staging_dir=tmp_path_factory.mktemp("abinitio3d_staging"),
        wheel=wheel,
    )
    yield from _serve(out_dir)


@pytest.fixture(scope="module")
def denovo3d_site(tmp_path_factory):
    """Export denovo3D and serve it; yield its base URL."""
    wheel = shinylive_build.build_wheel(out_dir=tmp_path_factory.mktemp("wheel"))
    out_dir = shinylive_build.build_bundle(
        shinylive_build.BUNDLES["denovo3d"],
        out_dir=tmp_path_factory.mktemp("denovo3d_site"),
        staging_dir=tmp_path_factory.mktemp("denovo3d_staging"),
        wheel=wheel,
    )
    yield from _serve(out_dir)


@pytest.fixture(scope="module")
def home_site(tmp_path_factory):
    """Export Home and serve it; yield its base URL."""
    wheel = shinylive_build.build_wheel(out_dir=tmp_path_factory.mktemp("wheel"))
    out_dir = shinylive_build.build_bundle(
        shinylive_build.BUNDLES["home"],
        out_dir=tmp_path_factory.mktemp("home_site"),
        staging_dir=tmp_path_factory.mktemp("home_staging"),
        wheel=wheel,
    )
    yield from _serve(out_dir)


class TestWhereIsMyClassExport:
    """mode="package" with plotly + shinywidgets (anywidget/ipywidgets/
    traitlets/psygnal, ...) -- the untested risk the survey flagged; all of
    it installs and imports cleanly under Pyodide."""

    def test_builds_a_static_site(self, where_is_my_class_site):
        assert where_is_my_class_site.startswith("http://127.0.0.1:")

    def test_runs_with_no_server(self, page: Page, where_is_my_class_site):
        page.goto(where_is_my_class_site)
        app = page.frame_locator("iframe.app-frame")
        app.get_by_role("button", name="Run").wait_for(timeout=120_000)

    def test_no_tracebacks(self, page: Page, where_is_my_class_site):
        # see TestHI3DExport.test_no_tracebacks: not "no console errors",
        # since the pyarrow/pyodide-unix-timezones fetch failure is benign
        # noise common to every mode="package" bundle that pulls in pandas
        errors = []
        page.on(
            "console", lambda m: errors.append(m.text) if m.type == "error" else None
        )
        page.goto(where_is_my_class_site)
        app = page.frame_locator("iframe.app-frame")
        app.get_by_role("button", name="Run").wait_for(timeout=120_000)
        page.wait_for_timeout(5_000)
        tracebacks = [e for e in errors if "Traceback" in e]
        assert tracebacks == []


# Chromium cannot drive the browser's own folder dialog, so the page is given a
# stand-in for it that returns a real directory handle: a folder of the page's
# origin-private file system, filled with the project below. Everything after the
# dialog (the handle through IndexedDB to Pyodide's worker, the read-only mount, the
# file dialog of the app, the loading of the files) is the real thing.
_PICK_A_FOLDER = """
window.__project = %s;
window.showDirectoryPicker = async () => {
  const root = await navigator.storage.getDirectory();
  const dir = await root.getDirectoryHandle("myproject", {create: true});
  for (const [path, b64] of Object.entries(window.__project)) {
    const parts = path.split("/");
    let d = dir;
    for (const part of parts.slice(0, -1)) {
      d = await d.getDirectoryHandle(part, {create: true});
    }
    const w = await (await d.getFileHandle(parts.at(-1), {create: true})).createWritable();
    await w.write(Uint8Array.from(atob(b64), c => c.charCodeAt(0)));
    await w.close();
  }
  return dir;
};
"""


def _class2d_project(n_classes=3, box=32):
    """A tiny RELION Class2D job: ``{path in the folder: bytes}``."""
    import tempfile

    import mrcfile
    import numpy as np

    images = np.random.default_rng(0).normal(size=(n_classes, box, box))
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "classes.mrcs"
        with mrcfile.new(str(path)) as mrc:
            mrc.set_data(images.astype(np.float32))
            mrc.voxel_size = (2.0, 2.0, 2.0)
        classes = path.read_bytes()
    rows = [
        f"{i + 1:06d}@Extract/job001/particles.mrcs MicrographsA/mic{i // 6}.mrc "
        f"{1 + i % n_classes} {1 + i // 6} {100.0 + 10 * i:.1f} 50.0 {20.0 * (i % 6):.1f}"
        for i in range(12)
    ]
    star = (
        "\ndata_particles\n\nloop_\n_rlnImageName #1\n_rlnMicrographName #2\n"
        "_rlnClassNumber #3\n_rlnHelicalTubeID #4\n_rlnCoordinateX #5\n"
        "_rlnCoordinateY #6\n_rlnHelicalTrackLengthAngst #7\n" + "\n".join(rows) + "\n"
    )
    return {
        "Class2D/job001/run_it025_data.star": star.encode(),
        "Class2D/job001/run_it025_classes.mrcs": classes,
    }


def _open_with_a_folder_to_give(page: Page, site):
    """Open the exported app at ``site`` with a project the page can be given.

    Returns
    -------
    playwright.sync_api.FrameLocator
        The app, once it shows its "local folder" input source.
    """
    import base64
    import json

    project = {
        path: base64.b64encode(data).decode()
        for path, data in _class2d_project().items()
    }
    page.add_init_script(_PICK_A_FOLDER % json.dumps(project))
    # the example data the tabs start with are public files: these tests do not need
    # them, and one that fails to arrive shows a popup that holds up the page
    page.route(
        "**/*",
        lambda route: (
            route.abort() if "ebi.ac.uk" in route.request.url else route.continue_()
        ),
    )
    app = page.frame_locator("iframe.app-frame")
    # the first visit may stall while the service worker takes over: look again
    for _ in range(3):
        page.goto(site)
        try:
            app.get_by_label("local folder").first.wait_for(timeout=120_000)
            return app
        except Exception:
            pass
    raise AssertionError("the app did not start")


def _dismiss_popups(page: Page, app):
    """Close what a failed example download left over the page."""
    app.locator("body").evaluate(
        """() => {
          document.querySelectorAll('.modal, .modal-backdrop').forEach(e => e.remove());
          document.body.classList.remove('modal-open');
          document.body.style.overflow = '';
        }"""
    )


# the tabs that start by downloading example data
_LOADS_EXAMPLES = ("helical_pitch", "abinitio3d", "helical_projection", "hi3d")


class TestEveryTabLocalFolder:
    """The "server" input source of every tab with one is the folder of the visitor."""

    @pytest.mark.parametrize(
        "tab",
        [
            "where_is_my_class",
            "helical_pitch",
            "abinitio3d",
            "hi3d",
            "hill",
            "denovo3d",
            "helical_projection",
        ],
    )
    def test_a_folder_is_given_and_browsed(self, tab, page: Page, request):
        site = request.getfixturevalue(f"{tab}_site")
        app = _open_with_a_folder_to_give(page, site)
        if tab in _LOADS_EXAMPLES:
            try:
                app.locator("#shiny-modal").first.wait_for(
                    state="attached", timeout=120_000
                )
                page.wait_for_timeout(8_000)  # the second file's popup
            except Exception:
                pass
            _dismiss_popups(page, app)
        app.get_by_label("local folder").first.check()
        app.get_by_role("button", name="Choose local folder").first.click()
        app.locator(".shiny-notification").filter(has_text="2 files").wait_for(
            timeout=60_000
        )
        _dismiss_popups(page, app)  # one may have come while the folder was read
        app.locator(".hfp-field button").first.click()
        app.locator(".hfp-row").first.wait_for(timeout=30_000)
        # the folders given, and nothing above them
        assert app.locator(".hfp").get_attribute("data-cwd") == "/local"
        names = app.locator(".hfp-row").evaluate_all(
            "rs => rs.map(r => r.dataset.name)"
        )
        assert names == ["myproject"]
        assert app.locator(".hfp-crumb").evaluate_all("cs => cs.map(c => c.title)") == [
            "/local"
        ]


class TestWhereIsMyClassLocalFolder:
    """A file of a chosen folder loads, as a file on a server does."""

    def _open(self, page: Page, site):
        return _open_with_a_folder_to_give(page, site)

    def test_a_file_of_a_chosen_folder_loads(self, page: Page, where_is_my_class_site):
        app = self._open(page, where_is_my_class_site)
        # no server files in a browser-side copy: the folder takes their place
        modes = app.locator("input[name$=wimc_input_mode]")
        assert modes.evaluate_all("els => els.map(e => e.value)") == ["server", "url"]
        app.get_by_label("local folder").check()
        browse = app.locator("button[id$=star_browse-open]")

        # nothing to browse before a folder is given
        browse.click()
        app.locator(".shiny-notification").filter(
            has_text="no folder to browse yet"
        ).wait_for(timeout=30_000)

        app.get_by_role("button", name="Choose local folder").click()
        app.locator(".shiny-notification").filter(has_text="2 files").wait_for(
            timeout=60_000
        )

        # the dialog shows the folders given and nothing above them
        browse.click()
        app.locator(".hfp-row").first.wait_for(timeout=30_000)
        assert app.locator(".hfp").get_attribute("data-cwd") == "/local"
        names = app.locator(".hfp-row").evaluate_all(
            "rs => rs.map(r => r.dataset.name)"
        )
        assert names == ["myproject"]
        assert app.locator(".hfp-crumb").evaluate_all("cs => cs.map(c => c.title)") == [
            "/local"
        ]
        app.locator(".hfp-row[data-name=myproject]").dblclick()
        app.locator(".hfp-row[data-name=Class2D]").dblclick()
        app.locator(".hfp-row[data-name=job001]").dblclick()
        app.locator(".hfp-row[data-name='run_it025_data.star']").dblclick()
        app.locator(".modal").wait_for(state="detached")
        field = app.locator("input[id$=wimc_server_star]")
        assert field.input_value() == (
            "/local/myproject/Class2D/job001/run_it025_data.star"
        )

        # and the file loads: its class averages are read from the same folder
        app.get_by_role("button", name="Run").click()
        app.get_by_text("3/3 classes | 32x32 pixels").wait_for(timeout=60_000)
        assert app.locator(".modal").count() == 0


class TestHILLExport:
    """mode="package" combining bokeh and shinywidgets (output_widget/
    render_widget), neither alone as HI3D and where_is_my_class each use --
    the real blocker this one found was jupyter_bokeh (shinywidgets' own
    BokehModel wrapper) not being one of shinywidgets' transitive
    dependencies, so it needs listing explicitly. finufft (no Pyodide/wasm
    wheel) is not a blocker: hill_compute.fft_rescale falls back to
    helicon.lib.transforms._nonuniform_grid_dft, an exact direct DFT, when
    finufft is unavailable."""

    def test_builds_a_static_site(self, hill_site):
        assert hill_site.startswith("http://127.0.0.1:")

    def test_runs_with_no_server(self, page: Page, hill_site):
        page.goto(hill_site)
        app = page.frame_locator("iframe.app-frame")
        app.get_by_role("button").first.wait_for(timeout=120_000)

    def test_the_page_loads_bokeh_for_the_figures(self, page: Page, hill_site):
        # `helicon webApps` puts BokehJS in its own page for the tab; alone,
        # the bundle's page must, or no figure is ever drawn
        page.goto(hill_site)
        app = page.frame_locator("iframe.app-frame")
        app.get_by_role("button").first.wait_for(timeout=120_000)
        frame = page.frames[-1]
        frame.wait_for_function("typeof window.Bokeh === 'object'", timeout=60_000)

    def test_no_tracebacks(self, page: Page, hill_site):
        # see TestHI3DExport.test_no_tracebacks
        errors = []
        page.on(
            "console", lambda m: errors.append(m.text) if m.type == "error" else None
        )
        page.goto(hill_site)
        app = page.frame_locator("iframe.app-frame")
        app.get_by_role("button").first.wait_for(timeout=120_000)
        page.wait_for_timeout(5_000)
        tracebacks = [e for e in errors if "Traceback" in e]
        assert tracebacks == []


class TestHelicalPitchExport:
    """mode="package"; starfile and scikit-image (skimage.transform.iradon,
    reached via helical_pitch_map through helical_pitch_mapfit/
    helical_pitch_phase) are its own extra requirements beyond
    helical_projection's."""

    def test_builds_a_static_site(self, helical_pitch_site):
        assert helical_pitch_site.startswith("http://127.0.0.1:")

    def test_runs_with_no_server(self, page: Page, helical_pitch_site):
        page.goto(helical_pitch_site)
        app = page.frame_locator("iframe.app-frame")
        app.get_by_role("button").first.wait_for(timeout=120_000)

    def test_no_tracebacks(self, page: Page, helical_pitch_site):
        errors = []
        page.on(
            "console", lambda m: errors.append(m.text) if m.type == "error" else None
        )
        page.goto(helical_pitch_site)
        app = page.frame_locator("iframe.app-frame")
        app.get_by_role("button").first.wait_for(timeout=120_000)
        page.wait_for_timeout(5_000)
        tracebacks = [e for e in errors if "Traceback" in e]
        assert tracebacks == []


class TestAbinitio3DExport:
    """mode="package"; shares helical_pitch_*'s own lib modules, so the same
    extra requirements (starfile, scikit-image) apply."""

    def test_builds_a_static_site(self, abinitio3d_site):
        assert abinitio3d_site.startswith("http://127.0.0.1:")

    def test_runs_with_no_server(self, page: Page, abinitio3d_site):
        page.goto(abinitio3d_site)
        app = page.frame_locator("iframe.app-frame")
        app.get_by_role("button").first.wait_for(timeout=120_000)

    def test_no_tracebacks(self, page: Page, abinitio3d_site):
        errors = []
        page.on(
            "console", lambda m: errors.append(m.text) if m.type == "error" else None
        )
        page.goto(abinitio3d_site)
        app = page.frame_locator("iframe.app-frame")
        app.get_by_role("button").first.wait_for(timeout=120_000)
        page.wait_for_timeout(5_000)
        tracebacks = [e for e in errors if "Traceback" in e]
        assert tracebacks == []


class TestDenovo3DExport:
    """mode="package"; the numba kernels of denovo3d_solver.py
    (build_A_helical_sym_matrix/build_A_data_matrix's loop_kji) now have
    numpy fallbacks (_HAS_NUMBA dispatch), which is what unblocked this
    bundle -- scikit-learn (regression-based rise/twist fits) and joblib
    (an on-disk cache only, no parallel execution) are its own extra
    requirements beyond helical_projection's."""

    def test_builds_a_static_site(self, denovo3d_site):
        assert denovo3d_site.startswith("http://127.0.0.1:")

    def test_runs_with_no_server(self, page: Page, denovo3d_site):
        page.goto(denovo3d_site)
        app = page.frame_locator("iframe.app-frame")
        app.get_by_role("button").first.wait_for(timeout=120_000)

    def test_no_tracebacks(self, page: Page, denovo3d_site):
        errors = []
        page.on(
            "console", lambda m: errors.append(m.text) if m.type == "error" else None
        )
        page.goto(denovo3d_site)
        app = page.frame_locator("iframe.app-frame")
        app.get_by_role("button").first.wait_for(timeout=120_000)
        page.wait_for_timeout(5_000)
        tracebacks = [e for e in errors if "Traceback" in e]
        assert tracebacks == []


class TestHomeExport:
    """mode="package"; pure shiny + stdlib, nothing it reaches is gated
    behind a heavy optional import -- the interesting part is its own click
    wiring, not the export itself: deployment.is_browser() makes an
    integrated app's chip link to its sibling bundle instead of a navbar
    tab that would not exist on this single-tab page (see home_tab._app_svg
    and tests/test_deployment.py's own, no-export unit tests of that)."""

    def test_builds_a_static_site(self, home_site):
        assert home_site.startswith("http://127.0.0.1:")

    def test_runs_with_no_server(self, page: Page, home_site):
        page.goto(home_site)
        app = page.frame_locator("iframe.app-frame")
        app.locator(".hh-title", has_text="Helicon data processing workflow").wait_for(
            timeout=120_000
        )

    def test_an_integrated_apps_chip_links_to_its_sibling_bundle(
        self, page: Page, home_site
    ):
        page.goto(home_site)
        app = page.frame_locator("iframe.app-frame")
        app.locator(".hh-title", has_text="Helicon data processing workflow").wait_for(
            timeout=120_000
        )
        chip = app.locator('[data-name="HelicalProjection"]')
        assert chip.get_attribute("data-url") == "helical_projection/"
        assert chip.get_attribute("data-tab") is None

    def test_no_tracebacks(self, page: Page, home_site):
        errors = []
        page.on(
            "console", lambda m: errors.append(m.text) if m.type == "error" else None
        )
        page.goto(home_site)
        app = page.frame_locator("iframe.app-frame")
        app.locator(".hh-title", has_text="Helicon data processing workflow").wait_for(
            timeout=120_000
        )
        page.wait_for_timeout(5_000)
        tracebacks = [e for e in errors if "Traceback" in e]
        assert tracebacks == []
