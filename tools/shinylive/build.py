#!/usr/bin/env python

"""Export selected `helicon webApps` tabs as standalone Shinylive bundles.

Each bundle is a tiny Shiny app of its own -- an ``app.py`` that wires up one
tab's ``<tab>_ui``/``<tab>_server`` pair. Two ways a bundle gets that tab's
code, picked per :class:`Bundle` by its ``mode``:

- ``"vendor"``: the tab's own source file, copied in verbatim, with no
  dependency on the ``helicon`` package at all. For a tab that needs nothing
  from ``helicon`` (as HelicalLattice does not).
- ``"package"``: the real ``helicon`` wheel, built from this checkout and
  installed by micropip like any other requirement, so ``app.py`` imports the
  tab the ordinary way (``from helicon.webApps.tabs.<tab> import ...``). This
  needs nothing special from the tab's own code -- what makes it possible is
  that numba, finufft, cryosparc-tools and psutil are now optional extras
  (pyproject.toml), not hard dependencies micropip cannot satisfy in Pyodide
  (none has a Pyodide wheel); every call site already tolerates their
  absence. See this directory's README.md.

``shinylive export`` then turns the staged app directory into a static site
that runs entirely in the browser.

Run directly: ``python tools/shinylive/build.py [bundle ...]`` (default: every
bundle in :data:`BUNDLES`). Needs the ``shinylive`` package (``pip install
shinylive``, not a `helicon` dependency -- it is a build-time tool, not
something an exported app needs); the first run downloads its web assets
(~100 MB) and caches them in ``~/.cache/shinylive`` (Linux) or the
platform equivalent.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src" / "helicon"
BUILD_DIR = REPO_ROOT / "build" / "shinylive"


@dataclass
class Bundle:
    """One tab, exported as its own Shinylive site.

    Parameters
    ----------
    name : str
        Output directory name (``build/shinylive/<name>/``).
    title : str
        The tab's display name, as `helicon webApps` shows it -- used for
        the bundle's page title and the landing page :func:`write_index`
        writes.
    module : str
        ``mode="vendor"``: the vendored module's import name inside the
        bundle (also its filename stem). ``mode="package"``: the tab
        module's full dotted path inside the real package, e.g.
        ``"helicon.webApps.tabs.helical_projection_tab"``. Either way,
        ``app.py`` imports ``<last component>_ui``/``_server`` from it.
    requirements : tuple of str
        Packages ``app.py`` needs beyond what Shinylive's Pyodide image
        already carries (``shiny`` itself and its own dependencies need not
        be listed -- they are already installed in every Shinylive image).
        ``mode="package"`` needs the real ``helicon``'s own dependencies
        listed too (whatever is not already in the Shinylive image);
        :func:`build_bundle` prepends the wheel itself.
    mode : str, optional
        ``"vendor"`` (default) or ``"package"``; see the module docstring.
    source : Path, optional
        ``mode="vendor"`` only: the tab source file to copy in verbatim,
        relative to ``src/helicon``.
    extra_files : tuple of (Path, str), optional
        ``mode="vendor"`` only: further ``(source, dest_name)`` pairs to
        copy into the bundle alongside ``module``, for a tab that is not a
        single file.
    is_module : bool, optional
        Whether ``<last component>_ui``/``_server`` are a Shiny module pair
        (``_ui`` takes a namespace id, ``_server`` is wrapped in
        ``@module.server``) -- true for every tab but Home, which is a
        singleton rendered straight into the page (``_ui()`` takes no id;
        ``_server(input, session)`` has no ``output``, since it never
        renders anything of its own). Picks which of the two ``app.py``
        templates :func:`_stage` writes.
    head : str, optional
        ``"<module>:<function>"`` of a function whose result (head content)
        goes into the bundle's page, for scripts the tab needs that the
        page of `helicon webApps` loads for it rather than the tab itself
        (HILL: BokehJS, without which its figures are never drawn).
    """

    name: str
    title: str
    module: str
    requirements: tuple[str, ...]
    mode: str = "vendor"
    source: Path | None = None
    extra_files: tuple[tuple[Path, str], ...] = field(default_factory=tuple)
    is_module: bool = True
    head: str = ""

    def __post_init__(self):
        if self.mode not in ("vendor", "package"):
            raise ValueError(
                f"Bundle.mode must be 'vendor' or 'package', not {self.mode!r}"
            )
        if self.mode == "vendor" and self.source is None:
            raise ValueError("mode='vendor' needs a source file")


BUNDLES: dict[str, Bundle] = {
    "helical_lattice": Bundle(
        name="helical_lattice",
        title="HelicalLattice",
        mode="vendor",
        module="helical_lattice_tab",
        source=Path("webApps/tabs/helical_lattice_tab.py"),
        # numpy/pandas/plotly only (see the module's own imports); shiny
        # itself comes from the Shinylive image.
        requirements=("plotly",),
    ),
    "helical_projection": Bundle(
        name="helical_projection",
        title="HelicalProjection",
        mode="package",
        module="helicon.webApps.tabs.helical_projection_tab",
        # beyond helicon's own base dependencies, which _stage adds via
        # wheel_base_requirements(): psutil is deliberately not among them
        # (it has no Pyodide wheel, and every reader of it already falls
        # back gracefully without it -- helicon.lib.system._psutil_or_none())
        # -- and torch is reached only by map_gauss_fit.GaussSet.to_set(),
        # which this tab never calls, so it is left out too.
        requirements=("plotly", "Pillow", "threadpoolctl"),
    ),
    "hi3d": Bundle(
        name="hi3d",
        title="HI3D",
        mode="package",
        module="helicon.webApps.tabs.hi3d_tab",
        # bokeh (not plotly/shinywidgets): HI3D's plots are all bokeh
        # figures embedded via bokeh.embed.components (with window.Bokeh
        # injected once into <head> via helicon.shiny.bokeh_dependency()).
        # trackpy (not in the Shinylive image) is optional: hi3d_core.find_peaks
        # falls back to a pure numpy/scipy 2D Crocker-Grier peak detector and
        # centre-of-mass refiner when trackpy is absent. mrcfile is already one
        # of helicon's base dependencies (wheel_base_requirements adds it).
        requirements=("bokeh",),
    ),
    "where_is_my_class": Bundle(
        name="where_is_my_class",
        title="whereIsMyClass",
        mode="package",
        module="helicon.webApps.tabs.where_is_my_class_tab",
        requirements=("plotly", "shinywidgets"),
    ),
    "helical_pitch": Bundle(
        name="helical_pitch",
        title="HelicalPitch",
        mode="package",
        module="helicon.webApps.tabs.helical_pitch_tab",
        # starfile (RELION .star I/O) and scikit-image (helical_pitch_map's
        # skimage.transform.iradon back-projection, reached via
        # helical_pitch_mapfit/helical_pitch_phase) are this tab's own extra
        # requirements beyond helical_projection's (plotly, threadpoolctl).
        # Its numba kernels (helical_pitch_context._period_scores) and its
        # ProcessPoolExecutor/ThreadPoolExecutor sites (helical_pitch_phase,
        # helical_pitch_mixture, helical_pitch_map, pools.TaskPool) all
        # already fall back through _HAS_NUMBA / has_real_threads+
        # pool_executor when unavailable, as under Pyodide.
        requirements=("plotly", "threadpoolctl", "starfile", "scikit-image"),
    ),
    "abinitio3d": Bundle(
        name="abinitio3d",
        title="abinitio3D",
        mode="package",
        module="helicon.webApps.tabs.abinitio3d_tab",
        # shares helical_pitch_*'s own lib modules (see "helical_pitch"
        # above), so the same extra requirements apply.
        requirements=("plotly", "threadpoolctl", "starfile", "scikit-image"),
    ),
    "denovo3d": Bundle(
        name="denovo3d",
        title="denovo3D",
        mode="package",
        module="helicon.webApps.tabs.denovo3d_tab",
        # scikit-learn (and its dependency joblib): denovo3d_solver's
        # regression-based rise/twist fits; helicon.lib.cache bypasses
        # joblib.Memory on Emscripten via DummyMemory so nothing pickles to
        # ephemeral WASM MEMFS. Its numba kernels, _apply_helical_symmetry_np
        # (Z-window culled), and score_only parameter scans all run via fast
        # single-core numpy/scipy paths when numba/threads are unavailable.
        requirements=(
            "plotly",
            "threadpoolctl",
            "scikit-image",
            "scikit-learn",
            "joblib",
        ),
    ),
    "hill": Bundle(
        name="hill",
        title="HILL",
        mode="package",
        module="helicon.webApps.tabs.hill_tab",
        # bokeh figures rendered through shinywidgets' output_widget/
        # render_widget (not bokeh.embed.components, as HI3D uses, and not
        # plotly, as where_is_my_class uses) -- shinywidgets wraps a bokeh
        # figure as an ipywidget via jupyter_bokeh (BokehModel), which is not
        # one of shinywidgets' own transitive dependencies, so it needs
        # listing explicitly (pure-python wheel, installs fine under
        # Pyodide), while bokeh_dependency injects BokehJS inline into <head>
        # without an external CDN fetch. finufft (no Pyodide/wasm wheel) is
        # not needed: hill_compute.fft_rescale falls back to
        # helicon.lib.transforms._nonuniform_grid_dft, an exact direct DFT,
        # when finufft is unavailable.
        requirements=("bokeh", "shinywidgets", "jupyter_bokeh"),
        head="helicon.webApps.tabs.hill_bookmark:bokeh_dependency",
    ),
    "home": Bundle(
        name="home",
        title="Home",
        mode="package",
        module="helicon.webApps.tabs.home_tab",
        # pure shiny + stdlib (deployment.py, imported as a package sibling,
        # is lightweight too: see its own module docstring) -- no extra
        # requirements of its own. Its reactive.effect that would spawn a
        # helicon subcommand as a detached process never runs: it returns at
        # once under deployment.is_cloud(), true here by is_browser() alone,
        # before any of the heavy imports (helicon.lib.terminal, ...) it
        # only reaches past that check.
        requirements=(),
        # not a Shiny module -- see Bundle.is_module's docstring.
        is_module=False,
    ),
}


_APP_PY_TEMPLATE = '''\
"""Standalone Shinylive app: the {title} tab of `helicon webApps`, alone.

Generated by tools/shinylive/build.py -- do not edit the copy in
build/shinylive/ by hand; edit the source tab module and re-run the build.
"""

from shiny import App, ui

from {module} import {attr}_ui, {attr}_server
{head_import}
# page_fillable, not page_fluid: the tab's sidebar layout then fills the window
# and its sidebar (and main panel) scroll on their own, as in `helicon
# webApps`. In a fluid page the layout grows to its content, the window is
# never bounded, and no sidebar scrolls.
app_ui = ui.page_fillable(
    ui.tags.title("{title} — Helicon"),
{head_call}    {attr}_ui("{attr}"),
)


def server(input, output, session):
    {attr}_server("{attr}")


app = App(app_ui, server)
'''

# A tab that is not a Shiny module (currently just Home -- see
# Bundle.is_module's docstring): _ui takes no id, and _server has no
# `output` (it renders nothing through output; only reactive.effect).
_APP_PY_TEMPLATE_PLAIN = '''\
"""Standalone Shinylive app: the {title} tab of `helicon webApps`, alone.

Generated by tools/shinylive/build.py -- do not edit the copy in
build/shinylive/ by hand; edit the source tab module and re-run the build.
"""

from shiny import App, ui

from {module} import {attr}_ui, {attr}_server
{head_import}
app_ui = ui.page_fluid(
    ui.tags.title("{title} — Helicon"),
{head_call}    {attr}_ui(),
)


def server(input, output, session):
    {attr}_server(input, session)


app = App(app_ui, server)
'''


def build_wheel(out_dir: Path = BUILD_DIR) -> Path:
    """Build the ``helicon`` wheel from this checkout into *out_dir*.

    Pure Python (no compiled extension of its own, and -- since numba,
    finufft, cryosparc-tools and psutil became optional extras -- none of
    its base dependencies needs one either), so this is a plain
    ``pip wheel``, not a cross-compile. Returns the wheel's path.

    The checkout's version (setuptools-scm, from ``git describe``) can
    change between two calls with the same *out_dir* -- a commit, or just the
    working tree changing -- so an earlier call's wheel can still be sitting
    there under a different filename. Removing every other ``helicon-*.whl``
    in *out_dir* once the current one is known keeps a later call's own
    lookup here from ever finding more than one.
    """
    import fcntl
    import tempfile

    out_dir.mkdir(parents=True, exist_ok=True)
    # pip builds in the source tree (its build/ folder), so two builds at once
    # -- parallel test workers -- break each other: one at a time
    with open(Path(tempfile.gettempdir()) / "helicon-build-wheel.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "-w",
                str(out_dir),
                str(REPO_ROOT),
            ],
            check=True,
        )
    # pip (re)writes the wheel matching the checkout's current version,
    # whether or not that filename already existed -- its mtime is always
    # the freshest among any helicon-*.whl here
    wheels = sorted(out_dir.glob("helicon-*.whl"), key=lambda p: p.stat().st_mtime)
    wheel = wheels[-1]
    for stale in wheels[:-1]:
        stale.unlink()
    return wheel


def wheel_base_requirements(wheel: Path) -> tuple[str, ...]:
    """The wheel's own unconditional ``Requires-Dist`` entries (no extra).

    Listed explicitly in a ``mode="package"`` bundle's own requirements.txt
    (see :func:`_stage`) rather than left for micropip to resolve from the
    wheel's metadata on its own, by necessity, not preference: found by
    trial, a package reached only as a *transitive* dependency of a
    URL-installed wheel is not force-loaded even when it is one of
    Pyodide's own bundled packages (``numpy``, say) -- the import then
    fails with "included in the Pyodide distribution, but not installed".
    Reading them from the wheel instead of copying pyproject.toml's
    ``dependencies`` by hand keeps the two from drifting apart.
    """
    import re
    import zipfile

    with zipfile.ZipFile(wheel) as zf:
        (metadata_name,) = (
            n for n in zf.namelist() if n.endswith(".dist-info/METADATA")
        )
        metadata = zf.read(metadata_name).decode("utf-8")
    reqs = []
    for line in metadata.splitlines():
        m = re.match(r"Requires-Dist:\s*(.+)", line)
        if m and "extra ==" not in m.group(1):
            # "pandas[html]" -> "pandas": Pyodide's lockfile matches by bare
            # package name; the [html] extra's own deps (lxml, html5lib, ...)
            # are not needed for anything the web app tabs do (the one
            # former reader, dataset.get_amyloid_atlas, now reads an
            # already-parsed mirror instead of scraping HTML directly)
            reqs.append(re.split(r"[\[;]", m.group(1).strip())[0].strip())
    return tuple(reqs)


def _stage(bundle: Bundle, staging_dir: Path, wheel: Path | None) -> None:
    """Assemble *bundle*'s app source tree in *staging_dir*."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    attr = bundle.module.rsplit(".", 1)[-1]
    if bundle.mode == "vendor":
        shutil.copyfile(SRC / bundle.source, staging_dir / f"{bundle.module}.py")
        for src, dest_name in bundle.extra_files:
            shutil.copyfile(SRC / src, staging_dir / dest_name)
        requirements = bundle.requirements
        import_module = bundle.module
    else:
        if wheel is None:
            raise ValueError(f"mode='package' bundle {bundle.name!r} needs a wheel")
        # fetched at run time from one level up from the app's own virtual
        # path, which lands back at this bundle's own exported directory --
        # build_bundle() copies the wheel there once export is done (it is
        # not staged here: see build_bundle's own "wheel" parameter doc).
        # helicon's own base dependencies are listed explicitly too: see
        # wheel_base_requirements's docstring for why that is necessary.
        # pyodide-http patches requests (one of those base dependencies) to
        # route through the browser's own fetch/XMLHttpRequest instead of a
        # real socket connection, which Pyodide has none of -- needed by
        # every package-mode bundle, since lib.path_utils.download_file_from_url
        # (an "EMDB ID" or "url" input, a RELION star file's own micrograph
        # or map paths, ...) is reachable from the real helicon package they
        # all install, not just the one tab a given bundle wires up.
        # pygments is rich's own dependency (rich.logging imports it, so
        # helicon.getLogger needs it): transitive again, so listed too.
        requirements = (
            f"../{wheel.name}",
            *wheel_base_requirements(wheel),
            "pyodide-http",
            "pygments",
            *bundle.requirements,
        )
        import_module = bundle.module
    (staging_dir / "requirements.txt").write_text(
        "\n".join(requirements) + ("\n" if requirements else "")
    )
    template = _APP_PY_TEMPLATE if bundle.is_module else _APP_PY_TEMPLATE_PLAIN
    head_import = head_call = ""
    if bundle.head:
        head_module, head_function = bundle.head.split(":")
        head_import = f"from {head_module} import {head_function}\n"
        head_call = f"    {head_function}(),\n"
    (staging_dir / "app.py").write_text(
        template.format(
            module=import_module,
            attr=attr,
            title=bundle.title,
            head_import=head_import,
            head_call=head_call,
        )
    )


def build_bundle(
    bundle: Bundle,
    *,
    out_dir: Path | None = None,
    staging_dir: Path | None = None,
    wheel: Path | None = None,
    keep_staging: bool = False,
) -> Path:
    """Stage and ``shinylive export`` *bundle*; return its output directory.

    Parameters
    ----------
    bundle : Bundle
    out_dir, staging_dir : Path, optional
        Where to write the exported site and the intermediate app source,
        respectively. Default to ``build/shinylive/<bundle.name>`` and
        ``build/shinylive/_staging/<bundle.name>``; a test passes its own
        ``tmp_path`` instead.
    wheel : Path, optional
        The ``helicon`` wheel (see :func:`build_wheel`), required when
        ``bundle.mode == "package"``. ``shinylive export`` embeds a staged
        requirement instead of serving it (micropip then cannot read it --
        the one thing that does not work about this build, found by trial),
        so the wheel is not staged; this copies it into *out_dir* itself,
        after export, as a plain static file alongside the exported
        ``index.html``, a sibling ``requirements.txt`` line
        (``../<wheel name>``) fetches over HTTP.
    keep_staging : bool, optional
    """
    if staging_dir is None:
        staging_dir = BUILD_DIR / "_staging" / bundle.name
    if out_dir is None:
        out_dir = BUILD_DIR / bundle.name
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    _stage(bundle, staging_dir, wheel)
    shinylive_cli = shutil.which("shinylive")
    if shinylive_cli is None:
        raise RuntimeError(
            "the 'shinylive' command was not found; install it with "
            "'pip install shinylive' (a build-time tool, not a helicon "
            "dependency -- see tools/shinylive/README.md)"
        )
    subprocess.run(
        [shinylive_cli, "export", str(staging_dir), str(out_dir)], check=True
    )
    if bundle.mode == "package":
        shutil.copyfile(wheel, out_dir / wheel.name)
    if not keep_staging:
        shutil.rmtree(staging_dir)
    return out_dir


_INDEX_TEMPLATE = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Helicon web apps (browser builds)</title>
<style>
body {{ font: 16px/1.5 system-ui, sans-serif; max-width: 40em; margin: 3em auto; padding: 0 1em; }}
li {{ margin: 0.5em 0; }}
</style>
</head>
<body>
<h1>Helicon web apps -- in your browser</h1>
<p>Each one runs entirely client-side (Shinylive/Pyodide): nothing is sent to
a server. See
<a href="https://github.com/jianglab/helicon/blob/main/tools/shinylive/README.md">tools/shinylive/README.md</a>
for which tabs qualify and why.</p>
<ul>
{items}
</ul>
</body>
</html>
"""


def write_index(bundles: list[Bundle], out_dir: Path = BUILD_DIR) -> Path:
    """Write a landing page at *out_dir*/index.html linking each of *bundles*."""
    items = "\n".join(f'<li><a href="{b.name}/">{b.title}</a></li>' for b in bundles)
    index_path = out_dir / "index.html"
    index_path.write_text(_INDEX_TEMPLATE.format(items=items))
    return index_path


def promote_to_root(bundle_dir: Path, out_dir: Path = BUILD_DIR) -> None:
    """Move an exported bundle's own files up into *out_dir* itself.

    Used for the "home" bundle only, so its own page becomes *out_dir*'s
    ``index.html`` -- the site's landing page -- in place of
    :func:`write_index`'s plain link list, with every sibling bundle still
    one level below it at ``<out_dir>/<name>/``, exactly what Home's own
    chips already assume (``home_tab.HomeApp.bundle``: the bare
    ``"<bundle>/"``, not ``"../<bundle>/"``). *bundle_dir* is removed once
    empty, so nothing is left behind at its old, nested path.
    """
    for entry in bundle_dir.iterdir():
        dest = out_dir / entry.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        shutil.move(str(entry), str(dest))
    bundle_dir.rmdir()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "bundles",
        nargs="*",
        choices=list(BUNDLES),
        help="Bundle(s) to build (default: all of them).",
    )
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="Leave the generated app source under build/shinylive/_staging/ "
        "(for inspection) instead of deleting it after export.",
    )
    args = parser.parse_args(argv)
    names = args.bundles or list(BUNDLES)
    wheel = None
    if any(BUNDLES[name].mode == "package" for name in names):
        wheel = build_wheel()
        print(f"wheel: {wheel}")
    for name in names:
        out_dir = build_bundle(
            BUNDLES[name], wheel=wheel, keep_staging=args.keep_staging
        )
        print(f"{name}: {out_dir}")
    if "home" in names:
        # Home becomes the landing page itself, not a link to it.
        promote_to_root(BUILD_DIR / "home")
        print(f"index: {BUILD_DIR / 'index.html'} (Home)")
    else:
        index_path = write_index([BUNDLES[name] for name in names])
        print(f"index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
