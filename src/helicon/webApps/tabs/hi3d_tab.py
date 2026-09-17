"""HI3D tab — helical indexing using cylindrical projection of a 3D map.

Faithful Shiny port of HI3D.git (Streamlit) 4-column layout.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import gzip
import shutil
from pathlib import Path

import numpy as np

import helicon
from shiny import reactive, render, ui, module, req

from ..lib.shared_state import ProjectState

logger = logging.getLogger(__name__)

_DEFAULT_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/"
    "EMD-10499/map/emd_10499.map.gz"
)


def lattice_line_segments(twist, phase_degree, rise, y_min, y_max):
    """One lattice line, wrapped into the plot's twist axis and cut at its edges.

    The markers of a given C-symmetry copy sit at ``(twist * n + phase, rise * n)``
    for integer ``n``, so in unwrapped twist they are colinear -- the direction the
    arrow shows. The plot's twist axis only spans one turn, though, so that line
    leaves one edge and comes back at the other.

    Each returned segment therefore ENDS exactly on +-180 and the next BEGINS
    exactly on the opposite edge at the same rise, so the family reads as one
    continuous line if the picture were tiled side by side. The crossings are
    solved for rather than found by sampling: a sampled split stops a little
    short of the edge, leaving a ragged gap exactly where continuity is the
    thing being shown.

    Returns ``(xs, ys)`` lists ready for a Bokeh ``multi_line``.
    """
    if rise <= 0 or y_max <= y_min:
        return [], []

    n_lo, n_hi = sorted((y_min / rise, y_max / rise))

    def wrapped(n):
        return (twist * n + phase_degree + 180.0) % 360.0 - 180.0

    if abs(twist) < 1e-12:  # a vertical line: no crossings to find
        x = wrapped(0.0)
        return [[x, x]], [[n_lo * rise, n_hi * rise]]

    # twist * n + phase == -180 (mod 360) is where the line leaves the axis.
    k_lo = int(np.floor((twist * n_lo + phase_degree + 180.0) / 360.0))
    k_hi = int(np.ceil((twist * n_hi + phase_degree + 180.0) / 360.0))
    crossings = sorted(
        n
        for k in range(min(k_lo, k_hi) - 1, max(k_lo, k_hi) + 2)
        for n in [(-180.0 + 360.0 * k - phase_degree) / twist]
        if n_lo < n < n_hi
    )

    xs, ys = [], []
    edges = [n_lo] + crossings + [n_hi]
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a <= 1e-12:
            continue
        # On a crossing the wrap is exact, so take the edge value rather than
        # whatever floating point lands on: leaving at +180 must arrive at -180.
        xa = (
            -180.0
            if (a in crossings and twist > 0)
            else (180.0 if a in crossings else wrapped(a))
        )
        xb = (
            180.0
            if (b in crossings and twist > 0)
            else (-180.0 if b in crossings else wrapped(b))
        )
        xs.append([xa, xb])
        ys.append([a * rise, b * rise])
    return xs, ys


@module.ui
def hi3d_tab_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.navset_pill(
                ui.nav_panel(
                    "Inputs",
                    ui.div(
                        # ── Inbox 1: Input source + Section controls ──
                        ui.div(
                            ui.accordion(
                                ui.accordion_panel(
                                    "README",
                                    ui.p(
                                        "This Web app considers a biological helical "
                                        "structure as a 2D crystal that has been rolled "
                                        "up into a cylindrical tube while preserving the "
                                        "original lattice. The indexing process is thus to "
                                        "computationally reverse this process: the 3D "
                                        "helical structure is first unrolled into a 2D "
                                        "image using cylindrical projection, and then the "
                                        "2D lattice parameters are automatically identified "
                                        "from which the helical parameters (twist, rise, "
                                        "and cyclic symmetry) are derived.",
                                        style="font-size:9pt; line-height:1.4;",
                                    ),
                                    ui.p(
                                        "Tips: play with the rmin/rmax, #peaks, axial step "
                                        "size parameters if consistent helical parameters "
                                        "cannot be obtained with the default parameters.",
                                        style="font-size:9pt; line-height:1.4;",
                                    ),
                                ),
                                open=False,
                                id="hi3d_readme",
                            ),
                            ui.input_radio_buttons(
                                "hi3d_input_mode",
                                "How to obtain the input map:",
                                choices=["upload", "url", "emd-xxxxx"],
                                selected="emd-xxxxx",
                                inline=True,
                            ),
                            ui.panel_conditional(
                                "input.hi3d_input_mode === 'upload'",
                                ui.input_file(
                                    "hi3d_upload_map",
                                    "Upload a map in MRC or CCP4 format",
                                    accept=[".mrc", ".mrc.gz", ".map", ".map.gz"],
                                ),
                            ),
                            ui.panel_conditional(
                                "input.hi3d_input_mode === 'url'",
                                ui.input_text(
                                    "hi3d_url_map",
                                    "Input the url of a 3D map:",
                                    value=_DEFAULT_URL,
                                ),
                            ),
                            ui.panel_conditional(
                                "input.hi3d_input_mode === 'emd-xxxxx'",
                                ui.output_ui("hi3d_emdb_helical_link"),
                                ui.input_action_button(
                                    "hi3d_change_emd", "Change EMDB ID"
                                ),
                                ui.input_text(
                                    "hi3d_emd_id",
                                    "Input an EMDB ID (emd-xxxxx):",
                                    value="emd-10499",
                                ),
                                ui.output_ui("hi3d_emdb_info"),
                            ),
                            ui.output_ui("hi3d_map_info"),
                            style="flex:1 1 180px; min-width:150px;",
                        ),
                        # ── Inbox 2: Section display + Transform ──
                        ui.div(
                            ui.panel_conditional(
                                "true",
                                ui.input_radio_buttons(
                                    "hi3d_section_axis",
                                    "Display a section along this axis:",
                                    choices={"0": "X/Y", "1": "X", "2": "Y", "3": "Z"},
                                    selected="0",
                                    inline=True,
                                ),
                                ui.input_slider(
                                    "hi3d_section_index",
                                    "Choose a section to display:",
                                    min=-1,
                                    max=1,
                                    value=0,
                                    step=1,
                                ),
                            ),
                            ui.output_ui("hi3d_original_image"),
                            ui.accordion(
                                ui.accordion_panel(
                                    "Transform the map",
                                    ui.input_checkbox(
                                        "hi3d_do_transform",
                                        "Center & verticalize",
                                        value=False,
                                    ),
                                    ui.output_ui("hi3d_transform_controls"),
                                ),
                                open=False,
                                id="hi3d_transform",
                            ),
                            style="flex:1 1 180px; min-width:150px;",
                        ),
                        # ── Inbox 3: Radial profile + range ──
                        ui.div(
                            ui.output_ui("hi3d_radial_profile"),
                            ui.accordion(
                                ui.accordion_panel(
                                    "Select radial range",
                                    ui.output_ui("hi3d_radial_range"),
                                ),
                                ui.accordion_panel(
                                    "Download data",
                                    ui.download_button(
                                        "hi3d_download_radial",
                                        "Radial profile (.csv)",
                                    ),
                                ),
                                open=False,
                                id="hi3d_radial_range_panel",
                            ),
                            style="flex:1 1 180px; min-width:150px;",
                        ),
                        style="display:flex; flex-wrap:wrap; gap:6px;",
                    ),
                ),
                ui.nav_panel(
                    "Parameters",
                    ui.div(
                        ui.input_numeric(
                            "hi3d_da",
                            "Angular step size (°)",
                            value=1.0,
                            min=0.1,
                            max=10.0,
                            step=0.1,
                        ),
                        ui.input_numeric(
                            "hi3d_dz",
                            "Axial step size (Å)",
                            value=1.0,
                            min=0.1,
                            max=10.0,
                            step=0.1,
                        ),
                        ui.input_numeric(
                            "hi3d_peak_width",
                            "Peak width (°)",
                            value=9.0,
                            min=0.1,
                            max=60.0,
                            step=1.0,
                        ),
                        ui.input_numeric(
                            "hi3d_peak_height",
                            "Peak height (Å)",
                            value=9.0,
                            min=0.1,
                            max=30.0,
                            step=1.0,
                        ),
                        ui.output_ui("hi3d_npeaks_ui"),
                        ui.input_checkbox("hi3d_acf_2x", "ACF 2x", value=False),
                        ui.input_checkbox("hi3d_show_scf", "SCF", value=False),
                        ui.hr(),
                        ui.h5("Display:"),
                        ui.input_checkbox(
                            "hi3d_show_cylproj", "Cylindrical projection", value=True
                        ),
                        ui.output_ui("hi3d_cylproj_params"),
                        ui.input_checkbox("hi3d_show_acf", "ACF", value=True),
                        ui.input_checkbox("hi3d_show_peaks", "Peaks", value=True),
                        ui.input_checkbox("hi3d_show_arrow", "Arrow", value=True),
                        ui.input_checkbox("hi3d_show_lattice", "Lattice", value=True),
                        ui.tooltip(
                            ui.input_checkbox(
                                "hi3d_show_lattice_lines",
                                "Lattice lines",
                                value=True,
                            ),
                            "Draw a dashed line through each C-symmetry copy's"
                            " lattice markers, along the twist/rise direction the"
                            " arrow shows. A line leaves one edge of the twist axis"
                            " and returns at the other, so it reads as continuous if"
                            " the plot were tiled side by side.",
                        ),
                    ),
                ),
            ),
        ),
        ui.h1(
            "HI3D: Helical indexing using the cylindrical projection of a 3D map",
            style="font-weight: bold;",
        ),
        ui.layout_columns(
            ui.div(
                ui.output_ui("hi3d_indexing_title"),
                ui.output_ui("hi3d_indexing_plot"),
            ),
            ui.div(
                ui.output_ui("hi3d_cylproj_plot_ui"),
                ui.output_ui("hi3d_acf_plot_ui"),
                ui.output_ui("hi3d_twist_rise_controls"),
                style="padding-left:8px;",
            ),
            col_widths=[8, 4],
            gap="4px",
        ),
        ui.HTML(
            "<i><p style='margin:2px 0'>Developed by the <a href='https://jianglab.science.psu.edu/helicon' target='_blank'>Jiang Lab</a>. "
            "Report issues to <a href='https://github.com/jianglab/helicon/issues' target='_blank'>helicon@GitHub</a>.</p></i>"
        ),
    )


@module.server
def hi3d_tab_server(input, output, session, project: ProjectState):
    from ..lib.hi3d_core import (
        cylindrical_projection,
        auto_correlation,
        find_peaks,
        fit_helical_lattice,
        refine_twist_rise,
        consistent_twist_rise_cn_sets,
        generate_bokeh_figure,
        make_square_shape,
        compute_radial_profile,
        estimate_radial_range,
        normalize,
        get_emdb_map_url,
    )
    from bokeh.plotting import figure as bk_figure
    from bokeh.models import Arrow, VeeHead, ColumnDataSource, Scatter
    from bokeh.models.tools import HoverTool, CrosshairTool
    from bokeh.models import LinearColorMapper
    from bokeh.palettes import Category10 as _Category10
    from shiny.types import SilentException

    # ── Reactive state ──────────────────────────────────────
    map_data = reactive.value(None)
    map_apix = reactive.value(1.0)
    map_crs = reactive.value([1, 2, 3])
    map_info_text = reactive.value("")

    cylproj = reactive.value(None)
    cylproj_work = reactive.value(None)
    cylproj_square = reactive.value(None)
    acf_img = reactive.value(None)
    peaks_data = reactive.value(None)
    masses_data = reactive.value(None)
    npeaks_all = reactive.value(0)
    npeaks_used = reactive.value(15)

    twist_val = reactive.value(0.0)
    rise_val = reactive.value(0.0)
    csym_val = reactive.value(1)
    fitted = reactive.value(False)

    dz_val = reactive.value(1.0)
    da_val = reactive.value(1.0)

    radial_profile_data = reactive.value(None)
    rmin_val = reactive.value(0.0)
    rmax_val = reactive.value(100.0)

    section_image = reactive.value(None)
    section_axis = reactive.value(0)  # 0=X/Y, 1=X, 2=Y, 3=Z
    section_index = reactive.value(0)
    apix_from_file = reactive.value(1.0)
    # ── URL query parameter init ─────────────────────────────
    # Set by _init_from_query_once so _auto_load_default_map can
    # skip the default map when a display button provides a file.
    skip_default_map = reactive.value(False)

    # ── URL query parameter init ─────────────────────────────
    # Set by _init_from_query_once so _auto_load_default_map can
    # skip the default map when a display button provides a file.
    skip_default_map = reactive.value(False)

    @reactive.effect(priority=10)
    async def _init_from_query_once():
        qs = session.clientdata.url_search()
        from urllib.parse import parse_qs

        qp = {
            k: v[0] if len(v) == 1 else v for k, v in parse_qs(qs.lstrip("?")).items()
        }
        url_img = qp.get("img_file_url")
        if url_img:
            skip_default_map.set(True)
            import asyncio

            try:
                ui.update_radio_buttons("hi3d_input_mode", selected="url")
                ui.update_text("hi3d_url_map", value=url_img)
            except Exception:
                pass
            try:
                result = await asyncio.to_thread(_download_url, url_img)
                _set_map(*result)
            except Exception as e:
                ui.modal_show(
                    ui.modal(
                        str(e),
                        title="Download failed",
                        easy_close=True,
                        footer=None,
                    )
                )

    @reactive.effect(priority=10)
    async def _init_from_query_once():
        qs = session.clientdata.url_search()
        from urllib.parse import parse_qs

        qp = {
            k: v[0] if len(v) == 1 else v for k, v in parse_qs(qs.lstrip("?")).items()
        }
        url_img = qp.get("img_file_url")
        if url_img:
            skip_default_map.set(True)
            import asyncio

            try:
                ui.update_radio_buttons("hi3d_input_mode", selected="url")
                ui.update_text("hi3d_url_map", value=url_img)
            except Exception:
                pass
            try:
                result = await asyncio.to_thread(_download_url, url_img)
                _set_map(*result)
            except Exception as e:
                ui.modal_show(
                    ui.modal(
                        str(e),
                        title="Download failed",
                        easy_close=True,
                        footer=None,
                    )
                )

    # ── Auto-load default map on startup ─────────────────────
    @reactive.effect
    async def _auto_load_default_map():
        if map_data() is not None or skip_default_map():
            return
        logger.debug("[HI3D] Auto-loading default map emd-10499")
        try:
            result = await asyncio.to_thread(_download_emd, "10499")
            _set_map(*result)
        except Exception:
            pass  # silent failure on startup

    # ── EMDB ID change + load ───────────────────────────────
    @reactive.effect
    @reactive.event(input.hi3d_change_emd)
    async def _change_and_load_emd():
        logger.debug("[HI3D] _change_and_load_emd: fired")
        import random

        try:
            emdb_ids = helicon.dataset.EMDB().helical_structure_ids()
            if emdb_ids:
                new_id = f"emd-{random.choice(emdb_ids)}"
                ui.update_text("hi3d_emd_id", value=new_id)
                await _do_load_emd(new_id.replace("emd-", ""))
        except Exception as e:
            ui.modal_show(ui.modal(str(e), title="Error", easy_close=True, footer=None))

    # ── EMDB helical-structures link ────────────────────────
    @render.ui
    def hi3d_emdb_helical_link():
        try:
            ids = helicon.dataset.EMDB().helical_structure_ids()
        except Exception:
            return None
        if not ids:
            return None
        n = len(ids)
        url = (
            "https://www.ebi.ac.uk/emdb/emsearch/"
            "*%20AND%20structure_determination_method:%22helical%22"
            "?rows=10&sort=release_date%20desc"
        )
        return ui.p(
            ui.a(f"All {n} helical structures in EMDB", href=url, target="_blank"),
            style="font-size:9pt; color:var(--bs-secondary-color); margin-top:2px;",
        )

    # ── EMDB info ───────────────────────────────────────────
    @render.ui
    def hi3d_emdb_info():
        emd_id_raw = input.hi3d_emd_id()
        if not emd_id_raw:
            return None
        emd_id = emd_id_raw.lower().replace("emd-", "")
        try:
            from helicon.lib.dataset import EMDB as _EMDB_cls

            params = _EMDB_cls().get_info(emd_id)
            if params is None:
                return ui.p(
                    f"EMD-{emd_id}: could not retrieve information",
                    style="color:#e67e22; font-size:9pt;",
                )
            entry_url = f"https://www.ebi.ac.uk/emdb/entry/EMD-{emd_id}"
            rest = f" | resolution={params.get('resolution', '?')} Å"
            if "twist" in params:
                rest += f" | twist={params['twist']}° | rise={params['rise']}Å"
                if "csym" in params:
                    rest += f" | {params['csym'].upper()}"
            return ui.p(
                ui.a(f"EMD-{emd_id}", href=entry_url, target="_blank"),
                rest,
                style="font-size:9pt; color:var(--bs-secondary-color); margin-top:2px;",
            )
        except Exception:
            return None

    # ── Load map from various sources ───────────────────────
    @reactive.effect
    @reactive.event(input.hi3d_upload_map)
    def _load_upload():
        fi = input.hi3d_upload_map()
        if not fi:
            return
        try:
            import mrcfile

            with mrcfile.open(fi[0]["datapath"]) as mrc:
                d = np.array(mrc.data, dtype=np.float32)
                apix_val = float(mrc.voxel_size.x)
                crs = [int(mrc.header.mapc), int(mrc.header.mapr), int(mrc.header.maps)]
            _set_map(d, apix_val, crs, fi[0]["name"])
        except Exception as e:
            ui.modal_show(ui.modal(str(e), title="Error", easy_close=True, footer=None))

    @reactive.effect
    @reactive.event(input.hi3d_input_mode, input.hi3d_url_map)
    async def _load_url_map():
        if input.hi3d_input_mode() != "url":
            return
        url = input.hi3d_url_map()
        if not url or not url.strip():
            return
        progress = {"done": 0, "total": 0}
        result = await _run_download(
            url.strip(), lambda: _download_url(url, progress), progress
        )
        if result is not None:
            _set_map(*result)

    def _fetch_to_file(url, handle, progress=None):
        """Stream ``url`` into an open file, recording progress as it goes.

        Streamed rather than buffered through ``resp.content`` so the caller can
        say how far along it is. The EMDB helical set runs to half a gigabyte
        for the largest maps, and a modal that says only "Downloading..." for
        several minutes is indistinguishable from one that has hung -- which is
        exactly how it was reported.

        ``progress`` is a plain dict, not a reactive value: this runs in a
        worker thread, where reactive writes are not allowed. The coroutine that
        started the thread polls it.
        """
        import requests

        resp = requests.get(url, timeout=60, stream=True)
        resp.raise_for_status()
        if progress is not None:
            progress["total"] = int(resp.headers.get("content-length") or 0)
        for chunk in resp.iter_content(chunk_size=1 << 20):
            if chunk:
                handle.write(chunk)
                if progress is not None:
                    progress["done"] += len(chunk)

    async def _run_download(label, work, progress):
        """Run a download off the loop, reporting bytes as they arrive.

        A progress bar, not a modal with a text output in it. Two earlier
        attempts failed for instructive reasons: re-showing the modal to
        refresh the count rebuilds the dialog several times a second, which
        reads as flashing; and putting a text output inside the modal never
        updated at all, because a reactive value set from inside an effect that
        is still running is not flushed to the browser until that effect
        returns -- which here is when the download has already finished.
        Progress messages are sent to the client as they are made, which is the
        whole point of them.

        Returns the result, or None if it failed, the failure already on screen.
        """
        with ui.Progress(min=0, max=1) as bar:
            bar.set(0, message=f"Downloading {label}", detail="connecting...")
            task = asyncio.create_task(asyncio.to_thread(work))
            shown = -1
            while not task.done():
                await asyncio.sleep(0.3)
                done, total = progress.get("done", 0), progress.get("total", 0)
                if total:
                    pct = int(100 * done / total)
                    if pct != shown:
                        shown = pct
                        bar.set(
                            done / total,
                            message=f"Downloading {label}",
                            detail=f"{done / 1e6:.0f} of {total / 1e6:.0f} MB ({pct}%)",
                        )
                elif done:
                    bar.set(
                        0,
                        message=f"Downloading {label}",
                        detail=f"{done / 1e6:.0f} MB",
                    )
            try:
                return await task
            except Exception as e:
                ui.modal_show(
                    ui.modal(
                        str(e), title="Download failed", easy_close=True, footer=None
                    )
                )
                return None

    def _download_emd(emd_id, progress=None):
        """Download map from EMDB and return (data, apix, crs, label).

        Pure I/O: no reactive writes, no UI calls. Safe to run in a
        background thread via :func:`asyncio.to_thread`.  Caller must
        invoke :func:`_set_map` on the main asyncio loop so reactive
        invalidation cascades run on the correct event loop.
        """
        logger.debug("[HI3D] _download_emd: starting download for emd-%s", emd_id)
        import mrcfile

        url = get_emdb_map_url(f"emd-{emd_id}")
        logger.debug("[HI3D] _download_emd: url=%s", url)
        with tempfile.NamedTemporaryFile(suffix=".map.gz", delete=False) as tmp:
            _fetch_to_file(url, tmp, progress)
            gz_path = tmp.name
        mrc_path = gz_path[:-3]
        with gzip.open(gz_path, "rb") as f_in, open(mrc_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        with mrcfile.open(mrc_path) as mrc:
            d = np.array(mrc.data, dtype=np.float32)
            apix_val = float(mrc.voxel_size.x)
            crs = [int(mrc.header.mapc), int(mrc.header.mapr), int(mrc.header.maps)]
        Path(gz_path).unlink(missing_ok=True)
        Path(mrc_path).unlink(missing_ok=True)
        logger.debug("[HI3D] _download_emd: download complete, shape=%s", d.shape)
        return d, apix_val, crs, f"EMD-{emd_id}"

    def _download_url(url, progress=None):
        """Fetch a map from an ``http(s)``/``ftp`` URL (or local path) and return
        ``(data, apix, crs, label)``.  Pure I/O — no reactive writes, no UI.

        Accepts plain ``.map``/``.mrc`` files and gzip-compressed
        ``.map.gz``/``.mrc.gz``.  Local file paths are also supported.
        """
        import mrcfile

        url_stripped = url.strip()
        suffix = Path(url_stripped).suffixes[-1] if url_stripped else ".map"
        is_gz = suffix == ".gz"
        if is_gz:
            suffix += ".map.gz"
        else:
            suffix = ".map"

        logger.debug("[HI3D] _download_url: starting for %s", url_stripped)

        if url_stripped.startswith(("http://", "https://", "ftp://")):
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                import requests

                _fetch_to_file(url_stripped, tmp, progress)
                local_path = Path(tmp.name)
        else:
            local_path = Path(url_stripped)

        if is_gz:
            mrc_path = local_path.with_suffix("")
            with gzip.open(local_path, "rb") as f_in, open(mrc_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            is_temp = True
        else:
            mrc_path = local_path
            is_temp = False

        with mrcfile.open(mrc_path) as mrc:
            d = np.array(mrc.data, dtype=np.float32)
            apix_val = float(mrc.voxel_size.x)
            crs = [int(mrc.header.mapc), int(mrc.header.mapr), int(mrc.header.maps)]

        if is_temp:
            local_path.unlink(missing_ok=True)
            mrc_path.unlink(missing_ok=True)

        label = Path(url_stripped).name or url_stripped
        logger.debug("[HI3D] _download_url: download complete, shape=%s", d.shape)
        return d, apix_val, crs, label

    async def _do_load_emd(emd_id):
        """Load map from EMDB with modal feedback. Call from main thread (async).

        The download runs on a worker thread; ``_set_map`` runs on the main
        asyncio loop so that reactive invalidation ``session._increment_busy_count``
        can ``get_running_loop()`` successfully.  Calling ``reactive.value.set``
        from inside an ``asyncio.to_thread`` worker previously raised
        ``RuntimeError: There is no current event loop`` mid-invalidiation,
        which dropped ``_run_computation`` from the flush queue and left the
        main indexing plot showing the old map.
        """
        progress = {"done": 0, "total": 0}
        result = await _run_download(
            f"EMD-{emd_id}", lambda: _download_emd(emd_id, progress), progress
        )
        if result is not None:
            _set_map(*result)

    def _set_map(d, apix_val, crs, label):
        if d.ndim != 3 or d.shape[0] < 32:
            ui.modal_show(
                ui.modal(
                    f"Not a 3D map (shape={d.shape})",
                    title="Error",
                    easy_close=True,
                    footer=None,
                )
            )
            return
        # Ensure axes are x,y,z
        if crs != [1, 2, 3]:
            from ..lib.hi3d_core import change_mrc_map_crs_order

            d = change_mrc_map_crs_order(d, crs, [1, 2, 3])
            crs = [1, 2, 3]
        logger.error(
            "[HI3D] _set_map: setting map_data shape=%s apix=%s", d.shape, apix_val
        )
        logger.debug("[HI3D] _set_map: setting map_data shape=%s", d.shape)
        map_data.set(d)
        map_apix.set(apix_val)
        map_crs.set(crs)
        apix_from_file.set(apix_val)
        project.input_map.set(d)
        project.input_map_apix.set(apix_val)
        nz, ny, nx = d.shape
        map_info_text.set(f"{nx}×{ny}×{nz} voxels | {apix_val:.4g} Å/voxel")
        # Radial range: where the density actually is, not the whole box.
        #
        # ``estimate_radial_range`` was ported into hi3d_core along with the
        # rest of HI3D but never called, so the range defaulted to 0..half-box
        # and every map started by including the empty middle and the empty
        # corners. Matching the original: threshold the radial profile at a
        # tenth of its range above the background taken from the outermost
        # bins, and take the first and last radius above it, in angstroms.
        #
        # Failures here fall back to the old whole-box default rather than
        # leaving the range unset, since a profile that defeats the estimate --
        # a blank map, say -- should still give the user something to adjust.
        rmin_default, rmax_default = 0.0, round(min(nx, ny) / 2 * apix_val, 1)
        try:
            rp = compute_radial_profile(d)
            radial_profile_data.set(rp)
            r_lo, r_hi = estimate_radial_range(rp, thresh_ratio=0.1)
            if r_hi > r_lo:
                rmin_default = round(r_lo * apix_val, 1)
                rmax_default = round(r_hi * apix_val, 1)
        except Exception:
            logger.debug("[HI3D] auto radial range failed", exc_info=True)
        rmin_val.set(rmin_default)
        rmax_val.set(rmax_default)
        # Reset section controls
        section_index.set(0)
        # Reset fitting
        fitted.set(False)
        cylproj.set(None)
        cylproj_work.set(None)
        cylproj_square.set(None)
        acf_img.set(None)
        peaks_data.set(None)

    # ── Map info ────────────────────────────────────────────
    @render.ui
    def hi3d_map_info():
        info = map_info_text()
        if not info:
            return None
        return ui.p(
            info,
            style="font-size:9pt; color:var(--bs-secondary-color); margin-top:4px;",
        )

    @reactive.effect
    def _update_section():
        data = map_data()
        if data is None:
            logger.debug("[HI3D] _update_section: map_data is None, skipping")
            return
        logger.debug("[HI3D] _update_section: map_data shape=%s", data.shape)
        try:
            axis_str = input.hi3d_section_axis()
            half_offset = input.hi3d_section_index()  # centered: -n//2 .. n-1-n//2
        except Exception as e:
            logger.debug("[HI3D] _update_section: input not ready: %s", e)
            axis_str = "0"
            half_offset = None
        axis = int(axis_str) if axis_str is not None else 0
        nz, ny, nx = data.shape
        if axis == 0:
            n = min(nx, ny)
        else:
            mapping = {1: 2, 2: 1, 3: 0}
            np_axis = mapping.get(axis, 2)
            n = data.shape[np_axis]
        idx = int(half_offset) + n // 2 if half_offset is not None else n // 2
        idx = min(max(0, idx), n - 1)
        if axis == 0:
            image = np.zeros((nz, max(ny, nx)), dtype=data.dtype)
            ix = np.squeeze(np.take(data, indices=[idx], axis=2))
            iy = np.squeeze(np.take(data, indices=[idx], axis=1))
            image[:, : ny // 2] = ix[:, : ny // 2]
            image[:, -nx // 2 :] = iy[:, nx // 2 :]
            image[:, ny // 2 - 1] = np.max(image)
        else:
            image = np.squeeze(np.take(data, indices=[idx], axis=np_axis))
        section_image.set(image)
        section_axis.set(axis)
        # Update slider range (centered convention: 0 = central slice)
        try:
            half = n // 2
            slider_min = -half
            slider_max = n - 1 - half
            if slider_max < slider_min:
                slider_max = slider_min
            slider_val = 0 if half_offset is None else int(half_offset)
            slider_val = min(max(slider_min, slider_val), slider_max)
            ui.update_slider(
                "hi3d_section_index",
                min=slider_min,
                max=slider_max,
                value=slider_val,
                step=1,
            )
        except Exception:
            pass

    # ── Original image ──────────────────────────────────────
    @render.ui
    def hi3d_original_image():
        image = section_image()
        logger.error(
            "[HI3D] hi3d_original_image: image=%s",
            None if image is None else image.shape,
        )
        if image is None:
            return None
        apix = map_apix()
        h, w = image.shape
        tooltips = [("x", "$x"), ("y", "$y"), ("val", "@image")]
        fig = generate_bokeh_figure(
            image,
            apix,
            apix,
            title="Original",
            title_location="below",
            plot_width=None,
            plot_height=None,
            x_axis_label=None,
            y_axis_label=None,
            tooltips=tooltips,
            show_axis=False,
            show_toolbar=False,
            crosshair_color="white",
            aspect_ratio=w / h,
        )
        fig.sizing_mode = "stretch_width"
        from bokeh.embed import components

        script, div = components(fig)
        return ui.HTML(script + div)

    @reactive.effect
    @reactive.event(input.hi3d_rmin, input.hi3d_rmax)
    def _sync_radial_range():
        """Carry the typed radial range into the values everything else reads.

        Without this the rmin/rmax inputs were inert: nothing read
        ``input.hi3d_rmin`` anywhere, so the markers on the radial profile and
        the cylindrical projection that follows kept the defaults computed when
        the map loaded.

        It has to be written back rather than read where needed, because the
        panel holding these inputs is rendered from ``rmin_val``/``rmax_val``
        and is re-rendered whenever the radial profile is recomputed. Left
        unsynced, that re-render reseeds the input from the stale default and
        the user's entry vanishes as they watch.
        """
        try:
            lo = float(input.hi3d_rmin())
            hi = float(input.hi3d_rmax())
        except (SilentException, TypeError, ValueError):
            return
        if lo != rmin_val():
            rmin_val.set(lo)
        if hi != rmax_val():
            rmax_val.set(hi)

    # ── Radial profile ──────────────────────────────────────
    @render.ui
    def hi3d_radial_profile():
        data = map_data()
        if data is None:
            return None
        try:
            rp = compute_radial_profile(data)
            radial_profile_data.set(rp)
            from bokeh.plotting import figure as bk_fig
            from bokeh.models import Span

            apix = map_apix()
            r = np.arange(len(rp)) * apix
            fig = bk_fig(
                x_axis_label="r (Å)",
                height=200,
                width=280,
                sizing_mode="stretch_width",
                tools="box_zoom,crosshair,pan,reset,save,wheel_zoom",
            )
            fig.line(r, rp, line_width=2, color="red", legend_label="radial")
            fig.add_layout(
                Span(
                    location=rmin_val(),
                    dimension="height",
                    line_color="green",
                    line_dash="dashed",
                    line_width=3,
                )
            )
            fig.add_layout(
                Span(
                    location=rmax_val(),
                    dimension="height",
                    line_color="green",
                    line_dash="dashed",
                    line_width=3,
                )
            )
            fig.legend.visible = False
            from bokeh.embed import components

            script, div = components(fig)
            return ui.HTML(script + div)
        except Exception:
            return None

    # ── Radial range controls ───────────────────────────────
    @render.ui
    def hi3d_radial_range():
        data = map_data()
        rp = radial_profile_data()
        if data is None or rp is None:
            return None
        apix = map_apix()
        rmax_half = min(data.shape[1], data.shape[2]) / 2 * apix
        return ui.div(
            ui.input_numeric(
                "hi3d_rmin",
                "Inner radius (Å)",
                value=rmin_val(),
                min=0.0,
                max=rmax_half,
                step=1.0,
            ),
            ui.input_numeric(
                "hi3d_rmax",
                "Outer radius (Å)",
                value=rmax_val(),
                min=0.0,
                max=rmax_half,
                step=1.0,
            ),
        )

    # ── Download radial profile ─────────────────────────────
    @render.download(filename="radial_profile.csv", media_type="text/csv")
    def hi3d_download_radial():
        import io
        import pandas as pd

        rp = radial_profile_data()
        if rp is None:
            data = map_data()
            if data is None:
                return
            rp = compute_radial_profile(data)
        apix = map_apix()
        rad = np.arange(len(rp)) * apix
        df = pd.DataFrame(
            np.hstack((rad.reshape(-1, 1), rp.reshape(-1, 1))),
            columns=["radius (Å)", "density"],
        ).round(6)
        yield df.to_csv(index=False)

    # ── Cylindrical projection parameters ───────────────────
    @render.ui
    def hi3d_cylproj_params():
        if not input.hi3d_show_cylproj():
            return None
        cp = cylproj()
        if cp is None:
            return None
        da = input.hi3d_da()
        dz = input.hi3d_dz()
        nz_cp = cp.shape[0]
        z_half = round(nz_cp // 2 * dz, 1)
        return ui.div(
            ui.input_numeric(
                "hi3d_ang_min",
                "Minimal angle (°)",
                value=-180.0,
                min=-180.0,
                max=180.0,
                step=1.0,
            ),
            ui.input_numeric(
                "hi3d_ang_max",
                "Maximal angle (°)",
                value=180.0,
                min=-180.0,
                max=180.0,
                step=1.0,
            ),
            ui.input_numeric("hi3d_z_min", "Minimal z (Å)", value=-z_half, step=1.0),
            ui.input_numeric("hi3d_z_max", "Maximal z (Å)", value=z_half, step=1.0),
        )

    # ── Npeaks UI ───────────────────────────────────────────
    @render.ui
    def hi3d_npeaks_ui():
        n = npeaks_all()
        if n < 3:
            return None
        return ui.input_numeric(
            "hi3d_npeaks",
            "# peaks to use",
            value=min(npeaks_used(), n),
            min=3,
            max=n,
            step=2,
        )

    # ── Run indexing computation ─────────────────────────────
    @reactive.effect
    @reactive.event(
        input.hi3d_da,
        input.hi3d_dz,
        input.hi3d_peak_width,
        input.hi3d_peak_height,
        # The reactive values, NOT input.hi3d_rmin/rmax. Those inputs live in a
        # panel that renders only once a map and its radial profile exist, so at
        # first load they do not exist at all -- and naming a missing input here
        # silences the whole effect, which meant the indexing never ran and the
        # results pane sat on "Run indexing to see results" forever. The sync
        # effect above pushes the typed values into these, so changes still
        # arrive; these merely always exist.
        rmin_val,
        rmax_val,
        map_data,
    )
    def _run_computation():
        data = map_data()
        if data is None:
            return
        try:
            req(
                input.hi3d_da(),
                input.hi3d_dz(),
                input.hi3d_peak_width(),
                input.hi3d_peak_height(),
            )
            apix = map_apix()
            rmin = rmin_val()
            rmax = rmax_val()
            da = input.hi3d_da()
            dz = input.hi3d_dz()

            cp = cylindrical_projection(
                data,
                da=da,
                dz=dz / apix,
                dr=1,
                rmin=rmin / apix,
                rmax=rmax / apix,
                interpolation_order=1,
            )
            cylproj.set(cp)

            # Apply ROI
            cp_work = cp.copy()
            nz_cp, na_cp = cp.shape
            z_min = -nz_cp // 2 * dz
            z_max = nz_cp // 2 * dz
            try:
                ang_min = input.hi3d_ang_min()
                ang_max = input.hi3d_ang_max()
                z_min = input.hi3d_z_min()
                z_max = input.hi3d_z_max()
            except Exception:
                ang_min = -180.0
                ang_max = 180.0

            draw_box = not (
                ang_min == -180
                and ang_max == 180
                and abs(z_min + nz_cp // 2 * dz) < 0.1
                and abs(z_max - nz_cp // 2 * dz) < 0.1
            )
            if draw_box:
                if ang_min < ang_max:
                    if ang_min > -180:
                        a0 = int(round(ang_min / da)) + na_cp // 2
                        cp_work[:, :a0] = 0
                    if ang_max < 180:
                        a1 = int(round(ang_max / da)) + na_cp // 2
                        cp_work[:, a1:] = 0
                else:
                    if ang_min < 180:
                        a0 = int(round(ang_min / da)) + na_cp // 2
                        cp_work[a0:] = 0
                    if ang_max > -180:
                        a1 = int(round(ang_max / da)) + na_cp // 2
                        cp_work[:a1] = 0
                if z_min > -nz_cp // 2 * dz:
                    z0 = int(round(z_min / dz)) + nz_cp // 2
                    cp_work[:z0, :] = 0
                if z_max < nz_cp // 2 * dz:
                    z1 = int(round(z_max / dz)) + nz_cp // 2
                    cp_work[z1:, :] = 0
            cylproj_work.set(cp_work)

            cp_sq = make_square_shape(cp_work)
            cylproj_square.set(cp_sq)

            do_scf = input.hi3d_show_scf()
            do_2x = input.hi3d_acf_2x()
            acf_result = auto_correlation(
                cp_sq,
                sqrt_transform=do_scf,
                high_pass_fraction=1.0 / cp_sq.shape[0],
            )
            if do_2x:
                acf_result = auto_correlation(
                    acf_result,
                    sqrt_transform=do_scf,
                    high_pass_fraction=1.0 / cp_sq.shape[0],
                )
            acf_img.set(acf_result)

            peak_w = input.hi3d_peak_width()
            peak_h = input.hi3d_peak_height()
            pks, m = find_peaks(
                acf_result,
                da=da,
                dz=dz,
                peak_width=peak_w,
                peak_height=peak_h,
                minmass=1.0,
            )
            if pks is not None and len(pks) >= 3:
                peaks_data.set(pks)
                masses_data.set(m)
                npeaks_all.set(len(pks))

                # Fit helical lattice
                npeaks = len(pks)  # use all peaks for initial fit
                try:
                    trc1, trc2 = fit_helical_lattice(
                        pks[:npeaks],
                        acf_result,
                        da=da,
                        dz=dz,
                    )
                    trc_mean = consistent_twist_rise_cn_sets(
                        [trc1],
                        [trc2],
                        epsilon=1.0,
                    )
                    if trc_mean:
                        t, r, c = trc_mean[0]
                        t_ref, r_ref = refine_twist_rise(
                            acf_result,
                            da,
                            dz,
                            t,
                            r,
                            c,
                        )
                        twist_val.set(t_ref)
                        rise_val.set(r_ref)
                        csym_val.set(c)
                    else:
                        twist_val.set(trc1[0])
                        rise_val.set(trc1[1])
                        csym_val.set(int(trc1[2]))
                    fitted.set(True)
                    project.twist.set(round(twist_val(), 3))
                    project.rise.set(round(rise_val(), 3))
                    project.csym.set(int(csym_val()))
                except Exception as e:
                    logger.error("Fit failed: %s", e)
                    fitted.set(False)
            else:
                peaks_data.set(None)
                masses_data.set(None)
                npeaks_all.set(0)
                fitted.set(False)
        except Exception as e:
            logger.error("Computation failed: %s", e, exc_info=True)

    # ── Cylindrical projection plot (col4) ──────────────────
    @render.ui
    def hi3d_cylproj_plot_ui():
        cp = cylproj()
        if cp is None or not input.hi3d_show_cylproj():
            return None
        da = input.hi3d_da()
        dz = input.hi3d_dz()
        h, w = cp.shape
        tooltips = [("angle", "$x °"), ("z", "$y Å"), ("cylproj", "@image")]
        fig = generate_bokeh_figure(
            cp,
            da,
            dz,
            title=f"Cylindrical Projection ({w}×{h})",
            title_location="below",
            plot_width=None,
            plot_height=None,
            x_axis_label=None,
            y_axis_label=None,
            tooltips=tooltips,
            show_axis=False,
            show_toolbar=True,
            crosshair_color="white",
            aspect_ratio=w / h,
        )
        fig.sizing_mode = "stretch_width"
        # Draw ROI box if needed
        try:
            ang_min = input.hi3d_ang_min()
            ang_max = input.hi3d_ang_max()
            z_min = input.hi3d_z_min()
            z_max = input.hi3d_z_max()
            draw = not (ang_min == -180 and ang_max == 180)
            if draw:
                if ang_min < ang_max:
                    fig.quad(
                        left=ang_min,
                        right=ang_max,
                        bottom=z_min,
                        top=z_max,
                        line_color=None,
                        fill_color="yellow",
                        fill_alpha=0.5,
                    )
                else:
                    fig.quad(
                        left=ang_min,
                        right=180,
                        bottom=z_min,
                        top=z_max,
                        line_color=None,
                        fill_color="yellow",
                        fill_alpha=0.5,
                    )
                    fig.quad(
                        left=-180,
                        right=ang_max,
                        bottom=z_min,
                        top=z_max,
                        line_color=None,
                        fill_color="yellow",
                        fill_alpha=0.5,
                    )
        except Exception:
            pass
        from bokeh.embed import components

        script, div = components(fig)
        return ui.HTML(script + div)

    # ── ACF plot (col4) ─────────────────────────────────────
    @render.ui
    def hi3d_acf_plot_ui():
        acf_result = acf_img()
        if acf_result is None or not input.hi3d_show_acf():
            return None
        da = input.hi3d_da()
        dz = input.hi3d_dz()
        h, w = acf_result.shape
        tooltips = [("twist", "$x °"), ("rise", "$y Å"), ("acf", "@image")]
        fig = generate_bokeh_figure(
            acf_result,
            da,
            dz,
            title=f"Auto-Correlation ({w}×{h})",
            title_location="below",
            plot_width=None,
            plot_height=None,
            x_axis_label=None,
            y_axis_label=None,
            tooltips=tooltips,
            show_axis=False,
            show_toolbar=True,
            crosshair_color="white",
            aspect_ratio=w / h,
        )
        fig.sizing_mode = "stretch_width"
        if input.hi3d_show_peaks() and peaks_data() is not None:
            pks = peaks_data()
            try:
                n = min(input.hi3d_npeaks(), len(pks))
            except SilentException:
                n = min(npeaks_used(), len(pks))
            fig.ellipse(
                pks[:n, 0],
                pks[:n, 1],
                width=input.hi3d_peak_width(),
                height=input.hi3d_peak_height(),
                line_width=1,
                line_color="yellow",
                fill_alpha=0,
            )
        from bokeh.embed import components

        script, div = components(fig)
        return ui.HTML(script + div)

    # ── Twist/Rise/Csym controls (col4) ─────────────────────
    @render.ui
    def hi3d_twist_rise_controls():
        acf_result = acf_img()
        if acf_result is None:
            return None
        h = acf_result.shape[0]
        dz = input.hi3d_dz()
        t = twist_val() if fitted() else 0.0
        r = rise_val() if fitted() else 0.0
        c = csym_val() if fitted() else 1
        return ui.div(
            ui.div(
                ui.div(
                    ui.input_numeric(
                        "hi3d_twist_manual",
                        "Twist (°):",
                        value=round(t, 2),
                        min=-180.0,
                        max=180.0,
                        step=0.01,
                    ),
                    style="flex:1 1 100px; min-width:50px;",
                ),
                ui.div(
                    ui.input_numeric(
                        "hi3d_rise_manual",
                        "Rise (Å):",
                        value=round(r, 2),
                        min=0.0,
                        max=h * dz,
                        step=0.01,
                    ),
                    style="flex:1 1 100px; min-width:50px;",
                ),
                ui.div(
                    ui.input_numeric(
                        "hi3d_csym_manual", "Csym:", value=int(c), min=1, max=64, step=1
                    ),
                    style="flex:1 1 100px; min-width:50px;",
                ),
                style="display:flex; flex-wrap:wrap; gap:4px; margin-top:20px;",
            ),
            ui.div(
                ui.input_action_button(
                    "hi3d_refine_btn",
                    "Refine twist/rise",
                    class_="btn-primary",
                    style="width:100%;",
                ),
                style="margin-top:4px;",
            ),
        )

    @reactive.effect
    @reactive.event(input.hi3d_refine_btn)
    def _refine():
        acf_result = acf_img()
        if acf_result is None:
            return
        da = input.hi3d_da()
        dz = input.hi3d_dz()
        t = input.hi3d_twist_manual()
        r = input.hi3d_rise_manual()
        c = int(input.hi3d_csym_manual())
        try:
            t_opt, r_opt = refine_twist_rise(acf_result, da, dz, t, r, c)
            twist_val.set(t_opt)
            rise_val.set(r_opt)
            csym_val.set(c)
            ui.update_numeric("hi3d_twist_manual", value=round(t_opt, 2))
            ui.update_numeric("hi3d_rise_manual", value=round(r_opt, 2))
            project.twist.set(round(t_opt, 3))
            project.rise.set(round(r_opt, 3))
            project.csym.set(c)
        except Exception as e:
            logger.error("Refine failed: %s", e)

    # ── Indexing title (col2) ───────────────────────────────
    @render.ui
    def hi3d_indexing_title():
        t = input.hi3d_twist_manual() if fitted() else 0.0
        r = input.hi3d_rise_manual() if fitted() else 0.0
        c = int(input.hi3d_csym_manual()) if fitted() else 1
        if abs(t) > 0.01 and r > 0.01:
            pitch = 360.0 / abs(t) * r
            title = (
                f"twist={round(t,3):g}° (pitch={round(pitch, 2):g}Å) "
                f"rise={round(r,3):g}Å  csym=c{c:d}"
            )
        else:
            title = "Run indexing to see results"
        return ui.h4(title, style="text-align:center; font-weight:normal;")

    # ── Main indexing plot (col2) ────────────────────────────
    @render.ui
    def hi3d_indexing_plot():
        acf_result = acf_img()
        if acf_result is None:
            return None
        da = input.hi3d_da()
        dz = input.hi3d_dz()
        h, w = acf_result.shape
        tooltips = [("twist", "$x °"), ("rise", "$y Å"), ("acf", "@image")]
        fig = generate_bokeh_figure(
            acf_result,
            da,
            dz,
            title="",
            title_location="above",
            plot_width=None,
            plot_height=None,
            x_axis_label="twist (°)",
            y_axis_label="rise (Å)",
            tooltips=tooltips,
            show_axis=True,
            show_toolbar=True,
            crosshair_color="white",
            aspect_ratio=w / h,
        )
        fig.sizing_mode = "stretch_width"
        fig.line(
            [-w // 2 * da, (w // 2 - 1) * da],
            [0, 0],
            line_width=2,
            line_color="yellow",
            line_dash="dashed",
        )
        if fitted():
            t = round(twist_val(), 3)
            r = round(rise_val(), 3)
            c = int(csym_val())

            if input.hi3d_show_arrow():
                fig.add_layout(
                    Arrow(
                        x_start=0,
                        y_start=0,
                        x_end=t,
                        y_end=r,
                        line_color="yellow",
                        line_width=4,
                        end=VeeHead(
                            line_color="yellow", fill_color="yellow", line_width=2
                        ),
                    )
                )

            if input.hi3d_show_lattice():
                colors_avail = _Category10[max(3, min(10, c))]
                colors = [colors_avail[si % len(colors_avail)] for si in range(c)]
                nn = int(h / 2 * dz / r) + 1
                nn = np.arange(-nn, nn + 1)
                xs = np.fmod(t * nn + np.max(nn) * 360, 360)
                xs[xs > 180] -= 360
                ys = r * nn
                # Lines first, markers second, so a line passes under the
                # circles it threads rather than through them. Bokeh draws in
                # the order glyphs are added; "underlay" is not the way to get
                # this, since that puts them beneath the ACF image itself, where
                # nothing can be seen of them at all.
                if input.hi3d_show_lattice_lines():
                    for si in range(c):
                        seg_x, seg_y = lattice_line_segments(
                            t, 360.0 / c * si, r, -(h // 2) * dz, (h // 2) * dz
                        )
                        if seg_x:
                            fig.multi_line(
                                xs=seg_x,
                                ys=seg_y,
                                line_color=colors[si],
                                line_width=1,
                                line_dash="dashed",
                            )

                for si in range(c):
                    xsym = np.fmod(xs + 360 / c * si, 360)
                    xsym[xsym > 180] -= 360
                    source = ColumnDataSource(dict(x=xsym, y=ys))
                    glyph = Scatter(
                        x="x",
                        y="y",
                        marker="circle",
                        size=10,
                        line_width=3,
                        line_color=colors[si],
                        fill_color=None,
                    )
                    fig.add_glyph(source, glyph)

        fig.xaxis.axis_label_text_font_size = "14pt"
        fig.yaxis.axis_label_text_font_size = "14pt"
        fig.xaxis.major_label_text_font_size = "11pt"
        fig.yaxis.major_label_text_font_size = "11pt"

        from bokeh.embed import components

        script, div = components(fig)
        return ui.HTML(script + div)
