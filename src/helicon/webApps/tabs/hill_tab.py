"""HILL tab — helical indexing via Fourier layer lines.

Ported from HILL.git into the Shiny module pattern.
Uses Bokeh widgets for interactive power spectrum / layer line analysis.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import helicon
from shiny import reactive, render, ui, module, req
from shiny.types import SilentException
from shinywidgets import output_widget, render_widget


try:
    from ..lib import hill_compute as hill
except Exception:
    hill = None

# helicon.get_images_from_url does not exist; use the shared webApps lib.
from .. import auto_method_ui, bookmark, deployment
from ..lib import denovo3d_pipeline

logger = logging.getLogger(__name__)

# ── Module-level constants ────────────────────────────────────────

MODULE_PREFIX = "hill"

from .hill_bookmark import BOOKMARK_DEFAULTS  # noqa: E402

# One row, whatever the sidebar's width: the pieces wrap rather than overflow,
# and the radio's label sits beside its options instead of above them.
_IMAGE_ROW_CSS = """
.hill-img-row { display: flex; flex-wrap: wrap; align-items: center; gap: 0 14px; }
.hill-img-row .form-group { margin-bottom: 0.5rem; }
.hill-img-row .shiny-input-radiogroup { display: flex; flex-wrap: wrap;
    align-items: baseline; column-gap: 8px; }
.hill-img-row .shiny-input-radiogroup > label { margin-bottom: 0; }
"""

_INIT_TWIST = 29.40
_INIT_RISE = 21.92
_INIT_PITCH = 268.41
_INIT_APIX = 2.3438
_INIT_NX = 256
_INIT_NY = 256
_INIT_PNX = 512
_INIT_PNY = 1024
_INIT_CUTOFF_X = 7.03
_INIT_CUTOFF_Y = 4.69
_INIT_HELICAL_RADIUS = 77.6


def _set_helix_sliders(slider_twist, slider_pitch, slider_rise, twist, pitch, rise):
    """Set the helix sliders and their ranges to a twist, pitch and rise.

    This is what the browser does for a value that arrives from the server
    (``apply_values`` in ``_HELIX_CONTROL_JS``), done on the model itself: a
    widget the browser does not have yet is built from the model, and a value
    sent to it before then is lost.

    Parameters
    ----------
    slider_twist, slider_pitch, slider_rise : bokeh.models.Slider
        The three sliders.
    twist, pitch, rise : float
        The values to show.

    Returns
    -------
    bool
        Whether anything of the sliders changed.
    """
    changed = False

    def same(a, b):
        return abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b))

    def set_range(slider, start, end, step=None):
        nonlocal changed
        if slider.start != start or slider.end != end:
            slider.start, slider.end = start, end
            changed = True
        if step is not None and slider.step != step:
            slider.step = step
            changed = True

    set_range(slider_twist, min(-180.0, twist), max(180.0, twist))
    for slider, value in ((slider_pitch, pitch), (slider_rise, rise)):
        if value < slider.start or value > slider.end:
            end = value * 2
            set_range(slider, value / 2, end, max(end * 0.001, np.finfo(float).eps))
    for slider, value in (
        (slider_twist, twist),
        (slider_pitch, pitch),
        (slider_rise, rise),
    ):
        if not same(slider.value, value):
            slider.value = value
            changed = True
    return changed


def _as_csym(value) -> int:
    """A csym as the whole number it must be.

    Parameters
    ----------
    value : int, float or str
        What an input holds: a bookmark's ``csym=2.5``, a number typed with a
        decimal point, an EMDB entry.

    Returns
    -------
    int
        The nearest whole number, and at least 1.
    """
    return max(1, int(round(float(value))))


def _progress_bar(done: int, total: int):
    """A Bootstrap progress bar for a search that has done ``done`` of ``total``
    steps; an animated full-width one while the number of steps is not known yet.

    Parameters
    ----------
    done, total : int
        Steps finished and steps in all; ``total`` 0 when it is not known.

    Returns
    -------
    shiny.ui.Tag
    """
    if total <= 0:
        label, width = "Searching", 100
        classes = " progress-bar-striped progress-bar-animated"
    else:
        width = max(0, min(100, round(100 * done / total)))
        label, classes = f"{width}%", ""
    return ui.div(
        ui.div(
            label if total > 0 else "",
            class_="progress-bar" + classes,
            role="progressbar",
            # As tall as the track, which clips it. The ipywidgets stylesheet this
            # tab loads sets every .progress-bar to 28 px, so the height must be
            # said here; the bottom of a taller fill is cut off.
            style=f"width: {width}%; height: 100%; line-height: 1.3rem; "
            "font-size: 0.8rem; padding: 0;",
            **{
                "aria-valuenow": str(width),
                "aria-valuemin": "0",
                "aria-valuemax": "100",
                "aria-label": label,
            },
        ),
        class_="progress hill-search-progress",
        style="height: 1.3rem; margin: 0.4em 0;",
    )


@contextlib.contextmanager
def _held(document):
    """Send the changes made inside as few messages as Bokeh can combine them to.

    Every change to a figure that is on a page is sent to the browser as soon as
    it is made, and the document's graph of models is walked again each time;
    redrawing the layer lines removes and adds about fifty renderers one at a
    time. Held, the changes to one attribute are combined into one message, and
    with the graph frozen it is walked once, at the end. Either alone saves
    nothing measurable: together a redraw takes a tenth of the time.

    Parameters
    ----------
    document : bokeh.document.Document or None
        The document of the figures, None when they are not on a page yet.
    """
    if document is None:
        yield
        return
    # ``models.freeze`` belongs to Bokeh's document internals rather than its
    # documented API: without it the changes are still combined, only slower.
    freeze = getattr(getattr(document, "models", None), "freeze", None)
    with freeze() if freeze is not None else contextlib.nullcontext():
        document.hold("combine")
        try:
            yield
        finally:
            document.unhold()


def _resolution_limits_from_apix(apix: float) -> tuple[float, float]:
    """Return HILL's default X/Y resolution limits for a pixel size."""
    apix = float(apix)
    return round(3 * apix, 4), round(2 * apix, 4)


def _update_with_apix_from_file(apix: float) -> None:
    """Apply a loaded file's pixel size to HILL's dependent inputs."""
    apix = float(apix)
    if not np.isfinite(apix) or apix <= 0:
        return

    cutoff_res_x, cutoff_res_y = _resolution_limits_from_apix(apix)
    nyquist_res = cutoff_res_y
    ui.update_numeric("hill_apix", value=round(apix, 4))
    ui.update_numeric("hill_cutoff_res_x", value=cutoff_res_x, min=nyquist_res)
    ui.update_numeric("hill_cutoff_res_y", value=cutoff_res_y, min=nyquist_res)


# Default 2D image stack pre-filled in the URL input so the field is not
# blank on launch.
_DEFAULT_URL = BOOKMARK_DEFAULTS["url"][1]


# ── UI ────────────────────────────────────────────────────────────


@module.ui
def hill_tab_ui():
    return ui.page_fillable(
        ui.tags.style(
            """
        .hill-scrollable-sidebar {
            height: 100%; max-height: 100%; overflow-y: auto;
            border: 1px solid #ccc; padding: 10px;
        }
        .hill-wrap-text { word-wrap: break-word; overflow-wrap: break-word; white-space: normal; }
        .hill-inline-box label { display: table-cell; text-align: left; vertical-align: middle; }
        .hill-inline-box input { width: 4em; margin-left: 1em; }
        .hill-inline-box .form-group { display: table-row; }
        .hill-params .accordion-body { padding: 0.6rem 0.5rem; }
        /* Every row is two columns. bslib's twelve-column grid has eleven gaps,
           which in a narrow sidebar leave its columns no width at all and the
           fields run out of the panel. */
        .hill-params .bslib-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
            column-gap: 0.75rem;
        }
        .hill-params .bslib-grid > * { grid-column: auto !important; min-width: 0; }
        .hill-params .form-control { min-width: 0; }
        .hill-params .shiny-input-container { width: 100%; }
        #hill-hill_main_plots { overflow-y: auto; }
        """
        ),
        ui.layout_sidebar(
            ui.sidebar(
                ui.navset_pill(
                    ui.nav_panel(
                        "Inputs",
                        ui.accordion(
                            ui.accordion_panel(
                                "README",
                                ui.p(
                                    "This Web app considers a biological helical structure as the product of "
                                    "a continuous helix and a set of parallel planes. Based on the convolution "
                                    "theory, the Fourier Transform (FT) of a helical structure would be the "
                                    "convolution of the FT of the continuous helix and the FT of the planes. "
                                    "The FT of a continuous helix consists of equally spaced layer planes (3D) "
                                    "or layer lines (2D projection) that can be described by Bessel functions "
                                    "of increasing orders (0, ±1, ±2, ...) from the Fourier origin."
                                ),
                                value="hill_readme_panel",
                            ),
                            id="hill_sidebar_accordion",
                            open=False,
                        ),
                        ui.accordion(
                            ui.accordion_panel(
                                "Input Mode",
                                ui.input_radio_buttons(
                                    "hill_input_mode_params",
                                    "How to obtain the input image/map:",
                                    helicon.shiny.source_modes(
                                        deployment.is_cloud(),
                                        (
                                            ("1", "upload"),
                                            ("2", "url"),
                                            ("3", "emd-xxxxx"),
                                        ),
                                    ),
                                    selected="2",
                                    inline=True,
                                ),
                                ui.output_ui("hill_conditional_input_uis"),
                                value="hill_input_mode",
                            ),
                            open=True,
                        ),
                        ui.output_ui("hill_conditional_3d_uis"),
                        ui.div(
                            ui.output_ui("hill_select_image_gallery"),
                            style="max-height: 500px; overflow-y: auto;",
                        ),
                        ui.div(
                            ui.output_ui("hill_active_image_gallery"),
                            style="max-height: 300px; overflow-y: auto;",
                        ),
                        ui.accordion(
                            ui.accordion_panel(
                                ui.output_text("hill_image_params_title", inline=True),
                                # "Input is:", the contrast and the auto-transform on
                                # one row (wrapping when the sidebar is too narrow)
                                ui.tags.style(_IMAGE_ROW_CSS),
                                ui.div(
                                    ui.input_radio_buttons(
                                        "hill_input_type",
                                        "Input is:",
                                        choices=["Image", "PS", "PD"],
                                        inline=True,
                                        width="auto",
                                    ),
                                    ui.output_ui("hill_img_flags_ui"),
                                    class_="hill-img-row",
                                ),
                                ui.output_ui("hill_img_2d_uis"),
                                value="hill_image_params",
                            ),
                            # on the accordion: on a panel it is only an HTML
                            # attribute, and the first panel is then open
                            open=False,
                        ),
                        output_widget("hill_display_transformed", height="auto"),
                        output_widget("hill_plot_radial_profile", height="auto"),
                        output_widget("hill_plot_acf", width="auto", height="auto"),
                        ui.output_ui("hill_get_avg_ps_ui"),
                        ui.div(
                            ui.div(
                                ui.tooltip(
                                    ui.input_action_button(
                                        "hill_suggest_symmetry",
                                        "Suggest csym, twist and rise",
                                        class_="btn-primary",
                                    ),
                                    "Search the layer lines of the selected images for "
                                    "the helical symmetries that explain them, best "
                                    "first. Uses the filament diameter and the "
                                    "resolution limit X (Parameters tab); several class "
                                    "averages of one helix do better than one.",
                                ),
                                class_="hill-suggest-button",
                                style="text-align: center;",
                            ),
                            ui.output_ui("hill_symmetry_message"),
                            ui.output_ui("hill_symmetry_progress"),
                            ui.output_data_frame("hill_symmetry_candidates"),
                            style="margin: 0.5em 0 1em 0;",
                        ),
                        ui.HTML(
                            "<i><p>Developed by the <a href='https://jianglab.science.psu.edu/helicon' target='_blank'>Jiang Lab</a>. "
                            "Report issues to <a href='https://github.com/jianglab/helicon/issues' target='_blank'>helicon@GitHub</a>.</p></i>"
                        ),
                        value="hill_input_sidebar",
                    ),
                    ui.nav_panel(
                        "Parameters",
                        # One panel per function, the controls of one type together
                        # inside it. Every input keeps its id (bookmarks, effects).
                        ui.accordion(
                            ui.accordion_panel(
                                "Helical symmetry",
                                ui.layout_columns(
                                    ui.input_numeric(
                                        "hill_diameter",
                                        "Filament/tube diameter (Å)",
                                        value=155.2,
                                        min=1.0,
                                        max=1000.0,
                                        step=10.0,
                                        update_on="blur",
                                    ),
                                    ui.input_numeric(
                                        "hill_out_of_plane_tilt",
                                        "Out-of-plane tilt (°)",
                                        value=0.0,
                                        min=-90.0,
                                        max=90.0,
                                        step=1.0,
                                        update_on="blur",
                                    ),
                                    col_widths=6,
                                    style="align-items: flex-end;",
                                ),
                                ui.input_radio_buttons(
                                    "hill_use_twist_pitch",
                                    "Keep twist/pitch when changing rise:",
                                    choices=["Twist", "Pitch"],
                                    selected="Twist",
                                    inline=True,
                                ),
                                value="hill_params_symmetry",
                            ),
                            ui.accordion_panel(
                                "Image alignment",
                                ui.tooltip(
                                    ui.input_select(
                                        "hill_auto_method",
                                        auto_method_ui.LABEL,
                                        choices=auto_method_ui.CHOICES,
                                        selected=auto_method_ui.DEFAULT,
                                    ),
                                    auto_method_ui.HELP,
                                ),
                                ui.input_checkbox(
                                    "hill_inhibit_update",
                                    "Inhibit automatic update",
                                    value=False,
                                ),
                                value="hill_params_alignment",
                            ),
                            ui.accordion_panel(
                                "Fourier transform",
                                ui.layout_columns(
                                    ui.input_numeric(
                                        "hill_cutoff_res_x",
                                        "Resolution limit X (Å)",
                                        value=7.03,
                                        min=2.0,
                                        step=1.0,
                                        update_on="blur",
                                    ),
                                    ui.input_numeric(
                                        "hill_cutoff_res_y",
                                        "Resolution limit Y (Å)",
                                        value=4.69,
                                        min=2.0,
                                        step=1.0,
                                        update_on="blur",
                                    ),
                                    ui.input_numeric(
                                        "hill_pnx",
                                        "FFT X-dim (px)",
                                        value=512,
                                        min=128,
                                        step=2,
                                        update_on="blur",
                                    ),
                                    ui.input_numeric(
                                        "hill_pny",
                                        "FFT Y-dim (px)",
                                        value=1024,
                                        min=512,
                                        step=2,
                                        update_on="blur",
                                    ),
                                    col_widths=6,
                                    style="align-items: flex-end;",
                                ),
                                ui.tooltip(
                                    ui.input_radio_buttons(
                                        "hill_ps_scaling",
                                        "Scale the spectra by:",
                                        choices={
                                            "log": "Log + filter",
                                            "shell": "Shell median",
                                        },
                                        selected="log",
                                        inline=True,
                                    ),
                                    "Amplitudes fall by orders of magnitude from low to "
                                    "high resolution. Log + filter compresses that range "
                                    "with a log and a Fourier high/low-pass. Shell median "
                                    "divides each Fourier shell (the pixels of one spatial "
                                    "frequency) by its median, so a layer line shows by how "
                                    "far it stands above the background at its own "
                                    "resolution, then takes the log of the ratio.",
                                ),
                                ui.panel_conditional(
                                    "input.hill_ps_scaling == 'log'",
                                    ui.layout_columns(
                                        ui.input_numeric(
                                            "hill_hp_fraction",
                                            "Fourier high-pass (%)",
                                            value=0.40,
                                            min=0.0,
                                            max=100.0,
                                            step=0.1,
                                            update_on="blur",
                                        ),
                                        ui.input_numeric(
                                            "hill_lp_fraction",
                                            "Fourier low-pass (%)",
                                            value=0.00,
                                            min=0.0,
                                            max=100.0,
                                            step=0.1,
                                            update_on="blur",
                                        ),
                                        col_widths=6,
                                        style="align-items: flex-end;",
                                    ),
                                    ui.input_checkbox(
                                        "hill_log_amp", "Log (amplitude)", value=True
                                    ),
                                ),
                                value="hill_params_fft",
                            ),
                            ui.accordion_panel(
                                "Y profile",
                                ui.input_checkbox(
                                    "hill_yp_peak_detect",
                                    "Detect peaks in Y-Profile",
                                    value=True,
                                ),
                                ui.panel_conditional(
                                    "input.hill_yp_peak_detect",
                                    ui.layout_columns(
                                        ui.input_numeric(
                                            "hill_yp_peak_prominence",
                                            "Peak prominence",
                                            value=0.2,
                                            min=0.001,
                                            max=1.0,
                                            step=0.01,
                                            update_on="blur",
                                        ),
                                        ui.input_numeric(
                                            "hill_yp_peak_fit_hw",
                                            "Fit half-width (px)",
                                            value=5,
                                            min=2,
                                            max=50,
                                            step=1,
                                            update_on="blur",
                                        ),
                                        col_widths=6,
                                        style="align-items: flex-end;",
                                    ),
                                ),
                                value="hill_params_yprofile",
                            ),
                            ui.accordion_panel(
                                "Layer lines",
                                ui.layout_columns(
                                    ui.input_checkbox(
                                        "hill_LL", "Layer line", value=True
                                    ),
                                    ui.input_checkbox(
                                        "hill_LLText", "Layer line text", value=True
                                    ),
                                    col_widths=6,
                                    style="align-items: flex-end;",
                                ),
                                ui.div(
                                    ui.input_numeric(
                                        "hill_m_max",
                                        "Max M =",
                                        value=3,
                                        min=1,
                                        step=1,
                                        update_on="blur",
                                        width="4em",
                                    ),
                                    class_="hill-inline-box",
                                ),
                                ui.input_checkbox_group(
                                    "hill_ms",
                                    "m:",
                                    ["3", "2", "1", "0", "-1", "-2", "-3"],
                                    selected=["-1", "0", "1"],
                                    inline=True,
                                ),
                                ui.input_text(
                                    "hill_ll_colors",
                                    "Layerline colors",
                                    value="lime cyan violet salmon silver",
                                    update_on="blur",
                                ),
                                value="hill_params_layerlines",
                            ),
                            ui.accordion_panel(
                                "Display",
                                ui.layout_columns(
                                    ui.input_checkbox(
                                        "hill_PS", "Power spectra", value=True
                                    ),
                                    ui.input_checkbox(
                                        "hill_YP", "Y Profile", value=True
                                    ),
                                    ui.input_checkbox(
                                        "hill_PD", "Phase difference", value=True
                                    ),
                                    ui.input_checkbox(
                                        "hill_Phase", "Phase hover", value=False
                                    ),
                                    ui.input_checkbox(
                                        "hill_Color", "Color", value=True
                                    ),
                                    ui.input_checkbox(
                                        "hill_fft_top_only",
                                        "Only display top half of FFT",
                                        value=False,
                                    ),
                                    col_widths=6,
                                    style="align-items: flex-end;",
                                ),
                                ui.input_text(
                                    "hill_const_image_color",
                                    "Flatten PS/PD image in this color",
                                    value="",
                                    placeholder="white black",
                                    update_on="blur",
                                ),
                                value="hill_params_display",
                            ),
                            id="hill_params_accordion",
                            class_="hill-params",
                            multiple=True,
                            open=True,
                        ),
                        ui.div(
                            ui.input_numeric(
                                "hill_twist",
                                "Twist (°)",
                                value=_INIT_TWIST,
                                min=-180.0,
                                max=180.0,
                                update_on="blur",
                            ),
                            ui.input_numeric(
                                "hill_pitch",
                                "Pitch (Å)",
                                value=_INIT_PITCH,
                                min=1.0,
                                update_on="blur",
                            ),
                            ui.input_numeric(
                                "hill_rise",
                                "Rise (Å)",
                                value=_INIT_RISE,
                                min=1.0,
                                update_on="blur",
                            ),
                            ui.input_numeric(
                                "hill_helix_revision",
                                "Helix control revision",
                                value=0,
                            ),
                            # What the csym box beside the sliders sets and shows;
                            # kept as an input for the bookmark, the candidate rows
                            # and everything that reads input.hill_csym().
                            ui.input_numeric(
                                "hill_csym",
                                "csym",
                                value=6,
                                min=1,
                                step=1,
                                update_on="blur",
                            ),
                            hidden=True,  # only for JS communication
                        ),
                        value="hill_params_sidebar",
                    ),
                    ui.nav_menu(
                        "Advanced",
                        ui.nav_panel(
                            "Filament Straightening",
                            ui.div(
                                ui.input_numeric(
                                    "hill_num_markers_straighten",
                                    "Number of markers",
                                    value=10,
                                    min=3,
                                    step=1,
                                    update_on="blur",
                                ),
                                ui.input_numeric(
                                    "hill_template_diameter_straighten",
                                    "Template diameter (Å)",
                                    value=None,
                                    min=1,
                                    step=1,
                                    update_on="blur",
                                ),
                                ui.input_numeric(
                                    "hill_output_width_straighten",
                                    "Output width (px)",
                                    value=512,
                                    min=16,
                                    step=1,
                                    update_on="blur",
                                ),
                                ui.input_numeric(
                                    "hill_mask_radius_straighten",
                                    "Mask radius (%)",
                                    value=90,
                                    min=1,
                                    step=1,
                                    update_on="blur",
                                ),
                                ui.input_numeric(
                                    "hill_mask_len_straighten",
                                    "Mask length (%)",
                                    value=90,
                                    min=1,
                                    step=1,
                                    update_on="blur",
                                ),
                                ui.input_checkbox(
                                    "hill_do_auto_sample_axis_straighten",
                                    "Auto sample axis",
                                    value=False,
                                ),
                                style="text-align: left; border: 1px solid #ddd; padding: 10px;",
                                class_="hill-wrap-text",
                            ),
                        ),
                        ui.nav_panel(
                            "Simulation",
                            ui.input_numeric(
                                "hill_ball_radius_sim",
                                "Gaussian radius (Å)",
                                value=0.0,
                                min=0.0,
                                max=100.0,
                                step=5.0,
                                update_on="blur",
                            ),
                            ui.input_numeric(
                                "hill_tilt_sim",
                                "Out-of-plane tilt (°)",
                                value=0.0,
                                min=0.0,
                                max=90.0,
                                step=1.0,
                                update_on="blur",
                            ),
                            ui.input_numeric(
                                "hill_gauss_noise_std_sim",
                                "Gaussian noise std",
                                value=0.0,
                                min=0.0,
                                max=1.0,
                                step=0.1,
                                update_on="blur",
                            ),
                            ui.input_action_button(
                                "hill_run_simulation", "Run Simulation"
                            ),
                        ),
                        ui.nav_panel(
                            "Average Power Spectra",
                            ui.output_ui("hill_avg_ps_meta_ui"),
                            ui.h6("Average Power Spectra results:"),
                            ui.output_data_frame("hill_display_avg_ps_df"),
                        ),
                    ),
                    id="hill_sidebar_navset",
                ),
                class_="hill-scrollable-sidebar",
                width="25vw",
            ),
            ui.navset_hidden(
                ui.nav_panel(
                    None,
                    ui.h1(
                        "HILL: Helical Indexing using Layer Lines",
                        style="font-weight: bold;",
                    ),
                    ui.div(
                        ui.row(
                            ui.column(
                                12,
                                ui.output_ui("hill_error_display"),
                            )
                        ),
                        ui.row(
                            ui.column(
                                12,
                                output_widget("hill_main_plots", height="100%"),
                                ui.tags.script(_BOKEH_SLIDER_EDIT_JS),
                            )
                        ),
                        class_="scrollable-container",
                    ),
                    value="hill_main_content_tab",
                ),
                ui.nav_panel(
                    None,
                    ui.output_ui("hill_filament_straightening_uis"),
                    value="hill_straighten_tab",
                ),
                id="hill_main_tabs",
            ),
        ),
        title="HILL: Helical Indexing using Layer Lines",
    )


# ── Server ────────────────────────────────────────────────────────


@module.server
def hill_tab_server(input, output, session):
    # Browse... in the server mode: pick the files on this computer
    from bokeh.models import CustomJS
    from scipy.signal import find_peaks
    from scipy.interpolate import splev

    if not deployment.is_cloud():
        helicon.shiny.file_picker_fill(
            "image_browse",
            "hill_img_file_server",
            input,
            ("*.mrcs", "*.mrc", "*.map", "*.mrc.gz", "*.map.gz"),
            "Select 2D image(s) or a 3D map",
        )

    if hill is None:
        ui.modal_show(
            ui.modal(
                "hill_compute module failed to import. Check dependencies.",
                title="Import Error",
                easy_close=True,
                footer=None,
            )
        )
        return

    # ── Reactive state ──────────────────────────────────────────
    input_data = reactive.value(None)
    apix_from_file = reactive.value(_INIT_APIX)
    data_all_2d = reactive.value([])
    data_all_2d_labels = reactive.value([])
    initial_selected_image_indices = reactive.value([0])
    image_display_size = reactive.value(128)
    selected_images = reactive.value([])
    selected_image_labels = reactive.value([])

    # The layer-line figures need to know whether there are images, not which:
    # depending on the list itself redrew them (about a second) every time the
    # selection was touched, and again for the diameter that change led to. A
    # value, unlike a calc, tells its dependents only when it changes.
    has_images = reactive.value(False)

    @reactive.effect
    def _track_has_images():
        has_images.set(len(selected_images()) > 0)

    straighten_tab_activated = reactive.value(False)
    markers_straighten = reactive.value(None)
    mask_radius_auto = reactive.value(0.0)
    mask_len_percent_auto = reactive.value(0.0)
    data_2d_transformed = reactive.value(None)
    ny_curr_img = reactive.value(_INIT_NY)
    nx_curr_img = reactive.value(_INIT_NX)
    nz_curr_img = reactive.value(_INIT_NX)
    emd_msg = reactive.value("")
    ps_data = reactive.value(None)
    phase_data = reactive.value(None)
    pd_data = reactive.value(None)
    _last_transform_key = [None]
    _layerline_frame_key = [None]
    _last_ps_pd_selection = [None]
    _previous_input_mode = [None]
    prev_data_from = reactive.value("main")
    avg_ps_pd_df = reactive.value({})
    avg_ps_pd_main_in_df = reactive.value({})
    # The image (its index in data_all_2d) that the transform controls hold the
    # values of; None when they hold no image's values (new data, new estimates).
    _active_orig = [None]
    straightened_filaments_for_avg = reactive.value([])

    # Error display state
    error_msg = reactive.value("")

    # ── Bokeh plot objects ──────────────────────────────────────
    init_img = np.ones((_INIT_NY, _INIT_NX))
    pwr_init = np.zeros((_INIT_PNY, _INIT_PNX))
    phase_diff_init = np.zeros((_INIT_PNY, _INIT_PNX))
    phase_diff_init[0][0] = 1.0

    from bokeh.events import DoubleTap
    from bokeh.layouts import gridplot, layout, row
    from bokeh.models import (
        ColumnDataSource,
        CustomJS,
        Span,
        Slider,
        Spinner,
        LinearColorMapper,
    )
    from bokeh.models.tools import CrosshairTool, HoverTool, PointDrawTool
    from bokeh.plotting import figure
    from jupyter_bokeh import BokehModel

    span_width = Span(dimension="width", line_color="red")
    span_height = Span(dimension="height", line_color="red")

    # ── PS figure ───────────────────────────────────────────────
    tooltips_ps = [
        ("Res r", "A"),
        ("Res y", "A"),
        ("Res x", "A"),
        ("Jn", "@bessel"),
        ("Amp", "@image"),
    ]
    fig_ps, data_source_ps = hill.create_layerline_image_figure(
        data=pwr_init,
        cutoff_res_x=_INIT_CUTOFF_X,
        cutoff_res_y=_INIT_CUTOFF_Y,
        helical_radius=_INIT_HELICAL_RADIUS,
        tilt=0.0,
        phase=None,
        fft_top_only=False,
        pseudo_color=True,
        const_image_color="",
        title="Power Spectra",
        yaxis_visible=False,
        tooltips=tooltips_ps,
    )
    fig_ps.add_tools(CrosshairTool(overlay=(span_width, span_height)))

    # ── YP figure ───────────────────────────────────────────────
    ny_init, nx_init = _INIT_PNY, _INIT_PNX
    dsy_init = 1.0 / (ny_init // 2 * _INIT_CUTOFF_Y)
    y_vals = np.arange(-ny_init // 2, ny_init // 2) * dsy_init
    yinv = y_vals.copy()
    yinv[yinv == 0] = 1e-10
    yinv = 1.0 / np.abs(yinv)
    yprofile_init = np.max(pwr_init, axis=1)
    yprofile_init /= max(yprofile_init.max(), 1e-10)
    data_source_yp = ColumnDataSource(
        data=dict(yprofile=yprofile_init, y=y_vals, resy=yinv)
    )
    tools_yp = "box_zoom,hover,pan,reset,save,wheel_zoom"
    tooltips_yp = [("Res y", "@resy A"), ("Amp", "$x")]
    fig_yp = figure(
        frame_width=nx_init // 2,
        frame_height=fig_ps.frame_height,
        x_range=(0, 1),
        y_range=fig_ps.y_range,
        y_axis_location="right",
        title=None,
        tools=tools_yp,
        tooltips=tooltips_yp,
    )
    # Keep YP responsive with the PS/PD figures so the gridplot row fits the
    # available width including the right-side toolbar (no right-edge crop).
    fig_yp.width_policy = "fit"
    fig_yp.height_policy = "fit"
    yp_line = fig_yp.line(
        source=data_source_yp, x="yprofile", y="y", line_width=2, color="blue"
    )
    fig_yp.yaxis.visible = False
    fig_yp.hover[0].attachment = "vertical"
    fig_yp.hover[0].renderers = [yp_line]

    # YP peak detection data sources
    data_source_yp_peaks = ColumnDataSource(data=dict(x=[], y=[], label=[]))
    data_source_yp_fits = ColumnDataSource(data=dict(x=[], y=[]))
    fig_yp.multi_line(
        source=data_source_yp_fits,
        xs="x",
        ys="y",
        line_width=2,
        line_color="red",
        line_dash="dashed",
    )
    fig_yp.scatter(
        source=data_source_yp_peaks, x="x", y="y", size=8, color="red", marker="circle"
    )
    fig_yp.text(
        source=data_source_yp_peaks,
        x="x",
        y="y",
        text="label",
        text_color="red",
        text_font_size="9pt",
        x_offset=5,
        y_offset=0,
        text_baseline="middle",
    )
    fig_yp.add_tools(CrosshairTool(overlay=(span_width)))

    # ── PD figure ───────────────────────────────────────────────
    tooltips_pd = [
        ("Res r", "A"),
        ("Res y", "A"),
        ("Res x", "A"),
        ("Jn", "@bessel"),
        ("Phase Diff", "@image °"),
    ]
    fig_pd, data_source_pd = hill.create_layerline_image_figure(
        data=phase_diff_init,
        cutoff_res_x=_INIT_CUTOFF_X,
        cutoff_res_y=_INIT_CUTOFF_Y,
        helical_radius=_INIT_HELICAL_RADIUS,
        tilt=0.0,
        phase=None,
        fft_top_only=False,
        pseudo_color=True,
        const_image_color="",
        title="Phase Diff Across Meridian",
        yaxis_visible=False,
        tooltips=tooltips_pd,
    )
    fig_pd.x_range = fig_ps.x_range
    fig_pd.y_range = fig_ps.y_range
    fig_pd.add_tools(CrosshairTool(overlay=(span_width, span_height)))

    figs = [fig_ps, fig_yp, fig_pd]
    figs_image = [fig_ps, fig_pd]

    # ── Layer line ellipses (initial) ───────────────────────────
    fig_ellipses = []
    _init_layer_lines(fig_ellipses, figs_image)

    # ── Bokeh sliders for helix params (their numbers are edited in place) ──
    curr_twist = _INIT_TWIST
    curr_rise = _INIT_RISE
    curr_pitch = hill.twist2pitch(curr_twist, curr_rise)

    slider_twist = Slider(
        start=-180.0,
        end=180.0,
        value=curr_twist,
        step=0.01,
        title="Twist (°)",
        sizing_mode="stretch_width",
        syncable=False,
    )
    slider_pitch = Slider(
        start=curr_pitch / 2,
        end=curr_pitch * 2.0,
        value=curr_pitch,
        step=curr_pitch * 0.002,
        title="Pitch (Å)",
        sizing_mode="stretch_width",
        syncable=False,
    )
    slider_rise = Slider(
        start=curr_rise / 2,
        end=min(curr_pitch, curr_rise * 2.0),
        value=curr_rise,
        step=min(curr_pitch, curr_rise * 2.0) * 0.001,
        title="Rise (Å)",
        sizing_mode="stretch_width",
        syncable=False,
    )
    for slider in (slider_twist, slider_pitch, slider_rise):
        slider.css_classes = _editable_slider_classes(slider)

    # csym, which is part of indexing too, is set beside the sliders it goes with,
    # at their left.
    spinner_csym = Spinner(
        title="Csym",
        low=1,
        step=1,
        mode="int",
        value=BOOKMARK_DEFAULTS["csym"][1],
        width=64,
        syncable=False,
    )
    csym_controls = ColumnDataSource(
        data=dict(value=[spinner_csym.value], revision=[0])
    )
    csym_controls.syncable = False
    csym_callbacks = _setup_csym_control_js(
        spinner_csym, csym_controls, session.ns("hill_csym")
    )

    # Send complete parameter sets through the module's actual input IDs.
    helix_input_ids = {
        name: session.ns(f"hill_{name}")
        for name in (
            "twist",
            "pitch",
            "rise",
            "use_twist_pitch",
            "helix_revision",
        )
    }
    helix_controls = ColumnDataSource(
        data=dict(
            twist=[curr_twist],
            rise=[curr_rise],
            pitch=[curr_pitch],
            keep=["Twist"],
            tilt=[0.0],
            revision=[0],
        )
    )
    helix_controls.syncable = False
    slider_callbacks = _setup_helix_control_js(
        slider_twist,
        slider_pitch,
        slider_rise,
        fig_ellipses,
        helix_input_ids,
        helix_controls,
    )

    # ── Main plot layout ────────────────────────────────────────
    figs_row = gridplot(
        children=[figs], toolbar_location="right", sizing_mode="stretch_width"
    )
    figs_grid = layout(
        children=[
            row(
                spinner_csym,
                slider_twist,
                slider_pitch,
                slider_rise,
                sizing_mode="stretch_width",
            ),
            figs_row,
        ],
        sizing_mode="stretch_width",
    )
    main_plot_widget = BokehModel(figs_grid)

    # ── Straightening figures ───────────────────────────────────
    fig_straighten, data_source_straighten = hill.create_image_figure(
        init_img,
        _INIT_APIX,
        _INIT_APIX,
        title=f"Original image ({_INIT_NX}x{_INIT_NY})",
        title_location="below",
        show_axis=False,
        show_toolbar=True,
        crosshair_color="white",
    )
    markers_data_source = ColumnDataSource({"x": [], "y": []})
    straighten_marker = fig_straighten.scatter(
        x="x", y="y", source=markers_data_source, color="red", size=20
    )
    draw_tool = PointDrawTool(
        renderers=[straighten_marker], default_overrides={"color": "red", "size": 20}
    )
    fig_straighten.add_tools(draw_tool)
    fig_straighten.toolbar.active_tap = draw_tool

    spline_source = ColumnDataSource({"x": [], "y": []})
    fig_straighten.line("x", "y", source=spline_source, line_width=3, line_color="red")

    # JS bridge for straightening markers
    markers_data_source.js_on_change(
        "data",
        CustomJS(
            args=dict(src=markers_data_source),
            code=r"""
        const xs = src.data.x || [], ys = src.data.y || [];
        const payload = { x: xs, y: ys, n: xs.length, ts: Date.now() };
        function send(w) {
            try { if (w && w.Shiny && w.Shiny.setInputValue) {
            w.Shiny.setInputValue("hill_straighten_pts", payload, {priority: "event"});
            }} catch(e) {}
        }
        send(window);
        """,
        ),
    )

    fig_after_straighten, data_source_after = hill.create_image_figure(
        init_img,
        _INIT_APIX,
        _INIT_APIX,
        title=f"Straightened image ({_INIT_NX}x{_INIT_NY})",
        title_location="below",
        show_axis=False,
        show_toolbar=True,
        crosshair_color="white",
    )

    # ── Sidebar image display figures ──────────────────────────
    fig_transformed_img, source_transformed_img = hill.create_image_figure(
        init_img,
        _INIT_APIX,
        _INIT_APIX,
        title=f"Transformed Image ({_INIT_NX}x{_INIT_NY})",
        title_location="below",
        show_axis=False,
        show_toolbar=False,
        crosshair_color="white",
    )

    # ── Radial profile figure ──────────────────────────────────
    _rp_x = np.arange(-_INIT_NX // 2, _INIT_NX // 2) * _INIT_APIX
    _rp_ymax = np.max(init_img, axis=0)
    _rp_ymean = np.mean(init_img, axis=0)

    _rp_tools = "box_zoom,crosshair,hover,pan,reset,save,wheel_zoom"
    _rp_tooltips = [("X", "@x{0.0}Å")]
    fig_radial_profile = figure(
        x_axis_label="x (Å)",
        y_axis_label="pixel value",
        frame_height=200,
        tools=_rp_tools,
        tooltips=_rp_tooltips,
    )
    fig_radial_profile_line_max = fig_radial_profile.line(
        _rp_x, _rp_ymax, line_width=2, color="red", legend_label="max"
    )
    fig_radial_profile_line_max_flipped = fig_radial_profile.line(
        -_rp_x,
        _rp_ymax,
        line_width=2,
        color="red",
        line_dash="dashed",
        legend_label="max flipped",
    )
    fig_radial_profile_line_mean = fig_radial_profile.line(
        _rp_x, _rp_ymean, line_width=2, color="blue", legend_label="mean"
    )
    fig_radial_profile_line_mean_flipped = fig_radial_profile.line(
        -_rp_x,
        _rp_ymean,
        line_width=2,
        color="blue",
        line_dash="dashed",
        legend_label="mean flipped",
    )
    _rp_rmin_span = Span(
        location=-_INIT_HELICAL_RADIUS,
        dimension="height",
        line_color="green",
        line_dash="dashed",
        line_width=3,
    )
    _rp_rmax_span = Span(
        location=_INIT_HELICAL_RADIUS,
        dimension="height",
        line_color="green",
        line_dash="dashed",
        line_width=3,
    )
    fig_radial_profile.add_layout(_rp_rmin_span)
    fig_radial_profile.add_layout(_rp_rmax_span)
    fig_radial_profile.yaxis.visible = False
    fig_radial_profile.legend.visible = False
    fig_radial_profile.legend.location = "top_right"
    fig_radial_profile.legend.click_policy = "hide"
    _toggle_legend_rp = CustomJS(
        args=dict(leg=fig_radial_profile.legend[0]),
        code="leg.visible = !leg.visible;",
    )
    fig_radial_profile.js_on_event(DoubleTap, _toggle_legend_rp)

    # ── ACF figure ─────────────────────────────────────────────
    _init_acf = hill.auto_correlation(init_img, sqrt=True, high_pass_fraction=0.1)
    _acf_ny = _INIT_NY
    _acf_y = np.arange(-_acf_ny // 2, _acf_ny // 2) * _INIT_APIX
    _acf_xmax = np.max(_init_acf, axis=1)

    _acf_tools = "box_zoom,crosshair,hover,pan,reset,save,wheel_zoom"
    _acf_tooltips = [("Axial Shift", "@y{0.0}Å")]
    fig_acf = figure(
        x_axis_label="Auto-correlation",
        y_axis_label="Axial Shift (Å)",
        frame_height=_acf_ny,
        y_range=[_acf_ny // 2 * _INIT_APIX, -_acf_ny // 2 * _INIT_APIX],
        sizing_mode="scale_both",
        tools=_acf_tools,
        tooltips=_acf_tooltips,
    )
    fig_acf_line = fig_acf.line(
        _acf_xmax, _acf_y, line_width=2, color="red", legend_label="ACF"
    )
    fig_acf.hover[0].attachment = "above"
    fig_acf.legend.visible = False
    fig_acf.legend.location = "top_right"
    fig_acf.legend.click_policy = "hide"
    _toggle_legend_acf = CustomJS(
        args=dict(leg=fig_acf.legend[0]),
        code="leg.visible = !leg.visible;",
    )
    fig_acf.js_on_event(DoubleTap, _toggle_legend_acf)

    # ── Reactive effects ────────────────────────────────────────

    # ── Re-sizer ─────────────────────────────────────────────────
    # Resize only when plot dimension inputs or visibility change.
    # Reading clientdata.output_width/output_height non-reactively
    # (via isolate) prevents the feedback loop where setting frame
    # sizes causes a Bokeh re-render, which changes the container size,
    # which re-triggers this effect, progressively shrinking the plots.
    _last_plot_size = [None]

    @reactive.effect(priority=-100)
    @reactive.event(
        input.hill_pnx,
        input.hill_pny,
        input.hill_PS,
        input.hill_YP,
        input.hill_PD,
        input.hill_sidebar_navset,
        ignore_init=False,
    )
    def _resize_main_plots():
        with reactive.isolate():
            pw = session.clientdata.output_width("hill_main_plots")
            ph = session.clientdata.output_height("hill_main_plots")
        if pw is None or pw <= 0 or ph is None or ph <= 0:
            return
        # Subtract space for the slider row (~40px),
        # plot title (~30px), toolbar (~60px), and margins (~20px).
        available_width = max(int(pw) - 60, 1)
        available_height = max(int(ph) - 100, 1)
        # Keep the figures responsive in width so the gridplot row (PS, YP, PD)
        # plus its right-side toolbar fits the container instead of overflowing
        # and cropping the right edge. ``width_policy="fit"`` makes each figure
        # scale to its allocated column; the frame height keeps the aspect.
        for fig in [fig_ps, fig_yp, fig_pd]:
            fig.width_policy = "fit"
            fig.height_policy = "fit"

    # ── Sidebar tab switching ────────────────────────────────────
    @reactive.effect
    @reactive.event(input.hill_sidebar_navset)
    def _sidebar_tab_switch():
        if input.hill_sidebar_navset() == "Filament Straightening":
            ui.update_navset("hill_main_tabs", selected="hill_straighten_tab")
            straighten_tab_activated.set(True)
        else:
            ui.update_navset("hill_main_tabs", selected="hill_main_content_tab")

    @reactive.effect
    @reactive.event(input.hill_button_back_to_main_page, ignore_init=True)
    def _back_to_main():
        ui.update_navset("hill_main_tabs", selected="hill_main_content_tab")
        ui.update_navset("hill_sidebar_navset", selected="hill_input_sidebar")

    # ── M choices update ─────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.hill_m_max, ignore_init=True)
    def _update_m_choices():
        prev = input.hill_ms()
        ms = [str(m) for m in range(input.hill_m_max(), -input.hill_m_max() - 1, -1)]
        # inline, as it was drawn: an update that leaves it out stacks them again
        ui.update_checkbox_group("hill_ms", choices=ms, selected=prev, inline=True)

    @reactive.effect(priority=10)
    @reactive.event(
        input.hill_twist,
        input.hill_rise,
        input.hill_use_twist_pitch,
        input.hill_out_of_plane_tilt,
        input.hill_helix_revision,
    )
    def _sync_helix_controls():
        twist, rise = input.hill_twist(), input.hill_rise()
        req(np.isfinite(twist), np.isfinite(rise), rise > 0)
        revision = input.hill_helix_revision() or 0
        helix_controls.data = dict(
            twist=[twist],
            rise=[rise],
            pitch=[hill.twist2pitch(twist, rise)],
            keep=[input.hill_use_twist_pitch()],
            tilt=[input.hill_out_of_plane_tilt()],
            revision=[int(revision)],
        )

    def _model_follows_inputs():
        """Bring the Bokeh model up to date with the inputs.

        What a bookmark restored (twist, rise, csym) can arrive after the model
        was built and before the browser has the widget, and a patch sent then
        is lost; the widget is built from the model as it stands, so the model
        must be right.

        Returns
        -------
        bool
            Whether the model had to change, that is, whether the widget must be
            built again.
        """
        try:
            twist, rise, csym = input.hill_twist(), input.hill_rise(), _csym()
        except SilentException:
            return False  # the inputs are not there yet: nothing to follow
        if not (np.isfinite(twist) and np.isfinite(rise) and rise > 0):
            return False
        pitch = hill.twist2pitch(twist, rise)
        changed = _set_helix_sliders(
            slider_twist, slider_pitch, slider_rise, twist, pitch, rise
        )
        if spinner_csym.value != csym:
            spinner_csym.value = csym
            changed = True
        return changed

    revision_of_csym = [0]

    @reactive.effect
    @reactive.event(input.hill_csym)
    def _sync_csym_spinner():
        """Show a csym that was set elsewhere (a bookmark, a candidate row)."""
        csym = _csym()
        if input.hill_csym() != csym:
            # not a whole number (a bookmark's csym=2.5, a decimal typed): the
            # input is made one, so that everything that reads it agrees
            ui.update_numeric("hill_csym", value=csym)
        # Two ways, for two cases. The model's own value is what a box that is
        # still being drawn starts from (the browser's callbacks run only on a
        # change after that, which is how a bookmark's csym was missed); it is
        # not changed by what is typed in the browser, so setting it again to the
        # value it has says nothing, and the revision of the channel does.
        spinner_csym.value = csym
        revision_of_csym[0] += 1
        csym_controls.data = dict(value=[csym], revision=[revision_of_csym[0]])

    # ── Layer line / display param changes ──────────────────────
    @reactive.effect
    @reactive.event(
        has_images,
        input.hill_input_type,
        input.hill_m_max,
        input.hill_diameter,
        input.hill_csym,
        input.hill_out_of_plane_tilt,
        input.hill_cutoff_res_x,
        input.hill_cutoff_res_y,
        input.hill_fft_top_only,
        input.hill_ll_colors,
        input.hill_LL,
        input.hill_LLText,
        input.hill_pnx,
        input.hill_pny,
        input.hill_twist,
        input.hill_rise,
    )
    def _update_layerline_figures():
        with _held(figs[0].document):
            _redraw_layerline_figures()

    def _redraw_layerline_figures():
        # Frame sizing is managed by _resize_main_plots — don't override it here.
        for e in fig_ellipses:
            for f in figs_image:
                if e in f.renderers:
                    f.renderers.remove(e)
        fig_ellipses.clear()
        _update_slider_renderers(slider_callbacks, fig_ellipses)
        if not has_images():
            return
        if input.hill_LL():
            curr_m_groups = hill.compute_layer_line_positions(
                twist=input.hill_twist(),
                rise=input.hill_rise(),
                csym=_csym(),
                radius=input.hill_diameter() / 2,
                tilt=input.hill_out_of_plane_tilt(),
                cutoff_res=input.hill_cutoff_res_y(),
                m_max=input.hill_m_max(),
            )
            colors = input.hill_ll_colors().split()
            if max(curr_m_groups[0]["LL"][0]) > 0:
                x0, _, _ = curr_m_groups[0]["LL"]
                tmp = np.sort(np.unique(x0))
                width = np.mean(tmp[1:] - tmp[:-1])
                height = width / 5
                for mi, m_key in enumerate(curr_m_groups.keys()):
                    x, y, bessel_n = curr_m_groups[m_key]["LL"]
                    texts = (
                        [str(int(n)) for n in bessel_n] if input.hill_LLText() else None
                    )
                    tags = [m_key, bessel_n]
                    color = colors[abs(m_key) % len(colors)]
                    alpha = [n % 2 * 1.0 for n in bessel_n]
                    for f in figs_image:
                        if input.hill_LLText():
                            gl = f.text(
                                x,
                                y,
                                y_offset=0.0,
                                text=texts,
                                text_color=color,
                                text_baseline="middle",
                                text_align="center",
                                visible=str(m_key) in input.hill_ms(),
                            )
                        else:
                            gl = f.ellipse(
                                x,
                                y,
                                width=width,
                                height=height,
                                line_color=color,
                                fill_color=color,
                                fill_alpha=alpha,
                                line_width=1.0,
                                visible=str(m_key) in input.hill_ms(),
                            )
                        gl.tags = tags
                        # Layer-line previews are browser-local while a control is
                        # moving.  The server rebuilds them after the final commit.
                        gl.data_source.syncable = False
                        fig_ellipses.append(gl)

        # The frame and the Bessel image depend on the sampling, the filament
        # and the tilt, not on twist, rise or the layer lines drawn. Sending them
        # again is most of the cost of a redraw (two 1 MB arrays to the browser),
        # so it is done only when one of those changed.
        ny = input.hill_pny()
        nx = input.hill_pnx()
        frame_key = (
            ny,
            nx,
            input.hill_cutoff_res_x(),
            input.hill_cutoff_res_y(),
            input.hill_diameter(),
            input.hill_out_of_plane_tilt(),
            input.hill_fft_top_only(),
        )
        if frame_key != _layerline_frame_key[0]:
            _layerline_frame_key[0] = frame_key
            dsy = 1.0 / (ny // 2 * input.hill_cutoff_res_y())
            dsx = 1.0 / (nx // 2 * input.hill_cutoff_res_x())
            x_range = (-(nx // 2) * dsx, (nx // 2) * dsx)
            if input.hill_fft_top_only():
                y_range = (-(ny // 2 * 0.01) * dsy, (ny // 2 - 0.5) * dsy)
            else:
                y_range = (-(ny // 2 + 0.5) * dsy, (ny // 2 - 0.5) * dsy)

            bessel = hill.bessel_n_image(
                ny,
                nx,
                input.hill_cutoff_res_x(),
                input.hill_cutoff_res_y(),
                input.hill_diameter() / 2,
                input.hill_out_of_plane_tilt(),
            )

            for f in figs:
                f.y_range.update(start=y_range[0], end=y_range[1])
            for f in figs_image:
                f.x_range.update(start=x_range[0], end=x_range[1])

            for source in (data_source_ps, data_source_pd):
                source.data = {
                    **source.data,
                    "x": [-nx // 2 * dsx],
                    "y": [-ny // 2 * dsy],
                    "dw": [nx * dsx],
                    "dh": [ny * dsy],
                    "bessel": [bessel],
                }

        _update_slider_renderers(slider_callbacks, fig_ellipses)

    # ── Color palette updates ────────────────────────────────────
    @reactive.effect
    @reactive.event(input.hill_const_image_color, input.hill_Color)
    def _update_color_palette():
        from bokeh.models.glyphs import Image as ImageGlyph
        from bokeh.palettes import Viridis256, Greys256

        if input.hill_const_image_color():
            palette = tuple(input.hill_const_image_color().split())
        else:
            palette = Viridis256 if input.hill_Color() else Greys256
        for f in figs_image:
            for rend in f.renderers:
                glyph = getattr(rend, "glyph", None)
                if glyph and isinstance(glyph, ImageGlyph):
                    glyph.color_mapper.palette = palette

    # ── Phase hover toggle ───────────────────────────────────────
    @reactive.effect
    @reactive.event(input.hill_Phase)
    def _toggle_phase_hover():
        ps_h = next(t for t in fig_ps.tools if isinstance(t, HoverTool))
        pd_h = next(t for t in fig_pd.tools if isinstance(t, HoverTool))
        if input.hill_Phase():
            ps_h.tooltips = [
                ("Res r", "A"),
                ("Res y", "A"),
                ("Res x", "A"),
                ("Jn", "@bessel"),
                ("Amp", "@image"),
                ("Phase", "@phase °"),
            ]
            pd_h.tooltips = [
                ("Res r", "A"),
                ("Res y", "A"),
                ("Res x", "A"),
                ("Jn", "@bessel"),
                ("Phase Diff", "@image °"),
                ("Phase", "@phase °"),
            ]
        else:
            ps_h.tooltips = [
                ("Res r", "A"),
                ("Res y", "A"),
                ("Res x", "A"),
                ("Jn", "@bessel"),
                ("Amp", "@image"),
            ]
            pd_h.tooltips = [
                ("Res r", "A"),
                ("Res y", "A"),
                ("Res x", "A"),
                ("Jn", "@bessel"),
                ("Phase Diff", "@image °"),
            ]

    # ── Plot visibility toggles ─────────────────────────────────
    @reactive.effect
    @reactive.event(input.hill_PS, input.hill_YP, input.hill_PD, ignore_init=True)
    def _toggle_plot_visibility():
        figs[0].visible = input.hill_PS()
        figs[1].visible = input.hill_YP()
        figs[2].visible = input.hill_PD()

    @reactive.effect
    @reactive.event(input.hill_ms)
    def _toggle_layerline_visibility():
        selected = [str(m) for m in input.hill_ms()]
        for el in fig_ellipses:
            if str(el.tags[0]) in selected:
                el.visible = True
            else:
                el.visible = False

    # ── Input UI: conditional input panels ──────────────────────
    @output
    @render.ui
    def hill_conditional_input_uis():
        sel = input.hill_input_mode_params()
        if sel == "1":
            return [
                ui.p("Upload a mrc or mrcs file"),
                ui.input_file(
                    "hill_img_file_upload",
                    "Upload the image file (.mrcs, .mrc, .map)",
                    accept=[".mrcs", ".mrc", ".map", ".map.gz"],
                    placeholder="image file",
                ),
                ui.input_checkbox("hill_is_3d", "The input is a 3D map", value=False),
            ]
        elif sel == "2":
            return [
                ui.input_text(
                    "hill_img_file_url",
                    "Input a url of 2D image(s) or a 3D map:",
                    value=_DEFAULT_URL,
                    update_on="blur",
                ),
                ui.input_checkbox("hill_is_3d", "The input is a 3D map", value=False),
            ]
        elif sel == helicon.shiny.SERVER:
            return [
                helicon.shiny.file_picker_field(
                    ui.input_text(
                        "hill_img_file_server",
                        "2D image(s) or a 3D map on the server:",
                        value="",
                        placeholder="Browse, or type a path",
                        update_on="blur",
                    ),
                    "image_browse",
                ),
                ui.input_checkbox("hill_is_3d", "The input is a 3D map", value=False),
            ]
        elif sel == "3":
            return [
                ui.input_text(
                    "hill_img_file_emd_id",
                    "Input an EMDB ID (emd-xxxxx):",
                    value="emd-10499",
                    update_on="blur",
                ),
                ui.input_action_button(
                    "hill_select_random_emdb", "Select a random EMDB ID"
                ),
                ui.output_ui("hill_emdb_info_uis"),
                ui.input_checkbox("hill_is_3d", "The input is a 3D map", value=True),
            ]
        return ui.p("Please select an option.")

    @output
    @render.ui
    def hill_conditional_3d_uis():
        if input.hill_is_3d():
            return [
                ui.accordion(
                    ui.accordion_panel(
                        "Generate 2D projection from 3D map",
                        ui.input_checkbox(
                            "hill_apply_helical_sym",
                            "Apply helical symmetry",
                            value=False,
                        ),
                        ui.output_ui("hill_ahs_uis"),
                        ui.input_numeric(
                            "hill_az",
                            "Rotation around helical axis (°):",
                            min=0.0,
                            max=360.0,
                            value=0.0,
                            step=1.0,
                            update_on="blur",
                        ),
                        ui.input_numeric(
                            "hill_tilt",
                            "Tilt (°):",
                            min=-180.0,
                            max=180.0,
                            value=0.0,
                            step=1.0,
                            update_on="blur",
                        ),
                        ui.input_numeric(
                            "hill_gauss_noise_std",
                            "Add Gaussian noise (sigma):",
                            min=0.0,
                            value=0.0,
                            step=0.5,
                            update_on="blur",
                        ),
                        ui.input_action_button(
                            "hill_generate_2d_projection", "Generate 2D projection"
                        ),
                        value="hill_2d_proj",
                    ),
                    # it holds the controls a 3D map needs, so it is open
                    open=True,
                ),
            ]
        return None

    @output
    @render.ui
    @reactive.event(input.hill_apply_helical_sym)
    def hill_ahs_uis():
        req(input.hill_apply_helical_sym())
        return [
            ui.input_numeric(
                "hill_twist_ahs",
                "Twist (°):",
                value=input.hill_twist(),
                min=-180.0,
                max=180.0,
                step=1.0,
            ),
            ui.input_numeric(
                "hill_rise_ahs", "Rise (Å):", value=input.hill_rise(), min=0.0, step=1.0
            ),
            ui.input_numeric("hill_csym_ahs", "Csym:", value=_csym(), min=1, step=1),
            ui.input_numeric(
                "hill_apix_map",
                "Current map pixel size (Å):",
                value=apix_from_file(),
                min=0.0,
                step=1.0,
            ),
            ui.input_numeric(
                "hill_apix_ahs",
                "New map pixel size (Å):",
                value=apix_from_file(),
                min=0.0,
                step=1.0,
            ),
            ui.input_numeric(
                "hill_fraction_ahs",
                "Center fraction (0-1):",
                value=1.0,
                min=0,
                max=1.0,
                step=0.1,
            ),
            ui.input_numeric(
                "hill_length_ahs",
                "Box length (pixels):",
                value=max(nx_curr_img(), nz_curr_img()),
                min=0,
                step=1,
            ),
            ui.input_numeric(
                "hill_width_ahs",
                "Box width (pixels):",
                value=max(nx_curr_img(), nz_curr_img()),
                min=0,
                step=1,
            ),
            ui.hr(),
        ]

    # ── EMDB info ────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.hill_input_mode_params, input.hill_img_file_emd_id)
    def _get_emdb_link():
        req(input.hill_input_mode_params() == "3")
        req(input.hill_img_file_emd_id())
        try:
            emdb = helicon.dataset.EMDB()
            df = emdb.meta.loc[emdb.meta["emd_id"].isin(emdb.helical_structure_ids())]
            eid = input.hill_img_file_emd_id().split("-")[-1]
            entry = df[df["emd_id"] == eid]
            if not entry.empty:
                resolution = float(entry["resolution"].values[0])
                twist = float(entry["twist"].values[0])
                rise = float(entry["rise"].values[0])
                csym = int(str(entry["csym"].values[0]).lstrip("C"))
                ui.update_numeric("hill_rise", value=rise)
                ui.update_numeric("hill_twist", value=twist)
                ui.update_numeric("hill_csym", value=csym)
                url = f"https://www.ebi.ac.uk/emdb/entry/{input.hill_img_file_emd_id()}"
                msg = (
                    f'<a href="{url}" target="_blank" rel="noopener noreferrer">'
                    f"{input.hill_img_file_emd_id()}</a>"
                    f" | resolution={resolution}Å"
                    f" | twist={twist}° | rise={rise}Å"
                    f" | axial sym=C{csym}"
                )
                emd_msg.set(msg)
            else:
                emd_msg.set(f"{input.hill_img_file_emd_id()} not in EMDB helical list.")
        except Exception as e:
            emd_msg.set(f"EMDB lookup failed: {e}")

    @reactive.effect
    @reactive.event(input.hill_select_random_emdb)
    def _random_emdb():
        try:
            emdb = helicon.dataset.EMDB()
            ids = emdb.amyloid_atlas_ids()
            import random

            eid = f"EMD-{random.choice(ids)}"
            ui.update_text("hill_img_file_emd_id", value=eid)
        except Exception:
            pass

    @output
    @render.ui
    @reactive.event(emd_msg)
    def hill_emdb_info_uis():
        req(input.hill_input_mode_params() == "3")
        return ui.markdown(emd_msg())

    # ── Image gallery ────────────────────────────────────────────
    @output
    @render.ui
    def hill_select_image_gallery():
        if input.hill_is_3d():
            return None
        if len(data_all_2d()) > 0:
            return helicon.shiny.image_gallery(
                id=session.ns("hill_select_image"),
                label=reactive.value("Select image(s):"),
                images=data_all_2d,
                image_labels=data_all_2d_labels,
                image_size=image_display_size,
                initial_selected_indices=initial_selected_image_indices,
                enable_selection=True,
                allow_multiple_selection=True,
            )
        return ui.div("No images available")

    @render.text
    def hill_image_params_title():
        """The title of the Image Parameters panel: whose parameters they are."""
        labels = selected_image_labels()
        if not len(labels):
            return "Image Parameters"
        label = str(labels[min(_active_position(), len(labels) - 1)])
        if label.isdigit():
            return f"Image Parameters (image {label})"
        return f"Image Parameters ({label})"  # "Simulation", "Projection"

    @output
    @render.ui
    def hill_active_image_gallery():
        """The selected images; the one clicked here is the one the transform
        controls below act on."""
        return helicon.shiny.image_gallery(
            id=session.ns("hill_active_image"),
            label=reactive.value(
                "Selected images (click one to set its transformation):"
                if len(selected_images()) > 1
                else "Selected image:"
            ),
            images=selected_images,
            image_labels=selected_image_labels,
            image_size=image_display_size,
            justification="left",
            enable_selection=True,
            allow_multiple_selection=False,
            initial_selected_indices=reactive.value([0]),
        )

    # ── Image loading ────────────────────────────────────────────

    @reactive.effect(priority=20)
    @reactive.event(input.hill_input_mode_params)
    def _reset_helix_after_emdb():
        mode = input.hill_input_mode_params()
        if _previous_input_mode[0] == "3" and mode in ("2", helicon.shiny.SERVER):
            ui.update_numeric("hill_twist", value=_INIT_TWIST)
            ui.update_numeric("hill_rise", value=_INIT_RISE)
            ui.update_numeric("hill_csym", value=BOOKMARK_DEFAULTS["csym"][1])
        _previous_input_mode[0] = mode

    @reactive.effect
    @reactive.event(input.hill_input_mode_params, input.hill_img_file_url)
    def _load_input_data_from_url():
        req(input.hill_input_mode_params() == "2")
        if deployment.refuse_local_path(input.hill_img_file_url()):
            return
        _load_input_data(input.hill_img_file_url())

    @reactive.effect
    @reactive.event(input.hill_input_mode_params, input.hill_img_file_server)
    def _load_input_data_from_server():
        req(input.hill_input_mode_params() == helicon.shiny.SERVER)
        if deployment.refuse_server_mode():
            return
        _load_input_data(input.hill_img_file_server().strip())

    def _load_input_data(url):
        """Read the image(s) or map from a URL or a server file."""
        if not url:
            return
        # Loading an image must not overwrite restored or user-edited helix params.
        try:
            data, apix = denovo3d_pipeline.get_images_from_url(url)
            input_data.set(data)
            apix_from_file.set(apix)
            _update_with_apix_from_file(apix)
            _update_image_dims_from_input_data()
        except Exception as e:
            logger.error("Failed to load data from URL: %s", e)
            ui.modal_show(
                ui.modal(
                    f"Failed to load data: {e}",
                    title="Error",
                    easy_close=True,
                    footer=None,
                )
            )

    @reactive.effect
    @reactive.event(input.hill_input_mode_params, input.hill_img_file_upload)
    def _load_input_data_from_upload():
        req(input.hill_input_mode_params() == "1")
        fi = input.hill_img_file_upload()
        if not fi:
            return
        try:
            import mrcfile

            with mrcfile.open(fi[0]["datapath"]) as mrc:
                d = mrc.data
                apix = float(mrc.voxel_size.x)
            input_data.set(d)
            apix_from_file.set(apix)
            _update_with_apix_from_file(apix)
            _update_image_dims_from_input_data()
        except Exception as e:
            logger.error("Failed to load data from upload: %s", e)
            ui.modal_show(
                ui.modal(
                    f"Failed to load data: {e}",
                    title="Error",
                    easy_close=True,
                    footer=None,
                )
            )

    @reactive.effect
    @reactive.event(input.hill_input_mode_params, input.hill_img_file_emd_id)
    def _load_input_data_from_emdb():
        req(input.hill_input_mode_params() == "3")
        eid = input.hill_img_file_emd_id()
        if not eid:
            return
        try:
            emdb = helicon.dataset.EMDB()
            data, apix = emdb(eid)
            input_data.set(data)
            apix_from_file.set(apix)
            _update_with_apix_from_file(apix)
            ui.update_checkbox("hill_is_3d", value=True)
            _update_image_dims_from_input_data()
        except Exception as e:
            logger.error("Failed to load data from EMDB: %s", e)
            ui.modal_show(
                ui.modal(
                    f"Failed to load data: {e}",
                    title="Error",
                    easy_close=True,
                    footer=None,
                )
            )

    def _update_image_dims_from_input_data():
        d = input_data()
        if d is None:
            return
        shape = d.shape
        if len(shape) == 2:
            ny_curr_img.set(shape[0])
            nx_curr_img.set(shape[1])
            ui.update_numeric("hill_pnx", min=min(128, shape[1]))
            ui.update_numeric("hill_pny", min=min(512, shape[0]))
        elif len(shape) == 3:
            nz_curr_img.set(shape[0])
            ny_curr_img.set(shape[1])
            nx_curr_img.set(shape[2])

    @reactive.effect(priority=10)
    @reactive.event(input.hill_is_3d)
    def _clear_display_for_3d():
        if input.hill_is_3d():
            ui.update_slider("hill_angle", value=0.0)
            ui.update_slider("hill_dx", value=0.0)
            ui.update_slider("hill_dy", value=0.0)
            ui.update_slider("hill_mask_radius", value=0.0)
            ui.update_slider("hill_mask_len", value=100.0)

    @reactive.effect
    @reactive.event(input_data, input.hill_is_3d)
    def _process_input_data():
        req(input_data() is not None)
        if input.hill_is_3d():
            return
        d = input_data()
        if d.ndim == 2:
            d = np.expand_dims(d, axis=0)
        images = [d[i] for i in range(len(d))]
        included = []
        included_images = []
        for i, img in enumerate(images):
            if np.max(img) > np.min(img):
                included.append(i)
                included_images.append(img)
        labels = [f"{i + 1}" for i in included]
        data_all_2d.set(included_images)
        data_all_2d_labels.set(labels)
        initial_selected_image_indices.set([0])
        prev_data_from.set("main")
        if included_images:
            selected_images.set(included_images[:1])
            selected_image_labels.set(labels[:1])
            # Auto-estimate and transform the first image so PS/PD
            # always use the transformed image from the start.
            img0 = included_images[0].astype(np.float64)
            if input.hill_input_type() == "Image":
                try:
                    auto = hill.auto_upright_transform(
                        img0, input.hill_apix(), input.hill_auto_method()
                    )
                    angle, dx_val, dy_val = auto["angle"], auto["dx"], auto["dy"]
                    mr = auto["mask_radius"]
                    ml = 90.0
                    ui.update_slider("hill_angle", value=angle)
                    ui.update_slider("hill_dx", value=dx_val)
                    ui.update_slider("hill_dy", value=dy_val)
                    ui.update_slider("hill_mask_radius", value=mr)
                    ui.update_slider("hill_mask_len", value=ml)
                    img0 = _transformed(
                        img0,
                        dict(
                            apix=input.hill_apix(),
                            negate=input.hill_negate(),
                            angle=angle,
                            dx=dx_val,
                            dy=dy_val,
                            mask_radius=mr,
                            mask_len=ml,
                        ),
                    )
                except Exception:
                    pass
            data_2d_transformed.set(img0)
        else:
            selected_images.set([])
            selected_image_labels.set([])

    @reactive.effect(priority=10)
    @reactive.event(input.hill_auto_method, ignore_init=True)
    def _forget_estimates_of_the_other_method():
        _active_orig[0] = None
        avg_ps_pd_main_in_df.set({})

    def _csym():
        """The csym, always a whole number (see :func:`_as_csym`).

        Returns
        -------
        int
        """
        value = input.hill_csym()
        req(value is not None)
        req(np.isfinite(value))  # an emptied box
        return _as_csym(value)

    def _selected_indices():
        """The images selected in the gallery, as indices into ``data_all_2d``.

        The gallery's selection can name images that a newly loaded, shorter
        stack no longer has, until the gallery is drawn again; those are left
        out, as they are from ``selected_images``, so that the two line up.

        Returns
        -------
        list of int
        """
        n = len(data_all_2d())
        return [i for i in (input.hill_select_image() or []) if i < n]

    def _active_position():
        """Position, among the selected images, of the one clicked in the
        "Selected image(s)" gallery (the first until one is clicked).

        Returns
        -------
        int
            An index into ``selected_images()``.
        """
        try:
            sel = input.hill_active_image()
        except SilentException:
            sel = None
        n = len(selected_images())
        if sel and 0 <= int(sel[0]) < n:
            return int(sel[0])
        return 0

    def _current_params():
        """What the transform controls hold, in the form saved per image.

        Returns
        -------
        dict
            ``apix``, ``negate``, ``angle``, ``dx``, ``dy``, ``mask_radius`` and
            ``mask_len``, as :func:`_transformed` takes them.
        """
        return {
            "apix": input.hill_apix(),
            "negate": input.hill_negate(),
            "angle": input.hill_angle(),
            "dx": input.hill_dx(),
            "dy": input.hill_dy(),
            "mask_radius": input.hill_mask_radius(),
            "mask_len": input.hill_mask_len(),
        }

    def _transformed(img, params):
        """``img`` inverted, rotated, shifted and masked as ``params`` says.

        Parameters
        ----------
        img : numpy.ndarray
            The 2D image.
        params : dict
            ``apix``, and any of ``angle``, ``dx``, ``dy``, ``negate``,
            ``mask_radius`` and ``mask_len`` (percent); what is missing is 0, so
            that nothing is done for it.

        Returns
        -------
        numpy.ndarray
            The transformed image, as float64.
        """
        apix = params["apix"]
        angle, dx, dy = params.get("angle", 0), params.get("dx", 0), params.get("dy", 0)
        negate = params.get("negate", False)
        mask_radius, mask_len = params.get("mask_radius", 0), params.get("mask_len", 0)
        temp = img.astype(np.float64)
        if angle or dx or dy or negate:
            temp = hill.transform_2d_image(temp, angle, dx, dy, negate, apix)
        if mask_radius > 0 and 0 < mask_len <= 100:
            temp = hill.mask_2d_filament(temp, mask_radius, apix, mask_len / 100.0)
        return temp

    def _ps_scaling():
        """How the spectra are scaled for display, as the keywords of the
        spectrum functions: the log and Fourier filters, or the shell median.

        Returns
        -------
        dict
            ``shell_median``, ``log``, ``low_pass_fraction`` and
            ``high_pass_fraction``, for :func:`hill_compute.compute_power_spectra`
            and :func:`hill_compute.resize_rescale_power_spectra`.
        """
        return dict(
            shell_median=input.hill_ps_scaling() == "shell",
            log=input.hill_log_amp(),
            low_pass_fraction=input.hill_lp_fraction() / 100.0,
            high_pass_fraction=input.hill_hp_fraction() / 100.0,
        )

    def _stash_active_params():
        """Keep what was set for the image the controls are showing, so that
        clicking another image and coming back finds it as it was left."""
        orig = _active_orig[0]
        if orig is None or input.hill_is_3d() or input.hill_input_type() != "Image":
            return
        if not input.hill_apix() > 0:
            return
        avg_ps_pd_main_in_df.set({**avg_ps_pd_main_in_df(), orig: _current_params()})

    @reactive.effect
    @reactive.event(input.hill_select_image, input.hill_auto_method)
    def _update_selected_images():
        sel = input.hill_select_image()
        if sel is None or len(sel) == 0:
            _active_orig[0] = None
            selected_images.set([])
            selected_image_labels.set([])
            return
        indices = list(sel)
        all_imgs = data_all_2d()
        all_labels = data_all_2d_labels()
        # The first selected image is the one the gallery shows as active, and
        # the one whose parameters are loaded below.
        _stash_active_params()
        _active_orig[0] = indices[0] if indices[0] < len(all_imgs) else None
        chosen = [all_imgs[i] for i in indices if i < len(all_imgs)]
        chosen_labels = [all_labels[i] for i in indices if i < len(all_labels)]
        selected_images.set(chosen)
        selected_image_labels.set(chosen_labels)

        prev_data_from.set("main")

        mode = input.hill_input_type()
        if mode in ("PS", "PD") or input.hill_is_3d():
            if not input.hill_inhibit_update():
                ui.update_slider("hill_angle", value=0.0)
                ui.update_slider("hill_dx", value=0.0)
                ui.update_slider("hill_dy", value=0.0)
        else:
            if input.hill_inhibit_update():
                ui.update_slider("hill_mask_len", value=90.0)
                ui.update_checkbox("hill_inhibit_update", value=False)
            else:
                valid_indices = [i for i in indices if i < len(all_imgs)]
                if valid_indices:
                    sel_idx = valid_indices[0]
                    if input.hill_input_type() == "Image":
                        if sel_idx not in avg_ps_pd_main_in_df():
                            auto = hill.auto_upright_transform(
                                chosen[0], input.hill_apix(), input.hill_auto_method()
                            )
                            angle, dx_val, dy_val = (
                                auto["angle"],
                                auto["dx"],
                                auto["dy"],
                            )
                            mr = auto["mask_radius"]
                            ml = 90.0
                            ui.update_slider("hill_angle", value=angle)

                            ui.update_slider("hill_dx", value=dx_val)

                            ui.update_slider("hill_dy", value=dy_val)

                            ui.update_slider("hill_mask_radius", value=mr)

                            ui.update_slider("hill_mask_len", value=ml)

                        else:
                            saved = avg_ps_pd_main_in_df()[sel_idx]
                            angle = saved["angle"]
                            dx_val = saved["dx"]
                            dy_val = saved["dy"]
                            mr = saved["mask_radius"]
                            ml = saved["mask_len"]
                            ui.update_slider("hill_angle", value=angle)

                            ui.update_slider("hill_dx", value=dx_val)

                            ui.update_slider("hill_dy", value=dy_val)

                            ui.update_slider("hill_mask_radius", value=mr)

                            ui.update_slider("hill_mask_len", value=ml)

                        # apply transform with local params
                        data_2d_transformed.set(
                            _transformed(
                                chosen[0],
                                dict(
                                    apix=input.hill_apix(),
                                    negate=input.hill_negate(),
                                    angle=angle,
                                    dx=dx_val,
                                    dy=dy_val,
                                    mask_radius=mr,
                                    mask_len=ml,
                                ),
                            )
                        )

            # save transformation parameters for all selected images
            valid_indices = [i for i in indices if i < len(all_imgs)]
            apix_val = input.hill_apix()
            for i, sel_idx in enumerate(valid_indices):
                if not input.hill_inhibit_update():
                    if sel_idx not in avg_ps_pd_main_in_df():
                        if i > 0:
                            auto = hill.auto_upright_transform(
                                all_imgs[sel_idx], apix_val, input.hill_auto_method()
                            )
                        else:
                            pass  # already estimated above
                        d = avg_ps_pd_main_in_df()
                        d[sel_idx] = {
                            "apix": apix_val,
                            "negate": input.hill_negate(),
                            "angle": auto["angle"],
                            "dx": auto["dx"],
                            "dy": auto["dy"],
                            "mask_radius": auto["mask_radius"],
                            "mask_len": 90.0,
                        }
                        avg_ps_pd_main_in_df.set(d)

    # ── Clear saved params when data changes ────────────────────
    @reactive.effect
    @reactive.event(data_all_2d)
    def _clear_avg_ps_pd_main_in_df():
        _active_orig[0] = None
        avg_ps_pd_main_in_df.set({})

    # ── Multi-selection: the controls follow the image clicked in the gallery ─
    def _show_transformed(img, params):
        """Show ``img`` transformed as ``params`` says, below the controls.

        Parameters
        ----------
        img : numpy.ndarray
            The 2D image.
        params : dict
            As for :func:`_transformed`.
        """
        temp = _transformed(img, params)
        hill.update_image_figure(
            fig_transformed_img,
            source_transformed_img,
            temp,
            params["apix"],
            title="Transformed image",
        )
        # data_2d_transformed stays with the first image while several are
        # selected (setting it would also re-estimate the diameter), so the
        # auto-correlation of the image shown here is drawn from here.
        if input.hill_input_type() == "Image":
            _set_acf(temp)

    @reactive.effect
    @reactive.event(input.hill_active_image)
    def _update_current_image_from_multiselect():
        req(len(selected_images()) > 1)
        req(input.hill_apix() > 0)
        valid = _selected_indices()
        pos = _active_position()
        if pos >= len(valid):
            return
        new = valid[pos]
        if new == _active_orig[0]:
            return  # the controls already hold this image's parameters
        _stash_active_params()
        _active_orig[0] = new
        record = avg_ps_pd_main_in_df().get(new)
        if record is None:
            return
        record = {**record, "negate": bool(record.get("negate"))}
        ui.update_numeric("hill_apix", value=record["apix"])
        ui.update_slider("hill_angle", value=record["angle"])
        ui.update_slider("hill_dx", value=record["dx"])
        ui.update_slider("hill_dy", value=record["dy"])
        ui.update_slider("hill_mask_radius", value=record["mask_radius"])
        ui.update_slider("hill_mask_len", value=record["mask_len"])
        ui.update_checkbox("hill_negate", value=record["negate"])
        # Shown now rather than when the controls echo the new values, which
        # they do not when the two images happen to have the same ones.
        _show_transformed(data_all_2d()[new], record)

    # ── Multi-selection: update transformed display on param change ─
    @reactive.effect
    @reactive.event(
        input.hill_negate,
        input.hill_angle,
        input.hill_dx,
        input.hill_dy,
        input.hill_mask_radius,
        input.hill_mask_len,
        input.hill_apix,
    )
    def _update_transformed_from_multiselect():
        req(len(selected_images()) > 1)
        req(input.hill_apix() > 0)
        curr_idx = _active_position()
        valid = _selected_indices()
        if curr_idx >= len(valid):
            return
        _show_transformed(data_all_2d()[valid[curr_idx]], _current_params())

    # ── Average PS & PD from selected images ─────────────────────
    @reactive.effect
    @reactive.event(input.hill_run_get_average_power_spectra)
    def _average_ps_pd_from_multiselect():
        req(len(selected_images()) > 1)
        _stash_active_params()  # the image being edited has not been left yet
        valid = _selected_indices()
        all_imgs = data_all_2d()
        avg_ps = np.zeros((input.hill_pny(), input.hill_pnx()))
        avg_pd = np.zeros((input.hill_pny(), input.hill_pnx()))
        for i, sel_idx in enumerate(valid):
            img = all_imgs[sel_idx]
            record = avg_ps_pd_main_in_df().get(sel_idx, {})
            temp = _transformed(img, {"apix": input.hill_apix(), **record})
            itype = input.hill_input_type()
            if itype == "Image":
                pwr, phase = hill.compute_power_spectra(
                    temp,
                    apix=record.get("apix", input.hill_apix()),
                    cutoff_res=(input.hill_apix() * 2, input.hill_apix() * 2),
                    output_size=(input.hill_pny(), input.hill_pnx()),
                    log=False,
                    do_normalize=False,
                    low_pass_fraction=0,
                    high_pass_fraction=0,
                )
                curr_pd = hill.compute_phase_difference_across_meridian(phase)
                avg_ps += pwr
                avg_pd += curr_pd
            elif itype == "PS":
                avg_ps += temp
            elif itype == "PD":
                avg_pd += temp
        count = len(valid)
        avg_ps /= count
        avg_pd /= count
        avg_ps_pd_df.set(
            pd.DataFrame(
                {
                    "ps_avg": [avg_ps],
                    "pd_avg": [avg_pd],
                    "group name": ["average power spectra from selected images"],
                    "#images": [count],
                }
            )
        )
        ps_data.set(
            hill.resize_rescale_power_spectra(
                avg_ps,
                nyquist_res=input.hill_apix() * 2,
                cutoff_res=(input.hill_cutoff_res_y(), input.hill_cutoff_res_x()),
                output_size=(input.hill_pny(), input.hill_pnx()),
                **_ps_scaling(),
                norm=1,
            )
        )
        pd_data.set(
            hill.resize_rescale_power_spectra(
                avg_pd,
                nyquist_res=input.hill_apix() * 2,
                cutoff_res=(input.hill_cutoff_res_y(), input.hill_cutoff_res_x()),
                output_size=(input.hill_pny(), input.hill_pnx()),
                log=0,
                low_pass_fraction=0,
                high_pass_fraction=0,
                norm=0,
            )
        )
        prev_data_from.set("avg_ps_pd")

    # ── Straightened-filament pooled averaging ───────────────────
    @reactive.effect
    @reactive.event(input.hill_button_add_for_avg_ps_pd)
    def _add_straightened_for_avg():
        curr_list = straightened_filaments_for_avg()
        curr_list.append(data_2d_transformed())
        straightened_filaments_for_avg.set(list(curr_list))

    @reactive.effect
    @reactive.event(input.hill_button_clear_straightened)
    def _clear_straightened():
        straightened_filaments_for_avg.set([])

    @reactive.effect
    @reactive.event(input.hill_button_avg_ps_from_straightened)
    def _average_straightened():
        req(len(straightened_filaments_for_avg()) > 1)
        straightened = straightened_filaments_for_avg()
        avg_ps = np.zeros((input.hill_pny(), input.hill_pnx()))
        avg_pd = np.zeros((input.hill_pny(), input.hill_pnx()))
        for img in straightened:
            pwr, phase = hill.compute_power_spectra(
                img,
                apix=input.hill_apix(),
                cutoff_res=(input.hill_apix() * 2, input.hill_apix() * 2),
                output_size=(input.hill_pny(), input.hill_pnx()),
                log=False,
                do_normalize=False,
                low_pass_fraction=0,
                high_pass_fraction=0,
            )
            avg_ps += pwr
            avg_pd += hill.compute_phase_difference_across_meridian(phase)
        count = len(straightened)
        avg_ps /= count
        avg_pd /= count
        avg_ps_pd_df.set(
            pd.DataFrame(
                {
                    "ps_avg": [avg_ps],
                    "pd_avg": [avg_pd],
                    "group name": ["average PS & PD from straightened filaments"],
                    "#images": [count],
                }
            )
        )
        ps_data.set(
            hill.resize_rescale_power_spectra(
                avg_ps,
                nyquist_res=input.hill_apix() * 2,
                cutoff_res=(input.hill_cutoff_res_y(), input.hill_cutoff_res_x()),
                output_size=(input.hill_pny(), input.hill_pnx()),
                **_ps_scaling(),
                norm=1,
            )
        )
        pd_data.set(
            hill.resize_rescale_power_spectra(
                avg_pd,
                nyquist_res=input.hill_apix() * 2,
                cutoff_res=(input.hill_cutoff_res_y(), input.hill_cutoff_res_x()),
                output_size=(input.hill_pny(), input.hill_pnx()),
                log=0,
                low_pass_fraction=0,
                high_pass_fraction=0,
                norm=0,
            )
        )
        prev_data_from.set("avg_ps_pd")

    # ── 2D transformation UI ─────────────────────────────────────
    # The Image Parameters panel starts collapsed, and Shiny does not run an output
    # that is hidden. These two draw the pixel size, the contrast and the
    # auto-transform that the transform, the spectra and the symmetry search all
    # read: not drawn, hill_apix() has no value and they silently do nothing.
    @output(suspend_when_hidden=False)
    @render.ui
    def hill_img_flags_ui():
        """The contrast and auto-transform controls, beside "Input is:"."""
        # Redrawn with the controls below, as the contrast box always was, so
        # that an image of another size starts not inverted.
        nx_curr_img()
        ny_curr_img()
        if input.hill_input_type() != "Image":
            return None
        return [
            ui.input_checkbox(
                "hill_negate", "Invert image contrast", value=False, width="auto"
            ),
            ui.tooltip(
                ui.input_action_button(
                    "hill_auto_transform",
                    "Auto-transform",
                    class_="btn-sm btn-primary",
                ),
                "Estimate the rotation, shift and mask of this image again "
                "(by the method chosen under Parameters) and set the controls "
                "to them.",
            ),
        ]

    @output(suspend_when_hidden=False)
    @render.ui
    def hill_img_2d_uis():
        itype = input.hill_input_type()
        if itype == "Image":
            return [
                ui.layout_columns(
                    ui.input_numeric(
                        "hill_apix",
                        "Pixel size (A/pixel)",
                        value=apix_from_file(),
                        min=0.1,
                        max=1000.0,
                        step=0.01,
                        update_on="blur",
                    ),
                    helicon.shiny.slider(
                        "hill_angle",
                        "Rotate (°)",
                        min=-180.0,
                        max=180.0,
                        value=0.0,
                        step=1.0,
                    ),
                    helicon.shiny.slider(
                        "hill_dx",
                        "Shift X (Å)",
                        min=-max(1.0, nx_curr_img() * apix_from_file()),
                        max=max(1.0, nx_curr_img() * apix_from_file()),
                        value=0.0,
                        step=1.0,
                    ),
                    helicon.shiny.slider(
                        "hill_dy",
                        "Shift Y (Å)",
                        min=-max(1.0, ny_curr_img() * apix_from_file()),
                        max=max(1.0, ny_curr_img() * apix_from_file()),
                        value=0.0,
                        step=1.0,
                    ),
                    helicon.shiny.slider(
                        "hill_mask_radius",
                        "Mask radius (Å, 0: no mask)",
                        min=0.0,
                        max=max(1.0, nx_curr_img() / 2 * apix_from_file()),
                        value=min(
                            max(mask_radius_auto(), 0.0),
                            max(1.0, nx_curr_img() / 2 * apix_from_file()),
                        ),
                        step=1.0,
                    ),
                    helicon.shiny.slider(
                        "hill_mask_len",
                        "Mask length (%, 0: no mask)",
                        min=0.0,
                        max=100.0,
                        value=min(max(mask_len_percent_auto(), 0.0), 100.0),
                        step=1.0,
                    ),
                    col_widths=6,
                    style="align-items: flex-end;",
                ),
            ]
        else:
            return [
                ui.layout_columns(
                    ui.input_numeric(
                        "hill_apix",
                        "Pixel size (Å)",
                        value=apix_from_file(),
                        min=0.1,
                        max=1000.0,
                        step=0.01,
                        update_on="blur",
                    ),
                    helicon.shiny.slider(
                        "hill_angle",
                        "Rotate (°)",
                        min=-180.0,
                        max=180.0,
                        value=0.0,
                        step=1.0,
                    ),
                    helicon.shiny.slider(
                        "hill_dx",
                        "Shift X (Å)",
                        min=-max(1.0, nx_curr_img() * apix_from_file()),
                        max=max(1.0, nx_curr_img() * apix_from_file()),
                        value=0.0,
                        step=1.0,
                    ),
                    helicon.shiny.slider(
                        "hill_dy",
                        "Shift Y (Å)",
                        min=-max(1.0, ny_curr_img() * apix_from_file()),
                        max=max(1.0, ny_curr_img() * apix_from_file()),
                        value=0.0,
                        step=1.0,
                    ),
                    col_widths=6,
                    style="align-items: flex-end;",
                ),
            ]

    @reactive.effect
    @reactive.event(input.hill_auto_transform)
    def _auto_transform_current_image():
        req(input.hill_input_type() == "Image")
        req(input.hill_apix() > 0)
        images = selected_images()
        req(len(images) > 0)
        try:
            auto = hill.auto_upright_transform(
                images[_active_position()], input.hill_apix(), input.hill_auto_method()
            )
        except Exception as e:
            logger.warning("auto-transform failed: %s", e)
            ui.notification_show(
                f"Could not estimate the transform of this image: {e}",
                type="warning",
                duration=8,
            )
            return
        ui.update_slider("hill_angle", value=auto["angle"])
        ui.update_slider("hill_dx", value=auto["dx"])
        ui.update_slider("hill_dy", value=auto["dy"])
        ui.update_slider("hill_mask_radius", value=auto["mask_radius"])
        ui.update_slider("hill_mask_len", value=90.0)

    # ── 2D transform reactive updates ───────────────────────────
    @reactive.effect(priority=10)
    @reactive.event(
        selected_images,
        input.hill_input_type,
        input.hill_apix,
        input.hill_angle,
        input.hill_dx,
        input.hill_dy,
    )
    def _apply_2d_transform_ps_pd():
        req(input.hill_input_type() in ("PS", "PD"))
        images = selected_images()
        req(len(images) == 1, input.hill_apix() > 0)
        angle, dx, dy = input.hill_angle(), input.hill_dx(), input.hill_dy()
        if images is not _last_ps_pd_selection[0]:
            _last_ps_pd_selection[0] = images
            if not input.hill_inhibit_update():
                angle = dx = dy = 0.0
                for name in ("angle", "dx", "dy"):
                    ui.update_numeric(f"hill_{name}", value=0.0)
        temp = hill.transform_2d_image(
            images[0].astype(np.float64), angle, dx, dy, False, input.hill_apix()
        )
        _last_transform_key[0] = None
        data_2d_transformed.set(temp)

    @reactive.effect(priority=10)
    @reactive.event(
        input.hill_input_type,
        input.hill_apix,
        input.hill_angle,
        input.hill_dx,
        input.hill_dy,
        input.hill_mask_radius,
        input.hill_negate,
        input.hill_mask_len,
    )
    def _apply_2d_transform():
        req(input.hill_input_type() == "Image")
        req(len(selected_images()) == 1)
        req(input.hill_apix() > 0)
        temp = _transformed(selected_images()[0], _current_params())
        # Skip if param key unchanged (redundant round-trip)
        key = (
            id(selected_images()[0]),
            input.hill_input_type(),
            input.hill_angle(),
            input.hill_dx(),
            input.hill_dy(),
            input.hill_mask_radius(),
            input.hill_mask_len(),
            input.hill_negate(),
            input.hill_apix(),
        )
        if key == _last_transform_key[0]:
            return
        _last_transform_key[0] = key
        hill.update_image_figure(
            fig_transformed_img,
            source_transformed_img,
            temp,
            input.hill_apix(),
            title=f"Transformed Image ({nx_curr_img()}x{ny_curr_img()})",
        )
        data_2d_transformed.set(temp)

    # ── Suggest csym, twist and rise from the layer lines ────────
    symmetry_candidates = reactive.value([])
    symmetry_message = reactive.value("")
    symmetry_progress = reactive.value((0, 0))  # steps done, steps in all
    # Written by the thread the search runs in, which may not touch a reactive
    # value, and read by _poll_symmetry_progress.
    _symmetry_steps = {"done": 0, "total": 0}

    @reactive.extended_task
    async def _symmetry_search(images, apix, radius, cutoff_res):
        import asyncio

        from ..lib import layerline_index

        def progress(done, total):
            _symmetry_steps.update(done=done, total=total)

        _symmetry_steps.update(done=0, total=0)
        # in a thread, so that the page answers; the search spreads itself
        # over processes
        return await asyncio.to_thread(
            layerline_index.index_images,
            images,
            apix,
            radius,
            cutoff_res,
            top=8,
            progress=progress,
        )

    @reactive.effect
    def _poll_symmetry_progress():
        """Follow the search while it runs: the steps it reports, 3 times a second."""
        if _symmetry_search.status() != "running":
            symmetry_progress.set((0, 0))
            return
        reactive.invalidate_later(0.3)
        symmetry_progress.set((_symmetry_steps["done"], _symmetry_steps["total"]))

    @render.ui
    def hill_symmetry_progress():
        if _symmetry_search.status() != "running":
            return None
        return _progress_bar(*symmetry_progress())

    @reactive.effect
    @reactive.event(input.hill_suggest_symmetry)
    def _start_symmetry_search():
        if _symmetry_search.status() == "running":
            ui.notification_show("A search is running. Wait for it.", duration=5)
            return
        if input.hill_is_3d() or input.hill_input_type() != "Image":
            symmetry_message.set(
                "The search needs 2D images, not power spectra or maps."
            )
            return
        valid = _selected_indices()
        if not valid:
            symmetry_message.set("Select one or more images first.")
            return
        apix = input.hill_apix()
        if len(valid) == 1:
            # the transform of the image shown is the one in the controls
            records = [
                dict(
                    angle=input.hill_angle(),
                    dx=input.hill_dx(),
                    dy=input.hill_dy(),
                    negate=input.hill_negate(),
                    apix=apix,
                )
            ]
        else:
            _stash_active_params()  # the image being edited has not been left yet
            records = [avg_ps_pd_main_in_df().get(i) for i in valid]
        images = hill.images_for_indexing(
            [data_all_2d()[i] for i in valid], records, apix, input.hill_auto_method()
        )
        if len({im.shape for im in images}) > 1:
            symmetry_message.set(
                "The selected images differ in size; select images of one size."
            )
            return
        radius = input.hill_diameter() / 2
        cutoff = max(input.hill_cutoff_res_x(), 2.0 * apix)
        symmetry_candidates.set([])
        symmetry_message.set(
            f"Searching {len(images)} image{'s' if len(images) > 1 else ''} "
            f"(radius {radius:.0f} Å, resolution limit {cutoff:.1f} Å) ..."
        )
        _symmetry_search(images, apix, radius, cutoff)

    @reactive.effect
    def _symmetry_search_finished():
        status = _symmetry_search.status()
        if status == "success":
            found = _symmetry_search.result()
            symmetry_candidates.set(list(found))
            symmetry_message.set(
                "Click a row to use it. The hand of the helix (the sign of the twist) "
                "cannot be seen in a power spectrum: the twists have the sign of the "
                "current one. Candidates that put their lines at the same heights "
                "(the same rise and csym × twist) differ only in where each line "
                "starts, which depends on the diameter."
                if found
                else "No helical symmetry stood out in these images."
            )
        elif status == "error":
            symmetry_message.set(f"The search failed: {_symmetry_search.error()}")

    @render.ui
    def hill_symmetry_message():
        text = symmetry_message()
        req(text)
        return ui.p(text, style="font-size: 0.85em; margin: 0.4em 0;")

    @render.data_frame
    def hill_symmetry_candidates():
        found = symmetry_candidates()
        req(len(found))
        with reactive.isolate():
            sign = -1.0 if (input.hill_twist() or 0) < 0 else 1.0
        return render.DataGrid(
            hill.candidates_table(found, sign), selection_mode="row", width="100%"
        )

    @reactive.effect
    @reactive.event(hill_symmetry_candidates.cell_selection)
    def _use_symmetry_candidate():
        try:
            rows = list(hill_symmetry_candidates.cell_selection().get("rows", []))
        except Exception:
            return
        found = symmetry_candidates()
        req(len(rows), len(found))
        choice = found[int(rows[0])]
        sign = -1.0 if (input.hill_twist() or 0) < 0 else 1.0
        ui.update_numeric("hill_csym", value=int(choice.csym))
        ui.update_numeric("hill_rise", value=round(float(choice.rise), 3))
        ui.update_numeric("hill_twist", value=round(sign * float(choice.twist), 3))

    # ── Auto-detect filament diameter ────────────────────────────
    @reactive.effect
    @reactive.event(data_2d_transformed)
    def _auto_filament_diameter():
        req(prev_data_from() == "main")
        req(input.hill_input_type() in ("Image",))
        req(data_2d_transformed() is not None)
        if input.hill_inhibit_update():
            return
        radius_auto, _ = hill.estimate_radial_range(
            data_2d_transformed(), thresh_ratio=0.1
        )
        ui.update_numeric(
            "hill_diameter", value=round(radius_auto * input.hill_apix() * 2, 2)
        )

    # ── Compute PS/PD data ──────────────────────────────────────
    @reactive.effect
    @reactive.event(
        input.hill_input_type,
        data_2d_transformed,
        input.hill_apix,
        input.hill_cutoff_res_y,
        input.hill_cutoff_res_x,
        input.hill_pny,
        input.hill_pnx,
        input.hill_ps_scaling,
        input.hill_log_amp,
        input.hill_lp_fraction,
        input.hill_hp_fraction,
    )
    def _compute_ps_pd():
        req(prev_data_from() == "main")
        req(len(selected_images()) == 1)
        req(data_2d_transformed() is not None)
        req(input.hill_apix() > 0)
        itype = input.hill_input_type()
        if itype == "Image":
            pwr, phase = hill.compute_power_spectra(
                data_2d_transformed(),
                apix=input.hill_apix(),
                cutoff_res=(input.hill_cutoff_res_y(), input.hill_cutoff_res_x()),
                output_size=(input.hill_pny(), input.hill_pnx()),
                **_ps_scaling(),
            )
            phase_data.set(phase)
            ps_data.set(pwr)
            pd_data.set(hill.compute_phase_difference_across_meridian(phase))
        elif itype == "PS":
            ps_data.set(
                hill.resize_rescale_power_spectra(
                    data_2d_transformed(),
                    nyquist_res=input.hill_apix() * 2,
                    cutoff_res=(input.hill_cutoff_res_y(), input.hill_cutoff_res_x()),
                    output_size=(input.hill_pny(), input.hill_pnx()),
                    **_ps_scaling(),
                    norm=1,
                )
            )
        elif itype == "PD":
            pd_data.set(
                hill.resize_rescale_power_spectra(
                    data_2d_transformed(),
                    nyquist_res=input.hill_apix() * 2,
                    cutoff_res=(input.hill_cutoff_res_y(), input.hill_cutoff_res_x()),
                    output_size=(input.hill_pny(), input.hill_pnx()),
                    log=0,
                    low_pass_fraction=0,
                    high_pass_fraction=0,
                    norm=0,
                )
            )

    # ── Update Bokeh figure data from computed PS/PD ────────────
    @reactive.effect
    @reactive.event(
        ps_data,
        input.hill_yp_peak_detect,
        input.hill_yp_peak_prominence,
        input.hill_yp_peak_fit_hw,
    )
    def _update_figure_pwr_yp():
        pwr = ps_data()
        req(pwr is not None)
        phase = phase_data()
        # Only the columns that changed are sent (assigning a new dict sends them
        # all, the unchanged Bessel image too), and the image as float32: these
        # arrays are about 4 MB each and every control change sends them.
        changed = {"image": [np.asarray(pwr, dtype=np.float32)]}
        if phase is not None:
            changed["phase"] = [
                np.fmod(np.rad2deg(phase) + 360, 360).astype(np.float32)
            ]
        data_source_ps.data.update(changed)

        ny = input.hill_pny()
        dsy = 1.0 / (ny // 2 * input.hill_cutoff_res_y())
        y = np.arange(-ny // 2, ny // 2) * dsy
        yinv = y.copy()
        yinv[yinv == 0] = 1e-10
        yinv = 1.0 / np.abs(yinv)
        yprofile = np.max(pwr, axis=1)
        yprofile /= max(yprofile.max(), 1e-10)

        data_source_yp.data = {
            **data_source_yp.data,
            "yprofile": yprofile,
            "y": y,
            "resy": yinv,
        }
        _yp_lo = float(yprofile.min()) * 0.95
        _yp_hi = float(yprofile.max()) * 1.1
        fig_yp.x_range.update(start=_yp_lo, end=_yp_hi)

        # Peak detection
        if input.hill_yp_peak_detect():
            prominence = input.hill_yp_peak_prominence()
            hw = input.hill_yp_peak_fit_hw()
            peak_indices, _ = find_peaks(yprofile, prominence=prominence)
            peak_xs, peak_ys, peak_labels = [], [], []
            fit_xs, fit_ys = [], []
            for pi in peak_indices:
                lo = max(0, pi - hw)
                hi = min(len(yprofile) - 1, pi + hw)
                if hi - lo < 2:
                    continue
                y_vals = y[lo : hi + 1]
                amp_vals = yprofile[lo : hi + 1]
                coeffs = np.polyfit(y_vals, amp_vals, 2)
                a, b, c = coeffs
                if a >= 0:
                    continue
                y_peak = -b / (2 * a)
                amp_peak = np.polyval(coeffs, y_peak)
                y_fine = np.linspace(y_vals[0], y_vals[-1], 50)
                amp_fine = np.polyval(coeffs, y_fine)
                peak_xs.append(float(amp_peak))
                peak_ys.append(float(y_peak))
                peak_labels.append(
                    f"{1 / abs(y_peak):.2f} A" if abs(y_peak) > 1e-100 else "inf A"
                )
                fit_xs.append(amp_fine.tolist())
                fit_ys.append(y_fine.tolist())
            data_source_yp_peaks.data = dict(x=peak_xs, y=peak_ys, label=peak_labels)
            data_source_yp_fits.data = dict(x=fit_xs, y=fit_ys)
        else:
            data_source_yp_peaks.data = dict(x=[], y=[], label=[])
            data_source_yp_fits.data = dict(x=[], y=[])

    @reactive.effect
    @reactive.event(pd_data)
    def _update_figure_pd():
        pd = pd_data()
        req(pd is not None)
        phase = phase_data()
        data_source_pd.data.update(
            {
                "image": [np.asarray(pd, dtype=np.float32)],
                "phase": (
                    [np.fmod(np.rad2deg(phase) + 360, 360).astype(np.float32)]
                    if phase is not None
                    else []
                ),
            }
        )

    # ── Sidebar figure updates ──────────────────────────────────
    n_triggers = reactive.value(0)

    @reactive.effect
    @reactive.event(data_2d_transformed)
    def _update_transformed_figure():
        req(data_2d_transformed() is not None)
        n_triggers.set(n_triggers() + 1)
        curr_img = data_2d_transformed()
        h, w = curr_img.shape
        dx = input.hill_apix()
        dy = input.hill_apix()
        aspect_ratio = w * dx / (h * dy)
        fig_transformed_img.x_range.start = -w // 2 * dx
        fig_transformed_img.x_range.end = (w // 2 - 1) * dx
        fig_transformed_img.y_range.start = (h // 2 - 1) * dy
        fig_transformed_img.y_range.end = -h // 2 * dy
        fig_transformed_img.aspect_ratio = aspect_ratio
        fig_transformed_img.title.text = (
            f"Transformed Image ({nx_curr_img()}x{ny_curr_img()})"
        )
        source_transformed_img.data = {
            **source_transformed_img.data,
            "image": [curr_img],
            "x": [-w // 2 * dx],
            "y": [-h // 2 * dy],
            "dw": [w * dx],
            "dh": [h * dy],
        }

    @reactive.effect
    @reactive.event(
        selected_images,
        input.hill_active_image,
        input.hill_apix,
        input.hill_angle,
        input.hill_dx,
        input.hill_dy,
        input.hill_mask_radius,
        input.hill_negate,
        input.hill_mask_len,
    )
    def _update_radial_profile_data():
        req(input.hill_input_type() in ("Image",))
        req(len(selected_images()) > 0)
        req(input.hill_apix() != 0)
        curr_img = selected_images()[_active_position()]
        if (
            input.hill_angle()
            or input.hill_dx()
            or input.hill_dy()
            or input.hill_negate()
        ):
            unmasked_data = hill.transform_2d_image(
                curr_img,
                input.hill_angle(),
                input.hill_dx(),
                input.hill_dy(),
                input.hill_negate(),
                input.hill_apix(),
            )
        else:
            unmasked_data = curr_img
        curr_x = np.arange(-nx_curr_img() // 2, nx_curr_img() // 2) * input.hill_apix()
        curr_ymax = np.max(unmasked_data, axis=0)
        curr_ymean = np.mean(unmasked_data, axis=0)

        fig_radial_profile_line_max.data_source.data = {"x": curr_x, "y": curr_ymax}
        fig_radial_profile_line_max_flipped.data_source.data = {
            "x": -curr_x,
            "y": curr_ymax,
        }
        fig_radial_profile_line_mean.data_source.data = {"x": curr_x, "y": curr_ymean}
        fig_radial_profile_line_mean_flipped.data_source.data = {
            "x": -curr_x,
            "y": curr_ymean,
        }
        _rp_rmin_span.location = -input.hill_mask_radius()
        _rp_rmax_span.location = input.hill_mask_radius()

    def _set_acf(img):
        """Plot the auto-correlation of the transformed image ``img``.

        Parameters
        ----------
        img : numpy.ndarray
            The transformed 2D image.
        """
        ny, _nx = img.shape
        curr_acf = hill.auto_correlation(img, sqrt=True, high_pass_fraction=0.1)
        if curr_acf is None or np.isnan(curr_acf).any():
            return
        fig_acf.y_range.start = ny // 2 * input.hill_apix()
        fig_acf.y_range.end = -ny // 2 * input.hill_apix()
        fig_acf.frame_height = ny
        fig_acf_line.data_source.data = {
            "x": np.max(curr_acf, axis=1),
            "y": np.arange(-ny // 2, ny // 2) * input.hill_apix(),
        }

    @reactive.effect
    @reactive.event(data_2d_transformed)
    def _update_acf_data():
        req(input.hill_input_type() in ("Image",))
        req(data_2d_transformed() is not None)
        _set_acf(data_2d_transformed())

    # ── Main plot renderer ──────────────────────────────────────
    @render_widget
    def hill_main_plots():
        # What the browser is first sent is the widget as it was built. A change
        # to the model before the browser has the widget (a bookmark's twist, rise
        # and csym arrive about then) is sent as a patch to a widget that does not
        # exist yet, and is lost: so the model is made to follow the inputs and
        # the widget is built again from it.
        with reactive.isolate():
            stale = _model_follows_inputs()
        if stale:
            main_plot_widget.update_from_model(figs_grid)
        return main_plot_widget

    # ── Sidebar render functions ───────────────────────────────
    @output
    @render.ui
    def hill_error_display():
        return None

    @render_widget
    def hill_display_transformed():
        return fig_transformed_img

    @render_widget
    @reactive.event(input.hill_input_type)
    def hill_plot_radial_profile():
        req(input.hill_input_type() in ("Image",))
        return fig_radial_profile

    @render_widget
    @reactive.event(input.hill_input_type)
    def hill_plot_acf():
        req(input.hill_input_type() in ("Image",))
        return fig_acf

    # ── Straightening ────────────────────────────────────────────
    @output
    @render.ui
    def hill_filament_straightening_uis():
        return ui.div(
            ui.layout_columns(
                output_widget(
                    "hill_figure_get_markers_straighten", width="100%", height="auto"
                ),
                output_widget(
                    "hill_figure_after_straighten", width="100%", height="auto"
                ),
                col_widths=6,
            ),
            ui.input_action_button("hill_reset_markers", "Clear markers"),
            ui.input_action_button("hill_straighten_run", "Straighten"),
            ui.input_action_button(
                "hill_button_back_to_main_page", "Back to Main Plots"
            ),
            ui.input_action_button(
                "hill_button_add_for_avg_ps_pd", "Add for average PS & PD"
            ),
            ui.input_action_button(
                "hill_button_clear_straightened", "Clear added filaments"
            ),
            ui.input_action_button(
                "hill_button_avg_ps_from_straightened",
                "Average PS & PD from straightened",
            ),
            style="text-align: center; border: 1px solid #ddd; padding: 10px;",
            class_="hill-wrap-text",
        )

    @render_widget
    def hill_figure_get_markers_straighten():
        return fig_straighten

    @render_widget
    def hill_figure_after_straighten():
        return fig_after_straighten

    # Auto-sample axis
    @reactive.effect
    @reactive.event(
        selected_images,
        input.hill_num_markers_straighten,
        input.hill_template_diameter_straighten,
    )
    def _auto_sample_straighten():
        req(input.hill_do_auto_sample_axis_straighten())
        req(len(selected_images()) == 1)
        ny, nx = selected_images()[0].shape
        td = input.hill_template_diameter_straighten()
        if td is not None:
            fr = td / input.hill_apix()
        else:
            fr = None
        xs, ys = hill.sample_filament_axis(
            selected_images()[0],
            num_points=input.hill_num_markers_straighten(),
            filament_radius_pixels=fr,
        )
        xs = (xs - nx // 2) * input.hill_apix()
        ys = (ys - ny // 2) * input.hill_apix()
        markers_data_source.data = {"x": list(xs), "y": list(ys)}

    # JS bridge → spline update
    @reactive.effect
    @reactive.event(input.hill_straighten_pts)
    def _update_fitted_spline():
        payload = input.hill_straighten_pts()
        if not payload:
            return
        xs = list(map(float, payload["x"]))
        ys = list(map(float, payload["y"]))
        ny, nx = selected_images()[0].shape
        sorted_i = np.argsort(ys)
        xs = (np.array(xs)[sorted_i]) / input.hill_apix() + nx // 2
        ys = np.array(ys)[sorted_i] / input.hill_apix() + ny // 2
        tck = hill.fit_spline(xs, ys)
        if tck is not None:
            spline_ys = np.linspace(0, ny, 1000)
            spline_xs = splev(spline_ys, tck)
            spline_ys_out = (spline_ys - ny // 2) * input.hill_apix()
            spline_xs_out = (spline_xs - nx // 2) * input.hill_apix()
            spline_source.data = {"x": list(spline_xs_out), "y": list(spline_ys_out)}
            markers_straighten.set((tck, ys[0], ys[-1]))

    @reactive.effect
    @reactive.event(input.hill_reset_markers)
    def _clear_markers():
        markers_straighten.set(None)
        markers_data_source.data = {"x": [], "y": []}
        spline_source.data = {"x": [], "y": []}

    @reactive.effect(priority=100)
    @reactive.event(input.hill_straighten_run)
    def _run_straightening():
        req(markers_straighten() is not None)
        tck, y_start, y_end = markers_straighten()
        straightened = np.nan_to_num(
            hill.filament_straighten(
                selected_images()[0],
                tck,
                input.hill_output_width_straighten() // 2,
                y_start,
                y_end,
            ),
            0.0,
        )
        straightened = hill.mask_2d_filament(
            straightened,
            input.hill_mask_radius_straighten()
            / 200.0
            * input.hill_output_width_straighten(),
            input.hill_apix(),
            input.hill_mask_len_straighten() / 100.0,
        )
        data_2d_transformed.set(straightened)
        hill.update_image_figure(
            fig_after_straighten,
            data_source_after,
            straightened,
            input.hill_apix(),
            title="Straightened image",
        )

    @reactive.effect
    @reactive.event(input.hill_sidebar_navset)
    def _update_straighten_fig():
        req(input.hill_sidebar_navset() == "Filament Straightening")
        req(len(selected_images()) == 1)
        hill.update_image_figure(
            fig_straighten,
            data_source_straighten,
            selected_images()[0],
            input.hill_apix(),
            title="Original image",
        )

    # ── Simulation ───────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.hill_run_simulation)
    def _run_simulation():
        req(input.hill_ball_radius_sim() > 0)
        sim = hill.simulate_helix(
            twist=input.hill_twist(),
            rise=input.hill_rise(),
            csym=_csym(),
            helical_radius=input.hill_diameter() / 2,
            ball_radius=input.hill_ball_radius_sim(),
            ny=input.hill_pny(),
            nx=input.hill_pnx(),
            apix=input.hill_apix(),
            tilt=input.hill_tilt_sim(),
        )
        if input.hill_gauss_noise_std_sim() > 0:
            sigma = np.std(sim[sim > 1e-3])
            sim += np.random.normal(
                scale=sigma * input.hill_gauss_noise_std_sim(), size=sim.shape
            )
        proj = sim[np.newaxis, :, :]
        nz_curr_img.set(1)
        ny_curr_img.set(proj.shape[1])
        nx_curr_img.set(proj.shape[2])
        data_all_2d.set(proj)
        data_all_2d_labels.set(["Simulation"])
        selected_images.set(proj)
        selected_image_labels.set(["Simulation"])
        data_2d_transformed.set(proj[0])
        prev_data_from.set("main")

    # ── 3D projection generation ─────────────────────────────────
    @reactive.effect
    @reactive.event(input.hill_generate_2d_projection, input_data)
    def _generate_2d_projection():
        req(input_data() is not None)
        req(input.hill_is_3d())
        data_3d = input_data()
        if input.hill_apply_helical_sym():
            m = hill.symmetrize_transform_map(
                data=data_3d,
                apix=input.hill_apix_map(),
                twist_degree=input.hill_twist_ahs(),
                rise_angstrom=input.hill_rise_ahs(),
                csym=_as_csym(input.hill_csym_ahs()),
                new_size=(
                    input.hill_length_ahs(),
                    input.hill_width_ahs(),
                    input.hill_width_ahs(),
                ),
                new_apix=input.hill_apix_ahs(),
                axial_rotation=input.hill_az(),
                tilt=input.hill_tilt(),
            )
            ui.update_numeric("hill_apix", value=input.hill_apix_ahs())
        else:
            m = helicon.transform_map(
                data_3d, rot=input.hill_az(), tilt=input.hill_tilt()
            )
        proj = np.transpose(m.sum(axis=-1))[:, ::-1]
        proj = proj[np.newaxis, :, :]
        if input.hill_gauss_noise_std() > 0:
            sigma = np.std(proj[proj > 1e-3])
            proj += np.random.normal(
                scale=sigma * input.hill_gauss_noise_std(), size=proj.shape
            )
        proj = np.transpose(proj, (0, 2, 1))
        nz_curr_img.set(proj.shape[0])
        ny_curr_img.set(proj.shape[1])
        nx_curr_img.set(proj.shape[2])
        ui.update_numeric("hill_pnx", min=min(128, proj.shape[2]))
        ui.update_numeric("hill_pny", min=min(512, proj.shape[1]))
        data_all_2d.set(proj)
        data_all_2d_labels.set(["Projection"])
        selected_images.set(proj)
        selected_image_labels.set(["Projection"])
        data_2d_transformed.set(proj[0])
        prev_data_from.set("main")

    # ── Average power spectra ────────────────────────────────────
    @output
    @render.ui
    def hill_get_avg_ps_ui():
        if len(selected_images()) >= 2:
            return ui.div(
                ui.input_action_button(
                    "hill_run_get_average_power_spectra",
                    "Average PS & PD",
                    class_="btn-primary",
                ),
                style="text-align: center; margin-top: 8px;",
            )
        return None

    @output
    @render.ui
    def hill_avg_ps_meta_ui():
        return [
            ui.input_file(
                "hill_input_metadata_file",
                "Upload metadata file",
                accept=[".star", ".cs", ".lst"],
                multiple=False,
            ),
            ui.accordion(
                ui.accordion_panel(
                    ui.p("Computing settings"),
                    ui.input_numeric(
                        "hill_avg_ps_from_meta_num_cpus",
                        "Number of CPUs:",
                        min=1,
                        max=32,
                        value=5,
                        step=1,
                        update_on="blur",
                    ),
                    ui.input_numeric(
                        "hill_avg_ps_from_meta_batch_size",
                        "Batch size:",
                        min=1,
                        max=1000,
                        value=100,
                        step=1,
                        update_on="blur",
                    ),
                    value="hill_avg_ps_from_meta_computing_settings",
                ),
                open=True,
            ),
            ui.input_action_button(
                "hill_run_avg_ps_from_meta",
                "Average Power Spectra",
            ),
            ui.hr(),
        ]

    @reactive.effect
    @reactive.event(input.hill_run_avg_ps_from_meta)
    def _run_avg_ps_from_meta():
        fileinfo = input.hill_input_metadata_file()
        if fileinfo is not None:
            file_path = fileinfo[0]["datapath"]
            from helicon.lib.average_power_spectra import average_power_spectra

            ret = average_power_spectra(
                file_path,
                apix=0.0,
                groupby=[],
                cutoff_res=[0.0, 0.0],
                min_particles=-1,
                force_phase_diff=False,
                batch_size=input.hill_avg_ps_from_meta_batch_size(),
                cpu=input.hill_avg_ps_from_meta_num_cpus(),
                diameter_mask=0,
                align=0,
                fft_x=input.hill_pnx(),
                fft_y=input.hill_pny(),
            )
            if ret is None:
                return
            result, used_apix, used_cutoff_res = ret
            result = pd.DataFrame(result).T
            avg_ps_pd_df.set(result)
            ui.update_numeric("hill_apix", value=float(round(used_apix, 4)))
            ui.update_numeric(
                "hill_cutoff_res_x", value=float(round(used_cutoff_res[1], 4))
            )
            ui.update_numeric(
                "hill_cutoff_res_y", value=float(round(used_cutoff_res[0], 4))
            )
            prev_data_from.set("avg_ps_pd")

    @render.data_frame
    @reactive.event(avg_ps_pd_df)
    def hill_display_avg_ps_df():
        req(len(avg_ps_pd_df()) > 0)
        return render.DataGrid(
            avg_ps_pd_df()[["group name", "#images"]],
            selection_mode="row",
            filters=True,
            height="30vh",
            width="100%",
        )

    @reactive.effect(priority=1000)
    @reactive.event(input.hill_display_avg_ps_df_cell_selection)
    def _set_prev_data_from_avg_ps_pd():
        prev_data_from.set("avg_ps_pd")

    @reactive.effect(priority=100)
    @reactive.event(input.hill_display_avg_ps_df_cell_selection)
    def _display_avg_ps_df_selected_plots_df_change():
        req(prev_data_from() == "avg_ps_pd")
        req(len(avg_ps_pd_df()) > 0)
        req(input.hill_apix() > 0)
        df_selected_idx = list(input.hill_display_avg_ps_df_cell_selection()["rows"])
        try:
            ps_data.set(
                hill.resize_rescale_power_spectra(
                    np.array(avg_ps_pd_df().iloc[df_selected_idx[0]]["ps_avg"]),
                    nyquist_res=input.hill_apix() * 2,
                    cutoff_res=(
                        input.hill_cutoff_res_y(),
                        input.hill_cutoff_res_x(),
                    ),
                    output_size=(input.hill_pny(), input.hill_pnx()),
                    **_ps_scaling(),
                    norm=1,
                )
            )
            pd_data.set(
                hill.resize_rescale_power_spectra(
                    np.array(avg_ps_pd_df().iloc[df_selected_idx[0]]["pd_avg"]),
                    nyquist_res=input.hill_apix() * 2,
                    cutoff_res=(
                        input.hill_cutoff_res_y(),
                        input.hill_cutoff_res_x(),
                    ),
                    output_size=(input.hill_pny(), input.hill_pnx()),
                    log=0,
                    low_pass_fraction=0,
                    high_pass_fraction=0,
                    norm=0,
                )
            )
        except Exception:
            logger.warning("No row selected in avg_ps_pd_df")

    @reactive.effect(priority=100)
    @reactive.event(
        input.hill_cutoff_res_y,
        input.hill_cutoff_res_x,
        input.hill_pny,
        input.hill_pnx,
        input.hill_ps_scaling,
        input.hill_log_amp,
        input.hill_lp_fraction,
        input.hill_hp_fraction,
    )
    def _display_avg_ps_df_selected_plots_plotting_params_change():
        req(prev_data_from() == "avg_ps_pd")
        req(len(avg_ps_pd_df()) > 0)
        req(input.hill_apix() > 0)
        df_selected_idx = [0]
        try:
            ps_data.set(
                hill.resize_rescale_power_spectra(
                    np.array(avg_ps_pd_df().iloc[df_selected_idx[0]]["ps_avg"]),
                    nyquist_res=input.hill_apix() * 2,
                    cutoff_res=(
                        input.hill_cutoff_res_y(),
                        input.hill_cutoff_res_x(),
                    ),
                    output_size=(input.hill_pny(), input.hill_pnx()),
                    **_ps_scaling(),
                    norm=1,
                )
            )
            pd_data.set(
                hill.resize_rescale_power_spectra(
                    np.array(avg_ps_pd_df().iloc[df_selected_idx[0]]["pd_avg"]),
                    nyquist_res=input.hill_apix() * 2,
                    cutoff_res=(
                        input.hill_cutoff_res_y(),
                        input.hill_cutoff_res_x(),
                    ),
                    output_size=(input.hill_pny(), input.hill_pnx()),
                    log=0,
                    low_pass_fraction=0,
                    high_pass_fraction=0,
                    norm=0,
                )
            )
        except Exception:
            logger.warning("Error displaying average power spectra")

    # ── Utility: match types for bookmark restoration ────────────

    # ── Cleanup on session end ──────────────────────────────────
    @reactive.effect
    def _on_session_end():
        pass


# ── Helpers (module-level to avoid pickling issues) ──────────────


def _init_layer_lines(fig_ellipses, figs_image):
    """Initialize layer line renderers with default values."""
    pass  # Not strictly needed; layer lines are built reactively


# A Bokeh slider of the helix whose number can be typed in place, as on the
# Shiny sliders of helicon.shiny.slider: double-clicking the slider's title or
# number opens an editor on the number; Enter applies it (the range of the slider
# follows), Escape or clicking away leaves the slider as it was. A typed number
# is applied as the end of a drag is (``value`` then ``value_throttled``), so the
# layer lines preview and the server is told once, by the slider's own callbacks.
_EDITABLE_SLIDER = "helicon-editable-bokeh-slider"
_MODEL_CLASS = "helicon-bokeh-model-"


def _editable_slider_classes(slider):
    """The CSS classes that make a Bokeh slider editable by
    :data:`_BOKEH_SLIDER_EDIT_JS`: one marks it, the other names its model."""
    return [_EDITABLE_SLIDER, f"{_MODEL_CLASS}{slider.id}"]


# Bokeh draws its widgets in shadow DOM, out of reach of the page's selectors: a
# double-click is found on its composed path, and the slider's model by the id
# its host element carries.
_BOKEH_SLIDER_EDIT_JS = """
(function () {
  if (window.__heliconBokehSliderEdit) { return; }
  window.__heliconBokehSliderEdit = true;
  var MARK = '%s', MODEL = '%s';

  function modelOf(host) {
    var id = null;
    host.classList.forEach(function (c) {
      if (c.indexOf(MODEL) === 0) { id = c.slice(MODEL.length); }
    });
    var docs = (window.Bokeh && window.Bokeh.documents) || [];
    for (var i = docs.length - 1; id && i >= 0; i--) {
      var model = docs[i].get_model_by_id(id);
      if (model) { return model; }
    }
    return null;
  }

  function digits(step) {
    var t = String(step);
    if (t.indexOf('e-') >= 0) { return Math.min(6, parseInt(t.split('e-')[1], 10)); }
    var dot = t.indexOf('.');
    return dot < 0 ? 0 : Math.min(6, t.length - dot - 1);
  }

  document.addEventListener('dblclick', function (event) {
    var label = null, host = null;
    var path = event.composedPath();
    for (var i = 0; i < path.length; i++) {
      var el = path[i];
      if (!el.classList) { continue; }
      if (!label && (el.classList.contains('bk-slider-value') ||
                     el.classList.contains('bk-slider-title'))) { label = el; }
      if (el.classList.contains(MARK)) { host = el; break; }
    }
    if (!label || !host) { return; }
    var slider = modelOf(host);
    if (!slider) { return; }
    event.preventDefault();
    var shown = label.classList.contains('bk-slider-value') ? label :
      (label.querySelector('.bk-slider-value') || label);
    var rect = shown.getBoundingClientRect();
    var current = Number(Number(slider.value).toFixed(Math.max(digits(slider.step), 2)));
    var box = document.createElement('input');
    box.type = 'text';
    box.inputMode = 'decimal';
    box.value = String(current);
    Object.assign(box.style, {
      position: 'fixed', left: (rect.left - 8) + 'px', top: (rect.top - 3) + 'px',
      width: Math.max(rect.width + 16, 72) + 'px', height: (rect.height + 6) + 'px',
      zIndex: 10000, fontSize: '14px', textAlign: 'center',
      border: '1px solid #0d6efd', borderRadius: '3px', padding: '0 2px'
    });
    document.body.appendChild(box);
    box.focus();
    box.select();
    // removing the box blurs it, which calls this again
    var closed = false;
    var done = function () {
      if (closed) { return; }
      closed = true;
      box.remove();
    };
    box.addEventListener('keydown', function (ev) {
      if (ev.key === 'Enter') {
        var v = parseFloat(String(box.value).replace(/[ ,]/g, ''));
        done();
        if (!isFinite(v)) { return; }
        slider.setv({value: v, value_throttled: v});
      } else if (ev.key === 'Escape') {
        done();
      }
    });
    box.addEventListener('blur', done);
  }, true);
})();
""" % (
    _EDITABLE_SLIDER,
    _MODEL_CLASS,
)


# Shared by browser previews, final commits, and server restoration.  A change is
# identified by its originating control, so delayed callbacks from dependent
# assignments cannot be mistaken for a newer user edit.
_HELIX_CONTROL_JS = """
const same = (a, b) => Math.abs(a - b) <= 1e-12 * Math.max(1, Math.abs(a), Math.abs(b));
const same_values = (a, b) => a != null && b != null &&
    a.length === b.length && a.every((value, i) => same(value, b[i]));

function ensure_state() {
    if (controls._hill_committed == null) {
        const d = controls.data;
        controls._hill_committed = [d.twist[0], d.pitch[0], d.rise[0]];
    }
    if (controls._hill_preview == null)
        controls._hill_preview = controls._hill_committed.slice();
    if (controls._hill_revision == null) {
        const revision = controls.data.revision;
        controls._hill_revision = revision == null ? 0 : revision[0];
    }
}

function keep_mode() {
    const host = window.Shiny ? window : window.parent;
    const radio = Array.from(host.document.getElementsByName(input_ids.use_twist_pitch))
        .find(el => el.checked);
    return radio ? radio.value : controls.data.keep[0];
}

function update_layer_lines(twist, pitch, rise) {
    const tilt = controls.data.tilt[0] * Math.PI / 180;
    for (const el of fig_ellipses) {
        const m = el.tags[0];
        const ns = el.tags[1];
        const data = el.data_source.data;
        for (let i = 0; i < ns.length; i++)
            data.y[i] = (m / rise + ns[i] / pitch) / Math.cos(tilt);
        el.data_source.change.emit();
    }
}

function apply_values(values) {
    const [twist, pitch, rise] = values;
    slider_twist.start = Math.min(-180, twist);
    slider_twist.end = Math.max(180, twist);
    if (pitch < slider_pitch.start || pitch > slider_pitch.end) {
        slider_pitch.start = pitch / 2;
        slider_pitch.end = pitch * 2;
        slider_pitch.step = Math.max(slider_pitch.end * 0.001, Number.EPSILON);
    }
    if (rise < slider_rise.start || rise > slider_rise.end) {
        slider_rise.start = rise / 2;
        slider_rise.end = rise * 2;
        slider_rise.step = Math.max(slider_rise.end * 0.001, Number.EPSILON);
    }
    controls._hill_preview = values.slice();
    if (!same(slider_twist.value, twist)) slider_twist.value = twist;
    if (!same(slider_pitch.value, pitch)) slider_pitch.value = pitch;
    if (!same(slider_rise.value, rise)) slider_rise.value = rise;
    update_layer_lines(twist, pitch, rise);
}

function preview_value(origin, value) {
    ensure_state();
    if (same(controls._hill_preview[origin], value)) return true;
    let [twist, pitch, rise] = controls._hill_preview;
    if (origin === 0) {
        twist = value;
        pitch = twist === 0 ? rise : Math.abs(360 * rise / twist);
    } else if (origin === 1) {
        pitch = value;
        const sign = twist < 0 ? -1 : 1;
        twist = sign * 360 * rise / pitch;
    } else {
        rise = value;
        if (keep_mode() === "Pitch") {
            const sign = twist < 0 ? -1 : 1;
            twist = sign * 360 * rise / pitch;
        } else {
            pitch = twist === 0 ? rise : Math.abs(360 * rise / twist);
        }
    }
    const values = [twist, pitch, rise];
    if (!values.every(Number.isFinite) || pitch <= 0 || rise <= 0) {
        apply_values(controls._hill_preview);
        return false;
    }
    apply_values(values);
    controls._hill_dirty = !same_values(values, controls._hill_committed);
    return true;
}
"""


def _setup_helix_control_js(
    slider_twist,
    slider_pitch,
    slider_rise,
    fig_ellipses,
    input_ids,
    controls,
):
    """Preview locally on value changes and commit once interaction finishes."""
    from bokeh.models import CustomJS

    args = dict(
        slider_twist=slider_twist,
        slider_pitch=slider_pitch,
        slider_rise=slider_rise,
        controls=controls,
        input_ids=input_ids,
        fig_ellipses=list(fig_ellipses),
    )
    restore_callback = CustomJS(
        args=args,
        code=_HELIX_CONTROL_JS
        + """
        ensure_state();
        const d = controls.data;
        const revision = d.revision == null ? 0 : d.revision[0];
        // A server response for an older commit must not overwrite an active
        // browser preview or a newer commit.
        if (controls._hill_dirty || revision < controls._hill_revision) return;
        const values = [d.twist[0], d.pitch[0], d.rise[0]];
        if (!values.every(Number.isFinite) || values[1] <= 0 || values[2] <= 0) return;
        controls._hill_revision = revision;
        controls._hill_committed = values.slice();
        controls._hill_dirty = false;
        apply_values(values);
    """,
    )
    controls.js_on_change(
        "data",
        restore_callback,
    )
    callbacks = [restore_callback]
    for origin, control in enumerate((slider_twist, slider_pitch, slider_rise)):
        preview_callback = CustomJS(
            args={**args, "origin": origin, "control": control},
            code=_HELIX_CONTROL_JS
            + """
            preview_value(origin, control.value);
        """,
        )
        commit_callback = CustomJS(
            args={**args, "origin": origin, "control": control},
            code=_HELIX_CONTROL_JS
            + """
            // Compilation or event scheduling can occasionally let the final
            // callback run before the last preview callback.
            if (!preview_value(origin, control.value)) return;
            ensure_state();
            const values = controls._hill_preview;
            if (!controls._hill_dirty || same_values(values, controls._hill_committed)) {
                controls._hill_dirty = false;
                return;
            }
            controls._hill_revision += 1;
            controls._hill_committed = values.slice();
            controls._hill_dirty = false;
            const host = window.Shiny ? window : window.parent;
            // Default priority coalesces this complete tuple into one Shiny batch.
            host.Shiny.setInputValue(input_ids.twist, values[0]);
            host.Shiny.setInputValue(input_ids.pitch, values[1]);
            host.Shiny.setInputValue(input_ids.rise, values[2]);
            host.Shiny.setInputValue(input_ids.helix_revision, controls._hill_revision);
        """,
        )
        control.js_on_change("value", preview_callback)
        control.js_on_change("value_throttled", commit_callback)
        callbacks.extend((preview_callback, commit_callback))
    return callbacks


def _setup_csym_control_js(spinner, controls, input_id):
    """Send the csym box's value to the server, and take values the server sends.

    Parameters
    ----------
    spinner : bokeh.models.Spinner
        The csym box.
    controls : bokeh.models.ColumnDataSource
        ``value`` and ``revision``, written by the server.
    input_id : str
        The (namespaced) Shiny input that holds the csym.

    Returns
    -------
    list of bokeh.models.CustomJS
    """
    from bokeh.models import CustomJS

    commit = CustomJS(
        args=dict(spinner=spinner, controls=controls, input_id=input_id),
        code="""
        const value = Number(spinner.value);
        if (!Number.isInteger(value) || value < 1) {
            // an emptied or half-typed box is not a csym: back to the last one
            const last = controls.data.value[0];
            if (Number.isInteger(last) && last >= 1) spinner.value = last;
            return;
        }
        const host = window.Shiny ? window : window.parent;
        host.Shiny.setInputValue(input_id, value);
    """,
    )
    restore = CustomJS(
        args=dict(spinner=spinner, controls=controls),
        code="""
        const value = controls.data.value[0];
        if (Number.isInteger(value) && value >= 1 && spinner.value !== value)
            spinner.value = value;
    """,
    )
    spinner.js_on_change("value", commit)
    controls.js_on_change("data", restore)
    return [commit, restore]


def _update_slider_renderers(callbacks, renderers):
    """Replace references without registering more callbacks or mutating old lists."""
    for callback in callbacks:
        callback.args = {**callback.args, "fig_ellipses": list(renderers)}
