"""HILL tab — helical indexing via Fourier layer lines.

Ported from HILL.git into the Shiny module pattern.
Uses Bokeh widgets for interactive power spectrum / layer line analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import splev
from scipy.signal import find_peaks

import helicon
from shiny import reactive, render, ui, module, req
from shinywidgets import output_widget, render_widget

from bokeh.models import CustomJS
from bokeh.resources import Resources

from ..lib.shared_state import ProjectState


def bokeh_dependency():
    """Return HTMLDependency for Bokeh with widget bundle via CDN."""
    resources = Resources(
        mode="cdn", components=["bokeh", "bokeh-widgets", "bokeh-tables"]
    )
    return ui.head_content(ui.HTML(resources.render()))


try:
    from ..lib import hill_compute as hill
except Exception:
    hill = None

# helicon.get_images_from_url does not exist; use the shared webApps lib.
from ..lib import denovo3d_pipeline

logger = logging.getLogger(__name__)

# ── Module-level constants ────────────────────────────────────────

MODULE_PREFIX = "hill"

BOOKMARK_DEFAULTS = {
    "input_mode": ("hill_input_mode", "url"),
    "twist": ("hill_twist", 29.40),
    "rise": ("hill_rise", 21.92),
    "csym": ("hill_csym", 6),
    "diameter": ("hill_diameter", 155.2),
    "apix": ("hill_apix", 2.3438),
    "cutoff_res_x": ("hill_cutoff_res_x", 7.03),
    "cutoff_res_y": ("hill_cutoff_res_y", 4.69),
    "log_amp": ("hill_log_amp", True),
    "pnx": ("hill_pnx", 512),
    "pny": ("hill_pny", 1024),
    "hp_fraction": ("hill_hp_fraction", 0.40),
    "lp_fraction": ("hill_lp_fraction", 0.00),
    "m_max": ("hill_m_max", 3),
    "const_image_color": ("hill_const_image_color", ""),
    "ll_colors": ("hill_ll_colors", "lime cyan violet salmon silver"),
    "fft_top_only": ("hill_fft_top_only", False),
    "use_twist_pitch": ("hill_use_twist_pitch", "Twist"),
    "out_of_plane_tilt": ("hill_out_of_plane_tilt", 0.0),
    "input_type": ("hill_input_type", "Image"),
    "angle": ("hill_angle", 0.0),
    "dx": ("hill_dx", 0.0),
    "dy": ("hill_dy", 0.0),
}

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

# Default 2D image stack pre-filled in the URL input so the field is not
# blank on launch.
_DEFAULT_URL = (
    "https://tinyurl.com/y5tq9fqa"
)


# ── UI ────────────────────────────────────────────────────────────


@module.ui
def hill_tab_ui():
    return ui.page_fillable(
        bokeh_dependency(),
        ui.tags.style("""
        .hill-scrollable-sidebar {
            height: 100%; max-height: 100%; overflow-y: auto;
            border: 1px solid #ccc; padding: 10px;
        }
        .hill-wrap-text { word-wrap: break-word; overflow-wrap: break-word; white-space: normal; }
        .hill-inline-box label { display: table-cell; text-align: left; vertical-align: middle; }
        .hill-inline-box input { width: 4em; margin-left: 1em; }
        .hill-inline-box .form-group { display: table-row; }
        #hill-hill_main_plots { overflow-y: auto; }
        """),
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
                                open=False,
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
                                    {"1": "upload", "2": "url", "3": "emd-xxxxx"},
                                    selected="2",
                                    inline=True,
                                ),
                                value="hill_input_mode",
                                open=True,
                            )
                        ),
                        ui.output_ui("hill_conditional_input_uis"),
                        ui.output_ui("hill_conditional_3d_uis"),
                        ui.div(
                            ui.output_ui("hill_select_image_gallery"),
                            style="max-height: 500px; overflow-y: auto;",
                        ),
                        output_widget("hill_display_selected_image", height="auto"),
                        ui.accordion(
                            ui.accordion_panel(
                                "Image Parameters",
                                ui.output_ui("hill_select_from_multiselect_ui"),
                                ui.input_radio_buttons(
                                    "hill_input_type",
                                    "Input is:",
                                    choices=["Image", "PS", "PD"],
                                    inline=True,
                                ),
                                ui.output_ui("hill_img_2d_uis"),
                                ui.output_ui("hill_img_update_buttons"),
                                value="hill_image_params",
                                open=False,
                            )
                        ),
                        output_widget("hill_display_transformed", height="auto"),
                        output_widget("hill_plot_radial_profile", height="auto"),
                        output_widget("hill_plot_acf", width="auto"),
                        ui.output_ui("hill_get_avg_ps_ui"),
                        ui.HTML(
                            "<i><p>Developed by the <a href='https://jianglab.science.psu.edu/helicon' target='_blank'>Jiang Lab</a>. "
                            "Report issues to <a href='https://github.com/jianglab/helicon/issues' target='_blank'>helicon@GitHub</a>.</p></i>"
                        ),
                        value="hill_input_sidebar",
                    ),
                    ui.nav_panel(
                        "Parameters",
                        ui.layout_columns(
                            ui.input_radio_buttons(
                                "hill_use_twist_pitch",
                                "Keep twist/pitch when changing rise:",
                                choices=["Twist", "Pitch"],
                                selected="Twist",
                                inline=True,
                            ),
                            ui.input_numeric(
                                "hill_csym",
                                "csym",
                                value=6,
                                min=1,
                                step=1,
                                update_on="blur",
                            ),
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
                            ui.input_checkbox(
                                "hill_fft_top_only",
                                "Only display top half of FFT",
                                value=False,
                            ),
                            ui.input_checkbox(
                                "hill_log_amp", "Log (amplitude)", value=True
                            ),
                            ui.input_text(
                                "hill_const_image_color",
                                "Flatten PS/PD image in this color",
                                value="",
                                placeholder="white black",
                                update_on="blur",
                            ),
                            ui.input_text(
                                "hill_ll_colors",
                                "Layerline colors",
                                value="lime cyan violet salmon silver",
                                update_on="blur",
                            ),
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
                        ui.input_checkbox(
                            "hill_inhibit_update",
                            "Inhibit automatic update",
                            value=False,
                        ),
                        ui.hr(),
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
                        ui.hr(),
                        ui.h5("Display:"),
                        ui.layout_columns(
                            ui.input_checkbox("hill_PS", "Power spectra", value=True),
                            ui.input_checkbox("hill_YP", "Y Profile", value=True),
                            ui.input_checkbox("hill_Phase", "Phase hover", value=False),
                            ui.input_checkbox(
                                "hill_PD", "Phase difference", value=True
                            ),
                            ui.input_checkbox("hill_Color", "Color", value=True),
                            ui.input_checkbox("hill_LL", "Layer line", value=True),
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
def hill_tab_server(input, output, session, project: ProjectState):
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
    prev_data_from = reactive.value("main")
    avg_ps_pd_df = reactive.value({})
    avg_ps_pd_main_in_df = reactive.value({})
    straightened_filaments_for_avg = reactive.value([])

    # Error display state
    error_msg = reactive.value("")

    init_done = reactive.value(False)

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
        Spinner,
        Slider,
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

    # ── Bokeh spinners and sliders for helix params ────────────
    curr_twist = _INIT_TWIST
    curr_rise = _INIT_RISE
    curr_pitch = _INIT_PITCH

    spinner_twist = Spinner(
        title="Twist (°)",
        low=-360.0,
        high=360.0,
        step=1.0,
        value=curr_twist,
        format="0.00",
        sizing_mode="stretch_width",
    )
    spinner_pitch = Spinner(
        title="Pitch (Å)",
        low=1.0,
        step=1.0,
        value=curr_pitch,
        format="0.00",
        sizing_mode="stretch_width",
    )
    spinner_rise = Spinner(
        title="Rise (Å)",
        low=1.0,
        step=1.0,
        value=curr_rise,
        format="0.00",
        sizing_mode="stretch_width",
    )

    slider_twist = Slider(
        start=-180.0,
        end=180.0,
        value=curr_twist,
        step=0.01,
        title="Twist (°)",
        sizing_mode="stretch_width",
    )
    slider_pitch = Slider(
        start=curr_pitch / 2,
        end=curr_pitch * 2.0,
        value=curr_pitch,
        step=curr_pitch * 0.002,
        title="Pitch (Å)",
        sizing_mode="stretch_width",
    )
    slider_rise = Slider(
        start=curr_rise / 2,
        end=min(curr_pitch, curr_rise * 2.0),
        value=curr_rise,
        step=min(curr_pitch, curr_rise * 2.0) * 0.001,
        title="Rise (Å)",
        sizing_mode="stretch_width",
    )

    # JS callbacks for spinners → Shiny inputs
    _setup_spinner_js(
        spinner_twist,
        spinner_pitch,
        spinner_rise,
        slider_twist,
        slider_pitch,
        slider_rise,
        MODULE_PREFIX,
    )
    _setup_slider_js(
        slider_twist,
        slider_pitch,
        slider_rise,
        spinner_twist,
        spinner_pitch,
        spinner_rise,
        fig_ellipses,
        MODULE_PREFIX,
    )
    _setup_spinner_throttle_js(spinner_rise, slider_rise, spinner_pitch, slider_pitch)

    # ── Main plot layout ────────────────────────────────────────
    figs_row = gridplot(
        children=[figs], toolbar_location="right", sizing_mode="stretch_width"
    )
    figs_grid = layout(
        children=[
            row(
                spinner_twist, spinner_pitch, spinner_rise, sizing_mode="stretch_width"
            ),
            row(slider_twist, slider_pitch, slider_rise, sizing_mode="stretch_width"),
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
    fig_selected_image, source_selected_image = hill.create_image_figure(
        init_img,
        _INIT_APIX,
        _INIT_APIX,
        title=f"Selected Image ({_INIT_NX}x{_INIT_NY})",
        title_location="below",
        show_axis=False,
        show_toolbar=False,
        crosshair_color="white",
    )

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

    @reactive.effect(priority=1000)
    def _init_from_query_once():
        if init_done():
            return
        init_done.set(True)
        qs = session.clientdata.url_search()
        from urllib.parse import parse_qs

        qp = {
            k: v[0] if len(v) == 1 else v for k, v in parse_qs(qs.lstrip("?")).items()
        }

        url_twist = qp.get("twist")
        url_rise = qp.get("rise")
        if url_twist and url_rise:
            try:
                ui.update_numeric("hill_twist", value=float(url_twist))
                ui.update_numeric("hill_rise", value=float(url_rise))
            except ValueError:
                pass
        url_csym = qp.get("csym")
        if url_csym:
            try:
                ui.update_numeric("hill_csym", value=int(url_csym))
            except ValueError:
                pass
        url_diameter = qp.get("filament_diameter")
        if url_diameter:
            try:
                ui.update_numeric("hill_diameter", value=float(url_diameter))
            except ValueError:
                pass
        url_input_mode = qp.get("input_mode")
        if url_input_mode:
            try:
                ui.update_radio_buttons(
                    "hill_input_mode_params", selected=url_input_mode
                )
            except Exception:
                pass
        url_input_type = qp.get("input_type")
        if url_input_type:
            try:
                ui.update_radio_buttons("hill_input_type", selected=url_input_type)
            except Exception:
                pass
        url_apix = qp.get("apix")
        if url_apix:
            try:
                apix_from_file.set(float(url_apix))
            except ValueError:
                pass
        url_rot = qp.get("rotate")
        if url_rot:
            try:
                ui.update_numeric("hill_angle", value=float(url_rot))
            except ValueError:
                pass
        url_dx = qp.get("dx")
        if url_dx:
            try:
                ui.update_numeric("hill_dx", value=float(url_dx))
            except ValueError:
                pass
        url_log = qp.get("log_amp")
        if url_log:
            try:
                ui.update_checkbox("hill_log_amp", value=bool(int(url_log)))
            except ValueError:
                pass
        # Load a file path provided via the display button query string.
        url_img = qp.get("img_file_url")
        if url_img:
            ui.update_text("hill_img_file_url", value=url_img)
            try:
                from ..lib import denovo3d_pipeline

                data, apix = denovo3d_pipeline.get_images_from_url(url_img)
                input_data.set(data)
                apix_from_file.set(apix)
                ui.update_numeric("hill_apix", value=apix)
                _update_image_dims_from_input_data()
            except Exception as e:
                import logging

                logging.getLogger(__name__).error(
                    "Failed to load image from query path %s: %s",
                    url_img,
                    e,
                    exc_info=True,
                )
                ui.modal_show(
                    ui.modal(
                        f"Failed to load: {e}",
                        title="Error",
                        easy_close=True,
                        footer=None,
                    )
                )
        # Replace the default URL with a file path passed from the display
        # button via the query string, then send it as a server-side input
        # message so _load_input_data_from_url picks it up (ignore_init
        # suppresses the initial ui.update_text value).
        url_img = qp.get("img_file_url")
        if url_img:
            ui.update_text("hill_img_file_url", value=url_img)
            # Load the file immediately instead of waiting for the reactive
            # event (which has ignore_init=True and would skip this).
            try:
                data, apix = denovo3d_pipeline.get_images_from_url(url_img)
                input_data.set(data)
                apix_from_file.set(apix)
                ui.update_numeric("hill_apix", value=apix)
                _update_image_dims_from_input_data()
            except Exception as e:
                logger.error(
                    "Failed to load image from query path %s: %s",
                    url_img,
                    e,
                    exc_info=True,
                )
                ui.modal_show(
                    ui.modal(
                        f"Failed to load: {e}",
                        title="Error",
                        easy_close=True,
                        footer=None,
                    )
                )

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
        # Subtract space for spinner row (~40px), slider row (~40px),
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
        ui.update_checkbox_group("hill_ms", choices=ms, selected=prev)

    # ── Layer line / display param changes ──────────────────────
    @reactive.effect
    @reactive.event(
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
        ignore_init=True,
    )
    def _update_layerline_figures():
        req(len(selected_images()) > 0)
        # Frame sizing is managed by _resize_main_plots — don't override it here.
        for e in fig_ellipses:
            for f in figs_image:
                if e in f.renderers:
                    f.renderers.remove(e)
        fig_ellipses.clear()
        if input.hill_LL():
            curr_m_groups = hill.compute_layer_line_positions(
                twist=input.hill_twist(),
                rise=input.hill_rise(),
                csym=input.hill_csym(),
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
                        fig_ellipses.append(gl)

        # Update ranges
        ny = input.hill_pny()
        nx = input.hill_pnx()
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

        data_source_ps.data = {
            **data_source_ps.data,
            "x": [-nx // 2 * dsx],
            "y": [-ny // 2 * dsy],
            "dw": [nx * dsx],
            "dh": [ny * dsy],
            "bessel": [bessel],
        }
        data_source_pd.data = {
            **data_source_pd.data,
            "x": [-nx // 2 * dsx],
            "y": [-ny // 2 * dsy],
            "dw": [nx * dsx],
            "dh": [ny * dsy],
            "bessel": [bessel],
        }

        # Re-setup slider JS with updated fig_ellipses references
        _setup_slider_js(
            slider_twist,
            slider_pitch,
            slider_rise,
            spinner_twist,
            spinner_pitch,
            spinner_rise,
            fig_ellipses,
            MODULE_PREFIX,
        )

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
                        open=False,
                    )
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
            ui.input_numeric(
                "hill_csym_ahs", "Csym:", value=input.hill_csym(), min=1, step=1
            ),
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
                spinner_rise.value = rise
                spinner_twist.value = twist
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

    # ── Image loading ────────────────────────────────────────────

    @reactive.effect
    @reactive.event(input.hill_input_mode_params, input.hill_img_file_url)
    def _load_input_data_from_url():
        req(input.hill_input_mode_params() == "2")
        url = input.hill_img_file_url()
        if not url:
            return
        # Reset helical params to URL-mode defaults (EMDB mode may have changed them).
        ui.update_numeric("hill_twist", value=_INIT_TWIST)
        ui.update_numeric("hill_rise", value=_INIT_RISE)
        ui.update_numeric("hill_csym", value=BOOKMARK_DEFAULTS["csym"][1])
        spinner_twist.value = _INIT_TWIST
        spinner_rise.value = _INIT_RISE
        try:
            data, apix = denovo3d_pipeline.get_images_from_url(url)
            input_data.set(data)
            apix_from_file.set(apix)
            ui.update_numeric("hill_apix", value=apix)
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
            ui.update_numeric("hill_apix", value=apix)
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
            ui.update_numeric("hill_apix", value=apix)
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
            ui.update_numeric("hill_angle", value=0.0)
            ui.update_numeric("hill_dx", value=0.0)
            ui.update_numeric("hill_dy", value=0.0)
            ui.update_numeric("hill_mask_radius", value=0.0)
            ui.update_numeric("hill_mask_len", value=100.0)

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
                    angle_auto, dy_auto, diameter_auto = (
                        helicon.estimate_helix_rotation_center_diameter(
                            img0, estimate_center=True, estimate_rotation=True
                        )
                    )
                    angle = round(angle_auto + 90, 2)
                    dx_val = 0.0
                    dy_val = round(dy_auto * input.hill_apix(), 2)
                    mr = round(diameter_auto / 2 * input.hill_apix(), 2)
                    ml = 90.0
                    ui.update_numeric("hill_angle", value=angle)
                    ui.update_numeric("hill_dx", value=dx_val)
                    ui.update_numeric("hill_dy", value=dy_val)
                    ui.update_numeric("hill_mask_radius", value=mr)
                    ui.update_numeric("hill_mask_len", value=ml)
                    if angle or dx_val or dy_val or input.hill_negate():
                        img0 = hill.transform_2d_image(
                            img0,
                            angle,
                            dx_val,
                            dy_val,
                            input.hill_negate(),
                            input.hill_apix(),
                        )
                    if mr > 0 and ml > 0 and ml <= 100:
                        img0 = hill.mask_2d_filament(
                            img0,
                            mr,
                            input.hill_apix(),
                            ml / 100.0,
                        )
                except Exception:
                    pass
            data_2d_transformed.set(img0)
        else:
            selected_images.set([])
            selected_image_labels.set([])

    @reactive.effect
    @reactive.event(input.hill_select_image)
    def _update_selected_images():
        sel = input.hill_select_image()
        if sel is None or len(sel) == 0:
            selected_images.set([])
            selected_image_labels.set([])
            return
        indices = list(sel)
        all_imgs = data_all_2d()
        all_labels = data_all_2d_labels()
        chosen = [all_imgs[i] for i in indices if i < len(all_imgs)]
        chosen_labels = [all_labels[i] for i in indices if i < len(all_labels)]
        selected_images.set(chosen)
        selected_image_labels.set(chosen_labels)

        prev_data_from.set("main")

        mode = input.hill_input_type()
        if mode in ("PS", "PD") or input.hill_is_3d():
            if not input.hill_inhibit_update():
                ui.update_numeric("hill_angle", value=0.0)
                ui.update_numeric("hill_dx", value=0.0)
                ui.update_numeric("hill_dy", value=0.0)
            # Pulse trigger for _apply_2d_transform_ps_pd
            # _apply_2d_transform_ps_pd handles this via selected_images trigger
        else:
            if input.hill_inhibit_update():
                ui.update_numeric("hill_mask_len", value=90.0)
                ui.update_checkbox("hill_inhibit_update", value=False)
            else:
                valid_indices = [i for i in indices if i < len(all_imgs)]
                if valid_indices:
                    sel_idx = valid_indices[0]
                    if input.hill_input_type() == "Image":
                        if sel_idx not in avg_ps_pd_main_in_df():
                            angle_auto, dy_auto, diameter_auto = (
                                helicon.estimate_helix_rotation_center_diameter(
                                    chosen[0],
                                    estimate_center=True,
                                    estimate_rotation=True,
                                )
                            )
                            angle = round(angle_auto + 90, 2)
                            dx_val = 0.0
                            dy_val = round(dy_auto * input.hill_apix(), 2)
                            mr = round(diameter_auto / 2 * input.hill_apix(), 2)
                            ml = 90.0
                            ui.update_numeric("hill_angle", value=angle)

                            ui.update_numeric("hill_dx", value=dx_val)

                            ui.update_numeric("hill_dy", value=dy_val)

                            ui.update_numeric("hill_mask_radius", value=mr)

                            ui.update_numeric("hill_mask_len", value=ml)

                        else:
                            saved = avg_ps_pd_main_in_df()[sel_idx]
                            angle = saved["angle"]
                            dx_val = saved["dx"]
                            dy_val = saved["dy"]
                            mr = saved["mask_radius"]
                            ml = saved["mask_len"]
                            ui.update_numeric("hill_angle", value=angle)

                            ui.update_numeric("hill_dx", value=dx_val)

                            ui.update_numeric("hill_dy", value=dy_val)

                            ui.update_numeric("hill_mask_radius", value=mr)

                            ui.update_numeric("hill_mask_len", value=ml)

                        # apply transform with local params
                        temp = chosen[0].astype(np.float64)
                        if angle or dx_val or dy_val or input.hill_negate():
                            temp = hill.transform_2d_image(
                                temp,
                                angle,
                                dx_val,
                                dy_val,
                                input.hill_negate(),
                                input.hill_apix(),
                            )
                        if mr > 0 and ml > 0 and ml <= 100:
                            temp = hill.mask_2d_filament(
                                temp,
                                mr,
                                input.hill_apix(),
                                ml / 100.0,
                            )
                        data_2d_transformed.set(temp)

            # save transformation parameters for all selected images
            valid_indices = [i for i in indices if i < len(all_imgs)]
            apix_val = input.hill_apix()
            for i, sel_idx in enumerate(valid_indices):
                if not input.hill_inhibit_update():
                    if sel_idx not in avg_ps_pd_main_in_df():
                        if i > 0:
                            img = all_imgs[sel_idx]
                            angle_auto, dy_auto, diameter_auto = (
                                helicon.estimate_helix_rotation_center_diameter(
                                    img, estimate_center=True, estimate_rotation=True
                                )
                            )
                        else:
                            pass  # already estimated above
                        d = avg_ps_pd_main_in_df()
                        d[sel_idx] = {
                            "apix": apix_val,
                            "negate": input.hill_negate(),
                            "angle": round(angle_auto + 90, 2),
                            "dx": 0.0,
                            "dy": round(dy_auto * apix_val, 2),
                            "mask_radius": round(diameter_auto / 2 * apix_val, 2),
                            "mask_len": 90.0,
                        }
                        avg_ps_pd_main_in_df.set(d)

    # ── Clear saved params when data changes ────────────────────
    @reactive.effect
    @reactive.event(data_all_2d)
    def _clear_avg_ps_pd_main_in_df():
        avg_ps_pd_main_in_df.set({})

    # ── Multi-selection: restore saved params on dropdown change ─
    @reactive.effect
    @reactive.event(input.hill_curr_img_in_selected_list)
    def _update_current_image_from_multiselect():
        req(len(selected_images()) > 1)
        req(input.hill_apix() > 0)
        valid = list(input.hill_select_image())
        curr_idx = int(input.hill_curr_img_in_selected_list())
        if curr_idx >= len(valid):
            return
        record = avg_ps_pd_main_in_df().get(valid[curr_idx])
        if record is None:
            return
        ui.update_numeric("hill_apix", value=record["apix"])
        ui.update_numeric("hill_angle", value=record["angle"])
        ui.update_numeric("hill_dx", value=record["dx"])
        ui.update_numeric("hill_dy", value=record["dy"])
        ui.update_numeric("hill_mask_radius", value=record["mask_radius"])
        ui.update_numeric("hill_mask_len", value=record["mask_len"])
        ui.update_checkbox("hill_negate", value=bool(record.get("negate")))
        img = data_all_2d()[valid[curr_idx]]
        hill.update_image_figure(
            fig_selected_image,
            source_selected_image,
            img,
            input.hill_apix(),
            title="Original image",
        )

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
        curr_idx = int(input.hill_curr_img_in_selected_list())
        valid = list(input.hill_select_image())
        if curr_idx >= len(valid):
            return
        all_imgs = data_all_2d()
        temp = all_imgs[valid[curr_idx]].astype(np.float64)
        if (
            input.hill_angle()
            or input.hill_dx()
            or input.hill_dy()
            or input.hill_negate()
        ):
            temp = hill.transform_2d_image(
                temp,
                input.hill_angle(),
                input.hill_dx(),
                input.hill_dy(),
                input.hill_negate(),
                input.hill_apix(),
            )
        if (
            input.hill_mask_radius() > 0
            and input.hill_mask_len() > 0
            and input.hill_mask_len() <= 100
        ):
            temp = hill.mask_2d_filament(
                temp,
                input.hill_mask_radius(),
                input.hill_apix(),
                input.hill_mask_len() / 100.0,
            )
        hill.update_image_figure(
            fig_transformed_img,
            source_transformed_img,
            temp,
            input.hill_apix(),
            title="Transformed image",
        )

    # ── Average PS & PD from selected images ─────────────────────
    @reactive.effect
    @reactive.event(input.hill_run_get_average_power_spectra)
    def _average_ps_pd_from_multiselect():
        req(len(selected_images()) > 1)
        valid = list(input.hill_select_image())
        all_imgs = data_all_2d()
        avg_ps = np.zeros((input.hill_pny(), input.hill_pnx()))
        avg_pd = np.zeros((input.hill_pny(), input.hill_pnx()))
        for i, sel_idx in enumerate(valid):
            img = all_imgs[sel_idx]
            record = avg_ps_pd_main_in_df().get(sel_idx, {})
            temp = img.astype(np.float64)
            if (
                record.get("angle")
                or record.get("dx")
                or record.get("dy")
                or record.get("negate")
            ):
                temp = hill.transform_2d_image(
                    temp,
                    record.get("angle", 0),
                    record.get("dx", 0),
                    record.get("dy", 0),
                    record.get("negate", False),
                    record.get("apix", input.hill_apix()),
                )
            mr = record.get("mask_radius", 0)
            ml = record.get("mask_len", 0)
            if mr > 0 and ml > 0 and ml <= 100:
                temp = hill.mask_2d_filament(
                    temp,
                    mr,
                    record.get("apix", input.hill_apix()),
                    ml / 100.0,
                )
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
                log=input.hill_log_amp(),
                low_pass_fraction=input.hill_lp_fraction() / 100.0,
                high_pass_fraction=input.hill_hp_fraction() / 100.0,
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

    # ── Multi-selection: use current transformed image for PS/PD ─
    @reactive.effect
    @reactive.event(input.hill_use_curr_transformed)
    def _use_curr_transformed_from_multiselect():
        req(prev_data_from() == "main")
        req(len(selected_images()) > 1)
        req(input.hill_apix() > 0)
        valid = list(input.hill_select_image())
        curr_idx = int(input.hill_curr_img_in_selected_list())
        if curr_idx >= len(valid):
            return
        all_imgs = data_all_2d()
        temp = all_imgs[valid[curr_idx]].astype(np.float64)
        if (
            input.hill_angle()
            or input.hill_dx()
            or input.hill_dy()
            or input.hill_negate()
        ):
            temp = hill.transform_2d_image(
                temp,
                input.hill_angle(),
                input.hill_dx(),
                input.hill_dy(),
                input.hill_negate(),
                input.hill_apix(),
            )
        if (
            input.hill_mask_radius() > 0
            and input.hill_mask_len() > 0
            and input.hill_mask_len() <= 100
        ):
            temp = hill.mask_2d_filament(
                temp,
                input.hill_mask_radius(),
                input.hill_apix(),
                input.hill_mask_len() / 100.0,
            )
        itype = input.hill_input_type()
        if itype == "Image":
            pwr, phase = hill.compute_power_spectra(
                temp,
                apix=input.hill_apix(),
                cutoff_res=(input.hill_cutoff_res_y(), input.hill_cutoff_res_x()),
                output_size=(input.hill_pny(), input.hill_pnx()),
                log=input.hill_log_amp(),
                low_pass_fraction=input.hill_lp_fraction() / 100.0,
                high_pass_fraction=input.hill_hp_fraction() / 100.0,
            )
            phase_data.set(phase)
            ps_data.set(pwr)
            pd_data.set(hill.compute_phase_difference_across_meridian(phase))
        elif itype == "PS":
            ps_data.set(
                hill.resize_rescale_power_spectra(
                    temp,
                    nyquist_res=input.hill_apix() * 2,
                    cutoff_res=(input.hill_cutoff_res_y(), input.hill_cutoff_res_x()),
                    output_size=(input.hill_pny(), input.hill_pnx()),
                    log=input.hill_log_amp(),
                    low_pass_fraction=input.hill_lp_fraction() / 100.0,
                    high_pass_fraction=input.hill_hp_fraction() / 100.0,
                    norm=1,
                )
            )
        elif itype == "PD":
            pd_data.set(
                hill.resize_rescale_power_spectra(
                    temp,
                    nyquist_res=input.hill_apix() * 2,
                    cutoff_res=(input.hill_cutoff_res_y(), input.hill_cutoff_res_x()),
                    output_size=(input.hill_pny(), input.hill_pnx()),
                    log=0,
                    low_pass_fraction=0,
                    high_pass_fraction=0,
                    norm=0,
                )
            )

    # ── Multi-selection: save current params to df ───────────────
    @reactive.effect
    @reactive.event(input.hill_update_transformation_params)
    def _save_curr_params_to_df():
        req(len(selected_images()) > 1)
        valid = list(input.hill_select_image())
        curr_idx = int(input.hill_curr_img_in_selected_list())
        if curr_idx >= len(valid):
            return
        sel_idx = valid[curr_idx]
        d = avg_ps_pd_main_in_df()
        d[sel_idx] = {
            "apix": input.hill_apix(),
            "negate": input.hill_negate(),
            "angle": input.hill_angle(),
            "dx": input.hill_dx(),
            "dy": input.hill_dy(),
            "mask_radius": input.hill_mask_radius(),
            "mask_len": input.hill_mask_len(),
        }
        avg_ps_pd_main_in_df.set(d)

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
                log=input.hill_log_amp(),
                low_pass_fraction=input.hill_lp_fraction() / 100.0,
                high_pass_fraction=input.hill_hp_fraction() / 100.0,
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
    @output
    @render.ui
    @reactive.event(selected_image_labels)
    def hill_select_from_multiselect_ui():
        req(len(selected_images()) > 1)
        return ui.input_select(
            "hill_curr_img_in_selected_list",
            "Transformation for:",
            choices={
                str(k): selected_image_labels()[k]
                for k in range(len(selected_images()))
            },
        )

    @output
    @render.ui
    def hill_img_2d_uis():
        itype = input.hill_input_type()
        if itype == "Image":
            return [
                ui.input_checkbox("hill_negate", "Invert image contrast", value=False),
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
                    ui.input_numeric(
                        "hill_angle",
                        "Rotate (°)",
                        value=0.0,
                        min=-180.0,
                        max=180.0,
                        step=1.0,
                        update_on="blur",
                    ),
                    ui.input_numeric(
                        "hill_dx",
                        "Shift X (Å)",
                        value=0.0,
                        min=-nx_curr_img() * apix_from_file(),
                        max=nx_curr_img() * apix_from_file(),
                        step=1.0,
                        update_on="blur",
                    ),
                    ui.input_numeric(
                        "hill_dy",
                        "Shift Y (Å)",
                        value=0.0,
                        min=-ny_curr_img() * apix_from_file(),
                        max=ny_curr_img() * apix_from_file(),
                        step=1.0,
                        update_on="blur",
                    ),
                    ui.input_numeric(
                        "hill_mask_radius",
                        "Mask radius (Å)",
                        value=mask_radius_auto(),
                        min=1.0,
                        max=nx_curr_img() / 2 * apix_from_file(),
                        step=1.0,
                        update_on="blur",
                    ),
                    ui.input_numeric(
                        "hill_mask_len",
                        "Mask length (%)",
                        value=mask_len_percent_auto(),
                        min=10.0,
                        max=100.0,
                        step=1.0,
                        update_on="blur",
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
                    ui.input_numeric(
                        "hill_angle",
                        "Rotate (°)",
                        value=0.0,
                        min=-180.0,
                        max=180.0,
                        step=1.0,
                        update_on="blur",
                    ),
                    ui.input_numeric(
                        "hill_dx",
                        "Shift X (Å)",
                        value=0.0,
                        min=-nx_curr_img() * apix_from_file(),
                        max=nx_curr_img() * apix_from_file(),
                        step=1.0,
                        update_on="blur",
                    ),
                    ui.input_numeric(
                        "hill_dy",
                        "Shift Y (Å)",
                        value=0.0,
                        min=-ny_curr_img() * apix_from_file(),
                        max=ny_curr_img() * apix_from_file(),
                        step=1.0,
                        update_on="blur",
                    ),
                    col_widths=6,
                    style="align-items: flex-end;",
                ),
            ]

    @output
    @render.ui
    def hill_img_update_buttons():
        req(len(selected_images()) > 1)
        return ui.layout_columns(
            ui.input_action_button("hill_use_curr_transformed", "Use Current Image"),
            ui.input_action_button(
                "hill_update_transformation_params", "Save Current Params"
            ),
            col_widths=6,
        )

    # ── 2D transform reactive updates ───────────────────────────
    @reactive.effect
    @reactive.event(
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
        temp = selected_images()[0].astype(np.float64)
        if (
            input.hill_angle()
            or input.hill_dx()
            or input.hill_dy()
            or input.hill_negate()
        ):
            temp = hill.transform_2d_image(
                temp,
                input.hill_angle(),
                input.hill_dx(),
                input.hill_dy(),
                input.hill_negate(),
                input.hill_apix(),
            )
        if (
            input.hill_mask_radius() > 0
            and input.hill_mask_len() > 0
            and input.hill_mask_len() <= 100
        ):
            temp = hill.mask_2d_filament(
                temp,
                input.hill_mask_radius(),
                input.hill_apix(),
                input.hill_mask_len() / 100.0,
            )
        # Skip if param key unchanged (redundant round-trip)
        key = (
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
                log=input.hill_log_amp(),
                low_pass_fraction=input.hill_lp_fraction() / 100.0,
                high_pass_fraction=input.hill_hp_fraction() / 100.0,
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
                    log=input.hill_log_amp(),
                    low_pass_fraction=input.hill_lp_fraction() / 100.0,
                    high_pass_fraction=input.hill_hp_fraction() / 100.0,
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
        if phase is not None:
            data_source_ps.data = {
                **data_source_ps.data,
                "image": [np.asarray(pwr)],
                "phase": [np.fmod(np.rad2deg(phase) + 360, 360).astype(np.float16)],
            }
        else:
            data_source_ps.data = {
                **data_source_ps.data,
                "image": [np.asarray(pwr)],
            }

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
        data_source_pd.data = {
            **data_source_pd.data,
            "image": [np.asarray(pd)],
            "phase": (
                [np.fmod(np.rad2deg(phase) + 360, 360).astype(np.float16)]
                if phase is not None
                else []
            ),
        }

    # ── Sidebar figure updates ──────────────────────────────────
    @reactive.effect
    @reactive.event(selected_images, input.hill_apix)
    def _update_selected_image_figure():
        req(len(selected_images()) > 0)
        curr_img = selected_images()[0]
        h, w = curr_img.shape
        dx = input.hill_apix()
        dy = input.hill_apix()
        aspect_ratio = w * dx / (h * dy)
        fig_selected_image.x_range.start = -w // 2 * dx
        fig_selected_image.x_range.end = (w // 2 - 1) * dx
        fig_selected_image.y_range.start = (h // 2 - 1) * dy
        fig_selected_image.y_range.end = -h // 2 * dy
        fig_selected_image.aspect_ratio = aspect_ratio
        fig_selected_image.title.text = (
            f"Selected Image ({nx_curr_img()}x{ny_curr_img()})"
        )
        source_selected_image.data = {
            **source_selected_image.data,
            "image": [curr_img],
            "x": [-w // 2 * dx],
            "y": [-h // 2 * dy],
            "dw": [w * dx],
            "dh": [h * dy],
        }

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
        if (
            input.hill_angle()
            or input.hill_dx()
            or input.hill_dy()
            or input.hill_negate()
        ):
            unmasked_data = hill.transform_2d_image(
                selected_images()[0],
                input.hill_angle(),
                input.hill_dx(),
                input.hill_dy(),
                input.hill_negate(),
                input.hill_apix(),
            )
        else:
            unmasked_data = selected_images()[0]
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

    @reactive.effect
    @reactive.event(data_2d_transformed)
    def _update_acf_data():
        req(input.hill_input_type() in ("Image",))
        req(data_2d_transformed() is not None)
        ny, _nx = data_2d_transformed().shape
        curr_acf = hill.auto_correlation(
            data_2d_transformed(),
            sqrt=True,
            high_pass_fraction=0.1,
        )
        if curr_acf is None or np.isnan(curr_acf).any():
            return
        fig_acf.y_range.start = ny // 2 * input.hill_apix()
        fig_acf.y_range.end = -ny // 2 * input.hill_apix()
        fig_acf.frame_height = ny
        fig_acf_line.data_source.data = {
            "x": np.max(curr_acf, axis=1),
            "y": np.arange(-ny // 2, ny // 2) * input.hill_apix(),
        }

    # ── Main plot renderer ──────────────────────────────────────
    @render_widget
    def hill_main_plots():
        return main_plot_widget

    # ── Sidebar render functions ───────────────────────────────
    @output
    @render.ui
    def hill_error_display():
        return None

    @render_widget
    def hill_display_selected_image():
        return fig_selected_image

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
            csym=input.hill_csym(),
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
                csym=input.hill_csym_ahs(),
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
            return ui.input_action_button(
                "hill_run_get_average_power_spectra",
                "Average PS & PD from selected images",
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
                    open=False,
                )
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
                    log=input.hill_log_amp(),
                    low_pass_fraction=input.hill_lp_fraction() / 100.0,
                    high_pass_fraction=input.hill_hp_fraction() / 100.0,
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
                    log=input.hill_log_amp(),
                    low_pass_fraction=input.hill_lp_fraction() / 100.0,
                    high_pass_fraction=input.hill_hp_fraction() / 100.0,
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


def _setup_spinner_js(
    spinner_twist,
    spinner_pitch,
    spinner_rise,
    slider_twist,
    slider_pitch,
    slider_rise,
    prefix,
):
    """Set up CustomJS for Bokeh spinners to send values to Shiny."""
    rise_code = f"""
        Shiny.setInputValue("{prefix}_rise", spinner_rise.value, {{priority: 'event'}});
        slider_rise.value = spinner_rise.value;
    """
    pitch_code = f"""
        Shiny.setInputValue("{prefix}_pitch", spinner_pitch.value, {{priority: 'event'}});
        slider_pitch.value = spinner_pitch.value;
    """
    twist_code = f"""
        Shiny.setInputValue("{prefix}_twist", spinner_twist.value, {{priority: 'event'}});
        slider_twist.value = spinner_twist.value;
    """
    args = dict(
        spinner_twist=spinner_twist,
        spinner_pitch=spinner_pitch,
        spinner_rise=spinner_rise,
        slider_twist=slider_twist,
        slider_pitch=slider_pitch,
        slider_rise=slider_rise,
    )
    spinner_twist.js_on_change("value", CustomJS(args=args, code=twist_code))
    spinner_pitch.js_on_change("value", CustomJS(args=args, code=pitch_code))
    spinner_rise.js_on_change("value", CustomJS(args=args, code=rise_code))


def _setup_slider_js(
    slider_twist,
    slider_pitch,
    slider_rise,
    spinner_twist,
    spinner_pitch,
    spinner_rise,
    fig_ellipses,
    prefix,
):
    """Set up CustomJS for Bokeh sliders to update layer lines and Shiny inputs."""
    rise_code = f"""
        var twist_sign = 1.;
        if (slider_twist.value < 0) {{ twist_sign = -1.; }}
        if ($("input[type='radio'][name='{prefix}_use_twist_pitch']:checked").val() == "Pitch") {{
            var t = twist_sign * 360/(slider_pitch.value/slider_rise.value);
            if (t != slider_twist.value) slider_twist.value = t;
        }} else {{
            var p = Math.abs(360/slider_twist.value * slider_rise.value);
            if (p != slider_pitch.value) slider_pitch.value = p;
        }}
        if (spinner_rise.value != slider_rise.value) spinner_rise.value = slider_rise.value;
        var pitch_inv = 1./slider_pitch.value;
        var rise_inv = 1./slider_rise.value;
        for (var fi = 0; fi < fig_ellipses.length; fi++) {{
            var el = fig_ellipses[fi];
            const m = el.tags[0];
            const ns = el.tags[1];
            var y = el.data_source.data.y;
            for (var i = 0; i < ns.length; i++) {{
                y[i] = m * rise_inv + ns[i] * pitch_inv;
            }}
            el.data_source.change.emit();
        }}
    """
    pitch_code = f"""
        var twist_sign = 1.;
        if (slider_twist.value < 0) {{ twist_sign = -1.; }}
        var t = twist_sign * 360/(slider_pitch.value/slider_rise.value);
        if (t != slider_twist.value) slider_twist.value = t;
        if (spinner_pitch.value != slider_pitch.value) spinner_pitch.value = slider_pitch.value;
        var pitch_inv = 1./slider_pitch.value;
        var rise_inv = 1./slider_rise.value;
        for (var fi = 0; fi < fig_ellipses.length; fi++) {{
            var el = fig_ellipses[fi];
            const m = el.tags[0];
            const ns = el.tags[1];
            var y = el.data_source.data.y;
            for (var i = 0; i < ns.length; i++) {{
                y[i] = m * rise_inv + ns[i] * pitch_inv;
            }}
            el.data_source.change.emit();
        }}
    """
    twist_code = f"""
        var p = Math.abs(360/slider_twist.value * slider_rise.value);
        if (p != slider_pitch.value) slider_pitch.value = p;
        if (spinner_twist.value != slider_twist.value) spinner_twist.value = slider_twist.value;
    """
    args = dict(
        fig_ellipses=fig_ellipses,
        slider_twist=slider_twist,
        slider_pitch=slider_pitch,
        slider_rise=slider_rise,
        spinner_twist=spinner_twist,
        spinner_pitch=spinner_pitch,
        spinner_rise=spinner_rise,
    )
    slider_twist.js_on_change("value", CustomJS(args=args, code=twist_code))
    slider_pitch.js_on_change("value", CustomJS(args=args, code=pitch_code))
    slider_rise.js_on_change("value", CustomJS(args=args, code=rise_code))


def _setup_spinner_throttle_js(spinner_rise, slider_rise, spinner_pitch, slider_pitch):
    """Set up CustomJS for spinner value_throttled to adjust slider ranges."""
    rise_code = """
        slider_rise.start = spinner_rise.value/2.0;
        slider_rise.end = Math.min(spinner_rise.value*2.0, spinner_pitch.value);
        slider_rise.step = slider_rise.end*0.001;
        spinner_pitch.low = slider_rise.value;
    """
    pitch_code = """
        slider_pitch.start = Math.max(spinner_pitch.value/2.0, spinner_rise.value);
        slider_pitch.end = Math.min(spinner_pitch.value*2.0, 10000.0);
        slider_pitch.step = slider_pitch.end*0.001;
        spinner_rise.high = slider_pitch.value;
    """
    rise_cb = CustomJS(
        args=dict(
            spinner_rise=spinner_rise,
            slider_rise=slider_rise,
            spinner_pitch=spinner_pitch,
        ),
        code=rise_code,
    )
    pitch_cb = CustomJS(
        args=dict(
            spinner_pitch=spinner_pitch,
            slider_pitch=slider_pitch,
            spinner_rise=spinner_rise,
        ),
        code=pitch_code,
    )
    spinner_rise.js_on_change("value_throttled", rise_cb)
    spinner_pitch.js_on_change("value_throttled", pitch_cb)
