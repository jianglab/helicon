"""The page of the hill tab: its inputs, controls and plots, and what only they need."""

from __future__ import annotations
import helicon
from shiny import ui, module
from shinywidgets import output_widget
from .. import auto_method_ui, deployment


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


_INIT_HELICAL_RADIUS = 77.6


# the largest diameter the slider offers before an image is loaded (then: its width)
_DIAMETER_SLIDER_MAX = 1000.0


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
                        ui.output_ui("hill_get_avg_ps_ui"),
                        ui.output_ui("hill_stitch_status"),
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
                        # under the profile plot, whose dashed lines show them: the
                        # mask (green) and the filament's diameter (orange)
                        ui.layout_columns(
                            ui.tooltip(
                                helicon.shiny.slider(
                                    "hill_mask_radius",
                                    "Mask radius (Å)",
                                    min=0.0,
                                    max=_DIAMETER_SLIDER_MAX / 2,
                                    value=0.0,
                                    step=1.0,
                                ),
                                "The radius beyond which the image is masked out, "
                                "shown as the green dashed lines in the profile above. "
                                "0: no mask.",
                            ),
                            ui.tooltip(
                                helicon.shiny.slider(
                                    "hill_diameter",
                                    "Tube/filament diameter (Å)",
                                    min=1.0,
                                    max=_DIAMETER_SLIDER_MAX,
                                    value=_INIT_HELICAL_RADIUS * 2,
                                    step=0.5,
                                ),
                                "The outer diameter of the filament or tube, shown as the orange "
                                "dashed lines in the profile above. It decides where each "
                                "layer line starts away from the meridian, so it matters for "
                                "the predicted lines and for 'Suggest csym, twist and rise'. "
                                "It is estimated from each image; double-click the number to "
                                "type one.",
                            ),
                            col_widths=6,
                            style="align-items: flex-end;",
                        ),
                        output_widget("hill_plot_acf", width="auto", height="auto"),
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
                                    "first. Uses the filament diameter (above) and the "
                                    "resolution limit X (Parameters tab); several class "
                                    "averages of one helix do better than one.",
                                ),
                                class_="hill-suggest-button",
                                style="text-align: center;",
                            ),
                            ui.output_ui("hill_symmetry_message"),
                            ui.output_ui("hill_symmetry_progress"),
                            ui.output_ui("hill_symmetry_warning"),
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
                                        "hill_out_of_plane_tilt",
                                        "Out-of-plane tilt (°)",
                                        value=0.0,
                                        min=-90.0,
                                        max=90.0,
                                        step=1.0,
                                        update_on="blur",
                                    ),
                                    ui.input_radio_buttons(
                                        "hill_use_twist_pitch",
                                        "Keep twist/pitch when changing rise:",
                                        choices=["Twist", "Pitch"],
                                        selected="Twist",
                                        inline=True,
                                    ),
                                    col_widths=6,
                                    style="align-items: flex-end;",
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


# A Bokeh slider of the helix whose number can be typed in place, as on the
# Shiny sliders of helicon.shiny.slider: double-clicking the slider's title or
# number opens an editor on the number; Enter applies it (the range of the slider
# follows), Escape or clicking away leaves the slider as it was. A typed number
# is applied as the end of a drag is (``value`` then ``value_throttled``), so the
# layer lines preview and the server is told once, by the slider's own callbacks.
_EDITABLE_SLIDER = "helicon-editable-bokeh-slider"


_MODEL_CLASS = "helicon-bokeh-model-"


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
