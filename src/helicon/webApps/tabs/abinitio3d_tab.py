"""AbInitio3D tab — an ab initio 3D map of a helical filament from its 2D classes.

Each class average of one helical type is a view of the filament at an
azimuthal angle; the segments of a filament run through the classes in the
order of those angles, one repeat per turn of the ring. The tab finds the
repeat and the class azimuths from the segment pairs, assigns every segment
its Euler angles and origin, and builds a 3D map from the class averages or,
with relion_reconstruct, from the segments.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import plotly.io as pio

import helicon
from shiny import reactive, ui, module, req, render
from shiny.module import resolve_id

from .. import bookmark, class2d_files, deployment

from ..lib import helical_pitch_compute as compute
from ..lib import helical_pitch_map as maps
from ..lib import helical_pitch_phase as phase
from ..lib import helical_pitch_relion as relion

logger = logging.getLogger(__name__)

_urls = {
    "empiar-10940_job010": (
        "https://ftp.ebi.ac.uk/empiar/world_availability/10940/data/EMPIAR/Class2D/job010/run_it020_data.star",
        "https://ftp.ebi.ac.uk/empiar/world_availability/10940/data/EMPIAR/Class2D/job010/run_it020_classes.mrcs",
    )
}
_url_key = "empiar-10940_job010"

BOOKMARK_DEFAULTS = {
    "mode_params": ("input_mode_params", "url"),
    "url_params": ("url_params", _urls[_url_key][0]),
    "mode_classes": ("input_mode_classes", "url"),
    "url_classes": ("url_classes", _urls[_url_key][1]),
    "server_params": ("server_params", ""),
    "server_classes": ("server_classes", ""),
    "ignore_blank": ("ignore_blank", True),
    "sort_abundance": ("sort_abundance", True),
    "rise": ("rise", 4.75),
    "min_twist": ("min_twist", 0.3),
    "merge_counterparts": ("merge_counterparts", True),
    "csym": ("rot_fold", 1),
    "hand": ("map_hand", "left"),
    "map_method": ("map_method", "joint"),
    "n_boot": ("phase_n_boot", 20),
    "split": ("split_axis_distance", 50),
    "two_rounds": ("map_refine", False),
    # suggested from the class fits of each result
    "fit_cut": ("fit_threshold", 0.1, bookmark.DERIVED),
}


def _fig_to_html(fig, multi_crosshair=False, plot_id=None):
    """Convert a plotly figure to responsive HTML for rendering via render.ui."""
    postscript = ""
    html = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=True,
        div_id=plot_id,
        config={"responsive": True, "displayModeBar": False},
        post_script=postscript,
    )
    return ui.HTML(html)


# The plots size themselves to their container, which has no
# height of its own inside a card, so each is given one; without it the three
# collapse onto each other and the histogram covers the download links.
_PLOT_BOX = "height: 340px; min-height: 340px; flex-shrink: 0; overflow: hidden;"


def _tip(element, text, placement="top"):
    """``element`` with ``text`` shown on hover."""
    return ui.tooltip(element, text, placement=placement)


def _info(text, placement="top"):
    """A small (i) that shows ``text`` on hover, to put after a title or label."""
    return ui.tooltip(
        ui.tags.span(
            " \u24d8", style="cursor: help; color: #6c757d; font-size: 0.9em;"
        ),
        text,
        placement=placement,
    )


# Alt-click on the first image of the class gallery selects (or unselects)
# every image, which is how the gallery does "select all"; clicking it twice
# when the first image is already selected ends with all selected
_SELECT_ALL_JS = """
(function() {
  var first = document.getElementById('%s');
  if (!first) return;
  var click = function() {
    first.dispatchEvent(new MouseEvent('click',
      {bubbles: true, cancelable: true, view: window, altKey: true}));
  };
  if (first.getAttribute('selected') === 'true') click();
  click();
})();
"""


# How the reasons for taking a class out are shown in a gallery label.
_SHORT_REASON = {
    phase.TOO_FEW: "too few pairs",
    phase.AT_CHANCE: "off the ring",
    phase.BELOW_THRESHOLD: "below the threshold",
}

# The pitch results, and everything that works from them, stay hidden until
# there is a result to show.
_RESULTS_BOX = "ab_results_box"
# The suggestions come in once the first ring is drawn, and stay while the
# selection is changed (which clears the result) until other data is loaded.
_SUGGESTIONS_BOX = "ab_suggestions_box"

# Lets the server enable or disable a button; a task button's own update
# message changes only its busy state.
_SET_DISABLED_JS = ui.tags.script(
    """
if (!window.__heliconSetDisabled) {
  window.__heliconSetDisabled = true;
  $(document).on('shiny:connected', function() {
    Shiny.addCustomMessageHandler('helicon-set-disabled', function(m) {
      var el = document.getElementById(m.id);
      if (el) el.disabled = m.disabled;
    });
  });
}
"""
)


# The results' controls (the sliders under the plots, the download, the map
# settings) wait for the plots: the plots are drawn on the server after a
# result arrives, and until the three every result has (repeat scan, ring,
# ranked fits) have come the controls stay invisible -- keeping their places,
# so nothing moves when they appear. By opacity, not visibility: the sliders'
# library sets its number labels visible itself, which showed them early. The per-filament histogram is not waited
# for: it is empty when no filament has a repeat of its own.
_RESULTS_ORDER_CSS = ui.tags.style(
    f"""
#{_RESULTS_BOX}:has(.ab-wait > .shiny-html-output:empty) .ab-after {{
  opacity: 0;
  pointer-events: none;
}}
"""
)

# The smallest-twist box and the Estimate pitch button on one row, the box
# narrow and both aligned at the bottom.
_PITCH_ROW_CSS = ui.tags.style(
    """
.ab-pitch-row {
  display: grid;
  grid-template-columns: minmax(9rem, 11rem) 1fr;
  gap: 8px;
  align-items: end;
}
.ab-pitch-row > .shiny-input-container { margin-bottom: 0; }
"""
)


@module.ui
def abinitio3d_tab_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.navset_pill(
                ui.nav_panel(
                    "Inputs",
                    ui.div(
                        ui.input_radio_buttons(
                            "input_mode_params",
                            "How to obtain the Class2D parameter file:",
                            choices=helicon.shiny.source_modes(deployment.is_cloud()),
                            selected="url",
                            inline=True,
                        ),
                        ui.panel_conditional(
                            "input.input_mode_params === 'upload'",
                            ui.input_file(
                                "upload_params",
                                "Upload the class2d parameters in a RELION star or cryoSPARC cs file",
                                accept=[".star", ".cs"],
                                placeholder="star or cs file",
                            ),
                        ),
                        ui.panel_conditional(
                            "input.input_mode_params === 'url'",
                            ui.input_text(
                                "url_params",
                                "Download URL for a RELION star or cryoSPARC cs file",
                                value=_urls[_url_key][0],
                            ),
                        ),
                        ui.panel_conditional(
                            "input.input_mode_params === 'server'",
                            helicon.shiny.file_picker_field(
                                ui.input_text(
                                    "server_params",
                                    "RELION star or cryoSPARC cs file on the server",
                                    value="",
                                    placeholder="Browse, or type a path",
                                ),
                                "params_browse",
                            ),
                        ),
                        ui.input_radio_buttons(
                            "input_mode_classes",
                            "How to obtain the class average images:",
                            choices=helicon.shiny.source_modes(deployment.is_cloud()),
                            selected="url",
                            inline=True,
                        ),
                        ui.panel_conditional(
                            "input.input_mode_classes === 'upload'",
                            ui.input_file(
                                "upload_classes",
                                "Upload the class averages in MRC format (.mrcs, .mrc)",
                                accept=[".mrcs", ".mrc"],
                                placeholder="mrcs or mrc file",
                            ),
                        ),
                        ui.panel_conditional(
                            "input.input_mode_classes === 'url'",
                            ui.input_text(
                                "url_classes",
                                "Download URL for a RELION or cryoSPARC Class2D output mrc(s) file",
                                value=_urls[_url_key][1],
                            ),
                        ),
                        ui.panel_conditional(
                            "input.input_mode_classes === 'server'",
                            helicon.shiny.file_picker_field(
                                ui.input_text(
                                    "server_classes",
                                    "RELION or cryoSPARC Class2D class averages (mrcs, mrc) "
                                    "on the server",
                                    value="",
                                    placeholder="Browse, or type a path",
                                ),
                                "classes_browse",
                            ),
                        ),
                        ui.input_task_button("run", label="Run", style="width: 100%;"),
                        id="input_files",
                        style="flex-shrink: 0;",
                    ),
                    ui.div(
                        ui.output_ui("select_classes_gallery"),
                        id="class-selection",
                        style="flex-grow: 1; overflow-y: auto;",
                    ),
                ),
                ui.nav_panel(
                    "Parameters",
                    ui.input_checkbox(
                        "ignore_blank", "Ignore blank classes", value=True
                    ),
                    ui.input_checkbox(
                        "sort_abundance", "Sort the classes by abundance", value=True
                    ),
                    ui.input_checkbox(
                        "ring_labels",
                        "Show class labels on the azimuthal-angle ring",
                        value=False,
                    ),
                    ui.input_numeric(
                        "rise",
                        "Helical rise (\u00c5)",
                        min=0.01,
                        max=1000.0,
                        value=4.75,
                        step=0.01,
                        update_on="blur",
                    ),
                    ui.input_checkbox(
                        "merge_counterparts",
                        "Merge classes that are 180\u00b0-rotated copies of each "
                        "other (registers the class averages; slow for many classes)",
                        value=True,
                    ),
                    ui.input_numeric(
                        "split_axis_distance",
                        ui.span(
                            "Split tube ids holding distinct filaments (\u00c5)",
                            _info(
                                "Segments of one tube id that lie farther than "
                                "this from each other's helical axis are taken "
                                "to be on different filaments -- as when "
                                "particles merged from several extractions "
                                "number their tubes alike -- and given ids of "
                                "their own, also in the exported star file. "
                                "Gaps along the axis never split a filament. "
                                "0: off."
                            ),
                        ),
                        min=0,
                        max=10000,
                        value=50,
                        step=10,
                        update_on="blur",
                    ),
                    ui.input_numeric(
                        "phase_n_boot",
                        ui.span(
                            "Bootstrap resamples",
                            _info(
                                "Resamples of the filaments for the uncertainty "
                                "of the repeat"
                            ),
                        ),
                        min=2,
                        max=200,
                        value=20,
                        step=1,
                        update_on="blur",
                    ),
                    ui.hr(),
                    ui.tags.b("3D map from classes"),
                    ui.div(
                        ui.input_checkbox(
                            "map_csym",
                            ui.span(
                                "Impose C",
                                _info(
                                    "Off, the C symmetry seen in the z "
                                    "section is an independent check"
                                ),
                            ),
                            value=False,
                        ),
                        ui.input_checkbox(
                            "map_refine",
                            ui.span(
                                "Two rounds",
                                _info(
                                    "Align the averages to the first map and rebuild"
                                ),
                            ),
                            value=False,
                        ),
                    ),
                    ui.input_radio_buttons(
                        "map_method",
                        "Reconstruction Method",
                        {
                            # "joint" kept as the key of elastic net, so older
                            # bookmarks still name it
                            "joint": "Linear regression - elasticnet",
                            "gauss": "Linear regression - gauss",
                            "backprojection": "Back-projection",
                        },
                        selected="joint",
                    ),
                    ui.div(
                        ui.panel_conditional(
                            "input.map_method === 'backprojection'",
                            ui.input_numeric(
                                "map_hsym",
                                ui.span(
                                    "Helical sym order",
                                    _info(
                                        "1: none, each average used "
                                        "once; n: each used n times "
                                        "along the filament, and the map "
                                        "symmetrized from its central third"
                                    ),
                                ),
                                min=1,
                                max=1000,
                                value=1,
                                step=1,
                                update_on="blur",
                            ),
                        ),
                        ui.panel_conditional(
                            "input.map_method !== 'backprojection'",
                            ui.tags.span(
                                "Helical sym: full (linear regression)",
                                class_="text-muted small",
                            ),
                        ),
                    ),
                ),
            ),
            width="33vw",
            style="display: flex; flex-direction: column; height: 100%;",
        ),
        ui.h1(
            "AbInitio3D: ab initio 3D map from the 2D classes of helical filaments",
            style="font-weight: bold;",
        ),
        ui.div(
            ui.layout_columns(
                ui.div(
                    ui.card(
                        ui.card_header(
                            "Selected classes",
                            _info(
                                "The classes the pitch is computed from; the "
                                "sidebar gallery shows the same selection. Click "
                                "a class to pick it (Shift-click for several, "
                                "Alt-click for all), then remove the picked ones."
                            ),
                        ),
                        ui.div(
                            ui.output_ui("accepted_gallery"),
                            style="max-height: 32vh; overflow-y: auto;",
                        ),
                        ui.layout_columns(
                            _tip(
                                ui.input_action_button(
                                    "accepted_select_all",
                                    "Select all",
                                    class_="btn-sm btn-outline-secondary",
                                    onclick=_SELECT_ALL_JS
                                    % module.resolve_id("select_classes_inner_image_1"),
                                ),
                                "Select every input class",
                            ),
                            _tip(
                                ui.input_action_button(
                                    "accepted_remove",
                                    "Remove picked",
                                    class_="btn-sm btn-outline-danger",
                                ),
                                "Take the picked classes out of the computation",
                            ),
                            col_widths=(6, 6),
                            fill=False,
                        ),
                    ),
                    ui.card(
                        ui.card_header("Suggestions"),
                        _tip(
                            ui.input_task_button(
                                "suggest_run",
                                label="Suggest classes",
                                class_="btn-outline-secondary",
                            ),
                            "More classes of this type: those that look like a "
                            "selected class turned 180\u00b0 (\u21bb, with the "
                            "correlation), and those on the same filaments as the "
                            "selected classes (\u25cf, the share of the filaments "
                            "having the class that are mostly selected classes). "
                            "Listed best first by the fit each would get on the "
                            "current ring (fit, on the scale of the ranked fits).",
                        ),
                        ui.div(
                            ui.output_ui("suggested_gallery"),
                            style="max-height: 32vh; overflow-y: auto;",
                        ),
                        ui.output_ui("suggested_buttons"),
                        id=_SUGGESTIONS_BOX,
                    ),
                ),
                # one block that scrolls: a card body is a flex container that
                # would otherwise shrink every plot and row to fit its height
                ui.div(
                    ui.div(
                        # the smallest twist sets how far the estimate searches,
                        # so it sits with the button that starts it
                        ui.input_numeric(
                            "min_twist",
                            ui.span(
                                "Smallest |twist| (\u00b0)",
                                _info(
                                    "With the rise (Parameters), sets the longest "
                                    "repeat the pitch estimate searches: "
                                    "360\u00b0 \u00d7 rise / |twist| (4.75 \u00c5 "
                                    "and 0.3\u00b0: 5700 \u00c5). Any C symmetry "
                                    "only shortens the repeat of a given twist. "
                                    "Lower it for filaments that twist very slowly."
                                ),
                            ),
                            min=0.05,
                            max=30.0,
                            value=0.3,
                            step=0.05,
                            update_on="blur",
                            width="100%",
                        ),
                        _tip(
                            ui.input_task_button(
                                "phase_run",
                                label="Estimate pitch",
                                style="width: 100%;",
                                # enabled once three classes are selected
                                disabled=True,
                            ),
                            "Uses all pairs of segments among the selected "
                            "classes, same-class and cross-class: each class is a "
                            "view at an azimuthal angle on a ring one repeat "
                            "around. Select the classes of one type only -- with "
                            "their 180\u00b0-rotated versions -- and leave out junk. "
                            "Gives the calibrated repeat, how well each class fits "
                            "the ring, and each long filament's own repeat.",
                        ),
                        _SET_DISABLED_JS,
                        _PITCH_ROW_CSS,
                        class_="ab-pitch-row",
                    ),
                    ui.output_ui("phase_summary"),
                    # takes no room, and is never hidden itself, so that it keeps
                    # running: it hides the results below until there are some
                    ui.div(
                        # hidden until results_visibility, drawn after this,
                        # shows them
                        ui.tags.style(
                            f"#{_RESULTS_BOX}, #{resolve_id(_SUGGESTIONS_BOX)} "
                            "{ display: none; }"
                        ),
                        ui.output_ui("results_visibility"),
                        _RESULTS_ORDER_CSS,
                        style="height: 0; margin: 0; padding: 0;",
                    ),
                    ui.div(
                        ui.layout_columns(
                            ui.div(
                                ui.output_ui("phase_scan_plot"),
                                style=_PLOT_BOX,
                                class_="ab-plot-box ab-wait",
                            ),
                            ui.div(
                                ui.output_ui("phase_circle_plot"),
                                style=_PLOT_BOX,
                                class_="ab-plot-box ab-wait",
                            ),
                            col_widths=(7, 5),
                            fill=False,
                        ),
                        ui.layout_columns(
                            # the histogram, and below it the two ranges that
                            # select filaments from it, one under the other
                            ui.div(
                                ui.div(
                                    ui.output_ui("filament_pitch_plot"),
                                    style=_PLOT_BOX,
                                    class_="ab-plot-box",
                                ),
                                ui.div(
                                    helicon.shiny.range_slider(
                                        "pitch_band",
                                        ui.span(
                                            "Repeat (\u00c5)",
                                            _info(
                                                "The filaments whose own repeat is in this "
                                                "range are downloaded and reconstructed. "
                                                "Double-click a number to type it."
                                            ),
                                        ),
                                        min=0,
                                        max=1000,
                                        value=(0, 1000),
                                        step=1,
                                    ),
                                    class_="ab-after",
                                ),
                                ui.div(
                                    helicon.shiny.range_slider(
                                        "length_range",
                                        ui.span(
                                            "Length (\u00c5)",
                                            _info(
                                                "The filaments whose length is in this "
                                                "range; double-click a number to type it."
                                            ),
                                        ),
                                        min=0,
                                        max=10000,
                                        value=(0, 10000),
                                        step=10,
                                    ),
                                    class_="ab-after",
                                ),
                            ),
                            # below the ring: the same fits, ranked
                            ui.div(
                                ui.div(
                                    ui.output_ui("class_fit_rank_plot"),
                                    style=_PLOT_BOX,
                                    class_="ab-plot-box ab-wait",
                                ),
                                ui.div(
                                    helicon.shiny.slider(
                                        "fit_threshold",
                                        ui.span(
                                            "Fit threshold",
                                            _info(
                                                "Classes whose fit is below this are "
                                                "picked to remove (the orange line). It "
                                                "starts in the first clear drop of the "
                                                "ranked fits below the best ones, and "
                                                "never below the \u2248 junk line (red, "
                                                "dashed). Double-click the number to "
                                                "type it."
                                            ),
                                        ),
                                        min=0.0,
                                        max=1.0,
                                        value=0.1,
                                        step=0.01,
                                    ),
                                    # what the two sliders select, to take away
                                    ui.output_ui("pitch_band_download"),
                                    class_="ab-after",
                                ),
                            ),
                            col_widths=(7, 5),
                            fill=False,
                        ),
                        ui.div(
                            ui.output_ui("relion_ui"),
                            ui.output_ui("relion_display"),
                            ui.hr(),
                            ui.layout_columns(
                                ui.input_numeric(
                                    "rot_fold",
                                    ui.span(
                                        "C symmetry",
                                        _info(
                                            "The pitch is C \u00d7 the repeat. It sets the "
                                            "twist and the rlnAngleRot of the star file, "
                                            "and the 3D maps."
                                        ),
                                    ),
                                    min=1,
                                    max=12,
                                    value=1,
                                    step=1,
                                    update_on="blur",
                                ),
                                ui.input_radio_buttons(
                                    "map_hand",
                                    ui.span(
                                        "Hand",
                                        _info(
                                            "Twist < 0 for a left-handed helix (RELION's "
                                            "sign). The images cannot tell: the two choices "
                                            "give mirror images."
                                        ),
                                    ),
                                    {"left": "Left", "right": "Right"},
                                    selected="left",
                                    inline=True,
                                ),
                                _tip(
                                    ui.input_task_button(
                                        "map_run",
                                        "3D map from classes",
                                        style="width: 100%;",
                                    ),
                                    "The class averages are placed at their azimuths on "
                                    "the ring and assembled into one helical map: shown "
                                    "as the averages tiled at those azimuths, the map's "
                                    "side projection over ~1.2 pitches, and its central z "
                                    "section.",
                                ),
                                col_widths=(3, 4, 5),
                                style="align-items: flex-end;",
                                fill=False,
                            ),
                            ui.output_ui("map_display"),
                            ui.output_ui("map_download"),
                            class_="ab-after",
                        ),
                        id=_RESULTS_BOX,
                    ),
                ),
                col_widths=(5, 7),
            ),
            style="overflow-y: auto; height: 100%;",
        ),
        ui.HTML(
            "<i><p style='margin:2px 0'>Developed by the <a href='https://jianglab.science.psu.edu/helicon' target='_blank'>Jiang Lab</a>. Report issues to <a href='https://github.com/jianglab/helicon/issues' target='_blank'>helicon@GitHub</a>.</p></i>"
        ),
    )


def _mrc_bytes(volume, apix):
    """The bytes of an MRC file holding ``volume``.

    Parameters
    ----------
    volume : numpy.ndarray
        The 3D map (z, y, x).
    apix : float
        The voxel size (Angstrom).

    Returns
    -------
    bytes
    """
    import tempfile

    import mrcfile

    fd, name = tempfile.mkstemp(suffix=".mrc")
    os.close(fd)
    path = Path(name)
    try:
        with mrcfile.new(name, overwrite=True) as mrc:
            mrc.set_data(np.asarray(volume, dtype=np.float32))
            mrc.voxel_size = float(apix)
        return path.read_bytes()
    finally:
        path.unlink(missing_ok=True)


@module.server
def abinitio3d_tab_server(input, output, session):
    # the Class2D parameters as read, and as used: with tube ids that hold
    # several filaments split (helicon.split_distinct_filaments)
    params_raw = reactive.value(None)
    params = reactive.value(None)
    data_all = reactive.value(None)

    # Run is pressed for the user, when the tab starts, if both inputs are URLs;
    # an upload needs a file the user has yet to choose, so nothing runs for it.
    auto_run = reactive.value(0)

    abundance = reactive.value([])
    image_size = reactive.value(0)

    displayed_class_ids = reactive.value([])
    displayed_class_images = reactive.value([])
    displayed_class_title = reactive.value("Select class(es):")
    displayed_class_labels = reactive.value([])

    initial_selected_image_indices = reactive.value([0])
    # bumped whenever the code, not the user, changes the selection, so that the
    # gallery is redrawn even when the new selection equals the previous one
    gallery_version = reactive.value(0)
    selected_images = reactive.value([])
    selected_image_labels = reactive.value([])

    # ── Data loading ──

    @reactive.effect
    @reactive.event(input.run, auto_run, ignore_none=False, ignore_init=True)
    def get_class2d_from_upload():
        req(input.input_mode_classes() == "upload")
        fileinfo = input.upload_classes()
        req(fileinfo)
        class_file = fileinfo[0]["datapath"]
        try:
            data, apix = compute.get_class2d_from_file(class_file)
            nx = data.shape[-1]
        except Exception as e:
            logger.error("Failed to read uploaded class images: %s", e)
            data, apix, nx = None, 0, 0
            ui.modal_show(
                ui.modal(
                    f"failed to read the uploaded 2D class average images from {fileinfo[0]['name']}",
                    title="File upload error",
                    easy_close=True,
                    footer=None,
                )
            )
        data_all.set((data, apix))
        image_size.set(nx)

    @reactive.effect
    @reactive.event(input.run, auto_run, ignore_none=False, ignore_init=True)
    def get_class2d_from_url():
        req(input.input_mode_classes() in ("url", "server"))
        url = _source(
            input.input_mode_classes(), input.url_classes, input.server_classes
        )
        req(len(url) > 0)
        if input.input_mode_classes() == "server" and deployment.refuse_server_mode():
            return
        if deployment.refuse_local_path(url):
            return
        try:
            data, apix = compute.get_class2d_from_url(url)
            nx = data.shape[-1]
        except Exception as e:
            logger.error("Failed to download class images: %s", e)
            data, apix, nx = None, 0, 0
            ui.modal_show(
                ui.modal(
                    f"failed to read 2D class average images from {url}",
                    title="File download error",
                    easy_close=True,
                    footer=None,
                )
            )
        data_all.set((data, apix))
        image_size.set(nx)

    @reactive.effect
    @reactive.event(input.run, auto_run, ignore_none=False, ignore_init=True)
    def get_params_from_upload():
        req(input.input_mode_params() == "upload")
        fileinfo = input.upload_params()
        req(fileinfo)
        param_file = fileinfo[0]["datapath"]
        msg = None
        try:
            tmp_params = compute.get_class2d_helix_params_from_file(param_file)
        except Exception as e:
            msg = str(e).replace(param_file, fileinfo[0]["name"])
            tmp_params = None
        params_raw.set(tmp_params)
        if tmp_params is None:
            if msg is None:
                msg = f"failed to parse the upload class2D parameters from {fileinfo[0]['name']}"
            msg_ui = ui.markdown(
                msg.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br><br>")
            )
            ui.modal_show(
                ui.modal(
                    msg_ui, title="File upload error", easy_close=True, footer=None
                )
            )

    @reactive.effect
    @reactive.event(input.run, auto_run, ignore_none=False, ignore_init=True)
    def get_params_from_url():
        req(input.input_mode_params() in ("url", "server"))
        url = _source(input.input_mode_params(), input.url_params, input.server_params)
        req(len(url) > 0)
        if input.input_mode_params() == "server" and deployment.refuse_server_mode():
            return
        if deployment.refuse_local_path(url):
            return
        msg = None
        try:
            tmp_params = compute.get_class2d_helix_params_from_url(url)
        except Exception as e:
            msg = str(e)
            tmp_params = None
        params_raw.set(tmp_params)
        if tmp_params is None:
            if msg is None:
                msg = f"failed to read class2D parameters from {url}"
            msg_ui = ui.markdown(
                msg.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br><br>")
            )
            ui.modal_show(
                ui.modal(
                    msg_ui, title="File download error", easy_close=True, footer=None
                )
            )

    @reactive.effect
    @reactive.event(params_raw, input.split_axis_distance)
    def _split_distinct_filaments():
        raw = params_raw()
        if raw is None:
            params.set(None)
            return
        distance = input.split_axis_distance() or 0
        try:
            fixed, n0, n1 = compute.split_distinct_filaments(raw, distance)
        except (KeyError, ValueError) as e:
            logger.warning("Tube ids were not checked for distinct filaments: %s", e)
            fixed, n0, n1 = raw, 0, 0
        if n1 > n0:
            ui.notification_show(
                f"{n1 - n0:,} tube ids held more than one filament (segments over "
                f"{distance:g} \u00c5 from each other's axis) and were split: "
                f"{n1:,} filaments. The exported star file has the new ids.",
                duration=10,
            )
        params.set(fixed)

    def _source(mode, url_input, server_input):
        """The path or URL the mode reads from."""
        return (server_input() if mode == "server" else url_input()).strip()

    # Browse... in the server mode: pick the files on this computer
    if not deployment.is_cloud():
        helicon.shiny.file_picker_fill(
            "params_browse",
            "server_params",
            input,
            ("*.star", "*.cs"),
            "Select the Class2D parameter file (RELION star or cryoSPARC cs)",
        )
        helicon.shiny.file_picker_fill(
            "classes_browse",
            "server_classes",
            input,
            ("*.mrcs", "*.mrc"),
            "Select the Class2D class averages (mrcs or mrc)",
        )

    # A server file chosen or typed in one of the two server fields brings the
    # other file of the same 2D classification with it, when it is beside it.
    def _fill_companion(given, other_id, other_value):
        if deployment.is_cloud():  # no looking at the server's files
            return
        found = class2d_files.companion(given)
        if (
            found
            and found != other_value
            and class2d_files.needs_filling(other_value, given)
        ):
            ui.update_text(other_id, value=found)

    @reactive.effect
    @reactive.event(input.server_params)
    def _fill_classes_from_params():
        if input.input_mode_classes() == "server":
            _fill_companion(
                input.server_params(), "server_classes", input.server_classes()
            )

    @reactive.effect
    @reactive.event(input.server_classes)
    def _fill_params_from_classes():
        if input.input_mode_params() == "server":
            _fill_companion(
                input.server_classes(), "server_params", input.server_params()
            )

    # ── Build class gallery ──

    # created after the Run handlers, so that they see it change rather than
    # its first value
    auto_started = []

    @reactive.effect
    def _auto_run_at_start():
        # the reads wait for the inputs to reach the server, and the effect
        # goes again until they have; once is all it is for
        if auto_started:
            return
        modes = (input.input_mode_params(), input.input_mode_classes())
        both_urls = (
            all(m in ("url", "server") for m in modes)
            and bool(_source(modes[0], input.url_params, input.server_params))
            and bool(_source(modes[1], input.url_classes, input.server_classes))
        )
        auto_started.append(True)
        if both_urls:
            with reactive.isolate():
                auto_run.set(1)

    @reactive.effect
    @reactive.event(params, data_all, input.ignore_blank, input.sort_abundance)
    def get_displayed_class_images():
        req(params() is not None)
        req(data_all() is not None)
        data, apix = data_all()
        n = len(data)
        images = [data[i] for i in range(n)]
        image_size.set(max(images[0].shape))
        try:
            df = params()
            abundance.set(compute.get_class_abundance(df, n))
        except Exception as e:
            logger.error("Failed to get class abundance: %s", e)
            ui.modal_show(
                ui.modal(
                    "Failed to get class abundance from the provided Class2D parameter and image files. "
                    "Make sure that the two files are for the same Class2D job",
                    title="Information error",
                    easy_close=True,
                    footer=None,
                )
            )
            return
        display_seq_all = np.arange(n, dtype=int)
        if input.sort_abundance():
            display_seq_all = np.argsort(abundance())[::-1]
        if input.ignore_blank():
            included = []
            for i in range(n):
                image = images[display_seq_all[i]]
                if np.max(image) > np.min(image):
                    included.append(display_seq_all[i])
            images = [images[i] for i in included]
        else:
            included = display_seq_all
        image_labels = [f"{i+1}: {abundance()[i]:,d}" for i in included]
        displayed_class_ids.set(included)
        displayed_class_images.set(images)
        displayed_class_title.set(
            f"{len(included)}/{n} classes | {_counts_text(int(i) + 1 for i in included)}"
            f" | {images[0].shape[1]}x{images[0].shape[0]} pixels | {apix} \u00c5/pixel"
        )
        displayed_class_labels.set(image_labels)

    # ── Sidebar class selection ──

    @reactive.calc
    def _segment_filament():
        """Filament number of every row of the Class2D parameters."""
        p = params()
        req(p is not None)
        return (
            p.groupby(["rlnMicrographName", "rlnHelicalTubeID"], sort=False)
            .ngroup()
            .values
        )

    def _counts_text(class_numbers):
        """'N filaments | M segments' for the segments of these classes."""
        p = params()
        if p is None:
            return ""
        mask = p["rlnClassNumber"].astype(int).isin(list(class_numbers)).values
        n_fil = len(np.unique(_segment_filament()[mask]))
        return f"{n_fil:,} filaments | {int(mask.sum()):,} segments"

    def _gallery(gallery_id, images, labels, label="", selection=False, initial=None):
        """An image gallery, with the value of its input cleared on each draw."""
        parts = helicon.shiny.image_gallery(
            id=session.ns(gallery_id),
            label=reactive.value(label),
            images=reactive.value(images),
            image_labels=reactive.value(labels),
            image_size=reactive.value(128),
            initial_selected_indices=reactive.value(initial or []),
            enable_selection=selection,
            allow_multiple_selection=selection,
        )
        if parts is None:
            parts = ui.div()
        elif isinstance(parts, tuple):
            parts = ui.TagList(parts[0], *parts[1])
        if selection and not initial:
            # a redrawn gallery has nothing picked; the browser still holds the
            # old value of its input until told otherwise
            parts = ui.TagList(
                parts,
                ui.tags.script(
                    f"Shiny.setInputValue('{session.ns(gallery_id)}', [], "
                    "{priority: 'event'});"
                ),
            )
        return parts

    # Drawn even while hidden: Remove picked and the suggestions' Add buttons
    # change the selection by redrawing this gallery, and with the Parameters
    # tab in front of it a hidden output waited -- so the buttons did nothing
    # until the Inputs tab was shown again.
    @output(suspend_when_hidden=False)
    @render.ui
    def select_classes_gallery():
        gallery_version()
        return _gallery(
            "select_classes_inner",
            displayed_class_images(),
            displayed_class_labels(),
            label=displayed_class_title(),
            selection=True,
            initial=initial_selected_image_indices(),
        )

    def _set_selection(positions):
        """Select these classes (positions in the class gallery) in the sidebar."""
        initial_selected_image_indices.set(sorted(set(int(i) for i in positions)))
        with reactive.isolate():
            gallery_version.set(gallery_version() + 1)

    @reactive.effect
    @reactive.event(input.select_classes_inner)
    def update_selected_images():
        sel = input.select_classes_inner()
        if sel is None or len(sel) == 0:
            selected_images.set([])
            selected_image_labels.set([])
            return
        selected_images.set([displayed_class_images()[i] for i in sel])
        selected_image_labels.set([displayed_class_labels()[i] for i in sel])

    # ── pitch from the phases of the selected classes ──

    MAX_PHASE_CLASSES = 200
    phase_result = reactive.value(None)

    def _selected_class_numbers():
        sel = input.select_classes_inner() or []
        return [int(displayed_class_ids()[i]) + 1 for i in sel]

    @reactive.effect
    @reactive.event(params, input.select_classes_inner)
    def _clear_phase_result():
        # a result belongs to the selection it was computed from
        phase_result.set(None)

    counterpart_cache = {}

    def _counterpart_pairs(class_numbers, ids, images):
        """Pairs of the selected classes that are each other's 180-degree rotation.

        ``ids`` and ``images`` are the displayed class ids and images, read by
        the caller: this runs in a worker thread, which cannot read them.
        """
        key = tuple(sorted(class_numbers))
        if key not in counterpart_cache:
            position = {int(c): i for i, c in enumerate(ids)}
            at = [position[int(c) - 1] for c in class_numbers if int(c) - 1 in position]
            found = phase.pair_counterparts(images, at)
            counterpart_cache[key] = [
                dict(
                    a=int(ids[d["a"]]) + 1,
                    b=int(ids[d["b"]]) + 1,
                    corr=d["corr"],
                    dx=d["dx"],
                )
                for d in found
            ]
        return counterpart_cache[key]

    # The pitch estimate, the suggestions and the two reconstructions take
    # from seconds to minutes: they run in worker threads (background_task),
    # so the event loop every session shares stays free. Their inputs are read
    # in the effect that starts them, and a result is applied only if it still
    # belongs to the data and selection it was computed from.

    def _phase_work(job, progress):
        counterparts = None
        if job["merge"]:
            progress.set(
                0, message="pairing classes that are turned copies of each other"
            )
            counterparts = _counterpart_pairs(
                job["class_ids"], job["ids"], job["images"]
            )
            progress.inc(1, message=f"{len(counterparts)} pair(s) of classes")
        return phase.analyze(
            job["params"],
            class_ids=job["class_ids"],
            n_boot=job["n_boot"],
            progress=lambda msg: progress.inc(1, message=msg),
            counterparts=counterparts,
            image_apix=job["apix"],
            max_sep=job["max_repeat"],
        )

    def _phase_apply(job, result):
        # a result belongs to the selection it was computed from
        if (
            params() is not job["params"]
            or _selected_class_numbers() != job["class_ids"]
        ):
            return
        phase_result.set(result)
        spans = result.filaments["span"]
        if len(spans):
            top = int(np.ceil(float(spans.max()) / 10.0) * 10)
            ui.update_slider("length_range", min=0, max=top, value=(0, top))
        fil = result.filaments
        pitches = fil["pitch"][fil["fitted"]]
        if len(pitches):
            lo_p = int(np.floor(float(pitches.min())))
            hi_p = int(np.ceil(float(pitches.max())))
            # centred on the pooled repeat (the green line), wide enough for
            # the 40% of the filaments closest to it
            half = float(np.percentile(np.abs(pitches - result.period), 40))
            low, high = result.period - half, result.period + half
            ui.update_slider(
                "pitch_band",
                min=lo_p,
                max=max(hi_p, lo_p + 1),
                value=(round(float(low)), round(float(high))),
            )

    def _phase_error(job, e):
        ui.modal_show(
            ui.modal(
                f"The class-azimuth pitch estimate failed: {e}",
                title="Pitch estimate error",
                easy_close=True,
                footer=None,
            )
        )

    phase_task = helicon.shiny.background_task(
        "phase_run",
        _phase_work,
        _phase_apply,
        _phase_error,
        progress_max=8,
        session=session,
        label="Class-azimuth pitch estimate",
    )

    def _max_repeat():
        """The longest repeat the pitch estimate searches, in A.

        One repeat turns the helix by 360 degrees at the C1 twist, the largest
        of the twists a repeat allows: 360 * rise / |smallest twist|.
        """
        rise = helicon.shiny.clamp_number(input.rise(), 4.75, 0.01, 1000.0, float)
        twist = helicon.shiny.clamp_number(
            abs(input.min_twist() or 0.3), 0.3, 0.05, 30.0, float
        )
        # the scan starts at 100 A
        return max(200.0, 360.0 * rise / twist)

    @reactive.effect
    @reactive.event(input.phase_run)
    def run_phase_pitch():
        req(params() is not None)
        class_ids = _selected_class_numbers()
        if len(class_ids) < MIN_PHASE_CLASSES:
            ui.modal_show(
                ui.modal(
                    "Select at least three classes of one helical type: the method "
                    "places the selected classes around a ring, and fewer than "
                    "three do not define one.",
                    title="Too few classes selected",
                    easy_close=True,
                    footer=None,
                )
            )
            return
        if len(class_ids) > MAX_PHASE_CLASSES:
            ui.modal_show(
                ui.modal(
                    f"{len(class_ids)} classes are selected; the method takes at "
                    f"most {MAX_PHASE_CLASSES}, since its cost grows with the "
                    "square of the number of classes. Remove the junk classes and "
                    "those of other helical types from the computation.",
                    title="Too many classes selected",
                    easy_close=True,
                    footer=None,
                )
            )
            return
        merge = bool(input.merge_counterparts())
        phase_task.invoke(
            dict(
                params=params(),
                class_ids=class_ids,
                # the UI's bounds, enforced here too: the browser's are not
                # binding
                n_boot=helicon.shiny.clamp_number(input.phase_n_boot(), 20, 2, 200),
                merge=merge,
                max_repeat=_max_repeat(),
                ids=list(displayed_class_ids()) if merge else None,
                images=list(displayed_class_images()) if merge else None,
                apix=float(data_all()[1]),
            )
        )

    # ── the classes in the computation, and suggestions for more ──

    @render.ui
    def accepted_gallery():
        numbers = _selected_class_numbers()
        r = phase_result()
        # the classes to take out come up picked, ready to remove, each
        # labelled with the reason
        poor = _poorly_fitting(r) if r is not None else {}
        labels = [
            f"{lab} \u2717 {_SHORT_REASON.get(poor[c], poor[c])}" if c in poor else lab
            for lab, c in zip(selected_image_labels(), numbers)
        ]
        return _gallery(
            "accepted_pick",
            selected_images(),
            labels,
            label=f"{len(selected_images())} class(es) | "
            + _counts_text(numbers)
            + (f" | {len(poor)} picked to remove" if poor else ""),
            selection=True,
            initial=[i for i, c in enumerate(numbers) if c in poor],
        )

    @reactive.effect
    @reactive.event(input.accepted_remove)
    def remove_picked_classes():
        sel = list(input.select_classes_inner() or [])
        picked = {sel[i] for i in (input.accepted_pick() or []) if i < len(sel)}
        if not picked:
            ui.notification_show("Pick the classes to remove first.", duration=4)
            return
        _set_selection([i for i in sel if i not in picked])

    # at most this many unselected classes, the most populated, are compared by
    # shape: the cost is one registration per (selected, candidate) pair
    MAX_COUNTERPART_CANDIDATES = 60

    # suggestions: {"seeds": display positions they were computed from, "items":
    # [{"pos": display position, "text": label}], "n_candidates": int}
    suggestions = reactive.value(None)

    @reactive.effect
    @reactive.event(params, input.select_classes_inner)
    def _update_suggestions():
        r = suggestions()
        if r is None:
            return
        sel = set(input.select_classes_inner() or [])
        if not r["seeds"] <= sel:
            # a class the suggestions rest on was removed
            suggestions.set(None)
        elif any(d["pos"] in sel for d in r["items"]):
            suggestions.set(
                dict(r, items=[d for d in r["items"] if d["pos"] not in sel])
            )

    def _suggest_work(job, progress):
        pos_of = {c: i for i, c in enumerate(job["ids"])}
        found = {}  # display position -> label parts
        progress.set(0.05, message="finding classes used by the same filaments")
        for d in phase.suggest_expansion(
            job["params"],
            [c + 1 for c in job["selected"]],
            [c + 1 for c in job["others"]],
        ):
            c = d["candidate"] - 1
            found.setdefault(pos_of[c], {}).update(
                share=d["share"], baseline=d["baseline"], filaments=d["filaments"]
            )
        if job["data"] is not None:
            ab = job["abundance"]
            by_size = sorted(job["others"], key=lambda c: -ab[c] if c < len(ab) else 0)
            progress.set(0.1, message="matching class averages turned by 180\u00b0")
            for d in phase.suggest_counterparts(
                job["data"],
                job["selected"],
                by_size[:MAX_COUNTERPART_CANDIDATES],
                progress=lambda k, n: progress.set(0.1 + 0.9 * k / max(n, 1)),
            ):
                found.setdefault(pos_of[d["candidate"]], {}).update(
                    corr=d["corr"], of=d["selected"] + 1
                )
        r = job["phase_result"]
        if r is not None and found:
            # the fit each would get on the ring of the current estimate, on
            # the same scale as the ranked fits
            progress.set(0.95, message="placing the suggestions on the ring")
            fits = phase.candidate_fit(
                job["params"], r, [job["ids"][pos] + 1 for pos in found]
            )
            for pos, d in found.items():
                d["fit"] = fits.get(job["ids"][pos] + 1, float("nan"))
        return found

    def _suggest_apply(job, found):
        # suggestions for a selection or data set no longer shown are dropped
        if params() is not job["params"] or list(
            input.select_classes_inner() or []
        ) != list(job["sel_disp"]):
            return
        labels = displayed_class_labels()
        items = []

        def best_fit_first(item):
            fit = item[1].get("fit", float("nan"))
            return (0, -fit) if np.isfinite(fit) else (1, item[0])

        for pos, d in sorted(found.items(), key=best_fit_first):
            why = []
            if np.isfinite(d.get("fit", float("nan"))):
                why.append(f"fit {d['fit']:.2f}")
            if "corr" in d:
                why.append(f"\u21bb{d['corr']:.2f}")
            if "share" in d:
                why.append(f"\u25cf{100 * d['share']:.0f}%")
            items.append(
                dict(
                    pos=pos,
                    text=f"{labels[pos]}  {' '.join(why)}",
                    fit=d.get("fit", float("nan")),
                )
            )
        suggestions.set(
            dict(
                seeds=set(job["sel_disp"]),
                items=items,
                n_candidates=len(job["others"]),
            )
        )

    def _suggest_error(job, e):
        ui.modal_show(
            ui.modal(
                f"The suggestions could not be computed: {e}",
                title="Suggestion error",
                easy_close=True,
                footer=None,
            )
        )

    suggest_task = helicon.shiny.background_task(
        "suggest_run",
        _suggest_work,
        _suggest_apply,
        _suggest_error,
        session=session,
        label="Class suggestions",
    )

    @reactive.effect
    @reactive.event(input.suggest_run)
    def run_suggestions():
        req(params() is not None)
        sel_disp = list(input.select_classes_inner() or [])
        if not sel_disp:
            ui.modal_show(
                ui.modal(
                    "Select the classes of one type first; other classes of that "
                    "type are then looked for among the rest.",
                    title="No classes selected",
                    easy_close=True,
                    footer=None,
                )
            )
            return
        ids = [int(c) for c in displayed_class_ids()]
        selected = [ids[i] for i in sel_disp]
        has_data = data_all() is not None and data_all()[0] is not None
        suggest_task.invoke(
            dict(
                params=params(),
                sel_disp=sel_disp,
                ids=ids,
                selected=selected,
                others=[c for c in ids if c not in set(selected)],
                data=data_all()[0] if has_data else None,
                abundance=abundance() if has_data else None,
                # the current estimate, to place the suggestions on its ring
                phase_result=phase_result(),
            )
        )

    @render.ui
    def suggested_gallery():
        r = suggestions()
        if r is None:
            return None
        items, n_below = _shown_suggestions()
        below = (
            f" ({n_below} more would fit below the fit threshold)" if n_below else ""
        )
        if not items:
            return ui.tags.small(
                f"None of the {r['n_candidates']} other classes stands out" + below,
                class_="text-muted",
            )
        images = displayed_class_images()
        return _gallery(
            "suggested_pick",
            [images[d["pos"]] for d in items],
            [d["text"] for d in items],
            label=f"{len(items)} suggested | "
            + _counts_text(int(displayed_class_ids()[d["pos"]]) + 1 for d in items)
            + below,
            selection=True,
        )

    @reactive.calc
    def _shown_suggestions():
        """The suggestions to offer, and how many are held back.

        A class whose predicted fit is below the Fit threshold would be picked
        to remove as soon as it was added -- it would lower the ring's
        agreement rather than add to it -- so it is not offered; moving the
        threshold brings it back. One without a predicted fit (no pairs with
        the classes of the fit) is offered, last.
        """
        r = suggestions()
        if r is None:
            return [], 0
        threshold = input.fit_threshold()
        if threshold is None:
            return list(r["items"]), 0
        shown = [
            d
            for d in r["items"]
            if not np.isfinite(d.get("fit", np.nan)) or d["fit"] >= float(threshold)
        ]
        return shown, len(r["items"]) - len(shown)

    @render.ui
    def suggested_buttons():
        if not _shown_suggestions()[0]:
            return None
        return ui.layout_columns(
            ui.input_action_button(
                "suggested_add_all",
                "Add all",
                class_="btn-sm btn-outline-primary",
            ),
            ui.input_action_button(
                "suggested_add_picked",
                "Add picked",
                class_="btn-sm btn-outline-primary",
            ),
            col_widths=(6, 6),
            fill=False,
        )

    def _add_suggested(indices):
        items = _shown_suggestions()[0]  # as the gallery shows them
        req(items)
        add = [items[i]["pos"] for i in indices if i < len(items)]
        if not add:
            ui.notification_show("Pick the suggested classes to add first.", duration=4)
            return
        _set_selection(list(input.select_classes_inner() or []) + add)

    @reactive.effect
    @reactive.event(input.suggested_add_all)
    def add_all_suggested():
        _add_suggested(range(len(_shown_suggestions()[0])))

    @reactive.effect
    @reactive.event(input.suggested_add_picked)
    def add_picked_suggested():
        _add_suggested(input.suggested_pick() or [])

    def _poorly_fitting(r):
        """Selected classes to take out, with the reason for each."""
        threshold = input.fit_threshold()
        return phase.diagnose_classes(
            r, threshold=None if threshold is None else float(threshold)
        )

    first_result = [True]

    @reactive.effect
    @reactive.event(phase_result)
    def _suggest_fit_threshold():
        # each new result starts from where its own ranked fits break; a
        # bookmarked threshold is kept for the first result (the slider is
        # hidden until there is one, so only a bookmark can have moved it)
        r = phase_result()
        if r is None:
            return
        if first_result[0]:
            first_result[0] = False
            default = BOOKMARK_DEFAULTS["fit_cut"][1]
            if input.fit_threshold() not in (None, default):
                return
        value = phase.fit_threshold(r.class_fit)
        if value is not None:
            ui.update_slider("fit_threshold", value=round(value, 2))

    MIN_PHASE_CLASSES = 3

    # whether a ring has been drawn for the data loaded now
    ring_drawn = reactive.value(False)

    @reactive.effect
    @reactive.event(params)
    def _forget_ring():
        ring_drawn.set(False)

    @reactive.effect
    def _remember_ring():
        if phase_result() is not None:
            ring_drawn.set(True)

    @render.ui
    def results_visibility():
        # Both start hidden in the page itself (_START_HIDDEN_CSS): hiding them
        # from here instead left them on screen while the tab started, before
        # this had run. This only shows them, once there is something to show.
        shown = []
        if phase_result() is not None:
            shown.append(f"#{_RESULTS_BOX} {{ display: block; }}")
        if ring_drawn():
            # a card's id is namespaced like an input's
            shown.append(f"#{session.ns(_SUGGESTIONS_BOX)} {{ display: flex; }}")
        if not shown:
            return None
        return ui.tags.style("\n".join(shown))

    @reactive.effect
    def _enable_phase_run():
        n = len(input.select_classes_inner() or [])
        if phase_task.status() == "running":
            return  # the task button shows it busy
        message = {"id": session.ns("phase_run"), "disabled": n < MIN_PHASE_CLASSES}

        # sent after the flush: the task button's own "ready" update, which
        # re-enables the button when a run ends, goes out with the flush
        async def _send():
            await session.send_custom_message("helicon-set-disabled", message)

        session.on_flushed(_send, once=True)

    @render.ui
    def phase_summary():
        r = phase_result()
        if r is None:
            n = len(input.select_classes_inner() or [])
            need = (
                ""
                if n >= MIN_PHASE_CLASSES
                else f"; select at least {MIN_PHASE_CLASSES} to estimate the pitch"
            )
            return _tip(
                ui.tags.small(f"{n} class(es) selected{need}", class_="text-muted"),
                "Select at least three classes of one helical type (shift-click to "
                "add) or add suggested classes, then press Estimate pitch. The result "
                "is cleared whenever the selection changes.",
            )
        rise = input.rise()
        het = r.heterogeneity
        n_sel = len(r.class_ids) + len(r.merged_classes)
        merged = (
            f" ({len(r.merged_classes)} merged -> {len(r.class_ids)} unique classes)"
            if r.merged_classes
            else ""
        )
        lines = [
            ui.p(
                ui.tags.b(f"Repeat {r.period:.1f} \u00b1 {r.period_sd:.1f} \u00c5"),
                f" \u00b7 {n_sel} classes{merged} \u00b7 {r.n_filaments:,} filaments",
                _info(
                    f"Raw peak {r.period_raw:.1f} \u00c5, corrected by "
                    f"{-100 * r.calibration['bias']:+.2f}% for this dataset's segment "
                    f"geometry; {r.n_pairs:,} segment pairs; positions "
                    + (
                        "along the refined filament path."
                        if r.refined_path
                        else "from the track length (no coordinates/origins)."
                    )
                    + (
                        f" {len(r.merged_classes)} classes were merged into their "
                        "180\u00b0-rotated counterparts."
                        if r.merged_classes
                        else ""
                    )
                ),
                style="margin-bottom: 2px;",
            )
        ]
        if rise is not None and rise > 0:
            twists = " \u00b7 ".join(
                f"C{c} {360.0 * rise / (r.period * c):.3f}\u00b0" for c in (1, 2, 3, 4)
            )
            lines.append(
                ui.p(
                    f"Twist: {twists}",
                    _info(
                        f"If the repeat is the pitch divided by C, at rise {rise} \u00c5 "
                        f"({r.period / rise:.1f} subunits per repeat)"
                    ),
                    style="margin-bottom: 2px;",
                )
            )
        if het["n"] >= 10:
            lo, hi = het["sd_between_ci"]
            lines.append(
                ui.p(
                    f"Filament spread {het['sd_between']:.0f} \u00c5 "
                    f"(95% CI {lo:.0f}\u2013{hi:.0f})",
                    _info(
                        f"Split-half test on {het['n']} filaments \u2265 1.2 repeats "
                        f"long; correlation between halves {het['corr']:.2f}"
                    ),
                    style="margin-bottom: 2px;",
                )
            )
        warnings = []
        if r.double_ratio >= phase.DOUBLE_RATIO_WARN:
            warnings.append(
                (
                    f"2\u00d7 ({2 * r.period:.0f} \u00c5) fits almost as well",
                    "The selected classes may cover only half of the views, which "
                    "places them around a ring half the true size. Add the missing "
                    "views -- often the 180\u00b0-rotated versions of the selected "
                    "classes -- or read the repeat as twice this.",
                )
            )
        if r.support < phase.MIN_SUPPORT:
            warnings.append(
                (
                    "Too few pairs beyond one repeat",
                    "Almost no segment pairs are further apart than this distance, so "
                    "it cannot be told apart from half the true value: the pitch may "
                    "be twice this.",
                )
            )
        if r.phase_gap > 120:
            warnings.append(
                (
                    f"{r.phase_gap:.0f}\u00b0 of the ring empty",
                    "Some views of this type are not selected, and the estimate rests "
                    "on the ones that are.",
                )
            )
        poor = _poorly_fitting(r)
        if poor:
            by_reason = {}
            for c, why in poor.items():
                by_reason.setdefault(why, []).append(c)
            warnings.append(
                (
                    f"{len(poor)} class(es) to remove: "
                    + "; ".join(
                        f"{', '.join(map(str, cs))} ({why})"
                        for why, cs in by_reason.items()
                    ),
                    "Judged from the segments alone, not the look of the averages "
                    "(numbers as before the colon in the labels). 'No better than "
                    "chance': set against segments a quarter repeat or more away, "
                    "the class's segments do not sit at one azimuth on the ring "
                    "(dim on the ring plot) -- junk, or another type. 'Too few "
                    "informative segment pairs': fewer than 5% of a typical "
                    "class's pairs a quarter repeat or more apart, too little to "
                    "tell either way. 'Below the threshold': a fit under the Fit "
                    "threshold set below the ranked fits. A 180\u00b0 copy goes "
                    "with the class it was merged into. They are picked in the "
                    "Selected classes gallery; press Remove picked to drop them.",
                )
            )
        if len(r.groups) > 1:
            parts = "; ".join(
                f"{len(g)} classes \u2192 {p:.1f} \u00c5"
                for g, p in zip(r.groups, r.group_periods)
            )
            warnings.append(
                (
                    f"{len(r.groups)} groups of classes",
                    "The selected classes fall into groups that rarely share "
                    f"filaments -- probably different helical types: {parts}. The "
                    "repeat above mixes them.",
                )
            )
        if het["n"] < 10:
            warnings.append(
                (
                    "Too few long filaments for the spread",
                    f"Only {het['n']} filaments are at least 1.2 repeats long.",
                )
            )
        for short, long in warnings:
            lines.append(
                ui.p(
                    "\u26a0\ufe0f " + short,
                    _info(long),
                    style="margin-bottom: 2px; color: #b35c00;",
                )
            )
        return ui.div(*lines)

    def _add_marker(fig, x, y0, y1):
        """A vertical line drawn as a trace, not a shape: the page script that
        gives the HelicalPitch histograms their crosshairs takes over the shapes of
        every plot on the page, and would move a shape with the mouse."""
        import plotly.graph_objects as go

        fig.add_trace(
            go.Scatter(
                x=[x, x],
                y=[y0, y1],
                mode="lines",
                line=dict(color="green", width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    @render.ui
    def phase_scan_plot():
        r = phase_result()
        req(r is not None)
        import plotly.graph_objects as go

        rise = input.rise()
        periods = np.asarray(r.scan_periods, dtype=float)
        hover = [
            f"Repeat: {p:.0f} \u00c5<br>Score: {v:.3f}"
            for p, v in zip(periods, r.scan_scores)
        ]
        if rise is not None and rise > 0:
            # the same read-out as the HelicalPitch histogram's hover text
            hover = [
                h
                + "".join(
                    f"<br>Twist for C{c}: {360.0 / (p * c / rise):.2f}\u00b0"
                    for c in (1, 2, 3, 4)
                )
                for h, p in zip(hover, periods)
            ]
        fig = go.Figure(
            go.Scatter(
                x=periods,
                y=r.scan_scores,
                mode="lines",
                name="score",
                text=hover,
                hoverinfo="text",
                showlegend=False,
            )
        )
        _add_marker(fig, r.period_raw, 0.0, float(np.max(r.scan_scores)))
        fig.update_layout(
            template="plotly_white",
            title_text="Repeat scan",
            title_x=0.5,
            title_font=dict(size=12),
            xaxis_title="Repeat distance (\u00c5)",
            yaxis_title="Top eigenvalue",
            hovermode="closest",
            hoverlabel=dict(bgcolor="white", font_size=12),
            margin=dict(t=40, b=50, l=50, r=20),
        )
        return _fig_to_html(fig)

    @render.ui
    def phase_circle_plot():
        r = phase_result()
        req(r is not None)
        import plotly.graph_objects as go

        w = np.sqrt(np.maximum(r.class_weight, 0))
        size = 6 + 24 * w / max(w.max(), 1e-12)
        fit = np.nan_to_num(np.asarray(r.class_fit, dtype=float), nan=0.0)
        fig = go.Figure(
            go.Scatterpolar(
                theta=np.rad2deg(r.phases) % 360.0,
                r=np.ones(len(r.phases)),
                mode="markers+text" if input.ring_labels() else "markers",
                text=[str(int(c)) for c in r.class_ids],
                textposition="top center",
                customdata=fit,
                marker=dict(
                    size=size,
                    color=fit,
                    colorscale="Viridis",
                    # the full 0-1 range, so the colours mean the same on
                    # every data set
                    cmin=0.0,
                    cmax=1.0,
                    colorbar=dict(title="fit", thickness=10, len=0.6),
                ),
                hovertemplate=(
                    "class %{text}<br>azimuth %{theta:.0f}\u00b0"
                    "<br>fit %{customdata:.2f}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            template="plotly_white",
            title_text="Class azimuths (one repeat)",
            title_x=0.5,
            title_font=dict(size=12),
            polar=dict(radialaxis=dict(visible=False, range=[0, 1.3])),
            showlegend=False,
            margin=dict(t=40, b=20, l=20, r=20),
        )
        return _fig_to_html(fig)

    @render.ui
    def class_fit_rank_plot():
        """The class fits of the ring, ranked: how they are spread.

        The ring shows where each class sits; this shows the fits themselves,
        best first, on the same fixed scale and colours, each marker's area
        following the class's number of segments, the classes to take out
        outlined in red (a class without a fit as a cross at 0) and the line
        below which a fit is taken as chance.
        """
        r = phase_result()
        req(r is not None)
        import plotly.graph_objects as go

        ids = [int(c) for c in r.class_ids]
        fit = np.asarray(r.class_fit, dtype=float)
        members = np.asarray(
            r.class_count if r.class_count is not None else r.class_weight,
            dtype=float,
        )
        order = np.argsort(-np.nan_to_num(fit, nan=-1.0), kind="stable")
        f = np.nan_to_num(fit[order], nan=0.0)
        m = members[order]
        # area by the number of segments
        size = 6 + 24 * np.sqrt(m) / max(float(np.sqrt(m).max()), 1e-12)
        poor = _poorly_fitting(r)
        reasons = [poor.get(ids[i], "") for i in order]
        text = [str(ids[i]) for i in order]
        no_fit = ~np.isfinite(fit[order])
        hover = [
            f"class {c}<br>"
            + ("no fit" if missing else f"fit {v:.2f}")
            + f"<br>{int(n):,} segments"
            + (f"<br>\u2717 {_SHORT_REASON.get(why, why)}" if why else "")
            for c, v, n, why, missing in zip(text, f, m, reasons, no_fit)
        ]
        rank = np.arange(1, len(order) + 1)
        fig = go.Figure(
            go.Scatter(
                x=rank,
                y=f,
                mode="markers+text" if input.ring_labels() else "markers",
                text=text,
                textposition="top center",
                hovertext=hover,
                hoverinfo="text",
                marker=dict(
                    size=size,
                    color=f,
                    colorscale="Viridis",
                    cmin=0.0,
                    cmax=1.0,
                    # a class without a fit is drawn at 0 as a cross
                    symbol=["x" if missing else "circle" for missing in no_fit],
                    line=dict(
                        color=[
                            "#d62728" if why else "rgba(0,0,0,0.3)" for why in reasons
                        ],
                        width=[2.5 if why else 0.5 for why in reasons],
                    ),
                ),
                showlegend=False,
            )
        )
        cut = phase.poorly_fitting_cut(fit)
        if cut is not None:
            # the chance level: a class below it is taken out. Drawn over the
            # markers and the grid, so it is not lost among them
            fig.add_hline(
                y=cut,
                line=dict(color="#b2182b", width=2, dash="dash"),
                layer="above",
                annotation_text=f"\u2248 junk ({cut:.2f})",
                # at the best-fitting end, clear of the poor classes
                annotation_position="top left",
                annotation_font=dict(size=10, color="#b2182b"),
            )
        threshold = input.fit_threshold()
        if threshold is not None:
            fig.add_hline(
                y=float(threshold),
                line=dict(color="#ff7f0e", width=2),
                layer="above",
                annotation_text=f"threshold ({float(threshold):.2f})",
                annotation_position="top left",
                annotation_font=dict(size=10, color="#d95f02"),
            )
        fig.update_layout(
            template="plotly_white",
            title_text="Class fits, ranked",
            title_x=0.5,
            title_font=dict(size=12),
            xaxis=dict(title="Rank", range=[0.3, len(order) + 0.7]),
            # the same fixed scale as the ring's colours
            # fixed at the top, as the ring's colours; down to the worst fit,
            # which can be below 0 (a class against the ring)
            yaxis=dict(title="Fit", range=[min(-0.05, float(f.min()) - 0.07), 1.05]),
            margin=dict(t=40, b=50, l=50, r=20),
            hovermode="closest",
        )
        return _fig_to_html(fig)

    def _length_range():
        lo, hi = input.length_range() or (None, None)
        return lo, hi

    def _pitch_band():
        lo, hi = input.pitch_band() or (None, None)
        return lo, hi

    @render.ui
    def filament_pitch_plot():
        r = phase_result()
        req(r is not None)
        lo, hi = _length_range()
        table = r.filaments
        table = table[(table["span"] >= lo) & (table["span"] <= hi)]
        n_all = len(table)
        table = table[table["fitted"]]
        pitches = table["pitch"].values
        spans = table["span"].values
        req(len(pitches) > 0)
        import plotly.graph_objects as go

        counts, edges = np.histogram(pitches, bins=40)
        top = float(counts.max()) * 1.05
        fig = go.Figure(
            go.Histogram(
                x=pitches,
                # the same bins as counted above, so the band reaches the top
                xbins=dict(start=edges[0], end=edges[-1], size=edges[1] - edges[0]),
                name="filaments",
                showlegend=False,
            )
        )
        low, high = _pitch_band()
        if low is not None and high is not None and high > low:
            fig.add_trace(
                go.Scatter(
                    x=[low, high, high, low, low],
                    y=[0, 0, top, top, 0],
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(255,165,0,0.2)",
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        _add_marker(fig, r.period, 0.0, top)
        fig.update_layout(
            template="plotly_white",
            title_text=(
                # two lines: it shares its row with the ranked fits
                f"Per-filament repeat \u00b7 {len(pitches)} filaments, "
                f"{spans.min():,.0f}\u2013{spans.max():,.0f} \u00c5 long"
                f"<br><sup>+{n_all - len(pitches)} without a clear repeat</sup>"
            ),
            title_x=0.5,
            title_font=dict(size=12),
            xaxis_title="Repeat distance (Å)",
            yaxis_title="# of filaments",
            margin=dict(t=62, b=50, l=50, r=20),
        )
        return _fig_to_html(fig)

    def _write_star(out):
        import starfile

        yield starfile.to_string(
            dict(optics=out.attrs["optics"], particles=out)
            if "optics" in out.attrs
            else dict(particles=out)
        )

    @reactive.calc
    def _angles():
        r = phase_result()
        req(r is not None)
        fold = helicon.shiny.clamp_number(input.rot_fold(), 1, 1, 12)
        return phase.segment_angles(params(), r, fold=fold)

    def _download_label(subset, prefix="Download"):
        n_fil = subset.groupby(["rlnMicrographName", "rlnHelicalTubeID"]).ngroups
        return f"{prefix} {n_fil:,} filaments / {len(subset):,} segments (.star)"

    @render.ui
    def pitch_band_download():
        r = phase_result()
        req(r is not None)
        low, high = _pitch_band()
        lo, hi = _length_range()
        req(low is not None and high is not None and high >= low)
        subset = phase.select_segments(
            params(),
            r.filaments,
            low,
            high,
            min_length=lo,
            max_length=hi,
            class_numbers=_selected_class_numbers(),
        )
        star_name = f"helices_pitch_{low:.0f}-{high:.0f}.star"
        csym = _imposed_csym()
        band_ui = render.download(
            label=_download_label(subset, "Download"),
            filename=star_name,
        )

        @band_ui
        def download_pitch_band():
            low, high = _pitch_band()
            lo, hi = _length_range()
            yield from _write_star(
                phase.select_segments(
                    params(),
                    r.filaments,
                    low,
                    high,
                    min_length=lo,
                    max_length=hi,
                    angles=_angles(),
                    class_numbers=_selected_class_numbers(),
                )
            )

        return ui.tooltip(
            band_ui,
            ui.HTML(
                "A RELION star file of the segments in the selected classes, with "
                'their estimated <span style="font-family: monospace; color: #ffd479;">rlnAngleRot</span>, <span style="font-family: monospace; color: #ffd479;">rlnAngleTilt</span> '
                '(90), <span style="font-family: monospace; color: #ffd479;">rlnAnglePsi</span> and origins, and each filament\'s '
                "pitch, length and direction score. It can be used directly to "
                "reconstruct a 3D map, without helical symmetry, for example (the "
                "rlnAngleRot follow the C symmetry set below; --sym imposes it when "
                "Impose C is ticked in Parameters):<br>"
                f'<span style="font-family: monospace; color: #ffd479;">relion_reconstruct --i {star_name} '
                f"--o map.mrc --ctf --sym c{csym}</span><br>"
                "and the map serves as a starting reference for refinement."
            ),
            placement="top",
        )

    # ── a 3D map from the segments, by relion_reconstruct ──

    def _imposed_csym():
        """The C symmetry to impose on a map: the tab's, when Impose C is ticked."""
        return (
            helicon.shiny.clamp_number(input.rot_fold(), 1, 1, 12)
            if input.map_csym()
            else 1
        )

    relion_map = reactive.value(None)

    @reactive.effect
    @reactive.event(phase_result)
    def _clear_relion_map():
        relion_map.set(None)

    def _relion_project_guess():
        """The directory the segments' image paths resolve from, if it can be found."""
        if deployment.is_cloud():  # no looking at the server's files
            return ""
        p = params()
        if p is None or "rlnImageName" not in p:
            return ""
        image = str(p["rlnImageName"].iloc[0]).split("@")[-1]
        if Path(image).is_absolute():
            return str(Path(image).parent) if Path(image).exists() else ""
        # the parameter file on this computer: chosen in the server mode, or a
        # path (or file:// URL) typed in the url mode
        mode = input.input_mode_params()
        found = class2d_files.project_folder(
            _source(mode, input.url_params, input.server_params), image
        )
        if found:
            return found
        for folder in [Path.cwd(), *Path.cwd().parents]:
            if (folder / image).exists():
                return str(folder)
        return ""

    @render.ui
    def relion_ui():
        r = phase_result()
        req(r is not None)
        if deployment.is_cloud():
            # a hosted copy runs no external programs on visitors' data: the
            # reconstruction would need a RELION project folder on the server
            # and minutes of the server's CPUs per visitor
            return _tip(
                ui.tags.small(
                    "relion_reconstruct: not available on the hosted web site",
                    class_="text-muted",
                ),
                "Download the star file and run relion_reconstruct where RELION is "
                "installed, or install helicon and run this tab locally "
                "(https://jianglab.science.psu.edu/helicon/).",
            )
        exe = relion.find_relion_reconstruct()
        if exe is None:
            return _tip(
                ui.tags.small("relion_reconstruct: not available", class_="text-muted"),
                "relion_reconstruct is not on this server's PATH; download the star "
                "file and reconstruct it where RELION is installed.",
            )
        return ui.layout_columns(
            ui.input_text(
                "relion_project_dir",
                ui.span(
                    "RELION project",
                    _info("The directory the image paths in the star file start from"),
                ),
                value=_relion_project_guess(),
                width="100%",
            ),
            _tip(
                ui.input_task_button(
                    "relion_run",
                    "relion_reconstruct",
                    style="width: 100%;",
                ),
                "Reconstruct the segments in the selected ranges with their angles "
                "and origins: no helical symmetry, no C symmetry unless Impose C is "
                "ticked in Parameters (as for the map from the class averages), and "
                "CTF correction when the star file has it. "
                + (
                    "Runs relion_reconstruct_mpi, with as many MPI processes as "
                    "the free CPUs and memory allow."
                    if relion.find_relion_reconstruct_mpi()
                    else "Runs relion_reconstruct with threads (no MPI version found)."
                ),
            ),
            col_widths=(7, 5),
            style="align-items: flex-end; margin-top: 8px;",
            fill=False,
        )

    def _relion_work(job, progress):
        progress.set(
            0,
            message=f"relion_reconstruct: {len(job['subset']):,} segments",
            detail="this can take a few minutes",
        )
        return relion.reconstruct(
            job["subset"],
            job["project_dir"],
            cpu=max(1, int(helicon.available_cpu())),
            csym=job["csym"],
        )

    def _relion_apply(job, result):
        # a map from segments of an older pitch result is not shown
        if phase_result() is not job["phase_result"]:
            return
        relion_map.set(result)

    def _relion_error(job, e):
        ui.modal_show(
            ui.modal(
                ui.tags.pre(str(e)[-3000:]),
                title="relion_reconstruct failed",
                easy_close=True,
                footer=None,
            )
        )

    relion_task = helicon.shiny.background_task(
        "relion_run",
        _relion_work,
        _relion_apply,
        _relion_error,
        session=session,
        label="relion_reconstruct",
    )

    @reactive.effect
    @reactive.event(input.relion_run)
    def run_relion():
        r = phase_result()
        req(r is not None)
        # the RELION project folder is a free-text path on the server
        if deployment.refuse_server_mode():
            return
        low, high = _pitch_band()
        lo, hi = _length_range()
        subset = phase.select_segments(
            params(),
            r.filaments,
            low,
            high,
            min_length=lo,
            max_length=hi,
            angles=_angles(),
            class_numbers=_selected_class_numbers(),
        )
        if not len(subset):
            ui.notification_show("No segment is in the selected ranges.", duration=5)
            return
        relion_task.invoke(
            dict(
                phase_result=r,
                subset=subset,
                project_dir=input.relion_project_dir(),
                csym=_imposed_csym(),
            )
        )

    @render.ui
    def relion_display():
        m = relion_map()
        req(m is not None)
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        volume = m["volume"]
        nz = volume.shape[0]
        side = volume.sum(axis=1)  # z rows, x columns
        z_view = volume[nz // 2]
        fig = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.6, 0.4],
            horizontal_spacing=0.04,
            subplot_titles=("Side projection", "Central z section"),
        )
        for image, col in ((side.T, 1), (z_view, 2)):
            fig.add_trace(
                go.Heatmap(
                    z=image, colorscale="gray", showscale=False, hoverinfo="skip"
                ),
                row=1,
                col=col,
            )
            fig.update_yaxes(
                autorange="reversed",
                scaleanchor=f"x{'' if col == 1 else col}",
                showticklabels=False,
                row=1,
                col=col,
            )
            fig.update_xaxes(showticklabels=False, row=1, col=col)
        fig.update_layout(
            template="plotly_white",
            title_text=(
                "relion_reconstruct"
                + (f" (MPI \u00d7{m['mpi']})" if m.get("mpi") else "")
                + f" \u00b7 {m['n_segments']:,} segments \u00b7 "
                f"{volume.shape[-1]}\u00b3 at {m['apix']:.3f} \u00c5 \u00b7 C{m['csym']}"
            ),
            title_x=0.5,
            title_font=dict(size=12),
            margin=dict(t=60, b=10, l=10, r=10),
            height=380,
        )
        btn = render.download(
            label="Download map (.mrc)",
            filename="abinitio3d_relion_map.mrc",
        )

        @btn
        def download_relion_map():
            # the run's own folder is gone (it was temporary): write the map
            # from memory
            yield _mrc_bytes(m["volume"], m["apix"])

        return ui.div(_fig_to_html(fig), btn)

    # ── a 3D map from the class averages ──

    map_result = reactive.value(None)

    @reactive.effect
    @reactive.event(phase_result)
    def _clear_map_result():
        map_result.set(None)

    def _map_work(job, progress):
        r = job["phase_result"]
        apix_orig = job["apix"]
        progress.set(0, message="straightening the class averages")
        prepared = maps.straighten_classes(
            job["images"],
            r.class_zdir,
            target_apix=max(5.0, apix_orig),
            apix=apix_orig,
        )
        progress.set(
            1,
            message=f"reconstructing from {len(prepared)} "
            + ("unique classes" if r.merged_classes else "classes"),
        )
        return maps.reconstruct_map(
            prepared,
            np.rad2deg(r.phases),
            r.period,
            job["csym"],
            job["rise"],
            left_handed=job["left_handed"],
            apix=max(5.0, apix_orig),
            helical_sym_order=job["hsym"],
            impose_csym=job["impose_csym"],
            method="backprojection" if job["method"] == "backprojection" else "joint",
            algorithm=dict(model="gauss") if job["method"] == "gauss" else None,
            rounds=job["rounds"],
            output_box=int(job["images"][0].shape[0]),
            output_apix=apix_orig,
            cpu=max(1, int(helicon.available_cpu())),
        )

    def _map_apply(job, result):
        # a map from an older pitch result is not shown
        if phase_result() is not job["phase_result"]:
            return
        map_result.set(result)

    def _map_error(job, e):
        ui.modal_show(
            ui.modal(
                f"The 3D map could not be reconstructed: {e}",
                title="3D map error",
                easy_close=True,
                footer=None,
            )
        )

    map_task = helicon.shiny.background_task(
        "map_run",
        _map_work,
        _map_apply,
        _map_error,
        progress_max=2,
        session=session,
        label="3D map from the class averages",
    )

    @reactive.effect
    @reactive.event(input.map_run)
    def run_map():
        r = phase_result()
        req(r is not None)
        position = {int(c): i for i, c in enumerate(displayed_class_ids())}
        images = displayed_class_images()
        at = [position[int(c)] for c in map(lambda k: int(k) - 1, r.class_ids)]
        map_task.invoke(
            dict(
                phase_result=r,
                images=[images[i] for i in at],
                apix=float(data_all()[1]),
                csym=helicon.shiny.clamp_number(input.rot_fold(), 1, 1, 12),
                rise=float(input.rise()),
                left_handed=input.map_hand() == "left",
                hsym=helicon.shiny.clamp_number(input.map_hsym(), 1, 1, 1000),
                impose_csym=bool(input.map_csym()),
                method=input.map_method(),
                rounds=2 if input.map_refine() else 1,
            )
        )

    def _map_classes_text(n_used):
        """The selected classes, and how many the map used after merging."""
        r = phase_result()
        n_merged = len(r.merged_classes) if r is not None else 0
        if not n_merged:
            return f"{n_used} classes"
        return (
            f"{n_used + n_merged} classes ({n_merged} merged -> {n_used} unique "
            "classes)"
        )

    @render.ui
    def map_display():
        m = map_result()
        if m is None:
            return None
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2,
            cols=2,
            column_widths=[0.72, 0.28],
            row_heights=[0.5, 0.5],
            horizontal_spacing=0.04,
            vertical_spacing=0.12,
            subplot_titles=(
                "Classes at their azimuths",
                "",
                "Map projection",
                "z section",
            ),
        )
        for image, row, col in (
            (m["tiles"], 1, 1),
            (m["projection"], 2, 1),
            (m["z_view"], 2, 2),
        ):
            fig.add_trace(
                go.Heatmap(
                    z=image, colorscale="gray", showscale=False, hoverinfo="skip"
                ),
                row=row,
                col=col,
            )
            fig.update_yaxes(
                autorange="reversed",
                scaleanchor=f"x{'' if (row, col) == (1, 1) else 2 * (row - 1) + col}",
                showticklabels=False,
                row=row,
                col=col,
            )
            fig.update_xaxes(showticklabels=False, row=row, col=col)
        fig.update_layout(
            template="plotly_white",
            title_text=(
                f"{_map_classes_text(len(m['phis']))} \u00b7 twist {m['twist']:.3f}\u00b0 \u00b7 "
                f"fit {m['score']:.2f}"
            ),
            title_x=0.5,
            title_font=dict(size=12),
            margin=dict(t=60, b=10, l=10, r=10),
            height=440,
        )
        return _fig_to_html(fig)

    @render.ui
    def map_download():
        m = map_result()
        req(m is not None)
        btn = render.download(
            label="Download map (.mrc)",
            filename="abinitio3d_class_map.mrc",
        )

        @btn
        def download_map():
            import tempfile

            import mrcfile

            with tempfile.TemporaryDirectory() as tmp:
                path = str(Path(tmp) / "map.mrc")
                with mrcfile.new(path) as mrc:
                    mrc.set_data(m["volume_out"].astype(np.float32))
                    mrc.voxel_size = m["apix_out"]
                yield Path(path).read_bytes()

        return btn
