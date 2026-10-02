"""HelicalPitch tab — determine helical pitch/twist using 2D Classification info.

Faithfully ported from the standalone HelicalPitch app's app.py
with minimal adaptation for the Shiny module pattern.
Uses pio.to_html() + @render.ui to avoid shinywidgets comm issues.
"""

from __future__ import annotations

import logging

import numpy as np
import plotly.io as pio

import helicon
from shiny import reactive, ui, module, req, render

from .. import bookmark, class2d_files, deployment
from ..lib.shared_state import ProjectState

from ..lib import helical_pitch_compute as compute

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
    "auto_min_len": ("auto_min_len", True),
    "max_len": ("max_len", -1),
    "max_pair_dist": ("max_pair_dist", -1),
    "bins": ("bins", 150),
    # set from the data while "Auto-set minimal filament length" is on
    "min_len": ("min_len", 0, bookmark.DERIVED),
    "rise": ("rise", 4.75),
    "split": ("split_axis_distance", 50),
}

# most histogram bins the server draws
MAX_BINS = 10000


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


# The plots size themselves to their container, which has no height of its
# own inside a layout column, so each is given one.
_PLOT_BOX = "height: 340px; min-height: 340px; flex-shrink: 0; overflow: hidden;"


_MIN_LEN_BOX = "hp_min_len_box"


def _info(text, placement="top"):
    """A small (i) that shows ``text`` on hover, to put after a title or label."""
    return ui.tooltip(
        ui.tags.span(
            " \u24d8", style="cursor: help; color: #6c757d; font-size: 0.9em;"
        ),
        text,
        placement=placement,
    )


@module.ui
def helical_pitch_tab_ui():
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
                        "auto_min_len", "Auto-set minimal filament length", value=True
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
                        "max_len",
                        "Maximal length (\u00c5)",
                        min=-1,
                        value=-1,
                        step=1.0,
                        update_on="blur",
                    ),
                    ui.input_numeric(
                        "max_pair_dist",
                        "Maximal pair distance (\u00c5) to plot",
                        min=-1,
                        value=-1,
                        step=1.0,
                        update_on="blur",
                    ),
                    ui.input_numeric(
                        "bins",
                        "Number of histogram bins",
                        min=1,
                        max=MAX_BINS,
                        value=150,
                        step=1,
                        update_on="blur",
                    ),
                ),
            ),
            width="33vw",
            style="display: flex; flex-direction: column; height: 100%;",
        ),
        ui.h1(
            "HelicalPitch: determine helical pitch/twist using 2D Classification info",
            style="font-weight: bold;",
        ),
        ui.div(
            ui.layout_columns(
                ui.div(
                    ui.div(
                        ui.output_ui("display_selected_images"),
                        style="max-height: 40vh; overflow-y: auto; margin-bottom: 0; padding-bottom: 0;",
                    ),
                    ui.div(ui.output_ui("lengths_histogram"), style=_PLOT_BOX),
                    # takes no room, and is never hidden itself, so that it
                    # keeps running: it shows or hides the length box below
                    ui.div(
                        ui.output_ui("min_len_visibility"),
                        style="height: 0; margin: 0; padding: 0;",
                    ),
                    ui.div(
                        ui.input_numeric(
                            "min_len",
                            "Minimal length (\u00c5)",
                            min=0.0,
                            value=0,
                            step=1.0,
                            update_on="blur",
                        ),
                        id=_MIN_LEN_BOX,
                    ),
                    ui.output_data_frame("helices_table"),
                ),
                ui.div(
                    ui.div(
                        ui.output_ui("pair_distances_histogram"),
                        style=_PLOT_BOX,
                    ),
                    ui.markdown(
                        "**How to interpret the histogram:** An informative histogram "
                        "should have clear peaks with equal spacing. If so, hover your "
                        "mouse pointer over the first major peak off the origin to align "
                        "the vertical lines well with the peaks. Once you have decided "
                        "on the line position, read the hover text, which shows the twist "
                        "values assuming the pair distance is the helical pitch (adjusted "
                        "for the cyclic symmetries around the helical axis). You need to "
                        "decide which cyclic symmetry and the corresponding twist should "
                        "be used.\n\n"
                        "If the histogram does not show clear peaks, it indicates that the "
                        "Class2D quality is bad. You might consider changing the "
                        "'Minimal length (\u00c5)' from 0 to a larger value (for example, "
                        "1000 \u00c5) to improve the peaks in the histogram."
                    ),
                    ui.output_ui("download_section"),
                    ui.output_ui("pair_distances_histogram_selected"),
                ),
                col_widths=(5, 7),
            ),
            style="overflow-y: auto; height: 100%;",
        ),
        ui.HTML(
            """
<i><p style='margin:2px 0'>Developed by the <a href='https://jianglab.science.psu.edu/helicon' target='_blank'>Jiang Lab</a>. Report issues to <a href='https://github.com/jianglab/helicon/issues' target='_blank'>helicon@GitHub</a>.</p></i>
<script>
(function() {
  if (window.__hp_xhair) return;
  window.__hp_xhair = true;
  var S = new WeakSet();
  function install(gd) {
    if (!gd || !gd._fullLayout || !gd.layout) return;
    if (S.has(gd)) return;
    // Check for pre-created hidden vline shapes (plot_histogram multi_crosshair)
    var n = (gd.layout.shapes || []).length;
    if (n === 0) return;
    S.add(gd);
    // Use native DOM events + Plotly axis mapping so the crosshair survives
    // multiple Plotly.js loads (include_plotlyjs=True re-executes the factory).
    gd.addEventListener('mousemove', function(e) {
      var rect = gd.getBoundingClientRect();
      var xa = gd._fullLayout && gd._fullLayout.xaxis;
      if (!xa) return;
      // p2c expects a pixel position relative to the axis frame, not the full
      // div. Subtract the axis offset (left padding / margin) to align lines
      // with the cursor.
      var hx = xa.p2c(e.clientX - rect.left - xa._offset);
      var u = {shapes:[]};
      for (var i = 0; i < n; i++) {
        var x = hx * (i + 1);
        u.shapes.push({type:'line', x0:x, x1:x, y0:0, y1:1, yref:'paper',
          line:{width:i===0?3:2, dash:i===0?'solid':'dash', color:'green'},
          visible: x <= xa.range[1]});
      }
      if (u.shapes.length) Plotly.relayout(gd, u);
    });
    gd.addEventListener('mouseleave', function() {
      var u = {shapes:[]};
      for (var i = 0; i < n; i++)
        u.shapes.push({type:'line', x0:0, x1:0, y0:0, y1:1, yref:'paper',
          line:{width:i===0?3:2, dash:i===0?'solid':'dash', color:'green'},
          visible:false});
      Plotly.relayout(gd, u);
    });
  }
  function retry(gd) {
    install(gd) || setTimeout(function(){install(gd)}, 200) || setTimeout(function(){install(gd)}, 600);
  }
  document.querySelectorAll('.js-plotly-plot,.plotly-graph-div').forEach(retry);
  new MutationObserver(function(){document.querySelectorAll('.js-plotly-plot,.plotly-graph-div').forEach(retry);})
    .observe(document.body, {childList:true, subtree:true});
})();
</script>
"""
        ),
    )


@module.server
def helical_pitch_tab_server(input, output, session, project: ProjectState):
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
    min_len_ready = reactive.value(False)
    selected_images = reactive.value([])
    selected_image_labels = reactive.value([])

    selected_helices = reactive.value(([], [], 0))
    selected_helices_min_len = reactive.value((([], [], 0), 0))
    retained_helices_by_length = reactive.value([])
    pair_distances = reactive.value([])

    df_selected_helices = reactive.value(([], [], 0))
    pair_distances_df_selected = reactive.value([])

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

    @render.ui
    def select_classes_gallery():
        return _gallery(
            "select_classes_inner",
            displayed_class_images(),
            displayed_class_labels(),
            label=displayed_class_title(),
            selection=True,
            initial=initial_selected_image_indices(),
        )

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

    # ── Main area: selected images ──

    @render.ui
    @reactive.event(selected_images)
    def display_selected_images():
        return helicon.shiny.image_gallery(
            id=session.ns("display_selected_image"),
            label=reactive.value(
                f"{len(selected_images())} selected class(es) | "
                + _counts_text(_selected_class_numbers())
            ),
            images=selected_images,
            image_labels=selected_image_labels,
        )

    # ── Helix selection + filtering ──

    @reactive.effect
    @reactive.event(input.select_classes_inner, params)
    def get_selected_helices():
        min_len_ready.set(False)
        req(params() is not None)
        req(image_size())
        req(len(abundance()))
        class_indices = [displayed_class_ids()[i] for i in input.select_classes_inner()]
        helices = compute.select_classes(params=params(), class_indices=class_indices)
        if len(helices):
            filement_lengths = compute.get_filament_length(helices=helices)
            segments_count = np.sum([abundance()[i] for i in class_indices])
        else:
            filement_lengths = []
            segments_count = 0
        selected_helices.set((helices, filement_lengths, segments_count))
        if not input.auto_min_len():
            selected_helices_min_len.set((selected_helices(), input.min_len()))
            min_len_ready.set(len(helices) > 0)

    @reactive.effect
    @reactive.event(selected_helices)
    def auto_set_filament_min_len():
        req(input.auto_min_len())
        helices, filament_lengths, _ = selected_helices()
        _, min_len_tmp = compute.compute_pair_distances(
            helices=helices, lengths=filament_lengths, target_total_count=1000
        )
        min_len_tmp = int(min_len_tmp)
        ui.update_numeric("min_len", value=min_len_tmp)
        selected_helices_min_len.set((selected_helices(), min_len_tmp))
        min_len_ready.set(len(helices) > 0)

    @render.ui
    def min_len_visibility():
        if min_len_ready():
            return None
        return ui.tags.style(f"#{_MIN_LEN_BOX} {{ display: none; }}")

    @reactive.effect
    @reactive.event(input.min_len)
    def update_selected_helices_min_len():
        selected_helices_min_len.set((selected_helices(), input.min_len()))

    @reactive.effect
    @reactive.event(selected_helices_min_len, input.max_len)
    def select_helices_by_length():
        previous = getattr(select_helices_by_length, "previous", ([], 0, None))
        selected_image_indices_previous, min_len_previous, max_len_previous = previous
        (helices, filement_lengths, _), min_len = selected_helices_min_len()
        max_len = input.max_len()
        max_len = 0 if max_len is None else max_len
        req(
            set(selected_image_indices_previous) != set(input.select_classes_inner())
            or min_len_previous != min_len
            or max_len_previous != max_len
        )
        if len(helices) == 0:
            retained_helices_by_length.set([])
        elif min_len == 0 and max_len <= 0:
            retained_helices_by_length.set(helices)
        else:
            helices_retained, _ = compute.select_helices_by_length(
                helices=helices,
                lengths=filement_lengths,
                min_len=min_len,
                max_len=max_len,
            )
            retained_helices_by_length.set(helices_retained)
        select_helices_by_length.previous = (
            input.select_classes_inner(),
            min_len,
            max_len,
        )

    @reactive.effect
    @reactive.event(retained_helices_by_length)
    def get_pair_lengths():
        if len(retained_helices_by_length()):
            dists, _ = compute.compute_pair_distances(
                helices=retained_helices_by_length()
            )
            pair_distances.set(dists)
        else:
            pair_distances.set([])

    # ── Update min_len/max_len constraints ──

    @reactive.effect
    @reactive.event(input.min_len)
    def _():
        ui.update_numeric("max_len", min=input.min_len())
        if 0 < input.max_len() < input.min_len():
            ui.update_numeric("max_len", value=-1)

    @reactive.effect
    @reactive.event(input.max_len)
    def _():
        if input.max_len() > 0:
            ui.update_numeric("min_len", max=input.max_len())
        if input.min_len() >= input.max_len():
            ui.update_numeric("min_len", value=0)

    # ── Helix data table ──

    @render.data_frame
    @reactive.event(params, input.select_classes_inner)
    def helices_table():
        df = params()
        summary_df = (
            df.groupby("helixID")
            .agg(
                length=("length", "first"),
                rlnClassNumber=(
                    "rlnClassNumber",
                    lambda x: list(x.value_counts().index),
                ),
                rlnMicrographName=("rlnMicrographName", "first"),
            )
            .reset_index()
            .rename(columns={"rlnClassNumber": "classes"})
        )
        if len(input.select_classes_inner()):
            selected_classes = [
                int(displayed_class_ids()[i]) + 1 for i in input.select_classes_inner()
            ]
            summary_df = summary_df[
                summary_df["classes"].apply(
                    lambda x: any(cls in selected_classes for cls in x)
                )
            ]
        summary_df["classes"] = summary_df["classes"].apply(
            lambda x: ",".join(map(str, x))
        )
        summary_df = summary_df.sort_values("length", ascending=False)
        return render.DataGrid(
            summary_df, selection_mode="rows", filters=True, height="30vh", width="100%"
        )

    @reactive.effect
    def get_df_selected_helices():
        try:
            df_selected = helices_table.data_view(selected=True)
        except Exception:
            return
        df_selected_helixids = df_selected["helixID"].tolist()
        mask = params()["helixID"].astype(int).isin(df_selected_helixids)
        particles = params().loc[mask, :]
        class_indices = [displayed_class_ids()[i] for i in input.select_classes_inner()]
        helices = compute.select_classes(params=particles, class_indices=class_indices)
        if len(helices):
            filement_lengths = compute.get_filament_length(helices=helices)
            segments_count = np.sum([abundance()[i] for i in class_indices])
        else:
            filement_lengths = []
            segments_count = 0
        df_selected_helices.set((helices, filement_lengths, segments_count))

    # ── Lengths histogram ──

    def _bins():
        # bounded here as well as in the browser: the bin count sets the work
        return helicon.shiny.clamp_number(input.bins(), 150, 1, MAX_BINS)

    @render.ui
    def lengths_histogram():
        req(input.bins() is not None and input.bins() > 0)
        helices, lengths, count = selected_helices()
        class_indices = [
            str(displayed_class_ids()[i] + 1) for i in input.select_classes_inner()
        ]
        title = (
            f"Filament Lengths: Class {' '.join(class_indices)}<br>"
            f"<i>{len(helices):,} filaments | {count:,} segments</i>"
        )
        fig = compute.plot_histogram(
            data=lengths,
            title=title,
            xlabel="Filament Length (\u00c5)",
            ylabel="# of Filaments",
            bins=_bins(),
            log_y=True,
            fig=None,
        )
        return _fig_to_html(fig)

    # ── Pair distances histogram ──

    @render.ui
    def pair_distances_histogram():
        req(input.bins() is not None and input.bins() > 0)
        req(input.max_pair_dist() is not None)
        req(input.rise() is not None and input.rise() > 0)
        data = pair_distances()
        segment_count = np.sum([len(h) for _, h in retained_helices_by_length()])
        if len(retained_helices_by_length()):
            class_indices = np.unique(
                np.concatenate(
                    [h["rlnClassNumber"] for _, h in retained_helices_by_length()]
                )
            ).astype(int)
        else:
            class_indices = []
        class_indices = [
            str(displayed_class_ids()[i] + 1)
            for i in input.select_classes_inner()
            if (displayed_class_ids()[i] + 1) in class_indices
        ]
        rise = input.rise()
        title = (
            f"Pair Distances: Class {' '.join(class_indices)}<br>"
            f"<i>{len(retained_helices_by_length())} filaments | "
            f"{segment_count:,} segments | {len(pair_distances()):,} segment pairs</i>"
        )
        fig = compute.plot_histogram(
            data=data,
            title=title,
            xlabel="Pair Distance (\u00c5)",
            ylabel="# of Pairs",
            max_pair_dist=input.max_pair_dist(),
            bins=_bins(),
            log_y=True,
            show_pitch_twist=dict(rise=rise, csyms=(1, 2, 3, 4)),
            multi_crosshair=True,
            fig=None,
        )
        return _fig_to_html(fig, multi_crosshair=True, plot_id="hp_pair_distances")

    # ── Download ──

    @render.ui
    def download_section():
        if len(pair_distances()) <= 0:
            return None
        download_ui = render.download(
            label="Download selected helices", filename="helices.star"
        )

        @download_ui
        def download_retained_helices():
            req(retained_helices_by_length())
            indices = np.concatenate([h.index for _, h in retained_helices_by_length()])
            params_to_save = params().iloc[indices]
            import starfile

            yield starfile.to_string(
                dict(optics=params_to_save.attrs["optics"], particles=params_to_save)
            )

        return download_ui

    def _selected_class_numbers():
        sel = input.select_classes_inner() or []
        return [int(displayed_class_ids()[i]) + 1 for i in sel]

    # ── Pair distances for selected helices ──

    @reactive.effect
    def get_pair_lengths_df_selected():
        (helices, _, _) = df_selected_helices()
        if len(helices):
            dists, _ = compute.compute_pair_distances(helices=helices)
            pair_distances_df_selected.set(dists)
        else:
            pair_distances_df_selected.set([])

    @render.ui
    def pair_distances_histogram_selected():
        req(input.bins() is not None and input.bins() > 0)
        req(input.max_pair_dist() is not None)
        req(input.rise() is not None and input.rise() > 0)
        data = pair_distances_df_selected()
        req(data is not None and len(data) > 0)
        (helices, _, _) = df_selected_helices()
        if len(helices):
            class_indices = np.unique(
                np.concatenate([h["rlnClassNumber"] for _, h in helices])
            ).astype(int)
        else:
            class_indices = []
        class_indices = [
            str(displayed_class_ids()[i] + 1)
            for i in input.select_classes_inner()
            if (displayed_class_ids()[i] + 1) in class_indices
        ]
        segment_count = np.sum([len(h) for _, h in helices])
        rise = input.rise()
        title = (
            f"Pair Distances: Class {' '.join(class_indices)}<br>"
            f"<i>{len(helices)} filaments | {segment_count:,} segments | "
            f"{len(pair_distances_df_selected()):,} segment pairs</i>"
        )
        fig = compute.plot_histogram(
            data=data,
            title=title,
            xlabel="Pair Distance (\u00c5)",
            ylabel="# of Pairs",
            max_pair_dist=input.max_pair_dist(),
            bins=_bins(),
            log_y=True,
            show_pitch_twist=dict(rise=rise, csyms=(1, 2, 3, 4)),
            multi_crosshair=True,
            fig=None,
        )
        return _fig_to_html(
            fig, multi_crosshair=True, plot_id="hp_pair_distances_selected"
        )
