"""The page of the abinitio3d tab: its inputs, controls and plots, and what only they need."""

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import helicon
from shiny import ui, module
from shiny.module import resolve_id
from .. import deployment
from ..lib import helical_pitch_types as types


_urls = {
    "empiar-10940_job010": (
        "https://ftp.ebi.ac.uk/empiar/world_availability/10940/data/EMPIAR/Class2D/job010/run_it020_data.star",
        "https://ftp.ebi.ac.uk/empiar/world_availability/10940/data/EMPIAR/Class2D/job010/run_it020_classes.mrcs",
    )
}


_url_key = "empiar-10940_job010"


# Plotly sizes a figure once, to its box as the page is first laid out, and after
# that only when the window is resized. The box changes without that (a scrollbar
# appears, the results are shown, the columns are re-flowed), which left a figure
# larger than its box and its left and bottom edges cut off. A figure made in a box
# that is not shown yet has no size to take, and keeps a default one. So the figure
# is compared with its box, and made to fit it, whenever the box or the window
# changes, when the figure comes into view, and a few times as the page settles.
_RESIZE_WITH_BOX = (
    "var gd = document.getElementById('{plot_id}');"
    "if (gd && gd.parentElement) {"
    "  var fit = function() {"
    # a box that is not shown cannot be fitted
    "    if (!gd.offsetWidth || !gd.offsetHeight || !gd._fullLayout) return;"
    "    var l = gd._fullLayout;"
    "    if (Math.abs(l.width - gd.offsetWidth) > 1 || Math.abs(l.height - gd.offsetHeight) > 1) {"
    "      try { Plotly.Plots.resize(gd); } catch (e) {}"
    "    }"
    "  };"
    "  if (window.ResizeObserver) {"
    "    var ro = new ResizeObserver(fit);"
    "    ro.observe(gd.parentElement); ro.observe(gd);"
    "  }"
    "  if (window.IntersectionObserver) {"
    "    new IntersectionObserver(fit).observe(gd);"
    "  }"
    "  window.addEventListener('resize', fit);"
    "  [0, 100, 400, 1000, 2500].forEach(function(t) { setTimeout(fit, t); });"
    "}"
)


def _filament_joint_figure(
    filaments, period, length_range=(None, None), repeat_band=(None, None), n_unfitted=0
):
    """The repeat of each filament against its length, with a histogram of each.

    A scatter of the filaments with a fitted repeat (length across, repeat up), a
    histogram of the lengths on top of it and one of the repeats at its right. The
    filaments in the selection (inside the length range and the repeat band set
    by the sliders) are blue and the others grey, in the scatter and in the
    histograms; The green line is the pooled repeat.

    Everything is a trace, not a shape: the page script that gives the
    HelicalPitch histograms their crosshairs takes over the shapes of every plot.

    Parameters
    ----------
    filaments : pandas.DataFrame
        The filaments with a fitted repeat: ``span`` (length, A) and ``pitch``
        (repeat, A), and ``rlnMicrographName`` and ``rlnHelicalTubeID`` for the hover.
    period : float
        The pooled repeat, in A.
    length_range, repeat_band : tuple of float or None
        The selection; a bound that is None is not applied.
    n_unfitted : int, optional
        Filaments without a clear repeat, which are not in the plot.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    length = np.asarray(filaments["span"], dtype=float)
    repeat = np.asarray(filaments["pitch"], dtype=float)
    l_lo, l_hi = length_range
    r_lo, r_hi = repeat_band
    chosen = np.ones(len(length), dtype=bool)
    if l_lo is not None:
        chosen &= length >= l_lo
    if l_hi is not None:
        chosen &= length <= l_hi
    if r_lo is not None:
        chosen &= repeat >= r_lo
    if r_hi is not None:
        chosen &= repeat <= r_hi

    x_counts, x_edges = np.histogram(length, bins=40)
    y_counts, y_edges = np.histogram(repeat, bins=40)
    x_top, y_top = float(x_counts.max()) * 1.05, float(y_counts.max()) * 1.05
    x_bins = dict(start=x_edges[0], end=x_edges[-1], size=x_edges[1] - x_edges[0])
    y_bins = dict(start=y_edges[0], end=y_edges[-1], size=y_edges[1] - y_edges[0])
    x_max = float(length.max()) * 1.03
    y_pad = 0.04 * max(float(repeat.max() - repeat.min()), 1.0)
    y_min, y_max = float(repeat.min()) - y_pad, float(repeat.max()) + y_pad

    fig = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.8, 0.2],
        row_heights=[0.2, 0.8],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.015,
        vertical_spacing=0.02,
    )
    grey, blue = "#c9c9c9", "#1f77b4"

    # the selection first, so that it lies behind the bars and the points
    def band(x, y, row, col):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                fill="toself",
                fillcolor="rgba(255,165,0,0.2)",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    sel_x = [l_lo if l_lo is not None else 0.0, l_hi if l_hi is not None else x_max]
    sel_y = [r_lo if r_lo is not None else y_min, r_hi if r_hi is not None else y_max]
    if sel_x[1] > sel_x[0] and sel_y[1] > sel_y[0]:
        a, b = sel_x
        c, d = sel_y
        band([a, b, b, a, a], [c, c, d, d, c], 2, 1)
        # in the histograms the edges of the selection, not a band: bars are drawn
        # below lines, so a band would tint them
        for x, y, row, col in (
            ([a, a], [0, x_top], 1, 1),
            ([b, b], [0, x_top], 1, 1),
            ([0, y_top], [c, c], 2, 2),
            ([0, y_top], [d, d], 2, 2),
        ):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(color="orange", width=2),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
    for mask, colour, name in (
        (chosen, blue, "selected"),
        (~chosen, grey, "not selected"),
    ):
        if not mask.any():
            continue
        fig.add_trace(
            go.Histogram(
                x=length[mask],
                xbins=x_bins,
                marker_color=colour,
                opacity=1,
                name=name,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Histogram(
                y=repeat[mask],
                ybins=y_bins,
                marker_color=colour,
                opacity=1,
                name=name,
                orientation="h",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2,
            col=2,
        )
    kind = go.Scattergl if len(length) > 3000 else go.Scatter
    name = filaments["rlnMicrographName"].astype(str).str.rsplit("/", n=1).str[-1]
    label = (name + ":" + filaments["rlnHelicalTubeID"].astype(str)).to_numpy()
    for mask, colour, text in (
        (~chosen, grey, "not selected"),
        (chosen, blue, "selected"),
    ):
        if not mask.any():
            continue
        fig.add_trace(
            kind(
                x=length[mask],
                y=repeat[mask],
                mode="markers",
                marker=dict(color=colour, size=4, opacity=0.7),
                customdata=label[mask],
                hovertemplate="length %{x:.0f} \u00c5<br>repeat %{y:.0f} "
                "\u00c5<br>%{customdata}<extra>" + text + "</extra>",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    for x, y, row, col in (
        ([0, x_max], [period, period], 2, 1),
        ([0, y_top], [period, period], 2, 2),
    ):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(color="green", width=2),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
    fig.update_layout(
        template="plotly_white",
        # selected below, the others on top: the whole bar is every filament
        barmode="stack",
        bargap=0.03,
        title_text=(
            # two lines: it shares its row with the ranked fits
            f"Per-filament repeat \u00b7 {len(length):,} filaments, "
            f"{int(chosen.sum()):,} selected"
            f"<br><sup>+{n_unfitted:,} without a clear repeat</sup>"
        ),
        title_x=0.5,
        title_font=dict(size=12),
        margin=dict(t=62, b=50, l=55, r=15),
        showlegend=False,
    )
    fig.update_xaxes(
        range=[0, x_max], row=2, col=1, title_text="Filament length (\u00c5)"
    )
    fig.update_yaxes(range=[y_min, y_max], row=2, col=1, title_text="Repeat (\u00c5)")
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(range=[0, x_top], title_text="#", row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=2)
    fig.update_xaxes(range=[0, y_top], title_text="#", row=2, col=2)
    fig.update_xaxes(visible=False, row=1, col=2)
    fig.update_yaxes(visible=False, row=1, col=2)
    return fig


def _fig_to_html(fig, multi_crosshair=False, plot_id=None):
    """Convert a plotly figure to responsive HTML for rendering via render.ui."""
    import plotly.io as pio

    html = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=False,
        div_id=plot_id,
        config={"responsive": True, "displayModeBar": False},
        post_script=_RESIZE_WITH_BOX,
    )
    return ui.HTML(html)


# The plots size themselves to their container, which has no
# height of its own inside a card, so each is given one, from the height of the
# window (two rows of plots fit one screen, between 280 and 640 px); without it the
# three collapse onto each other and the histogram covers the download links.
_PLOT_BOX = (
    "height: clamp(280px, calc((100vh - 300px) / 2), 640px); min-height: 280px; "
    "flex-shrink: 0; overflow: hidden;"
)


# The map figure: the width (px) it is laid out for, and the share of it that the
# strips of classes and map projection take, the z section the rest.
_MAP_PLOT_WIDTH = 900


# The reconstruction methods of the map. "joint" is kept as the key of elastic net,
# so that older bookmarks still name it.
_TYPES_METHODS = {
    "markers": "Marker classes (class averages)",
    "context": "Filament context (sequence of classes)",
}


_MAP_METHODS = {
    "joint": "Linear regression - elasticnet",
    "gauss": "Linear regression - gauss",
    "backprojection": "Back-projection",
}


_MAP_STRIP_SHARE = 0.72


def _map_figure(m, title):
    """The map from classes: the classes at their azimuths above the map's side
    projection, and its central z section beside both.

    Each row is as tall as its strip is at the width the two share, and the strips
    are pulled together (row 1 to its bottom, row 2 to its top, both toward the z
    section), so that the space the figure leaves over goes outside them, not
    between them.

    Parameters
    ----------
    m : dict
        From :func:`helical_pitch_map.reconstruct_map`: ``tiles``,
        ``projection`` and ``z_view``.
    title : str

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    strips = (m["tiles"], m["projection"])
    shape = [np.shape(s)[0] / max(np.shape(s)[1], 1) for s in strips]
    strip_px = [_MAP_PLOT_WIDTH * _MAP_STRIP_SHARE * a for a in shape]
    gap_px = 34  # room for the second title
    fig = make_subplots(
        rows=2,
        cols=2,
        column_widths=[_MAP_STRIP_SHARE, 1.0 - _MAP_STRIP_SHARE],
        row_heights=strip_px,
        specs=[[{}, {"rowspan": 2}], [{}, None]],
        horizontal_spacing=0.04,
        vertical_spacing=gap_px / (sum(strip_px) + gap_px),
    )
    panels = (
        # (image, row, col, title, side its x and y shrink toward)
        (m["tiles"], 1, 1, "Classes at their azimuths", "right", "bottom"),
        (m["projection"], 2, 1, "Map projection", "right", "top"),
        (m["z_view"], 1, 2, "z section", "left", "middle"),
    )
    for image, row, col, label, x_toward, y_toward in panels:
        fig.add_trace(
            go.Heatmap(z=image, colorscale="gray", showscale=False, hoverinfo="skip"),
            row=row,
            col=col,
        )
        sub = fig.get_subplot(row, col)
        x_ref = sub.xaxis.plotly_name.replace("axis", "")
        y_ref = sub.yaxis.plotly_name.replace("axis", "")
        fig.update_yaxes(
            autorange="reversed",
            scaleanchor=x_ref,
            constrain="domain",
            constraintoward=y_toward,
            showticklabels=False,
            showgrid=False,
            row=row,
            col=col,
        )
        fig.update_xaxes(
            constrain="domain",
            constraintoward=x_toward,
            showticklabels=False,
            showgrid=False,
            row=row,
            col=col,
        )
        # on the axis, not the page: the title follows its image when the image
        # is shrunk to keep its shape
        fig.add_annotation(
            text=label,
            x=0.5,
            y=1.0,
            xref=f"{x_ref} domain",
            yref=f"{y_ref} domain",
            xanchor="center",
            yanchor="bottom",
            showarrow=False,
            font=dict(size=14),
        )
    fig.update_layout(
        template="plotly_white",
        title_text=title,
        title_x=0.5,
        title_font=dict(size=12),
        margin=dict(t=60, b=10, l=10, r=10),
        # as tall as the strips are at a typical width; at other widths the
        # strips keep their shape and stay together
        height=int(round(70 + sum(strip_px) + gap_px)),
    )
    return fig


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
                    ui.input_checkbox(
                        "dihedral",
                        ui.span(
                            "Dihedral symmetry check",
                            _info(
                                "Dihedral (Dn) symmetry adds two-fold axes perpendicular to the "
                                "helical axis: the filament looks the same turned "
                                "upside down, so it has no polarity. If there is "
                                "such an axis, the view of a class is the mirror "
                                "image, along the filament, of the view at the "
                                "mirrored azimuth (the axis need not be centred "
                                "in any class). The plot scans the azimuth of the "
                                "axis and shows how well every class matches its "
                                "mirrored partner. How to read it: a sharp peak "
                                "close to 1, well above the rest of the curve, "
                                "means a two-fold axis at that azimuth; a flat "
                                "curve (peak below about 0.75) means plain helical "
                                "symmetry. Needs classes spread evenly over the "
                                "azimuths. The result only informs: it changes "
                                "nothing in the reconstruction."
                            ),
                        ),
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
                    ui.input_checkbox(
                        "types_restrict",
                        ui.span(
                            "Only the filaments of the type group in use",
                            _info(
                                "Use group (Types card) ticks this: Estimate "
                                "pitch, the suggestions, the maps and the "
                                "downloads then take the segments of that "
                                "group's filaments only. Untick it to use the "
                                "selected classes on all filaments; tick it again "
                                "for the group's filaments."
                            ),
                        ),
                        value=False,
                    ),
                    ui.output_ui("types_restrict_note"),
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
                    ui.input_checkbox(
                        "repeat_map_fit",
                        ui.span(
                            "Decide the repeat with maps from the class averages "
                            "too (minutes)",
                            _info(
                                "At every candidate repeat, a map is made from two "
                                "thirds of the selected averages at the ring's "
                                "azimuths, and the rest are scored by how well they "
                                "fit it. That score and the segments' are combined: "
                                "the segments refine the repeat where their pairs "
                                "reach half of it, the maps decide it where they do "
                                "not (filaments shorter than a repeat, which bias "
                                "the segments). The maps also tell whether the pitch "
                                "is one or two repeats (C1 or C2). Uses all the "
                                "CPUs available."
                            ),
                        ),
                        value=False,
                    ),
                    ui.hr(),
                    ui.tags.b("3D map from classes"),
                    ui.div(
                        ui.input_checkbox(
                            "map_csym",
                            ui.span(
                                "Impose C symmetry",
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
                        _MAP_METHODS,
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
        helicon.shiny.plotly_dependency(),
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
                                    class_="btn-sm btn-primary",
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
                                class_="btn-primary",
                            ),
                            "More classes of this type: those that look like a "
                            "selected class turned 180\u00b0 (\u21bb, with the "
                            "correlation), and those on the same filaments as the "
                            "selected classes (\u25cf, the share of the filaments "
                            "having the class that are mostly selected classes). "
                            "Listed best first by the fit each would get on the "
                            "current ring (fit, on the scale of the ranked fits), "
                            "with the pixel agreement its average would have "
                            "(pixels) for the best-fitting ones. Those below the "
                            "Fit or Pixel agreement threshold are held back.",
                        ),
                        ui.div(
                            ui.output_ui("suggested_gallery"),
                            style="max-height: 32vh; overflow-y: auto;",
                        ),
                        ui.output_ui("suggested_buttons"),
                        id=_SUGGESTIONS_BOX,
                    ),
                    ui.card(
                        ui.card_header("Types"),
                        ui.div(
                            _tip(
                                ui.input_select(
                                    "types_method",
                                    "Method",
                                    _TYPES_METHODS,
                                    selected="markers",
                                ),
                                "How the filaments are grouped. Marker classes: by the "
                                "classes the filaments of one group use and those of "
                                "the other never do, from the class averages (fast). "
                                "Filament context: by the sequence of classes along "
                                "each filament, a hidden Markov chain over its "
                                "segments whose type stays along the filament and "
                                "whose class follows the type's own transitions "
                                "(RELION's helix_filament_context). It does not "
                                "register the class averages. It is slower (a model "
                                "is fitted for each number of groups, minutes on a "
                                "large data set) and needs the segments in order along "
                                "their filaments. On the one set of several known "
                                "types it was measured on, it told the types apart "
                                "better with 4 or 5 groups, and as well with 6 or "
                                "more. The number of groups it starts at is chosen, "
                                "as in RELION, by how well each number predicts "
                                "filaments held out of the fit (the fewest within "
                                "0.01 nats per segment of the best, one type "
                                "included). Short filaments show only part of the "
                                "cycle of views, so types of one structure can "
                                "differ by the phase they show; the group list "
                                "says when the filaments are short, and how alike "
                                "the class averages of the groups look. The "
                                "slider changes it.",
                            ),
                            _tip(
                                ui.input_task_button(
                                    "types_run",
                                    label="Group filaments",
                                    class_="btn-primary",
                                    style="white-space: nowrap; margin-bottom: 1rem;",
                                ),
                                "Groups the filaments by type with the method above "
                                "(shown in its tooltip). Marker classes: every pair "
                                "of class averages is registered, classes whose "
                                "averages look alike form candidate sets, and the "
                                "filaments are split, one step at a time, by the set "
                                "that one side uses and the other never does (views "
                                "only one type shows). Filament context: a model of "
                                "the sequence of classes along the filaments is "
                                "fitted for each number of types from 2 to 8, and "
                                "each filament gets the type that most of its "
                                "segments are in; it takes minutes on a large data "
                                "set. In both, classes that look like no other "
                                "(junk, or a view of their own) are listed as Not "
                                "grouped, and the repeat is not used (different "
                                "types can share it). Use one group at a time, then "
                                "estimate its pitch.",
                            ),
                            style=(
                                "display: grid; grid-template-columns: minmax(0, 1fr) auto; "
                                "gap: 8px; align-items: end;"
                            ),
                        ),
                        helicon.shiny.slider(
                            "types_n",
                            ui.span(
                                "Groups",
                                _info(
                                    "How many groups of filaments. Marker classes: "
                                    "the splits are made best first, and it starts "
                                    "after the last clear split. Filament context: "
                                    "it starts at the number of types that predicts "
                                    "held-out filaments best (the fewest within 0.01 "
                                    "nats per segment, one type included), and a "
                                    "group is one type. Too "
                                    "many groups is the safer "
                                    "error: two groups of one type give the same "
                                    "structure and can be merged later, while a "
                                    "group of two types gives none. A group that "
                                    "lacks the views another group is told by may "
                                    "also hold short filaments of that group, and "
                                    "says so."
                                ),
                            ),
                            min=1,
                            max=types.MAX_MARKER_GROUPS,
                            value=1,
                            step=1,
                        ),
                        ui.output_ui("types_controls"),
                        ui.div(
                            ui.output_ui("types_gallery"),
                            style="max-height: 32vh; overflow-y: auto;",
                        ),
                        ui.output_ui("types_buttons"),
                        ui.output_ui("types_pitch_ui"),
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
                    ui.output_ui("type_filter_status"),
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
                        ui.output_ui("dihedral_plot"),
                        ui.layout_columns(
                            # the histogram, and below it the two ranges that
                            # select filaments from it, one under the other
                            ui.div(
                                ui.div(
                                    ui.output_ui("filament_pitch_plot"),
                                    style=_PLOT_BOX,
                                    class_="ab-plot-box",
                                ),
                                # the two ranges that select filaments from it
                                # on the left, what they select to take away
                                # on the right
                                ui.layout_columns(
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
                                    ),
                                    ui.output_ui("pitch_band_download"),
                                    col_widths=(7, 5),
                                    fill=False,
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
                                    helicon.shiny.slider(
                                        "agree_threshold",
                                        ui.span(
                                            "Pixel agreement threshold",
                                            _info(
                                                "Classes whose registered averages agree "
                                                "with the ring less than this are picked "
                                                "to remove (the purple line). 0 (the "
                                                "default) picks none. A class with no "
                                                "registered pair is not judged. Estimate "
                                                "the pitch again after removing."
                                            ),
                                        ),
                                        min=0.0,
                                        max=1.0,
                                        value=0.0,
                                        step=0.01,
                                    ),
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
                col_widths=(4, 8),
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
