"""HelicalProjection tab — compare 2D images with helical structure projections.

Ported from HelicalProjection.git and adapted to the Shiny module pattern.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time
import logging
import re

import numpy as np
import pandas as pd

import helicon
from shiny import reactive, render, ui, module, req
from shiny.types import SilentException
import plotly.express as px


from ..lib.shared_state import ProjectState
from ..lib import helical_projection_compute as compute
from ..lib import helix_transform
from ..lib import map_gauss_fit

logger = logging.getLogger(__name__)

# How much re-placing the gaussian mode does with the pixel aligner before the
# results are displayed, counted in image-map pairs rather than in maps: one
# pair costs about 0.7 s, so a fixed ten maps would cost more than the search
# itself once several images are selected. Ten pairs covers the top of a
# single-image search, which is where a user looks first.
POLISHED_PAIRS_FOR_DISPLAY = 10

# Above this many maps, generating the x/y/z previews is worth confirming:
# each map is a download of tens or hundreds of megabytes, and selecting a
# whole filtered table is one click.
MAPS_NEEDING_CONFIRMATION = 10


def _emdb_link_script():
    """Make the entry ids in the EMDB table open their entry.

    The link is added to the *rendered* cells rather than to the data. Putting
    an anchor in the frame makes the grid treat the column as HTML, and a
    column of HTML cannot be filtered: typing an id that was on screen matched
    nothing, which is a poor trade for a link in a table of 751 rows. This way
    the column stays plain text -- filterable, sortable, and still what the
    tab matches rows against.

    The grid draws only the rows in view and redraws them on scroll, sort and
    filter, so the work is repeated from a MutationObserver rather than done
    once. The anchor stops its own click from bubbling, so following a link
    does not also select that row.
    """
    return ui.tags.script(
        r"""
        (function () {
            var GRID = 'helical_projection-display_emdb_dataframe';
            var PATTERN = /^EMD-\d+$/;
            function linkify() {
                var grid = document.getElementById(GRID);
                if (!grid) return;
                var headers = grid.querySelectorAll('thead th');
                var column = -1;
                headers.forEach(function (th, i) {
                    if (th.textContent.trim().toLowerCase() === 'emdb_id') column = i;
                });
                if (column < 0) return;
                grid.querySelectorAll('tbody tr').forEach(function (tr) {
                    var cell = tr.children[column];
                    if (!cell || cell.querySelector('a')) return;
                    var id = cell.textContent.trim();
                    if (!PATTERN.test(id)) return;
                    var a = document.createElement('a');
                    a.href = 'https://www.ebi.ac.uk/emdb/' + id;
                    a.target = '_blank';
                    a.rel = 'noopener';
                    a.textContent = id;
                    a.addEventListener('click', function (e) { e.stopPropagation(); });
                    cell.textContent = '';
                    cell.appendChild(a);
                });
            }
            function watch() {
                var grid = document.getElementById(GRID);
                if (!grid) return setTimeout(watch, 500);
                linkify();
                new MutationObserver(function () { linkify(); }).observe(
                    grid, {childList: true, subtree: true});
            }
            document.addEventListener('DOMContentLoaded', watch);
            watch();
        })();
        """
    )


BOOKMARK_DEFAULTS = {
    "mode_images": ("input_mode_images", "url"),
    "url_images": ("url_images", ""),
    "mode_maps": ("input_mode_maps", "url"),
    "ignore_blank": ("ignore_blank", True),
    "show_pdb": ("show_pdb", False),
    "use_curated": ("use_curated_helical_parameters", True),
    "show_twist_star": ("show_twist_star", True),
    "proj_xyz": ("map_projection_xyz_choices", ["x", "y", "z"]),
    "xyz_size": ("map_xyz_projection_display_size", 128),
    "side_size": ("map_side_projection_vertical_display_size", 128),
    "length_z": ("length_z", 1),
    "length_xy": ("length_xy", 1.2),
    "scale_range": ("scale_range", 5),
    "rescale_apix": ("rescale_apix", True),
    "match_sf": ("match_sf", True),
    "projection_method": ("projection_method", "gaussian"),
    "plot_scores": ("plot_scores", True),
    "hide_query": ("hide_query_image", False),
}

_urls = {
    "empiar-10940_job010": (
        "https://ftp.ebi.ac.uk/empiar/world_availability/10940/data/EMPIAR/Class2D/job010/run_it020_classes.mrcs",
        "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-14046/map/emd_14046.map.gz",
    )
}
_url_key = "empiar-10940_job010"


@module.ui
def helical_projection_tab_ui():
    return ui.div(
        ui.layout_sidebar(
            ui.sidebar(
                ui.navset_pill(
                    ui.nav_panel(
                        "Input 2D Images",
                        ui.input_radio_buttons(
                            "input_mode_images",
                            "How to obtain the input images:",
                            choices=["upload", "url"],
                            selected="url",
                            inline=True,
                        ),
                        ui.output_ui("create_input_image_files_ui"),
                        ui.hr(),
                        ui.output_ui("select_image_gallery"),
                    ),
                    ui.nav_panel(
                        "Input 3D Maps",
                        ui.input_radio_buttons(
                            "input_mode_maps",
                            "How to obtain the 3D maps:",
                            choices=[
                                "upload",
                                "url",
                                "amyloid_atlas",
                                "EMDB-helical",
                                "EMDB",
                            ],
                            selected="url",
                            inline=True,
                        ),
                        ui.output_ui("create_input_map_files_ui"),
                        ui.panel_conditional(
                            "input.input_mode_maps === 'amyloid_atlas' || input.input_mode_maps === 'EMDB-helical' || input.input_mode_maps === 'EMDB'",
                            ui.output_data_frame("display_emdb_dataframe"),
                        ),
                        ui.output_ui("map_actions_ui"),
                        ui.output_ui("display_map_xyz_projections_gallery"),
                    ),
                    ui.nav_panel(
                        "Parameters",
                        ui.layout_columns(
                            ui.input_checkbox(
                                "ignore_blank", "Ignore blank input images", value=True
                            ),
                            ui.input_checkbox(
                                "show_pdb", "Show PDB ids in EMDB table", value=False
                            ),
                            ui.tooltip(
                                ui.input_checkbox(
                                    "use_curated_helical_parameters",
                                    "Use curated helical parameters",
                                    value=True,
                                ),
                                "When checked, uses curated values from jianglab/EMDB_helical_parameter_curation",
                            ),
                            ui.tooltip(
                                ui.input_checkbox(
                                    "show_twist_star",
                                    "Show twist* in EMDB table",
                                    value=True,
                                ),
                                "Displays twist adjusted for helical symmetry",
                            ),
                            col_widths=6,
                        ),
                        ui.layout_columns(
                            ui.input_checkbox_group(
                                "map_projection_xyz_choices",
                                "Show projections along:",
                                choices=["x", "y", "z"],
                                selected=["x", "y", "z"],
                                inline=True,
                            ),
                            col_widths=6,
                        ),
                        ui.layout_columns(
                            ui.input_numeric(
                                "map_xyz_projection_display_size",
                                "Map XYZ projection image size",
                                value=128,
                                min=32,
                                max=512,
                                step=16,
                                update_on="blur",
                            ),
                            ui.input_numeric(
                                "map_side_projection_vertical_display_size",
                                "Side projection display size",
                                value=128,
                                min=32,
                                max=512,
                                step=32,
                                update_on="blur",
                            ),
                            ui.input_numeric(
                                "length_z",
                                "Z-projection length (x rise)",
                                value=1,
                                min=0,
                                step=1,
                                update_on="blur",
                            ),
                            ui.input_numeric(
                                "length_xy",
                                "Side projection length (x pitch)",
                                value=1.2,
                                min=0,
                                step=0.1,
                                update_on="blur",
                            ),
                            ui.input_numeric(
                                "scale_range",
                                "Search image scale (%)",
                                value=5,
                                min=0,
                                max=100,
                                step=1,
                                update_on="blur",
                            ),
                            col_widths=6,
                        ),
                        ui.layout_columns(
                            ui.input_checkbox(
                                "rescale_apix",
                                "Resample to image pixel size",
                                value=True,
                            ),
                            ui.input_checkbox(
                                "match_sf", "Apply matched-filter", value=True
                            ),
                            ui.input_checkbox(
                                "plot_scores", "Plot matching scores", value=True
                            ),
                            ui.input_checkbox(
                                "hide_query_image", "Hide query image", value=False
                            ),
                            col_widths=6,
                        ),
                        ui.input_radio_buttons(
                            "sort_map_side_projections_by",
                            "Sort projections by",
                            choices=["selection", "similarity score"],
                            selected="similarity score",
                            inline=True,
                        ),
                        ui.accordion(
                            ui.accordion_panel(
                                "Filtering options:",
                                ui.input_numeric(
                                    "lp_angst_x",
                                    "Low pass filtering X (Å):",
                                    value=-1,
                                    step=0.1,
                                    update_on="blur",
                                ),
                                ui.input_numeric(
                                    "hp_angst_x",
                                    "High pass filtering X (Å):",
                                    value=-1,
                                    step=0.1,
                                    update_on="blur",
                                ),
                                ui.input_numeric(
                                    "aniso_ratio_xy",
                                    "Anisotropic ratio (X/Y):",
                                    value=1.0,
                                    step=0.1,
                                    update_on="blur",
                                ),
                            ),
                            id="filtering_options",
                            open=False,
                        ),
                        ui.input_radio_buttons(
                            "projection_method",
                            "Search mode",
                            choices={
                                "gaussian": "Gaussians, 2D and 3D (faster)",
                                "volume": "Voxels and pixels",
                            },
                            selected="gaussian",
                            inline=True,
                        ),
                    ),
                    id="hp_tab",
                ),
                width="33vw",
            ),
            ui.div(
                ui.h1(
                    "HelicalProjection: compare 2D images with helical structure projections",
                    style="font-weight: bold;",
                ),
                _emdb_link_script(),
                helix_transform.card_switching_script(
                    [
                        {
                            "input": "display_selected_image",
                            "cards": ".hp-pi-card",
                            "key": "pi",
                        }
                    ]
                ),
                # The gallery gets the full width of the viewport to itself, so
                # the images flow across and wrap rather than being squeezed
                # into a column beside the controls; the transform UI and the
                # buttons follow on a row of their own.
                ui.div(
                    ui.output_ui("display_selected_image_gallery"),
                    style="width: 100%; margin-bottom: 10px;",
                ),
                ui.div(
                    ui.output_ui("hp_per_image_transform_ui"),
                    ui.div(
                        ui.input_action_button(
                            "auto_transform", "Auto Transform", class_="btn-primary"
                        ),
                        ui.input_action_button("reset_transform", "Reset Transform"),
                        style="display: flex; flex-direction: column; gap: 6px;"
                        " min-width: 170px;",
                    ),
                    ui.div(
                        ui.input_task_button(
                            "compare_projections", "Compare projections"
                        ),
                        # The top-matches control belongs with the button that
                        # produces the matches, and the rest of this row is
                        # empty space it can use.
                        ui.output_ui("select_top_n_ui"),
                        style="display: flex; flex-direction: column;"
                        " min-width: 170px; gap: 8px;",
                    ),
                    style="display: flex; flex-direction: row; flex-wrap: wrap;"
                    " align-items: flex-start; gap: 10px; margin-bottom: 12px;",
                ),
                ui.div(
                    ui.output_ui("generate_score_plot_ui"),
                    ui.div(
                        ui.output_ui("display_map_side_projections_gallery"),
                        style="max-height: 80vh; overflow-y: auto;",
                    ),
                ),
                ui.HTML(
                    "<i><p>Developed by the <a href='https://jianglab.science.psu.edu/helicon' target='_blank'>Jiang Lab</a>. "
                    "Report issues to <a href='https://github.com/jianglab/helicon/issues' target='_blank'>helicon@GitHub</a>.</p></i>"
                ),
            ),
        ),
    )


@module.server
def helical_projection_tab_server(input, output, session, project: ProjectState):
    images_all = reactive.value([])
    image_size = reactive.value(0)
    image_apix = reactive.value(0)

    displayed_image_ids = reactive.value([])
    displayed_images = reactive.value([])
    displayed_image_title = reactive.value("Select an image:")
    displayed_image_labels = reactive.value([])

    initial_selected_image_indices = reactive.value([0])
    selected_images_original = reactive.value([])
    selected_images_labels = reactive.value([])
    selected_images_thresholded_rotated_shifted_cropped = reactive.value([])
    # key -> helix_transform.ImageTransform for every selected image, keyed by
    # the image's label rather than its position in the selection
    transform_state = reactive.value({})
    # the next generation number for controls, see ImageTransform.generation
    transform_generation = [0]
    # measurements already made, keyed by image content, so adding an image to
    # the selection measures that image and nothing else
    auto_transform_cache = {}

    emdb_df_original = reactive.value(None)
    emdb_df = reactive.value(None)

    maps = reactive.value([])
    # which of the selected images the transform card is editing
    active_selected_image = reactive.value(0)
    map_xyz_projections = reactive.value([])
    map_xyz_projection_title = reactive.value("Map XYZ projections:")
    map_xyz_projection_labels = reactive.value([])
    map_xyz_projection_display_size = reactive.value(128)

    map_side_projections_with_alignments = reactive.value([])
    map_side_projections_displayed = reactive.value([])
    map_side_projection_title = reactive.value("Map side projections:")
    map_side_projection_labels = reactive.value([])
    map_side_projection_links = reactive.value([])
    map_side_projection_vertical_display_size = reactive.value(128)

    # -- Render slots for image_gallery (must run inside session) --

    @render.ui
    def select_image_gallery():
        return helicon.shiny.image_gallery(
            id=session.ns("select_image"),
            label=displayed_image_title,
            images=displayed_images,
            image_labels=displayed_image_labels,
            image_size=reactive.value(128),
            initial_selected_indices=initial_selected_image_indices,
            enable_selection=True,
            # several class averages can be searched jointly: each is aligned
            # to the same projection and the scores averaged
            allow_multiple_selection=True,
        )

    @render.ui
    def display_map_xyz_projections_gallery():
        return helicon.shiny.image_gallery(
            id=session.ns("display_map_xyz_projections"),
            label=map_xyz_projection_title,
            images=map_xyz_projections,
            image_labels=map_xyz_projection_labels,
            image_size=map_xyz_projection_display_size,
            enable_selection=False,
        )

    @reactive.calc
    def selected_image_gallery_title():
        n = len(selected_images_thresholded_rotated_shifted_cropped())
        if n > 1:
            return "Selected images (searched jointly, scores averaged):"
        return "Selected image:"

    @render.ui
    def display_selected_image_gallery():
        return helicon.shiny.image_gallery(
            id=session.ns("display_selected_image"),
            label=selected_image_gallery_title,
            images=selected_images_thresholded_rotated_shifted_cropped,
            image_labels=selected_images_labels,
            image_size=map_side_projection_vertical_display_size,
            justification="left",
            # clicking picks which image's transform card is shown, so the
            # controls take one card's worth of space instead of N
            enable_selection=len(selected_images_original()) > 1,
            allow_multiple_selection=False,
            # The image being edited, not the first one. This gallery shows
            # the *transformed* images, so every slider move re-renders it,
            # and a gallery re-render clicks its initial selection for real --
            # which used to throw the card switcher back to image 1 in the
            # middle of editing image 5.
            initial_selected_indices=reactive.value([active_selected_image()]),
            display_dashed_line=True,
        )

    @render.ui
    def display_map_side_projections_gallery():
        return helicon.shiny.image_gallery(
            id=session.ns("display_map_side_projections"),
            label=map_side_projection_title,
            images=map_side_projections_displayed,
            image_labels=map_side_projection_labels,
            image_links=map_side_projection_links,
            image_size=map_side_projection_vertical_display_size,
            justification="left",
            enable_selection=False,
        )

    # -- Score plot --

    @render.ui
    @reactive.event(input.plot_scores, map_side_projections_with_alignments)
    def generate_score_plot_ui():
        req(input.plot_scores())
        req(len(map_side_projections_with_alignments()) > 1)
        work = sorted(map_side_projections_with_alignments(), key=lambda x: -x[4])
        scores = [item[4] for item in work]
        labels = [item[8] for item in work]
        titles = [""] * len(labels)
        try:
            df = emdb_df()
            if df is not None:
                for li, label in enumerate(labels):
                    if label in df["emdb_id"].values:
                        mask = df["emdb_id"] == label
                        titles[li] = str(df.loc[mask, "title"].values[0])
        except Exception:
            pass
        fig = px.scatter(
            x=range(1, len(scores) + 1),
            y=scores,
            hover_name=labels,
            hover_data=dict(titles=titles),
            labels={"x": "Rank", "y": "Similarity Score"},
        )
        fig.update_traces(
            hovertemplate="<b>%{hovertext}</b><br><i>%{customdata}</i><br>Score: %{y:.3f}<br>Rank: %{x}"
        )
        if len(labels) > 0:
            fig.add_annotation(
                x=1,
                y=scores[0],
                text=labels[0],
                yanchor="middle",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="black",
                ax=70,
                ay=0,
                standoff=5,
            )
        fig.update_layout(
            xaxis_title="Rank",
            yaxis_title="Similarity Score",
            showlegend=False,
            autosize=True,
            width=None,
        )
        import plotly.io as _pio

        return ui.HTML(_pio.to_html(fig, full_html=False))

    @render.ui
    @reactive.event(map_side_projections_with_alignments)
    def select_top_n_ui():
        req(len(map_side_projections_with_alignments()))
        n_results = len(map_side_projections_with_alignments())
        # Label beside the box, not above it: in a 150px column the label wrapped
        # onto a second line and left the field itself too narrow to read.
        return ui.div(
            ui.tags.style(
                ".hp-top-n .shiny-input-container { margin-bottom: 0; width: auto; }"
            ),
            ui.tags.label(
                "Number of top matches:",
                {"for": session.ns("select_top_n")},
                style="white-space: nowrap; margin: 0;",
            ),
            ui.input_numeric(
                "select_top_n",
                None,
                min=0,
                value=min(10, n_results),
                width="110px",
            ),
            ui.input_action_button(
                "select_top_n_button", "Select", style="white-space: nowrap;"
            ),
            class_="hp-top-n",
            style="display: flex; flex-direction: row; align-items: center;"
            " gap: 8px; margin-bottom: 8px;",
        )

    # -- Dynamic input UI --

    @render.ui
    @reactive.event(input.input_mode_images)
    def create_input_image_files_ui():
        displayed_images.set([])
        if input.input_mode_images() == "upload":
            return ui.input_file(
                "upload_images",
                "Upload input images (.mrcs, .mrc)",
                accept=[".mrcs", ".mrc"],
                placeholder="mrcs or mrc file",
            )
        elif input.input_mode_images() == "url":
            return ui.input_text(
                "url_images",
                "Download URL for MRC images file",
                value=_urls[_url_key][0],
            )
        return None

    @render.ui
    @reactive.event(input.input_mode_maps)
    def create_input_map_files_ui():
        mode = input.input_mode_maps()
        twist_val = project.twist() if project.twist() else 179.402
        rise_val = project.rise() if project.rise() else 2.378
        csym_val = project.csym() if project.csym() else 1
        if mode == "upload":
            return ui.div(
                ui.input_file(
                    "upload_map",
                    "Upload 3D map (.mrc, .map, .gz)",
                    accept=[".mrc", ".mrc.gz", ".map", ".map.gz"],
                ),
                ui.layout_columns(
                    ui.input_numeric(
                        "twist",
                        "Twist (°)",
                        value=twist_val,
                        min=-180,
                        max=180,
                        step=1,
                        update_on="blur",
                    ),
                    ui.input_numeric(
                        "rise",
                        "Rise (Å)",
                        value=rise_val,
                        min=0,
                        step=1,
                        update_on="blur",
                    ),
                    ui.input_numeric(
                        "csym", "Csym", value=csym_val, min=1, step=1, update_on="blur"
                    ),
                    col_widths=[4, 4, 4],
                ),
            )
        elif mode == "url":
            return ui.div(
                ui.input_text(
                    "url_map", "Download URL for 3D map", value=_urls[_url_key][1]
                ),
                ui.layout_columns(
                    ui.input_numeric(
                        "twist",
                        "Twist (°)",
                        value=twist_val,
                        min=-180,
                        max=180,
                        step=1,
                        update_on="blur",
                    ),
                    ui.input_numeric(
                        "rise",
                        "Rise (Å)",
                        value=rise_val,
                        min=0,
                        step=1,
                        update_on="blur",
                    ),
                    ui.input_numeric(
                        "csym", "Csym", value=csym_val, min=1, step=1, update_on="blur"
                    ),
                    col_widths=[4, 4, 4],
                ),
            )
        elif mode in ["amyloid_atlas", "EMDB-helical", "EMDB"]:
            emdb = helicon.dataset.EMDB()
            cols = ["emdb_id", "pdb", "resolution", "twist", "rise", "csym", "title"]
            if mode == "amyloid_atlas":
                emd_ids = emdb.amyloid_atlas_ids()
            elif mode == "EMDB-helical":
                emd_ids = emdb.helical_structure_ids()
            else:
                emd_ids = emdb.emd_ids
                cols = ["emdb_id", "pdb", "resolution", "title"]
            df = emdb.meta.loc[emdb.meta["emd_id"].isin(emd_ids)].copy()
            df["resolution"] = df["resolution"].astype(float)
            if "twist" in df:
                df["twist"] = df["twist"].astype(float)
            if "rise" in df:
                df["rise"] = df["rise"].astype(float)
            df = df[cols].round(3)
            df["rank"] = np.inf
            df = df[["rank"] + cols]
            emdb_df_original.set(df)
            return None
        return None

    @render.data_frame
    @reactive.event(emdb_df)
    def display_emdb_dataframe():
        df = emdb_df()
        if df is None or df.empty:
            return None
        return render.DataGrid(
            df,
            selection_mode="rows",
            filters=True,
            editable=True,
            height="40vh",
            width="100%",
        )

    @render.ui
    def map_actions_ui():
        """One row: take the table, drop the selection, make the previews.

        The grid selects a row at a time, or a run of them with shift; there
        is no select-all in it, and filtering the table down to a family of
        structures and then taking all of them is how this tab is used. The
        previews carry their count, because asking for 751 of them is 751
        downloads and that should be a decision rather than a surprise.

        The selection buttons belong to the three table modes; the previews
        belong to every mode, including the single map a URL or an upload
        gives. They share a row so the three do not each take one.

        "Select all displayed" carries no count on purpose. The count would
        have to come from the grid's filtered view, and read here it went
        stale -- it changed only after the button was pressed, which is worse
        than no number at all. The grid prints "Viewing rows 1 through N of M"
        directly above these buttons, so the number is already on screen; the
        click itself reads the view, and that read is always current.
        """
        buttons = []
        df = emdb_df()
        table_mode = input.input_mode_maps() in (
            "amyloid_atlas",
            "EMDB-helical",
            "EMDB",
        )
        if table_mode and df is not None and not df.empty:
            buttons += [
                ui.input_action_button(
                    "select_all_emdb_rows",
                    "Select all displayed",
                    class_="btn-sm btn-outline-secondary",
                ),
                ui.input_action_button(
                    "clear_emdb_rows",
                    "Clear selection",
                    class_="btn-sm btn-outline-secondary",
                ),
            ]
        n_maps = len(maps())
        if n_maps:
            buttons.append(
                ui.input_action_button(
                    "generate_xyz_projections",
                    "Generate x/y/z projections (%d map%s)"
                    % (n_maps, "" if n_maps == 1 else "s"),
                    class_="btn-sm btn-outline-secondary",
                )
            )
        if not buttons:
            return None
        return ui.div(
            *buttons,
            style="display: flex; flex-wrap: wrap; gap: 6px; margin: 6px 0;",
        )

    @reactive.effect
    @reactive.event(input.select_all_emdb_rows)
    async def _select_all_emdb_rows():
        rows = list(display_emdb_dataframe.data_view_rows())
        if not rows:
            return
        await display_emdb_dataframe.update_cell_selection(
            {"type": "row", "rows": rows}
        )

    @reactive.effect
    @reactive.event(input.clear_emdb_rows)
    async def _clear_emdb_rows():
        await display_emdb_dataframe.update_cell_selection({"type": "row", "rows": []})

    # -- Image loading --

    @reactive.effect
    @reactive.event(input.input_mode_images, input.upload_images)
    def _load_images_upload():
        req(input.input_mode_images() == "upload")
        fi = input.upload_images()
        req(fi)
        try:
            data, apix = compute.get_images_from_file(fi[0]["datapath"])
        except Exception as e:
            logger.error("Image upload failed: %s", e)
            ui.modal_show(
                ui.modal(
                    "Failed to read uploaded images: " + str(e),
                    title="Error",
                    easy_close=True,
                    footer=None,
                )
            )
            return
        images_all.set(data)
        image_size.set(min(data.shape))
        image_apix.set(apix)
        project.apix.set(apix)

    @reactive.effect
    @reactive.event(input.input_mode_images, input.url_images)
    def _load_images_url():
        req(input.input_mode_images() == "url")
        req(len(input.url_images()) > 0)
        try:
            data, apix = compute.get_images_from_url(input.url_images())
        except Exception as e:
            logger.error("Image URL download failed: %s", e)
            ui.modal_show(
                ui.modal(
                    "Failed to download images: " + str(e),
                    title="Error",
                    easy_close=True,
                    footer=None,
                )
            )
            return
        images_all.set(data)
        image_size.set(min(data.shape))
        image_apix.set(apix)
        project.apix.set(apix)

    @reactive.effect
    @reactive.event(images_all, input.ignore_blank)
    def _build_displayed_images():
        req(len(images_all()))
        data = images_all()
        n = len(data)
        ny, nx = data[0].shape[:2]
        images = [data[i] for i in range(n)]
        seq = np.arange(n, dtype=int)
        if input.ignore_blank():
            included = [i for i in seq if np.max(images[i]) > np.min(images[i])]
            images = [images[i] for i in included]
        else:
            included = list(seq)
        displayed_image_ids.set(included)
        displayed_images.set(images)
        displayed_image_title.set(
            "%d/%d images | %dx%d pixels | %s Å/pixel"
            % (len(images), n, nx, ny, image_apix())
        )
        displayed_image_labels.set([str(i + 1) for i in included])

    # -- Selected image processing --

    @reactive.effect
    @reactive.event(
        input.select_image,
        images_all,
        input.lp_angst_x,
        input.hp_angst_x,
        input.aniso_ratio_xy,
    )
    def _update_selected_images():
        sel = input.select_image()
        if not sel or len(displayed_images()) == 0:
            return
        images = [displayed_images()[i] for i in sel if i < len(displayed_images())]
        apix = image_apix()
        lp_x = input.lp_angst_x()
        hp_x = input.hp_angst_x()
        ratio = input.aniso_ratio_xy()
        if lp_x > 0 or hp_x > 0:
            lp_frac = 2 * apix / lp_x if lp_x > 0 else -1
            hp_frac = 2 * apix / hp_x if hp_x > 0 else -1
            images = [
                compute.anisotropic_low_high_pass_filter(
                    img,
                    low_pass_fraction_x=lp_frac,
                    high_pass_fraction_x=hp_frac,
                    ratio=ratio,
                )
                for img in images
            ]
        selected_images_original.set(images)
        selected_images_labels.set(
            [
                displayed_image_labels()[i]
                for i in sel
                if i < len(displayed_image_labels())
            ]
        )

    # ── Per-image transforms ────────────────────────────────────────
    # Several class averages can be searched jointly, and each sits at its own
    # angle and height: on ten good EMPIAR-10940 classes the rotations span
    # -20.1 to +10.2 degrees. So every selected image has its own transform,
    # held here keyed by the image's label, and its own card of four sliders --
    # one card for a single image, one per image for several, and no other
    # controls.
    #
    # There used to be a second set, shared sliders for a single image that
    # also acted as a "nudge" on every image once several were selected. Those
    # sliders were hidden in that mode, but a removed control keeps its last
    # value on the server, so the first image's rotation was silently added to
    # every image selected after it -- the new image appeared to copy the old
    # one's transform.

    def _pi_id(kind, key, generation):
        return "hp_pi_%s_%s_%d" % (kind, re.sub(r"\W", "_", str(key)), generation)

    def _input_or(name, default=0.0):
        """Read a dynamically created input, or a default before it exists.

        Reading an unset input registers the dependency and then raises
        SilentException, so catching it here still means the caller re-runs
        once the input appears or the user changes it.
        """
        try:
            value = input[name]()
        except SilentException:
            return default
        return default if value is None else value

    def _next_generation():
        transform_generation[0] += 1
        return transform_generation[0]

    def _selected_by_key():
        """The selected images keyed by label, in selection order."""
        labels = list(selected_images_labels())
        images = list(selected_images_original())
        if len(labels) != len(images):
            labels = [str(i + 1) for i in range(len(images))]
        return dict(zip(labels, images))

    def _auto_transforms(keys, images_by_key):
        """The automatic transform for these images, each measured on its own."""
        made = {}
        for key in keys:
            image = images_by_key[key]
            auto = helix_transform.auto_transform([image], cache=auto_transform_cache)
            rotation, shift = auto.per_image[0]
            made[key] = helix_transform.ImageTransform(
                rotation=float(rotation),
                shift_y=float(shift),
                crop_size=max(32, min(auto.crop_size, int(image.shape[0]) // 2 * 2)),
                threshold=float(np.min(image)),
                generation=_next_generation(),
            )
        return made

    def _as_controls_hold(state):
        """The transforms as the user has left them, read from their controls."""
        held = {}
        for key, t in state.items():
            held[key] = helix_transform.ImageTransform(
                rotation=float(_input_or(_pi_id("rot", key, t.generation), t.rotation)),
                shift_y=float(_input_or(_pi_id("dy", key, t.generation), t.shift_y)),
                crop_size=int(
                    _input_or(_pi_id("vcrop", key, t.generation), t.crop_size)
                ),
                threshold=float(
                    _input_or(_pi_id("threshold", key, t.generation), t.threshold)
                ),
                generation=t.generation,
            )
        return held

    @reactive.effect
    @reactive.event(selected_images_original)
    def _reconcile_transforms_with_selection():
        """Adding an image transforms that image; nothing else changes.

        Images already selected keep what they had, manual edits included, and
        the card shown is the one for the image just added. Re-running the
        automatic transform over the whole selection on every change is what
        used to throw those edits away.
        """
        images_by_key = _selected_by_key()
        previous = transform_state()
        previous_keys = list(previous)
        index = active_selected_image()
        previous_active = (
            previous_keys[index] if 0 <= index < len(previous_keys) else None
        )
        state, _added, active = helix_transform.reconcile_transforms(
            previous,
            list(images_by_key),
            _as_controls_hold(previous),
            lambda new_keys: _auto_transforms(new_keys, images_by_key),
            previous_active=previous_active,
        )
        active_selected_image.set(active)
        transform_state.set(state)

    @render.ui
    @reactive.event(transform_state)
    def hp_per_image_transform_ui():
        """One transform card per selected image, only the active one shown.

        All the cards stay in the DOM and are hidden with CSS, so each image
        keeps whatever was set on it while you click between them, and no
        effect can read an input that does not exist.
        """
        state = transform_state()
        images_by_key = _selected_by_key()
        if not state or any(key not in images_by_key for key in state):
            return ui.div()
        shown = active_selected_image()
        cards = []
        for i, (key, t) in enumerate(state.items()):
            img = images_by_key[key]
            # The four sliders, with ranges from this image: the crop cannot
            # exceed its own height, a threshold means nothing outside its own
            # range of values, and a shift beyond half the image moves the
            # filament out of it.
            ny_i = int(img.shape[0])
            v_min = float(np.min(img))
            v_max = float(np.max(img))
            v_step = max((v_max - v_min) / 100.0, 1e-3)
            shift_limit = max(1, ny_i // 2)
            cards.append(
                ui.div(
                    ui.card(
                        ui.div(
                            f"Image {key}",
                            style="font-weight: bold; margin-bottom: 4px;",
                        ),
                        ui.layout_columns(
                            ui.input_slider(
                                _pi_id("rot", key, t.generation),
                                "Rotation (°)",
                                min=-90,
                                max=90,
                                value=round(min(max(t.rotation, -90.0), 90.0), 1),
                                step=0.1,
                            ),
                            ui.input_slider(
                                _pi_id("threshold", key, t.generation),
                                "Threshold",
                                min=round(v_min, 3),
                                max=round(v_max, 3),
                                value=round(min(max(t.threshold, v_min), v_max), 3),
                                step=round(v_step, 3),
                            ),
                            ui.input_slider(
                                _pi_id("vcrop", key, t.generation),
                                "Vertical crop size (px)",
                                min=32,
                                max=ny_i,
                                value=int(min(max(t.crop_size, 32), ny_i)),
                                step=2,
                            ),
                            ui.input_slider(
                                _pi_id("dy", key, t.generation),
                                "Vertical shift (px)",
                                min=-shift_limit,
                                max=shift_limit,
                                value=int(
                                    round(
                                        min(max(t.shift_y, -shift_limit), shift_limit)
                                    )
                                ),
                                step=1,
                            ),
                            col_widths=6,
                        ),
                    ),
                    class_="hp-pi-card",
                    **{"data-pi": str(i)},
                    style="min-width: 360px;"
                    + ("" if i == shown else " display: none;"),
                )
            )
        heading = []
        if len(cards) > 1:
            heading = [
                ui.div(
                    "Per-image transform (click an image to edit it):",
                    style="font-weight: bold; margin-bottom: 4px;",
                )
            ]
        return ui.div(
            *heading,
            *cards,
            style="display: flex; flex-direction: column; gap: 4px;",
        )

    @reactive.effect
    @reactive.event(input.auto_transform)
    def _auto_transform():
        """Straighten and centre every selected image again, on request."""
        images_by_key = _selected_by_key()
        req(len(images_by_key))
        transform_state.set(_auto_transforms(list(images_by_key), images_by_key))

    @reactive.effect
    @reactive.event(input.display_selected_image)
    def _remember_active_selected_image():
        """Which card the user is on, so a re-render can put them back."""
        value = input.display_selected_image()
        if isinstance(value, (list, tuple)):
            value = value[0] if len(value) else None
        try:
            index = int(value)
        except (TypeError, ValueError):
            return
        if 0 <= index < len(selected_images_original()):
            active_selected_image.set(index)

    @reactive.effect
    @reactive.event(input.reset_transform)
    def _reset_transform():
        images_by_key = _selected_by_key()
        req(len(images_by_key))
        transform_state.set(
            {
                key: helix_transform.ImageTransform(
                    rotation=0.0,
                    shift_y=0.0,
                    crop_size=int(img.shape[0]) // 2 * 2,
                    threshold=float(np.min(img)),
                    generation=_next_generation(),
                )
                for key, img in images_by_key.items()
            }
        )

    # A plain effect, deliberately not reactive.event: the per-image inputs are
    # created dynamically, and reactive.event reads every dependency up front,
    # which would raise on inputs that do not exist yet.
    @reactive.effect
    def _transform_crop_images():
        images_by_key = _selected_by_key()
        state = transform_state()
        # the state catches up with a new selection one step later; until it
        # has, there is nothing consistent to draw
        if not images_by_key or list(state) != list(images_by_key):
            return
        transformed = []
        for key, img in images_by_key.items():
            t = state[key]
            transformed.append(
                helix_transform.apply_transform(
                    img,
                    rotation=_input_or(_pi_id("rot", key, t.generation), t.rotation),
                    shift_y=_input_or(_pi_id("dy", key, t.generation), t.shift_y),
                    crop_size=_input_or(
                        _pi_id("vcrop", key, t.generation), t.crop_size
                    ),
                    threshold=_input_or(
                        _pi_id("threshold", key, t.generation), t.threshold
                    ),
                )
            )
        selected_images_thresholded_rotated_shifted_cropped.set(transformed)

    # -- Map loading --

    @reactive.effect
    @reactive.event(
        input.input_mode_maps, input.upload_map, input.twist, input.rise, input.csym
    )
    def _load_map_upload():
        req(input.input_mode_maps() == "upload")
        fi = input.upload_map()
        req(fi)
        twist = input.twist() if input.twist() is not None else (project.twist() or 0)
        rise = input.rise() if input.rise() is not None else (project.rise() or 0)
        csym = input.csym() if input.csym() is not None else (project.csym() or 1)
        m_info = compute.MapInfo(
            filename=fi[0]["datapath"],
            twist=twist,
            rise=rise,
            csym=csym,
            label=fi[0]["name"],
        )
        maps.set([m_info])

    @reactive.effect
    @reactive.event(
        input.input_mode_maps, input.url_map, input.twist, input.rise, input.csym
    )
    def _load_map_url():
        req(input.input_mode_maps() == "url")
        req(len(input.url_map()) > 0)
        url_val = input.url_map()
        label = url_val.split("/")[-1].split(".")[0]
        twist = input.twist() if input.twist() is not None else (project.twist() or 0)
        rise = input.rise() if input.rise() is not None else (project.rise() or 0)
        csym = input.csym() if input.csym() is not None else (project.csym() or 1)
        m_info = compute.MapInfo(
            url=url_val, twist=twist, rise=rise, csym=csym, label=label
        )
        maps.set([m_info])

    # -- EMDB selection --

    @reactive.effect
    @reactive.event(
        emdb_df_original,
        input.use_curated_helical_parameters,
        input.show_pdb,
        input.show_twist_star,
    )
    def _update_emdb_df():
        df_orig = emdb_df_original()
        req(df_orig is not None and not df_orig.empty)
        df_updated = df_orig.copy()
        if not input.show_pdb():
            df_updated = df_updated.drop(columns=["pdb"], errors="ignore")
        target_cols = list(df_updated.columns)
        if (
            input.use_curated_helical_parameters()
            and "twist" in df_updated
            and "rise" in df_updated
        ):
            url = "https://raw.githubusercontent.com/jianglab/EMDB_helical_parameter_curation/refs/heads/main/EMDB_validation.csv"
            df_curated = pd.read_csv(url)
            df_curated = df_curated[df_curated["emdb_id"].isin(df_updated["emdb_id"])]
            df_curated = df_curated.rename(
                columns={
                    "twist_validated (°)": "twist",
                    "rise_validated (Å)": "rise",
                    "csym_validated": "csym",
                }
            )
            df_curated = df_curated[["emdb_id", "twist", "rise", "csym"]]
            df_updated = df_updated.merge(
                df_curated, on="emdb_id", how="left", suffixes=("", "_curated")
            )
            df_updated["twist"] = df_updated["twist_curated"].combine_first(
                df_updated["twist"]
            )
            df_updated["rise"] = df_updated["rise_curated"].combine_first(
                df_updated["rise"]
            )
            df_updated["csym"] = df_updated["csym_curated"].combine_first(
                df_updated["csym"]
            )
            df_updated["twist"] = pd.to_numeric(
                df_updated["twist"], errors="coerce"
            ).round(3)
            df_updated["rise"] = pd.to_numeric(
                df_updated["rise"], errors="coerce"
            ).round(3)
            df_updated = df_updated[target_cols]
        if input.show_twist_star() and "twist" in df_updated and "rise" in df_updated:
            rise = df_updated["rise"].astype(float).abs()
            twist_star = df_updated["twist"].astype(float).abs()
            for n in range(10, 1, -1):
                if n == 2:
                    mask = (
                        (rise * 2 < 5)
                        & (4.5 < rise * 2)
                        & ((360 - twist_star * 2) < 90)
                    )
                    mask |= (rise < 5) & (4.5 < rise) & (abs(360 - twist_star * 2) < 90)
                    twist_star[mask] = abs(360 - twist_star * 2)
                else:
                    mask = (
                        (rise * n < 5)
                        & (4.5 < rise * n)
                        & (abs(360 - twist_star * n) < 90)
                    )
                    twist_star[mask] = abs(360 - twist_star * n)
            cols = df_updated.columns.tolist()
            twist_index = cols.index("twist")
            cols.insert(twist_index, "twist*")
            df_updated["twist*"] = np.round(twist_star, 3)
            df_updated = df_updated.sort_values(by="twist*").reset_index(drop=True)
            df_updated = df_updated[cols]
        emdb_df.set(df_updated.copy())

    @reactive.effect
    @reactive.event(display_emdb_dataframe.cell_selection)
    def _get_map_from_emdb():
        try:
            sel_rows = set(display_emdb_dataframe.cell_selection().get("rows", set()))
        except Exception:
            return
        req(len(sel_rows))
        try:
            view_idx = display_emdb_dataframe.data_view().index
        except Exception:
            return
        sel_idx = [i for i in view_idx if i in sel_rows]
        if not sel_idx:
            return
        df_sel = display_emdb_dataframe.data().iloc[sel_idx]
        maps_tmp = []
        unreadable = []
        for _, row in df_sel.iterrows():
            # Whatever a row holds, reading it must not end the session: the
            # table is a merge of a deposited table and a curated one, and a
            # cell can be blank, a string, or the literal "Cnan" that a merge
            # with no curated value leaves behind.
            try:
                emdb_id = compute.extract_emdb_id(str(row["emdb_id"]))
                m_info = compute.MapInfo(
                    emd_id=emdb_id,
                    twist=compute.as_number(row.get("twist")),
                    rise=compute.as_number(row.get("rise")),
                    csym=compute.as_csym(row.get("csym")),
                    label=emdb_id,
                )
            except Exception as e:
                logger.error("Could not read the EMDB table row %s: %s", dict(row), e)
                unreadable.append(str(row.get("emdb_id", "?")))
                continue
            maps_tmp.append(m_info)
        if unreadable:
            warn(
                "Some rows could not be read",
                "These entries were left out of the map list.",
                unreadable,
            )
        maps.set(maps_tmp)

    def warn(title, what, items):
        """Tell the user what did not work, without ending their session.

        A map that cannot be downloaded, or one whose projection fails, is an
        ordinary event in a search over dozens of EMDB entries -- the entry may
        be withdrawn, the file may be too large, the connection may drop. It
        should cost that map and nothing else, so every such failure ends up
        here rather than in a traceback that disconnects the browser.
        """
        shown = [str(i) for i in items[:10]]
        if len(items) > len(shown):
            shown.append("... and %d more" % (len(items) - len(shown)))
        ui.modal_show(
            ui.modal(
                ui.p(what),
                ui.tags.ul(*[ui.tags.li(s) for s in shown]),
                title=title,
                easy_close=True,
                footer=None,
            )
        )

    # -- Map XYZ projections --

    @reactive.effect
    @reactive.event(maps)
    def _clear_map_xyz_projections():
        # what is on screen belongs to the previous selection
        map_xyz_projections.set([])
        map_xyz_projection_labels.set([])

    @reactive.effect
    @reactive.event(input.generate_xyz_projections)
    def _get_map_xyz_projections():
        req(len(maps()))
        if len(maps()) > MAPS_NEEDING_CONFIRMATION:
            ui.modal_show(
                ui.modal(
                    ui.p(
                        "This downloads %d maps, one after another, and each "
                        "is tens to hundreds of megabytes. The previews are "
                        "not needed to run a search." % len(maps())
                    ),
                    title="Generate projections for %d maps?" % len(maps()),
                    easy_close=True,
                    footer=ui.TagList(
                        ui.input_action_button(
                            "confirm_xyz_projections", "Generate", class_="btn-primary"
                        ),
                        ui.modal_button("Cancel"),
                    ),
                )
            )
            return
        make_map_xyz_projections()

    @reactive.effect
    @reactive.event(input.confirm_xyz_projections)
    def _get_map_xyz_projections_confirmed():
        ui.modal_remove()
        make_map_xyz_projections()

    def make_map_xyz_projections():
        map_xyz_projections.set([])
        images = []
        image_labels = []
        failures = []
        xyz_tag = "".join([s.upper() for s in input.map_projection_xyz_choices()])
        map_xyz_projection_title.set("Map %s projections:" % xyz_tag)
        with ui.Progress(min=0, max=len(maps())) as p:
            p.set(
                message="Generating x/y/z projections",
                detail="This may take a while ...",
            )
            for mi, m in enumerate(maps()):
                p.set(
                    mi,
                    message="%d/%d: x/y/z projecting %s"
                    % (mi + 1, len(maps()), m.label),
                )
                try:
                    tmp_images, tmp_labels = compute.get_one_map_xyz_projects(
                        map_info=m,
                        length_z=input.length_z(),
                        map_projection_xyz_choices=input.map_projection_xyz_choices(),
                    )
                    images += tmp_images
                    image_labels += tmp_labels
                    map_xyz_projection_labels.set(image_labels)
                    map_xyz_projections.set(images)
                except Exception as e:
                    logger.error("Failed to get XYZ projections for %s: %s", m.label, e)
                    failures.append("%s: %s" % (m.label, e))
        if failures:
            warn(
                "Some maps could not be projected",
                "These maps were skipped. The others are unaffected, and the "
                "search can still be run.",
                failures,
            )

    # -- Compare projections --

    @reactive.effect
    @reactive.event(input.compare_projections)
    def _compare_projections():
        req(len(maps()))
        req(len(selected_images_thresholded_rotated_shifted_cropped()))
        query_imgs = list(selected_images_thresholded_rotated_shifted_cropped())
        query_lbls = list(selected_images_labels()) or ["query"] * len(query_imgs)
        query_apix = image_apix()
        rescale = input.rescale_apix()
        length_xy_factor = input.length_xy()
        match_sf = input.match_sf()
        projection_method = input.projection_method()
        scale_range = input.scale_range() / 100.0
        active_maps = [m for m in maps() if compute.has_twist(m)]
        no_twist = [m.label for m in maps() if not compute.has_twist(m)]
        if not active_maps:
            warn(
                "Nothing to search",
                "None of the selected maps has a helical twist, so no side "
                "projection can be made. Set a twist, or select maps whose "
                "helical parameters are known.",
                no_twist,
            )
            return
        errors = {}

        # In gaussian mode the queries become gaussians too, once for the whole
        # search: every map is then matched by an integral over mixtures rather
        # than by correlating images. If a query cannot be fitted -- a blank
        # image, or one whose background leaves nothing above it -- the search
        # falls back to the pixel aligner rather than refusing to run.
        query_fits = None
        if projection_method == "gaussian":
            try:
                query_fits = map_gauss_fit.fit_queries(query_imgs, query_apix)
            except Exception as e:
                logger.warning("Could not fit the query images to gaussians: %s", e)
                query_fits = None
        results = []
        with ui.Progress(min=0, max=len(active_maps)) as p:
            p.set(
                message="Generating side projections",
                detail="This may take a while ...",
            )
            t0 = time()
            # Sized by memory as well as by cores: each map in flight holds a
            # volume of its own, so a pool of one-per-core asks for several
            # gigabytes at once on a many-core machine.
            n_workers = compute.projection_workers(len(active_maps))
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(
                        compute.symmetrize_project_align_one_map,
                        m,
                        query_imgs,
                        query_lbls,
                        query_apix,
                        rescale,
                        length_xy_factor,
                        match_sf,
                        0,
                        scale_range,
                        projection_method,
                        query_fits,
                    ): m
                    for m in active_maps
                }
                for f in as_completed(futures):
                    # f.result() re-raises whatever the worker raised, and an
                    # exception escaping a reactive effect disconnects the
                    # browser. A map that fails is one map missing from the
                    # results, which is what the warning below reports.
                    try:
                        m_info, res = f.result()
                    except Exception as e:
                        m_info, res = futures[f], None
                        logger.error("Failed to search %s: %s", m_info.label, e)
                        errors[m_info.label] = str(e)
                    t1 = time()
                    results.append((m_info, res))
                    n_done = len(results)
                    remaining = (len(futures) - n_done) / max(n_done, 1) * (t1 - t0)
                    p.set(
                        n_done,
                        message="%d/%d: symmetrizing/projecting/matching %s"
                        % (n_done, len(active_maps), m_info.label),
                        detail="%s remaining" % helicon.timedelta2string(remaining),
                    )
        failed = [m_info.label for m_info, res in results if res is None]
        good = [res for _, res in results if res is not None]
        if no_twist:
            # Skipped quietly: a toast that fades, not a dialog to dismiss --
            # and shown now rather than when the search began, so it is on
            # screen when the user turns to the results instead of during a
            # minute of progress bar.
            ui.notification_show(
                "Skipped %d map%s with no helical twist"
                % (len(no_twist), "" if len(no_twist) == 1 else "s"),
                duration=15,
                type="warning",
            )
        if failed:
            warn(
                "Some maps could not be searched",
                "No side projection could be made for these maps, so they are "
                "not in the results below.",
                [
                    "%s: %s" % (label, errors[label]) if label in errors else label
                    for label in failed
                ],
            )
        # The gaussian search finds the right map as often as the pixel route
        # and much faster, but it does not vary scale, and sometimes the pixel
        # aligner's placement fits the projection visibly better. Sometimes it
        # fits far worse. So the matches a user actually looks at are offered
        # the pixel placement and keep whichever sits better on the projection
        # -- see refine_placement_for_display. The scores keep their ranking so
        # the list stays internally comparable.
        #
        # Including a single result: one map selected is precisely when the
        # placement is studied rather than the ranking, and requiring more
        # than one left that case showing the unscaled placement.
        if query_fits is not None and len(good):
            n_polish = max(1, POLISHED_PAIRS_FOR_DISPLAY // max(1, len(query_imgs)))
            order = sorted(range(len(good)), key=lambda i: -good[i][4])
            order = order[: min(n_polish, len(order))]
            with ui.Progress(min=0, max=len(order)) as p:
                p.set(message="Placing the top matches", detail="for display")
                for n, i in enumerate(order):
                    try:
                        good[i] = compute.refine_placement_for_display(
                            good[i], query_imgs, scale_range
                        )
                    except Exception as e:
                        logger.warning("Could not re-place %s: %s", good[i][8], e)
                    p.set(n + 1)

        map_side_projections_with_alignments.set(good)

    @reactive.effect
    @reactive.event(
        map_side_projections_with_alignments,
        input.sort_map_side_projections_by,
        input.hide_query_image,
    )
    def _update_side_projections_display():
        req(len(map_side_projections_with_alignments()))
        work = list(map_side_projections_with_alignments())
        if input.sort_map_side_projections_by() == "similarity score":
            work = sorted(work, key=lambda x: -x[4])
        df = emdb_df()
        if df is not None:
            df["rank"] = np.inf
        displayed = []
        labels = []
        links = []
        for i, item in enumerate(work):
            (
                flip,
                scale,
                rot_ang,
                shift_c,
                score,
                aligned_img,
                query_lbl,
                proj_img,
                proj_lbl,
            ) = item
            if df is not None and proj_lbl in df["emdb_id"].values:
                row_index = df.index[df["emdb_id"] == proj_lbl][0]
                df.loc[row_index, "rank"] = i + 1
            scale_r = round(scale, 3)
            rot_r = round(rot_ang, 1)
            if not input.hide_query_image():
                displayed.append(aligned_img)
                labels.append(
                    "%d/%d: %s%s%s%s%s"
                    % (
                        i + 1,
                        len(work),
                        query_lbl,
                        "|vflip" if flip else "",
                        "|%s" % scale_r if scale_r != 1 else "",
                        "|%s" % rot_r,
                        "°",
                    )
                )
                links.append("")
            displayed.append(proj_img)
            labels.append("%d/%d: %s|score=%.3f" % (i + 1, len(work), proj_lbl, score))
            if proj_lbl.startswith("emd_") or proj_lbl.startswith("EMD-"):
                num = proj_lbl.split("_")[-1].split("-")[-1]
                links.append("https://www.ebi.ac.uk/emdb/EMD-%s" % num)
            else:
                links.append("")
        map_side_projections_displayed.set(displayed)
        map_side_projection_labels.set(labels)
        map_side_projection_links.set(links)
        if df is not None:
            emdb_df.set(df.copy())

    @reactive.effect
    @reactive.event(input.map_xyz_projection_display_size)
    def _update_xyz_display_size():
        map_xyz_projection_display_size.set(input.map_xyz_projection_display_size())

    @reactive.effect
    @reactive.event(input.map_side_projection_vertical_display_size)
    def _update_side_display_size():
        map_side_projection_vertical_display_size.set(
            input.map_side_projection_vertical_display_size()
        )

    @reactive.effect
    @reactive.event(input.select_top_n_button)
    async def _select_top_n():
        req(len(map_side_projections_with_alignments()))
        df = display_emdb_dataframe.data()
        req(len(df))
        n = input.select_top_n()
        rank_col = df.columns.get_loc("rank")
        await display_emdb_dataframe.update_sort([{"col": rank_col, "desc": False}])
        await display_emdb_dataframe.update_filter([{"col": rank_col, "value": (1, n)}])
        df_view = display_emdb_dataframe.data_view()
        row_indices = list(df_view.index)
        cols = tuple(range(len(df_view.columns)))
        await display_emdb_dataframe.update_cell_selection(
            {"type": "row", "rows": row_indices, "cols": cols}
        )
