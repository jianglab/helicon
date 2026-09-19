"""HelicalProjection tab — compare 2D images with helical structure projections.

Ported from HelicalProjection.git and adapted to the Shiny module pattern.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time
import logging

import numpy as np
import pandas as pd

import helicon
from shiny import reactive, render, ui, module, req
from shiny.types import SilentException
import plotly.express as px


from ..lib.shared_state import ProjectState
from ..lib import helical_projection_compute as compute
from ..lib import helix_transform

logger = logging.getLogger(__name__)

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
    "projection_method": ("projection_method", "volume"),
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
                            "Side projection from",
                            choices={
                                "volume": "Symmetrized volume",
                                "gaussian": "Gaussian model (faster)",
                            },
                            selected="volume",
                            inline=True,
                        ),
                        ui.help_text(
                            "The gaussian model fits each map once, caches it, "
                            "and expands the helical symmetry on a few hundred "
                            "parameters instead of on voxels. Measured over 24 "
                            "EMDB maps it returns the same best match and runs "
                            "about twice as fast, but it cannot reproduce "
                            "detail finer than the sampling it was fitted at."
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
                    ui.output_ui("hp_shared_transform_ui"),
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
    selected_image_diameter = reactive.value(0)
    selected_images_thresholded_rotated_shifted_cropped = reactive.value([])
    # per-image (rotation degrees, vertical shift pixels), set by Auto Transform
    per_image_transforms = reactive.value([])
    # the crop Auto Transform worked out, so per-image cards can start from it
    auto_crop_size = reactive.value(0)
    # measurements already made, keyed by image content: the transform runs on
    # every change to the selection, and without this each change re-measures
    # every image that was already selected
    auto_transform_cache = {}

    emdb_df_original = reactive.value(None)
    emdb_df = reactive.value(None)

    maps = reactive.value([])
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
            initial_selected_indices=reactive.value([0]),
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

    # -- Auto-rotation/shift/diameter estimation --

    @reactive.effect
    @reactive.event(selected_images_original)
    def _auto_estimate_rotation_shift_diameter():
        req(len(selected_images_original()))
        imgs = selected_images_original()
        ny = int(np.max([img.shape[0] for img in imgs]))
        tmp = np.array(
            [
                helicon.estimate_helix_rotation_center_diameter(
                    img, threshold=np.max(img) * 0.2
                )
                for img in imgs
            ]
        )
        rotation = float(np.mean(tmp[:, 0]))
        shift_y = float(np.mean(tmp[:, 1]))
        diameter = float(np.max(tmp[:, 2]))
        crop_size = int(diameter * 3) // 4 * 4
        min_val = float(np.min([np.min(img) for img in imgs]))
        max_val = float(np.max([np.max(img) for img in imgs]))
        step_val = (max_val - min_val) / 100
        selected_image_diameter.set(diameter)
        ui.update_slider("pre_rotation", value=round(rotation, 1))
        ui.update_slider(
            "shift_y", value=shift_y, min=-crop_size // 2, max=crop_size // 2
        )
        ui.update_slider(
            "vertical_crop_size",
            value=max(32, crop_size),
            min=max(32, min(int(diameter) // 2 * 2, ny // 2)),
            max=ny,
        )
        ui.update_slider(
            "threshold",
            value=min_val,
            min=round(min_val, 3),
            max=round(max_val, 3),
            step=round(step_val, 3),
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
    # -20.1 to +10.2 degrees. One shared rotation cannot straighten them all,
    # so each image gets its own card and the shared sliders act as a nudge on
    # top. This is the pattern denovo3D uses, and both tabs now drive it from
    # helix_transform.

    def _pi_id(kind, i):
        return f"hp_pi_{kind}_{i}"

    _PI_SHARED_ID = {
        "rot": "pre_rotation",
        "dy": "shift_y",
        "vcrop": "vertical_crop_size",
        "threshold": "threshold",
    }

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

    def _per_image_active():
        """True when each selected image carries its own transform."""
        return len(selected_images_original()) > 1

    def _param(kind, i, default=0.0):
        """Effective value of one setting for image ``i``."""
        if _per_image_active():
            return _input_or(_pi_id(kind, i), default)
        return _input_or(_PI_SHARED_ID[kind], default)

    @render.ui
    def hp_shared_transform_ui():
        """The shared sliders, shown only when one image is selected.

        With several selected the per-image card is the manual transform UI --
        it is the one that says which image it edits -- and showing both
        invites edits to a control that image is not reading.
        """
        if _per_image_active():
            return ui.div()
        return ui.layout_columns(
            ui.input_slider(
                "pre_rotation", "Rotation (°)", min=-90, max=90, value=0, step=0.1
            ),
            ui.input_slider(
                "threshold", "Threshold", min=0.0, max=1.0, value=0.0, step=0.01
            ),
            ui.input_slider(
                "vertical_crop_size",
                "Vertical crop size (px)",
                min=32,
                max=512,
                value=128,
                step=2,
            ),
            ui.input_slider(
                "shift_y", "Vertical shift (px)", min=-64, max=64, value=0, step=1
            ),
            col_widths=6,
        )

    @render.ui
    @reactive.event(selected_images_original, per_image_transforms, auto_crop_size)
    def hp_per_image_transform_ui():
        """One transform card per selected image, only the clicked one shown.

        All the cards stay in the DOM and are hidden with CSS, so each image
        keeps whatever was typed into it while you click between them, and no
        effect can read an input that does not exist.
        """
        if not _per_image_active():
            return ui.div()
        labels = list(selected_images_labels())
        images = selected_images_original()
        per = per_image_transforms()
        ny = int(max(img.shape[0] for img in images))
        cards = []
        for i, img in enumerate(images):
            label = labels[i] if i < len(labels) else str(i)
            rot, shift = per[i] if i < len(per) else (0.0, 0.0)
            cards.append(
                ui.div(
                    ui.card(
                        ui.div(
                            f"Image {label}",
                            style="font-weight: bold; margin-bottom: 4px;",
                        ),
                        ui.layout_columns(
                            ui.input_numeric(
                                _pi_id("rot", i),
                                "Rotation (°)",
                                value=round(float(rot), 2),
                                step=0.1,
                                update_on="blur",
                            ),
                            ui.input_numeric(
                                _pi_id("dy", i),
                                "Vertical shift (px)",
                                value=round(float(shift), 2),
                                step=1,
                                update_on="blur",
                            ),
                            ui.input_numeric(
                                _pi_id("vcrop", i),
                                "Vertical crop size (px)",
                                value=int(
                                    auto_crop_size()
                                    or _input_or("vertical_crop_size", ny)
                                ),
                                min=32,
                                step=2,
                                update_on="blur",
                            ),
                            ui.input_numeric(
                                _pi_id("threshold", i),
                                "Threshold",
                                value=float(_input_or("threshold", 0.0)),
                                step=0.01,
                                update_on="blur",
                            ),
                            col_widths=6,
                        ),
                    ),
                    class_="hp-pi-card",
                    **{"data-pi": str(i)},
                    style="min-width: 360px;" + ("" if i == 0 else " display: none;"),
                )
            )
        return ui.div(
            ui.div(
                "Per-image transform (click an image to edit it):",
                style="font-weight: bold; margin-bottom: 4px;",
            ),
            *cards,
            style="display: flex; flex-direction: column; gap: 4px;",
        )

    def _apply_auto_transform():
        """Straighten and centre every selected image, as denovo3D does."""
        images = selected_images_original()
        if not len(images):
            return
        auto = helix_transform.auto_transform(images, cache=auto_transform_cache)
        selected_image_diameter.set(auto.diameter)
        crop = max(32, min(auto.crop_size, auto.ny // 2 * 2))
        auto_crop_size.set(crop)
        if len(images) > 1:
            per_image_transforms.set(auto.per_image)
            for i, (rot, shift) in enumerate(auto.per_image):
                ui.update_numeric(_pi_id("rot", i), value=round(float(rot), 2))
                ui.update_numeric(_pi_id("dy", i), value=round(float(shift), 2))
                ui.update_numeric(_pi_id("vcrop", i), value=crop)
            # the shared boxes stay at zero, where they nudge every image alike
            ui.update_slider("pre_rotation", value=0.0)
            ui.update_slider("shift_y", value=0)
        else:
            per_image_transforms.set([])
            rot, shift = auto.per_image[0]
            ui.update_slider("pre_rotation", value=round(float(rot), 1))
            ui.update_slider("shift_y", value=int(round(float(shift))))
        ui.update_slider("vertical_crop_size", value=crop)

    @reactive.effect
    @reactive.event(input.auto_transform)
    def _auto_transform():
        _apply_auto_transform()

    @reactive.effect
    @reactive.event(selected_images_original)
    def _auto_transform_on_selection():
        """Selecting images transforms them, without waiting to be asked.

        Every route out of this tab needs the filaments horizontal and centred,
        and doing it by hand first was a step with no decision in it. The button
        stays, for re-running it after the selection is changed by other means.
        """
        _apply_auto_transform()

    @reactive.effect
    @reactive.event(input.reset_transform)
    def _reset_transform():
        images = selected_images_original()
        req(len(images))
        ny = int(max(img.shape[0] for img in images))
        per_image_transforms.set([])
        for i in range(len(images)):
            ui.update_numeric(_pi_id("rot", i), value=0.0)
            ui.update_numeric(_pi_id("dy", i), value=0.0)
            ui.update_numeric(_pi_id("vcrop", i), value=ny // 2 * 2)
            ui.update_numeric(_pi_id("threshold", i), value=0.0)
        ui.update_slider("pre_rotation", value=0.0)
        ui.update_slider("shift_y", value=0)
        ui.update_slider("threshold", value=0.0)
        ui.update_slider("vertical_crop_size", value=ny // 2 * 2)

    # A plain effect, deliberately not reactive.event: the per-image inputs are
    # created dynamically, and reactive.event reads every dependency up front,
    # which would raise on inputs that do not exist yet.
    @reactive.effect
    def _transform_crop_images():
        orig = selected_images_original()
        if not orig:
            return
        shared_rot = _input_or("pre_rotation", 0.0)
        shared_shift = _input_or("shift_y", 0.0)
        transformed = []
        for i, img in enumerate(orig):
            thresh_val = _param("threshold", i, 0.0)
            rot_val = _param("rot", i, 0.0)
            shift_val = _param("dy", i, 0.0)
            crop_sz = _param("vcrop", i, img.shape[0])
            if _per_image_active():
                # the shared boxes are a common nudge on top of each image's own
                rot_val += shared_rot
                shift_val += shared_shift
            t_img = helix_transform.apply_transform(
                img,
                rotation=rot_val,
                shift_y=shift_val,
                crop_size=crop_sz,
                threshold=thresh_val,
            )
            transformed.append(t_img)
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
        for _, row in df_sel.iterrows():
            emdb_id = compute.extract_emdb_id(str(row["emdb_id"]))
            twist = row["twist"] if "twist" in row and not pd.isna(row["twist"]) else 0
            rise = row["rise"] if "rise" in row and not pd.isna(row["rise"]) else 0
            csym_str = (
                str(row["csym"]) if "csym" in row and not pd.isna(row["csym"]) else "C1"
            )
            csym = int(csym_str[1:]) if len(csym_str) > 1 else 1
            m_info = compute.MapInfo(
                emd_id=emdb_id, twist=twist, rise=rise, csym=csym, label=emdb_id
            )
            maps_tmp.append(m_info)
        maps.set(maps_tmp)

    # -- Map XYZ projections --

    @reactive.effect
    @reactive.event(maps, input.length_z, input.map_projection_xyz_choices)
    def _get_map_xyz_projections():
        req(len(maps()))
        map_xyz_projections.set([])
        images = []
        image_labels = []
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
        active_maps = [m for m in maps() if abs(m.twist) > 1e-3]
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
                    ): m
                    for m in active_maps
                }
                for f in as_completed(futures):
                    m_info, res = f.result()
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
        twist_zeros = [m.label for m in maps() if abs(m.twist) < 1e-3]
        failed = [m_info.label for m_info, res in results if res is None]
        good = [res for _, res in results if res is not None]
        if twist_zeros:
            ui.modal_show(
                ui.modal(
                    "WARNING: twist=0. Please set twist to a correct value for %s"
                    % " ".join(twist_zeros),
                    title="Twist value error",
                    easy_close=True,
                    footer=None,
                )
            )
        if failed:
            ui.modal_show(
                ui.modal(
                    "WARNING: failed to generate side projection of %s"
                    % " ".join(failed),
                    title="Projection error",
                    easy_close=True,
                    footer=None,
                )
            )
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
