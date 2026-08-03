"""whereIsMyClass tab — map 2D class images to helical tube/filament micrographs.

Faithfully ported from src/helicon/webApps/whereIsMyClass/app.py as a Shiny module.
"""

from __future__ import annotations

import logging
import pathlib

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

import helicon
from shiny import reactive, ui, module, req, render
from shinywidgets import render_plotly, render_widget, output_widget

from ..lib.shared_state import ProjectState

from ..lib import whereismyclass_compute as compute

logger = logging.getLogger(__name__)

BOOKMARK_DEFAULTS = {
    "input_mode": ("wimc_input_mode", "file selector"),
    "url_star": ("wimc_url_star", ""),
    "ignore_blank": ("wimc_ignore_blank", True),
    "sort_abundance": ("wimc_sort_abundance", True),
    "show_sharable": ("wimc_show_sharable_url", False),
    "rise": ("wimc_rise", 4.75),
    "target_apix": ("wimc_target_apix", 5),
    "low_pass": ("wimc_low_pass_angstrom", 20),
    "high_pass": ("wimc_high_pass_angstrom", 0),
    "max_len": ("wimc_max_len", -1),
    "max_pair_dist": ("wimc_max_pair_dist", -1),
    "bins": ("wimc_bins", 100),
    "plot_height": ("wimc_plot_height", 640),
}


@module.ui
def where_is_my_class_tab_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.navset_pill(
                ui.nav_panel(
                    "Inputs",
                    ui.input_radio_buttons(
                        "wimc_input_mode",
                        "Input source:",
                        choices=["file selector", "url"],
                        selected="file selector",
                        inline=True,
                    ),
                    ui.div(
                        ui.panel_conditional(
                            "input.wimc_input_mode === 'file selector'",
                            output_widget("wimc_filepath_params"),
                        ),
                        ui.panel_conditional(
                            "input.wimc_input_mode === 'url'",
                            ui.input_text(
                                "wimc_url_star",
                                "URL for a RELION star or cryoSPARC cs file",
                                value="",
                                placeholder="https://...",
                            ),
                        ),
                        ui.input_task_button(
                            "wimc_run",
                            label="Run",
                            style="width: 100%; margin-top: 5px;",
                        ),
                        id="wimc_input_files",
                        style="flex-shrink: 0;",
                    ),
                    ui.div(
                        ui.output_ui("wimc_select_classes"),
                        id="wimc_class_selection",
                        style="flex-grow: 1; overflow-y: auto;",
                    ),
                ),
                ui.nav_panel(
                    "Parameters",
                    ui.layout_columns(
                        ui.input_checkbox(
                            "wimc_ignore_blank", "Ignore blank classes", value=True
                        ),
                        ui.input_checkbox(
                            "wimc_sort_abundance",
                            "Sort the classes by abundance",
                            value=True,
                        ),
                        ui.input_checkbox(
                            "wimc_show_sharable_url", "Show sharable URL", value=False
                        ),
                        col_widths=6,
                        style="align-items: flex-start;",
                    ),
                    ui.layout_columns(
                        ui.input_numeric(
                            "wimc_rise",
                            "Helical rise (Å)",
                            min=0.01,
                            max=1000.0,
                            value=4.75,
                            step=0.01,
                            update_on="blur",
                        ),
                        ui.input_numeric(
                            "wimc_target_apix",
                            "Down scale to pixel size (Å)",
                            min=0,
                            value=5,
                            step=100,
                            update_on="blur",
                        ),
                        ui.input_numeric(
                            "wimc_low_pass_angstrom",
                            "Low-pass filter (Å)",
                            min=0,
                            value=20,
                            step=10,
                            update_on="blur",
                        ),
                        ui.input_numeric(
                            "wimc_high_pass_angstrom",
                            "High-pass filter (Å)",
                            min=0,
                            max=10000,
                            value=0,
                            step=100,
                            update_on="blur",
                        ),
                        ui.input_numeric(
                            "wimc_max_len",
                            "Maximal length (Å)",
                            min=-1,
                            value=-1,
                            step=1.0,
                            update_on="blur",
                        ),
                        ui.input_numeric(
                            "wimc_max_pair_dist",
                            "Maximal pair distance (Å) to plot",
                            min=-1,
                            value=-1,
                            step=1.0,
                            update_on="blur",
                        ),
                        ui.input_numeric(
                            "wimc_bins",
                            "Number of histogram bins",
                            min=1,
                            value=100,
                            step=1,
                            update_on="blur",
                        ),
                        ui.input_numeric(
                            "wimc_plot_height",
                            "Plot height (pixel)",
                            min=128,
                            value=640,
                            step=32,
                            update_on="blur",
                        ),
                        col_widths=6,
                        style="align-items: flex-start;",
                    ),
                ),
                id="wimc_sidebar_tabs",
            ),
            width="25vw",
            style="display: flex; flex-direction: column; height: 100%;",
        ),
        ui.h1(
            "WhereIsMyClass: map 2D classes to helical tube/filament images",
            style="font-weight: bold;",
        ),
        ui.layout_columns(
            ui.div(
                ui.output_ui("wimc_display_selected_images"),
                ui.output_data_frame("wimc_helices_dataframe"),
                ui.output_ui("wimc_classes_selected_helices"),
            ),
            ui.div(
                ui.div(
                    ui.input_checkbox_group(
                        "wimc_marked_helices_classes",
                        label=None,
                        choices=[],
                        inline=True,
                    ),
                    ui.input_action_button(
                        "wimc_select_all_marked_helices_classes",
                        label="Select all",
                    ),
                    ui.input_action_button(
                        "wimc_unselect_all_marked_helices_classes",
                        label="Unselect all",
                    ),
                    id="wimc_div_marked_classes",
                    style="display: none;",
                ),
                ui.div(
                    output_widget("wimc_display_micrograph"),
                    id="wimc_div_display_micrograph",
                    style="display: flex; justify-content: center;",
                ),
                output_widget("wimc_pair_distances_histogram"),
                ui.output_ui("update_checkbox_group_marked_helices_classes"),
            ),
            col_widths=(5, 7),
        ),
        ui.HTML(
            "<i><p style='margin:2px 0'>Developed by the <a href='https://jianglab.science.psu.edu/helicon' target='_blank'>Jiang Lab</a>. "
            "Report issues to <a href='https://github.com/jianglab/helicon/issues' target='_blank'>helicon@GitHub</a>.</p></i>"
        ),
    )


@module.server
def where_is_my_class_tab_server(
    input, output, session, project: ProjectState, wimc_filechooser=None
):
    # ── Reactive state (mirrors the original app.py) ──
    params = reactive.value(None)
    project_root_dir = reactive.value(None)
    filepath_classes = reactive.value(None)

    data_all = reactive.value(None)
    abundance = reactive.value([])

    displayed_class_ids = reactive.value([])
    displayed_class_images = reactive.value([])
    displayed_class_title = reactive.value("Select class(es):")
    displayed_class_labels = reactive.value([])
    initial_selected_image_indices = reactive.value([0])
    selected_images = reactive.value([])
    selected_image_labels = reactive.value([])

    displayed_micrograph_filename = reactive.value(None)
    displayed_micrograph_data = reactive.value(None)
    displayed_micrograph_apix_original = reactive.value(0)
    displayed_helix_ids = reactive.value([])
    displayed_helices_class_ids = reactive.value([])
    displayed_helices_class_images = reactive.value([])
    displayed_helices_class_labels = reactive.value([])
    displayed_helices_classes_xys = reactive.value(None)

    df_selected_helices = reactive.value(([], [], 0))
    pair_distances_df_selected = reactive.value([])

    first_point = reactive.Value(None)
    second_point = reactive.Value(None)

    # Track whether micrograph has been rendered at least once
    micrograph_rendered = reactive.value(False)

    @render_widget
    def wimc_filepath_params():
        if wimc_filechooser is not None:
            return wimc_filechooser
        from ipyfilechooser import FileChooser

        fc = FileChooser(
            path=".",
            select_desc="Select",
            show_hidden=False,
            filter_pattern=["*_data.star", "*.cs"],
            title="Select a RELION star or cryoSPARC cs file on the server",
        )
        return fc

    @reactive.effect
    @reactive.event(input.wimc_run)
    def get_params_from_file():
        req(input.wimc_input_mode() == "file selector")
        fc = wimc_filechooser
        req(fc is not None)
        filepath = fc.selected
        req(filepath is not None and len(filepath))
        req(pathlib.Path(filepath).exists())

        project_root_dir.set(compute.get_project_root_dir(filepath))
        filepath_classes.set(compute.get_class_file(filepath))

        try:
            df = compute.get_class2d_params_from_file(filepath)
            helices = df.groupby(["rlnMicrographName", "rlnHelicalTubeID"])
            for hi, (_, helix) in enumerate(helices):
                l = helix["rlnHelicalTrackLengthAngst"].astype(float).max().round()
                df.loc[helix.index, "length"] = l
                df.loc[helix.index, "helixID"] = hi + 1
            params.set(df)
        except Exception:
            logger.error("Failed to get class2d params from file", exc_info=True)
            m = ui.modal(
                f"failed to read class2D parameters from {filepath}",
                title="File read error",
                easy_close=True,
                footer=None,
            )
            ui.modal_show(m)

    # ── Load from URL ──
    @reactive.effect
    @reactive.event(input.wimc_run)
    def get_params_from_url():
        req(input.wimc_input_mode() == "url")
        url = input.wimc_url_star()
        req(url and url.strip())

        # If the "URL" is a local file path (as passed by the file browser),
        # resolve the project root so micrographs and the class-selection
        # checkboxes can be shown, mirroring get_params_from_file().
        local_path = pathlib.Path(url)
        if local_path.exists():
            project_root_dir.set(compute.get_project_root_dir(url))

        try:
            df = compute.get_class2d_params_from_url(url)
            helices = df.groupby(["rlnMicrographName", "rlnHelicalTubeID"])
            for hi, (_, helix) in enumerate(helices):
                l = helix["rlnHelicalTrackLengthAngst"].astype(float).max().round()
                df.loc[helix.index, "length"] = l
                df.loc[helix.index, "helixID"] = hi + 1
            params.set(df)
        except Exception:
            logger.error("Failed to get class2d params from URL", exc_info=True)
            ui.modal_show(
                ui.modal(
                    f"Failed to download class2D parameters from {url}",
                    title="URL download error",
                    easy_close=True,
                    footer=None,
                )
            )
            return

        from urllib.parse import urlparse

        parsed = urlparse(url)
        basename = parsed.path.rsplit("/", 1)[-1]
        class_stem = compute.get_class_file_stem(basename)
        if class_stem:
            class_url = compute.get_class_file_url(url, class_stem)
            try:
                data, apix = compute.get_class2d_from_url(class_url)
                data_all.set((data, apix))
            except Exception:
                logger.warning("Could not download class images from %s", class_url)
                ui.modal_show(
                    ui.modal(
                        f"Class parameters loaded, but class images could not be downloaded from {class_url}",
                        title="Partial download",
                        easy_close=True,
                        footer=None,
                    )
                )

    # ── Load class images from file ──
    @reactive.effect
    @reactive.event(filepath_classes)
    def get_2d_images_from_files():
        req(filepath_classes())
        if type(filepath_classes()) is list:
            data, apix, nx = compute.get_class3d_projections_from_files(
                filepath_classes()
            )
        else:
            try:
                data, apix = compute.get_class2d_from_file(filepath_classes())
                nx = data.shape[-1]
            except Exception:
                logger.error("Failed to read 2D class averages", exc_info=True)
                m = ui.modal(
                    f"failed to read 2D class average images from {filepath_classes()}",
                    title="File read error",
                    easy_close=True,
                    footer=None,
                )
                ui.modal_show(m)
                return
        data_all.set((data, apix))

    # ── Build class gallery ──
    @reactive.effect
    @reactive.event(
        params, data_all, input.wimc_ignore_blank, input.wimc_sort_abundance
    )
    def get_displayed_class_images():
        req(params() is not None)
        req(data_all() is not None)
        data, apix = data_all()
        n = len(data)
        images = [data[i] for i in range(n)]

        try:
            df = params()
            abundance.set(compute.get_class_abundance(df, n))
        except Exception:
            logger.error("Failed to get class abundance", exc_info=True)
            m = ui.modal(
                "Failed to get class abundance from the provided Class2D parameter "
                "and image files. Make sure that the two files are for the same Class2D job",
                title="Information error",
                easy_close=True,
                footer=None,
            )
            ui.modal_show(m)
            return None

        display_seq_all = np.arange(n, dtype=int)
        if input.wimc_sort_abundance():
            display_seq_all = np.argsort(abundance())[::-1]

        if input.wimc_ignore_blank():
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
            f"{len(included)}/{n} classes | {images[0].shape[1]}x{images[0].shape[0]} pixels | {apix} Å/pixel"
        )
        displayed_class_labels.set(image_labels)

    # ── Class selection gallery in sidebar ──
    @render.ui
    def wimc_select_classes():
        return helicon.shiny.image_gallery(
            id=session.ns("wimc_classes_gallery"),
            label=displayed_class_title,
            images=displayed_class_images,
            image_labels=displayed_class_labels,
            image_size=reactive.value(128),
            enable_selection=True,
            allow_multiple_selection=True,
            initial_selected_indices=initial_selected_image_indices,
        )

    @reactive.effect
    @reactive.event(input.wimc_classes_gallery)
    def update_selected_images():
        sel = input.wimc_classes_gallery()
        if sel is None or len(sel) == 0:
            selected_images.set([])
            selected_image_labels.set([])
            return
        selected_images.set([displayed_class_images()[i] for i in sel])
        selected_image_labels.set([displayed_class_labels()[i] for i in sel])

    # ── Main area: selected images display ──
    @render.ui
    @reactive.event(selected_images)
    def wimc_display_selected_images():
        return helicon.shiny.image_gallery(
            id=session.ns("wimc_display_selected_image"),
            label=reactive.value("Selected class(es):"),
            images=selected_images,
            image_labels=selected_image_labels,
        )

    # ── Main area: helix data table ──
    @render.data_frame
    @reactive.event(params, input.wimc_classes_gallery)
    def wimc_helices_dataframe():
        import pandas

        df = params()
        if df is None:
            return render.DataGrid(pandas.DataFrame(), selection_mode="row")
        summary_df = (
            df.groupby("helixID")
            .agg(
                {
                    "length": "first",
                    "rlnClassNumber": lambda x: list(x.value_counts().index),
                    "rlnMicrographName": "first",
                }
            )
            .reset_index()
        )
        summary_df = summary_df.rename(columns={"rlnClassNumber": "classes"})

        sel = input.wimc_classes_gallery()
        if sel and len(sel):
            selected_classes = [int(displayed_class_ids()[i]) + 1 for i in sel]
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
            summary_df,
            selection_mode="row",
            filters=True,
            height="30vh",
            width="100%",
        )

    # ── Compute df_selected_helices when table selection changes ──
    @reactive.effect
    def get_df_selected_helices():
        try:
            df_selected = wimc_helices_dataframe.data_view(selected=True)
        except Exception:
            return
        if df_selected is None or len(df_selected) == 0:
            df_selected_helices.set(([], [], 0))
            return
        df_selected_helixids = df_selected["helixID"].tolist()
        mask = params()["helixID"].astype(int).isin(df_selected_helixids)
        particles = params().loc[mask, :]

        class_indices = [
            int(i) - 1 for i in (input.wimc_marked_helices_classes() or [])
        ]

        helices = compute.select_classes(params=particles, class_indices=class_indices)
        if len(helices):
            filament_lengths = compute.get_filament_length(helices=helices)
            segments_count = np.sum([len(h) for _, h in helices])
        else:
            filament_lengths = []
            segments_count = 0

        df_selected_helices.set((helices, filament_lengths, segments_count))

    # ── Classes assigned to selected helices ──
    @render.ui
    def wimc_classes_selected_helices():
        return helicon.shiny.image_gallery(
            id=session.ns("wimc_helices_class_gallery"),
            label=reactive.value("Classes assigned to selected helices"),
            images=displayed_helices_class_images,
            image_labels=displayed_helices_class_labels,
            image_size=reactive.value(128),
        )

    # ── Micrograph display ──
    @reactive.effect
    def get_selected_helices():
        helices_selected = wimc_helices_dataframe.data_view(selected=True)
        req(project_root_dir() is not None)
        req(len(helices_selected) > 0)

        helix = helices_selected.iloc[0]
        micrograph = project_root_dir() / helix["rlnMicrographName"]
        if not micrograph.exists():
            m = ui.modal(
                f"{str(micrograph)} is not available",
                title="ERROR: micrograph not available",
                easy_close=True,
                footer=None,
            )
            ui.modal_show(m)
            # Clear stale micrograph state so the old plot disappears.
            displayed_micrograph_filename.set(None)
            displayed_micrograph_data.set(None)
            return

        displayed_micrograph_filename.set(micrograph)

        helix_ids = [int(helix["helixID"])]
        classe_ids = list(map(int, str(helix["classes"]).split(",")))

        displayed_helix_ids.set(helix_ids)
        displayed_helices_class_ids.set(classe_ids)

    @reactive.effect
    @reactive.event(
        displayed_micrograph_filename,
        input.wimc_target_apix,
        input.wimc_low_pass_angstrom,
        input.wimc_high_pass_angstrom,
    )
    def get_micrograph():
        req(displayed_micrograph_filename())
        try:
            data, apix, apix_original = compute.get_micrograph(
                filename=displayed_micrograph_filename(),
                target_apix=input.wimc_target_apix(),
                low_pass_angstrom=input.wimc_low_pass_angstrom(),
                high_pass_angstrom=input.wimc_high_pass_angstrom(),
            )
        except Exception as e:
            logger.error(
                "Failed to read micrograph %s: %s",
                displayed_micrograph_filename(),
                e,
                exc_info=True,
            )
            displayed_micrograph_data.set(None)
            ui.modal_show(
                ui.modal(
                    f"Could not read the micrograph file "
                    f"<b>{displayed_micrograph_filename().name}</b>.<br><br>"
                    f"<i>{e}</i>",
                    title="Micrograph read error",
                    easy_close=True,
                    footer=None,
                )
            )
            return
        displayed_micrograph_data.set(data)
        displayed_micrograph_apix_original.set(apix_original)

    @render_plotly
    @reactive.event(displayed_micrograph_data, input.wimc_plot_height)
    def wimc_display_micrograph():
        req(displayed_micrograph_data() is not None)

        fig = compute.plot_micrograph(
            micrograph=displayed_micrograph_data(),
            title=f"{displayed_micrograph_filename().name}",
            apix=input.wimc_target_apix(),
            plot_height=input.wimc_plot_height(),
        )

        def plot_micrograph_on_click(trace, points, selector):
            if selector.shift:
                first_point.set((points.xs[0], points.ys[0]))
            else:
                first_point.set(None)
                second_point.set(None)

        def plot_micrograph_on_hover(trace, points, selector):
            if first_point() is None:
                second_point.set(None)
                return
            if selector.shift:
                second_point.set((points.xs[0], points.ys[0]))

        for data in fig.data:
            if data.name == "image":
                data.on_click(plot_micrograph_on_click)
                data.on_hover(plot_micrograph_on_hover)

        micrograph_rendered.set(True)
        return fig

    # ── Checkbox group for marked helix classes ──
    @render.ui
    @reactive.event(displayed_helices_class_ids)
    def update_checkbox_group_marked_helices_classes():
        req(len(displayed_helices_class_ids()))
        choices = [str(class_id) for class_id in displayed_helices_class_ids()]
        ui.update_checkbox_group(
            id="wimc_marked_helices_classes",
            label="Mark these classes (ordered in decreasing abundance):",
            choices=choices,
            selected=choices,
        )
        return ui.tags.script(
            "document.getElementById('wimc_div_marked_classes').style.display = 'block';"
        )

    @reactive.effect
    @reactive.event(input.wimc_select_all_marked_helices_classes)
    def action_select_all():
        req(len(displayed_helices_class_ids()))
        choices = [str(class_id) for class_id in displayed_helices_class_ids()]
        ui.update_checkbox_group(
            id="wimc_marked_helices_classes",
            label="Mark these classes (ordered in decreasing abundance):",
            choices=choices,
            selected=choices,
        )

    @reactive.effect
    @reactive.event(input.wimc_unselect_all_marked_helices_classes)
    def action_unselect_all():
        choices = [str(class_id) for class_id in displayed_helices_class_ids()]
        ui.update_checkbox_group(
            id="wimc_marked_helices_classes",
            label="Mark these classes (ordered in decreasing abundance):",
            choices=choices,
            selected=[],
        )

    # ── Mark classes on micrograph ──
    @reactive.effect
    @reactive.event(input.wimc_marked_helices_classes)
    def update_displayed_helices_classes_xys():
        helix_ids = displayed_helix_ids()
        classe_ids = input.wimc_marked_helices_classes()
        apix = displayed_micrograph_apix_original()

        xys = {}
        for helix_id in helix_ids:
            for class_id in classe_ids:
                mask = (params()["helixID"] == helix_id) & (
                    params()["rlnClassNumber"] == int(class_id)
                )
                x = params().loc[mask, "rlnCoordinateX"].values * apix
                y = params().loc[mask, "rlnCoordinateY"].values * apix
                xys[(helix_id, class_id)] = dict(x=x, y=y)

        displayed_helices_classes_xys.set(xys)

    @reactive.effect
    @reactive.event(displayed_helices_classes_xys)
    def mark_classes_on_micrograph():
        req(micrograph_rendered())
        req(wimc_display_micrograph.widget is not None)
        compute.mark_classes_on_helices(
            fig=wimc_display_micrograph.widget,
            helices=displayed_helices_classes_xys(),
            marker_size=10,
        )

    # ── Distance measurement ──
    @reactive.effect
    @reactive.event(first_point, second_point, ignore_none=False)
    def display_distance_measurement_ui():
        req(micrograph_rendered())
        req(wimc_display_micrograph.widget is not None)
        compute.draw_distance_measurement(
            fig=wimc_display_micrograph.widget,
            first_point=first_point(),
            second_point=second_point(),
        )

    # ── Helix class images ──
    @reactive.effect
    @reactive.event(displayed_helices_class_ids)
    def get_selected_helix_classes():
        class_ids = displayed_helices_class_ids()
        req(len(class_ids) and data_all() is not None)
        data, _ = data_all()
        n = len(data)
        images = [data[i] for i in range(n)]

        class_images = [images[i - 1] for i in class_ids]
        image_labels = [f"{i}" for i in class_ids]
        displayed_helices_class_images.set(class_images)
        displayed_helices_class_labels.set(image_labels)

    # ── Pair distances histogram ──
    @reactive.effect
    @reactive.event(df_selected_helices)
    def get_pair_lengths_df_selected():
        helices, filament_lengths, _ = df_selected_helices()
        if len(helices):
            dists, _ = compute.compute_pair_distances(helices=helices)
            pair_distances_df_selected.set(dists)
        else:
            pair_distances_df_selected.set([])

    @render_plotly
    @reactive.event(
        pair_distances_df_selected,
        input.wimc_bins,
        input.wimc_max_pair_dist,
        input.wimc_rise,
    )
    def wimc_pair_distances_histogram():
        req(input.wimc_bins() is not None and input.wimc_bins() > 0)
        req(input.wimc_max_pair_dist() is not None)
        req(input.wimc_rise() is not None and input.wimc_rise() > 0)
        fig = getattr(wimc_pair_distances_histogram, "fig", None)
        data = pair_distances_df_selected()

        helices, filament_lengths, _ = df_selected_helices()

        if len(helices):
            class_indices = np.unique(
                np.concatenate([h["rlnClassNumber"] for _, h in helices])
            ).astype(int)
        else:
            class_indices = []

        sel = input.wimc_classes_gallery() or []
        class_indices = [
            str(displayed_class_ids()[i] + 1)
            for i in sel
            if (displayed_class_ids()[i] + 1) in class_indices
        ]
        segment_count = np.sum([len(h) for _, h in helices])
        rise = input.wimc_rise()
        title = (
            f"Pair Distances: Class {' '.join(class_indices)}<br>"
            f"<i>{len(helices)} filaments | {segment_count:,} segments | "
            f"{len(pair_distances_df_selected()):,} segment pairs"
        )

        fig = compute.plot_histogram(
            data=data,
            title=title,
            xlabel="Pair Distance (Å)",
            ylabel="# of Pairs",
            max_pair_dist=input.wimc_max_pair_dist(),
            bins=input.wimc_bins(),
            log_y=True,
            show_pitch_twist=dict(rise=rise, csyms=(1, 2, 3, 4)),
            multi_crosshair=True,
            fig=fig,
        )
        wimc_pair_distances_histogram.fig = fig

        return fig
