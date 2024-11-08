import pathlib
import numpy as np

from shiny import reactive, req
from shiny.express import input, ui, render
from shinywidgets import render_plotly

import helicon

from . import compute

params = reactive.value(None)

project_root_dir = reactive.value(None)
filepath_classes = reactive.value(None)

data_all = reactive.value(None)
abundance = reactive.value([])
image_size = reactive.value(0)

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

first_point = reactive.Value(None)
second_point = reactive.Value(None)


ui.head_content(ui.tags.title("HelicalClassesOnMicrographs"))
helicon.shiny.setup_ajdustable_sidebar(width="25vw")
ui.tags.style(
    """
    * { font-size: 10pt; padding:0; border: 0; margin: 0; }
    aside {--_padding-icon: 10px;}
    """
)

with ui.sidebar(
    width="25vw", style="display: flex; flex-direction: column; height: 100%;"
):
    with ui.navset_pill(id="tab"):
        with ui.nav_panel("Input Class2D file"):
            with ui.div(id="input_files", style="flex-shrink: 0;"):
                helicon.shiny.file_selection_ui(
                    id="filepath_params",
                    label="Choose a RELION star or cryoSPARC cs file on the server",
                    value=None,
                    width="100%",
                ),
                filepath_params = helicon.shiny.file_selection_server(
                    id="filepath_params", file_types=["_data.star", ".cs"]
                )

                ui.input_task_button("run", label="Run", style="width: 100%;")

            with ui.div(id="class-selection", style="flex-grow: 1; overflow-y: auto;"):
                helicon.shiny.image_select(
                    id="select_classes",
                    label=displayed_class_title,
                    images=displayed_class_images,
                    image_labels=displayed_class_labels,
                    image_size=reactive.value(128),
                    initial_selected_indices=initial_selected_image_indices,
                )

                @reactive.effect
                @reactive.event(input.select_classes)
                def update_selected_images():
                    selected_images.set(
                        [displayed_class_images()[i] for i in input.select_classes()]
                    )
                    selected_image_labels.set(
                        [displayed_class_labels()[i] for i in input.select_classes()]
                    )

        with ui.nav_panel("Parameters"):
            with ui.layout_columns(col_widths=6, style="align-items: flex-start;"):
                ui.input_checkbox("ignore_blank", "Ignore blank classes", value=True)
                ui.input_checkbox(
                    "sort_abundance",
                    "Sort the classes by abundance",
                    value=True,
                )
                ui.input_checkbox(
                    "show_sharable_url",
                    "Show sharable URL",
                    value=False,
                )

            with ui.layout_columns(col_widths=6, style="align-items: flex-start;"):
                ui.input_numeric(
                    "target_apix",
                    "Down scale to pixel size (Å)",
                    min=0,
                    value=5,
                    step=100,
                )

                ui.input_numeric(
                    "low_pass_angstrom", "Low-pass filter (Å)", min=0, value=20, step=10
                )

                ui.input_numeric(
                    "high_pass_angstrom",
                    "High-pass filter (Å)",
                    min=0,
                    max=10000,
                    value=0,
                    step=100,
                )

            with ui.layout_columns(col_widths=6, style="align-items: flex-start;"):
                ui.input_numeric(
                    "plot_height", "Plot height (pixel)", min=128, value=640, step=32
                )


title = "WhereIsMyClass: map 2D classes to helical tube/filament images"
ui.h1(title, style="font-weight: bold;")

with ui.layout_columns(col_widths=(5, 7), style="height: 100vh; overflow-y: auto;"):
    with ui.div():

        @render.ui
        @reactive.event(selected_images)
        def display_selected_images():
            return helicon.shiny.image_gallery(
                id="display_selected_image",
                label=reactive.value("Selected classe(s):"),
                images=selected_images,
                image_labels=selected_image_labels,
            )

        @render.data_frame
        @reactive.event(params, input.select_classes)
        def display_helices_dataframe():
            df = params()
            # Group df by helixID and create a summary dataframe
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

            if len(input.select_classes()):
                selected_classes = [
                    int(displayed_class_ids()[i]) + 1 for i in input.select_classes()
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

            # Use the summary dataframe for display
            df = summary_df
            return render.DataGrid(
                summary_df,
                selection_mode="row",
                filters=True,
                height="30vh",
                width="100%",
            )

        helicon.shiny.image_select(
            id="classes_selected_helices",
            label=reactive.value("Classes assigned to selected helices"),
            images=displayed_helices_class_images,
            image_labels=displayed_helices_class_labels,
            image_size=reactive.value(128),
            enable_selection=False,
        )

    with ui.div():
        with ui.div(id="div_marked_classes", style="display: none;"):
            ui.input_checkbox_group(
                "marked_helices_classes",
                label=None,
                choices=[],
                inline=True,
            )
            ui.input_action_button(
                "select_all_marked_helices_classes",
                label="Select all",
            )
            ui.input_action_button(
                "unselect_all_marked_helices_classes",
                label="Unselect all",
            )

        with ui.div(
            id="div_display_micrograph", style="display: flex; justify-content: center;"
        ):

            @render_plotly
            @reactive.event(displayed_micrograph_data, input.plot_height)
            def display_micrograph():
                req(displayed_micrograph_data() is not None)

                fig = compute.plot_micrograph(
                    micrograph=displayed_micrograph_data(),
                    title=f"{displayed_micrograph_filename().name}",
                    apix=input.target_apix(),
                    plot_height=input.plot_height(),
                )

                def plot_micrograph_on_click(trace, points, selector):
                    if selector.shift == True:
                        first_point.set((points.xs[0], points.ys[0]))
                    else:
                        first_point.set(None)
                        second_point.set(None)

                def plot_micrograph_on_hover(trace, points, selector):
                    if first_point() is None:
                        second_point.set(None)
                        return

                    if selector.shift == True:
                        second_point.set((points.xs[0], points.ys[0]))

                for data in fig.data:
                    if data.name == "image":
                        data.on_click(plot_micrograph_on_click)
                        data.on_hover(plot_micrograph_on_hover)

                return fig

    ui.HTML(
        "<i><p>Developed by the <a href='https://jiang.bio.purdue.edu/helicon' target='_blank'>Jiang Lab</a>. Report issues to <a href='https://github.com/jianglab/helicon/issues' target='_blank'>helicon@GitHub</a>.</p></i>"
    )


@reactive.effect
@reactive.event(input.run)
def get_params_from_file():
    filepath = filepath_params()
    req(len(filepath))
    req(pathlib.Path(filepath).exists())

    project_root_dir.set(compute.get_project_root_dir(filepath))
    filepath_classes.set(compute.get_class_file(filepath))

    msg = None
    try:
        df = compute.get_class2d_params_from_file(filepath)
        helices = df.groupby(["rlnMicrographName", "rlnHelicalTubeID"])
        for hi, (_, helix) in enumerate(helices):
            l = helix["rlnHelicalTrackLengthAngst"].astype(float).max().round()
            df.loc[helix.index, "length"] = l
            df.loc[helix.index, "helixID"] = hi + 1
        params.set(df)
    except Exception as e:
        print(e)
        msg = str(e)

    if params() is None:
        if msg is None:
            msg = f"failed to read class2D parameters from {filepath}"
        msg = ui.markdown(
            msg.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br><br>")
        )
        m = ui.modal(
            msg,
            title="File read error",
            easy_close=True,
            footer=None,
        )
        ui.modal_show(m)


@reactive.effect
@reactive.event(filepath_classes)
def get_class2d_from_file():
    req(filepath_classes())
    try:
        data, apix = compute.get_class2d_from_file(filepath_classes())
        nx = data.shape[-1]
    except Exception as e:
        print(e)
        data, apix = None, 0
        nx = 0
        m = ui.modal(
            f"failed to read 2D class average images from {filepath_classes()}",
            title="File read error",
            easy_close=True,
            footer=None,
        )
        ui.modal_show(m)
    data_all.set((data, apix))
    image_size.set(nx)


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
        print(e)
        m = ui.modal(
            f"Failed to get class abundance from the provided Class2D parameter and  image files. Make sure that the two files are for the same Class2D job",
            title="Information error",
            easy_close=True,
            footer=None,
        )
        ui.modal_show(m)
        return None

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
        f"{len(included)}/{n} classes | {images[0].shape[1]}x{images[0].shape[0]} pixels | {apix} Å/pixel"
    )
    displayed_class_labels.set(image_labels)


@reactive.effect
def get_selected_helices():
    helices_selected = display_helices_dataframe.data_view(selected=True)
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
        return

    displayed_micrograph_filename.set(micrograph)

    helix_ids = [int(helix["helixID"])]
    classe_ids = list(map(int, str(helix.classes).split(",")))

    displayed_helix_ids.set(helix_ids)
    displayed_helices_class_ids.set(classe_ids)


@reactive.effect
@reactive.event(
    displayed_micrograph_filename,
    input.target_apix,
    input.low_pass_angstrom,
    input.high_pass_angstrom,
)
def get_micrograph():
    req(displayed_micrograph_filename())
    data, apix, apix_original = compute.get_micrograph(
        filename=displayed_micrograph_filename(),
        target_apix=input.target_apix(),
        low_pass_angstrom=input.low_pass_angstrom(),
        high_pass_angstrom=input.high_pass_angstrom(),
    )
    displayed_micrograph_data.set(data)
    displayed_micrograph_apix_original.set(apix_original)


@render.ui
@reactive.event(displayed_helices_class_ids)
def update_checbox_group_marked_helices_classes():
    req(len(displayed_helices_class_ids()))
    choices = [str(class_id) for class_id in displayed_helices_class_ids()]
    ui.update_checkbox_group(
        id="marked_helices_classes",
        label="Mark these classes (ordered in decreasing abundance):",
        choices=choices,
        selected=choices,
    )
    return ui.tags.script(
        "document.getElementById('div_marked_classes').style.display = 'block';"
    )


@reactive.effect
@reactive.event(input.select_all_marked_helices_classes)
def action_select_all_mark_all_helices_classes():
    req(len(displayed_helices_class_ids()))
    choices = [str(class_id) for class_id in displayed_helices_class_ids()]
    ui.update_checkbox_group(
        id="marked_helices_classes",
        label="Mark these classes (ordered in decreasing abundance):",
        choices=choices,
        selected=choices,
    )


@reactive.effect
@reactive.event(input.unselect_all_marked_helices_classes)
def action_unselect_all_mark_all_helices_classes():
    choices = [str(class_id) for class_id in displayed_helices_class_ids()]
    ui.update_checkbox_group(
        id="marked_helices_classes",
        label="Mark these classes (ordered in decreasing abundance):",
        choices=choices,
        selected=[],
    )


@reactive.effect
@reactive.event(input.marked_helices_classes)
def update_displayed_helices_classes_xys():
    helix_ids = displayed_helix_ids()
    classe_ids = input.marked_helices_classes()
    apix = displayed_micrograph_apix_original()

    xys = {}
    for helix_id in helix_ids:
        for class_id in classe_ids:
            if class_id not in classe_ids:
                continue
            mask = (params()["helixID"] == helix_id) & (
                params()["rlnClassNumber"] == int(class_id)
            )
            x = params().loc[mask, "rlnCoordinateX"].values * apix
            y = params().loc[mask, "rlnCoordinateY"].values * apix
            xys[(helix_id, class_id)] = dict(x=x, y=y)

    displayed_helices_classes_xys.set(xys)


@reactive.effect
@reactive.event(displayed_helices_class_ids)
def get_selected_helix_classes():
    class_ids = displayed_helices_class_ids()
    req(len(class_ids))

    data, apix = data_all()
    n = len(data)
    images = [data[i] for i in range(n)]

    class_images = [images[i - 1] for i in class_ids]
    image_labels = [f"{i}" for i in class_ids]
    displayed_helices_class_images.set(class_images)
    displayed_helices_class_labels.set(image_labels)


@reactive.effect
@reactive.event(displayed_helices_classes_xys)
def mark_classes_on_micrograph():
    req(display_micrograph)
    compute.mark_classes_on_helices(
        fig=display_micrograph.widget,  # plotly figure
        helices=displayed_helices_classes_xys(),
        marker_size=10,
    )


@reactive.effect
@reactive.event(first_point, second_point, ignore_none=False)
def display_distance_measurement_ui():
    compute.draw_distance_measurement(
        fig=display_micrograph.widget,
        first_point=first_point(),
        second_point=second_point(),
    )
