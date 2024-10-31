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
                ui.input_text(
                    "filepath_params",
                    "File path for a RELION star or cryoSPARC cs file on the server",
                    value="/Users/wjiang/temp/empiar-10940-tau-easy/data/EMPIAR/Class2D/job010/run_it020_data.star",
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

        @render_plotly
        @reactive.event(displayed_micrograph_data, input.plot_height)
        def display_micrograph():
            req(displayed_micrograph_data() is not None)

            fig = compute.plot_micrograph(
                micrograph=displayed_micrograph_data(),
                title=f"{displayed_micrograph_filename().name}",
                apix=input.target_apix(),
                plot_height=input.plot_height(),
                plot_width=input.plot_height(),
            )
            return fig

        @reactive.effect
        @reactive.event(displayed_helices_classes_xys)
        def mark_classes_on_micrograph():
            req(display_micrograph)
            compute.mark_classes_on_helices(
                fig=display_micrograph.widget,  # plotly figure
                helices=displayed_helices_classes_xys(),
                marker_size=10,
            )

        ui.HTML(
            "<i><p>Developed by the <a href='https://jiang.bio.purdue.edu/helicon' target='_blank'>Jiang Lab</a>. Report issues to <a href='https://github.com/jianglab/helicon/issues' target='_blank'>helicon@GitHub</a>.</p></i>"
        )


@reactive.effect
def get_params_from_file():
    filepath = input.filepath_params()
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
    except Exception as e:
        print(e)
        msg = str(e)
        tmp_params = None
    params.set(df)

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
    assert micrograph.exists()

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


import numpy as np
import pandas as pd
import pathlib

from shiny import reactive
from shiny.express import ui, render

import plotly.graph_objects as go
import plotly.colors


def get_project_root_dir(param_file):
    f = pathlib.Path(param_file)
    if param_file.endswith(".star"):
        return f.parent.parent.parent
    elif param_file.endswith(".cs"):
        return f.parent.parent
    else:
        return None


def get_micrograph(filename, target_apix, low_pass_angstrom, high_pass_angstrom):
    import mrcfile

    with mrcfile.open(filename) as mrc:
        apix = round(float(mrc.voxel_size.x), 4)
        data = mrc.data.squeeze()
    ny, nx = data.shape
    from skimage.transform import resize_local_mean

    new_ny = int(ny * apix / target_apix + 0.5) // 2 * 2
    new_nx = int(nx * apix / target_apix + 0.5) // 2 * 2
    data = resize_local_mean(image=data, output_shape=(new_ny, new_nx))
    import helicon

    if low_pass_angstrom > 0 or high_pass_angstrom > 0:
        low_pass_fraction = (
            2 * target_apix / low_pass_angstrom if low_pass_angstrom > 0 else 0
        )
        high_pass_fraction = (
            2 * target_apix / high_pass_angstrom if high_pass_angstrom > 0 else 0
        )
        data = helicon.low_high_pass_filter(
            data,
            low_pass_fraction=low_pass_fraction,
            high_pass_fraction=high_pass_fraction,
        )
    return data, target_apix, apix


def get_class_file(param_file):
    f = pathlib.Path(param_file)
    if param_file.endswith(".star"):
        return f.parent / (f.stem[:10] + "classes.mrcs")
    elif param_file.endswith(".cs"):
        return f.parent / (f.stem[:7] + "class_averages.mrc")
    else:
        return None


def get_filament_length(helices, particle_box_length=0):
    filement_lengths = []
    for gn, g in helices:
        track_lengths = g["rlnHelicalTrackLengthAngst"].astype(float).values
        length = track_lengths.max() - track_lengths.min() + particle_box_length
        filement_lengths.append(length)
    return filement_lengths


def select_classes(params, class_indices):
    class_indices_tmp = np.array(class_indices) + 1
    mask = params["rlnClassNumber"].astype(int).isin(class_indices_tmp)
    particles = params.loc[mask, :]
    helices = list(particles.groupby(["rlnMicrographName", "rlnHelicalTubeID"]))
    return helices


def get_class_abundance(params, nClass):
    abundance = np.zeros(nClass, dtype=int)
    for gn, g in params.groupby("rlnClassNumber"):
        abundance[int(gn) - 1] = len(g)
    return abundance


def get_class2d_from_file(classFile):
    import mrcfile

    with mrcfile.open(classFile) as mrc:
        apix = float(mrc.voxel_size.x)
        data = mrc.data
    return data, round(apix, 4)


def get_class2d_params_from_file(params_file):
    if params_file.endswith(".star"):
        params = star_to_dataframe(params_file)
    elif params_file.endswith(".cs"):
        params = cs_to_dataframe(params_file)
    required_attrs = np.unique(
        "rlnImageName rlnHelicalTubeID rlnHelicalTrackLengthAngst rlnClassNumber rlnCoordinateX rlnCoordinateY".split()
    )
    missing_attrs = [attr for attr in required_attrs if attr not in params]
    if missing_attrs:
        raise ValueError(f"ERROR: parameters {missing_attrs} are not available")
    return params


def star_to_dataframe(starFile):
    import starfile

    d = starfile.read(starFile, always_dict=True)
    assert (
        "optics" in d and "particles" in d
    ), f"ERROR: {starFile} has {' '.join(d.keys())} but optics and particles are expected"
    data = d["particles"]
    data.attrs["optics"] = d["optics"]
    data.attrs["starFile"] = starFile
    return data


def cs_to_dataframe(cs_file):
    cs = np.load(cs_file)
    data = pd.DataFrame.from_records(cs.tolist(), columns=cs.dtype.names)
    required_attrs = "blob/idx blob/path filament/filament_uid filament/arc_length_A alignments2D/class alignments2D/pose location/center_x_frac location/center_y_frac location/micrograph_shape".split()
    missing_attrs = [attr for attr in required_attrs if attr not in data]
    if missing_attrs:
        msg = f"ERROR: required attrs '{', '.join(missing_attrs)}' are not included in {cs_file}"
        msg += "\nIf the particles in this CryoSPARC job were imported from a RELION star file, you can use the following command to create a star file and load that star file to HelicalPitch:\n"
        msg += "helicon images2star <this cs file> <output star file> --copyParm <original star file>"
        raise ValueError(msg)
    ret = pd.DataFrame()
    ret["rlnImageName"] = (
        (data["blob/idx"].astype(int) + 1).map("{:06d}".format)
        + "@"
        + data["blob/path"].str.decode("utf-8")
    )
    if "micrograph_blob/path" in data:
        ret["rlnMicrographName"] = data["micrograph_blob/path"]
    else:
        ret["rlnMicrographName"] = data["blob/path"].str.decode("utf-8")

    if data["filament/filament_uid"].min() > 1000:
        micrographs = data.groupby(["blob/path"])
        for _, m in micrographs:
            mapping = {
                v: i + 1
                for i, v in enumerate(sorted(m["filament/filament_uid"].unique()))
            }
            ret.loc[m.index, "rlnHelicalTubeID"] = m["filament/filament_uid"].map(
                mapping
            )
    else:
        ret["rlnHelicalTubeID"] = data["filament/filament_uid"].astype(int)

    ret["rlnHelicalTrackLengthAngst"] = (
        data["filament/arc_length_A"].astype(np.float32).values.round(2)
    )

    if (
        "location/center_x_frac" in data
        and "location/center_y_frac" in data
        and "location/micrograph_shape" in data
    ):
        locations = pd.DataFrame(data["location/micrograph_shape"].tolist())
        my = locations.iloc[:, 0]
        mx = locations.iloc[:, 1]
        ret["rlnCoordinateX"] = (
            (data["location/center_x_frac"] * mx).astype(float).round(2)
        )
        ret["rlnCoordinateY"] = (
            (data["location/center_y_frac"] * my).astype(float).round(2)
        )

    # 2D class assignments
    ret["rlnClassNumber"] = data["alignments2D/class"].astype(int) + 1
    return ret


def plot_micrograph(
    micrograph, title, apix, plot_height, plot_width  # np array of shape (ny, nx)
):
    import plotly.graph_objects as go
    import numpy as np

    fig = go.FigureWidget()

    h, w = micrograph.shape

    fig.add_trace(
        go.Heatmap(
            z=micrograph,
            x=np.arange(w) * apix,
            y=np.arange(h) * apix,
            colorscale="Greys",
            showscale=False,  # Removed colorbar
            hoverongaps=False,
            hovertemplate=(
                "x: %{x:.1f} Å<br>"
                + "y: %{y:.1f} Å<br>"
                + "val: %{z:.1f}<br>"
                + "<extra></extra>"
            ),
        )
    )

    layout_params = {
        "title": {
            "text": title,
            "x": 0.5,  # Center the title
            "xanchor": "center",
            "font": {"size": 18},
        },
        "xaxis": {"visible": False, "range": [0, (w - 1) * apix]},
        "yaxis": {
            "visible": False,
            "range": [0, (h - 1) * apix],
            "scaleanchor": "x",
        },
        "plot_bgcolor": "white",
        "showlegend": False,
        "width": plot_width,
        "height": plot_height,
    }

    if plot_width or plot_height:
        if plot_width:
            layout_params["width"] = plot_width
        if plot_height:
            layout_params["height"] = plot_height
    else:
        layout_params["autosize"] = True

    layout_params["margin"] = dict(l=0, r=0, t=50, b=0)

    layout_params["modebar"] = {
        "remove": [],
        "bgcolor": "rgba(255, 255, 255, 0.7)",
    }

    fig.update_layout(**layout_params)

    return fig


def mark_classes_on_helices(
    fig,  # plotly figure
    helices,  #  {(helix_id, class_id): {'x':[], 'y'=[]}}
    marker_size,
):
    assert fig is not None

    fig.data = fig.data[:1]

    if helices is None or len(helices) == 0:
        return

    import plotly.graph_objects as go

    color_palette = plotly.colors.qualitative.Plotly
    marker_symbols = [
        "circle",
        "square",
        "diamond",
        "cross",
        "x",
        "triangle-up",
        "triangle-down",
        "triangle-left",
        "triangle-right",
        "pentagon",
        "hexagon",
        "octagon",
        "star",
        "bowtie",
    ]

    new_traces = []
    # {(helix_id, class_id): {'x':[], 'y'=[]}}
    for hi, helix in enumerate(helices):
        helix_id, class_id = helix
        xy = helices[helix]
        x = xy["x"]
        y = xy["y"]
        new_traces.append(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=color_palette[hi % len(color_palette)],
                    opacity=0.6,
                    symbol=marker_symbols[hi % len(marker_symbols)],
                ),
                hovertemplate=f"Class {class_id}: {len(x)} segments<extra></extra>",
            )
        )

    fig.add_traces(new_traces)
