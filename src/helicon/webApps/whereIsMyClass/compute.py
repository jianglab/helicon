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
