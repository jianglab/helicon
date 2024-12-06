import pathlib

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go

import mrcfile

import helicon


def get_project_root_dir(param_file):
    f = pathlib.Path(param_file)
    if param_file.endswith(".star"):
        return f.parent.parent.parent
    elif param_file.endswith(".cs"):
        return f.parent.parent
    else:
        return None


def get_micrograph(filename, target_apix, low_pass_angstrom, high_pass_angstrom):
    from skimage.transform import resize_local_mean

    with mrcfile.open(filename) as mrc:
        apix = round(float(mrc.voxel_size.x), 4)
        data = mrc.data.squeeze()
    ny, nx = data.shape

    new_ny = int(ny * apix / target_apix + 0.5) // 2 * 2
    new_nx = int(nx * apix / target_apix + 0.5) // 2 * 2
    data = resize_local_mean(image=data, output_shape=(new_ny, new_nx))

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
        if 'Class3D' in f.as_posix():
            class_files = f.parent.glob(f.stem[:10] + "class*.mrc")
            return sorted([f for f in class_files])
        else:
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

def select_helices_from_helixID(params, ids):
    mask = params["helixID"].astype(int).isin(ids)
    particles = params.loc[mask, :]
    helices = list(particles.groupby(["rlnMicrographName", "rlnHelicalTubeID"]))
    return helices

def compute_pair_distances(helices, lengths=None, target_total_count=-1):
    if lengths is not None:
        sorted_indices = (np.argsort(lengths))[::-1]
    else:
        sorted_indices = range(len(helices))
    min_len = 0
    dists_same_class = []
    for i in sorted_indices:
        _, segments_all_classes = helices[i]
        class_ids = np.unique(segments_all_classes["rlnClassNumber"])
        for ci in class_ids:
            mask = segments_all_classes["rlnClassNumber"] == ci
            segments = segments_all_classes.loc[mask, :]
            pos_along_helix = segments["rlnHelicalTrackLengthAngst"].values.astype(
                float
            )
            psi = segments["rlnAnglePsi"].values.astype(float)

            distances = np.abs(pos_along_helix[:, None] - pos_along_helix)
            distances = np.triu(distances)

            # Calculate pairwise distances only for segments with the same polarity
            mask = np.abs((psi[:, None] - psi + 180) % 360 - 180) < 90
            distances = distances[mask]
            dists_same_class.extend(
                distances[distances > 0]
            )  # Exclude zero distances (self-distances)
        if (
            lengths is not None
            and target_total_count > 0
            and len(dists_same_class) > target_total_count
        ):
            min_len = lengths[i]
            break
    if not dists_same_class:
        return [], 0
    else:
        return np.sort(dists_same_class), min_len



def estimate_inter_segment_distance(data):
    # data must have been sorted by micrograph, rlnHelicalTubeID, and rlnHelicalTrackLengthAngst
    helices = data.groupby(["rlnMicrographName", "rlnHelicalTubeID"], sort=False)

    import numpy as np

    dists_all = []
    for _, particles in helices:
        if len(particles) < 2:
            continue
        dists = np.sort(particles["rlnHelicalTrackLengthAngst"].astype(float).values)
        dists = dists[1:] - dists[:-1]
        dists_all.append(dists)
    dists_all = np.hstack(dists_all)
    dist_seg = np.median(dists_all)  # Angstrom
    return dist_seg

def get_class_abundance(params, nClass):
    abundance = np.zeros(nClass, dtype=int)
    for gn, g in params.groupby("rlnClassNumber"):
        abundance[int(gn) - 1] = len(g)
    return abundance

def get_class3d_projections_from_files(classFiles):
    projections = []
    nx = 0
    for f in classFiles:
        with mrcfile.open(f) as mrc:
            apix = float(mrc.voxel_size.x)
            data = mrc.data
            nx = mrc.header['nx']
        img = get_one_map_xyz_projects(data, nx)
        projections.append(img)

    return np.array(projections), apix, nx


@helicon.cache(expires_after=7, cache_dir=helicon.cache_dir / "whereIsMyClass", verbose=0)
def get_one_map_xyz_projects(data, nx): 
    min_data = np.min(data)
    max_data = np.max(data)
    if max_data-min_data != 0:
        data = (data-min_data)/(max_data-min_data)
    image = np.zeros((nx,nx*3+2))
    
    #image[:,0:nx] = data.sum(axis=0)
    image[:,0:nx] = data[int(nx/2),:,:]*nx
    image[:,nx+1:nx*2+1] = data.sum(axis=1)
    image[:,nx*2+2:nx*3+2] = data.sum(axis=2)
        
    return image

def get_class2d_from_file(classFile):
    with mrcfile.open(classFile) as mrc:
        apix = float(mrc.voxel_size.x)
        data = mrc.data
    print(np.shape(data))

    return data, round(apix, 4)


def get_class2d_params_from_file(params_file):
    if params_file.endswith(".star"):
        params = star_to_dataframe(params_file)
    elif params_file.endswith(".cs"):
        params = cs_to_dataframe(params_file)
    else:
        raise ValueError(
            f"ERROR: {params_file} is not a valid Class2D parameter file. Only star or cs files are supported"
        )
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


def plot_micrograph(micrograph, title, apix, plot_height=None, plot_width=None):

    fig = go.FigureWidget()

    h, w = micrograph.shape

    fig.add_trace(
        go.Heatmap(
            name="image",
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
            "x": 0.5,
            "y": 0.95,
            "xanchor": "center",
            "font": {"size": 14},
        },
        "xaxis": {"visible": False, "range": [0, w * apix]},
        "yaxis": {
            "visible": False,
            "range": [0, h * apix],
            "scaleanchor": "x",
            "autorange": "reversed",
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

    fig.data = [d for d in fig.data if not d.name.startswith("class_")]

    if helices is None or len(helices) == 0:
        return

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
                name=f"class_{class_id}",
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


def draw_distance_measurement(
    fig,  # plotly figure
    first_point,
    second_point,
):
    assert fig is not None
    if first_point is not None and second_point is not None:
        x = [first_point[0], second_point[0]]
        y = [first_point[1], second_point[1]]
        dist = np.hypot(x[1] - x[0], y[1] - y[0])
        line = go.Scatter(
            name="distance_line",
            x=x,
            y=y,
            mode="lines",
            line=dict(color="white", dash="dot"),
            hovertemplate=f"{dist:.1f} Å<extra></extra>",
        )
        other_traces = [d for d in fig.data if d.name != "distance_line"]
        with fig.batch_update():
            fig.data = other_traces
            fig.add_trace(line)
    else:
        other_traces = [d for d in fig.data if d.name != "distance_line"]
        if len(other_traces) < len(fig.data):
            fig.data = other_traces

def plot_histogram(
    data,
    title,
    xlabel,
    ylabel,
    max_pair_dist=None,
    bins=50,
    log_y=True,
    show_pitch_twist={},
    multi_crosshair=False,
    fig=None,
):
    import plotly.graph_objects as go

    if max_pair_dist is not None and max_pair_dist > 0:
        data = [d for d in data if d <= max_pair_dist]

    hist, edges = np.histogram(data, bins=bins)
    hist_linear = hist
    if log_y:
        hist = np.log10(1 + hist)

    center = (edges[:-1] + edges[1:]) / 2

    hover_text = []
    for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        hover_info = f"{xlabel.replace(" (Å)", "")}: {center[i]:.0f} ({left:.0f}-{right:.0f})Å<br>{ylabel}: {hist_linear[i]}"
        if show_pitch_twist:
            rise = show_pitch_twist["rise"]
            csyms = show_pitch_twist["csyms"]
            for csym in csyms:
                twist = 360 / (center[i] * csym / rise)
                hover_info += f"<br>Twist for C{csym}: {twist:.2f}°"
        hover_text.append(hover_info)

    if fig:
        fig.data[0].x = center
        fig.data[0].y = hist
        fig.data[0].text = hover_text
        fig.layout.title.text = title
    else:
        fig = go.FigureWidget()

        histogram = go.Bar(
            x=center,
            y=hist,
            name="Histogram",
            marker_color="blue",
            hoverinfo="none",
        )

        fig.add_trace(histogram)

        fig.data[0].text = hover_text
        fig.data[0].hoverinfo = "text"
        fig.update_layout(
            template="plotly_white",
            title_text=title,
            title_x=0.5,
            title_font=dict(size=12),
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            autosize=True,
            hovermode="closest",
            hoverlabel=dict(bgcolor="white", font_size=12),
        )

        if multi_crosshair:
            for i in range(20):
                fig.add_vline(
                    x=0,
                    line_width=3 if i == 0 else 2,
                    line_dash="solid" if i == 0 else "dash",
                    line_color="green",
                    visible=False,
                )

            def update_vline(trace, points, state):
                if points.point_inds:
                    hover_x = points.xs[0]
                    with fig.batch_update():
                        for i, vline in enumerate(fig.layout.shapes):
                            x = hover_x * (i + 1)
                            vline.x0 = x
                            vline.x1 = x
                            if x <= fig.data[0].x.max():
                                vline.visible = True
                            else:
                                vline.visible = False

            fig.data[0].on_hover(update_vline)

    return fig

