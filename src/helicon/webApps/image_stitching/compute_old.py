
import numpy as np
import pandas as pd
import os, pathlib

from joblib import Memory
username = os.getenv("USER")
memory = Memory(location=f"/tmp/{username}_joblib_cache", verbose=0)

from shiny import reactive
from shiny.express import ui, render, module, expressify

def encode_numpy(img, hflip=False, vflip=False):
    if img.dtype != np.dtype("uint8"):
        vmin, vmax = img.min(), img.max()
        if vmax > vmin:
            tmp = (255 * (img - vmin) / (vmax - vmin)).astype(np.uint8)
        else:
            tmp = np.zeros_like(img, dtype=np.uint8)
    else:
        tmp = img
    if hflip:
        tmp = tmp[:, ::-1]
    if vflip:
        tmp = tmp[::-1, :]
    from PIL import Image

    pil_img = Image.fromarray(tmp)
    return encode_PIL_Image(pil_img)


def encode_PIL_Image(img, hflip=False, vflip=False):
    import io, base64

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64, {encoded}"


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
            pos_along_helix = segments["rlnHelicalTrackLengthAngst"].values.astype(float)
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


def select_helices_by_length(helices, lengths, min_len, max_len):
    min_len = 0 if min_len is None else min_len
    max_len = -1 if min_len is None else max_len

    helices_retained = []
    n_ptcls = 0
    for gi, (gn, g) in enumerate(helices):
        cond = max_len <= 0 and min_len <= lengths[gi]
        cond = cond or (
            max_len > 0 and (max_len > min_len and (min_len <= lengths[gi] < max_len))
        )
        if cond:
            n_ptcls += len(g)
            helices_retained.append((gn, g))
    return helices_retained, n_ptcls


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


def get_number_helices_classes(params):
    nHelices = len(list(params.groupby(["rlnMicrographName", "rlnHelicalTubeID"])))
    nClasses = len(params["rlnClassNumber"].unique())
    return nHelices, nClasses


def get_pixel_size(
    data,
    attrs=[
        "micrograph_blob/psize_A",
        "rlnMicrographPixelSize",
        "rlnMicrographOriginalPixelSize",
        "blob/psize_A",
        "rlnImagePixelSize",
    ],
    return_source=False,
):
    try:
        sources = [data.attrs["optics"]]
    except:
        sources = []
    sources += [data]
    for source in sources:
        for attr in attrs:
            if attr in source:
                if attr in ["rlnImageName", "rlnMicrographName"]:
                    import mrcfile, pathlib

                    folder = pathlib.Path(data["starFile"].iloc[0])
                    if folder.is_symlink():
                        folder = folder.readlink()
                    folder = folder.resolve().parent
                    filename = source[attr].iloc[0].split("@")[-1]
                    filename = str((folder / "../.." / filename).resolve())
                    with mrcfile.open(filename, header_only=True) as mrc:
                        apix = float(mrc.voxel_size.x)
                else:
                    apix = float(source[attr].iloc[0])
                if return_source:
                    return apix, attr
                else:
                    return apix
    return None


def assign_segment_id(data, inter_segment_distance):
    assert "rlnHelicalTrackLengthAngst" in data
    tmp = (
        data.loc[:, "rlnHelicalTrackLengthAngst"].astype(float) / inter_segment_distance
    )
    err = (tmp - tmp.round()).abs()
    if np.sum(np.where(err > 0.1)) > 0:
        print(
            f"WARNING: it appears that the helical segments were extracted with different inter-segment distances"
        )
    helical_segment_id = tmp.round().astype(int)
    return helical_segment_id


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


def get_class2d_from_uploaded_file(fileobj):
    import os, tempfile

    orignal_filename = fileobj.name
    suffix = os.path.splitext(orignal_filename)[-1]
    with tempfile.NamedTemporaryFile(suffix=suffix) as temp:
        temp.write(fileobj.read())
        return get_class2d_from_file(temp.name)


@memory.cache
def get_class2d_from_url(url):
    url_final = get_direct_url(url)  # convert cloud drive indirect url to direct url
    fileobj = download_file_from_url(url_final)
    if fileobj is None:
        raise ValueError(
            f"ERROR: {url} could not be downloaded. If this url points to a cloud drive file, make sure the link is a direct download link instead of a link for preview"
        )
    data = get_class2d_from_file(fileobj.name)
    return data


def get_class2d_from_file(classFile):
    import mrcfile

    with mrcfile.open(classFile) as mrc:
        apix = float(mrc.voxel_size.x)
        data = mrc.data
    return data, round(apix, 4)


@memory.cache
def get_class2d_params_from_url(url, url_cs_pass_through=None):
    url_final = get_direct_url(url)  # convert cloud drive indirect url to direct url
    fileobj = download_file_from_url(url_final)
    if fileobj is None:
        raise ValueError(
            f"ERROR: {url} could not be downloaded. If this url points to a cloud drive file, make sure the link is a direct download link instead of a link for preview"
        )
    if url_cs_pass_through is None:
        data = get_class2d_params_from_file(fileobj.name)
        return data
    url_final_cs_pass_through = get_direct_url(
        url_cs_pass_through
    )  # convert cloud drive indirect url to direct url
    fileobj_cs_pass_through = download_file_from_url(url_final_cs_pass_through)
    if fileobj_cs_pass_through is None:
        raise ValueError(
            f"ERROR: {url_cs_pass_through} could not be downloaded. If this url points to a cloud drive file, make sure the link is a direct download link instead of a link for preview"
        )
    data = get_class2d_params_from_file(fileobj.name, fileobj_cs_pass_through.name)
    return data


def get_class2d_params_from_file(params_file, cryosparc_pass_through_file=None):
    if params_file.endswith(".star"):
        params = star_to_dataframe(params_file)
    elif params_file.endswith(".cs"):
        assert cryosparc_pass_through_file is not None
        params = cs_to_dataframe(params_file, cryosparc_pass_through_file)
    required_attrs = np.unique(
        "rlnImageName rlnHelicalTubeID rlnHelicalTrackLengthAngst rlnClassNumber rlnAnglePsi".split()
    )
    missing_attrs = [attr for attr in required_attrs if attr not in params]
    if missing_attrs:
        raise ValueError(f"ERROR: parameters {missing_attrs} are not available")
    return params


def star_to_dataframe(starFile):
    import pandas as pd
    from gemmi import cif

    star = cif.read_file(starFile)
    if len(star) == 2:
        optics = pd.DataFrame()
        for item in star[0]:
            for tag in item.loop.tags:
                value = star[0].find_loop(tag)
                optics[tag.strip("_")] = np.array(value)
    else:
        optics = None

    data = pd.DataFrame()
    for item in star[-1]:
        for tag in item.loop.tags:
            value = star[-1].find_loop(tag)
            tag = tag.strip("_")
            data[tag] = np.array(value)

    if optics is not None:
        data.attrs["optics"] = optics

    data.attrs["starFile"] = starFile

    return data


def cs_to_dataframe(cs_file, cs_pass_through_file):
    cs = np.load(cs_file)
    df_cs = pd.DataFrame.from_records(cs.tolist(), columns=cs.dtype.names)
    cs_passthrough = np.load(cs_pass_through_file)
    df_cs_passthrough = pd.DataFrame.from_records(
        cs_passthrough.tolist(), columns=cs_passthrough.dtype.names
    )
    data = pd.concat([df_cs, df_cs_passthrough], axis=1)
    data = data.loc[:, ~data.columns.duplicated()]
    # rlnImageName rlnHelicalTubeID rlnHelicalTrackLengthAngst rlnCoordinateX rlnCoordinateY rlnClassNumber rlnAnglePsi
    ret = pd.DataFrame()
    if "blob/idx" in data and "blob/path" in data:
        ret["rlnImageName"] = (
            (data["blob/idx"].astype(int) + 1).map("{:06d}".format)
            + "@"
            + data["blob/path"].str.decode("utf-8")
        )
    if "blob/psize_A" in data:
        ret["rlnImagePixelSize"] = data["blob/psize_A"]
        ret["blob/psize_A"] = data["blob/psize_A"]
    if "micrograph_blob/path" in data:
        ret["rlnMicrographName"] = data["micrograph_blob/path"]
    if "micrograph_blob/psize_A" in data:
        ret["rlnMicrographPixelSize"] = data["micrograph_blob/psize_A"]
        ret["micrograph_blob/psize_A"] = data["micrograph_blob/psize_A"]
    if "location/micrograph_path" in data:
        ret["rlnMicrographName"] = data["location/micrograph_path"]
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
    if "filament/filament_uid" in data:
        if "blob/path" in data:
            if data["filament/filament_uid"].min() > 1000:
                micrographs = data.groupby(["blob/path"])
                for _, m in micrographs:
                    mapping = {
                        v: i + 1
                        for i, v in enumerate(
                            sorted(m["filament/filament_uid"].unique())
                        )
                    }
                    ret.loc[m.index, "rlnHelicalTubeID"] = m[
                        "filament/filament_uid"
                    ].map(mapping)
            else:
                ret.loc[:, "rlnHelicalTubeID"] = data["filament/filament_uid"].astype(
                    int
                )

            if "filament/position_A" in data:
                filaments = data.groupby(["blob/path", "filament/filament_uid"])
                for _, f in filaments:
                    val = f["filament/position_A"].astype(np.float32).values
                    val -= np.min(val)
                    ret.loc[f.index, "rlnHelicalTrackLengthAngst"] = val.round(2)
        else:
            mapping = {
                v: i + 1
                for i, v in enumerate(sorted(data["filament/filament_uid"].unique()))
            }
            ret.loc[:, "rlnHelicalTubeID"] = data["filament/filament_uid"].map(mapping)
    if "filament/filament_pose" in data:
        ret.loc[:, "rlnAnglePsi"] = np.round(
            -np.rad2deg(data["filament/filament_pose"]), 1
        )
    # 2D class assignments
    if "alignments2D/class" in data:
        ret["rlnClassNumber"] = data["alignments2D/class"].astype(int) + 1
    if "alignments2D/shift" in data:
        shifts = pd.DataFrame(data["alignments2D/shift"].tolist()).round(2)
        ret["rlnOriginX"] = -shifts.iloc[:, 0]
        ret["rlnOriginY"] = -shifts.iloc[:, 1]
    if "alignments2D/pose" in data:
        ret["rlnAnglePsi"] = -np.rad2deg(data["alignments2D/pose"]).round(2)
    return ret


def dataframe2star(data, starFile):
    if hasattr(starFile, "read") and callable(starFile.read):
        fp = starFile
    else:
        fp = open(starFile, "wt")

    try:
        optics = data.attrs["optics"]
        if len(optics) > 0:
            fp.write("\n# version 30001\n")
            fp.write("\ndata_optics\n\nloop_ \n")
            keys = [k for k in optics.columns if k.startswith("rln")]
            for ki, k in enumerate(keys):
                fp.write("_%s #%d \n" % (k, ki + 1))
            lines = optics[keys[0]].astype(str)
            for k in keys[1:]:
                if optics[k].dtype == np.float64:
                    lines += "\t" + optics[k].round(6).astype(str)
                else:
                    lines += "\t" + optics[k].astype(str)
            fp.write("\n".join(lines))
            fp.write("\n\n")
    except:
        pass

    fp.write(f"\ndata_particles\n\nloop_ \n")
    keys = [k for k in data.columns if k.startswith("rln")]
    for ki, k in enumerate(keys):
        fp.write("_%s #%d \n" % (k, ki + 1))
    lines = data[keys[0]].astype(str)
    for k in keys[1:]:
        if data[k].dtype == np.float64:
            lines += "\t" + data[k].round(6).astype(str)
        else:
            lines += "\t" + data[k].astype(str)
    fp.write("\n".join(lines))
    fp.write("\n")


def download_file_from_url(url):
    import tempfile
    import requests
    import os

    if pathlib.Path(url).is_file():
        return open(url, "rb")
    try:
        filesize = get_file_size(url)
        local_filename = url.split("/")[-1]
        suffix = "." + local_filename
        fileobj = tempfile.NamedTemporaryFile(suffix=suffix)
        with requests.get(url) as r:
            r.raise_for_status()  # Check for request success
            fileobj.write(r.content)
        return fileobj
    except requests.exceptions.RequestException as e:
        print(e)
        return None


def get_direct_url(url):
    import re

    if url.startswith("https://drive.google.com/file/d/"):
        hash = url.split("/")[5]
        return f"https://drive.google.com/uc?export=download&id={hash}"
    elif url.startswith("https://app.box.com/s/"):
        hash = url.split("/")[-1]
        return f"https://app.box.com/shared/static/{hash}"
    elif url.startswith("https://www.dropbox.com"):
        if url.find("dl=1") != -1:
            return url
        elif url.find("dl=0") != -1:
            return url.replace("dl=0", "dl=1")
        else:
            return url + "?dl=1"
    elif url.find("sharepoint.com") != -1 and url.find("guestaccess.aspx") != -1:
        return url.replace("guestaccess.aspx", "download.aspx")
    elif url.startswith("https://1drv.ms"):
        import base64

        data_bytes64 = base64.b64encode(bytes(url, "utf-8"))
        data_bytes64_String = (
            data_bytes64.decode("utf-8").replace("/", "_").replace("+", "-").rstrip("=")
        )
        return (
            f"https://api.onedrive.com/v1.0/shares/u!{data_bytes64_String}/root/content"
        )
    else:
        return url


def get_file_size(url):
    import requests

    response = requests.head(url)
    if "Content-Length" in response.headers:
        file_size = int(response.headers["Content-Length"])
        return file_size
    else:
        return None


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



@module
def image_select(
    input,
    output,
    session,
    label="Select Image(s):",
    images=reactive.value([]),
    display_image_labels=True,
    image_labels=reactive.value([]),
    image_size=reactive.value(128),
    initial_selected_indices=reactive.value([]),
    disable_selection=False,
    image_border=2,
    gap=0,
):
    import numpy as np
    from PIL import Image

    selection_return = reactive.value(None)
    selection = reactive.value([])
    initial_selection = reactive.value([])
    bids_react = reactive.value([])

    @render.ui
    def display_images():
        if images() is None or len(images()) == 0:
            return None

        images_final = []
        for i, image in enumerate(images()):
            if isinstance(image, str):
                tmp = image
            elif isinstance(image, Image.Image):
                tmp = encode_PIL_Image(image)
            elif isinstance(image, np.ndarray) and image.ndim == 2:
                tmp = encode_numpy(image)
            else:
                raise ValueError(
                    "image must be an image file, a PIL Image, or a 2D numpy array"
                )
            images_final.append(tmp)

        assert len(image_labels()) == 0 or len(image_labels()) == len(images_final)

        if len(image_labels()):
            image_labels_final = image_labels()
        else:
            image_labels_final = list(range(1, len(images_final) + 1))

        assert image_size() >= 32

        bids = [f"image_select_{i+1}" for i in range(len(images_final))]

        def create_image_button(i, image, label, bid):
            img = ui.img(
                src=image,
                alt=f"Image {i+1}",
                title=str(label),
                style=f"object-fit: contain; max-width: {image_size()-image_border*2}px; max-height: {image_size()-image_border*2}px; border: {image_border}px solid transparent;",
            )

            return ui.input_action_button(
                id=bid,
                label=(
                    ui.div(
                        img,
                        ui.p(
                            label,
                            style="text-align: left; color: white; text-shadow: -1px -1px 0.5px rgba(0,0,0,0.5), 1px -1px 0.5px rgba(0,0,0,0.5), -1px 1px 0.5px rgba(0,0,0,0.5), 1px 1px 0.5px rgba(0,0,0,0.5); position: absolute; top: 2px; left: 5px;",
                        ),
                        style="position: relative;",
                    )
                    if display_image_labels
                    else img
                ),
                disabled=disable_selection,
                style=f"padding: 0px; border: 0px; margin: 0px; background-color: transparent;",
                onmouseover=f"if (this.querySelector('img').style.border !== '{image_border}px solid red') {{this.querySelector('img').style.border='{image_border}px solid blue'; this.querySelector('p').style.color='blue';}}",
                onmouseout=f"if (this.querySelector('img').style.border !== '{image_border}px solid red') {{this.querySelector('img').style.border='{image_border}px solid transparent';  this.querySelector('p').style.color='white';}}",
                onclick=f"""var bid = '{session.ns}-{bid}';
                            var count0 = parseInt(this.getAttribute('click_count')) || 0;
                            count = count0 + 1
                            var info = {{
                                count: count,
                                selected: (count)%2,
                                ctrlKey: event.ctrlKey,
                                shiftKey: event.shiftKey,
                                altKey: event.altKey,
                                metaKey: event.metaKey
                            }};

                            var img  = this.querySelector("img");
                            var text = this.querySelector("p");
                            img.style.border = count%2 ? "{image_border}px solid red" : "{image_border}px solid transparent";
                            if (text) {{
                                text.style.color = count%2 ? "red" : "white";
                            }}

                            this.setAttribute('click_count', count);
                            //Shiny.setInputValue(bid, count, {{priority: 'deferred'}});
                            Shiny.setInputValue(bid + '_click', info, {{priority: 'deferred'}});
                            //console.log("click", bid, parseInt(this.getAttribute('click_count')), info);
                        """,
            )

        ui_images = ui.div(
            *[
                create_image_button(i, image, image_labels_final[i], bid)
                for i, (image, bid) in enumerate(zip(images_final, bids))
            ],
            style=f"display: flex; flex-flow: row wrap; justify-content: center; justify-items: center; align-items: center; gap: {gap}px {gap}px; margin: 0 0 {image_border}px 0",
        )

        if len(label):
            ui_images = ui.div(
                ui.h6(
                    label,
                    style="text-align: center; margin: 0;",
                    title="Hold the Shift key while clicking to select multiple images",
                ),
                ui_images,
                style=f"display: flex; flex-direction: column; gap: {gap}px; margin: 0",
            )

        bids_react.set(bids)
        initial_selection.set([0] * len(images()))

        if not disable_selection and len(initial_selected_indices()) > 0:
            tmp = initial_selection()
            for i in initial_selected_indices():
                tmp[i] = 1
            initial_selection.set(tmp)

            click_scripts = []
            for i in initial_selected_indices():
                click_scripts.append(
                    ui.tags.script(
                        f"""
                            var bid = '{session.ns}-{bids[i]}';
                            var element = document.getElementById(bid);
                            var event = new MouseEvent('click', {{
                                bubbles: true,
                                cancelable: true,
                                view: window,
                                shiftKey: true
                            }});
                            element.dispatchEvent(event);
                        """
                    )
                )
            return (ui_images, click_scripts)
        else:
            return ui_images

    @render.ui
    def ordered_selection():
        status = [
            (input[bid]() + initial_selection()[i]) % 2 == 1
            for i, bid in enumerate(bids_react())
        ]
        current = [i for i, is_selected in enumerate(status) if is_selected]
        previous = getattr(ordered_selection, "previous", [])
        result = [i for i in previous if i in current]
        result += [i for i in current if i not in result]
        scripts = None
        if len(result) > 1:
            shiftKey = input[f"{bids_react()[result[-1]]}_click"]()["shiftKey"]
            if not shiftKey:
                scripts = []
                for i in result[:-1]:
                    bid = f"{session.ns}-{bids_react()[i]}"
                    script = ui.tags.script(
                        f"""document.getElementById('{bid}').click();"""
                    )
                    scripts.append(script)
                result = [result[-1]]

        ordered_selection.previous = result
        selection.set(result)

        return scripts

    @reactive.effect
    @reactive.event(selection)
    def _():
        if selection() != selection_return():
            selection_return.set(selection())

    if disable_selection:
        return None
    else:
        return selection_return


@expressify
def google_analytics(id):
    ui.head_content(
        ui.HTML(
            f"""
            <script async src="https://www.googletagmanager.com/gtag/js?id={id}"></script>
            <script>
            window.dataLayer = window.dataLayer || [];
            function gtag(){{dataLayer.push(arguments);}}
            gtag('js', new Date());
            gtag('config', '{id}');
            </script>
            """
        )
    )


def get_client_url(input):
    d = input._map
    url = f"{d['.clientdata_url_protocol']()}//{d['.clientdata_url_hostname']()}:{d['.clientdata_url_port']()}{d['.clientdata_url_pathname']()}{d['.clientdata_url_search']()}"
    return url


def get_client_url_query_params(input, keep_list=True):
    d = input._map
    qs = d[".clientdata_url_search"]().strip("?")
    import urllib.parse

    parsed_qs = urllib.parse.parse_qs(qs)
    if not keep_list:
        for k, v in parsed_qs.items():
            if isinstance(v, list) and len(v) == 1:
                parsed_qs[k] = v[0]
    return parsed_qs


def set_client_url_query_params(query_params):
    import urllib.parse

    encoded_query_params = urllib.parse.urlencode(query_params, doseq=True)
    script = ui.tags.script(
        f"""
                var url = new URL(window.location.href);
                url.search = '{encoded_query_params}';
                window.history.pushState(null, '', url.toString());
            """
    )
    return script
