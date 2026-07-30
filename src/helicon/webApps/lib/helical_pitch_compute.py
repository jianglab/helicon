from __future__ import annotations

"""Compute functions for the HelicalPitch tab.

Ported from /Users/wjiang/software/helical-index/HelicalPitch.git/compute.py
and adapted to use helicon library utilities where available.
"""


import numpy as np
import pandas as pd
import pathlib

import helicon


# ── Pair distance computation ────────────────────────────────────


def compute_pair_distances(helices, lengths=None, target_total_count=-1):
    """Compute same-polarity pair distances along each helix."""
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

            mask = np.abs((psi[:, None] - psi + 180) % 360 - 180) < 90
            distances = distances[mask]
            dists_same_class.extend(distances[distances > 0])

        if (
            lengths is not None
            and target_total_count > 0
            and len(dists_same_class) > target_total_count
        ):
            min_len = lengths[i]
            break
    if not dists_same_class:
        return [], 0
    return np.sort(dists_same_class), min_len


def find_first_peak(distances, cutoff=500):
    """Find the first significant peak in the distance distribution."""
    from scipy.stats import gaussian_kde
    from scipy.signal import find_peaks

    distances_work = np.sort(np.hstack((-distances, distances)))
    kde = gaussian_kde(distances_work)
    kde_x = np.arange(
        np.min(distances_work).round(), np.max(distances_work).round() + 0.5
    )
    kde_y = kde.pdf(kde_x)
    peaks, _ = find_peaks(kde_y, prominence=(0.01 / len(kde_x), None))
    peaks = np.sort(peaks)
    index_0 = np.where(kde_x >= 0)[0][0]
    peaks = peaks[peaks >= index_0] - index_0
    first_peak = 0
    if peaks is not None and len(peaks) > 0:
        if len(peaks) == 1:
            if peaks[0] >= cutoff:
                first_peak = peaks[0]
        else:
            if peaks[0] >= cutoff:
                first_peak = peaks[0]
            elif peaks[1] >= cutoff:
                first_peak = peaks[1]
    return first_peak, peaks


# ── Helix selection ───────────────────────────────────────────────


def select_helices_by_length(helices, lengths, min_len, max_len):
    """Filter helices by filament length range."""
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
    """Return the length of each filament."""
    filement_lengths = []
    for gn, g in helices:
        track_lengths = g["rlnHelicalTrackLengthAngst"].astype(float).values
        length = track_lengths.max() - track_lengths.min() + particle_box_length
        filement_lengths.append(length)
    return filement_lengths


def select_classes(params, class_indices):
    """Return helices (grouped by micrograph + tube ID) for the given classes."""
    class_indices_tmp = np.array(class_indices) + 1
    mask = params["rlnClassNumber"].astype(int).isin(class_indices_tmp)
    particles = params.loc[mask, :]
    helices = list(particles.groupby(["rlnMicrographName", "rlnHelicalTubeID"]))
    return helices


def get_class_abundance(params, nClass):
    """Count particles per class."""
    abundance = np.zeros(nClass, dtype=int)
    for gn, g in params.groupby("rlnClassNumber"):
        abundance[int(gn) - 1] = len(g)
    return abundance


# ── File I/O (delegated to helicon where possible) ───────────────


@helicon.cache(
    cache_dir=str(helicon.cache_dir / "helical_lab"), expires_after=7, verbose=0
)
def get_class2d_from_url(url):
    url_final = helicon.get_direct_url(url)
    fileobj = helicon.download_file_from_url(url_final)
    if fileobj is None:
        raise ValueError(
            f"ERROR: {url} could not be downloaded. If this url points to a "
            "cloud drive file, make sure the link is a direct download link."
        )
    return get_class2d_from_file(fileobj.name)


def get_class2d_from_file(classFile):
    import mrcfile

    with mrcfile.open(classFile) as mrc:
        apix = float(mrc.voxel_size.x)
        data = mrc.data
    return data, round(apix, 4)


@helicon.cache(
    cache_dir=str(helicon.cache_dir / "helical_lab"), expires_after=7, verbose=0
)
def get_class2d_params_from_url(url):
    url_final = helicon.get_direct_url(url)
    fileobj = helicon.download_file_from_url(url_final)
    if fileobj is None:
        raise ValueError(
            f"ERROR: {url} could not be downloaded. If this url points to a "
            "cloud drive file, make sure the link is a direct download link."
        )
    return get_class2d_params_from_file(fileobj.name)


def get_class2d_helix_params_from_url(url):
    df = get_class2d_params_from_url(url)
    return _annotate_helix_ids(df)


def get_class2d_helix_params_from_file(params_file):
    df = get_class2d_params_from_file(params_file)
    return _annotate_helix_ids(df)


def _annotate_helix_ids(df):
    helices = df.groupby(["rlnMicrographName", "rlnHelicalTubeID"])
    for hi, (_, helix) in enumerate(helices):
        l = helix["rlnHelicalTrackLengthAngst"].astype(float).max().round()
        df.loc[helix.index, "length"] = l
        df.loc[helix.index, "helixID"] = hi + 1
    return df


def get_class2d_params_from_file(params_file):
    if params_file.endswith(".star"):
        params = _star_to_dataframe(params_file)
    elif params_file.endswith(".cs"):
        params = _cs_to_dataframe(params_file)
    else:
        raise ValueError(
            f"ERROR: {params_file} is not a valid Class2D parameter file. "
            "Only star or cs files are supported"
        )
    required_attrs = np.unique(
        "rlnImageName rlnHelicalTubeID rlnHelicalTrackLengthAngst "
        "rlnClassNumber rlnAnglePsi".split()
    )
    missing_attrs = [attr for attr in required_attrs if attr not in params]
    if missing_attrs:
        raise ValueError(f"ERROR: parameters {missing_attrs} are not available")
    return params


def _star_to_dataframe(starFile):
    import starfile

    d = starfile.read(starFile, always_dict=True)
    assert "optics" in d and "particles" in d, (
        f"ERROR: {starFile} has {' '.join(d.keys())} "
        "but optics and particles are expected"
    )
    data = d["particles"]
    data.attrs["optics"] = d["optics"]
    data.attrs["starFile"] = starFile
    return data


def _cs_to_dataframe(cs_file):
    cs = np.load(cs_file)
    data = pd.DataFrame.from_records(cs.tolist(), columns=cs.dtype.names)
    required_attrs = (
        "blob/idx blob/path filament/filament_uid filament/position_A "
        "alignments2D/class alignments2D/pose "
        "location/micrograph_path".split()
    )
    missing_attrs = [attr for attr in required_attrs if attr not in data]
    if missing_attrs:
        msg = (
            f"ERROR: required attrs '{', '.join(missing_attrs)}' "
            f"are not included in {cs_file}"
        )
        msg += (
            "\nIf the particles were imported from a RELION star file, "
            "use: helicon images2star <cs file> <output star> "
            "--copyParm <original star>"
        )
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
        ret["rlnMicrographName"] = data["location/micrograph_path"].str.decode("utf-8")
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
        data["filament/position_A"].astype(np.float32).values.round(2)
    )
    ret["rlnClassNumber"] = data["alignments2D/class"].astype(int) + 1
    ret["rlnAnglePsi"] = -np.rad2deg(data["alignments2D/pose"]).round(2)
    return ret


# ── Histogram plotting ────────────────────────────────────────────


def plot_histogram(
    data,
    title,
    xlabel,
    ylabel,
    max_pair_dist=None,
    bins=50,
    log_y=True,
    show_pitch_twist=None,
    multi_crosshair=False,
    fig=None,
):
    """Plot a pair-distance histogram with optional twist/pitch crosshairs."""
    import plotly.graph_objects as go

    if show_pitch_twist is None:
        show_pitch_twist = {}

    if max_pair_dist is not None and max_pair_dist > 0:
        data = [d for d in data if d <= max_pair_dist]

    hist, edges = np.histogram(data, bins=bins)
    hist_linear = hist
    if log_y:
        hist = np.log10(1 + hist)

    center = (edges[:-1] + edges[1:]) / 2

    hover_text = []
    for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        hover_info = (
            f"{xlabel.replace(' (Å)', '')}: {center[i]:.0f} "
            f"({left:.0f}-{right:.0f})Å<br>{ylabel}: {hist_linear[i]}"
        )
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
        fig = go.Figure()
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
            margin=dict(t=40, b=50, l=50, r=20),
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

            # Spikes disabled: multi-crosshair handled via JS injection in _fig_to_html

    return fig
