from __future__ import annotations

import logging, os
from pathlib import Path
from typing import Any, Callable
import numpy as np
import pandas as pd
from .exceptions import HeliconIOError, HeliconValueError, HeliconConfigError

pd.options.mode.copy_on_write = True

logger = logging.getLogger(__name__)

from .util import available_cpu

from .io_mrc import get_image_number, get_image_size

__all__ = [
    "Relion_OpticsGroup_Parameters",
    "assign_beamshifts_to_cluster",
    "clean_cs_micrograph_path",
    "cistem2dataframe",
    "connect_cryosparc",
    "cs2dataframe",
    "dataframe2cs",
    "dataframe2file",
    "dataframe2star",
    "dataframe_convert",
    "dataframe_cryosparc_to_relion",
    "dataframe_guess_data_type",
    "dataframe_normalize_filename",
    "eman_astigmatism_to_relion",
    "get_dataframe_convention",
    "get_relion_project_folder",
    "guess_data_type",
    "image2dataframe",
    "images2dataframe",
    "mrc2mrcs",
    "relion_astigmatism_to_eman",
    "star2dataframe",
    "star_build_opticsgroup",
    "star_dissolve_opticsgroup",
    "getPixelSize",
    "setPixelSize",
    "pixelSizeAttrForImageAttr",
]


def preferred_relion_star_column_order() -> list[str]:
    """Return the preferred column order for RELION STAR files.

    This order is based on common conventions and may not be strictly defined.
    Columns not in this list will be appended at the end in their original order.

    Returns
    -------
    list of str
        Preferred column names in order.
    """

    cols = "rlnAngleRot rlnAngleTilt rlnAnglePsi rlnOriginXAngst rlnOriginYAngst rlnCoordinateX rlnCoordinateY rlnDefocusU rlnDefocusV rlnDefocusAngle rlnPhaseShift rlnCtfBfactor rlnCtfScalefactor rlnLogLikeliContribution rlnRandomSubset rlnClassNumber rlnImageName rlnMicrographName rlnMovieName rlnOpticsGroup".split()
    return cols


def reorder_dataframe_columns(
    data: pd.DataFrame, column_order: list[str] = None
) -> pd.DataFrame:
    """Reorder the columns of a DataFrame according to a specified order.

    Columns not included in the specified order will be appended at the end
    in their original order.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame to reorder.
    column_order : list of str
        Desired order of columns. Columns not in this list will be appended.

    Returns
    -------
    pd.DataFrame
        DataFrame with reordered columns.
    """
    if column_order is None:
        column_order = preferred_relion_star_column_order()
    existing_columns = [col for col in column_order if col in data.columns]
    remaining_columns = [col for col in data.columns if col not in existing_columns]
    new_column_order = existing_columns + remaining_columns
    return data[new_column_order]


def pixelSizeAttrForImageAttr(imageAttr: str) -> str | None:
    """Return the corresponding pixel size attribute for an image attribute.

    Parameters
    ----------
    imageAttr : str
        Image attribute name (e.g. ``rlnImageName``).

    Returns
    -------
    str or None
        The matching pixel size attribute, or None if not found.
    """
    mapping = {
        "rlnImageName": "rlnImagePixelSize",
        "rlnMicrographName": "rlnMicrographPixelSize",
        "rlnMicrographMovieName": "rlnMicrographOriginalPixelSize",
    }
    if imageAttr in mapping:
        return mapping[imageAttr]
    return None


def getPixelSize(
    data: pd.DataFrame,
    attrs: list[str] = [
        "rlnImagePixelSize",
        "rlnMicrographPixelSize",
        "rlnMicrographOriginalPixelSize",
        "rlnImageName",
        "rlnMicrographName",
    ],
    return_pixelSize_source: bool = False,
):
    """Get the pixel size from a DataFrame or its optics group.

    Searches the specified attributes in order, falling back to reading
    the MRC header if the pixel size is not stored directly.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame with optional ``.attrs["optics"]``.
    attrs : list of str, optional
        Attributes to search for pixel size.
    return_pixelSize_source : bool, optional
        If True, also return the attribute name used.

    Returns
    -------
    float or (float, str) or None
        Pixel size value, optionally with the source attribute name.
    """
    try:
        sources = [data.attrs["optics"]]
    except KeyError:
        sources = []
    sources += [data]
    apix = None
    for source in sources:
        if source is None:
            continue
        for attr in attrs:
            if attr in source:
                if attr in ["rlnImageName", "rlnMicrographName"]:
                    import mrcfile

                    if isinstance(data.attrs["source_path"], list):
                        folder = Path(data.attrs["source_path"][0])
                    else:
                        folder = Path(data.attrs["source_path"])
                    if folder.is_symlink():
                        folder = folder.readlink()
                    folder = folder.resolve().parent
                    filename = source[attr].iloc[0].split("@")[-1]
                    filename = str((folder / "../.." / filename).resolve())
                    try:
                        with mrcfile.open(filename, header_only=True) as mrc:
                            apix = float(mrc.voxel_size.x)
                    except (OSError, ValueError, TypeError):
                        pass
                else:
                    apix = float(source[attr].iloc[0])
                if apix is not None:
                    if return_pixelSize_source:
                        return apix, attr
                    return apix
    if return_pixelSize_source:
        return None, None
    return None


def setPixelSize(
    data: pd.DataFrame, apix_new: float, update_defocus: bool = False
) -> None:
    """Set the pixel size on a DataFrame and optionally rescale defocus values.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame with optional ``.attrs["optics"]``.
    apix_new : float
        New pixel size in Angstroms.
    update_defocus : bool, optional
        If True, rescale defocus values by ``(apix_new / apix_old)^2``.
    """
    apix_old, pixelSize_source = getPixelSize(data, return_pixelSize_source=True)
    if update_defocus:
        for attr in "rlnDefocusU rlnDefocusV".split():
            if attr in data:
                data.loc[:, attr] = data.loc[:, attr].astype(float) * (
                    (apix_new / apix_old) ** 2
                )
    try:
        data.attrs["optics"].loc[:, pixelSize_source] = apix_new
    except (KeyError, AttributeError):
        pass
    if pixelSize_source in data:
        data.loc[:, pixelSize_source] = apix_new


def get_relion_project_folder(starFile: str) -> str | None:
    """Extract the RELION project folder from a STAR file path.

    Parameters
    ----------
    starFile : str
        Path to a STAR file within a RELION project.

    Returns
    -------
    str or None
        Absolute path to the project folder, or None if it cannot be determined.
    """
    filename_abs = str(Path(starFile).resolve())
    if filename_abs.find("/job") == -1:
        return None
    parts = filename_abs.split("/")
    for pi, p in enumerate(parts):
        if p.startswith("job"):
            break
    job_folder = "/".join(parts[: pi + 1])
    if not Path(job_folder, "default_pipeline.star").exists():
        return None
    pi = max(0, pi - 1)
    proj_folder = "/".join(parts[:pi])
    if not Path(proj_folder, "default_pipeline.star").exists():
        return None
    return proj_folder


def __process_cluster(
    X: np.ndarray, n_clusters: int, min_cluster_size: int
) -> tuple[int, float, np.ndarray]:
    """Cluster data and evaluate with the silhouette score.

    Parameters
    ----------
    X : np.ndarray
        2D array of features to cluster.
    n_clusters : int
        Requested number of clusters.
    min_cluster_size : int
        Minimum number of samples per cluster.

    Returns
    -------
    tuple of (int, float, np.ndarray)
        (n_clusters, silhouette_avg, cluster_labels).
    """
    from sklearn.metrics import silhouette_score
    from .analysis import AgglomerativeClusteringWithMinSize

    clustering_method = AgglomerativeClusteringWithMinSize(
        n_clusters=n_clusters, min_cluster_size=min_cluster_size
    )
    cluster_labels = clustering_method.fit_predict(
        X
    )  # note: the number of clusters can be smaller than the requested number of clusters due to small clusters being merged
    if len(np.unique(cluster_labels)) > 1:
        silhouette_avg = silhouette_score(X, cluster_labels)
    else:
        silhouette_avg = -1
    return n_clusters, silhouette_avg, cluster_labels


def assign_beamshifts_to_cluster(
    beamshifts: list | np.ndarray,
    min_cluster_size: int = 4,
    range_n_clusters: range = range(2, 200),
    cpu: int = -1,
    verbose: int = 2,
) -> np.ndarray:
    """Find the optimal clustering of beam shift positions using silhouette scores.

    Parameters
    ----------
    beamshifts : list or np.ndarray
        List of beam shift coordinate tuples or array-like.
    min_cluster_size : int, optional
        Minimum number of samples per cluster. Defaults to 4.
    range_n_clusters : range, optional
        Range of cluster numbers to evaluate. Defaults to range(2, 200).
    cpu : int, optional
        Number of CPU workers (-1 for all available). Defaults to -1.
    verbose : int, optional
        Verbosity level (0=quiet, 1=result, 2+=per-iteration). Defaults to 2.

    Returns
    -------
    np.ndarray
        1-indexed array of cluster labels for each beam shift.
    """
    X = np.array(beamshifts)

    # Evaluate silhouette scores for different numbers of clusters
    best_n_clusters = range_n_clusters[0]
    best_score = -1
    best_cluster_labels = None

    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(
        max_workers=cpu if cpu > 0 else available_cpu()
    ) as executor:
        futures = [
            executor.submit(__process_cluster, X, n, min_cluster_size)
            for n in range_n_clusters
        ]

        for future in as_completed(futures):
            n_clusters, silhouette_avg, cluster_labels = future.result()

            if verbose > 2:
                logger.info(
                    f"\t{n_clusters} -> {len(np.unique(cluster_labels))} clusters: silhouette score={silhouette_avg}{' *' if silhouette_avg >= best_score else ''}"
                )

            if silhouette_avg >= best_score:
                best_score = silhouette_avg
                best_cluster_labels = cluster_labels

    if verbose:
        logger.info(
            f"The optimal number of clusters is {len(np.unique(best_cluster_labels))} with a silhouette score of {best_score:.3f}"
        )

    cluster_labels = np.array(best_cluster_labels) + 1
    return cluster_labels


try:
    from numba import jit, set_num_threads, prange
except ImportError:
    logger.warning(
        "failed to load numba. The program will run correctly but will be much slower. Run 'pip install numba' to install numba and speed up the program"
    )

    def jit(*args: Any, **kwargs: Any) -> Any:
        """No-op numba.jit fallback when numba is not installed."""
        return lambda f: f

    def set_num_threads(n: int) -> None:
        """No-op numba.set_num_threads fallback when numba is not installed."""
        return

    prange = range


########################################################################################################################


def images2dataframe(
    inputFiles: str | list[str],
    csparc_passthrough_files: list[str] = [],
    alternative_folders: list[str] = [],
    ignore_bad_particle_path: int = 0,
    ignore_bad_micrograph_path: int = 1,
    warn_missing_ctf: int = 1,
    target_convention: str | None = None,
) -> pd.DataFrame:
    """Read one or more image metadata files into a single DataFrame.

    Parameters
    ----------
    inputFiles : str or list of str
        Path to a single file, or a list of file paths.
    csparc_passthrough_files : list of str, optional
        List of cryoSPARC passthrough files for v2+.
    alternative_folders : list of str, optional
        List of alternative folders to search for paths.
    ignore_bad_particle_path : int, optional
        If True, skip particles with missing file paths.
    ignore_bad_micrograph_path : int, optional
        If True, skip micrographs with missing file paths.
    warn_missing_ctf : int, optional
        If True, warn when CTF parameters are missing.
    target_convention : str, optional
        Target Euler angle convention (``"relion"`` or ``"cryosparc"``).

    Returns
    -------
    pd.DataFrame
        DataFrame combining data from all input files, with ``attrs`` containing
        optics, convention, and source_path metadata.
    """
    if isinstance(inputFiles, str):
        data = image2dataframe(
            inputFiles,
            csparc_passthrough_files,
            alternative_folders,
            ignore_bad_particle_path,
            ignore_bad_micrograph_path,
            warn_missing_ctf,
        )
        if target_convention:
            data = dataframe_convert(data, target=target_convention)
        return data

    datalist = []
    opticslist = []
    for fi, inputFile in enumerate(inputFiles):
        p = image2dataframe(
            inputFile,
            csparc_passthrough_files,
            alternative_folders,
            ignore_bad_particle_path,
            ignore_bad_micrograph_path,
            warn_missing_ctf,
        )
        datalist.append(p)
        try:
            if p.attrs["optics"] is not None:
                opticslist.append(p.attrs["optics"])
        except (KeyError, AttributeError):
            pass

    convention = None
    if target_convention is not None:
        convention = target_convention
    else:
        types = set()
        for f in inputFiles:
            t = f.split(".")[-1]
            if t == "star":
                types.add("relion")
            elif t == "cs":
                types.add("cryosparc")  # v2.x
        if len(types) > 1:
            if "relion" in types:
                convention = "relion"
            elif "cryosparc" in types:
                convention = "cryosparc"
    if convention:  # don't convert convention by default
        for pi, p in enumerate(datalist):
            p = dataframe_convert(p, target=target_convention)
            datalist[pi] = p

    data = pd.concat(datalist, sort=False)
    if len(opticslist):
        optics = pd.concat(opticslist, sort=False)
    else:
        optics = None
    data.attrs["optics"] = optics
    data.attrs["convention"] = target_convention
    data.attrs["source_path"] = inputFiles
    data.reset_index(drop=True, inplace=True)  # important to do this
    return data


def image2dataframe(
    inputFile: str,
    csparc_passthrough_files: list[str] = [],
    alternative_folders: list[str] = [],
    ignore_bad_particle_path: int = 0,
    ignore_bad_micrograph_path: int = 1,
    warn_missing_ctf: int = 1,
) -> pd.DataFrame:
    """Read a single image metadata file into a DataFrame.

    Supports .star (RELION), .csv (cryoSPARC v0/v1), .cs (cryoSPARC v2+),
    .db (cisTEM), and MRC/other image formats.

    Parameters
    ----------
    inputFile : str
        Path to the input file. For .db files, the format is
        ``<image_number>@<filename>.db``.
    csparc_passthrough_files : list of str, optional
        List of cryoSPARC passthrough files for v2+.
    alternative_folders : list of str, optional
        List of alternative folders to search for paths.
    ignore_bad_particle_path : int, optional
        If True, skip particles with missing file paths.
    ignore_bad_micrograph_path : int, optional
        If True, skip micrographs with missing file paths.
    warn_missing_ctf : int, optional
        If True, warn when CTF parameters are missing.

    Returns
    -------
    pd.DataFrame
        DataFrame with particle metadata and a ``source_path`` attribute.
    """
    if inputFile.endswith(".db"):
        realInputFile = inputFile.split("@")[-1]
    else:
        realInputFile = inputFile
    if not Path(realInputFile).exists():
        raise HeliconIOError("cannot find file %s" % (realInputFile))

    if inputFile.endswith(".star"):  # relion
        p = star2dataframe(
            inputFile,
            alternative_folders,
            ignore_bad_particle_path,
            ignore_bad_micrograph_path,
        )
    elif inputFile.endswith(".csv"):  # cryosparc v0, v1
        p = csv2dataframe(
            inputFile,
            alternative_folders,
            ignore_bad_particle_path,
            ignore_bad_micrograph_path,
        )
    elif inputFile.endswith(".cs"):  # cryosparc v2+
        p = cs2dataframe(
            inputFile,
            csparc_passthrough_files,
            alternative_folders,
            ignore_bad_particle_path,
            ignore_bad_micrograph_path,
            warn_missing_ctf,
        )
    elif inputFile.endswith(".db"):  # cisTEM
        # will directly read and convert Relion convention
        p = cistem2dataframe(
            inputFile,
            alternative_folders,
            ignore_bad_particle_path,
            ignore_bad_micrograph_path,
        )
    else:
        import collections

        data = []
        imgnum = get_image_number(inputFile, as2D=True)
        for i in range(imgnum):
            d = collections.OrderedDict()
            d["rlnImageName"] = f"{i+1}@{inputFile}"
            data.append(d)
        p = pd.DataFrame(data)
        p.convention = "relion"
    p.attrs["source_path"] = inputFile
    return p


def dataframe2file(data: pd.DataFrame, outputFile: str) -> None:
    """Save a DataFrame to a file in the appropriate format based on extension.

    Supports .star (v3 or old format), .csv, and .cs (cryoSPARC) formats.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame to save.
    outputFile : str
        Output file path. The extension determines the format.
    """
    if len(data) < 1:
        raise HeliconValueError(
            f"dataframe2file(data, outputFile={outputFile}): data is empty, nothing to save"
        )
    if outputFile.endswith(".oldformat.star"):
        dataframe2star(data, outputFile, format="old")
    elif outputFile.endswith(".star"):
        dataframe2star(data, outputFile, format="v3")
    elif outputFile.endswith(".csv"):
        data.to_csv(outputFile)
    elif outputFile.endswith(".cs"):
        dataframe2cs(data, outputFile)
    else:
        raise HeliconValueError(
            "dataframe2file(data, outputFile=%s) is called with a unsupported file format. Only .star and .cs formats are supported"
            % (outputFile)
        )


def guess_data_type(string: str) -> type:
    """Guess the Python type of a string value.

    Tries int, then float, and falls back to str.

    Parameters
    ----------
    string : str
        The string to evaluate.

    Returns
    -------
    type
        One of ``int``, ``float``, or ``str``.
    """
    try:
        v = int(string)
        return int
    except ValueError:
        try:
            v = float(string)
            return float
        except ValueError:
            return str


def dataframe_guess_data_type(data: pd.DataFrame) -> pd.DataFrame:
    """Assign appropriate dtypes to DataFrame columns based on known column names.

    Parameters
    ----------
    data : pd.DataFrame
        pandas DataFrame with cryo-EM metadata columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns cast to int, float, str, or bytes as appropriate.
    """
    intVars = set(
        "pid ppid frame set class sym helicaltube helicalclass vppid vpppos rlnRandomSubset rlnClassNumber classID rlnHelicalTubeID rlnBeamTiltClass rlnClass3DNumber rlnOpticsGroup rlnImageSize rlnImageDimensionality alignments.model-best.k".split()
    )
    floatVars = set(
        "voltage cs ampcont defocus dfdiff dfang btamp btang vps scale asamp asang score".split()
    )
    floatVars.update(
        set(
            [
                "rlnAmplitudeContrast",
                "rlnAnglePsi",
                "rlnAngleRot",
                "rlnAngleTilt",
                "rlnCoordinateX",
                "rlnCoordinateY",
                "rlnDefocusAngle",
                "rlnDefocusU",
                "rlnDefocusV",
                "rlnDetectorPixelSize",
                "rlnImagePixelSize",
                "rlnLogLikeliContribution",
                "rlnMagnification",
                "rlnOriginX",
                "rlnOriginXAngst",
                "rlnOriginY",
                "rlnOriginYAngst",
                "rlnPhaseShift",
                "rlnSphericalAberration",
                "rlnVoltage",
                "rlnCtfMaxResolution",
            ]
        )
    )
    floatVars.update(set(["ctf/ctf_fit_to_A"]))
    strVars = set(
        "emdid filename micrograph movie rlnImageName rlnMicrographName rlnMicrographMovieName".split()
    )
    bytesVars = set(
        "blob/path micrograph_blob/path location/micrograph_path ctf/path mscope_params/defect_path".split()
    )
    known_type_dict = {}
    unknown_type_cols = []
    bytes2str_cols = []
    for col in data:
        if col in intVars:
            known_type_dict[col] = int
        elif col in bytesVars:
            bytes2str_cols.append(col)
        elif col in strVars:
            known_type_dict[col] = str
        elif (
            col in floatVars
            or col.lower().endswith("change")
            or col.lower().endswith("sigma")
            or col.lower().endswith("score")
        ):
            known_type_dict[col] = float
        else:
            unknown_type_cols.append(col)
    if len(known_type_dict):
        for col, dtype in known_type_dict.items():
            data[col] = data[col].astype(dtype)
    if bytes2str_cols:
        for col in bytes2str_cols:
            data[col] = data[col].str.decode("utf-8")
    if unknown_type_cols:
        pass
        # data[unknown_type_cols] = data[unknown_type_cols].apply(pd.to_numeric)

    try:
        optics_df = data.attrs["optics"]
        if optics_df is not None:
            data.attrs["optics"] = dataframe_guess_data_type(optics_df)
    except (KeyError, AttributeError):
        pass

    return data


def star_dissolve_opticsgroup(data: pd.DataFrame) -> None:
    """Copy parameters from the optics block to the main data block.

    Useful for converting a RELION v3+ STAR file to an older format.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with RELION convention and an ``optics`` attribute.
    """
    assert (
        data.attrs["convention"] == "relion"
    ), f"star_dissolve_opticsgroup: requires data in RELION convention. current convention is {data.attrs['convention']}"
    try:
        optics = data.attrs["optics"]
        optics.loc[:, "rlnOpticsGroup"] = optics.loc[:, "rlnOpticsGroup"].astype(str)
    except (KeyError, AttributeError):
        optics = None
    if optics is not None:
        og_names = set(optics["rlnOpticsGroup"].unique())
        data.loc[:, "rlnOpticsGroup"] = data.loc[:, "rlnOpticsGroup"].astype(str)
        for gn, g in data.groupby("rlnOpticsGroup", sort=False):
            if gn not in og_names:
                raise HeliconValueError(
                    f"optic group {gn} not available ({sorted(og_names)})"
                )
            ptcl_indices = g.index
            og_index = optics["rlnOpticsGroup"] == gn
            if "rlnAmplitudeContrast" in optics:
                data.loc[ptcl_indices, "rlnAmplitudeContrast"] = optics.loc[
                    og_index, "rlnAmplitudeContrast"
                ]
            if "rlnImagePixelSize" in optics:
                data.loc[ptcl_indices, "rlnImagePixelSize"] = optics.loc[
                    og_index, "rlnImagePixelSize"
                ]
            if "rlnSphericalAberration" in optics:
                data.loc[ptcl_indices, "rlnSphericalAberration"] = optics.loc[
                    og_index, "rlnSphericalAberration"
                ]
            if "rlnVoltage" in optics:
                data.loc[ptcl_indices, "rlnVoltage"] = optics.loc[
                    og_index, "rlnVoltage"
                ]
            if "rlnMagnification" in optics:
                data.loc[ptcl_indices, "rlnMagnification"] = optics.loc[
                    og_index, "rlnMagnification"
                ]
            if "rlnDetectorPixelSize" in optics:
                data.loc[ptcl_indices, "rlnDetectorPixelSize"] = optics.loc[
                    og_index, "rlnDetectorPixelSize"
                ]
    data.attrs["optics"] = None


# all relion variables are defined here:
# https://github.com/3dem/relion/blob/600499f35c721e2009135ee027078e69414f7edb/src/metadata_label.h
Relion_OpticsGroup_Parameters = (
    "rlnOpticsGroup rlnOpticsGroupName rlnMtfFileName "
    "rlnVoltage rlnSphericalAberration rlnAmplitudeContrast "
    "rlnMagnification rlnDetectorPixelSize "
    "rlnMicrographOriginalPixelSize rlnMicrographPixelSize rlnMicrographBinning "
    "rlnImagePixelSize rlnImageSize rlnImageDimensionality "
    "rlnBeamTiltX rlnBeamTiltY "
    "rlnOddZernike rlnEvenZernike "
    "rlnMagMat00 rlnMagMat01 rlnMagMat10 rlnMagMat11 "
    "rlnCtfDataAreCtfPremultiplied"
).split()


def star_build_opticsgroup(data: pd.DataFrame) -> None:
    """Build optics group block from particle data parameters.

    Extracts optics group parameters present in the main data block,
    groups particles by unique parameter combinations, and creates an
    optics group DataFrame stored in ``data.attrs["optics"]``.

    If ``rlnOpticsGroup`` already exists in the data (e.g. from a
    CryoSPARC exposure group mapping), it is used directly instead of
    generating new sequential IDs from parameter combinations.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with particle data using RELION convention.
    """
    assert (
        data.attrs["convention"] == "relion"
    ), f"star_build_opticsgroup: requires data in RELION convention. current convention is {data.attrs['convention']}"

    vars = [
        v for v in Relion_OpticsGroup_Parameters if v in data and v != "rlnOpticsGroup"
    ]

    if "rlnOpticsGroup" in data:
        if not vars:
            return
        ogp_list = []
        for gn, gdata in data.groupby("rlnOpticsGroup", sort=False):
            d = {"rlnOpticsGroup": gn, "rlnOpticsGroupName": f"opticsGroup{gn}"}
            for v in vars:
                d[v] = gdata[v].values[0]
            ogp_list.append(d)
        optics = pd.DataFrame(ogp_list)
        data.attrs["optics"] = optics
        data.drop(columns=vars, inplace=True)
        return

    if not vars:
        return

    ogp_list = []
    for gi, g in enumerate(
        data.groupby(vars if len(vars) > 1 else vars[0], sort=False)
    ):
        gn, gdata = g
        d = {"rlnOpticsGroup": gi + 1, "rlnOpticsGroupName": f"opticsGroup{gi + 1}"}
        for v in vars:
            d[v] = gdata[v].values[0]
        ogp_list.append(d)
        data.loc[gdata.index, "rlnOpticsGroup"] = gi + 1

    optics = pd.DataFrame(ogp_list)
    data.attrs["optics"] = optics
    data.drop(columns=vars, inplace=True)


def remove_invalid_opticsgroup_parameters(data: pd.DataFrame) -> None:
    """Remove invalid optics group parameters from the optics group.

    Drops columns from ``data.attrs["optics"]`` that are not
    recognised RELION optics group parameters.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing optics group metadata.
    """
    try:
        badVars = []
        if data.attrs["optics"] is not None:
            badVars = [
                v
                for v in data.attrs["optics"]
                if v not in Relion_OpticsGroup_Parameters
            ]
        if "rlnImageName" not in data:
            badVars += [
                v
                for v in "rlnImagePixelSize rlnImageSize".split()
                if v in data.attrs["optics"]
            ]
        if badVars:
            data.attrs["optics"].drop(badVars, axis=1, inplace=True)
    except (KeyError, AttributeError):
        pass

    def missing_required_opticsgroup_parameters(data: pd.DataFrame) -> list[str]:
        """Identify required optics group parameters that are missing.

        Checks the optics group for all required RELION optics group
        parameters and returns a list of those that are absent.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing optics group metadata.

        Returns
        -------
        list of str
            List of missing required optics group parameter names.
        """
        requiredVars = (
            "rlnVoltage rlnSphericalAberration rlnMicrographOriginalPixelSize".split()
        )
        if "rlnImageName" in data:
            requiredVars += "rlnImageSize rlnImagePixelSize".split()
        # if "rlnMicrographName" in data:
        #    requiredVars += "rlnMicrographPixelSize".split()
        missingVars = []
        try:
            if data.attrs["optics"] is not None:
                missingVars = [v for v in requiredVars if v not in data.attrs["optics"]]
            else:
                missingVars = requiredVars
        except (KeyError, AttributeError):
            missingVars = requiredVars
        if "rlnImagePixelSize" in missingVars and "rlnImageName" not in data:
            missingVars.remove("rlnImagePixelSize")
        return missingVars

    missingVars = missing_required_opticsgroup_parameters(data)
    if not missingVars:
        return

    vars = [
        v for v in Relion_OpticsGroup_Parameters if v in data and v != "rlnOpticsGroup"
    ]
    if vars:
        ogp_list = []
        for gi, g in enumerate(
            data.groupby(vars if len(vars) > 1 else vars[0], sort=False)
        ):
            gn, gdata = g
            d = {}
            d["rlnOpticsGroup"] = int(gi + 1)
            d["rlnOpticsGroupName"] = f"opticsGroup{int(gi+1)}"
            for v in vars:
                d[v] = gdata[v].values[0]
            ogp_list.append(d)
            data.loc[gdata.index, "rlnOpticsGroup"] = int(gi + 1)
        optics = pd.DataFrame(ogp_list)
        data.attrs["optics"] = optics
        data.drop(columns=vars, inplace=True)

    try:
        optics = data.attrs["optics"]
        if optics is None:
            raise KeyError(optics)
        if "rlnImageSize" in missingVars and "rlnImageSize" not in optics:
            var = None
            if "rlnImageName" in data:
                var = "rlnImageName"
            elif "rlnMicrographName" in data:
                var = "rlnMicrographName"
            if var:
                imageFileName = data.loc[data.index[0], var].split("@")[-1]
                if Path(imageFileName).exists():
                    nx, ny, nz = get_image_size(imageFileName)
                    optics.loc[:, "rlnImageSize"] = ny
                    optics.loc[:, "rlnImageDimensionality"] = 2
                else:
                    logger.warning(
                        "failed to obtain rlnImageSize, rlnImageDimensionality from non-existing file %s. You should manually add both parameters to the optics group of the star file",
                        imageFileName,
                    )

        """
        if "rlnMicrographPixelSize" in missingVars and "rlnMicrographPixelSize" not in optics:
            if "rlnMicrographOriginalPixelSize" in optics:
                optics.loc[:, "rlnMicrographPixelSize"] = optics.loc[:, "rlnMicrographOriginalPixelSize"]
                logger.warning(f"'rlnMicrographPixelSize' is copied from 'rlnMicrographOriginalPixelSize'. Please manually edit it if it is incorrect")
            elif "rlnImagePixelSize" in optics:
                optics.loc[:, "rlnMicrographPixelSize"] = optics.loc[:, "rlnImagePixelSize"]
                logger.warning(f"'rlnMicrographPixelSize' is copied from 'rlnImagePixelSize'. Please manually edit it if it is incorrect")
        """
        if (
            "rlnMicrographOriginalPixelSize" in missingVars
            and "rlnMicrographOriginalPixelSize" not in optics
        ):
            if "rlnMicrographPixelSize" in optics:
                optics.loc[:, "rlnMicrographOriginalPixelSize"] = optics.loc[
                    :, "rlnMicrographPixelSize"
                ]
                logger.warning(
                    "'rlnMicrographOriginalPixelSize' is copied from 'rlnMicrographPixelSize'. Please manually edit it if it is incorrect"
                )
            elif "rlnImagePixelSize" in optics:
                optics.loc[:, "rlnMicrographOriginalPixelSize"] = optics.loc[
                    :, "rlnImagePixelSize"
                ]
                logger.warning(
                    "'rlnMicrographOriginalPixelSize' is copied from 'rlnImagePixelSize'. Please manually edit it if it is incorrect"
                )
    except KeyError:
        pass

    missingVars = missing_required_opticsgroup_parameters(data)
    if missingVars:
        varval = " ".join([f"{v} <val>" for v in missingVars])
        logger.warning(
            "required OpticsGroup parameters %s are missing. Use \n\timages2star.py <input.star> <output.star> --setParm %s\nto add these parameters",
            " ".join(missingVars),
            varval,
        )


# https://github.com/dzyla/Follow_Relion_gracefully/blob/main/follow_relion_gracefully_lib.py#L352
def star2dataframe(
    starFile: str,
    alternative_folders: list[str] = [],
    ignore_bad_particle_path: int = 0,
    ignore_bad_micrograph_path: int = 1,
) -> pd.DataFrame:
    """Read a RELION .star file into a pandas DataFrame.

    Parses the star file using ``starfile``, selects the first recognised
    data block (particles, micrographs, movies, or coordinate_files),
    attaches the optics group block to ``data.attrs["optics"]``, and
    normalises file paths.

    Parameters
    ----------
    starFile : str
        Path to the .star file.
    alternative_folders : list of str, optional
        List of alternative directory paths to search when resolving
        relative file references.
    ignore_bad_particle_path : int, optional
        If non-zero, skip particles whose image path does not exist.
    ignore_bad_micrograph_path : int, optional
        If non-zero, skip particles whose micrograph path does not exist.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the particle/micrograph data.
    """
    import starfile

    data = None
    d = starfile.read(starFile, always_dict=True)
    for k in d:
        if k in ["movies", "micrographs", "particles", "coordinate_files"]:
            data = d[k]
            break

    if "images" in d:
        if "particles" not in d:
            data = d["images"]
        else:
            logger.warning(
                f"{starFile} contains both 'images' and 'particles' data blocks. Only 'particles' will be read"
            )

    assert (
        data is not None
    ), f"ERROR: {starFile} does not have a required data block (movies, micrographs, or particles/images)"

    if "optics" in d:
        data.attrs["optics"] = d["optics"]

    data = dataframe_guess_data_type(data)
    nans = data.isnull().any(axis=1)
    if nans.sum() > 0:
        logger.warning(
            "%s: %d/%d particle rows are corrupted and thus ignored",
            starFile,
            nans.sum(),
            len(data),
        )
        logger.warning(
            "    Corrupted particle indices:\n%s",
            nans.to_numpy().nonzero()[0],
        )
        if nans.sum() < 100:
            # with pd.option_context("display.max_colwidth", None):
            logger.warning("\n%s", data[nans == True])
        data = data[nans == False]

    data.attrs["source_path"] = starFile
    data.attrs["convention"] = "relion"
    dataframe_normalize_filename(
        data, alternative_folders, ignore_bad_particle_path, ignore_bad_micrograph_path
    )

    return data


def star_to_dataframe(starFile, logger=None):
    """Convert a RELION STAR file to a DataFrame with image index and filename.

    Parses the STAR file, identifies the image name column
    (``rlnImageName`` or ``rlnReferenceImage``), splits the ``index@filename``
    format into separate ``pid`` (0-based) and ``filename`` columns.

    Parameters
    ----------
    starFile : str
        Path to the STAR file.
    logger : logging.Logger, optional
        Logger for error messages.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional ``pid`` and ``filename`` columns.
    """
    df = star2dataframe(starFile=starFile)

    fileNameCol = ""
    for col in ["rlnImageName", "rlnReferenceImage"]:
        if col in df:
            fileNameCol = col
            break
    if not fileNameCol:
        msg = f"ERROR: cannot find 'rlnImageName' or 'rlnReferenceImage' in the input {starFile}"
        if logger:
            logger.error(msg)
        raise KeyError(msg)

    tmp = df[fileNameCol].str.split("@", expand=True)
    indices, filenames = tmp.iloc[:, 0], tmp.iloc[:, -1]
    indices = indices.astype(int) - 1
    df["pid"] = indices
    df["filename"] = filenames
    return df


def dataframe2star(data: pd.DataFrame, starFile: str | Any, format: str = "v3") -> None:
    """Write a pandas DataFrame to a RELION .star file.

    Converts the DataFrame to RELION convention, optionally builds or
    dissolves optics groups depending on the output format version, and
    writes the result to disk (or a file-like object).

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame (any convention supported by ``dataframe_convert``).
    starFile : str or file-like
        Output file path or a file-like object with a ``write`` method.
    format : str, optional
        Output star file version. ``"v3"`` or ``"relion3"`` builds an
        optics group block; any other value dissolves it.
    """
    data2 = dataframe_convert(data, target="relion")

    if "rlnImageName" in data2:
        data2 = mrc2mrcs(data2)
        # check image formats
        micrographNames = data2["rlnImageName"].str.split("@", expand=True).iloc[:, -1]
        suffix = micrographNames.str.rsplit(".", n=1, expand=True).iloc[:, -1].unique()
        if not set(suffix).issubset(["mrcs", "mrc", "tnf", "spi", "img", "hed"]):
            mgraphs = micrographNames.groupby(micrographNames, sort=False)
            for mi, mgraph in enumerate(mgraphs):
                mgraphName, mgraphParticles = mgraph
                # make sure that the particles are in an image format supported by Relion
                # https://www2.mrc-lmb.cam.ac.uk/relion/index.php/Conventions_%26_File_formats#Image_I.2FO
                suffix = Path(mgraphName).suffix
                if suffix not in [".mrcs", ".mrc", ".tnf", ".spi", ".img", ".hed"]:
                    logger.warning(
                        "RELION does not support image format: %s",
                        mgraphName,
                    )

    if format in ["v3", "relion3"]:
        star_build_opticsgroup(data2)
        remove_invalid_opticsgroup_parameters(data2)
        if "rlnImageName" not in data and "rlnMicrographName" in data:
            data_block_tag = "data_micrographs"
        else:
            data_block_tag = "data_particles"
    else:
        star_dissolve_opticsgroup(data2)
        data_block_tag = "data_"

    data2 = dataframe_guess_data_type(data2)

    if hasattr(starFile, "read") and callable(starFile.read):
        fp = starFile
    else:
        fp = open(starFile, "wt")

    try:
        optics = data2.attrs.get("optics")
        if optics is not None and len(optics) > 0:
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
    except (KeyError, OSError):
        pass

    fp.write(f"\n{data_block_tag}\n\nloop_ \n")
    keys = [k for k in data2.columns if k.startswith("rln")]
    for ki, k in enumerate(keys):
        fp.write("_%s #%d \n" % (k, ki + 1))
    lines = data2[keys[0]].astype(str)
    for k in keys[1:]:
        if data2[k].dtype == np.float64:
            lines += "\t" + data2[k].round(6).astype(str)
        else:
            lines += "\t" + data2[k].astype(str)
    fp.write("\n".join(lines))
    fp.write("\n")


def _detect_cs_import_origin(csFile: str) -> tuple:
    """Detect if a .cs file originated from a RELION STAR import.

    Reads the first ``blob/path`` to extract the import job name, then
    checks for ``{project_dir}/{import_job}/particles.star`` and
    ``{project_dir}/{import_job}/imported_particles.cs``.

    Returns
    -------
    tuple
        ``(detected, import_star_path, import_uids, uid_to_row)``.
        When *detected* is ``False``, the remaining entries are
        ``("", [], {})``.
    """
    try:
        cs_path = Path(csFile).resolve()
        cs = np.load(str(cs_path), allow_pickle=True)
        cs_dtype = cs.dtype

        if cs_dtype.names is None or "blob/path" not in cs_dtype.names or len(cs) == 0:
            return (False, "", [], {})

        raw_path = cs[0]["blob/path"]
        first_path = raw_path.decode() if isinstance(raw_path, bytes) else str(raw_path)

        first_slash = first_path.find("/")
        if first_slash < 0:
            return (False, "", [], {})
        import_job = first_path[:first_slash]

        project_dir = str(cs_path.parent.parent)
        import_star_path = f"{project_dir}/{import_job}/particles.star"
        import_cs_path = f"{project_dir}/{import_job}/imported_particles.cs"

        if not (Path(import_star_path).exists() and Path(import_cs_path).exists()):
            return (False, "", [], {})

        # Read imported_particles.cs to get uid→row mapping
        cs_imp = np.load(import_cs_path, allow_pickle=True)
        if cs_imp.dtype.names is None or "uid" not in cs_imp.dtype.names:
            return (False, "", [], {})

        import_uids = [int(row["uid"]) for row in cs_imp]
        uid_to_row = {uid: i for i, uid in enumerate(import_uids)}

        logger.info(
            "Detected .cs from RELION import. Using original STAR: %s "
            "(total=%d, selected=%d)",
            import_star_path,
            len(import_uids),
            len(cs),
        )
        return (True, import_star_path, import_uids, uid_to_row)

    except Exception:
        return (False, "", [], {})


def _cs2dataframe_from_star_import(
    csFile: str,
    passthrough_files: list[str],
    import_star_path: str,
    import_uids: list,
    uid_to_row: dict,
    alternative_folders: list[str],
    ignore_bad_particle_path: int,
    ignore_bad_micrograph_path: int,
) -> pd.DataFrame:
    """Convert a .cs file using the original RELION STAR as data source.

    The .cs file's particles are a subset of the original STAR's particles.
    The original STAR file is used as the data source (preserving all original
    RELION fields) and the .cs file as a subset selector via uid matching.
    CryoSPARC-refined fields from the .cs file (class, alignments, CTF) are
    overlaid on the selected particles.

    Parameters
    ----------
    csFile : str
        Path to the .cs file (subset selector + overlay fields).
    passthrough_files : list of str
        CryoSPARC passthrough .cs files (currently unused in this path;
        the target .cs supplies overlay fields directly).
    import_star_path : str
        Path to the original imported RELION STAR file.
    import_uids : list of int
        UIDs from ``imported_particles.cs``, in row order (same as STAR rows).
    uid_to_row : dict
        Mapping uid → row index in *import_uids*.
    alternative_folders, ignore_bad_particle_path, ignore_bad_micrograph_path
        Forwarded to ``dataframe_normalize_filename``.
    """
    import starfile

    # 1. Read target .cs (overlay fields + selected uids)
    cs = np.load(csFile, allow_pickle=True)
    cs_df = pd.DataFrame.from_records(cs.tolist(), columns=cs.dtype.names)
    selected_uids = set(int(uid) for uid in cs_df["uid"]) if "uid" in cs_df else set()

    # 2. Read original STAR file via star2dataframe (handles optics, typing, etc.)
    # Pass ignore_bad_*=2 to skip path resolution: the original STAR's
    # rlnImageName paths are RELION-relative and may not exist on the current
    # filesystem.  The actual image paths come from the .cs blob/path, which
    # is resolved separately through the CryoSPARC project structure.
    star_data = star2dataframe(
        import_star_path,
        alternative_folders,
        ignore_bad_particle_path=2,
        ignore_bad_micrograph_path=2,
    )

    # 3. Validate sizes
    if len(star_data) != len(import_uids):
        logger.warning(
            "%s: STAR has %d rows but imported_particles.cs has %d uids. Truncating.",
            csFile,
            len(star_data),
            len(import_uids),
        )
        min_len = min(len(star_data), len(import_uids))
        star_data = star_data.iloc[:min_len].reset_index(drop=True)
        import_uids = import_uids[:min_len]
        uid_to_row = {uid: i for i, uid in enumerate(import_uids)}

    # 4. Filter STAR data to selected uids
    if not selected_uids:
        logger.warning("%s: no uid field, returning original STAR data as-is", csFile)
        return star_data

    star_data["_uid"] = import_uids
    data = star_data[star_data["_uid"].isin(selected_uids)].copy()

    if len(data) == 0:
        raise HeliconIOError(
            f"_cs2dataframe_from_star_import: no matching uids in {csFile}"
        )

    uids_in_data = list(data["_uid"])
    data.drop(columns=["_uid"], inplace=True)
    data.reset_index(drop=True, inplace=True)

    # 5. Overlay CryoSPARC-refined fields
    cs_by_uid = cs_df.set_index("uid")

    if "alignments2D/class" in cs.dtype.names:
        cls_map = {}
        for uid in uids_in_data:
            try:
                cls_map[uid] = int(cs_by_uid.loc[uid, "alignments2D/class"]) + 1
            except (KeyError, TypeError, ValueError):
                pass
        if cls_map:
            data["rlnClassNumber"] = data.index.to_series().map(
                lambda i: cls_map.get(uids_in_data[i])
            )

    if "alignments2D/shift" in cs.dtype.names:
        sx_map, sy_map = {}, {}
        for uid in uids_in_data:
            try:
                shift = np.atleast_1d(
                    np.asarray(cs_by_uid.loc[uid, "alignments2D/shift"], dtype=float)
                )
                sx = float(shift[0])
                sy = float(shift[1]) if len(shift) > 1 else 0.0
                apix = 1.0
                if "blob/psize_A" in cs.dtype.names:
                    try:
                        apix = float(cs_by_uid.loc[uid, "blob/psize_A"])
                    except (KeyError, TypeError, ValueError):
                        pass
                sx_map[uid] = (-sx) * apix
                sy_map[uid] = (-sy) * apix
            except (KeyError, TypeError, ValueError):
                pass
        if sx_map:
            data["rlnOriginXAngst"] = data.index.to_series().map(
                lambda i: sx_map.get(uids_in_data[i])
            )
            data["rlnOriginYAngst"] = data.index.to_series().map(
                lambda i: sy_map.get(uids_in_data[i])
            )

    if "alignments2D/pose" in cs.dtype.names:
        psi_map = {}
        for uid in uids_in_data:
            try:
                psi = float(cs_by_uid.loc[uid, "alignments2D/pose"])
                psi_map[uid] = -psi * (180.0 / np.pi)
            except (KeyError, TypeError, ValueError):
                pass
        if psi_map:
            data["rlnAnglePsi"] = data.index.to_series().map(
                lambda i: psi_map.get(uids_in_data[i])
            )

    # CTF overlay (overrides original values with CryoSPARC-refined ones)
    ctf_overlays = [
        ("ctf/df1_A", "rlnDefocusU", 1.0),
        ("ctf/df2_A", "rlnDefocusV", 1.0),
        ("ctf/df_angle_rad", "rlnDefocusAngle", 180.0 / np.pi),
        ("ctf/phase_shift_rad", "rlnPhaseShift", 1.0),
        ("ctf/bfactor", "rlnCtfBfactor", 1.0),
        ("ctf/scale", "rlnCtfScalefactor", 1.0),
    ]
    for cs_field, rln_name, mul in ctf_overlays:
        if cs_field not in cs.dtype.names:
            continue
        val_map = {}
        for uid in uids_in_data:
            try:
                val_map[uid] = float(cs_by_uid.loc[uid, cs_field]) * mul
            except (KeyError, TypeError, ValueError):
                pass
        if val_map:
            data[rln_name] = data.index.to_series().map(
                lambda i: val_map.get(uids_in_data[i])
            )

    # 6. Final cleanup: remove NaN overlay entries (where .cs lacked a value)
    data.attrs["source_path"] = csFile
    data.attrs["convention"] = "relion"
    return data


def cs2dataframe(
    csFile: str,
    passthrough_files: list[str] = [],
    alternative_folders: list[str] = [],
    ignore_bad_particle_path: int = 0,
    ignore_bad_micrograph_path: int = 1,
    warn_missing_ctf: int = 1,
) -> pd.DataFrame:
    """Read a CryoSPARC v2+ .cs file into a pandas DataFrame.

    Loads the numpy structured array from a CryoSPARC metadata file,
    optionally merges additional passthrough files on the ``uid`` column,
    guesses column data types, drops corrupted rows, normalises file
    paths, and sets the convention to ``"cryosparc"``.

    Parameters
    ----------
    csFile : str
        Path to the .cs file.
    passthrough_files : list of str, optional
        List of additional .cs passthrough files to merge. If empty,
        attempts to auto-discover a matching passthrough file.
    alternative_folders : list of str, optional
        Alternative directory paths to search when resolving relative
        file references.
    ignore_bad_particle_path : int, optional
        If non-zero, skip particles whose image path does not exist.
    ignore_bad_micrograph_path : int, optional
        If non-zero, skip particles whose micrograph path does not exist.
    warn_missing_ctf : int, optional
        If non-zero, print a warning when CTF information is absent
        (unless the input is a ``templates_selected.cs`` file).

    Returns
    -------
    pd.DataFrame
        DataFrame containing the CryoSPARC particle/micrograph data.
    """
    # Auto-detect if this .cs file originated from a RELION STAR import.
    # If so, use the original STAR file as data source (preserving all original
    # RELION fields) with the .cs as subset selector, overlaying CryoSPARC-
    # refined fields (class, alignments, CTF) on the selected particles.
    _detected, _star_path, _uids, _uid_row = _detect_cs_import_origin(csFile)
    if _detected:
        return _cs2dataframe_from_star_import(
            csFile,
            passthrough_files,
            _star_path,
            _uids,
            _uid_row,
            alternative_folders,
            ignore_bad_particle_path,
            ignore_bad_micrograph_path,
        )

    # read CryoSPARC v2+ meta data
    cs = np.load(csFile, allow_pickle=True)
    data = pd.DataFrame.from_records(cs.tolist(), columns=cs.dtype.names)
    if passthrough_files:
        passthrough_files_final = passthrough_files * 1
    else:
        passthrough_files_final = []
        p = Path(csFile)
        if p.name.startswith("particles_"):
            ptFile = f"*J[0-9]*_passthrough_{p.name}"  # Select2D
        else:
            ptFile = f"*J[0-9]*_passthrough_particles.cs"  # Class2D
        ptfs = list(p.parent.glob(ptFile))
        if ptfs:
            passthrough_files_final.append(ptfs[0])
    if passthrough_files_final:
        extra_data = []
        for f in passthrough_files_final:
            cs = np.load(f)
            extra_data.append(
                pd.DataFrame.from_records(cs.tolist(), columns=cs.dtype.names)
            )
        for extra_df in extra_data:
            # Drop passthrough columns that already exist in the input file
            # to avoid _x/_y suffix renaming from merge()
            cols_to_drop = [
                c for c in extra_df.columns if c != "uid" and c in data.columns
            ]
            if cols_to_drop:
                extra_df = extra_df.drop(columns=cols_to_drop)
            data = data.merge(extra_df, on="uid", how="left")
        data = data.loc[:, ~data.columns.duplicated()]
    if "blob/path" not in data and "micrograph_blob/path" not in data:
        raise HeliconIOError(
            f"it appears that you have specified a CryoSPARC v2 passthrough file that does not have particle/micrograph path info. Available parameters are: {data.columns.values}"
        )
    if (
        warn_missing_ctf
        and "ctf/accel_kv" not in data
        and csFile.find("templates_selected.cs") == -1
    ):
        logger.warning(
            "CTF info not found. You should also provide the passthrough file that has CTF info"
        )
    if "ctf/type" in data:
        data = data.drop("ctf/type", axis=1)

    data = dataframe_guess_data_type(data)
    nans = data.isnull().any(axis=1)
    if nans.sum() > 0:
        logger.warning(
            "%s: %d/%d particle rows are corrupted and thus ignored",
            csFile,
            nans.sum(),
            len(data),
        )
        logger.warning(
            "    Corrupted particle indices:\n%s",
            nans.to_numpy().nonzero()[0],
        )
        if nans.sum() < 100:
            logger.warning("\n%s", data[nans == True])
        data = data[nans == False]
    data.attrs["source_path"] = csFile
    data.attrs["convention"] = "cryosparc"
    dataframe_normalize_filename(
        data, alternative_folders, ignore_bad_particle_path, ignore_bad_micrograph_path
    )
    return data


def dataframe2cs(data: pd.DataFrame, csFile: str) -> None:
    """Write a pandas DataFrame to a CryoSPARC .cs file.

    Parameters
    ----------
    data : pd.DataFrame
        pandas DataFrame with cryosparc-convention columns.
    csFile : str
        Path to the output .cs file.
    """
    structured_array = data.to_records(index=False)
    dtypes = []
    for col_name in structured_array.dtype.names:
        if structured_array[col_name].dtype.kind == "O":
            max_len = max(map(len, structured_array[col_name]))
            dtypes.append((col_name, f"S{max_len}"))
        else:
            dtypes.append((col_name, structured_array[col_name].dtype.kind))
    structured_array = structured_array.astype(dtypes)
    with open(csFile, "wb") as f:
        np.save(f, structured_array)


def cistem2dataframe(
    dbFile: str,
    alternative_folders: list[str] = [],
    ignore_bad_particle_path: int = 0,
    ignore_bad_micrograph_path: int = 1,
) -> pd.DataFrame:
    """Read a cisTEM SQLite database and return a DataFrame in RELION convention.

    Parameters
    ----------
    dbFile : str
        Path to the cisTEM SQLite database. May be prefixed with an
        iteration number and ``@`` (e.g. ``"3@/path/to/db"``).
    alternative_folders : list of str, optional
        Additional folders to search for files.
    ignore_bad_particle_path : int, optional
        If 1, skip missing particle files without error.
    ignore_bad_micrograph_path : int, optional
        If 1, skip missing micrograph files without error.

    Returns
    -------
    pd.DataFrame
        DataFrame with relion-convention column names.
    """
    import sqlalchemy

    if dbFile.find("@") == -1:
        iter = -1
    else:
        iter, dbFile = dbFile.split("@")
        iter = int(iter)
    db = sqlalchemy.create_engine("sqlite:///%s" % (dbFile))
    refinement_list = pd.read_sql_table("REFINEMENT_LIST", db).set_index(
        "REFINEMENT_ID"
    )
    if iter < 0:
        refinement_id = refinement_list.index.max() + 1 + iter
    else:
        refinement_id = iter
    refinement_package_asset_id = refinement_list.loc[
        refinement_id, "REFINEMENT_PACKAGE_ASSET_ID"
    ]

    refinement_package_asset_table = pd.read_sql_table(
        "REFINEMENT_PACKAGE_ASSETS", db
    ).set_index("REFINEMENT_PACKAGE_ASSET_ID")
    stack_filename = str(
        refinement_package_asset_table.loc[
            refinement_package_asset_id, "STACK_FILENAME"
        ]
    )

    refinement_input_table_name = "REFINEMENT_PACKAGE_CONTAINED_PARTICLES_%s" % (
        refinement_package_asset_id
    )
    input_table = pd.read_sql_table(refinement_input_table_name, db).set_index(
        "POSITION_IN_STACK"
    )

    if "PARENT_IMAGE_ASSET_ID" in input_table:
        image_asset_table = pd.read_sql_table("IMAGE_ASSETS", db).set_index(
            "IMAGE_ASSET_ID"
        )

        input_table_group = input_table.groupby("PARENT_IMAGE_ASSET_ID")
        for image_asset_id, mgraphParticles in input_table_group:
            if int(image_asset_id) < 1:
                continue  # sometimes cisTEM db file set this field to -1. why?
            input_table.loc[mgraphParticles.index, "FILENAME"] = image_asset_table.loc[
                image_asset_id, "FILENAME"
            ]

    refinement_result_table_name = "REFINEMENT_RESULT_%s_%s" % (refinement_id, 1)
    result_cistem = pd.read_sql_table(refinement_result_table_name, db).set_index(
        "POSITION_IN_STACK"
    )

    duplicate_cols = [c for c in input_table if c in result_cistem]
    input_table.drop(duplicate_cols, inplace=True, axis=1)

    cols = list(input_table.columns) + list(result_cistem.columns)
    data_cistem = result_cistem.combine_first(input_table)
    cols = [c for c in cols if c in data_cistem]
    data_cistem = data_cistem[cols]
    data_cistem.reset_index(drop=False, inplace=True)

    data = pd.DataFrame()

    data["rlnImageName"] = (
        data_cistem["POSITION_IN_STACK"].astype(int).map("{:06d}".format)
        + "@"
        + stack_filename
    )

    mapping = {
        "PSI": "rlnAnglePsi",
        "THETA": "rlnAngleTilt",
        "PHI": "rlnAngleRot",
        "XSHIFT": "rlnOriginX",
        "YSHIFT": "rlnOriginY",
        "DEFOCUS1": "rlnDefocusU",
        "DEFOCUS2": "rlnDefocusV",
        "DEFOCUS_ANGLE": "rlnDefocusAngle",
        "PHASE_SHIFT": "rlnPhaseShift",
        "LOGP": "rlnLogLikeliContribution",
    }
    mapping.update(
        {
            "FILENAME": "rlnMicrographName",
            "X_POSITION": "rlnCoordinateX",
            "Y_POSITION": "rlnCoordinateY",
            "PIXEL_SIZE": "rlnMagnification",
            "SPHERICAL_ABERRATION": "rlnSphericalAberration",
            "MICROSCOPE_VOLTAGE": "rlnVoltage",
            "AMPLITUDE_CONTRAST": "rlnAmplitudeContrast",
        }
    )

    for key in data_cistem.columns:
        if key in mapping:
            if key == "PIXEL_SIZE":
                data["rlnImagePixelSize"] = data_cistem[key].astype(float)
                # data["rlnDetectorPixelSize"] = 5
                # data["rlnMagnification"] = (5e4/data_cistem[key].astype(float)).round(1)
            else:
                value = mapping[key]
                if key in "X_POSITION Y_POSITION XSHIFT YSHIFT":
                    data_cistem[key] = (
                        (
                            data_cistem[key].astype(float)
                            / data_cistem["PIXEL_SIZE"].astype(float)
                        )
                    ).round(3)
                    if key in "XSHIFT YSHIFT":
                        data_cistem[key] *= -1
                elif key == "PHASE_SHIFT":
                    data_cistem[key] = np.rad2deg(data_cistem[key].astype(float)).round(
                        1
                    )
                data[value] = data_cistem[key]

    data.attrs["source_path"] = dbFile
    data.attrs["convention"] = "relion"
    dataframe_normalize_filename(
        data, alternative_folders, ignore_bad_particle_path, ignore_bad_micrograph_path
    )

    return data


def dataframe_normalize_filename(
    data: pd.DataFrame,
    alternative_folders: list[str] = [],
    ignore_bad_particle_path: int = 0,
    ignore_bad_micrograph_path: int = 1,
) -> pd.DataFrame:
    """Normalize filenames in a DataFrame by resolving relative paths.

    For each column containing filenames (e.g. rlnImageName, rlnMicrographName),
    relative paths are resolved to absolute paths using a variety of search
    strategies including the source file directory and alternative folders.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with filename columns to normalize.
    alternative_folders : list of str, optional
        Additional folders to search for files.
    ignore_bad_particle_path : int, optional
        If >= 2, skip filename normalization entirely.
    ignore_bad_micrograph_path : int, optional
        If 1, skip missing micrograph files without error.

    Returns
    -------
    pd.DataFrame
        DataFrame with resolved absolute file paths.
    """
    if ignore_bad_particle_path >= 2:
        return data

    def getRealFileName(
        filename: str,
        source_path: str | list[str],
        alternative_folders: list[str] = [],
        ignore_bad_path: int = 0,
    ) -> str:
        """Resolve a filename to its real absolute path.

        Searches for the file in the source directory, alternative folders,
        and the RELION project folder. Results are cached in
        ``getRealFileName.mapping`` to avoid repeated filesystem checks.

        Parameters
        ----------
        filename : str
            The filename to resolve (may be relative).
        source_path : str or list of str
            Path to the source file for deriving search folders.
        alternative_folders : list of str, optional
            Additional folders to search.
        ignore_bad_path : int, optional
            If 1, return the original filename if not found.

        Returns
        -------
        str
            The resolved absolute path to the file.
        """
        if filename in getRealFileName.mapping:
            return getRealFileName.mapping[filename]

        basenames = []
        if not Path(filename).is_absolute():
            basenames.append(filename)
        basenames.append(Path(filename).name)

        basenames += [f[:-4] + ".mrcs" for f in basenames if f.endswith(".mrc")]

        folders = [folder for folder in alternative_folders]
        if isinstance(source_path, str):
            folders += [str(Path(source_path).resolve().parent)]
        elif isinstance(source_path, (list, tuple, set)):
            folders += [str(Path(sp).resolve().parent) for sp in source_path]

        relion_folder = get_relion_project_folder(filename)
        if relion_folder is not None:
            folders.append(relion_folder)

        filenameChoices = [filename]

        for basename in basenames:
            for folder in folders:
                filenameChoices += [str(Path(folder) / basename)]
                filenameChoices += [str(Path(folder) / ".." / basename)]
                filenameChoices += [str(Path(folder) / "../.." / basename)]

        match = None
        match_link = None
        for fci, fc in enumerate(filenameChoices):
            if Path(fc).is_file():
                match = fc
                break
            if Path(fc).is_symlink():
                match_link = fc

        if match:
            ret = str(Path(match).resolve())

            # when a new folder is found to have a matching file, all files of the same file type in the folder
            # will be pre-mapped to avoid multiple file checks later
            import glob

            suffix = Path(filename).suffix
            filename_dir = Path(filename).parent
            allfiles = glob.glob(str(Path(match).parent / ("*" + Path(match).suffix)))
            for f in allfiles:
                tmp_basename = Path(f).stem + suffix
                getRealFileName.mapping[str(filename_dir / tmp_basename)] = str(
                    Path(f).resolve()
                )
        else:
            if ignore_bad_path:
                ret = filename
            elif match_link:
                raise HeliconIOError(
                    "image %s in file %s is found at %s that is a broken link to %s"
                    % (
                        filename,
                        source_path,
                        str(Path(match_link).resolve()),
                        os.readlink(match_link),
                    )
                )
            else:
                raise HeliconIOError(
                    f"cannot find image {filename} in file {source_path} after trying these choices: {filenameChoices}"
                )
        return ret

    getRealFileName.mapping = {}

    cache = {}

    def buildFileNameCache(
        filenames: Any,
        source_path: str | list[str],
        alternative_folders: list[str] = [],
        ignore_bad_path: int = 0,
    ) -> None:
        """Build a filename resolution cache for a list of filenames.

        Populates ``cache`` with the resolved absolute path for each filename.

        Parameters
        ----------
        filenames : iterable
            Iterable of filenames to resolve.
        source_path : str or list of str
            Path to the source file for deriving search folders.
        alternative_folders : list of str, optional
            Additional folders to search.
        ignore_bad_path : int, optional
            If 1, return the original filename if not found.
        """
        for fi, filename in enumerate(filenames):
            cache[filename] = getRealFileName(
                filename, source_path, alternative_folders, ignore_bad_path
            )

    attrs = []
    attrs_with_at = []
    for (
        attr
    ) in "rlnImageName rlnMicrographName rlnMicrographMovieName rlnMicrographCoordinates".split():
        if attr in data:
            if attr == "rlnImageName":
                ignore_bad_path = ignore_bad_particle_path
            else:
                ignore_bad_path = ignore_bad_micrograph_path
            if "@" in data[attr][0]:
                attrs_with_at.append((attr, ignore_bad_path))
            else:
                attrs.append((attr, ignore_bad_path))
    for attr in "data_input_relpath blob/path filename".split():
        if attr in data:
            attrs.append((attr, ignore_bad_particle_path))
    for attr in "micrograph_blob/path location/micrograph_path micrograph".split():
        if attr in data:
            attrs.append((attr, ignore_bad_micrograph_path))
    if len(attrs):
        for attr, ignore_bad_path in attrs:
            filenames = data[attr].unique()
            buildFileNameCache(
                filenames,
                data.attrs["source_path"],
                alternative_folders,
                ignore_bad_path=ignore_bad_path,
            )
            data[attr] = data[attr].map(cache)
    if len(attrs_with_at):
        for attr, ignore_bad_path in attrs_with_at:
            tmp = data[attr].str.split("@", expand=True)
            indices, filenames = tmp.iloc[:, 0], tmp.iloc[:, -1]
            buildFileNameCache(
                filenames.unique(),
                data.attrs["source_path"],
                alternative_folders,
                ignore_bad_path=ignore_bad_path,
            )
            data[attr] = indices + "@" + filenames.map(cache)

    return data


# see Figure 1 and Eq 5 of https://doi.org/10.1016/j.jsb.2015.08.008
def relion_astigmatism_to_eman(
    rlnDefocusU: float, rlnDefocusV: float, rlnDefocusAngle: float
) -> tuple[float, float, float]:
    """Convert RELION astigmatism parameters to EMAN2 format.

    Parameters
    ----------
    rlnDefocusU : float
        Defocus U in Angstroms (RELION convention).
    rlnDefocusV : float
        Defocus V in Angstroms (RELION convention).
    rlnDefocusAngle : float
        Defocus angle in degrees (RELION convention).

    Returns
    -------
    tuple of (float, float, float)
        (defocus, dfdiff, dfang) in EMAN2 convention (units of um).
    """
    rlnDefocusU = float(rlnDefocusU)
    rlnDefocusV = float(rlnDefocusV)
    rlnDefocusAngle = float(rlnDefocusAngle)
    # check ctffind3.f function CTF(CS,WL,WGH1,WGH2,DFMID1,DFMID2,ANGAST,THETATR, IX, IY)
    defocus = (rlnDefocusU + rlnDefocusV) / 2 / 1e4
    dfdiff = (
        math.fabs(rlnDefocusU - rlnDefocusV) / 2 / 1e4
    )  # rlnDefocusU can be larger or smaller than rlnDefocusV
    if rlnDefocusU > rlnDefocusV:
        dfang = math.fmod(
            rlnDefocusAngle + 360.0 + 90.0, 360.0
        )  # largest defocus direction to smallest defocus direction
    else:
        dfang = rlnDefocusAngle  # already at smallest defocus direction
    return (defocus, dfdiff, dfang)


def eman_astigmatism_to_relion(
    defocus: float, dfdiff: float, dfang: float
) -> tuple[float, float, float]:
    """Convert EMAN2 astigmatism parameters to RELION format.

    Parameters
    ----------
    defocus : float
        Average defocus in um (EMAN2 convention).
    dfdiff : float
        Defocus difference in um (EMAN2 convention).
    dfang : float
        Defocus angle in degrees (EMAN2 convention).

    Returns
    -------
    tuple of (float, float, float)
        (rlnDefocusU, rlnDefocusV, rlnDefocusAngle) in RELION convention
        (Angstroms, degrees).
    """
    if math.fmod(dfang + 360, 180) < 90:
        rlnDefocusU = defocus - dfdiff
        rlnDefocusV = defocus + dfdiff
    else:
        rlnDefocusU = defocus + dfdiff
        rlnDefocusV = defocus - dfdiff
    rlnDefocusAngle = math.fmod(dfang + 360, 90)
    return (rlnDefocusU * 1e4, rlnDefocusV * 1e4, rlnDefocusAngle)


def get_dataframe_convention(data: pd.DataFrame) -> str:
    """Get or guess the naming convention of a DataFrame.

    Checks the ``convention`` attribute first; if not set, guesses based on
    which column names are present (relion vs cryosparc).

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame to inspect.

    Returns
    -------
    str
        ``"relion"`` or ``"cryosparc"``.

    Raises
    ------
    AttributeError
        If the convention cannot be determined.
    """
    try:
        c = data.attrs["convention"]  # test if the convention is set
        assert c is not None and len(c) > 0
    except (
        KeyError,
        AssertionError,
    ):  # let's guess the convention if it is not set yet
        if any(
            k in data
            for k in "rlnImageName rlnMicrographName rlnMicrographMovieName rlnVoltage".split()
        ):
            c = "relion"
        elif any(
            k in data
            for k in "blob/path micrograph_blob/path movie_blob/path location/micrograph_path".split()
        ):
            c = "cryosparc"
        else:
            msg = "ERROR: get_dataframe_convention(): unrecognized dataframe. only relion, and cryosparc conventions are supported"
            raise AttributeError(msg)
    return c


def dataframe_convert(data: pd.DataFrame, target: str = "relion") -> pd.DataFrame:
    """Convert a DataFrame between naming conventions.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame to convert.
    target : str, optional
        Target convention, ``"relion"`` or ``"cryosparc"``.

    Returns
    -------
    pd.DataFrame
        DataFrame in the target convention.

    Raises
    ------
    AttributeError
        If the conversion is not supported.
    """
    data.attrs["convention"] = get_dataframe_convention(data)

    if data.attrs["convention"] == target:
        return data

    msg = f"ERROR: dataframe_convert(): unavailable conversion of convention from {data.attrs['convention']} to {target}"
    if data.attrs["convention"] == "relion":
        if target == "cryosparc":
            return dataframe_relion_to_cryosparc(data)
        else:
            raise AttributeError(msg)
    elif data.attrs["convention"] == "cryosparc":
        if target == "relion":
            return dataframe_cryosparc_to_relion(data)
        else:
            raise AttributeError(msg)
    else:
        raise AttributeError(msg)


def _electron_wavelength(voltage_kv: float | np.ndarray) -> np.ndarray:
    """Calculate the relativistic electron wavelength.

    Parameters
    ----------
    voltage_kv : float or np.ndarray
        Acceleration voltage in kV (scalar or array-like).

    Returns
    -------
    np.ndarray
        Wavelength in Angstroms (same shape as input).
    """
    h = 6.62607015e-34
    m_e = 9.1093837e-31
    e = 1.602176634e-19
    c = 299792458
    V = np.asarray(voltage_kv, dtype=float) * 1000.0
    lam = h / np.sqrt(2 * m_e * e * V * (1 + e * V / (2 * m_e * c**2)))
    return lam * 1e10


def clean_cs_micrograph_path(path: str) -> str:
    """Strip cryoSPARC hash prefix and ``_patch_aligned_doseweighted`` suffix from a micrograph filename.

    Parameters
    ----------
    path : str
        CryoSPARC micrograph path (absolute or relative).

    Returns
    -------
    str
        Cleaned filename with hash and ``_patch_aligned_doseweighted`` removed.
    """
    name = Path(path).name
    parts = name.split("_", 1)
    if len(parts) == 2 and len(parts[0]) > 10 and parts[0].isdigit():
        name = parts[1]
    name = name.replace("_patch_aligned_doseweighted", "")
    return name


def dataframe_cryosparc_to_relion(data: pd.DataFrame) -> pd.DataFrame:
    """Convert a CryoSPARC-convention DataFrame to RELION convention.

    Maps cryosparc column names (e.g. ``blob/path``, ``ctf/df1_A``) to
    their RELION equivalents (``rlnImageName``, ``rlnDefocusU``, etc.).

    Both CryoSPARC and RELION use top-left origin for particle
    coordinates in the micrograph, so no Y-axis inversion is applied:
    ``rlnCoordinateY = center_y_frac * height``.

    Parameters
    ----------
    data : pd.DataFrame
        pandas DataFrame with cryosparc-convention columns.

    Returns
    -------
    pd.DataFrame
        pandas DataFrame with relion-convention column names.

    Raises
    ------
    AttributeError
        If the input is not in cryosparc convention.
    """
    data.attrs["convention"] = get_dataframe_convention(data)

    if data.attrs["convention"] == "relion":
        return data

    if data.attrs["convention"] != "cryosparc":
        msg = f"ERROR: dataframe_cryosparc_to_relion(): input dataframe is in {data.attrs['convention']} instead of the required cryosparc convention"
        raise AttributeError(msg)

    ret = pd.DataFrame()
    if "blob/idx" in data and "blob/path" in data:
        ret["rlnImageName"] = (
            (data["blob/idx"].astype(int) + 1).map("{:06d}".format)
            + "@"
            + data["blob/path"]
        )
    if "micrograph_blob/path" in data:
        ret["rlnMicrographName"] = data["micrograph_blob/path"]
    if "location/micrograph_path" in data:
        ret["rlnMicrographName"] = data["location/micrograph_path"]
    if "movie_blob/path" in data:
        ret["rlnMicrographMovieName"] = data["movie_blob/path"]
    if "ctf/accel_kv" in data:
        ret["rlnVoltage"] = data["ctf/accel_kv"]
    if "ctf/cs_mm" in data:
        ret["rlnSphericalAberration"] = data["ctf/cs_mm"]
    if "ctf/amp_contrast" in data:
        ret["rlnAmplitudeContrast"] = data["ctf/amp_contrast"]
    if "ctf/df1_A" in data and "ctf/df2_A" in data and "ctf/df_angle_rad" in data:
        ret["rlnDefocusU"] = data["ctf/df1_A"]
        ret["rlnDefocusV"] = data["ctf/df2_A"]
        ret["rlnDefocusAngle"] = np.rad2deg(data["ctf/df_angle_rad"])
    if "ctf/phase_shift_rad" in data:
        ret["rlnPhaseShift"] = np.rad2deg(data["ctf/phase_shift_rad"])
    if "ctf/ctf_fit_to_A" in data:
        ret["rlnCtfMaxResolution"] = data["ctf/ctf_fit_to_A"]
    if "blob/psize_A" in data:
        ret["rlnImagePixelSize"] = data["blob/psize_A"]
        # ret["rlnDetectorPixelSize"] = 5.0
        # ret["rlnMagnification"] = (ret["rlnDetectorPixelSize"]*1e4/data["blob/psize_A"]).round(1)
    if "micrograph_blob/psize_A" in data:
        ret["rlnMicrographPixelSize"] = data["micrograph_blob/psize_A"]
    if "alignments3D/split" in data:
        ret["rlnRandomSubset"] = data["alignments3D/split"] + 1

    # 2D class assignments
    if "alignments2D/class" in data:
        ret["rlnClassNumber"] = data["alignments2D/class"].astype(int) + 1
    origin_x = origin_y = None
    if "alignments2D/shift" in data:
        shifts = pd.DataFrame(data["alignments2D/shift"].tolist()).round(2)
        origin_x = -shifts.iloc[:, 0]
        origin_y = -shifts.iloc[:, 1]
    if "alignments2D/pose" in data:
        ret["rlnAnglePsi"] = -np.rad2deg(data["alignments2D/pose"]).round(2)

    # 3D class assignments
    if "alignments3D/class" in data:
        ret["rlnClassNumber"] = data["alignments3D/class"].astype(int) + 1
    if "alignments3D/cross_cor" in data:
        ret["rlnLogLikeliContribution"] = data[
            "alignments3D/cross_cor"
        ]  # not a good pair relationship. other better choice?

    if "alignments3D/pose" in data:
        # cryosparc rotation r vector is in the rotvec format
        from scipy.spatial.transform import Rotation as R

        rotvecs = list(data["alignments3D/pose"].values)
        r = R.from_rotvec(rotvecs)
        e = r.as_euler("ZYZ", degrees=True)
        ret["rlnAngleRot"] = e[:, 0]
        ret["rlnAngleTilt"] = e[:, 1]
        ret["rlnAnglePsi"] = e[:, 2]

    if "alignments3D/shift" in data:
        shifts = pd.DataFrame(data["alignments3D/shift"].tolist()).round(2)
        origin_x = shifts.iloc[:, 0]
        origin_y = shifts.iloc[:, 1]

    # Output Angstrom origins for RELION 3.1+ convention (pixel origins deprecated)
    if origin_x is not None and "blob/psize_A" in data:
        apix = data["blob/psize_A"]
        ret["rlnOriginXAngst"] = (origin_x * apix).round(6)
        ret["rlnOriginYAngst"] = (origin_y * apix).round(6)

    if "location/center_x_frac" in data and "location/center_y_frac" in data:
        if "location/micrograph_shape" in data:
            loc_shape = data["location/micrograph_shape"]
        elif "micrograph_blob/shape" in data:
            loc_shape = data["micrograph_blob/shape"]
        else:
            loc_shape = None
        if loc_shape is not None:
            shape_df = pd.DataFrame(loc_shape.tolist())
            my = shape_df.iloc[:, 0]
            mx = shape_df.iloc[:, 1]
            y_frac = data["location/center_y_frac"]
            ret["rlnCoordinateX"] = (
                (data["location/center_x_frac"] * mx).astype(float).round(2)
            )
            ret["rlnCoordinateY"] = (y_frac * my).astype(float).round(2)

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
                    val = f["filament/position_A"].astype(np.float32).values.copy()
                    val -= np.min(val)
                    ret.loc[f.index, "rlnHelicalTrackLengthAngst"] = val.round(2)
        else:
            mapping = {
                v: i + 1
                for i, v in enumerate(sorted(data["filament/filament_uid"].unique()))
            }
            ret.loc[:, "rlnHelicalTubeID"] = data["filament/filament_uid"].map(mapping)

    if "filament/filament_pose" in data:
        ret.loc[:, "rlnAngleRotPrior"] = 0.0
        ret.loc[:, "rlnAngleTiltPrior"] = 90.0
        ret.loc[:, "rlnAnglePsiPrior"] = np.round(
            -np.rad2deg(data["filament/filament_pose"]), 1
        )
        ret.loc[:, "rlnAnglePsiFlipRatio"] = 0.5

    if "ctf/bfactor" in data:
        ret["rlnCtfBfactor"] = data["ctf/bfactor"]

    if "ctf/scale" in data:
        ret["rlnCtfScalefactor"] = data["ctf/scale"]

    # High-order aberrations
    # Beam tilt: ctf/tilt_A (Å) → rlnBeamTiltX/Y (mrad)
    # Using the established formula from pyem:
    #   tilt_mrad = arcsin(tilt_A / cs_mm * 1e-7) * 1e3
    # sign convention: positive mrad corresponds to positive Å displacement
    if "ctf/tilt_A" in data and "ctf/cs_mm" in data:
        cs_mm = data["ctf/cs_mm"].values
        tilt_vals = np.stack(data["ctf/tilt_A"].values)
        tilt_x_A = tilt_vals[:, 0]
        tilt_y_A = tilt_vals[:, 1]
        ret["rlnBeamTiltX"] = (np.arcsin(tilt_x_A / cs_mm * 1e-7) * 1e3).round(8)
        ret["rlnBeamTiltY"] = (np.arcsin(tilt_y_A / cs_mm * 1e-7) * 1e3).round(8)

    # Trefoil: ctf/trefoil_A cannot be reliably converted to RELION's
    # rlnOddZernike coefficients because the Å-to-Zernike conversion
    # depends on spatial frequency (not a single constant).
    # This matches pyem's behaviour (pass / not implemented).
    # Silence the warning when all values are zero (cryoSPARC default).
    if "ctf/trefoil_A" in data:
        vals = np.stack(data["ctf/trefoil_A"].values)
        if not np.allclose(vals, 0):
            logger.warning(
                "ctf/trefoil_A found in cryoSPARC data but not converted to STAR file. "
                "RELION encodes trefoil via rlnOddZernike in the optics group; "
                "the Å-to-Zernike conversion is frequency-dependent and not yet implemented. "
                "Run CtfRefine in RELION to fit trefoil from the data."
            )

    # Tetrafoil: ctf/tetra_A cannot be reliably converted (same reason as trefoil).
    # Silence the warning when all values are zero (cryoSPARC default).
    if "ctf/tetra_A" in data:
        vals = np.stack(data["ctf/tetra_A"].values)
        if not np.allclose(vals, 0):
            logger.warning(
                "ctf/tetra_A found in cryoSPARC data but not converted to STAR file. "
                "RELION encodes tetrafoil via rlnEvenZernike in the optics group; "
                "the Å-to-Zernike conversion is frequency-dependent and not yet implemented. "
                "Run CtfRefine in RELION to fit tetrafoil from the data."
            )

    # Anisotropic magnification: direct copy (matches pyem convention)
    if "ctf/anisomag" in data:
        anisomag_vals = np.stack(data["ctf/anisomag"].values)
        ret["rlnMagMat00"] = anisomag_vals[:, 0]
        ret["rlnMagMat01"] = anisomag_vals[:, 1]
        ret["rlnMagMat10"] = anisomag_vals[:, 2]
        ret["rlnMagMat11"] = anisomag_vals[:, 3]

    # Exposure group → optics group
    for exp_col in [
        "ctf/exp_group_id",
        "location/exp_group_id",
        "mscope_params/exp_group_id",
    ]:
        if exp_col in data:
            ret["rlnOpticsGroup"] = data[exp_col].astype(int)
            break

    # 3D variability introduced in v2.9
    import fnmatch

    v3d_cols = [
        col for col in data.columns if fnmatch.fnmatch(col, "components_mode_*/value")
    ]
    for col in v3d_cols:
        ci = col.split("/")[0].split("_")[-1]
        col_name = "v3d%s" % (ci)
        ret[col_name] = data[col]

    if len(ret.columns) == 0:
        raise HeliconValueError(
            "dataframe_cryosparc_to_relion(): none of the parameters %s is supported"
            % (list(data.columns))
        )

    ret = reorder_dataframe_columns(ret)

    try:
        ret.attrs["source_path"] = data.attrs["source_path"]
    except KeyError:
        ret.attrs["source_path"] = None
    ret.attrs["convention"] = "relion"

    return ret


def mrc2mrcs(data: pd.DataFrame) -> pd.DataFrame:
    """Convert .mrc file references in a DataFrame to .mrcs symlinks.

    For each unique .mrc file, checks for a companion .mrcs that is a
    symlink or hard link to the same file; if none exists, creates one.

    Parameters
    ----------
    data : pd.DataFrame
        pandas DataFrame with a ``filename`` column (or an
        ``rlnImageName`` column from which the filename is extracted).

    Returns
    -------
    pd.DataFrame
        pandas DataFrame with .mrcs references.
    """
    if "rlnImageName" in data:  # Relion star file
        tmp = data["rlnImageName"].str.split("@", expand=True)
        pid = tmp.iloc[:, 0]
        data.loc[:, "filename"] = tmp.iloc[:, -1]
    if "filename" not in data:
        return data

    names = set(data["filename"])
    mapping = {f: f for f in names}
    mrc_names = [f for f in names if f.endswith(".mrc")]
    if mrc_names:
        for name in mrc_names:
            mrc_path = Path(name)
            mrc_resolved = mrc_path.resolve()
            mrcs_path = mrc_path.with_suffix(".mrcs")

            # Check if companion .mrcs already links to the same file
            if mrcs_path.is_symlink() and mrcs_path.resolve() == mrc_resolved:
                mapping[name] = str(mrcs_path)
                continue
            if (
                mrcs_path.exists()
                and mrcs_path.stat().st_ino == mrc_resolved.stat().st_ino
                and mrcs_path.stat().st_dev == mrc_resolved.stat().st_dev
            ):
                mapping[name] = str(mrcs_path)
                continue

            folder = mrc_path.parent
            if not os.access(str(folder), os.W_OK):
                folder = Path("./mrc2mrcs")
                folder.mkdir(parents=True, exist_ok=True)
                mrcs_path = folder / (mrc_path.name + "s")
            mapping[name] = str(mrcs_path)
            if not mrcs_path.exists():
                if mrcs_path.is_symlink():
                    mrcs_path.unlink()
                os.symlink(str(mrc_resolved), str(mrcs_path))
        data.loc[:, "filename"] = data["filename"].map(mapping)
    if "rlnImageName" in data:  # Relion star file
        data.loc[:, "rlnImageName"] = pid.astype(str) + "@" + data["filename"]
        data.drop(["filename"], axis=1, inplace=True)
    return data


#####################################################################################
def connect_cryosparc(
    cryosparc_server_info_file: str = "$HOME/.cryosparc/cryosparc.toml",
) -> Any:
    """Connect to a CryoSPARC server using credentials from a TOML file.

    Parameters
    ----------
    cryosparc_server_info_file : str, optional
        Path to the TOML file containing server credentials.
        Defaults to ``~/.cryosparc/cryosparc.toml``.

    Returns
    -------
    cryosparc.tools.CryoSPARC
        Connected CryoSPARC client instance.

    Raises
    ------
    HeliconConfigError
        If the credentials file is missing or has insecure permissions.
    """

    def print_instructions():
        """Print setup instructions for the CryoSPARC credentials file."""
        info = "To connect to CryoSPARC server, please follow these instructions:\n"
        info += f"1. create a text file {cryosparc_server_info_file}\n"
        info += "2. change its permission to user readable/writable only by running this command:\n"
        info += f"   chmod 600 {cryosparc_server_info_file}\n"
        info += f"3. add the following info to {cryosparc_server_info_file}:\n\n"
        info += 'license = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"\n'
        info += 'host = "xxx.yyy.zzz.edu"\n'
        info += "base_port = 39000\n"
        info += 'email = "xxx@yyy.edu"\n'
        info += 'password = "yourpassowrd"\n\n'
        info += "Remember to change the placeholder text to your own information\n\n"
        logger.info(info)

    p = Path(os.path.expandvars(cryosparc_server_info_file))
    if not p.exists():
        print_instructions()
        raise HeliconConfigError(
            f"CryoSPARC server info file not found: {cryosparc_server_info_file}"
        )
    elif oct(p.stat().st_mode)[-3:] != "600":
        raise HeliconConfigError(
            f"Please run command 'chmod 600 {cryosparc_server_info_file}' to keep your server info secure"
        )

    with open(p, mode="rb") as fp:
        import tomllib

        info = tomllib.load(fp)

        from cryosparc.tools import CryoSPARC

        cs = CryoSPARC(
            license=info["license"],
            host=info["host"],
            base_port=info["base_port"],
            email=info["email"],
            password=info["password"],
        )
        assert cs.test_connection()
    return cs
