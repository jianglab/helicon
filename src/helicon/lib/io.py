import os, sys
from pathlib import Path
import numpy as np
import pandas as pd

pd.options.mode.copy_on_write = True

from .util import color_print


def get_image_number(imageFile, as2D=False):
    if not os.path.exists(imageFile):
        color_print(f"ERROR: cannot find image file {imageFile}")
        sys.exit()
    import mrcfile

    with mrcfile.open(imageFile, header_only=True) as mrc:
        if as2D:
            n = mrc.header.nz
        else:
            n = 1
    return n


def get_image_size(imageFile):
    if not os.path.exists(imageFile):
        color_print(f"ERROR: cannot find image file {imageFile}")
        sys.exit()
    import mrcfile

    with mrcfile.open(imageFile, header_only=True) as mrc:
        nz = mrc.header.nz
        ny = mrc.header.ny
        nx = mrc.header.nx
    return (nx, ny, nz)


def read_image_2d(imageFile, i):
    if not os.path.exists(imageFile):
        color_print(f"ERROR: cannot find image file {imageFile}")
        sys.exit()
    i = int(i)
    import mrcfile

    with mrcfile.open(imageFile) as mrc:
        nz = mrc.header.nz
        if 0 <= i < nz:
            return mrc.data[i]
        else:
            color_print(
                f"ERROR: the requested image {i} is out of the valid range [0, {nz}) for image file {imageFile}"
            )
            sys.exit()


def change_map_axes_order(data, header, new_axes=["x", "y", "z"]):
    import numpy as np

    map_axes = {"x": 0, "y": 1, "z": 2}
    try:
        current_axes_int = [header.mapc - 1, header.mapr - 1, header.maps - 1]
    except:
        current_axes_int = [0, 1, 2]
    new_axes_int = [map_axes[a] for a in new_axes]
    data2 = np.moveaxis(data, current_axes_int, new_axes_int)
    header2 = header.copy()
    header2.mapc = new_axes_int[0] + 1
    header2.mapr = new_axes_int[1] + 1
    header2.maps = new_axes_int[2] + 1
    return data2, header2


def display_map_orthoslices(data, title, hold=False):
    if not sys.__stdin__.isatty():
        return
    nz, ny, nx = data.shape
    sz = data[nz // 2, :, :]
    sy = data[:, ny // 2, :]
    sx = data[:, :, nx // 2]
    images = [sx, sy, sz]
    titles = ["X=%d" % (nx // 2), "Y=%d" % (ny // 2), "Z=%d" % (nz // 2)]
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 3.5), facecolor="w", edgecolor="w")
    fig.suptitle(title, fontsize=16)
    for i in range(3):
        sub = fig.add_subplot(1, 3, i + 1)
        sub.imshow(images[i], cmap="gray", aspect="equal", interpolation="bicubic")
        sub.set_title(titles[i], fontsize=12)
    fig.tight_layout()
    if hold:
        plt.show()
    else:
        plt.draw()
    plt.pause(0.05)


def get_relion_project_folder(starFile):
    filename_abs = os.path.abspath(starFile)
    if filename_abs.find("/job") == -1:
        return None
    parts = filename_abs.split("/")
    for pi, p in enumerate(parts):
        if p.startswith("job"):
            break
    job_folder = "/".join(parts[: pi + 1])
    if not os.path.exists(os.path.join(job_folder, "default_pipeline.star")):
        return None
    pi = max(0, pi - 1)
    proj_folder = "/".join(parts[:pi])
    if not os.path.exists(os.path.join(proj_folder, "default_pipeline.star")):
        return None
    return proj_folder


def movie_filename_patterns():
    # EPU:
    # FoilHole_30593197_Data_30537205_30537207_20230430_084907_fractions_patch_aligned_doseweighted.mrc
    # FoilHole_28788144_Data_28764755_46_20240328_192116_fractions.tiff
    d = dict(
        EPU_old=r"FoilHole_\d{7,8}_Data_\d{7,8}_\d{8}_\d{8}_\d{6}_",
        EPU=r"FoilHole_\d{7,8}_Data_\d{7,8}_(\d{1,3})_\d{8}_\d{6}_",
        serialEM_pncc=r"([XY][\+-]\d[XY][\+-]\d-\d)",
    )
    return d


def guess_data_collection_software(filename):
    import re

    format = None
    patterns = movie_filename_patterns()
    for p in patterns:
        if re.search(patterns[p], filename) is not None:
            format = p
            break
    return format


def verify_data_collection_software(filename, software):
    import re

    match = re.search(movie_filename_patterns()[software], filename)
    return match


def extract_EPU_data_collection_time(filename):
    import re

    pattern = r"FoilHole_\d{7,8}_Data_\d{7,8}_\d{1,3}_(\d{8}_\d{6})_"
    match = re.search(pattern, filename)
    if match:
        from datetime import datetime

        datetime_str = match.group(1)
        datetime_obj = datetime.strptime(datetime_str, "%Y%m%d_%H%M%S")
        timestamp = datetime_obj.timestamp()
        return timestamp
    else:
        print(filename)
        print(pattern)
        raise
    return 0


def extract_EPU_old_data_collection_time(filename):
    import re

    pattern = r"FoilHole_\d{8}_Data_\d{8}_\d{8}_(\d{8}_\d{6})_"
    match = re.search(pattern, filename)
    if match:
        from datetime import datetime

        datetime_str = match.group(1)
        datetime_obj = datetime.strptime(datetime_str, "%Y%m%d_%H%M%S")
        timestamp = datetime_obj.timestamp()
        return timestamp
    else:
        print(filename)
        print(pattern)
        raise
    return 0


def extract_EPU_beamshift_pos(filename):
    import re

    pattern = r"FoilHole_\d{7,8}_Data_\d{7}_(\d{1,3})_\d{8}_\d{6}_"
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    else:
        print(filename)
        print(pattern)
        raise
    return ""


def extract_serialEM_pncc_beamshift(filename):
    import re

    pattern = r"([XY][\+-]\d[XY][\+-]\d-\d)"
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    else:
        print(filename)
        print(pattern)
        raise
    return ""


def EPU_micrograph_path_2_movie_xml_path(micrograph_path, movies_folder):
    import re

    pattern = r"(\d{21}_FoilHole_\d{8}_Data_\d{8}_\d{8}_\d{8}_\d{6}_fractions)"
    match = re.search(pattern, micrograph_path)
    if match:
        mid = match.group(1)
        from pathlib import Path

        xml_path = (Path(movies_folder) / (mid + ".tiff")).resolve()
        xml_path = str(xml_path).replace("_fractions.tiff", ".xml")
        assert Path(xml_path).exists(), f"{xml_path} does not exist"
        return xml_path
    else:
        raise ValueError("ERROR: bad micrograph path: {micrograph_path}")


def EPU_xml_2_beamshift(xml_file):
    import xmltodict

    with open(xml_file, "rb") as fp:
        xml = xmltodict.parse(fp, dict_constructor=dict)
    beamshift = xml["MicroscopeImage"]["microscopeData"]["optics"]["BeamShift"]
    beamshift = (float(beamshift["a:_x"]), float(beamshift["a:_y"]))
    return beamshift


def assign_beamshifts_to_cluster(
    beamshifts, min_cluster_size=4, range_n_clusters=range(2, 200), verbose=True
):
    from sklearn.metrics import silhouette_score
    from .analysis import AgglomerativeClusteringWithMinSize

    X = np.array(beamshifts)

    # Evaluate silhouette scores for different numbers of clusters
    best_n_clusters = range_n_clusters[0]
    best_score = -1
    best_cluster_labels = None
    for n_clusters in range_n_clusters:
        clustering_method = AgglomerativeClusteringWithMinSize(
            n_clusters=n_clusters, min_cluster_size=min_cluster_size
        )
        cluster_labels = clustering_method.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_n_clusters = n_clusters
            best_cluster_labels = cluster_labels
    if verbose:
        print(
            f"The optimal number of clusters is {best_n_clusters} with a silhouette score of {best_score:.2f}"
        )
    beamshifts_tuples = [tuple(v) for v in beamshifts]
    data_cluster_dict = {
        tuple(data_point): cluster_id + 1
        for data_point, cluster_id in zip(beamshifts_tuples, best_cluster_labels)
    }
    return data_cluster_dict


def euler_relion2eman(rot, tilt, psi):
    # order of rotation: rot around z, tilt around y, psi around z
    # order of rotation: az around z, alt around x, phi around z
    az = rot + 90.0
    alt = tilt
    phi = psi - 90.0
    return az, alt, phi


def euler_eman2relion(az, alt, phi):
    # order of rotation: rot around z, tilt around y, psi around z
    # order of rotation: az around z, alt around x, phi around z
    rot = az - 90
    tilt = alt
    psi = phi + 90
    return rot, tilt, psi


def eman_euler2quaternion(az, alt, phi):
    import quaternionic as qtn

    alpha_beta_gamma = np.vstack(
        (np.deg2rad(az - 90), np.deg2rad(alt), np.deg2rad(phi + 90))
    ).T  # z,y,z convention
    q = qtn.array.from_euler_angles(alpha_beta_gamma, beta=None, gamma=None).normalized
    q = np.array(q)  # qtn.array cannot be pickled for joblib Parallel processing
    return q


def relion_euler2quaternion(rot, tilt, psi):
    import quaternionic as qtn

    alpha_beta_gamma = np.vstack(
        (np.deg2rad(rot), np.deg2rad(tilt), np.deg2rad(psi))
    ).T  # z,y,z convention
    q = qtn.array.from_euler_angles(alpha_beta_gamma, beta=None, gamma=None).normalized
    q = np.array(q)  # qtn.array cannot be pickled for joblib Parallel processing
    return q


def quaternion2euler(q, euler_convention="relion"):
    import quaternionic as qtn

    alpha_beta_gamma = np.rad2deg(qtn.array(q).reshape((-1, 4)).to_euler_angles)
    rot, tilt, psi = (
        alpha_beta_gamma[:, 0],
        alpha_beta_gamma[:, 1],
        alpha_beta_gamma[:, 2],
    )
    if len(q.shape) == 1:
        rot, tilt, psi = rot[0], tilt[0], psi[0]
    rot = set_angle_range(rot, range=[-180, 180])
    tilt = set_angle_range(tilt, range=[-180, 180])
    psi = set_angle_range(psi, range=[-180, 180])
    if euler_convention in ["relion"]:
        return rot, tilt, psi
    elif euler_convention in ["eman"]:
        return euler_relion2eman(rot, tilt, psi)
    raise ValueError


# https://github.com/christophhagen/averaging-quaternions


# Q is a Nx4 numpy matrix and contains the quaternions to average in the rows.
# The quaternions are arranged as (w,x,y,z), with w being the scalar
# The result will be the average quaternion of the input. Note that the signs
# of the output quaternion can be reversed, since q and -q describe the same orientation
# The weight vector w must be of the same length as the number of rows in the
# quaternion maxtrix Q
def average_quaternions(Q, w=None):
    # Number of quaternions to average
    assert w is None or len(w) == Q.shape[0]

    import numpy
    import numpy.matlib as npm

    M = Q.shape[0]
    A = npm.zeros(shape=(4, 4))
    weightSum = 0

    if w is None:
        w = numpy.ones(M)

    for i in range(0, M):
        q = Q[i, :]
        A = w[i] * numpy.outer(q, q) + A
        weightSum += w[i]

    # scale
    A = (1.0 / weightSum) * A

    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = numpy.linalg.eig(A)

    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:, eigenValues.argsort()[::-1]]

    # return the real part of the largest eigenvector (has only real part)
    ret = np.real(eigenVectors[:, 0].A1)
    ret = ret * 1.0  # ensure contigous array
    return ret


def average_relion_eulers(rot, tilt, psi, weights=None, return_quaternion=False):
    assert len(rot) == len(tilt) and len(rot) == len(psi)
    if weights:
        assert len(weights) == len(rot)
    Q = relion_euler2quaternion(rot, tilt, psi)
    qm = average_quaternions(Q, w=weights)
    if return_quaternion:
        return qm
    else:
        rot_mean, tilt_mean, psi_mean = quaternion2relion_euler(qm)
        return rot_mean, tilt_mean, psi_mean


def angular_distance(rotation_1, rotation_2):
    # rotation_1/rotation_2: scipy.spatial.transform.Rotation
    mag = (rotation_1.inv() * rotation_2).magnitude()
    return np.rad2deg(mag)


try:
    from numba import jit, set_num_threads, prange
except ImportError:
    color_print(
        f"WARNING: failed to load numba. The program will run correctly but will be much slower. Run 'pip install numba' to install numba and speed up the program"
    )

    def jit(*args, **kwargs):
        return lambda f: f

    def set_num_threads(n: int):
        return

    prange = range


@jit(nopython=True, cache=False, nogil=True, parallel=True)
def apply_helical_symmetry(
    data,
    apix,
    twist_degree,
    rise_angstrom,
    csym=1,
    fraction=1.0,
    new_size=None,
    new_apix=None,
    cpu=1,
):
    if new_apix is None:
        new_apix = apix
    nz0, ny0, nx0 = data.shape
    if new_size != data.shape:
        nz1, ny1, nx1 = new_size
        nz2, ny2, nx2 = max(nz0, nz1), max(ny0, ny1), max(nx0, nx1)
        data_work = np.zeros((nz2, ny2, nx2), dtype=np.float32)
    else:
        data_work = np.zeros((nz0, ny0, nx0), dtype=np.float32)

    nz, ny, nx = data_work.shape
    w = np.zeros((nz, ny, nx), dtype=np.float32)

    hsym_max = max(1, int(nz * new_apix / rise_angstrom))
    hsyms = range(-hsym_max, hsym_max + 1)
    csyms = range(csym)

    mask = (data != 0) * 1
    z_nonzeros = np.nonzero(mask)[0]
    z0 = np.min(z_nonzeros)
    z1 = np.max(z_nonzeros)
    z0 = max(z0, nz0 // 2 - int(nz0 * fraction + 0.5) // 2)
    z1 = min(nz0 - 1, min(z1, nz0 // 2 + int(nz0 * fraction + 0.5) // 2))

    set_num_threads(cpu)

    for hi in hsyms:
        for k in prange(nz):
            k2 = ((k - nz // 2) * new_apix + hi * rise_angstrom) / apix + nz0 // 2
            if k2 < z0 or k2 >= z1:
                continue
            k2_floor, k2_ceil = int(np.floor(k2)), int(np.ceil(k2))
            wk = k2 - k2_floor

            for ci in csyms:
                rot = np.deg2rad(twist_degree * hi + 360 * ci / csym)
                m = np.array([[np.cos(rot), np.sin(rot)], [-np.sin(rot), np.cos(rot)]])
                for j in prange(ny):
                    for i in prange(nx):
                        j2 = (
                            m[0, 0] * (j - ny // 2) + m[0, 1] * (i - nx / 2)
                        ) * new_apix / apix + ny0 // 2
                        i2 = (
                            m[1, 0] * (j - ny // 2) + m[1, 1] * (i - nx / 2)
                        ) * new_apix / apix + nx0 // 2

                        j2_floor, j2_ceil = int(np.floor(j2)), int(np.ceil(j2))
                        i2_floor, i2_ceil = int(np.floor(i2)), int(np.ceil(i2))
                        if j2_floor < 0 or j2_floor >= ny0 - 1:
                            continue
                        if i2_floor < 0 or i2_floor >= nx0 - 1:
                            continue

                        wj = j2 - j2_floor
                        wi = i2 - i2_floor

                        data_work[k, j, i] += (
                            (1 - wk)
                            * (1 - wj)
                            * (1 - wi)
                            * data[k2_floor, j2_floor, i2_floor]
                            + (1 - wk)
                            * (1 - wj)
                            * wi
                            * data[k2_floor, j2_floor, i2_ceil]
                            + (1 - wk)
                            * wj
                            * (1 - wi)
                            * data[k2_floor, j2_ceil, i2_floor]
                            + (1 - wk) * wj * wi * data[k2_floor, j2_ceil, i2_ceil]
                            + wk
                            * (1 - wj)
                            * (1 - wi)
                            * data[k2_ceil, j2_floor, i2_floor]
                            + wk * (1 - wj) * wi * data[k2_ceil, j2_floor, i2_ceil]
                            + wk * wj * (1 - wi) * data[k2_ceil, j2_ceil, i2_floor]
                            + wk * wj * wi * data[k2_ceil, j2_ceil, i2_ceil]
                        )
                        w[k, j, i] += 1.0
    mask = w > 0
    data_work = np.where(mask, data_work / w, data_work)
    if data_work.shape != new_size:
        nz1, ny1, nx1 = new_size
        data_work = data_work[
            nz // 2 - nz1 // 2 : nz // 2 + nz1 // 2,
            ny // 2 - ny1 // 2 : ny // 2 + ny1 // 2,
            nx // 2 - nx1 // 2 : nx // 2 + nx1 // 2,
        ]
    data_work = np.ascontiguousarray(data_work)
    return data_work


########################################################################################################################


def images2dataframe(
    inputFiles,
    csparc_passthrough_files=[],
    alternative_folders=[],
    ignore_bad_particle_path=0,
    ignore_bad_micrograph_path=1,
    warn_missing_ctf=1,
    target_convention=None,
):
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
        except:
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
    inputFile,
    csparc_passthrough_files=[],
    alternative_folders=[],
    ignore_bad_particle_path=0,
    ignore_bad_micrograph_path=1,
    warn_missing_ctf=1,
):
    if inputFile.endswith(".db"):
        realInputFile = inputFile.split("@")[-1]
    else:
        realInputFile = inputFile
    if not os.path.exists(realInputFile):
        color_print("ERROR: cannot find file %s" % (realInputFile))
        sys.exit(-1)

    if inputFile.endswith(".star"):  # relion
        p = star2dataframe(
            inputFile,
            alternative_folders,
            ignore_bad_particle_path,
            ignore_bad_micrograph_path,
        )
    elif inputFile.endswith(".csv"):  # cryosparc v0.x
        p = csv2dataframe(
            inputFile,
            alternative_folders,
            ignore_bad_particle_path,
            ignore_bad_micrograph_path,
        )
    elif inputFile.endswith(".cs"):  # cryosparc v2.x
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


def dataframe2file(data, outputFile):
    if len(data) < 1:
        color_print(
            f"WARNING: dataframe2file(data, outputFile={outputFile}): data is empty, nothing to save"
        )
        sys.exit(-1)
    if outputFile.endswith(".oldformat.star"):
        dataframe2star(data, outputFile, format="old")
    elif outputFile.endswith(".star"):
        dataframe2star(data, outputFile, format="v3")
    elif outputFile.endswith(".csv"):
        data.to_csv(outputFile)
    elif outputFile.endswith(".cs"):
        dataframe2cs(data, outputFile)
    else:
        color_print(
            "ERROR: dataframe2file(data, outputFile=%s) is called with a unsupported file format. Only .star and .cs formats are supported"
            % (outputFile)
        )
        sys.exit(-1)


def guess_data_type(string):
    try:
        v = int(string)
        return int
    except:
        try:
            v = float(string)
            return float
        except:
            return str


def dataframe_guess_data_type(data):
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
        data.attrs["optics"] = dataframe_guess_data_type(data.attrs["optics"])
    except:
        pass

    return data


def star_dissolve_opticsgroup(data):
    """copy parameters from optics block to the main data block.
    useful for converting a new star file for Relion v3+ to older star file format
    """
    assert (
        data.attrs["convention"] == "relion"
    ), f"star_dissolve_opticsgroup: requires data in relion convention. current convention is {data.attrs["convention"]}"
    try:
        optics = data.attrs["optics"]
        optics.loc[:, "rlnOpticsGroup"] = optics.loc[:, "rlnOpticsGroup"].astype(str)
    except:
        optics = None
    if optics is not None:
        og_names = set(optics["rlnOpticsGroup"].unique())
        data.loc[:, "rlnOpticsGroup"] = data.loc[:, "rlnOpticsGroup"].astype(str)
        for gn, g in data.groupby("rlnOpticsGroup", sort=False):
            if gn not in og_names:
                color_print(
                    f"ERROR: optic group {gn} not available ({sorted(og_names)})"
                )
                sys.exit(-1)
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
Relion_OpticsGroup_Parameters = "rlnOpticsGroup rlnOpticsGroupName rlnMtfFileName rlnVoltage rlnSphericalAberration rlnAmplitudeContrast rlnMagnification rlnDetectorPixelSize rlnMicrographOriginalPixelSize rlnMicrographPixelSize rlnMicrographBinning rlnImagePixelSize rlnImageSize rlnImageDimensionality rlnBeamTiltX rlnBeamTiltY rlnOddZernike rlnEvenZernike".split()


def star_build_opticsgroup(data):
    assert (
        data.attrs["convention"] == "relion"
    ), f"star_build_opticsgroup: requires data in relion convention. current convention is {data.attrs["convention"]}"

    def remove_invalid_opticsgroup_parameters(data):
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
        except:
            pass

    def missing_required_opticsgroup_parameters(data):
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
        except:
            missingVars = requiredVars
        if "rlnImagePixelSize" in missingVars and "rlnImageName" not in data:
            missingVars.remove("rlnImagePixelSize")
        return missingVars

    remove_invalid_opticsgroup_parameters(data)

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
        if "rlnImageSize" in missingVars and "rlnImageSize" not in optics:
            var = None
            if "rlnImageName" in data:
                var = "rlnImageName"
            elif "rlnMicrographName" in data:
                var = "rlnMicrographName"
            if var:
                imageFileName = data.loc[data.index[0], var].split("@")[-1]
                if os.path.exists(imageFileName):
                    nx, ny, nz = get_image_size(imageFileName)
                    dim = 3 if nz > 1 else 2
                    optics.loc[:, "rlnImageSize"] = ny
                    optics.loc[:, "rlnImageDimensionality"] = dim
                else:
                    color_print(
                        f"WARNING: failed to obtain rlnImageSize, rlnImageDimensionality from non-existing file{imageFileName}. You should manually add both parameters to the optics group of the star file"
                    )

        """
        if "rlnMicrographPixelSize" in missingVars and "rlnMicrographPixelSize" not in optics:
            if "rlnMicrographOriginalPixelSize" in optics:
                optics.loc[:, "rlnMicrographPixelSize"] = optics.loc[:, "rlnMicrographOriginalPixelSize"]
                color_print(f"WARNING: 'rlnMicrographPixelSize' is copied from 'rlnMicrographOriginalPixelSize'. Please manually edit it if it is incorrect")
            elif "rlnImagePixelSize" in optics:
                optics.loc[:, "rlnMicrographPixelSize"] = optics.loc[:, "rlnImagePixelSize"]
                color_print(f"WARNING: 'rlnMicrographPixelSize' is copied from 'rlnImagePixelSize'. Please manually edit it if it is incorrect")
        """
        if (
            "rlnMicrographOriginalPixelSize" in missingVars
            and "rlnMicrographOriginalPixelSize" not in optics
        ):
            if "rlnMicrographPixelSize" in optics:
                optics.loc[:, "rlnMicrographOriginalPixelSize"] = optics.loc[
                    :, "rlnMicrographPixelSize"
                ]
                color_print(
                    f"WARNING: 'rlnMicrographOriginalPixelSize' is copied from 'rlnMicrographPixelSize'. Please manually edit it if it is incorrect"
                )
            elif "rlnImagePixelSize" in optics:
                optics.loc[:, "rlnMicrographOriginalPixelSize"] = optics.loc[
                    :, "rlnImagePixelSize"
                ]
                color_print(
                    f"WARNING: 'rlnMicrographOriginalPixelSize' is copied from 'rlnImagePixelSize'. Please manually edit it if it is incorrect"
                )
    except:
        pass
    missingVars = missing_required_opticsgroup_parameters(data)
    if missingVars:
        varval = " ".join([f"{v} <val>" for v in missingVars])
        color_print(
            f"WARNING: required OpticsGroup parameters {' '.join(missingVars)} are missing. Use \n\timages2star.py <input.star> <output.star> --setParm {varval}\nto add these parameters"
        )


# https://github.com/dzyla/Follow_Relion_gracefully/blob/main/follow_relion_gracefully_lib.py#L352
def star2dataframe(
    starFile,
    alternative_folders=[],
    ignore_bad_particle_path=0,
    ignore_bad_micrograph_path=1,
):
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
            data[tag.strip("_")] = np.array(value)

    data = dataframe_guess_data_type(data)
    nans = data.isnull().any(axis=1)
    if nans.sum() > 0:
        color_print(
            "WARNING: %s: %d/%d particle rows are corrupted and thus ignored"
            % (starFile, nans.sum(), len(data))
        )
        color_print(
            "    Corrupted particle indices:\n%s" % (nans.to_numpy().nonzero()[0])
        )
        if nans.sum() < 100:
            with pd.option_context("display.max_colwidth", -1):
                color_print("\n", data[nans == True])
        data = data[nans == False]
    data.attrs["optics"] = optics
    data.attrs["source_path"] = starFile
    data.attrs["convention"] = "relion"
    dataframe_normalize_filename(
        data, alternative_folders, ignore_bad_particle_path, ignore_bad_micrograph_path
    )
    return data


def dataframe2star(data, starFile, format="v3"):
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
                prefix, suffix = os.path.splitext(mgraphName)
                if suffix not in [".mrcs", ".mrc", ".tnf", ".spi", ".img", ".hed"]:
                    color_print(
                        f"WARNING: RELION does not support image format: {mgraphName}"
                    )

    if format in ["v3", "relion3"]:
        star_build_opticsgroup(data2)
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
        optics = data2.attrs["optics"]
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


def cs2dataframe(
    csFile,
    passthrough_files=[],
    alternative_folders=[],
    ignore_bad_particle_path=0,
    ignore_bad_micrograph_path=1,
    warn_missing_ctf=1,
):
    # read CryoSPARC v2/3/4 meta data
    cs = np.load(csFile)
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
            data = data.merge(extra_df, on="uid", how="left")
        data = data.loc[:, ~data.columns.duplicated()]
    if "blob/path" not in data and "micrograph_blob/path" not in data:
        color_print(
            f"ERROR: it appears that you have specified a CryoSPARC v2 passthrough file that does not have particle/micrograph path info. Available parameters are: {data.columns.values}"
        )
        sys.exit(-1)
    if (
        warn_missing_ctf
        and "ctf/accel_kv" not in data
        and csFile.find("templates_selected.cs") == -1
    ):
        color_print(
            "WARNING: CTF info not found. You should also provide the passthrough file that has CTF info"
        )
    if "ctf/type" in data:
        data = data.drop("ctf/type", axis=1)

    data = dataframe_guess_data_type(data)
    nans = data.isnull().any(axis=1)
    if nans.sum() > 0:
        color_print(
            "WARNING: %s: %d/%d particle rows are corrupted and thus ignored"
            % (csFile, nans.sum(), len(data))
        )
        color_print(
            "    Corrupted particle indices:\n%s" % (nans.to_numpy().nonzero()[0])
        )
        color_print(
            "    Sample of a corrupted particle info:\n%s"
            % (data.iloc[nans.to_numpy().nonzero()[0][0], :])
        )
        data = data[nans == False]
    data.attrs["source_path"] = csFile
    data.attrs["convention"] = "cryosparc"
    dataframe_normalize_filename(
        data, alternative_folders, ignore_bad_particle_path, ignore_bad_micrograph_path
    )
    return data


def dataframe2cs(data, csFile):
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
    dbFile,
    alternative_folders=[],
    ignore_bad_particle_path=0,
    ignore_bad_micrograph_path=1,
):
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
    data,
    alternative_folders=[],
    ignore_bad_particle_path=0,
    ignore_bad_micrograph_path=1,
):
    if ignore_bad_particle_path >= 2:
        return data

    def getRealFileName(
        filename, source_path, alternative_folders=[], ignore_bad_path=0
    ):
        if filename in getRealFileName.mapping:
            return getRealFileName.mapping[filename]

        basenames = []
        if not os.path.isabs(filename):
            basenames.append(filename)
        basenames.append(os.path.basename(filename))

        basenames += [f[:-4] + ".mrcs" for f in basenames if f.endswith(".mrc")]

        folders = [folder for folder in alternative_folders]
        if isinstance(source_path, str):
            folders += [os.path.dirname(os.path.realpath(source_path))]
        elif isinstance(source_path, (list, tuple, set)):
            folders += [os.path.dirname(os.path.realpath(sp)) for sp in source_path]

        relion_folder = get_relion_project_folder(filename)
        if relion_folder is not None:
            folders.append(relion_folder)

        filenameChoices = [filename]

        for basename in basenames:
            for folder in folders:
                filenameChoices += [os.path.join(folder, basename)]
                filenameChoices += [os.path.join(folder, "..", basename)]
                filenameChoices += [os.path.join(folder, "../..", basename)]

        match = None
        match_link = None
        for fci, fc in enumerate(filenameChoices):
            if os.path.isfile(fc):
                match = fc
                break
            if os.path.islink(fc):
                match_link = fc

        if match:
            ret = os.path.normpath(match)

            # when a new folder is found to have a matching file, all files of the same file type in the folder
            # will be pre-mapped to avoid multiple file checks later
            import glob

            suffix = os.path.splitext(filename)[-1]
            filename_dir = os.path.dirname(filename)
            allfiles = glob.glob(
                os.path.join(os.path.dirname(match), "*" + os.path.splitext(match)[-1])
            )
            for f in allfiles:
                tmp_basename = os.path.splitext(os.path.basename(f))[0] + suffix
                getRealFileName.mapping[os.path.join(filename_dir, tmp_basename)] = (
                    os.path.normpath(f)
                )
        else:
            if ignore_bad_path:
                ret = filename
            elif match_link:
                msg = (
                    "ERROR: image %s in file %s is found at %s that is a broken link to %s"
                    % (
                        filename,
                        source_path,
                        os.path.normpath(match_link),
                        os.readlink(match_link),
                    )
                )
                color_print(msg)
                sys.exit(-1)
            else:
                msg = f"ERROR: cannot find image {filename} in file {source_path} after trying these choices: {filenameChoices}"
                color_print(msg)
                sys.exit(-1)
        return ret

    getRealFileName.mapping = {}

    cache = {}

    def buildFileNameCache(
        filenames, source_path, alternative_folders=[], ignore_bad_path=0
    ):
        for fi, filename in enumerate(filenames):
            cache[filename] = getRealFileName(
                filename, source_path, alternative_folders, ignore_bad_path
            )

    attrs = []
    attrs_with_at = []
    for attr in "rlnImageName rlnMicrographName rlnMicrographMovieName".split():
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
def relion_astigmatism_to_eman(rlnDefocusU, rlnDefocusV, rlnDefocusAngle):
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


def eman_astigmatism_to_relion(defocus, dfdiff, dfang):
    if math.fmod(dfang + 360, 180) < 90:
        rlnDefocusU = defocus - dfdiff
        rlnDefocusV = defocus + dfdiff
    else:
        rlnDefocusU = defocus + dfdiff
        rlnDefocusV = defocus - dfdiff
    rlnDefocusAngle = math.fmod(dfang + 360, 90)
    return (rlnDefocusU * 1e4, rlnDefocusV * 1e4, rlnDefocusAngle)


def get_dataframe_convention(data):
    try:
        c = data.attrs["convention"]  # test if the convention is set
        assert c is not None and len(c) > 0
    except:  # let's guess the convention if it is not set yet
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


def dataframe_convert(data, target="relion"):

    data.attrs["convention"] = get_dataframe_convention(data)

    if data.attrs["convention"] == target:
        return data

    msg = f"ERROR: dataframe_convert(): unavailable conversion of convention from {data.attrs["convention"]} to {target}"
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


def dataframe_cryosparc_to_relion(data):
    data.attrs["convention"] = get_dataframe_convention(data)

    if data.attrs["convention"] == "relion":
        return data

    if data.attrs["convention"] != "cryosparc":
        msg = (
            "ERROR: dataframe_cryosparc_to_relion(): input dataframe is in %s instead of the required cryosparc convention"
            % (data.attrs["convention"])
        )
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
    if "alignments2D/shift" in data:
        shifts = pd.DataFrame(data["alignments2D/shift"].tolist()).round(2)
        ret["rlnOriginX"] = -shifts.iloc[:, 0]
        ret["rlnOriginY"] = -shifts.iloc[:, 1]
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
        e = r.as_euler("zyz", degrees=True)
        ret["rlnAngleRot"] = e[:, 2]
        ret["rlnAngleTilt"] = e[:, 1]
        ret["rlnAnglePsi"] = e[:, 0]

    if "alignments3D/shift" in data:
        shifts = pd.DataFrame(data["alignments3D/shift"].tolist()).round(2)
        ret["rlnOriginX"] = shifts.iloc[:, 0]
        ret["rlnOriginY"] = shifts.iloc[:, 1]

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
        ret.loc[:, "rlnAngleRotPrior"] = 0.0
        ret.loc[:, "rlnAngleTiltPrior"] = 90.0
        ret.loc[:, "rlnAnglePsiPrior"] = np.round(
            -np.rad2deg(data["filament/filament_pose"]), 1
        )
        ret.loc[:, "rlnAnglePsiFlipRatio"] = 0.5

    # TODO: convert high order aberrations: beam tilt, trifoil, tetrafoil, anisomag: 'ctf/tilt_A', 'ctf/trefoil_A', 'ctf/tetra_A', 'ctf/anisomag'
    # color_print(data.columns)
    #

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
        color_print(
            "ERROR: dataframe_cryosparc_to_relion(): none of the parameters %s is supported"
            % (list(data.columns))
        )
        sys.exit(-1)

    try:
        ret.attrs["source_path"] = data.attrs["source_path"]
    except:
        ret.attrs["source_path"] = None
    ret.attrs["convention"] = "relion"

    return ret


def mrc2mrcs(data):
    if "rlnImageName" in data:  # Relion star file
        tmp = data["rlnImageName"].str.split("@", expand=True)
        pid = tmp.iloc[:, 0]
        data.loc[:, "filename"] = tmp.iloc[:, -1]
    if "filename" not in data:
        return data

    names = set(data["filename"])
    mapping = {f: f for f in names}
    names = set([f for f in names if f.endswith(".mrc")])
    if len(names):
        for name in names:
            folder = os.path.dirname(name)
            if not os.access(folder, os.W_OK):
                folder = "./mrc2mrcs"
                if not os.path.exists(folder):
                    os.makedirs(folder)
            name2 = os.path.basename(name) + "s"
            name2 = os.path.join(folder, name2)
            mapping[name] = name2
            if not os.path.exists(name2):
                if os.path.islink(name2):
                    os.remove(name2)
                name_abs = os.path.abspath(os.path.normpath(name))
                os.symlink(name_abs, name2)
        data.loc[:, "filename"] = data["filename"].map(mapping)
    if "rlnImageName" in data:  # Relion star file
        data.loc[:, "rlnImageName"] = pid.astype(str) + "@" + data["filename"]
        data.drop(["filename"], axis=1, inplace=True)
    return data


#####################################################################################
def connect_cryosparc(cryosparc_server_info_file="$HOME/.cryosparc/cryosparc.toml"):
    def print_instructions():
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
        color_print(info)

    p = Path(os.path.expandvars(cryosparc_server_info_file))
    if not p.exists():
        print_instructions()
        sys.exit(-1)
    elif oct(p.stat().st_mode)[-3:] != "600":
        color_print(
            f"Please run command 'chmod 600 {cryosparc_server_info_file}' to keep your server info secure"
        )
        sys.exit(-1)

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
