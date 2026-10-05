"""Handler for the saveStartEnd option."""

from __future__ import annotations
import logging
import re
from pathlib import Path, PurePosixPath
import pandas as pd
import helicon
from helicon.lib.exceptions import HeliconError, HeliconFileExistsError

logger = logging.getLogger(__name__)

option_name = "saveStartEnd"

# the files of a RELION ManualPick job
START_END_COORDINATE_LIST = "manualpick.star"
START_END_COORDINATE_SUFFIX = "_manualpick.star"


def add_args(parser):
    parser.add_argument(
        "--saveStartEnd",
        metavar="<dirname>",
        type=str,
        help=(
            "save the start-end coordinates of the helical filaments to this folder, "
            "in the format of the output of a RELION ManualPick job: one "
            "<micrograph>_manualpick.star file per micrograph and the coordinate list "
            "file <dirname>/manualpick.star. Use the coordinate list file as the input "
            "coordinates of a RELION Extract job (with 'Coordinates are start-end only?' "
            "= Yes) to re-extract the filaments with a new inter-box distance, e.g. after "
            "joining the filaments from multiple input files. The filaments are saved as "
            "they are at the position of this option among the other options"
        ),
        default="",
    )


def handle(data, args, index_d, param):
    """Handle the saveStartEnd option.

    Parameters
    ----------
    data : pd.DataFrame
        The particle data DataFrame.
    args : argparse.Namespace
        CLI arguments.
    index_d : dict
        Option index tracker.
    param : object
        The parameter value for this option.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        (data, index_d), with data unchanged.
    """
    if param:
        coord_list_file = Path(param) / START_END_COORDINATE_LIST
        if coord_list_file.exists() and getattr(args, "force", 0) != 1:
            raise HeliconFileExistsError(
                "the start-end coordinate file (%s) exists. Use --force=1 to overwrite it"
                % (coord_list_file)
            )
        save_filament_start_end(data, param, verbose=getattr(args, "verbose", 0))
    return data, index_d


def _start_end_file_subpath(micrograph: str) -> PurePosixPath:
    """Path of the start-end coordinate file of a micrograph, relative to its folder.

    Like RELION, the folders up to the job folder of the micrograph are
    dropped: ``MotionCorr/job002/Movies/mic1.mrc`` ->
    ``Movies/mic1_manualpick.star``.
    """
    p = PurePosixPath(micrograph.replace("\\", "/"))
    parts = p.parts[:-1]
    sub = PurePosixPath(p.name)
    for i, part in enumerate(parts):
        if re.fullmatch(r"job\d+", part):
            sub = PurePosixPath(*p.parts[i + 1 :])
            break
    return sub.with_name(f"{sub.stem}{START_END_COORDINATE_SUFFIX}")


def save_filament_start_end(
    data: pd.DataFrame, folder: str, verbose: int = 0
) -> pd.DataFrame:
    """Save the start-end coordinates of the helical filaments as a RELION picking job.

    Each filament is represented by the straight line through its particles
    (see :func:`helicon.filament_start_end`). As in the output of a RELION
    ManualPick job, the start and end points of the filaments on a
    micrograph are written as consecutive rows of
    ``<folder>/<micrograph>_manualpick.star``, and the coordinate list file
    ``<folder>/manualpick.star`` maps each micrograph to its coordinate file.
    Its ``data_general`` block sets ``rlnMicrographPickType`` to ``startend``,
    as RELION 5.1 and later require.
    The coordinate list file can be the input coordinates of a RELION Extract
    job with ``Coordinates are start-end only?`` = Yes, which re-extracts the
    filaments with a new inter-box distance. Filaments with a single particle
    have no start-end line and are skipped.

    Parameters
    ----------
    data : pd.DataFrame
        Particle data with ``rlnMicrographName``, ``rlnHelicalTubeID``,
        ``rlnCoordinateX`` and ``rlnCoordinateY`` columns.
    folder : str
        Output folder.
    verbose : int, optional
        Verbosity level.

    Returns
    -------
    pd.DataFrame
        The coordinate list: ``rlnMicrographName`` and
        ``rlnMicrographCoordinates`` of each micrograph with filaments.
    """
    import starfile

    required_attrs = (
        "rlnMicrographName rlnHelicalTubeID rlnCoordinateX rlnCoordinateY".split()
    )
    missing_attrs = [a for a in required_attrs if a not in data]
    if missing_attrs:
        raise HeliconError(
            "--saveStartEnd requires %s in the particles" % (" ".join(missing_attrs))
        )

    filament_attrs = ["rlnMicrographName", "rlnHelicalTubeID"]
    lines = helicon.filament_start_end(data, filament_attrs)
    single = lines["n"] < 2
    lines = lines[~single]
    if single.any():
        logger.info(
            "--saveStartEnd: skipped %d filaments with only one particle",
            single.sum(),
        )
    if len(lines) == 0:
        raise HeliconError("--saveStartEnd: no filament has two or more particles")

    inter_box_distance = helicon.estimate_inter_box_distance(data, filament_attrs)
    if inter_box_distance:
        curved = lines["maxOffset"] > inter_box_distance / 2
        if curved.any():
            logger.warning(
                "--saveStartEnd: %d/%d filaments are not straight: their particles are "
                "up to %.1f pixels away from the start-end line (inter-box distance: "
                "%.1f pixels). Their start-end lines are the best-fit straight lines",
                curved.sum(),
                len(lines),
                lines["maxOffset"].max(),
                inter_box_distance,
            )

    coord_list_file_given = Path(folder) / START_END_COORDINATE_LIST
    folder = Path(folder).resolve()
    coord_list_file = folder / START_END_COORDINATE_LIST
    relion_proj_folder = helicon.get_relion_project_folder(str(coord_list_file))

    def relion_path(p: Path) -> str:
        # RELION file names are relative to the project folder
        if relion_proj_folder:
            try:
                return p.relative_to(relion_proj_folder).as_posix()
            except ValueError:
                pass
        return p.as_posix()

    micrographs = lines["rlnMicrographName"].unique()
    subpaths = [_start_end_file_subpath(m) for m in micrographs]
    if len(set(subpaths)) < len(subpaths):
        # micrographs with the same name in different folders: keep all folders
        subpaths = []
        for m in micrographs:
            parts = PurePosixPath(m.replace("\\", "/")).parts
            p = PurePosixPath(
                *[s for s in parts if s not in ("/", "..") and not s.endswith(":")]
            )
            subpaths.append(p.with_name(f"{p.stem}{START_END_COORDINATE_SUFFIX}"))

    coord_list = []
    for (mgraph, filaments), subpath in zip(
        lines.groupby("rlnMicrographName", sort=False), subpaths
    ):
        xy = filaments[["startX", "startY", "endX", "endY"]].values.reshape(-1, 2)
        coords = pd.DataFrame(
            {
                "rlnCoordinateX": xy[:, 0].round(2),
                "rlnCoordinateY": xy[:, 1].round(2),
                # initialized as RELION AutoPick does
                "rlnClassNumber": 0,
                "rlnAnglePsi": -999.0,
                "rlnAutopickFigureOfMerit": 0.0,
            }
        )
        coord_file = folder / subpath
        coord_file.parent.mkdir(parents=True, exist_ok=True)
        starfile.write({"": coords}, coord_file, overwrite=True)
        mgraph_path = Path(mgraph)
        if relion_proj_folder and mgraph_path.is_absolute():
            mgraph = relion_path(mgraph_path)
        coord_list.append((mgraph, relion_path(coord_file)))
    coord_list = pd.DataFrame(
        coord_list, columns=["rlnMicrographName", "rlnMicrographCoordinates"]
    )
    # RELION 5.1+ requires the pick type of the coordinates
    starfile.write(
        {
            "general": {"rlnMicrographPickType": "startend"},
            "coordinate_files": coord_list,
        },
        coord_list_file,
        overwrite=True,
    )

    if verbose:
        logger.info(
            "Saved the start-end coordinates of %d filaments in %d micrographs to %s. "
            "Use it as the input coordinates of a RELION Extract job with "
            "'Coordinates are start-end only?' = Yes to re-extract the filaments",
            len(lines),
            len(coord_list),
            coord_list_file_given,
        )
    return coord_list
