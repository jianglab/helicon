"""Handler for the assignExposureGroupByBeamShiftXY option."""

from __future__ import annotations
import argparse
import helicon
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from helicon.lib.exceptions import HeliconError
import logging

logger = logging.getLogger(__name__)


option_name = "assignExposureGroupByBeamShiftXY"


def add_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for the assignExposureGroupByBeamShiftXY option.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to attach arguments to.
    """
    parser.add_argument(
        "--assignExposureGroupByBeamShiftXY",
        type=str,
        metavar="0|1|xml_folder=<path>:min_micrographs_per_group=<n>",
        help="assign images to exposure groups by beam shift XY coordinates from EPU XML files. Requires EPU_old FoilHole XML files. disabled by default",
        default=None,
    )


def handle(
    data,
    args: argparse.Namespace,
    index_d: dict,
    param: object,
    output_title: str,
    output_slots: set,
    exp_group_id_name: str,
    micrograph_name: str,
    original_exp_group_ids: list,
):
    """Handle the assignExposureGroupByBeamShiftXY option.

    Parameters
    ----------
    data : Dataset
        The cryosparc Dataset.
    args : argparse.Namespace
        CLI arguments.
    index_d : dict
        Option index tracker.
    param : object
        The parameter value for this option.
    output_title : str
        Title for output filename construction.
    output_slots : set
        Output slot names.
    exp_group_id_name : str
        Name of the exposure group ID column.
    micrograph_name : str
        Name of the micrograph name column.
    original_exp_group_ids : list
        Original exposure group IDs.

    Returns
    -------
    tuple
        (data, output_title, output_slots, index_d) after processing.
    """
    if param is not None and param != "0":

        micrographs = np.sort(np.unique(data[micrograph_name]))

        _, param_dict = helicon.parse_param_str(param)
        xml_folder = param_dict.get("xml_folder", "")

        def has_xml(xml_folder: str, micrograph_path: str) -> bool:
            if xml_folder:
                xfp = Path(xml_folder)
                if xfp.exists() and xfp.is_dir() and xfp.glob("FoilHole_*.xml"):
                    return True
            if Path(micrograph_path).exists():
                if Path(micrograph_path).parent.glob("FoilHole_*.xml"):
                    return True
            return False

        if not has_xml(xml_folder=xml_folder, micrograph_path=micrographs[0]):
            raise HeliconError(
                f"Cannot find FoilHole XML files for {micrographs[0]}. Specify xml_folder=<path> in the parameter string."
            )

        min_cluster_size = int(param_dict.get("min_micrographs_per_group", 4))

        from tqdm import tqdm

        @helicon.cache(
            cache_dir=str(helicon.cache_dir / "cryosparc"),
            expires_after=7,
            verbose=0,
        )
        def EPU_micrograph_path_2_movie_xml_path(
            micrograph_path: str | Path, xml_folder: str
        ) -> Path:
            return helicon.EPU_micrograph_path_2_movie_xml_path(
                micrograph_path=micrograph_path, xml_folder=xml_folder
            )

        xml_files_dict = {
            m: EPU_micrograph_path_2_movie_xml_path(
                micrograph_path=input_project_folder / m,
                xml_folder=xml_folder,
            )
            for m in tqdm(
                micrographs,
                total=len(micrographs),
                desc="Finding xml files",
                unit="micrograph",
            )
        }

        @helicon.cache(
            cache_dir=str(helicon.cache_dir / "cryosparc"),
            expires_after=7,
            verbose=0,
        )
        def EPU_micrograph_path_2_beamshift(m: str) -> tuple[float, float]:
            xml_file = xml_files_dict[m]
            beamshift = helicon.EPU_xml_2_beamshift(xml_file=xml_file)
            return beamshift

        micrographs_to_beamshifts = {
            m: EPU_micrograph_path_2_beamshift(m)
            for m in tqdm(
                micrographs,
                total=len(micrographs),
                desc="Parsing xml files",
                unit="micrograph",
            )
        }

        @helicon.cache(
            cache_dir=str(helicon.cache_dir / "cryosparc"),
            ignore=["cpu", "verbose"],
            expires_after=7,
            verbose=0,
        )
        def assign_beamshifts_to_cluster(
            beamshifts: list | np.ndarray,
            range_n_clusters: range,
            min_cluster_size: int,
            cpu: int,
            verbose: int,
        ) -> np.ndarray:
            return helicon.assign_beamshifts_to_cluster(
                beamshifts=beamshifts,
                range_n_clusters=range_n_clusters,
                min_cluster_size=min_cluster_size,
                cpu=cpu,
                verbose=verbose,
            )

        beamshifts = np.array(list(micrographs_to_beamshifts.values()))
        beamshift_clusters = assign_beamshifts_to_cluster(
            beamshifts=beamshifts,
            range_n_clusters=range(2, 200),
            min_cluster_size=min_cluster_size,
            cpu=args.cpu,
            verbose=args.verbose,
        )
        assert len(beamshifts) == len(beamshift_clusters)
        micrograph_to_beamshift_clusters = {
            m: beamshift_clusters[mi]
            for mi, m in enumerate(micrographs_to_beamshifts.keys())
        }

        if "mscope_params/beam_shift" in data:
            data["mscope_params/beam_shift"] = np.array(
                [micrographs_to_beamshifts[row[micrograph_name]] for row in data.rows()]
            )

        exposure_groups = [
            micrograph_to_beamshift_clusters[row[micrograph_name]]
            for row in data.rows()
        ]
        data[exp_group_id_name] = helicon.combine_groups(
            data[exp_group_id_name], np.array(exposure_groups)
        )
        if len(exp_group_id_names_all) > 1:
            for attr in exp_group_id_names_all:
                if attr != exp_group_id_name:
                    data[attr] = data[exp_group_id_name]

        group_ids = np.sort(np.unique(data[exp_group_id_name]))
        for gi in group_ids:
            mask = np.where(data[exp_group_id_name] == gi)
            for (
                col
            ) in "ctf/cs_mm ctf/phase_shift_rad ctf/shift_A ctf/tilt_A ctf/trefoil_A ctf/tetra_A ctf/anisomag".split():
                if col in data:
                    data[col][mask] = np.median(data[col][mask])

        output_slots.add(exp_group_id_name.split("/")[0])
        output_title += (
            f" {len(source_group_ids)}->{len(group_ids)} beamshift XY groups"
        )

        if args.verbose > 1:
            logger.info(
                f"\t{len(source_group_ids)} -> {len(group_ids)} exposure groups stored in {' '.join(exp_group_id_names_all)}"
            )

        if args.verbose > 1:
            if args.csFile:
                output_file = (
                    f"{Path(args.csFile).stem}"
                    + (output_title if output_title else ".output")
                    + ".pdf"
                )
            else:
                output_file = (
                    f"{args.projectID}_{args.outputWorkspaceID}_{'-'.join(args.jobID)}"
                    + output_title
                    + ".pdf"
                )
            output_file = "-".join(output_file.split())
            output_file = output_file.replace(" ", "-")
            output_file = output_file.replace("->", "-")
            output_file = output_file.replace("/", "_")

            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 8))
            scatter = plt.scatter(
                beamshifts[:, 0],
                beamshifts[:, 1],
                c=beamshift_clusters,
                cmap="tab20",
                s=2,
            )
            plt.colorbar(scatter, label="Exposure Group")
            plt.xlabel("Beam Shift X")
            plt.ylabel("Beam Shift Y")
            plt.title("Exposure groups by beam shifts")
            plt.savefig(output_file)
            logger.info(
                f"\tPlot of exposure group assignments based on beam shifts is saved to {output_file}"
            )
            plt.show()
            plt.close()
    return data, output_title, output_slots, index_d
