"""Handler for the assignOpticGroupByBeamShiftXY option."""

from __future__ import annotations
import helicon
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from helicon.lib.exceptions import HeliconError, HeliconIOError
import logging

logger = logging.getLogger(__name__)


option_name = "assignOpticGroupByBeamShiftXY"


def add_args(parser):
    parser.add_argument(
        "--assignOpticGroupByBeamShiftXY",
        type=str,
        metavar="0|1|xml_folder=<path>:min_micrographs_per_group=<n>",
        help="assign images to optic groups by beam shift XY coordinates from EPU FoilHole XML files. Requires EPU_old FoilHole XML files. disabled by default",
        default="0",
    )


def handle(data, args, index_d, param):
    """Handle the assignOpticGroupByBeamShiftXY option.

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
        (data, index_d) after processing.
    """
    if param is not None and param != "0":
        try:
            optics_orig = data.attrs["optics"]
        except Exception:
            raise HeliconError("\tdata_optics block must be available")

        image_name = helicon.first_matched_attr(
            data,
            attrs="rlnMicrographMovieName rlnMicrographName rlnImageName".split(),
        )
        if image_name is None:
            raise HeliconError(
                "\trlnMicrographMovieName, rlnMicrographName or rlnImageName must be available"
            )

        required_cols = "rlnOpticsGroup".split()
        missing_cols = [c for c in required_cols if c not in data]
        if missing_cols:
            raise HeliconError(
                f"\trequired attrs {' '.join(missing_cols)} must be available"
            )

        _, param_dict = helicon.parse_param_str(str(param))
        xml_folder = param_dict.get("xml_folder", "")
        min_cluster_size = int(param_dict.get("min_micrographs_per_group", 4))

        micrographs = np.sort(data[image_name].unique())
        helicon.check_foilhole_xml_files(micrographs, xml_folder)

        @helicon.cache(
            cache_dir=str(helicon.cache_dir / "images2star"),
            expires_after=7,
            verbose=0,
        )
        def _cached_xml_path(micrograph_path: str, xml_folder: str) -> Path:
            return helicon.EPU_micrograph_path_2_movie_xml_path(
                micrograph_path=micrograph_path, xml_folder=xml_folder
            )

        xml_files_dict = {
            m: _cached_xml_path(micrograph_path=m, xml_folder=xml_folder)
            for m in tqdm(
                micrographs,
                total=len(micrographs),
                desc="Finding xml files",
                unit="micrograph",
            )
        }

        @helicon.cache(
            cache_dir=str(helicon.cache_dir / "images2star"),
            expires_after=7,
            verbose=0,
        )
        def _cached_beamshift(m: str) -> tuple[float, float]:
            return helicon.EPU_xml_2_beamshift(xml_file=xml_files_dict[m])

        micrographs_to_beamshifts = {
            m: _cached_beamshift(m)
            for m in tqdm(
                micrographs,
                total=len(micrographs),
                desc="Parsing xml files",
                unit="micrograph",
            )
        }

        @helicon.cache(
            cache_dir=str(helicon.cache_dir / "images2star"),
            ignore=["cpu", "verbose"],
            expires_after=7,
            verbose=0,
        )
        def _cached_cluster(
            beamshifts: list | np.ndarray,
            min_cluster_size: int,
            cpu: int,
            verbose: int,
        ) -> np.ndarray:
            return helicon.assign_beamshifts_to_cluster(
                beamshifts=beamshifts,
                range_n_clusters=range(2, 200),
                min_cluster_size=min_cluster_size,
                cpu=cpu,
                verbose=verbose,
            )

        beamshifts_list = list(micrographs_to_beamshifts.values())
        beamshift_clusters = _cached_cluster(
            beamshifts=beamshifts_list,
            min_cluster_size=min_cluster_size,
            cpu=args.cpu,
            verbose=args.verbose,
        )

        micrograph_to_cluster = {
            m: beamshift_clusters[mi]
            for mi, m in enumerate(micrographs_to_beamshifts.keys())
        }

        existing_groups = data["rlnOpticsGroup"].copy()
        new_subgroups = data[image_name].map(micrograph_to_cluster)
        data["rlnOpticsGroup"] = helicon.combine_groups(
            existing_groups.values, new_subgroups.values
        )

        optics = optics_orig.copy().iloc[0:0]
        pairs = pd.DataFrame(
            {"existing": existing_groups, "combined": data["rlnOpticsGroup"]}
        ).drop_duplicates()
        og_count = 0
        for _, pair in pairs.iterrows():
            parent = (
                optics_orig[
                    optics_orig["rlnOpticsGroup"].astype(str) == str(pair["existing"])
                ]
                .iloc[[0]]
                .copy()
            )
            combined_id = pair["combined"]
            parent["rlnOpticsGroup"] = combined_id
            parent["rlnOpticsGroupName"] = f"opticsGroup{combined_id}"
            optics = pd.concat([optics, parent], ignore_index=True)
            og_count += 1

        data.attrs["optics"] = optics

        if args.verbose > 1:
            n_existing = len(existing_groups.unique())
            n_new = len(optics)
            logger.info(f"\t{n_existing} optics groups -> {n_new} optic groups")

    return data, index_d
