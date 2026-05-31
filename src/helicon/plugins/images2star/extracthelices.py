"""Handler for the extractHelices option."""

from __future__ import annotations
import logging
import helicon
import numpy as np
import pandas as pd
from pathlib import Path
import mrcfile

logger = logging.getLogger(__name__)


option_name = "extractHelices"


def add_args(parser):
    parser.add_argument(
        "--extractHelices",
        type=str,
        metavar="width=300:outPath=./helicon.helices/:topLength=<10>[:topLengthFraction=<0.1>][lengthCutoffAngst=<1000>]",
        action="append",
        help="extract helical filaments from input files. disabled by default",
        default=[],
    )


def handle(data, args, index_d, param):
    """Handle the extractHelices option.

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
    if len(param):
        if param.find("=") != -1:
            # width=300:outPath=./helicon.helices/:topLength=<10>[:topLengthFraction=<0.1>][lengthCutoffAngst=<1000>]
            _, param_dict = helicon.parse_param_str(param)
        else:
            param_dict = {}

        if len(param.split(":")) > 3:
            logger.warning(
                "here might be multiple selection criteria. Will use the intersection of them."
            )

        width = param_dict.get("width", None)
        outPath = param_dict.get("outPath", "./helicon.helices/")
        topLength = param_dict.get("topLength", None)
        topLengthFraction = param_dict.get("topLengthFraction", None)
        lengthCutoffAngst = param_dict.get("lengthCutoffAngst", None)

        outPath = str(Path(outPath).resolve())

        import starfile

        get_apix = True
        coord_df = pd.DataFrame(
            columns=[
                "startX",
                "startY",
                "endX",
                "endY",
                "rlnMicrographName",
                "helixLength",
            ]
        )
        for _, mic_name, coordfile in data.itertuples():
            if get_apix:
                import mrcfile

                with mrcfile.open(mic_name, "r") as mic:
                    mic_data = np.array(mic.data, dtype=np.float32)
                    apix = mic.voxel_size["x"]
                    get_apix = False
            cf = starfile.read(coordfile)
            if cf is not None and not isinstance(cf, dict):
                cf = cf.reset_index(drop=True)
                cf = cf.loc[:, ["rlnCoordinateX", "rlnCoordinateY"]]
                starts = cf.iloc[::2].reset_index(drop=True)
                ends = cf.iloc[1::2].reset_index(drop=True)
                filaments = pd.DataFrame(
                    {
                        "startX": starts["rlnCoordinateX"],
                        "startY": starts["rlnCoordinateY"],
                        "endX": ends["rlnCoordinateX"],
                        "endY": ends["rlnCoordinateY"],
                        "rlnMicrographName": mic_name,
                    }
                )
                filaments["helixLength"] = np.sqrt(
                    (filaments["endX"] - filaments["startX"]) ** 2
                    + (filaments["endY"] - filaments["startY"]) ** 2
                )
                # print(filaments)
                coord_df = pd.concat([coord_df, filaments])

        coord_df["helixLength"] *= apix
        coord_df = coord_df.sort_values(by="helixLength", ascending=False)

        if topLengthFraction:
            coord_df = coord_df.iloc[0 : np.floor(len(coord_df) * topLengthFraction), :]

        if topLength:
            if len(coord_df) > topLength:
                coord_df = coord_df.iloc[0:topLength, :]

        if lengthCutoffAngst:
            coord_df = coord_df[coord_df["helixLength"] >= lengthCutoffAngst]

        cpu = args.cpu
        tasks = []
        helix_idx = 0
        coord_df.reset_index(drop=True)
        out_names = []
        for _, startX, startY, endX, endY, mic_name, _ in coord_df.itertuples():
            mic_prefix = ".".join(mic_name.split("/")[-1].split(".")[:-1])
            out_name = (
                outPath
                + "/helix_"
                + str(helix_idx)
                + "_width_"
                + str(width)
                + "px_"
                + mic_prefix
                + ".mrc"
            )
            out_names.append(out_name)
            tasks.append((startX, startY, endX, endY, mic_name, width, out_name))
            helix_idx += 1

        coord_df["rlnHelixImageName"] = out_names

        def process_one_task(startX, startY, endX, endY, mic_name, width, out_name):
            import mrcfile

            with mrcfile.open(mic_name, "r") as mic:
                mic_data = np.array(mic.data, dtype=np.float32)
                apix = mic.voxel_size["x"]
                helix_image = helicon.get_rotated_clip(
                    mic_data, startY, startX, endY, endX, width
                )
                with mrcfile.new(out_name, overwrite=True) as o_mrc:
                    o_mrc.set_data(np.array(helix_image, dtype=np.float32))
                    o_mrc.voxel_size = apix

        if not Path(outPath).is_dir():
            Path(outPath).mkdir(parents=True)

        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=cpu) as executor:
            future_tasks = [executor.submit(process_one_task, *task) for task in tasks]
            results = []
            for completed_task in as_completed(future_tasks):
                result = completed_task.result()
                results.append(result)

        data = coord_df
        # print(data)
    return data, index_d
