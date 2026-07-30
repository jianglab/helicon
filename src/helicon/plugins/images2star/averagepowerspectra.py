"""Handler for the averagePowerSpectra images2star option."""

from __future__ import annotations

import logging
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd

import helicon
from helicon.lib.average_power_spectra import average_power_spectra_from_dataframe

logger = logging.getLogger(__name__)

option_name = "averagePowerSpectra"


def add_args(parser):
    parser.add_argument(
        "--averagePowerSpectra",
        metavar=(
            "groupby=<col1,col2,...>:"
            "fftX=<nx>:fftY=<ny>:"
            "batchSize=<n>:cpu=<n>:"
            "outdir=<path>"
        ),
        type=str,
        action="append",
        help=(
            "Compute average 2D power spectra (and phase differences) "
            "for particle groups.  Saves ``.mrcs`` files per group. "
            "Example: ``--averagePowerSpectra groupby=class:fftX=512:fftY=1024``"
        ),
        default=None,
    )


def handle(data, args, index_d, param):
    """Handle the averagePowerSpectra option.

    Parameters
    ----------
    data : pd.DataFrame
        The particle data DataFrame.
    args : argparse.Namespace
        CLI arguments.
    index_d : dict
        Option index tracker.
    param : object
        Parameter string (colon-delimited key=value pairs).

    Returns
    -------
    tuple[pd.DataFrame, dict]
        ``(data, index_d)`` after processing (DataFrame is not modified).
    """
    if not param:
        return data, index_d

    _, param_dict = helicon.parse_param_str(param)

    # Parse parameters with defaults matching the original CLI
    groupby_str = param_dict.get("groupby", "")
    groupby = (
        [g.strip() for g in groupby_str.split(",") if g.strip()]
        if groupby_str
        else None
    )
    fft_x = int(param_dict.get("fftX", 512))
    fft_y = int(param_dict.get("fftY", 1024))
    batch_size = int(param_dict.get("batchSize", 100))
    cpu = int(param_dict.get("cpu", 1))
    outdir = Path(param_dict.get("outdir", Path(args.output_starFile).stem))
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Computing average power spectra: fft=%dx%d, groupby=%s, batch=%d, cpu=%d",
        fft_x,
        fft_y,
        groupby or "auto",
        batch_size,
        cpu,
    )

    results, used_apix, (cutoff_res_y, cutoff_res_x) = (
        average_power_spectra_from_dataframe(
            data,
            apix=0.0,
            groupby=groupby,
            cutoff_res=[0.0, 0.0],
            min_particles=-1,
            force_phase_diff=False,
            batch_size=batch_size,
            cpu=cpu,
            diameter_mask=0,
            align=0,
            fft_x=fft_x,
            fft_y=fft_y,
        )
    )

    if results is None:
        logger.error("averagePowerSpectra: no results (check column names)")
        return data, index_d

    logger.info(
        "  pixel size = %.4f A, cutoff resolution = %.2f x %.2f A",
        used_apix,
        cutoff_res_y,
        cutoff_res_x,
    )

    for gi in sorted(results):
        r = results[gi]
        group_label = _sanitise_filename(str(r["group name"]))
        n = r["#images"]

        def _save_mrc(name_prefix: str, data: np.ndarray):
            path = outdir / f"{name_prefix}_{group_label}.mrcs"
            with mrcfile.new(path, overwrite=True) as mrc:
                mrc.set_data(data.astype(np.float32))
            return path

        ps_path = _save_mrc("ps_avg", r["ps_avg"])
        logger.info(
            "  Group %d (%s): %d images, PS saved to %s", gi, group_label, n, ps_path
        )

        if "pd_avg" in r:
            pd_path = _save_mrc("pd_avg", r["pd_avg"])
            logger.info("  Group %d (%s): PD saved to %s", gi, group_label, pd_path)

        if "image_avg" in r:
            _save_mrc("image_avg", r["image_avg"])

    index_d[option_name] += 1
    return data, index_d


def _sanitise_filename(name: str) -> str:
    """Replace characters that are problematic in filenames."""
    for ch in r'"*:<>?|\\/':
        name = name.replace(ch, "_")
    return name.strip().replace(" ", "_") or "unknown"
