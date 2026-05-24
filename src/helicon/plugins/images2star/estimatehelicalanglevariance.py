"""Handler for the estimateHelicalAngleVariance option."""

from __future__ import annotations
import helicon
import numpy as np
import os
from tqdm import tqdm


option_name = "estimateHelicalAngleVariance"


def add_args(parser):
    parser.add_argument(
        "--estimateHelicalAngleVariance",
        metavar="<0|1>",
        type=int,
        help="estimate the variance of the tilt, psi, rot angles of segments in the same helical tube/filament",
        default=0,
    )


def handle(data, args, index_d, param):
    """Handle the estimateHelicalAngleVariance option.

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
    if param:
        missing_attrs = [
            p
            for p in "rlnImageName rlnHelicalTubeID rlnHelicalTrackLengthAngst rlnAngleTilt rlnAnglePsi rlnAngleRot".split()
            if p not in data
        ]
        assert (
            missing_attrs == []
        ), f"\tERROR: requried parameters {' '.join(missing_attrs)} are not available"

        from helicon import convert_dataframe_file_path

        data.loc[:, "rlnImageName_abs"] = (
            convert_dataframe_file_path(data, "rlnImageName", to="abs")
            .str.split("@", expand=True)
            .iloc[:, -1]
        )
        groups = data.groupby(["rlnImageName_abs", "rlnHelicalTubeID"], sort=False)
        groups = [group_particles for group_name, group_particles in groups]
        nsegments = []
        tilt_means = []
        tilt_sigmas = []
        psi_sigmas = []
        rot_sigmas = []
        from scipy.stats import circmean, circstd
        from tqdm import tqdm

        for group_particles in tqdm(
            groups, unit=" filaments", disable=args.verbose < 1
        ):
            nsegments.append(len(group_particles))
            tilt = group_particles["rlnAngleTilt"].astype(np.float32).values
            tilt_means.append(np.rad2deg(circmean(np.deg2rad(tilt))))
            tilt_sigma = np.rad2deg(circstd(np.deg2rad(tilt)))
            data.loc[group_particles.index, "rlnAngleTiltSigma"] = round(tilt_sigma, 2)
            tilt_sigmas.append(tilt_sigma)
            psi = group_particles["rlnAnglePsi"].astype(np.float32).values
            psi = np.rad2deg(
                np.arccos(np.cos(2 * np.deg2rad(psi)))
            )  # to make the psi angles independent of polarity
            psi_sigma = np.rad2deg(circstd(np.deg2rad(psi))) / 2
            data.loc[group_particles.index, "rlnAnglePsiSigma"] = round(psi_sigma, 2)
            psi_sigmas.append(psi_sigma)
            rot = group_particles["rlnAngleRot"].astype(np.float32).values
            if len(rot) > 1:
                pos = (
                    group_particles["rlnHelicalTrackLengthAngst"]
                    .astype(np.float32)
                    .values
                )
                delta_rot = helicon.angular_difference(rot[1:], rot[:-1]) / (
                    pos[1:] - pos[:-1]
                )
                # since rot angle should change linearly along the helical track, we cannot directly calculate the circular std of rot angles
                # instead, we calculate the circular std of delta_rot which represents the change of rot angle per Angstrom along the helical track
                # delta_rot should be a constant if there is no error in assigning rot angles
                # the unit is degree/Angstrom
                rot_sigma = np.rad2deg(circstd(np.deg2rad(delta_rot)))
            else:
                rot_sigma = 0.0
            data.loc[group_particles.index, "rlnAngleRotSigma"] = round(rot_sigma, 2)
            rot_sigmas.append(rot_sigma)
        data = data.drop(["rlnImageName_abs"], inplace=False, axis=1)
        data = data.reset_index(drop=True)  # important to do this
        index_d[option_name] += 1
        if args.verbose > 1:
            if args.verbose > 1:
                import matplotlib

                matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(18, 14))
            for ai, angle in enumerate("Tilt Psi Rot".split()):
                axes[0, ai].hist(
                    data[f"rlnAngle{angle}"],
                    bins=50,
                    edgecolor="white",
                    linewidth=1,
                )
                axes[0, ai].set(xlabel=f"{angle} (°)", ylabel="# Filaments")
            axes[0, 3].scatter(tilt_means, tilt_sigmas)
            axes[0, 3].set(xlabel=f"Tilt (°)", ylabel="Tilt Sigma (°)")
            angles = [
                ("Tilt", tilt_sigmas),
                ("Psi", psi_sigmas),
                ("Rot", rot_sigmas),
            ]
            for ai, angle in enumerate(angles):
                angle_str, angle_sigma = angle
                axes[1, ai].hist(angle_sigma, bins=50, edgecolor="white", linewidth=1)
                axes[2, ai].plot(range(len(angle_sigma)), sorted(angle_sigma))
                hbin = axes[3, ai].hexbin(
                    nsegments, angle_sigma, bins="log", gridsize=50, cmap="jet"
                )
                fig.colorbar(hbin, ax=axes[3, ai], label="# Filaments")
                axes[1, ai].set(xlabel=f"{angle_str} Sigma (°)", ylabel="# Filaments")
                axes[2, ai].set(
                    xlabel="Rank (# Filaments)", ylabel=f"{angle_str} Sigma (°)"
                )
                axes[3, ai].set(
                    xlabel="Filament Length (# Segments)",
                    ylabel=f"{angle_str} Sigma (°)",
                )
            from itertools import combinations

            for pi, pair in enumerate(combinations(angles, 2)):
                (angle_str_1, angle_sigma_1), (angle_str_2, angle_sigma_2) = pair
                hbin = axes[pi + 1, 3].hexbin(
                    angle_sigma_1,
                    angle_sigma_2,
                    bins="log",
                    gridsize=50,
                    cmap="jet",
                )
                fig.colorbar(hbin, ax=axes[pi + 1, 3], label="# Filaments")
                axes[pi + 1, 3].set(
                    xlabel=f"{angle_str_1} Sigma (°)",
                    ylabel=f"{angle_str_2} Sigma (°)",
                )
            plt.savefig(
                f"{os.path.splitext(args.output_starFile)[0]}.tilt_psi_rot_sigma.pdf"
            )
            plt.tight_layout()
            plt.show()
    return data, index_d


import logging

logger = logging.getLogger(__name__)
