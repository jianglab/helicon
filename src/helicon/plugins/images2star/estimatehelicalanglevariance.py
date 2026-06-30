"""Handler for the estimateHelicalAngleVariance option."""

from __future__ import annotations
import helicon
import numpy as np
from pathlib import Path
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
        loglikeli_means = []
        loglikeli_stds = []
        maxvalueprob_means = []
        maxvalueprob_stds = []
        rot_rates_medians = []
        rot_rates_mads = []
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
            # Per-filament rot change rate: median of all pairwise (rot_j-rot_i)/(pos_j-pos_i)
            # Use np.unwrap to handle 360° wrapping — converts discontinuous angles
            # (e.g. 350° → 10°) into a continuous monotonic sequence (350° → 370°)
            if len(rot) >= 2:
                rot_unwrapped = np.rad2deg(
                    np.unwrap(np.deg2rad(rot.astype(np.float64)))
                )
                i, j = np.triu_indices(len(rot_unwrapped), k=1)
                delta_rot_rate = (rot_unwrapped[j] - rot_unwrapped[i]) / (
                    pos[j] - pos[i]
                )
                abs_delta_rot_rate = np.abs(delta_rot_rate)
                rot_rates_medians.append(float(np.median(abs_delta_rot_rate)))
                rot_rates_mads.append(
                    float(
                        np.median(
                            np.abs(abs_delta_rot_rate - np.median(abs_delta_rot_rate))
                        )
                    )
                )
            else:
                rot_rates_medians.append(0.0)
                rot_rates_mads.append(0.0)
            if "rlnLogLikeliContribution" in data:
                ll_vals = group_particles["rlnLogLikeliContribution"].values
                loglikeli_means.append(float(np.mean(ll_vals)))
                loglikeli_stds.append(float(np.std(ll_vals)))
            if "rlnMaxValueProbDistribution" in data:
                mvp_vals = group_particles["rlnMaxValueProbDistribution"].values
                maxvalueprob_means.append(float(np.mean(mvp_vals)))
                maxvalueprob_stds.append(float(np.std(mvp_vals)))
        data = data.drop(["rlnImageName_abs"], inplace=False, axis=1)
        data = data.reset_index(drop=True)  # important to do this
        index_d[option_name] += 1
        if args.verbose > 1:
            if args.verbose > 1:
                import matplotlib

                matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(
                nrows=4, ncols=7, figsize=(33, 14), constrained_layout=True
            )
            for ai, angle in enumerate("Tilt Psi Rot".split()):
                axes[0, ai].hist(
                    data[f"rlnAngle{angle}"],
                    bins=50,
                    edgecolor="white",
                    linewidth=1,
                )
                axes[0, ai].set(xlabel=f"{angle} (°)", ylabel="# Filaments")
            axes[0, 6].scatter(tilt_means, tilt_sigmas)
            axes[0, 6].set(xlabel=f"Tilt (°)", ylabel="Tilt Sigma (°)")
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
                hbin = axes[pi + 1, 6].hexbin(
                    angle_sigma_1,
                    angle_sigma_2,
                    bins="log",
                    gridsize=50,
                    cmap="jet",
                )
                fig.colorbar(hbin, ax=axes[pi + 1, 6], label="# Filaments")
                axes[pi + 1, 6].set(
                    xlabel=f"{angle_str_1} Sigma (°)",
                    ylabel=f"{angle_str_2} Sigma (°)",
                )
            has_ll = "rlnLogLikeliContribution" in data
            has_mvp = "rlnMaxValueProbDistribution" in data

            if has_ll:
                axes[0, 4].hist(
                    data["rlnLogLikeliContribution"],
                    bins=50,
                    edgecolor="white",
                    linewidth=1,
                )
                axes[0, 4].set(xlabel="LogLikeliContribution", ylabel="# Particles")
            if has_mvp:
                axes[0, 5].hist(
                    data["rlnMaxValueProbDistribution"],
                    bins=50,
                    edgecolor="white",
                    linewidth=1,
                )
                axes[0, 5].set(xlabel="MaxValueProbDistribution", ylabel="# Particles")

            if has_ll and has_mvp:
                hbin = axes[1, 4].hexbin(
                    data["rlnLogLikeliContribution"],
                    data["rlnMaxValueProbDistribution"],
                    bins="log",
                    gridsize=50,
                    cmap="jet",
                )
                fig.colorbar(hbin, ax=axes[1, 4], label="# Particles")
                axes[1, 4].set(
                    xlabel="LogLikeliContribution",
                    ylabel="MaxValueProbDistribution",
                )

            if has_ll and has_mvp and loglikeli_means:
                axes[1, 5].scatter(loglikeli_means, maxvalueprob_means, s=3, alpha=0.5)
                axes[1, 5].set(
                    xlabel="Mean LogLikeli",
                    ylabel="Mean MaxValueProb",
                )

            if has_ll and loglikeli_stds:
                axes[2, 4].hist(loglikeli_stds, bins=50, edgecolor="white", linewidth=1)
                axes[2, 4].set(
                    xlabel="Std LogLikeliContribution",
                    ylabel="# Filaments",
                )
            if has_mvp and maxvalueprob_stds:
                axes[2, 5].hist(
                    maxvalueprob_stds, bins=50, edgecolor="white", linewidth=1
                )
                axes[2, 5].set(
                    xlabel="Std MaxValueProbDist",
                    ylabel="# Filaments",
                )

            if has_ll and has_mvp and loglikeli_means:
                hbin = axes[3, 4].hexbin(
                    loglikeli_means,
                    maxvalueprob_means,
                    bins="log",
                    gridsize=50,
                    cmap="jet",
                )
                fig.colorbar(hbin, ax=axes[3, 4], label="# Filaments")
                axes[3, 4].set(
                    xlabel="Mean LogLikeli",
                    ylabel="Mean MaxValueProb",
                )
                axes[3, 5].scatter(loglikeli_stds, maxvalueprob_stds, s=3, alpha=0.5)
                axes[3, 5].set(
                    xlabel="Std LogLikeli",
                    ylabel="Std MaxValueProb",
                )

            rot_min = min(rot_rates_medians)
            rot_p99 = np.percentile(rot_rates_medians, 99)
            bin_width = (rot_p99 - rot_min) / 50
            rot_bins = np.linspace(rot_min, rot_p99, 51)  # 50 bins from min to p99
            rot_bins = np.append(rot_bins, rot_p99 + bin_width)  # 51st bin, same width
            clipped = np.clip(np.array(rot_rates_medians), None, rot_bins[-1] - 1e-12)
            axes[0, 3].hist(
                clipped,
                bins=rot_bins,
                edgecolor="white",
                linewidth=1,
            )
            axes[0, 3].axvline(
                np.mean(rot_rates_medians),
                color="red",
                ls="--",
                lw=1,
                label=f"Mean={np.mean(rot_rates_medians):.3g}",
            )
            axes[0, 3].axvline(
                np.median(rot_rates_medians),
                color="orange",
                ls=":",
                lw=1,
                label=f"Median={np.median(rot_rates_medians):.3g}",
            )
            hist_counts, _ = np.histogram(clipped, bins=rot_bins)
            peak_bin_center = (
                rot_bins[np.argmax(hist_counts)] + rot_bins[np.argmax(hist_counts) + 1]
            ) / 2
            axes[0, 3].axvline(
                peak_bin_center,
                color="green",
                ls="-.",
                lw=1,
                label=f"Peak={peak_bin_center:.3g}",
            )
            axes[0, 3].legend(fontsize=7)
            axes[0, 3].set(
                xlabel="Rot Change Rate (°/Å)",
                ylabel="# Filaments",
            )

            axes[1, 3].hist(rot_rates_mads, bins=50, edgecolor="white", linewidth=1)
            axes[1, 3].set(
                xlabel="MAD Rot Rate (°/Å)",
                ylabel="# Filaments",
            )

            axes[2, 3].plot(range(len(rot_rates_medians)), sorted(rot_rates_medians))
            axes[2, 3].set(
                xlabel="Rank (# Filaments)",
                ylabel="Rot Change Rate (°/Å)",
            )

            hbin = axes[3, 3].hexbin(
                nsegments,
                rot_rates_medians,
                bins="log",
                gridsize=50,
                cmap="jet",
            )
            fig.colorbar(hbin, ax=axes[3, 3], label="# Filaments")
            axes[3, 3].set(
                xlabel="Filament Length (# Segments)",
                ylabel="Rot Change Rate (°/Å)",
            )

            plt.savefig(
                f"{Path(args.output_starFile).with_suffix('')}.tilt_psi_rot_sigma.pdf"
            )
            plt.show()
    return data, index_d


import logging

logger = logging.getLogger(__name__)
