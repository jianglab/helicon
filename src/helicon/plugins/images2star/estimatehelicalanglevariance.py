"""Handler and reusable helpers for helical-angle variance estimation."""

from __future__ import annotations

from pathlib import Path

import helicon
import numpy as np
from matplotlib.figure import Figure


option_name = "estimateHelicalAngleVariance"

_REQUIRED_COLUMNS = (
    "rlnImageName",
    "rlnHelicalTubeID",
    "rlnHelicalTrackLengthAngst",
    "rlnAngleTilt",
    "rlnAnglePsi",
    "rlnAngleRot",
)


def add_args(parser):
    parser.add_argument(
        "--estimateHelicalAngleVariance",
        metavar="<0|1>",
        type=int,
        help="estimate the variance of the tilt, psi, rot angles of segments in the same helical tube/filament",
        default=0,
    )


def _finite(values) -> np.ndarray:
    """Return the finite values in *values* as a one-dimensional float array."""
    values = np.asarray(values, dtype=np.float64).ravel()
    return values[np.isfinite(values)]


def _hist(ax, values, **kwargs) -> None:
    """Draw a histogram without failing on empty or non-finite inputs."""
    values = _finite(values)
    if values.size:
        ax.hist(values, **kwargs)
    else:
        ax.text(0.5, 0.5, "No finite data", ha="center", va="center", transform=ax.transAxes)


def _hexbin(fig, ax, x, y, *, label: str) -> None:
    """Draw a finite-data hexbin and its colorbar when at least one pair exists."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = min(x.size, y.size)
    valid = np.isfinite(x[:n]) & np.isfinite(y[:n])
    if valid.any():
        artist = ax.hexbin(
            x[:n][valid],
            y[:n][valid],
            bins="log",
            gridsize=50,
            cmap="jet",
        )
        fig.colorbar(artist, ax=ax, label=label)
    else:
        ax.text(0.5, 0.5, "No finite data", ha="center", va="center", transform=ax.transAxes)


def _build_figure(data, metrics: dict) -> Figure:
    """Build the existing 4x7 diagnostic dashboard without selecting a GUI backend."""
    figure = Figure(figsize=(33, 14), constrained_layout=True)
    axes = figure.subplots(nrows=4, ncols=7, squeeze=False)

    nsegments = metrics["nsegments"]
    tilt_means = metrics["tilt_means"]
    tilt_sigmas = metrics["tilt_sigmas"]
    psi_sigmas = metrics["psi_sigmas"]
    rot_sigmas = metrics["rot_sigmas"]
    rot_rates_medians = metrics["rot_rates_medians"]
    rot_rates_mads = metrics["rot_rates_mads"]
    loglikeli_means = metrics["loglikeli_means"]
    loglikeli_stds = metrics["loglikeli_stds"]
    maxvalueprob_means = metrics["maxvalueprob_means"]
    maxvalueprob_stds = metrics["maxvalueprob_stds"]

    for ai, angle in enumerate(("Tilt", "Psi", "Rot")):
        _hist(
            axes[0, ai],
            data[f"rlnAngle{angle}"],
            bins=50,
            edgecolor="white",
            linewidth=1,
        )
        axes[0, ai].set(xlabel=f"{angle} (°)", ylabel="# Filaments")

    axes[0, 6].scatter(tilt_means, tilt_sigmas)
    axes[0, 6].set(xlabel="Tilt (°)", ylabel="Tilt Sigma (°)")

    angles = [
        ("Tilt", tilt_sigmas),
        ("Psi", psi_sigmas),
        ("Rot", rot_sigmas),
    ]
    for ai, (angle_name, angle_sigma) in enumerate(angles):
        _hist(
            axes[1, ai],
            angle_sigma,
            bins=50,
            edgecolor="white",
            linewidth=1,
        )
        finite_sigma = np.sort(_finite(angle_sigma))
        axes[2, ai].plot(range(len(finite_sigma)), finite_sigma)
        _hexbin(
            figure,
            axes[3, ai],
            nsegments,
            angle_sigma,
            label="# Filaments",
        )
        axes[1, ai].set(xlabel=f"{angle_name} Sigma (°)", ylabel="# Filaments")
        axes[2, ai].set(
            xlabel="Rank (# Filaments)",
            ylabel=f"{angle_name} Sigma (°)",
        )
        axes[3, ai].set(
            xlabel="Filament Length (# Segments)",
            ylabel=f"{angle_name} Sigma (°)",
        )

    from itertools import combinations

    for pi, ((angle_1, sigma_1), (angle_2, sigma_2)) in enumerate(
        combinations(angles, 2)
    ):
        _hexbin(
            figure,
            axes[pi + 1, 6],
            sigma_1,
            sigma_2,
            label="# Filaments",
        )
        axes[pi + 1, 6].set(
            xlabel=f"{angle_1} Sigma (°)",
            ylabel=f"{angle_2} Sigma (°)",
        )

    has_ll = "rlnLogLikeliContribution" in data
    has_mvp = "rlnMaxValueProbDistribution" in data

    if has_ll:
        _hist(
            axes[0, 4],
            data["rlnLogLikeliContribution"],
            bins=50,
            edgecolor="white",
            linewidth=1,
        )
        axes[0, 4].set(xlabel="LogLikeliContribution", ylabel="# Particles")
    if has_mvp:
        _hist(
            axes[0, 5],
            data["rlnMaxValueProbDistribution"],
            bins=50,
            edgecolor="white",
            linewidth=1,
        )
        axes[0, 5].set(xlabel="MaxValueProbDistribution", ylabel="# Particles")

    if has_ll and has_mvp:
        _hexbin(
            figure,
            axes[1, 4],
            data["rlnLogLikeliContribution"],
            data["rlnMaxValueProbDistribution"],
            label="# Particles",
        )
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
        _hist(
            axes[2, 4],
            loglikeli_stds,
            bins=50,
            edgecolor="white",
            linewidth=1,
        )
        axes[2, 4].set(
            xlabel="Std LogLikeliContribution",
            ylabel="# Filaments",
        )
    if has_mvp and maxvalueprob_stds:
        _hist(
            axes[2, 5],
            maxvalueprob_stds,
            bins=50,
            edgecolor="white",
            linewidth=1,
        )
        axes[2, 5].set(
            xlabel="Std MaxValueProbDist",
            ylabel="# Filaments",
        )

    if has_ll and has_mvp and loglikeli_means:
        _hexbin(
            figure,
            axes[3, 4],
            loglikeli_means,
            maxvalueprob_means,
            label="# Filaments",
        )
        axes[3, 4].set(
            xlabel="Mean LogLikeli",
            ylabel="Mean MaxValueProb",
        )
        axes[3, 5].scatter(loglikeli_stds, maxvalueprob_stds, s=3, alpha=0.5)
        axes[3, 5].set(
            xlabel="Std LogLikeli",
            ylabel="Std MaxValueProb",
        )

    finite_rot_rates = _finite(rot_rates_medians)
    if finite_rot_rates.size:
        rot_min = float(np.min(finite_rot_rates))
        rot_p99 = float(np.percentile(finite_rot_rates, 99))
        if np.isclose(rot_min, rot_p99):
            half_width = max(abs(rot_min) * 0.05, 0.01)
            rot_bins = np.linspace(rot_min - half_width, rot_p99 + half_width, 51)
        else:
            bin_width = (rot_p99 - rot_min) / 50
            rot_bins = np.append(
                np.linspace(rot_min, rot_p99, 51),
                rot_p99 + bin_width,
            )
        clipped = np.clip(finite_rot_rates, rot_bins[0], rot_bins[-1] - 1e-12)
        axes[0, 3].hist(
            clipped,
            bins=rot_bins,
            edgecolor="white",
            linewidth=1,
        )
        axes[0, 3].axvline(
            np.mean(finite_rot_rates),
            color="red",
            ls="--",
            lw=1,
            label=f"Mean={np.mean(finite_rot_rates):.3g}",
        )
        axes[0, 3].axvline(
            np.median(finite_rot_rates),
            color="orange",
            ls=":",
            lw=1,
            label=f"Median={np.median(finite_rot_rates):.3g}",
        )
        hist_counts, _ = np.histogram(clipped, bins=rot_bins)
        peak_index = int(np.argmax(hist_counts))
        peak_bin_center = (rot_bins[peak_index] + rot_bins[peak_index + 1]) / 2
        axes[0, 3].axvline(
            peak_bin_center,
            color="green",
            ls="-.",
            lw=1,
            label=f"Peak={peak_bin_center:.3g}",
        )
        axes[0, 3].legend(fontsize=7)
    else:
        axes[0, 3].text(
            0.5,
            0.5,
            "No finite data",
            ha="center",
            va="center",
            transform=axes[0, 3].transAxes,
        )
    axes[0, 3].set(
        xlabel="Rot Change Rate (°/Å)",
        ylabel="# Filaments",
    )

    _hist(
        axes[1, 3],
        rot_rates_mads,
        bins=50,
        edgecolor="white",
        linewidth=1,
    )
    axes[1, 3].set(
        xlabel="MAD Rot Rate (°/Å)",
        ylabel="# Filaments",
    )

    sorted_rot_rates = np.sort(finite_rot_rates)
    axes[2, 3].plot(range(len(sorted_rot_rates)), sorted_rot_rates)
    axes[2, 3].set(
        xlabel="Rank (# Filaments)",
        ylabel="Rot Change Rate (°/Å)",
    )

    _hexbin(
        figure,
        axes[3, 3],
        nsegments,
        rot_rates_medians,
        label="# Filaments",
    )
    axes[3, 3].set(
        xlabel="Filament Length (# Segments)",
        ylabel="Rot Change Rate (°/Å)",
    )
    return figure


def estimate_helical_angle_variance(
    data,
    *,
    create_plot: bool = True,
    show_progress: bool = False,
):
    """Estimate per-filament angular variation and optionally build its dashboard.

    Parameters
    ----------
    data : pandas.DataFrame
        RELION particle metadata.
    create_plot : bool, optional
        Build and return the composite Matplotlib figure.
    show_progress : bool, optional
        Show a tqdm progress bar while processing filaments.

    Returns
    -------
    tuple[pandas.DataFrame, matplotlib.figure.Figure | None]
        The augmented particle table and optional backend-neutral figure.
    """
    missing = [column for column in _REQUIRED_COLUMNS if column not in data]
    if missing:
        raise ValueError(f"required parameters {' '.join(missing)} are not available")
    if len(data) == 0:
        raise ValueError("cannot estimate helical-angle variance for an empty dataset")

    from helicon import convert_dataframe_file_path
    from scipy.stats import circmean, circstd
    from tqdm import tqdm

    result = data.copy()
    result.loc[:, "rlnImageName_abs"] = (
        convert_dataframe_file_path(result, "rlnImageName", to="abs")
        .str.split("@", expand=True)
        .iloc[:, -1]
    )
    groups = result.groupby(["rlnImageName_abs", "rlnHelicalTubeID"], sort=False)

    metrics = {
        "nsegments": [],
        "tilt_means": [],
        "tilt_sigmas": [],
        "psi_sigmas": [],
        "rot_sigmas": [],
        "loglikeli_means": [],
        "loglikeli_stds": [],
        "maxvalueprob_means": [],
        "maxvalueprob_stds": [],
        "rot_rates_medians": [],
        "rot_rates_mads": [],
    }

    for _, particles in tqdm(
        groups,
        unit=" filaments",
        disable=not show_progress,
    ):
        metrics["nsegments"].append(len(particles))

        tilt = particles["rlnAngleTilt"].astype(np.float32).values
        metrics["tilt_means"].append(np.rad2deg(circmean(np.deg2rad(tilt))))
        tilt_sigma = np.rad2deg(circstd(np.deg2rad(tilt)))
        result.loc[particles.index, "rlnAngleTiltSigma"] = round(tilt_sigma, 2)
        metrics["tilt_sigmas"].append(tilt_sigma)

        psi = particles["rlnAnglePsi"].astype(np.float32).values
        psi = np.rad2deg(np.arccos(np.cos(2 * np.deg2rad(psi))))
        psi_sigma = np.rad2deg(circstd(np.deg2rad(psi))) / 2
        result.loc[particles.index, "rlnAnglePsiSigma"] = round(psi_sigma, 2)
        metrics["psi_sigmas"].append(psi_sigma)

        rot = particles["rlnAngleRot"].astype(np.float64).values
        positions = particles["rlnHelicalTrackLengthAngst"].astype(np.float64).values
        if len(rot) > 1:
            adjacent_distance = positions[1:] - positions[:-1]
            valid_adjacent = np.isfinite(adjacent_distance) & (adjacent_distance != 0)
            delta_rot = (
                helicon.angular_difference(rot[1:], rot[:-1])[valid_adjacent]
                / adjacent_distance[valid_adjacent]
            )
            delta_rot = _finite(delta_rot)
            rot_sigma = (
                float(np.rad2deg(circstd(np.deg2rad(delta_rot))))
                if delta_rot.size
                else 0.0
            )
        else:
            rot_sigma = 0.0
        result.loc[particles.index, "rlnAngleRotSigma"] = round(rot_sigma, 2)
        metrics["rot_sigmas"].append(rot_sigma)

        if len(rot) >= 2:
            rot_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(rot)))
            i, j = np.triu_indices(len(rot_unwrapped), k=1)
            distances = positions[j] - positions[i]
            valid_pairs = np.isfinite(distances) & (distances != 0)
            rates = np.abs(
                (rot_unwrapped[j][valid_pairs] - rot_unwrapped[i][valid_pairs])
                / distances[valid_pairs]
            )
            rates = _finite(rates)
        else:
            rates = np.asarray([], dtype=np.float64)
        if rates.size:
            rate_median = float(np.median(rates))
            rate_mad = float(np.median(np.abs(rates - rate_median)))
        else:
            rate_median = 0.0
            rate_mad = 0.0
        metrics["rot_rates_medians"].append(rate_median)
        metrics["rot_rates_mads"].append(rate_mad)

        if "rlnLogLikeliContribution" in result:
            values = _finite(particles["rlnLogLikeliContribution"])
            metrics["loglikeli_means"].append(
                float(np.mean(values)) if values.size else np.nan
            )
            metrics["loglikeli_stds"].append(
                float(np.std(values)) if values.size else np.nan
            )
        if "rlnMaxValueProbDistribution" in result:
            values = _finite(particles["rlnMaxValueProbDistribution"])
            metrics["maxvalueprob_means"].append(
                float(np.mean(values)) if values.size else np.nan
            )
            metrics["maxvalueprob_stds"].append(
                float(np.std(values)) if values.size else np.nan
            )

    result = result.drop(["rlnImageName_abs"], axis=1)
    result = result.reset_index(drop=True)
    figure = _build_figure(result, metrics) if create_plot else None
    return result, figure


@helicon.cache(
    expires_after=None,
    ignore=["output_star_file", "plot_file"],
)
def estimate_helical_angle_variance_from_star(
    input_star_file: str,
    output_star_file: str,
    plot_file: str,
):
    """Run the estimator for a STAR file, persist its artifacts, and cache the result."""
    data = helicon.images2dataframe(
        str(input_star_file),
        ignore_bad_micrograph_path=1,
        warn_missing_ctf=1,
        target_convention="relion",
    )
    result, figure = estimate_helical_angle_variance(
        data,
        create_plot=True,
        show_progress=False,
    )
    helicon.dataframe2file(result, str(output_star_file))
    figure.savefig(str(plot_file))
    return {
        "figure": figure,
        "output_star_file": str(output_star_file),
        "plot_file": str(plot_file),
    }


def handle(data, args, index_d, param):
    """Handle the estimateHelicalAngleVariance images2star option."""
    if not param:
        return data, index_d

    if args.verbose > 1:
        try:
            import matplotlib

            matplotlib.use("TkAgg")
        except Exception:
            pass
    data, figure = estimate_helical_angle_variance(
        data,
        create_plot=args.verbose > 1,
        show_progress=args.verbose >= 1,
    )
    index_d[option_name] += 1
    if figure is not None:
        plot_file = f"{Path(args.output_starFile).with_suffix('')}.tilt_psi_rot_sigma.pdf"
        figure.savefig(plot_file)
        # Keep the CLI's historical interactive-show behavior.  The reusable
        # function itself remains backend-neutral for Qt embedding and tests.
        try:
            import matplotlib.pyplot as plt
            from matplotlib._pylab_helpers import Gcf

            manager = plt._get_backend_mod().new_figure_manager_given_figure(
                id(figure),
                figure,
            )
            Gcf._set_new_active_manager(manager)
            plt.show()
        except Exception:
            pass
    return data, index_d
