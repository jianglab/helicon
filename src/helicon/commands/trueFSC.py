"""A command line tool to compute True FSC curve with optimal mask and phase randomization"""

import argparse
from asyncio import subprocess
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import ndimage
from scipy.interpolate import interp1d

import matplotlib

matplotlib.use("Agg")

import helicon
from helicon.lib.exceptions import HeliconError

logger = logging.getLogger(__name__)


def main(args):
    """Compute True FSC from two independently refined half-map reconstructions.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    """
    helicon.log_command_line()

    log_file = str(Path(args.plotFile).with_suffix(".log"))
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    if args.verbose <= 0:
        ch.setLevel(logging.CRITICAL)
    elif args.verbose == 1:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)

    logger.info(" ".join(sys.argv))
    logger.info("Started at %s", datetime.now())

    result = compute_truefsc(
        args.map1,
        args.map2,
        args.plotFile,
        apix=args.apix,
        cutoff_res=args.cutoffRes,
        one_mask=args.oneMask,
        mask_file=args.maskFile,
        mask_fraction_thresh=args.maskFractionThresh,
        mask_thresh=args.maskThresh,
        mask_mass=args.maskMass,
        mask_soft=args.maskSoft,
        refine_mask=args.refineMask,
    )

    if result["plot_file"]:
        if args.showPlot:
            _show_plot(result["plot_file"])

    return result


@helicon.cache(expires_after=None, ignore=["plot_file"])
def compute_truefsc(
    map1_file,
    map2_file,
    plot_file,
    apix=0,
    cutoff_res=0,
    one_mask=1,
    mask_file=None,
    mask_fraction_thresh=-1,
    mask_thresh=None,
    mask_mass=0,
    mask_soft=-1,
    refine_mask=1,
):
    """Compute True FSC curve with optimal mask and phase randomization.

    Parameters
    ----------
    map1_file : str
        Path to half-map 1.
    map2_file : str
        Path to half-map 2.
    plot_file : str
        Path to output plot file.
    apix : float, optional
        Pixel size in Angstrom. Read from map header if 0.
    cutoff_res : float, optional
        Cutoff resolution for phase randomization. Default: auto.
    one_mask : int, optional
        Use same mask for both maps (1) or separate (0).
    mask_file : list of str, optional
        User-provided mask file(s).
    mask_fraction_thresh : float, optional
        Mask threshold as fraction of max.
    mask_thresh : list of float, optional
        Mask threshold pixel value(s).
    mask_mass : float, optional
        Total mass in kDa.
    mask_soft : float, optional
        Mask slope width in Angstrom. Default: auto.
    refine_mask : int, optional
        Refine mask slope (0 or 1).

    Returns
    -------
    dict
        Result dictionary with keys: resolution, resolution_fit, plot_file, etc.
    """
    import mrcfile

    logger.info("Started at %s", datetime.now())

    map1_file = str(map1_file)
    map2_file = str(map2_file)
    plot_file = str(plot_file)

    if not Path(map1_file).exists():
        raise HeliconError(f"map1 not found: {map1_file}")
    if not Path(map2_file).exists():
        raise HeliconError(f"map2 not found: {map2_file}")

    with mrcfile.open(map1_file) as mrc:
        map1 = mrc.data.copy().astype(np.float64)
        apix1 = float(mrc.voxel_size.x)

    with mrcfile.open(map2_file) as mrc:
        map2 = mrc.data.copy().astype(np.float64)
        apix2 = float(mrc.voxel_size.x)

    if map1.shape != map2.shape:
        raise HeliconError(
            f"maps must have the same size: {map1.shape} != {map2.shape}"
        )

    if apix > 0:
        pix_size = apix
    elif apix1 == apix2:
        pix_size = apix1
    else:
        raise HeliconError(
            f"maps have different pixel sizes: {apix1} != {apix2}. Use --apix to specify."
        )

    logger.info("Sampling: %g Angstrom/pixel", pix_size)
    logger.info("Map size: %dx%dx%d", map1.shape[2], map1.shape[1], map1.shape[0])

    if np.allclose(np.mean(map1), np.mean(map2)) and np.allclose(
        np.std(map1), np.std(map2)
    ):
        logger.warning(
            "WARNING: the two maps appear to have identical statistics. "
            "Please provide two maps independently refined using half datasets."
        )

    fsc_prefix = str(Path(plot_file).with_suffix(""))

    # Precompute shell labels (reused by all FSC calls)
    n = map1.shape[0]
    _k2 = np.fft.fftfreq(n) ** 2
    _kr2 = np.fft.rfftfreq(n) ** 2
    _shell = np.round(
        np.sqrt(_k2[:, None, None] + _k2[None, :, None] + _kr2[None, None, :]) * n
    ).astype(np.int32)
    np.clip(_shell, 0, n // 2, out=_shell)
    _shell_flat = _shell.ravel()
    # Full symmetric-grid shell labels for calc_fsc_per_shell (used during
    # mask-slope optimisation). n is constant across calls, so precompute once
    # to avoid rebuilding the 3-cube meshgrid on every FSC evaluation.
    _shell_full = np.round(
        np.sqrt(_k2[:, None, None] + _k2[None, :, None] + _k2[None, None, :]) * n
    ).astype(np.int32)
    np.clip(_shell_full, 0, n // 2, out=_shell_full)
    _shell_flat_full = _shell_full.ravel()
    del _shell_full, _k2, _kr2

    # Compute FSC of original, unmasked maps
    logger.info("Calculating FSC of original maps")
    fsc_result = helicon.calc_fsc(map1, map2, pix_size, shell_flat=_shell_flat, n=n)
    saxis = fsc_result[:, 0]
    fsc_unmasked = fsc_result[:, 1]

    res_unmasked = _find_resolution(saxis, fsc_unmasked, 0.143)
    logger.info("Resolution at FSC=0.143 (unmasked): %.2f Angstrom", res_unmasked)

    # Determine cutoff resolution for phase randomization
    if cutoff_res > 2:
        cutoff_res_val = cutoff_res
    else:
        cutoff_res_val = _find_resolution(saxis, fsc_unmasked, 0.8)
        if cutoff_res_val > 100:
            saxis_fit, fsc_fit, _ = _fit_fsc_curve(saxis, fsc_unmasked)
            cutoff_res_val = _find_resolution(saxis_fit, fsc_fit, 0.8)
        if cutoff_res_val > 10:
            cutoff_res_val = round(cutoff_res_val)
        elif cutoff_res_val > 5:
            cutoff_res_val = round(cutoff_res_val * 2) / 2
        else:
            cutoff_res_val = round(cutoff_res_val * 4) / 4

    logger.info(
        "Cutoff resolution for phase randomization: %.2f Angstrom", cutoff_res_val
    )

    # Phase-randomize maps
    from helicon.lib.filters import randomize_phases_lowpass
    from scipy.fft import rfftn

    logger.info("Randomizing phases below %.2f Angstrom", cutoff_res_val)
    F1r = randomize_phases_lowpass(map1, pix_size, cutoff_res_val, return_fft=True)
    F2r = randomize_phases_lowpass(map2, pix_size, cutoff_res_val, return_fft=True)

    cutoff_res_i = int(map1.shape[0] * pix_size / cutoff_res_val)

    # Save unmasked FSC
    fscfile = fsc_prefix + ".unmasked.txt"
    np.savetxt(fscfile, np.column_stack([saxis[1:], fsc_unmasked[1:]]))

    # FSC of phase-randomized, unmasked maps (for reference)
    logger.info("Calculating FSC of phase randomized, unmasked maps")
    fsc_result_rand_unmasked = helicon.calc_fsc(
        None, None, pix_size, F1=F1r, F2=F2r, shell_flat=_shell_flat, n=n
    )
    fscfile = fsc_prefix + ".randomized-unmasked.txt"
    np.savetxt(
        fscfile,
        np.column_stack(
            [fsc_result_rand_unmasked[1:, 0], fsc_result_rand_unmasked[1:, 1]]
        ),
    )

    # Generate masks
    mask_file = mask_file or []
    user_mask = len(mask_file) > 0
    if user_mask:
        if len(mask_file) == 2:
            logger.info("Reading mask files: %s", " ".join(mask_file))
            mask1 = mrcfile.open(mask_file[0]).data.astype(np.float64)
            mask2 = mrcfile.open(mask_file[1]).data.astype(np.float64)
            if one_mask:
                mask_avg = (mask1 + mask2) / 2
                mask1, mask2 = mask_avg, mask_avg
        else:
            logger.info("Reading mask file: %s", mask_file[0])
            mask1 = mrcfile.open(mask_file[0]).data.astype(np.float64)
            mask2 = mask1
        logger.info("Using user-provided mask(s), skipping mask slope optimization")
    else:
        if one_mask:
            map_avg = (map1 + map2) / 2
            logger.info(
                "Map average: mask threshold automatically set using Otsu method"
            )
            logger.info("Map average: generating adaptive mask")
            mask1 = _generate_adaptive_mask(
                map_avg,
                pix_size,
                cutoff_res_val,
                mask_fraction_thresh,
                mask_thresh,
                mask_mass,
            )
            mask2 = mask1
        else:
            mask1 = _generate_adaptive_mask(
                map1,
                pix_size,
                cutoff_res_val,
                mask_fraction_thresh,
                mask_thresh,
                mask_mass,
            )
            mask2 = _generate_adaptive_mask(
                map2,
                pix_size,
                cutoff_res_val,
                mask_fraction_thresh,
                mask_thresh,
                mask_mass,
            )

    # Apply soft mask edge (skip when user provides mask)
    map1r = None
    map2r = None
    if not user_mask:
        mask_soft_val = mask_soft
        if mask_soft > 0:
            mask_soft_px = mask_soft_val / pix_size
            logger.info(
                "User provided mask slope width: %.1f Angstrom (%.1f pixels)",
                mask_soft_val,
                mask_soft_px,
            )
        elif refine_mask:
            from scipy.optimize import minimize_scalar
            from scipy.fft import irfftn

            logger.info("Searching for optimal mask slope width")
            map1r = irfftn(F1r, workers=-1)
            map2r = irfftn(F2r, workers=-1)

            def _fsc_score(x, map1, map2, map1r, map2r, mask_a, cutoff_i):
                mask_e = _soft_mask(mask_a, x)
                m1 = map1 * mask_e
                m2 = map2 * mask_e
                fsc_t = helicon.calc_fsc_per_shell(
                    m1, m2, pix_size, shell_flat=_shell_flat_full, n=n
                )

                m1r = map1r * mask_e
                m2r = map2r * mask_e
                fsc_n = helicon.calc_fsc_per_shell(
                    m1r, m2r, pix_size, shell_flat=_shell_flat_full, n=n
                )

                fsc_t_arr = fsc_t[cutoff_i:]
                fsc_n_arr = fsc_n[cutoff_i:]

                fsc_true = (fsc_t_arr - fsc_n_arr) / (1 - fsc_n_arr)
                fsc_true[np.isnan(fsc_true)] = 1.0

                score = (
                    np.mean(1 - np.abs(fsc_true))
                    + np.mean(np.abs(fsc_n_arr))
                    + np.mean(np.abs(fsc_t_arr - fsc_true))
                    + np.mean(1 - np.abs(fsc_true - fsc_n_arr))
                )

                if logger.isEnabledFor(logging.DEBUG):
                    nshells = len(fsc_t)
                    saxis_shells = np.arange(nshells) / (map1.shape[0] * pix_size)
                    res = _find_resolution(saxis_shells[cutoff_i:], fsc_true, 0.143)
                    logger.debug(
                        "\tMask width: %.2f Angstrom (%.2f pixels)\t->\t%.2f Angstrom at FSC=0.143\tfval=%g",
                        x * pix_size,
                        x,
                        res,
                        score,
                    )

                return score

            res_opt = minimize_scalar(
                _fsc_score,
                bounds=(0, map1.shape[0] / 3),
                method="bounded",
                args=(map1, map2, map1r, map2r, mask1, cutoff_res_i + 2),
                options={"xatol": 2},
            )
            mask_soft_px = res_opt.x
            logger.info(
                "Optimal mask slope width: %.1f Angstrom (%.1f pixels)",
                mask_soft_px * pix_size,
                mask_soft_px,
            )
        else:
            mask_soft_px = 3 * res_unmasked / pix_size
            logger.info(
                "Default mask slope width: %.1f Angstrom (%.1f pixels = 3 * %g / %g)",
                mask_soft_px * pix_size,
                mask_soft_px,
                res_unmasked,
                pix_size,
            )
        mask1 = _soft_mask(mask1, mask_soft_px)
        mask2 = _soft_mask(mask2, mask_soft_px)

    # Save masks (skip when user provides them)
    maskdir = str(Path(plot_file).parent) or "."
    basename1 = Path(map1_file).stem
    basename2 = Path(map2_file).stem

    mask1_file = None
    mask2_file = None
    if user_mask:
        # Reuse the user-provided mask file(s) for display, as if they were the
        # auto-generated masks. The mask panel shows the exact mask applied.
        mask1_file = str(mask_file[0])
        if len(mask_file) == 2:
            mask2_file = str(mask_file[1])
        else:
            # A single mask applies to both half-maps.
            mask2_file = mask1_file
    elif one_mask:
        mask1_file = str(
            Path(maskdir) / (basename1 + "_" + basename2 + ".common_mask.mrc")
        )
        logger.info("Saving final mask: %s", mask1_file)
        with mrcfile.new(mask1_file, overwrite=True) as mrc:
            mrc.set_data(mask1.astype(np.float32))
    else:
        mask1_file = str(Path(maskdir) / (basename1 + ".mask.mrc"))
        logger.info("Saving final mask: %s", mask1_file)
        with mrcfile.new(mask1_file, overwrite=True) as mrc:
            mrc.set_data(mask1.astype(np.float32))

        mask2_file = str(Path(maskdir) / (basename2 + ".mask.mrc"))
        logger.info("Saving final mask: %s", mask2_file)
        with mrcfile.new(mask2_file, overwrite=True) as mrc:
            mrc.set_data(mask2.astype(np.float32))

    # Apply masks and compute FSCs
    m1 = map1 * mask1
    m2 = map2 * mask2

    # Reconstruct real-space phase-randomized maps (already done during refinement if applicable)
    if map1r is None:
        from scipy.fft import irfftn as _irfftn

        map1r = _irfftn(F1r, workers=-1)
        map2r = _irfftn(F2r, workers=-1)
    m1r = map1r * mask1
    m2r = map2r * mask2

    # Save masked maps
    masked1_file = str(Path(maskdir) / (basename1 + ".masked.mrc"))
    logger.info("Saving final masked map: %s", masked1_file)
    with mrcfile.new(masked1_file, overwrite=True) as mrc:
        mrc.set_data(m1.astype(np.float32))

    masked2_file = str(Path(maskdir) / (basename2 + ".masked.mrc"))
    logger.info("Saving final masked map: %s", masked2_file)
    with mrcfile.new(masked2_file, overwrite=True) as mrc:
        mrc.set_data(m2.astype(np.float32))

    logger.info("Calculating FSC of masked maps (gold FSC)")
    fsc_result_masked = helicon.calc_fsc(m1, m2, pix_size, shell_flat=_shell_flat, n=n)
    saxis_m = fsc_result_masked[:, 0]
    fsc_t = fsc_result_masked[:, 1]

    res_masked = _find_resolution(saxis_m, fsc_t, 0.143)
    logger.info("Resolution at FSC=0.143 (masked): %.2f Angstrom", res_masked)

    fscfile = fsc_prefix + ".masked.txt"
    np.savetxt(fscfile, np.column_stack([saxis_m[1:], fsc_t[1:]]))

    logger.info("Calculating FSC of phase-randomized masked maps (noise FSC)")
    fsc_result_noise = helicon.calc_fsc(m1r, m2r, pix_size, shell_flat=_shell_flat, n=n)
    saxis_n = fsc_result_noise[:, 0]
    fsc_n = fsc_result_noise[:, 1]

    fscfile = fsc_prefix + ".randomized-masked.txt"
    np.savetxt(fscfile, np.column_stack([saxis_n[1:], fsc_n[1:]]))

    # Compute True FSC
    logger.info("Calculating True FSC")
    fsc_true = np.copy(fsc_t)
    fsc_true[cutoff_res_i + 1 :] = (
        fsc_t[cutoff_res_i + 1 :] - fsc_n[cutoff_res_i + 1 :]
    ) / (1 - fsc_n[cutoff_res_i + 1 :])
    fsc_true[np.isnan(fsc_true)] = 1.0

    fscfile = fsc_prefix + ".true.txt"
    np.savetxt(fscfile, np.column_stack([saxis_m[1:], fsc_true[1:]]))

    # Fit and find resolution
    saxis_fit, fsc_true_fit, res_true_fit = _fit_fsc_curve(saxis_m, fsc_true)
    res_true = _find_resolution(saxis_m, fsc_true, 0.143)
    res_true_fit = _find_resolution(saxis_fit, fsc_true_fit, 0.143)

    logger.info(
        "Resolution at FSC=0.143 (true): %.2f Angstrom (fit: %.2f Angstrom)",
        res_true,
        res_true_fit,
    )

    # Save fitted curve
    fscfile_fit = fsc_prefix + ".true.fit.txt"
    np.savetxt(fscfile_fit, np.column_stack([saxis_fit, fsc_true_fit]))

    # Plot
    fsc_curves = [
        (saxis[1:], fsc_unmasked[1:], f"unmasked ({res_unmasked:.2f} A)"),
        (saxis_m[1:], fsc_t[1:], f"masked ({res_masked:.2f} A)"),
        (saxis_n[1:], fsc_n[1:], "noise-substituted"),
        (saxis_m[1:], fsc_true[1:], f"corrected ({res_true:.2f} A)"),
    ]

    logger.info("Saving FSC curves: %s", plot_file)
    volumes = [
        (
            "Map 1",
            [
                ("Unmasked map 1", map1),
                ("Mask 1", mask1),
                ("Masked map 1", m1),
            ],
        ),
        (
            "Map 2",
            [
                ("Unmasked map 2", map2),
                ("Mask 2", mask2),
                ("Masked map 2", m2),
            ],
        ),
    ]
    plot_fsc(fsc_curves, plot_file, volumes=volumes)

    return {
        "resolution": res_true,
        "resolution_fit": res_true_fit,
        "resolution_unmasked": res_unmasked,
        "resolution_masked": res_masked,
        "plot_file": plot_file,
        "fsc_curves": fsc_curves,
        "volumes": {
            "mask1_file": mask1_file,
            "mask2_file": mask2_file,
            "masked_map1_file": masked1_file,
            "masked_map2_file": masked2_file,
        },
    }


def _show_plot(plot_file):
    """Show plot using system viewer."""
    if plot_file.lower().endswith(".pdf"):
        viewers = ["evince", "okular", "zathura", "xpdf", "open"]
    else:
        viewers = ["xdg-open", "open"]

    for viewer in viewers:
        try:
            import subprocess

            subprocess.Popen([viewer, plot_file])
            logger.info("Opening %s...", plot_file)
            return
        except FileNotFoundError:
            continue
    logger.warning(
        "Could not find a viewer to open the plot. Please open %s manually.",
        plot_file,
    )


def _find_resolution(saxis, fsc, threshold):
    """Find resolution where FSC curve crosses a threshold value.

    Parameters
    ----------
    saxis : np.ndarray
        Spatial frequency values (1/Angstrom).
    fsc : np.ndarray
        FSC values corresponding to saxis.
    threshold : float
        FSC threshold value (e.g. 0.143).

    Returns
    -------
    float
        Resolution in Angstroms. Returns 999 if the threshold is not crossed.
    """
    fsc_arr = np.asarray(fsc)
    saxis_arr = np.asarray(saxis)
    idx = np.where(fsc_arr < threshold)[0]
    if len(idx) == 0:
        return 999.0
    i = idx[0]
    if i == 0:
        return 1.0 / saxis_arr[0] if saxis_arr[0] > 0 else 999.0
    x0 = saxis_arr[i - 1]
    x1 = saxis_arr[i]
    y0 = fsc_arr[i - 1]
    y1 = fsc_arr[i]
    if y0 == y1:
        cross_x = x1
    else:
        cross_x = x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
    if cross_x <= 0:
        return 999.0
    return 1.0 / cross_x


def _fit_fsc_curve(saxis, fsc, order=4):
    """Fit a polynomial to the FSC curve.

    Parameters
    ----------
    saxis : np.ndarray
        Spatial frequency values (1/Angstrom).
    fsc : np.ndarray
        FSC values.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, float)
        Fitted spatial frequencies, FSC values on a finer grid, and
        resolution at FSC=0.143 from the fit.
    """
    from scipy.optimize import minimize

    mask = np.isfinite(fsc) & (fsc >= -0.1) & (fsc <= 1.1)
    s_fit = saxis[mask]
    f_fit = fsc[mask]
    if len(s_fit) < 3:
        return saxis, fsc, _find_resolution(saxis, fsc, 0.143)

    def fermi(mu, T, x):
        return 1.0 / (np.exp((x - mu) / T) + 1.0)

    def butterworth(omega, n, x):
        return 1.0 / (1.0 + (x / omega) ** n)

    def fit_score_fermi(params):
        mu, T = params
        if T <= 0:
            return 1e10
        a = 1.0 / fermi(mu, T, 0.0)
        pred = a * fermi(mu, T, s_fit)
        return np.mean(np.abs(f_fit - pred))

    def fit_score_butterworth(params):
        omega, n = params
        if omega <= 0 or n <= 0:
            return 1e10
        pred = butterworth(omega, n, s_fit)
        return np.mean(np.abs(f_fit - pred))

    best_error = np.inf
    best_fitted = f_fit.copy()
    best_res = _find_resolution(s_fit, f_fit, 0.143)

    # Try Fermi fit
    mu0 = s_fit[len(s_fit) // 2]
    res_fermi = minimize(
        fit_score_fermi,
        x0=[mu0, 0.01],
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-6},
    )
    if res_fermi.fun < best_error:
        best_error = res_fermi.fun
        mu, T = res_fermi.x
        a = 1.0 / fermi(mu, T, 0.0)
        s_fine = np.linspace(saxis[1], saxis[-1], 500)
        f_fine = a * fermi(mu, T, s_fine)
        f_fine = np.clip(f_fine, -1, 1)
        best_fitted_fine = f_fine
        best_s_fine = s_fine
        # Find resolution from fitted curve
        idx = np.where(f_fine < 0.143)[0]
        if len(idx) > 0:
            i = idx[0]
            if i > 0:
                x0, x1 = s_fine[i - 1], s_fine[i]
                y0, y1 = f_fine[i - 1], f_fine[i]
                cross = x0 + (0.143 - y0) * (x1 - x0) / (y1 - y0)
                best_res = 1.0 / cross if cross > 0 else 999.0

    # Try Butterworth fit
    omega0 = s_fit[len(s_fit) // 2]
    res_bw = minimize(
        fit_score_butterworth,
        x0=[omega0, 2.0],
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-6},
    )
    if res_bw.fun < best_error:
        best_error = res_bw.fun
        omega, n = res_bw.x
        s_fine = np.linspace(saxis[1], saxis[-1], 500)
        f_fine = butterworth(omega, n, s_fine)
        f_fine = np.clip(f_fine, -1, 1)
        best_fitted_fine = f_fine
        best_s_fine = s_fine
        idx = np.where(f_fine < 0.143)[0]
        if len(idx) > 0:
            i = idx[0]
            if i > 0:
                x0, x1 = s_fine[i - 1], s_fine[i]
                y0, y1 = f_fine[i - 1], f_fine[i]
                cross = x0 + (0.143 - y0) * (x1 - x0) / (y1 - y0)
                best_res = 1.0 / cross if cross > 0 else 999.0

    return best_s_fine, best_fitted_fine, best_res


def _fourier_lowpass(volume, cutoff_res, apix):
    """Apply Fourier lowpass matching EMAN2 filter.fourier with targetres.

    Parameters
    ----------
    volume : np.ndarray
        Input 3D map.
    cutoff_res : float
        Resolution cutoff in Angstroms.
    apix : float
        Pixel size in Angstroms.

    Returns
    -------
    np.ndarray
        Low-pass filtered map.
    """
    from scipy.fft import rfftn, irfftn

    nz, ny, nx = volume.shape
    F = rfftn(volume, workers=-1)

    bfactor = 4 * np.log(2) * cutoff_res**2
    ds = 1.0 / (apix * ny)

    k2 = np.fft.fftfreq(nx) ** 2
    kr2 = np.fft.rfftfreq(nx) ** 2
    # Vectorized filter for rfftn output
    f = np.exp(
        -bfactor
        * ds**2
        * (k2[:, None, None] + k2[None, :, None] + kr2[None, None, :])
        / 4.0
    )

    F_filtered = F * f
    return irfftn(F_filtered, workers=-1)


def _otsu_threshold_eman(volume, n_bins=256, ignore_zero=True):
    """Compute Otsu threshold matching EMAN2's implementation.

    Parameters
    ----------
    volume : np.ndarray
        Input 3D array.
    n_bins : int, optional
        Number of histogram bins. Defaults to 256.
    ignore_zero : bool, optional
        If True, exclude zero-valued voxels. Defaults to True.

    Returns
    -------
    float
        Otsu threshold value.
    """
    hmin = float(np.min(volume))
    hmax = float(np.max(volume))
    bin_width = (hmax - hmin) / n_bins

    flat = volume.ravel()
    if ignore_zero:
        flat = flat[flat != 0]
    if len(flat) == 0:
        return hmin

    hist, _ = np.histogram(flat, bins=n_bins, range=(hmin, hmax))
    hist = hist.astype(np.float64)

    total = hist.sum()
    if total == 0:
        return hmin

    sum_all = np.dot(np.arange(n_bins, dtype=np.float64), hist)
    cumsum = np.cumsum(hist)
    cumsum_val = np.cumsum(np.arange(n_bins, dtype=np.float64) * hist)

    wB = cumsum
    wF = total - wB
    mB = np.zeros(n_bins)
    mF = np.zeros(n_bins)
    valid = (wB > 0) & (wF > 0)
    mB[valid] = cumsum_val[valid] / wB[valid]
    mF[valid] = (sum_all - cumsum_val[valid]) / wF[valid]

    between = wB * wF * (mB - mF) ** 2
    max_bi = np.argmax(between[1:]) + 1  # skip empty first bin

    return hmin + (max_bi + 1) * bin_width


def _generate_adaptive_mask(
    volume, apix, cutoff_res, mask_fraction_thresh=-1, mask_thresh=None, mask_mass=0
):
    """Generate an adaptive mask using seed-and-grow (matching EMAN2 mask.auto3d).

    Parameters
    ----------
    volume : np.ndarray
        Input 3D map.
    apix : float
        Pixel size in Angstroms.
    cutoff_res : float
        Low-pass cutoff resolution in Angstroms.
    mask_fraction_thresh : float, optional
        Mask threshold as fraction of max pixel value. Default: -1 (auto).
    mask_thresh : list of float, optional
        Mask threshold pixel value(s).
    mask_mass : float, optional
        Total mass of structure in kDa.

    Returns
    -------
    np.ndarray
        Binary mask.
    """
    from scipy.ndimage import gaussian_filter, label

    # Low-pass filter the map (matching EMAN2 filter.fourier with targetres)
    if cutoff_res > 2 * apix:
        sigma = cutoff_res / (3.81 * apix)
        volume_lp = gaussian_filter(volume, sigma=sigma)
    else:
        volume_lp = volume.copy()

    # Determine threshold
    if mask_fraction_thresh > 0:
        thresh = mask_fraction_thresh * np.max(volume_lp)
    elif mask_thresh and mask_thresh[0] > 0:
        thresh = mask_thresh[0]
    elif mask_mass > 0:
        mass_kda = mask_mass
        vol_voxels = mass_kda * 1e3 / (0.81 * apix**3)
        sorted_vals = np.sort(volume_lp.ravel())[::-1]
        thresh = sorted_vals[min(int(vol_voxels), len(sorted_vals) - 1)]
    else:
        thresh = _otsu_threshold_eman(volume_lp)

    logger.info("Adaptive mask: threshold=%.7g", thresh)

    # Seed with brightest voxels from the LOW-PASS filtered map
    nmaxseed = 1000
    flat_idx = np.argpartition(volume_lp.ravel(), -nmaxseed)[-nmaxseed:]
    above_thresh = volume_lp > thresh
    seeds = np.zeros(volume_lp.shape, dtype=bool)
    seeds.ravel()[flat_idx] = True
    seeds &= above_thresh

    # Find connected components on the LP-filtered map
    struct = np.ones((3, 3, 3), dtype=bool)
    labeled, n_components = label(above_thresh, structure=struct)

    # Keep components containing seeds
    seed_labels = labeled.ravel()[flat_idx]
    seed_labels = seed_labels[seed_labels > 0]
    component_ids = np.unique(seed_labels)

    mask = np.isin(labeled, component_ids)

    if not np.any(mask):
        mask = above_thresh.copy()

    mask = mask.astype(np.float64)
    mass = np.count_nonzero(mask) * apix**3 * 0.81e-3
    logger.info("Adaptive mask: mass of masked region: %d kDa", int(round(mass)))
    return mask


def _soft_mask(mask, soft_width):
    """Apply a soft cosine edge to a binary mask (matching EMAN2 mask.distance).

    Uses chamfer distance (iterative propagation) instead of exact EDT
    for a ~5x speedup with sufficient accuracy for a smooth falloff.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (1 = inside, 0 = outside).
    soft_width : float
        Width of the cosine transition zone in pixels.

    Returns
    -------
    np.ndarray
        Soft-edged mask with values in [0, 1].
    """
    if soft_width <= 0:
        return mask.astype(np.float64)
    from scipy.ndimage import distance_transform_edt

    # Exact EDT on a downsampled version, then upsample
    # The cosine falloff is smooth so exact sub-pixel distances aren't critical
    nz, ny, nx = mask.shape
    step = max(1, int(soft_width / 4))
    mask_ds = mask[::step, ::step, ::step].astype(bool)
    dist_ds = distance_transform_edt(~mask_ds) * step
    # Upsample to full resolution via linear interpolation
    from scipy.ndimage import zoom

    dist = zoom(
        dist_ds,
        (nz / dist_ds.shape[0], ny / dist_ds.shape[1], nx / dist_ds.shape[2]),
        order=1,
    )
    dist = dist[:nz, :ny, :nx]

    soft = np.ones(mask.shape, dtype=np.float64)
    outside = ~mask.astype(bool)
    near_edge = outside & (dist > 0) & (dist <= soft_width)
    soft[near_edge] = (np.cos(dist[near_edge] / soft_width * np.pi / 2) + 1) / 2
    soft[outside & (dist > soft_width)] = 0.0
    return soft


def plot_fsc(fsccurves, fscfile, volumes=None, showPlot=False):
    """Plot FSC curves and central sections to a multi-page PDF.

    Parameters
    ----------
    fsccurves : list of tuple
        List of (x, y, label) tuples for FSC curves.
    fscfile : str
        Output file path (pdf, png, jpeg, etc.).
    volumes : list of tuple, optional
        List of (page_title, [(vol_name, vol_data), ...]) groups.
        Each group becomes one page with a row of central sections.
    showPlot : bool, optional
        If True, display the plot interactively. Defaults to False.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    from matplotlib.lines import Line2D
    from matplotlib.backends.backend_pdf import PdfPages

    is_pdf = fscfile.lower().endswith(".pdf")

    if is_pdf:
        pdf = PdfPages(fscfile)
    else:
        pdf = None

    # --- Page 1: FSC curves ---
    ymin = 1.0
    ymax = 0.0
    xmax = 0.0
    fig = plt.figure(figsize=(9, 6), facecolor="w", edgecolor="w")
    for x, y, label in fsccurves:
        xmax = max(xmax, np.max(x))
        ymin = min(ymin, np.min(y[len(y) // 2 :]))
        ymax = max(ymax, np.max(y))
        plt.plot(x, y, label=label)

    l = Line2D([0, xmax], [0.143, 0.143], linestyle="--", color="r")
    plt.gca().add_line(l)

    plt.xlim([0, xmax])
    xstep = round(xmax / 5 / 2, 2) * 2
    if xstep > 0:
        xticks = [xstep * i for i in range(int(xmax / xstep) + 1)]
        xlabels = [r"1/$\infty$"] + ["1/%.1f" % (1 / xt) for xt in xticks[1:]]
        plt.xticks(xticks, xlabels)
        plt.gca().xaxis.set_minor_locator(MultipleLocator(xstep / 10))

    plt.ylim([ymin, ymax])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.gca().yaxis.set_minor_locator(MultipleLocator(0.02))

    plt.gca().xaxis.grid(True)
    plt.gca().yaxis.grid(True)
    plt.gca().grid(linestyle="--", linewidth="0.5")

    plt.xlabel(r"Resolution (1/$\AA$)", fontsize=14)
    plt.ylabel("Fourier Shell Correlation", fontsize=14)

    plt.gca().legend(loc="best", shadow=False, frameon=True, fontsize=12)

    if pdf is not None:
        pdf.savefig(fig)
        plt.close(fig)
    else:
        plt.gcf().savefig(fscfile)
        plt.close()

    # --- Subsequent pages: grouped central sections ---
    if volumes:
        for page_title, vol_list in volumes:
            n_vols = len(vol_list)
            fig, axes = plt.subplots(
                n_vols, 3, figsize=(15, 4.5 * n_vols), facecolor="w", edgecolor="w"
            )
            fig.suptitle(page_title, fontsize=18, fontweight="bold")

            for row, (vol_name, vol_data) in enumerate(vol_list):
                _plot_central_sections_to_axes(axes[row], vol_data, vol_name)

            fig.tight_layout(rect=[0, 0, 1, 0.95])
            if pdf is not None:
                pdf.savefig(fig)
                plt.close(fig)
            else:
                outbase = str(Path(fscfile).with_suffix(""))
                safe_name = page_title.lower().replace(" ", "_")
                fig.savefig(f"{outbase}.{safe_name}.png")
                plt.close(fig)

    if pdf is not None:
        pdf.close()


def _plot_central_sections_to_axes(axes, volume, title):
    """Plot X, Y, Z central sections into existing axes.

    Parameters
    ----------
    axes : array of matplotlib.axes.Axes
        Row of 3 axes to plot into.
    volume : np.ndarray
        3D numpy array.
    title : str
        Column label for this volume.
    """
    nz, ny, nx = volume.shape
    sz = volume[nz // 2, :, :]
    sy = volume[:, ny // 2, :]
    sx = volume[:, :, nx // 2]

    vmin = np.percentile(volume, 1)
    vmax = np.percentile(volume, 99)

    images = [sz, sy, sx]
    labels = [f"XY (Z={nz // 2})", f"XZ (Y={ny // 2})", f"YZ (X={nx // 2})"]

    for col, (img, label) in enumerate(zip(images, labels)):
        axes[col].imshow(img, cmap="gray", aspect="equal", vmin=vmin, vmax=vmax)
        if col == 1:
            axes[col].set_title(f"{title}\n{label}", fontsize=11)
        else:
            axes[col].set_title(label, fontsize=11)
        axes[col].tick_params(labelsize=8)


def check_args(args, parser):
    """Validate trueFSC CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    parser : argparse.ArgumentParser
        The argument parser.

    Returns
    -------
    argparse.Namespace
        Validated arguments.
    """
    outputFormats = "eps jpeg jpg pdf pgf png ps raw rgba svg svgz tif tiff".split()
    if args.plotFile.rsplit(".", 1)[-1] not in outputFormats:
        parser.error(
            f"only these formats are allowed for output plot file: {' '.join(outputFormats)}"
        )
    if args.maskFile:
        if len(args.maskFile) not in [1, 2]:
            parser.error("only 1 or 2 mask files are allowed")
        elif len(args.maskFile) == 1:
            args.maskFile = args.maskFile * 2
    if args.maskThresh:
        if len(args.maskThresh) not in [1, 2]:
            parser.error("only 1 or 2 mask threshold values are allowed")
        elif len(args.maskThresh) == 1:
            args.maskThresh = args.maskThresh * 2
    if args.maskSoft > 0:
        args.refineMask = 0
    return args


def add_args(parser):
    """Add CLI arguments for the trueFSC command.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to attach arguments to.
    """
    parser.add_argument("map1", metavar="<map 1>", type=str, help="input half-map 1")
    parser.add_argument("map2", metavar="<map 2>", type=str, help="input half-map 2")
    parser.add_argument(
        "plotFile",
        metavar="<plot file>",
        type=str,
        nargs="?",
        default="trueFSC.pdf",
        help="output plot file (pdf, png, jpeg format). Default: trueFSC.pdf",
    )

    parser.add_argument(
        "--apix",
        metavar="<Angstrom/pixel>",
        type=float,
        help="pixel size (Angstrom/pixel). Read from map header if not provided",
        default=0,
    )
    parser.add_argument(
        "--cutoffRes",
        metavar="<Angstrom>",
        type=float,
        help="starting resolution for phase randomization. Default: FSC=0.8 of unmasked maps",
        default=0,
    )
    parser.add_argument(
        "--oneMask",
        metavar="<0|1>",
        type=int,
        default=1,
        help="use the same mask for both maps (1) or separate masks (0)",
    )
    parser.add_argument(
        "--showPlot", 
        metavar="<0|1>",
        type=int, 
        default=1, 
        help="show plots on screen (1) or not (0)"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--maskFile",
        metavar="<filename>",
        type=str,
        nargs="*",
        help="mask file(s)",
        default=[],
    )
    group.add_argument(
        "--maskFractionThresh",
        metavar="<0-1>",
        type=float,
        help="mask threshold as fraction of max pixel value",
        default=-1,
    )
    group.add_argument(
        "--maskThresh",
        metavar="<pixel value>",
        type=float,
        nargs="*",
        help="mask threshold pixel value",
        default=[],
    )
    group.add_argument(
        "--maskMass",
        metavar="<kDa>",
        type=float,
        help="total mass of structure in kDa",
        default=0,
    )

    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument(
        "--maskSoft",
        metavar="<Angstrom>",
        type=float,
        help="mask slope width in Angstrom. Default: auto",
        default=-1,
    )
    group2.add_argument(
        "--refineMask",
        metavar="<0|1>",
        type=int,
        help="refine mask slope (0 or 1). Default: 1 (ignored if --maskSoft is set)",
        default=1,
    )

    parser.add_argument(
        "--verbose", 
        metavar="<0-2>",
        type=int, 
        default=1, 
        help="verbose level (0-2). Default: 1"
    )
