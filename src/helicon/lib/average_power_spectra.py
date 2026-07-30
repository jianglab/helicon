"""Average power spectra computation for helical particle images.

Provides both a file-based entry point (used by the HILL web app) and a
DataFrame-based entry point (used by the ``images2star`` plugin).  Helper
functions that already exist in the canonical ``helicon.lib`` are imported
rather than duplicated.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import helicon
from helicon.lib.transforms import (
    compute_phase_difference_across_meridian,
    fft_rescale,
    rotate_shift_image,
)
from helicon.lib.filters import generate_tapering_filter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API — file-based entry point (used by the HILL web app)
# ---------------------------------------------------------------------------


def average_power_spectra(
    input_image: str | Path,
    apix: float = 0.0,
    groupby: list[str] | None = None,
    cutoff_res: list[float] | None = None,
    min_particles: int = -1,
    force_phase_diff: bool = False,
    batch_size: int = 100,
    cpu: int = 1,
    diameter_mask: float = 0,
    align: int = 0,
    fft_x: int = 512,
    fft_y: int = 1024,
):
    """Compute average power spectra from a particle data file.

    Parameters
    ----------
    input_image : str or Path
        Path to a ``.star``, ``.cs``, ``.lst``, or ``.mrcs`` file.
    apix : float, optional
        Pixel size in Å.  ``0`` = read from input file.
    groupby : list of str, optional
        Columns to group by (e.g. ``["class"]``, ``["helicaltube"]``).
    cutoff_res : list of float, optional
        ``[cutoff_res_y, cutoff_res_x]`` in Å.

    min_particles : int, optional
        Minimum particles per group.  ``-1`` = auto (1 % of total).

    force_phase_diff : bool, optional
        Compute phase differences even without ``phi0`` angles.

    batch_size : int, optional
        Particles per batch for parallel processing.

    cpu : int, optional
        Number of parallel workers.

    diameter_mask : float, optional
        Mask diameter in Å.  ``0`` = disabled.

    align : int, optional
        Whether to rotationally align particles.

    fft_x, fft_y : int, optional
        Output FFT dimensions.

    Returns
    -------
    tuple
        ``(results_dict, used_apix, used_cutoff_res)`` where *results_dict* is
        keyed by group index with keys ``group name``, ``ps_avg``,
        ``pd_avg`` (optional), ``image_avg`` (optional), ``#images``.
    """
    data = helicon.image2dataframe(str(input_image))
    data = _remap_columns(data, apix)
    return _average_power_spectra(
        data,
        apix=apix,
        groupby=groupby or [],
        cutoff_res=cutoff_res or [0.0, 0.0],
        min_particles=min_particles,
        force_phase_diff=force_phase_diff,
        batch_size=batch_size,
        cpu=cpu,
        diameter_mask=diameter_mask,
        align=align,
        fft_x=fft_x,
        fft_y=fft_y,
    )


# ---------------------------------------------------------------------------
# Public API — DataFrame-based entry point (used by images2star plugin)
# ---------------------------------------------------------------------------


def average_power_spectra_from_dataframe(
    data: pd.DataFrame,
    apix: float = 0.0,
    groupby: list[str] | None = None,
    cutoff_res: list[float] | None = None,
    min_particles: int = -1,
    force_phase_diff: bool = False,
    batch_size: int = 100,
    cpu: int = 1,
    diameter_mask: float = 0,
    align: int = 0,
    fft_x: int = 512,
    fft_y: int = 1024,
):
    """Compute average power spectra from a pre-loaded DataFrame.

    Expects columns ``filename``, ``pid``, ``apix``, and optionally
    ``phi0``, ``class``, ``helicaltube``.  RELION/CryoSPARC column names
    are mapped automatically.

    Returns the same structure as :func:`average_power_spectra`.
    """
    mapped = _remap_columns(data, apix)
    return _average_power_spectra(
        mapped,
        apix=apix,
        groupby=groupby or [],
        cutoff_res=cutoff_res or [0.0, 0.0],
        min_particles=min_particles,
        force_phase_diff=force_phase_diff,
        batch_size=batch_size,
        cpu=cpu,
        diameter_mask=diameter_mask,
        align=align,
        fft_x=fft_x,
        fft_y=fft_y,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Short column names that the compute code operates on
_COL_FILENAME = "filename"
_COL_PID = "pid"
_COL_APIX = "apix"
_COL_PHI0 = "phi0"
_COL_CLASS = "class"
_COL_HELICALTUBE = "helicaltube"


def _remap_columns(data: pd.DataFrame, apix: float = 0.0) -> pd.DataFrame:
    """Map RELION/CryoSPARC column names to internal short names.

    If the DataFrame already has the short names the original is returned
    (not copied).
    """
    # Already using short names — nothing to do.
    if _COL_FILENAME in data.columns and _COL_PID in data.columns:
        return data

    result = data.copy()

    # ---- filename + pid from rlnImageName ----
    if _COL_FILENAME not in result and "rlnImageName" in result:
        tmp = result["rlnImageName"].str.split("@", expand=True)
        result[_COL_FILENAME] = tmp[1]
        result[_COL_PID] = tmp[0].astype(int) - 1

    # ---- helicaltube ----
    if _COL_HELICALTUBE not in result and "rlnHelicalTubeID" in result:
        result[_COL_HELICALTUBE] = result["rlnHelicalTubeID"].astype(int) - 1

    # ---- class ----
    if _COL_CLASS not in result and "rlnClassNumber" in result:
        result[_COL_CLASS] = result["rlnClassNumber"]

    # ---- phi0 (in-plane angle) ----
    if _COL_PHI0 not in result:
        if "rlnAnglePsiPrior" in result:
            result[_COL_PHI0] = (result["rlnAnglePsiPrior"].astype(float) - 90.0).round(
                3
            )
        elif _COL_FILENAME in result and "phi0" in result.columns:
            pass  # already set

    # ---- apix ----
    if _COL_APIX not in result:
        if "rlnPixelSize" in result:
            result[_COL_APIX] = result["rlnPixelSize"].astype(float)
        elif apix > 0:
            result[_COL_APIX] = apix
        else:
            # Read from first MRC file
            import mrcfile

            sample = result[_COL_FILENAME].iloc[0]
            with mrcfile.open(sample, header_only=True) as mrc:
                result[_COL_APIX] = mrc.voxel_size.x

    return result


def _average_power_spectra(
    data: pd.DataFrame,
    apix: float = 0.0,
    groupby: list[str] | None = None,
    cutoff_res: list[float] | None = None,
    min_particles: int = -1,
    force_phase_diff: bool = False,
    batch_size: int = 100,
    cpu: int = 1,
    diameter_mask: float = 0,
    align: int = 0,
    fft_x: int = 512,
    fft_y: int = 1024,
):
    """Core computation shared by both entry points."""
    data = data.copy()
    cutoff_res_y, cutoff_res_x = cutoff_res

    # ---- resolve pixel size ----
    if _COL_APIX not in data:
        if apix > 0:
            data[_COL_APIX] = apix
        else:
            import mrcfile

            sample = data[_COL_FILENAME].iloc[0]
            with mrcfile.open(sample, header_only=True) as mrc:
                data[_COL_APIX] = mrc.voxel_size.x
    used_apix = data[_COL_APIX].iloc[0]

    # ---- resolve cutoff resolution ----
    if cutoff_res_y < used_apix * 2:
        cutoff_res_y = used_apix * 2
    if cutoff_res_x < used_apix * 2:
        cutoff_res_x = used_apix * 2
    used_cutoff = (cutoff_res_y, cutoff_res_x)

    # ---- auto-groupby ----
    if not groupby:
        if _COL_PHI0 in data:
            if _COL_CLASS in data:
                groupby = [_COL_CLASS]
            elif _COL_HELICALTUBE in data:
                groupby = [_COL_FILENAME, _COL_HELICALTUBE]
        elif len(data[_COL_FILENAME].unique()) == 1:
            import mrcfile

            sample = data[_COL_FILENAME].iloc[0]
            with mrcfile.open(sample, header_only=True) as mrc:
                nx = mrc.header.nx
                ny = mrc.header.ny
                nz = mrc.header.nz
            if nx == ny and nz != nx and nz <= 500 and _COL_HELICALTUBE not in data:
                groupby = [_COL_PID]

    if groupby == ["None"]:
        groupby = []

    # ---- required columns ----
    required = [_COL_APIX]
    if groupby:
        if _COL_HELICALTUBE in groupby and _COL_FILENAME not in groupby:
            groupby = [_COL_FILENAME] + groupby
        required += [_COL_PHI0] if _COL_PHI0 in groupby else []
        required += groupby
    missing = [c for c in required if c not in data]
    if missing:
        logger.error(
            "Parameters %s are not available. Available: %s",
            " ".join(missing),
            " ".join(data.columns),
        )
        return None

    # ---- group particles ----
    if groupby:
        groups = list(data.groupby(groupby, sort=True))
        if len(groups) > 1 and min_particles < 0:
            min_particles = int(len(data) * 0.01)
        if min_particles > 0:
            groups = [g for g in groups if len(g[1]) >= min_particles]
    else:
        groups = [("all_particles", data)]

    compute_phase_diff = force_phase_diff or _COL_PHI0 in data

    from joblib import Parallel, delayed

    fftavgs = Parallel(n_jobs=cpu, verbose=0, prefer="processes")(
        delayed(_average_one_batch)(
            batch,
            group_id,
            compute_phase_diff,
            diameter_mask,
            (cutoff_res_y, cutoff_res_x),
            fft_x,
            fft_y,
            align,
        )
        for batch, group_id in _particle_subsets(groups, batch_size)
    )

    # ---- accumulate batch results ----
    results = {}
    for ps_avg, pd_avg, image_avg, n_ptcls, group_id in fftavgs:
        gi, _, group_name, _, _ = group_id
        if gi not in results:
            d = {
                "group name": group_name,
                "ps_avg": np.zeros_like(ps_avg),
                "#images": 0,
            }
            if pd_avg is not None:
                d["pd_avg"] = np.zeros_like(pd_avg)
            if image_avg is not None:
                d["image_avg"] = np.zeros_like(image_avg)
            results[gi] = d
        results[gi]["ps_avg"] += ps_avg
        if pd_avg is not None:
            results[gi]["pd_avg"] += pd_avg
        if image_avg is not None:
            results[gi]["image_avg"] += image_avg
        results[gi]["#images"] += n_ptcls

    # ---- normalize ----
    for gi in results:
        n = results[gi]["#images"]
        results[gi]["ps_avg"] = np.fft.fftshift(results[gi]["ps_avg"] / n)
        if "pd_avg" in results[gi]:
            results[gi]["pd_avg"] = np.fft.fftshift(
                np.rad2deg(np.arccos(results[gi]["pd_avg"] / n))
            )
        if "image_avg" in results[gi]:
            results[gi]["image_avg"] /= n

    return results, used_apix, used_cutoff


def _particle_subsets(groups, max_batch_size=100):
    """Yield (batch_mgraphs, group_id) tuples for parallel processing."""
    for gi, (group_name, group_particles) in enumerate(groups):
        # Normalize display_name to a hashable form
        if isinstance(group_name, tuple):
            display_name = group_name
        else:
            display_name = (str(group_name),)

        mgraphs = list(group_particles.groupby([_COL_FILENAME], sort=True))
        n_per_mgraph = [len(m[1]) for m in mgraphs]

        batch_idx = 0
        i0 = 0
        while i0 < len(n_per_mgraph):
            for i in range(i0, len(n_per_mgraph)):
                if (
                    sum(n_per_mgraph[i0 : i + 1]) >= max_batch_size
                    or i == len(n_per_mgraph) - 1
                ):
                    batch = mgraphs[i0 : i + 1]
                    group_id = (gi, batch_idx, display_name, len(groups), i0)
                    yield batch, group_id
                    batch_idx += 1
                    i0 = i + 1
                    break


@helicon.cache(cache_dir=str(helicon.cache_dir / "hill"), expires_after=7, verbose=0)
def _average_one_batch(
    mgraphs,
    group_id,
    compute_phase_differences,
    diameter_mask,
    cutoff_res,
    pad_nx,
    pad_ny,
    align,
):
    """Process one batch of particles from one or more micrographs."""
    n_ptcls = sum(len(m[1]) for m in mgraphs)
    gi, bi, group_name, ng, nb = group_id

    phi0_angles = np.zeros(n_ptcls)
    used_apix = mgraphs[0][1][_COL_APIX].iloc[0]
    tapering_filter = None
    data_orig = None
    data_in = None
    i0 = 0

    for _mgraph_name, particles in mgraphs:
        n = len(particles)
        pids = particles[_COL_PID].astype(int).values
        filename = particles[_COL_FILENAME].iloc[0]
        import mrcfile

        with mrcfile.mmap(filename, mode="r") as mrc:
            ny, nx = mrc.data.shape[-2:]
            if data_orig is None:
                data_orig = np.zeros((n_ptcls, ny, nx), dtype=np.float32)
                if align:
                    data_in = np.zeros((n_ptcls, ny, nx), dtype=np.float32)
                else:
                    data_in = data_orig
            if tapering_filter is None:
                if diameter_mask > 0:
                    fraction_x = diameter_mask / used_apix / nx
                else:
                    fraction_x = 0.9
                tapering_filter = generate_tapering_filter(
                    image_size=(ny, nx),
                    fraction_start=[0.9, fraction_x],
                    fraction_slope=0.1,
                )
            if _COL_PHI0 in particles:
                phi0_angles[i0 : i0 + n] = particles[_COL_PHI0].astype(float).values
            for i in range(n):
                data_orig[i0 + i] = mrc.data[pids[i]]
        i0 += n

    if align:
        d_angles = np.zeros(n_ptcls)
        d_shift = np.zeros(n_ptcls)
        for pi in range(n_ptcls):
            d = data_orig[pi]
            dphi = -phi0_angles[pi]
            d_aligned, da, dxy = rotation_trans_align(
                image=d, angle0=dphi, dx0=0, dy0=0, mask=tapering_filter
            )
            data_in[pi] = d_aligned * tapering_filter
            d_angles[pi] = da
            d_shift[pi] = dxy
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Group %d/%d Batch %d/%d: mean rotation = %.2f deg, shift = %.1f A",
                gi + 1,
                ng,
                bi + 1,
                nb,
                np.mean(np.abs(d_angles)),
                np.mean(d_shift) * used_apix,
            )
    else:
        for pi in range(n_ptcls):
            d = data_orig[pi]
            dphi = -phi0_angles[pi]
            d_rotated = (
                rotate_shift_image(data=d, angle=dphi, post_shift=(0, 0))
                if dphi != 0
                else d
            )
            data_in[pi] = d_rotated * tapering_filter

    ps_avg = np.zeros((pad_ny, pad_nx), dtype=np.float64)
    pd_avg = (
        np.zeros((pad_ny, pad_nx), dtype=np.float64)
        if compute_phase_differences
        else None
    )

    for pi in range(n_ptcls):
        fft2d = fft_rescale(
            data=data_in[pi],
            apix=used_apix,
            cutoff_res=cutoff_res,
            output_size=(pad_ny, pad_nx),
        )
        ps_avg += np.abs(fft2d)
        if compute_phase_differences:
            phase = np.angle(fft2d)
            cos = compute_phase_difference_across_meridian(phase)
            pd_avg += np.cos(cos)

    image_avg = np.sum(data_in, axis=0) if logger.isEnabledFor(logging.DEBUG) else None

    return ps_avg, pd_avg, image_avg, n_ptcls, group_id


# ---------------------------------------------------------------------------
# Unique helper — rotation + translation alignment
# ---------------------------------------------------------------------------


def rotation_trans_align(image, angle0, dx0=0, dy0=0, mask=None):
    """Refine rotation and translation alignment.

    Searches for the rotation angle (around *angle0*) and shift that
    minimises the difference between the image and its 180-degree-rotated
    and mirrored versions.

    Parameters
    ----------
    image : np.ndarray
        Input 2-D image.
    angle0 : float
        Initial rotation angle in degrees.
    dx0, dy0 : float
        Initial translation in pixels.
    mask : np.ndarray or None
        Optional binary mask.

    Returns
    -------
    tuple
        ``(aligned_image, delta_angle, total_shift)``.
    """
    if mask is None:
        mask = 1.0

    def _score(x):
        da, dy, dx = x
        angle = angle0 + da
        tmp = rotate_shift_image(data=image, angle=angle, pre_shift=(dy, dx))
        tmps = [
            tmp,
            tmp[::-1, :],
            tmp[:, ::-1],
            tmp[::-1, ::-1],
        ]
        tmp2 = rotate_shift_image(data=image, angle=angle + 180, pre_shift=(dy, dx))
        tmps += [
            tmp2,
            tmp2[::-1, :],
            tmp2[:, ::-1],
            tmp2[::-1, ::-1],
        ]
        mean = sum(tmps) / len(tmps)
        err = sum(np.sum(np.abs(t - mean) * mask) for t in tmps)
        return err / (len(tmps) * image.size)

    from scipy.optimize import fmin

    res = fmin(_score, x0=(0, dy0, dx0), xtol=1e-4, disp=0)
    da, dy, dx = res
    aligned = rotate_shift_image(data=image, angle=angle0 + da, pre_shift=(dy, dx))
    return aligned, da, np.hypot(dy, dx)
