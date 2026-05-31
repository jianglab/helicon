#!/usr/bin/env python3
"""Helical segment consistency — detect and filter outlier helical segments."""

#  Section 0   imports

from __future__ import annotations
import argparse
import logging
import pandas as pd
import numpy as np
import math

from PIL import Image
from pathlib import Path

from scipy.optimize import minimize, curve_fit
from matplotlib import pyplot as plt

import os
import datetime, time, pytz

#!pip install pytz
local_tz = pytz.timezone("America/New_York")  # Change this to your timezone

logger = logging.getLogger(__name__)
# local_tz = pytz.timezone("America/Chicago")  # Change this to your timezone

from uuid import uuid4
import sys
from helicon.lib.exceptions import (
    HeliconError,
    HeliconValidationError,
    HeliconFileExistsError,
)

sys.executable

#!pip install starfile
import starfile

logger = logging.getLogger(__name__)

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module=r"starfile(\.|$)")

import platform

# !pip install pylops
# !pip install openpyxl

# Check that NumPy is being used under pylops


#  Section 5: Plot All the Data

# Batch 450 plots into 9 JPGs (each a 10x5 grid), then insert them into a PowerPoint.
# You can adapt `plot_one(ax, i)` to draw your real plot for item i.
#
# Outputs:
# - 9 JPEGs: /mnt/data/page_01.jpg ... /mnt/data/page_09.jpg
# - 1 PPTX:  /mnt/data/plots_batch.pptx
#
# Feel free to rerun after editing `plot_one` to use your real data.


# ---- Your per-item plotting goes here ----
def plot_one(ax, j):
    """Plot a sample sine-wave trace into a given axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw into.
    j : int
        Plot index for title labeling.
    """
    x = np.linspace(0, 2 * np.pi, 200)
    y = np.sin(x + (j * np.pi / 30.0)) * np.exp(-x / 5) + 0.05 * np.random.randn(len(x))
    ax.plot(x, y)
    ax.set_title(f"Plot {j+1}", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 2 * np.pi)


def jpgs_to_pdf(jpg_paths, pdf_path, dpi=200):
    """Combine a list of JPG images into a single multi-page PDF.

    Parameters
    ----------
    jpg_paths : list of str or Path
        Ordered list of image paths.
    pdf_path : str or Path
        Output PDF path.
    dpi : int, optional
        Nominal output DPI for PDF metadata. Defaults to 200.

    Returns
    -------
    str
        String form of the output PDF path.
    """
    jpg_paths = [str(p) for p in jpg_paths]
    if not jpg_paths:
        raise ValueError("jpg_paths is empty.")

    images = []
    for p in jpg_paths:
        im = Image.open(p)
        if im.mode != "RGB":
            im = im.convert("RGB")
        images.append(im)

    first, rest = images[0], images[1:]
    pdf_path = Path(pdf_path)
    first.save(
        pdf_path, "PDF", resolution=float(dpi), save_all=True, append_images=rest
    )
    return str(pdf_path)


def wrap_sym(z, P):
    """Wrap a value to the symmetric interval ``[-P/2, P/2)``.

    Parameters
    ----------
    z : ndarray or float
        Input value(s).
    P : float
        Period.

    Returns
    -------
    ndarray or float
        Wrapped value(s).
    """
    return (z + P / 2) % P - P / 2


def unwrap_sequence(y, P):
    """Phase-unwrap a 1D sequence measured modulo P.

    Adjusts by multiples of P so consecutive differences lie in
    ``[-P/2, P/2)``.

    Parameters
    ----------
    y : ndarray
        1D input sequence.
    P : float
        Period.

    Returns
    -------
    ndarray
        Unwrapped sequence.
    """
    y = np.asarray(y, dtype=float)
    if y.size <= 1:
        return y.copy()
    dy = np.diff(y)
    dy_wrapped = wrap_sym(dy, P)
    corr = np.cumsum(dy_wrapped - dy)  # cumulative  multiples of P
    return y + np.concatenate(([0.0], corr))


def solve_b_given_m_wrapped(x, o, P, m):
    """Solve for the L2-optimal intercept given a wrapped-periodic slope.

    Steps:
        1. Compute raw residuals ``r0 = o - m*x`` (modulo P).
        2. Unwrap the residuals.
        3. Set ``b = mean(unwrapped residuals)``.
        4. Score with shortest-arc residuals.

    Parameters
    ----------
    x : ndarray
        Independent variable.
    o : ndarray
        Observed values (modulo P).
    P : float
        Period (e.g. 2π or 360).
    m : float
        Slope.

    Returns
    -------
    tuple of (float, float)
        ``(b, SSE)`` where *b* is the optimal intercept and *SSE*
        is the sum of squared errors.
    """
    x = np.asarray(x, float)
    o = np.asarray(o, float)
    r0 = o - m * x
    r = unwrap_sequence(r0, P)
    b = float(np.mean(r))
    d = wrap_sym(o - (m * x + b), P)
    SSE = float(np.sum(d**2))
    return b, SSE


def fit_line_wrapped_by_m_grid(x, o, P, m_min, m_max, num_m=501):
    """Grid-search for the best slope via wrapped-periodic linear fitting.

    Parameters
    ----------
    x : ndarray
        Independent variable.
    o : ndarray
        Observed values (modulo P).
    P : float
        Period.
    m_min : float
        Minimum slope to search.
    m_max : float
        Maximum slope to search.
    num_m : int, optional
        Number of grid points. Defaults to 501.

    Returns
    -------
    tuple
        ``(m_best, b_best, SSE_best, (m_grid, b_grid, S_grid))``.
    """
    m_grid = np.linspace(m_min, m_max, num_m)
    b_grid = np.empty_like(m_grid)
    S_grid = np.empty_like(m_grid)
    for k, m in enumerate(m_grid):
        b, S = solve_b_given_m_wrapped(x, o, P, m)
        b_grid[k] = b
        S_grid[k] = S
    i = int(np.argmin(S_grid))
    return (
        float(m_grid[i]),
        float(b_grid[i]),
        float(S_grid[i]),
        (m_grid, b_grid, S_grid),
    )


# --- 1) Plain (non-wrapped) data ---
def plot_linear_fit_simple(x, o, m_best, b_best, P=2.0 * np.pi):
    """Generate data for plotting a linear fit with periodic wrapping.

    Parameters
    ----------
    x : ndarray
        Independent variable.
    o : ndarray
        Observed values.
    m_best : float
        Best-fit slope.
    b_best : float
        Best-fit intercept.
    P : float, optional
        Period. Defaults to 2π.

    Returns
    -------
    tuple of ndarray
        ``(xx, yy, yOldUnwrapped)`` for plotting.
    """
    x = np.asarray(x, float)
    o = np.asarray(o, float)

    # Line to draw across the x-range
    xx = np.linspace(x.min(), x.max(), 400)
    yy = m_best * xx + b_best
    yy = yy % (P)

    yOldUnwrapped = m_best * x + b_best
    if 0:
        plt.figure()
        plt.scatter(x, o, s=18, label="data")
        plt.plot(xx, yy, label=f"fit: y={m_best:.4g}x+{b_best:.4g}")
        plt.xlabel("x")
        plt.ylabel("o")
        plt.title("Linear fit")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return xx, yy, yOldUnwrapped


def get_angular_errors(x, o, m_best, b_best, Period):
    """Compute angular errors between observations and a linear fit.

    Parameters
    ----------
    x : ndarray
        Independent variable.
    o : ndarray
        Observed values (modulo Period).
    m_best : float
        Best-fit slope.
    b_best : float
        Best-fit intercept.
    Period : float
        Period.

    Returns
    -------
    tuple of ndarray
        ``(AngErrors, AbsAngErrors)`` — signed and absolute angular errors.
    """
    yy = m_best * x + b_best
    # This is the fit on the resampled data
    AngErrors = (o - yy + Period / 2.0) % Period - Period / 2.0
    AbsAngErrors = np.abs(AngErrors)
    return AngErrors, AbsAngErrors


# Checking if this appears elsewhere


def l1_core_average(x, k=50):
    """Compute the L1 core average — the point minimizing L1 deviation over the densest k points.

    Parameters
    ----------
    x : ndarray
        Input data.
    k : int, optional
        Number of closest points to consider. Defaults to 50.

    Returns
    -------
    tuple of (float, float, ndarray)
        ``(m_best, l1_sum, subset)`` where *m_best* is the L1 core average,
        *l1_sum* is the minimal sum of absolute deviations, and *subset*
        is the k-point window that determines *m_best*.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n == 0:
        raise ValueError("No finite data points.")
    k = min(k, n)

    xs = np.sort(x)
    pref = np.concatenate(([0.0], np.cumsum(xs)))

    best_sum = np.inf
    best_m = None
    best_slice = (0, k)

    if k % 2 == 1:  # odd k → median is xs[i+half]
        half = k // 2
        for i in range(0, n - k + 1):
            m_idx = i + half
            m = xs[m_idx]
            # left of m (i .. m_idx-1)
            left_sum = m * (m_idx - i) - (pref[m_idx] - pref[i])
            # right of m (m_idx+1 .. i+k-1)
            right_sum = (pref[i + k] - pref[m_idx + 1]) - m * (i + k - m_idx - 1)
            total = left_sum + right_sum
            if total < best_sum:
                best_sum = total
                best_m = m
                best_slice = (i, i + k)
    else:  # even k → any m in [xs[j], xs[j+1]] minimizes; pick midpoint
        half = k // 2
        for i in range(0, n - k + 1):
            j = i + half - 1
            a, b = xs[j], xs[j + 1]
            m = 0.5 * (a + b)
            # elements strictly to the left count = (j+1 - i) = half
            L = j + 1 - i
            left_sum = m * L - (pref[j + 1] - pref[i])
            # elements to the right count = (i+k - (j+1)) = k - half
            R = i + k - (j + 1)
            right_sum = (pref[i + k] - pref[j + 1]) - m * R
            total = left_sum + right_sum
            if total < best_sum:
                best_sum = total
                best_m = m
                best_slice = (i, i + k)

    subset = xs[best_slice[0] : best_slice[1]]
    return best_m, best_sum, subset


def HelicalSegmentConsistency(
    data: pd.DataFrame,
    *,
    convert_path_fn=None,
    verbose: int = 0,
    input_star_path: str | None = None,
    output_star_path: str | None = None,
    param: str | None = None,
):
    """Detect and filter outlier helical segments based on angular consistency.

    Analyzes the angular progression of rot/tilt/psi along each helix and
    computes per-segment and per-helix quality metrics.

    Parameters
    ----------
    data : pd.DataFrame
        Particle DataFrame with helical parameters.
    convert_path_fn : callable, optional
        Path conversion function (unused, kept for API compatibility).
    verbose : int, optional
        Verbosity level. Defaults to 0.
    input_star_path : str, optional
        Path to the input STAR file (for output naming).
    output_star_path : str, optional
        Path to the output STAR file.
    param : str, optional
        Extra parameter string.

    Returns
    -------
    tuple of (pd.DataFrame, dict)
        Updated DataFrame and stats dictionary.
    """
    # Mutate/extend data and return (data, stats).
    # Keep this ABI stable so it plugs into images2star.py cleanly.
    parts = data.copy()
    logger.debug("parts columns: %s", list(parts.columns))
    logger.info("Total # of columns =%d", len(parts.columns))

    StarFileName = input_star_path
    logger.info("STAR file path: %s", StarFileName)
    StarFileOutName = StarFileName[:-5] + "Out.star"

    num_m = 1201

    CurrentWorkingDir = os.getcwd()
    OutputFigsDir = "Figs25_" + StarFileName[:-5]
    OutputFigsDir = CurrentWorkingDir + "/Figs25_HOM_" + StarFileName[:-5]

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if 0:
        OutputFigsDir = "Figs" + stamp
    if not Path(OutputFigsDir).is_dir():
        os.mkdir(OutputFigsDir)

    logger.info("Output figs dir: %s", OutputFigsDir)
    OUTPDFDIR = Path(OutputFigsDir + "/Stuff/")
    #  This is the subdirectory  where the jpg and PDFs get stored
    pdf_path = "batch_" + StarFileName[:-5] + ".pdf"
    # This is the filename within Stuff

    QThreshhold = 10000
    NumSegmentsMin = 25

    StarFileNameKey = StarFileName[9:-5]

    logger.info("Section 0")
    # Get UTC time and convert to local time
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    now_local = now_utc.astimezone(local_tz)
    logger.info("PSU date and time: %s", now_local.strftime("%Y-%m-%d %H:%M:%S %Z"))

    stats = dict()
    # Will be populated later with keys like:
    # nsegments, tilt_means, tilt_sigmas, psi_sigmas, rot_sigmas, options

    #       Section 1    create some auxiliary columns
    logger.info("Section 1    create some auxiliary columns")

    # Parse rlnImageName into particle number (0-based) and micrograph string
    split = parts["rlnImageName"].str.split("@", n=1, expand=True)
    parts["rlnPartNum"] = split[0].astype(int) - 1  # zero-based, same as RELION
    parts["rlnMicrographFromImageName"] = split[1]

    # Assign unique integer ID per micrograph (also zero-based this time)
    micro_to_id = {
        name: i for i, name in enumerate(parts["rlnMicrographFromImageName"].unique())
    }
    parts["rlnMicUniqId"] = parts["rlnMicrographFromImageName"].map(micro_to_id)

    # Parse rlnImageName into particle number and micrograph string
    split = parts["rlnImageName"].str.split("@", n=1, expand=True)

    # Convert the left side to integers
    nums = pd.to_numeric(split[0], errors="coerce").astype("Int64")

    # If it looks 1-based (no zeros and min>=1), shift to 0-based
    if nums.notna().any():
        if (nums == 0).sum() == 0 and nums.min() >= 1:
            nums = nums - 1

    parts["rlnPartNum"] = nums.astype(int)  # now 0..N-1
    parts["rlnMicrographFromImageName"] = split[1]

    # Unique micrograph ID (zero-based for consistency)
    micro_to_id = {
        name: i for i, name in enumerate(parts["rlnMicrographFromImageName"].unique())
    }
    parts["rlnMicUniqId"] = parts["rlnMicrographFromImageName"].map(micro_to_id)

    # List of unique micrographs
    MicrographList = list(micro_to_id.keys())

    # (Optional) quick sanity checks
    logger.info(
        "min/max rlnPartNum: %d %d",
        int(parts["rlnPartNum"].min()),
        int(parts["rlnPartNum"].max()),
    )
    logger.info("unique micrographs: %d", len(MicrographList))

    # Collect unique micrograph list
    MicrographList = list(micro_to_id.keys())
    logger.info("Number of unique micrographs: %d", len(MicrographList))  # expect 3514
    if 0:
        print("First few micrographs:", MicrographList[:5])

    #  Created 'rlnPartNum' 'rlnMicrographFromImageName' 'rlnMicUniqId'

    #       Section 2:    create a categorical index for the pair (MicUniqId, TubeID)
    #   It is called  rlnHelicalTubeAndMicID
    #   parts is updated
    #     NumberUniqueHelices is found

    logger.info("Section 2  create some categorical index")

    parts["rlnHelicalTubeAndMicID"] = parts.groupby(
        ["rlnMicUniqId", "rlnHelicalTubeID"]
    ).ngroup()

    # ensure contiguous 0-based IDs
    parts["rlnHelicalTubeAndMicID"] = parts["rlnHelicalTubeAndMicID"].astype(int)

    # check result
    if 0:
        print(
            parts[["rlnMicUniqId", "rlnHelicalTubeID", "rlnHelicalTubeAndMicID"]].head(
                5
            )
        )
    NumberUniqueHelices = parts["rlnHelicalTubeAndMicID"].nunique()
    logger.info("Number of unique helices: %d", NumberUniqueHelices)
    if 0:
        print(max(parts["rlnHelicalTubeAndMicID"]))

    # Get UTC time and convert to local time
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    now_local = now_utc.astimezone(local_tz)
    logger.info("PSU date and time: %s", now_local.strftime("%Y-%m-%d %H:%M:%S %Z"))

    # Section 2.5: Empirically calculate period and symmetry

    NumSegmentsAllArray = []
    HelixMicIDsManySegments = []
    NumSegmentsManySegmentsArray = []
    AveAnglesTiltArray = []
    AllRotArray = []

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    now_local = now_utc.astimezone(local_tz)
    logger.info(
        "Section 2.5 PSU date and time: %s", now_local.strftime("%Y-%m-%d %H:%M:%S %Z")
    )

    os.chdir(OutputFigsDir)

    logger.info("CWD: %s", os.getcwd())

    # HelixMicID = 12;#44
    Count = 0

    logger.info("HelixMicID,NumSegments, AveAngles")

    for HelixMicID in range(NumberUniqueHelices):
        # if Count >5: continue
        subid = parts[parts["rlnHelicalTubeAndMicID"] == HelixMicID]
        NumSegments = subid.shape[0]
        NumSegmentsAllArray.append(NumSegments)
        if NumSegments < NumSegmentsMin:
            continue

        HelixMicIDsManySegments.append(HelixMicID)
        NumSegmentsManySegmentsArray.append(NumSegments)

        # idx = subid["rlnHelicalSegmentIndex"]
        idx = subid["rlnHelicalTrackLengthAngst"]
        angles_deg = subid["rlnAngleRot"]
        angles_tilt = subid["rlnAngleTilt"]
        AllRotArray.append(np.array(angles_deg))

        AveAnglesTiltArray.append(np.mean(angles_tilt))

    # %whos list

    logger.info(f"NumSegmentsAllArray          = {np.sum(NumSegmentsAllArray)}")
    logger.info(
        f"NumSegmentsManySegmentsArray = {np.sum(NumSegmentsManySegmentsArray)}"
    )
    logger.info(f"Total HelixMicIDsManySegments = {len(HelixMicIDsManySegments)}")

    #  Section 2.6  Calculate Period
    #      AllRotArrayNP =

    AllRotArrayNP = np.concatenate(AllRotArray)
    plt.figure(figsize=(3, 3))
    plt.hist(AllRotArrayNP, 100)

    AllRotArrayBreadth = np.max(AllRotArrayNP) - np.min(AllRotArrayNP)

    logger.info("AllRotArrayBreadth = " + str(AllRotArrayBreadth))

    SymPre = 360 / AllRotArrayBreadth
    SymGuess = int(SymPre + 0.3)

    logger.info("Guess for Symmetry = " + str(SymGuess))

    if SymGuess == 1:
        SymGuess = 2
    Period = 360 / SymGuess

    logger.info("Guess for Symmetry = " + str(SymGuess))

    logger.info("Period = " + str(Period))

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    now_local = now_utc.astimezone(local_tz)
    logger.info("PSU date and time: %s", now_local.strftime("%Y-%m-%d %H:%M:%S %Z"))

    logger.info("CWD: %s", os.getcwd())

    #################################################################################
    #################################################################################
    #  Section 3: Assemble  all the slopes.
    # subid are the particular helix under scrutiny
    #  mManySegmentsArray       slopes of helices with more than NumSegmentsMin segments
    #  SminManySegmentsArray    errors metric 1  of helices with more than NumSegmentsMin segments

    logger.info("Section 3: Get listing of all the slopes.  And make global plots")

    os.chdir(OutputFigsDir)
    # print('HelixMicID,NumSegments, m_best, SMin, AveAngles')

    Count = 0
    mManySegmentsArray = []
    SminManySegmentsArray = []
    DEBUG = 0

    for HelixMicID in range(NumberUniqueHelices):
        # if Count >25: continue
        subid = parts[parts["rlnHelicalTubeAndMicID"] == HelixMicID]
        NumSegments = subid.shape[0]
        if NumSegments < NumSegmentsMin:
            continue

        # idx = subid["rlnHelicalSegmentIndex"]
        idx = subid["rlnHelicalTrackLengthAngst"]
        angles_deg = subid["rlnAngleRot"] % Period
        angles_tilt = subid["rlnAngleTilt"]

        # unwrapped_Rot = unwrap_to_line(idx, angles_deg, period_candidates=(180, 360), discont_frac=0.5)

        m_best, b_best, Smin, _ = fit_line_wrapped_by_m_grid(
            idx, angles_deg, Period, m_min=-1.1, m_max=1.1, num_m=num_m
        )
        # print(f"m_best={m_best:.4f}, b_best={b_best:.4f}, Smin={Smin:.4f}")
        xx, yy, _ = plot_linear_fit_simple(idx, angles_deg, m_best, b_best, Period)

        SminManySegmentsArray.append(Smin)
        mManySegmentsArray.append(m_best)
        # print(HelixMicID,NumSegments, f"{m_best:.3g}", f"{Smin:.3g}")
        Count += 1

    plt.hist(mManySegmentsArray)
    plt.savefig("mManySegmentsArray.jpg")

    #################################################################################
    #     Section 3.1   Find the best helix, plot and save it. Determine mBest (which is positive)

    SminManySegmentsInd = np.argmin(SminManySegmentsArray)
    SminManySegmentsMin = SminManySegmentsArray[SminManySegmentsInd]
    #
    HelixMicIDsManySegmentsMin = HelixMicIDsManySegments[SminManySegmentsInd]
    #  27
    mManySegmentsMin = mManySegmentsArray[SminManySegmentsInd]
    # 0.245
    # mBest = np.abs(mManySegmentsMin)

    logger.info(
        " StarFile,  HelixMicIDsManySegmentsMin, SminManySegmentsMin, SminManySegmentsInd, mManySegmentsMin, Period"
    )
    logger.info(
        "%s %s %s %s %s %s",
        StarFileName,
        HelixMicIDsManySegmentsMin,
        SminManySegmentsMin,
        SminManySegmentsInd,
        mManySegmentsMin,
        Period,
    )
    # choose a particular micrograph and tube

    # StarFile,  HelixMicIDsManySegmentsMin, SminManySegmentsMin, SminManySegmentsInd, mManySegmentsMin
    # run_fahim_data_with_lineage.star 27 9.105462675670747 14 -0.245

    HelicalTubeAndMicID = HelixMicIDsManySegmentsMin
    sub = parts[
        (parts["rlnHelicalTubeAndMicID"] == HelicalTubeAndMicID)
    ]  # 27 for Fahim Data

    # sort by segment index to see order along filament
    # sub = sub.sort_values("rlnHelicalSegmentIndex")
    sub = sub.sort_values("rlnHelicalTrackLengthAngst")

    # plot AnglePsi vs. segment index
    plt.figure(figsize=(7, 3))
    # plt.plot(sub["rlnHelicalSegmentIndex"], sub["rlnAnglePsi"], marker="o")
    plt.plot(sub["rlnHelicalTrackLengthAngst"], sub["rlnAngleRot"] % Period, marker="o")
    plt.xlabel("Segment index along filament")
    plt.xlabel("Angstroms along filament")
    plt.ylabel("AngleRot (deg)")
    # plt.title(f"Micrograph {mic_id}, Tube {tube_id} " + StarFileName)
    plt.title(
        f"HelicalTubeAndMicID {HelicalTubeAndMicID} "
        f"Smin = {int(SminManySegmentsMin)}\n"
        f"for starfile: {StarFileName}\n"
        f"mManySegmentsMin = {mManySegmentsMin:.3g}"
    )
    plt.grid(True)
    # plt.show()

    YYY = sub["rlnAngleRot"] % Period
    XXX = sub["rlnHelicalTrackLengthAngst"]

    # Get UTC time and convert to local time

    plt.savefig(f"{StarFileNameKey}BestHelix.jpg")

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    now_local = now_utc.astimezone(local_tz)
    logger.info("PSU date and time: %s", now_local.strftime("%Y-%m-%d %H:%M:%S %Z"))

    mBest = np.abs(mManySegmentsMin)

    ###################################################3
    #       Section 3.2   plot distributions of lengths >25
    logger.info("SymGuess: %s", SymGuess)

    # plt.hist(SminManySegmentsArray)
    plt.figure(figsize=(3, 2))
    plt.hist(NumSegmentsManySegmentsArray, 50)
    plt.title("NumSegmentsManySegmentsArray")
    plt.xlabel("Number of Segments")
    plt.ylabel("Histogram")
    # np.argmax(NumSegmentsManySegmentsArray)
    # NumSegmentsManySegmentsArray[258]

    plt.savefig(f"{StarFileNameKey}DistLongSegmentLengths.jpg")

    logger.info("Number Of long Helices for " + StarFileName + " = " + str(Count))

    logger.info("Total Segments At Start = " + str(np.sum(NumSegmentsAllArray)))
    logger.info(
        "Total Segments In Long Helices = " + str(np.sum(NumSegmentsManySegmentsArray))
    )

    logger.info("CWD: %s", os.getcwd())

    # Get UTC time and convert to local time
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    now_local = now_utc.astimezone(local_tz)
    logger.info("PSU date and time: %s", now_local.strftime("%Y-%m-%d %H:%M:%S %Z"))

    #################################################################################
    #################################################################################
    #  Section 4.0: Make Global Plots

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    now_local = now_utc.astimezone(local_tz)
    logger.info("PSU date and time: %s", now_local.strftime("%Y-%m-%d %H:%M:%S %Z"))

    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    _ = plt.hist(np.abs(mManySegmentsArray), 30, log=True)
    plt.axvline(x=mBest, color="r", linestyle="-", linewidth=1)
    # plt.xlim(0,.1)
    plt.title("Hist of slope (Angle/Angstrom)")
    # plt.subplot(1,2,2)
    # plt.scatter(AveAnglesTiltArray,np.abs(mManySegmentsArray))
    plt.subplot(1, 2, 2)
    plt.scatter(np.abs(mManySegmentsArray), AveAnglesTiltArray, s=1)
    # plt.xlim(0,.1)
    plt.axvline(x=mBest * 1, color="r", linestyle="-", linewidth=1)
    plt.title("Ave Tilt vs slope \n" + StarFileNameKey)

    plt.savefig(f"{StarFileNameKey}HistogramSlopesTiltvsSlope.jpg")

    #################################################################################
    #  Section 4.1: Make   Global Plots on Long Helices
    # HelixMicIDsManySegments=[]
    # NumSegmentsManySegmentsArray = []
    # mManySegmentsArray  = []
    # SminManySegmentsArray = []

    NumLongHelices = len(mManySegmentsArray)
    logger.info("Total # long Helices (num segments >25)=" + str(NumLongHelices))
    plt.figure(figsize=(12, 3))

    plt.subplot(1, 3, 1)
    label = "hist num  long Helices"
    plt.hist(NumSegmentsManySegmentsArray, 20, label=label)
    plt.title(label)

    plt.subplot(1, 3, 2)
    label = "hist slopes  long Helices"
    plt.hist(mManySegmentsArray, 40, label=label)
    plt.title(label)

    plt.subplot(1, 3, 3)
    label = "hist S long Helices"
    plt.hist(SminManySegmentsArray, 20, label=label)
    plt.title(label)

    plt.savefig(f"{StarFileNameKey}StatsLongHelicesGT25.jpg")

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    now_local = now_utc.astimezone(local_tz)
    logger.info("PSU date and time: %s", now_local.strftime("%Y-%m-%d %H:%M:%S %Z"))

    logger.info("CWD: %s", os.getcwd())

    #################################################################################
    #  Section 4.2: Make Global Plots of Long Good Helices

    HelixMicIDsNP = np.asarray(HelixMicIDsManySegments)
    NumSegNP = np.asarray(NumSegmentsManySegmentsArray)
    mNP = np.asarray(mManySegmentsArray)
    SminNP = np.asarray(SminManySegmentsArray)

    mask = SminNP < 500  # boolean mask instead of np.where(...)[0]

    logger.info("len(mNP),sum(mask)")
    logger.info("len(mNP)=%d, sum(mask)=%d", len(mNP), sum(mask))

    logger.info(
        "Total # long Helices (num segments >25) and Smin<10000=" + str(sum(mask))
    )
    plt.figure(figsize=(12, 3))

    plt.subplot(1, 3, 1)
    label = "hist num  long, good Helices"
    plt.hist(NumSegNP[mask], 20, label=label)
    plt.title(label)

    plt.subplot(1, 3, 2)
    label = "hist slopes  long, good " + str(sum(mask)) + " Helices"
    plt.hist(np.abs(mNP[mask]), 20, label=label)
    plt.title(label)
    # plt.xlim(0.00,0.35)

    plt.subplot(1, 3, 3)
    label = "hist S long, good Helices"
    plt.hist(SminNP[mask], 20, label=label)
    plt.title(label)

    plt.savefig(f"{StarFileNameKey}PlotsForGoodLongHelices.jpg")

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    now_local = now_utc.astimezone(local_tz)
    logger.info("PSU date and time: %s", now_local.strftime("%Y-%m-%d %H:%M:%S %Z"))

    logger.info("StarFileName: %s", StarFileName)
    logger.info("m_best: %s", m_best)

    #################################################################################
    #      Section 4.3 Initialize two new columns in dataframe (which becomes output star)

    parts["rlnHelicalTubeAndMicIDGood"] = 0.0
    parts["rlnHelicalTubeAndMicIDGoodSegValue"] = 0.0

    # subid = parts[parts["rlnHelicalTubeAndMicID"] == HelixMicID ]
    # NumSegments = subid.shape[0]
    # idx = subid["rlnHelicalTrackLengthAngst"]
    # segmentIDs = subid["rlnPartNum"]
    # angles_deg= subid["rlnAngleRot"]
    # angles_tilt= subid["rlnAngleTilt"]
    # print(segmentIDs)

    #   Section 4.4 This is the globally best

    SGlobalMin = np.argmin(SminNP)
    HelixMicIDsBest = HelixMicIDsManySegments[SGlobalMin]
    logger.info("HelixMicIDsBest = " + str(HelixMicIDsBest))
    logger.info("mBest = " + str(mBest))
    logger.info("Period = " + str(Period))

    logger.info("HelixMicIDsManySegmentsMin = " + str(HelixMicIDsManySegmentsMin))

    def fit_line_wrapped_by_m_known(idx, angles_deg, Period, m_best):
        m_abs = np.abs(m_best)
        x = np.asarray(idx, float)
        o = np.asarray(angles_deg, float)

        #  Try positive solution
        mpos = m_abs
        r0pos = o - mpos * x
        rpos = unwrap_sequence(r0pos, Period)
        bpos = float(np.mean(rpos % Period))
        dpos = wrap_sym(o - (mpos * x + bpos), Period)
        SSEpos = float(np.sum(np.abs(dpos)))

        #  Try negative solution
        mneg = -m_abs
        r0neg = o - mneg * x
        rneg = unwrap_sequence(r0neg, Period)
        bneg = float(np.mean(rneg % Period))
        dneg = wrap_sym(o - (mneg * x + bneg), Period)
        SSEneg = float(np.sum(np.abs(dneg)))

        if SSEpos <= SSEneg:
            mFinal, rFinal, bFinal, dFinal, SSEfinal = mpos, rpos, bpos, dpos, SSEpos
        else:
            mFinal, rFinal, bFinal, dFinal, SSEfinal = mneg, rneg, bneg, dneg, SSEneg

        # print(SSEpos,SSEneg)
        return float(mFinal), float(bFinal), float(SSEfinal)

    # mFinal, bFinal, SSEfinal  = fit_line_wrapped_by_m_known(idx, angles_deg, P, m_best )

    # HelixMicID = 58; #HelixMicIDsBest
    # if Count >0: continue
    subid = parts[parts["rlnHelicalTubeAndMicID"] == HelixMicIDsBest]
    NumSegments = subid.shape[0]
    logger.info("Globally Best has NumSegments=" + str(NumSegments))

    # HelixMicIDsManySegments.append(HelixMicID)
    # NumSegmentsManySegmentsArray.append(NumSegments)

    # idx = subid["rlnHelicalSegmentIndex"]
    idx = subid["rlnHelicalTrackLengthAngst"]
    angles_deg_full = subid["rlnAngleRot"]
    angles_deg = angles_deg_full % Period
    # unwrapped_Rot = unwrap_to_line(idx, angles_deg, period_candidates=(180, 360), discont_frac=0.5)

    mFinal, bFinal, SSEfinal = fit_line_wrapped_by_m_known(
        idx, angles_deg, Period, mBest
    )
    xx, yy, _ = plot_linear_fit_simple(idx, angles_deg, mFinal, bFinal, Period)

    # SminManySegmentsArray.append(Smin)
    # mManySegmentsArray.append(m_best)

    logger.info("HelixMicID,NumSegments, m_best, Smin, b_best")
    logger.info(
        "%s %s %s %s %s",
        HelixMicID,
        NumSegments,
        f"{mFinal:.3g}",
        f"{Smin:.3g}",
        f"{bFinal:.3g}",
    )

    plt.figure(figsize=(8, 3))

    plt.subplot(1, 2, 1)
    plt.scatter(idx, angles_deg_full, marker="o")

    plt.subplot(1, 2, 2)
    # plot AngleRot vs. segment index
    # plt.subplot(1,3,1)
    # plt.scatter(idx, unwrapped_Rot['angles_unwrapped_deg'], marker="o")

    plt.scatter(idx, angles_deg, marker="o")
    plt.plot(xx, yy, "g-", lw=2, label="Fitted Line")
    # plt.xlabel("Segment index along filament")
    plt.xlabel("Angstroms along filament")
    plt.title(
        "AngleRot for HelixMicID="
        + str(HelixMicID)
        + "\n  Smin="
        + format(SSEfinal, ".2f")
        + " mFinal="
        + format(mFinal, ".3f")
    )

    plt.savefig("AnotherGoodHelixNP.jpg")

    logger.info("mBest = " + str(abs(mBest)))

    #### Good up to here
    #################################################################################
    ############################################################################################
    #    Section 5.0           This is a recalculation of S with the second metric (where mBest is fixed)

    # os.chdir(MainPSUDir)#
    os.chdir(OutputFigsDir)

    HelixMicIDsManySegments_2 = []
    NumSegmentsManySegmentsArray_2 = []
    mManySegmentsArray_2 = []
    SminManySegmentsArray_2 = []
    AveAnglesTiltArray_2 = []
    bFinalArray_2 = []

    for j, HelixMicID in enumerate(HelixMicIDsManySegments):
        # if Count >5: continue
        subid = parts[parts["rlnHelicalTubeAndMicID"] == HelixMicID]
        NumSegments = subid.shape[0]

        HelixMicIDsManySegments_2.append(HelixMicID)
        NumSegmentsManySegmentsArray_2.append(NumSegments)

        # idx = subid["rlnHelicalSegmentIndex"]
        idx = subid["rlnHelicalTrackLengthAngst"]
        angles_deg = subid["rlnAngleRot"] % Period
        angles_tilt = subid["rlnAngleTilt"]

        AveAnglesTiltArray_2.append(np.mean(angles_tilt))

        # unwrapped_Rot = unwrap_to_line(idx, angles_deg, period_candidates=(180, 360), discont_frac=0.5)

        mFinal, bFinal, SSEfinal = fit_line_wrapped_by_m_known(
            idx, angles_deg, Period, mBest
        )
        # print(f"m_best={m_best:.4f}, b_best={b_best:.4f}, Smin={Smin:.4f}")
        xx, yy, _ = plot_linear_fit_simple(idx, angles_deg, mFinal, bFinal, Period)

        SminManySegmentsArray_2.append(SSEfinal)
        mManySegmentsArray_2.append(mFinal)
        bFinalArray_2.append(bFinal)

        # print(HelixMicID,NumSegments, f"{m_best:.3g}", f"{Smin:.3g}")

        Count += 1
        if 1:
            continue
        plt.figure(figsize=(15, 3))

        # plot AngleRot vs. segment index
        plt.subplot(1, 3, 1)
        # plt.scatter(idx, unwrapped_Rot['angles_unwrapped_deg'], marker="o")
        plt.scatter(idx, angles_deg, marker="o")
        plt.plot(xx, yy, "g-", lw=2, label="Fitted Line")
        plt.xlabel("Dist along filament")
        plt.title(
            "AngleRot for "
            + str(HelixMicID)
            + "\n  Smin="
            + format(Smin, ".2f")
            + "\n  m_best="
            + format(m_best, ".2f")
        )

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    now_local = now_utc.astimezone(local_tz)
    logger.info("PSU date and time: %s", now_local.strftime("%Y-%m-%d %H:%M:%S %Z"))

    #################################################################################
    #    Section 5.1          This is a plotting of S with the second metric (where mBest is fixed)

    plt.figure(figsize=(12, 3))
    plt.subplot(1, 3, 1)
    plt.hist(SminManySegmentsArray_2, 20)
    plt.title("SminManySegArray_2")
    plt.subplot(1, 3, 2)
    plt.hist(mManySegmentsArray_2, 20)
    plt.title("mManySegArray_2")
    plt.subplot(1, 3, 3)
    plt.hist(bFinalArray_2)
    plt.title("bFinalArray_2")

    HelixMicIDs1NP = np.asarray(HelixMicIDsManySegments_2)
    Num1SegNP = np.asarray(NumSegmentsManySegmentsArray_2)
    m1NP = np.asarray(mManySegmentsArray_2)
    S1NP = np.asarray(SminManySegmentsArray_2)

    mask = S1NP < 600  # boolean mask instead of np.where(...)[0]

    idx1 = np.argsort(S1NP)  # ascending
    S1NP_sorted = S1NP[idx1]  # key sorted
    m1NP_sorted = m1NP[idx1]
    HelixMicIDs1NP_sorted = HelixMicIDs1NP[idx1]

    logger.info(" This is the number of long helices = " + str(len(m1NP)))

    plt.savefig(f"{StarFileNameKey}SmandbForSecondMetric.jpg")

    logger.info("mBest: %s", mBest)

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    now_local = now_utc.astimezone(local_tz)
    logger.info("PSU date and time: %s", now_local.strftime("%Y-%m-%d %H:%M:%S %Z"))

    #################################################################################
    #################################################################################
    #    Section 6:   Order Smin and bind all outputs for long chains all into a pdf
    logger.info(
        "Section 6. Order Smin and bind all outputs for long chains all into a pdf"
    )

    #  Section 6.1: Sort then  plot all the data
    #                   Sets up pages

    # Batch 450 plots into 9 JPGs (each a 10x5 grid), then insert them into a PowerPoint.
    # You can adapt `plot_one(ax, i)` to draw your real plot for item i.
    #
    # Outputs:
    # - 9 JPEGs: /mnt/data/page_01.jpg ... /mnt/data/page_09.jpg
    # - 1 PPTX:  /mnt/data/plots_batch.pptx
    #
    # Feel free to rerun after editing `plot_one` to use your real data.

    if not Path(OutputFigsDir).is_dir():
        os.mkdir(OutputFigsDir)

    os.chdir(OutputFigsDir)

    # ---- Config ----
    TOTAL = NumLongHelices
    # TOTAL = 48
    ROWS, COLS = 6, 8
    PER_PAGE = ROWS * COLS  # 50
    OUTDIR = Path(OutputFigsDir + "/Stuff")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ---- Your per-item plotting goes here ----

    def plot_one_unwrapped(ax, j, m_best):
        Smin_Now = S1NP_sorted[j]  # key sorted
        mNP_Now = m1NP_sorted[j]
        HelixMicID = HelixMicIDs1NP_sorted[j]

        subid = parts[parts["rlnHelicalTubeAndMicID"] == HelixMicID]
        NumSegments = subid.shape[0]
        # idx = subid["rlnHelicalSegmentIndex"]
        idx = subid["rlnHelicalTrackLengthAngst"]
        angles_deg = subid["rlnAngleRot"] % Period

        # print(angles_deg)
        idxSortedOnTrack = idx.sort_values()
        angles_sorted_on_track = angles_deg.loc[idxSortedOnTrack.index]

        # unwrapped_Rot = unwrap_to_line(idx, angles_deg, period_candidates=(180, 360), discont_frac=0.5)

        mFinal, bFinal, SSEfinal = fit_line_wrapped_by_m_known(
            idxSortedOnTrack, angles_sorted_on_track, Period, m_best
        )
        # print(f"m_best={m_best:.4f}, b_best={b_best:.4f}, Smin={Smin:.4f}")
        xx, yy, _ = plot_linear_fit_simple(
            idxSortedOnTrack, angles_sorted_on_track, mFinal, bFinal, Period
        )

        AngErrors, AbsAngErrors = get_angular_errors(
            idxSortedOnTrack, angles_sorted_on_track, mFinal, bFinal, Period
        )

        # !!!!!!!!!!!

        parts.loc[
            parts["rlnHelicalTubeAndMicID"] == HelixMicID, "rlnHelicalTubeAndMicIDGood"
        ] = np.mean(AbsAngErrors)

        for J, AbsAngErrorsJ in enumerate(AbsAngErrors):
            # Get the row key (index) corresponding to position j
            JID = idxSortedOnTrack.index[J]
            # Write to that row in parts
            parts.loc[JID, "rlnHelicalTubeAndMicIDGoodSegValue"] = AbsAngErrorsJ

        ax.scatter(idx, angles_deg, marker="o")
        ax.plot(xx, yy, "g-", lw=2, label="Fitted Line")
        # ax.set_xlabel("Segment index along filament")
        ax.set_xlabel("Distance (A) along filament")
        ax.set_title(
            "HelixMicID="
            + str(HelixMicID)
            + "\n  Smin="
            + format(SSEfinal / 100, ".2f")
            + ";  m ="
            + format(mFinal, ".2f")
        )
        ax.set_ylim(0, 1.1 * Period)

        return AngErrors, AbsAngErrors

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    now_local = now_utc.astimezone(local_tz)
    logger.info("PSU date and time: %s", now_local.strftime("%Y-%m-%d %H:%M:%S %Z"))

    #################################################################################
    #   Section 6.2   Block 2
    # ---- Generate 9 pages of 50 subplots each ----
    num_pages = math.ceil(TOTAL / PER_PAGE)
    jpg_paths = []
    AngErrorsChunks = []
    for p in range(num_pages):
        # if p>0: continue
        start = p * PER_PAGE
        end = min(start + PER_PAGE, TOTAL)
        count = end - start

        fig, axes = plt.subplots(ROWS, COLS, figsize=(COLS * 2.5, ROWS * 2.0), dpi=200)
        # Flatten axes for easy indexing
        axes_flat = axes.ravel()

        # Draw actual plots
        for k in range(count):
            # if k>0: continue
            ax = axes_flat[k]
            AngErrors, AbsAngErrors = plot_one_unwrapped(ax, start + k, mBest)
            AngErrorsChunks.append(AngErrors)
            if p + k == 0:
                AngErrors0 = AngErrors

        # Hide any unused axes on the last page
        for k in range(count, PER_PAGE):
            axes_flat[k].axis("off")

        fig.tight_layout()
        page_path = OUTDIR / f"page_{p+1:02d}.jpg"
        fig.savefig(page_path, format="jpg", bbox_inches="tight")
        plt.close(fig)
        jpg_paths.append(str(page_path))

    AngErrorsAll = np.concatenate(AngErrorsChunks)

    logger.info("Page path: %s", page_path)

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    now_local = now_utc.astimezone(local_tz)
    logger.info("PSU date and time: %s", now_local.strftime("%Y-%m-%d %H:%M:%S %Z"))

    #################################################################################
    #################################################################################
    #    Section 7:  More plots toward decision functions'
    logger.info("Section 7. More plots toward decision functions")

    Smin_Now = S1NP_sorted[0]  # key sorted
    mNP_Now = m1NP_sorted[0]
    HelixMicID = HelixMicIDs1NP_sorted[0]

    subid = parts[parts["rlnHelicalTubeAndMicID"] == HelixMicID]
    NumSegments = subid.shape[0]
    idx = subid["rlnHelicalTrackLengthAngst"]
    angles_deg = subid["rlnAngleRot"] % Period

    idxSortedOnTrack = idx.sort_values()
    angles_sorted_on_track = angles_deg.loc[idxSortedOnTrack.index]

    if 0:
        mFinal, bFinal, SSEfinal = fit_line_wrapped_by_m_known(
            idx, angles_deg, Period, mBest
        )
        xx, yy, _ = plot_linear_fit_simple(idx, angles_deg, mFinal, bFinal, Period)
        AngErrors, AbsAngErrors = get_angular_errors(
            idx, angles_deg, mFinal, bFinal, Period
        )
    else:
        mFinal, bFinal, SSEfinal = fit_line_wrapped_by_m_known(
            idxSortedOnTrack, angles_sorted_on_track, Period, mBest
        )
        xx, yy, _ = plot_linear_fit_simple(
            idxSortedOnTrack, angles_sorted_on_track, mFinal, bFinal, Period
        )
        AngErrors, AbsAngErrors = get_angular_errors(
            idxSortedOnTrack, angles_sorted_on_track, mFinal, bFinal, Period
        )

    subid = parts[parts["rlnHelicalTubeAndMicID"] == HelixMicID]
    NumSegments = subid.shape[0]
    idx = subid["rlnHelicalTrackLengthAngst"]
    angles_deg = subid["rlnAngleRot"] % Period

    mFinal, bFinal, SSEfinal = fit_line_wrapped_by_m_known(
        idx, angles_deg, Period, mBest
    )
    xx, yy, _ = plot_linear_fit_simple(idx, angles_deg, mFinal, bFinal, Period)

    AngErrors, AbsAngErrors = get_angular_errors(
        idx, angles_deg, mFinal, bFinal, Period
    )

    BiggestErrorOnBestCurve = np.max(AbsAngErrors)

    # --- Plot: data + fit on top, errors below ---
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(4, 4),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},  # top 3× taller than bottom
    )
    fig.suptitle(
        "Best fit "
        + StarFileNameKey
        + " \n Worst error here "
        + f"≈ {BiggestErrorOnBestCurve:.1f}"
        + " degrees",
        fontsize=12,
    )

    # Top: observed vs fitted wrapped line
    ax1.scatter(idx, angles_deg, marker="o", s=10, label="Observed")
    ax1.plot(xx, yy, "g-", lw=2, label="Fitted line")
    ax1.set_ylabel("Angle (deg)")
    ax1.legend(loc="best")

    # Bottom: residuals (error) in red
    ax2.axhline(0, color="k", ls="--", lw=1)
    ax2.scatter(idx, AngErrors, s=10, color="r", label="Error")
    ax2.set_xlabel("Distance (Å) along filament")
    ax2.set_ylabel("Error (deg)")
    ax2.legend(loc="best")

    plt.tight_layout()
    # plt.show()
    logger.error("BiggestErrorOnBestCurve = x ≈ %.1f Å", BiggestErrorOnBestCurve)

    plt.savefig(f"{StarFileNameKey}BestFitWorstError.jpg")

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    now_local = now_utc.astimezone(local_tz)
    logger.info("PSU date and time: %s", now_local.strftime("%Y-%m-%d %H:%M:%S %Z"))

    #    Section 7.2: Curve fitting toward cdf

    # --------------------------------------------------
    # A. Inputs
    # --------------------------------------------------
    # Period =          # your period (float)
    data = np.abs(AngErrorsAll)  # 1D numpy array of length N with samples in [0, P/2]

    # --------------------------------------------------
    # B. Build a histogram estimate of f
    #    Use Freedman–Diaconis rule for bin width via bins='fd'
    # --------------------------------------------------
    counts, edges = np.histogram(data, bins="fd", range=(0, Period / 2))
    bin_centers = 0.5 * (edges[:-1] + edges[1:])  # x-values for the fit
    widths = edges[1:] - edges[:-1]

    # Optional: treat histogram as counts with Poisson noise
    sigma_counts = np.sqrt(counts + 0.5)  # avoid zero error for empty bins

    # If you want to drop empty bins (helps the fit sometimes):
    mask = counts > 0
    x_fit = bin_centers[mask]
    y_fit = counts[mask]
    y_err = sigma_counts[mask]

    # --------------------------------------------------
    # 2. Define the periodic Gaussian model G(x; A, sigma)
    # --------------------------------------------------
    n_vals = np.array([-1, 0, 1, 2], dtype=float)  # the n-range you mentioned

    def G_model(x, A, sigma):
        x = np.asarray(x)
        dx = x[None, :] - n_vals[:, None] * Period  # shape (len(n_vals), len(x))
        return A * np.exp(-0.5 * (dx / sigma) ** 2).sum(axis=0)

    def G2_model(x, APeak, sigmaPeak, ATail, sigmaTail):
        x = np.asarray(x)
        return APeak * np.exp(-0.5 * (x / sigmaPeak) ** 2) + ATail * np.exp(
            -0.5 * (x / sigmaTail) ** 2
        )

    # --------------------------------------------------
    # 3. Fit A and sigma to the histogram   BiggestErrorOnBestCurve
    # --------------------------------------------------
    A0 = float(np.max(y_fit))  # crude amplitude guess
    sigma0 = BiggestErrorOnBestCurve  #   rude width guess; adjust if you like
    ATail = float(np.min(y_fit))
    sigmaTail = Period  #  rude width guess; adjust if you like

    if 0:
        popt, pcov = curve_fit(
            G_model, x_fit, y_fit, p0=[A0, sigma0], sigma=y_err, absolute_sigma=False
        )
    else:
        popt, pcov = curve_fit(
            G2_model,
            x_fit,
            y_fit,
            p0=[A0, sigma0, ATail, sigmaTail],
            sigma=y_err,
            absolute_sigma=False,
        )

    APeak_fit, sigmaPeak_fit, ATail_fit, sigmaTail_fit = popt

    if 0:
        print("A_fit   =", A_fit)
        print("sigma_fit =", sigma_fit)
    else:
        logger.debug("APeak_fit = %s", APeak_fit)
        logger.debug("sigmaPeak_fit = %s", sigmaPeak_fit)
        logger.debug("ATail_fit = %s", ATail_fit)
        logger.debug("sigmaTail_fit = %s", sigmaTail_fit)

    # --------------------------------------------------
    # 4. Optional: evaluate fitted curve on a dense grid for plotting
    # --------------------------------------------------
    xx = np.linspace(0, Period / 2, 500)

    if 0:
        yy = G_model(xx, A_fit, sigma_fit)
    else:
        yy = G2_model(xx, APeak_fit, sigmaPeak_fit, ATail_fit, sigmaTail_fit)

    logger.info("len(xx)=%d, len(yy)=%d", len(xx), len(yy))

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    now_local = now_utc.astimezone(local_tz)
    logger.info(
        "Section 7 PSU date and time: %s", now_local.strftime("%Y-%m-%d %H:%M:%S %Z")
    )

    #       Section -1
    # --- your real analysis here ---
    # Example placeholder:
    # parts["rlnHOMConsistencyScore"] = 1.0
    # stats = {"note": "HOM placeholder ran", "nrows": len(data)}
    return parts, stats


def _read_star(path: str) -> pd.DataFrame:
    """Read a RELION STAR file into a DataFrame.

    Parameters
    ----------
    path : str
        Path to the STAR file.

    Returns
    -------
    pd.DataFrame
        The particles table.
    """
    if starfile is None:
        raise RuntimeError(
            "Reading .star requires the 'starfile' package for robustness.\n"
            "Install with: pip install starfile"
        )
    tables = starfile.read(path)
    # starfile returns a dict for multi-table files; pick common particle table
    if isinstance(tables, dict):
        # Typical keys: 'data_optics', 'data_particles'
        for key in ("data_particles", "particles", "data_"):
            if key in tables:
                return tables[key]
        # Fallback to the first table
        return next(iter(tables.values()))
    return tables  # already a DataFrame


def _write_star(df: pd.DataFrame, path: str, like: str | None = None):
    """Write a DataFrame to a STAR file.

    If ``like`` is provided, preserves the optics table from the source file.

    Parameters
    ----------
    df : pd.DataFrame
        The particles table.
    path : str
        Output STAR file path.
    like : str, optional
        Path to a source STAR file whose optics table to preserve.
    """
    if starfile is None:
        raise RuntimeError(
            "Writing .star requires the 'starfile' package.\n"
            "Install with: pip install starfile"
        )
    if like and Path(like).exists():
        src = starfile.read(like)
        if isinstance(src, dict) and "data_optics" in src:
            out = dict(src)  # shallow copy optics etc.
            out["data_particles"] = df
            starfile.write(out, path, overwrite=True)
            return
    # single-table write
    starfile.write(df, path, overwrite=True)


def add_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for the HOM_containerC command.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to attach arguments to.
    """
    parser.add_argument(
        "input_star", help="Input STAR (e.g., Refine3D/.../run_data.star)"
    )
    parser.add_argument("output_star", help="Output STAR to write")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument(
        "--param",
        type=str,
        default=None,
        help="Optional JSON or k=v pairs for extra options",
    )
    parser.add_argument(
        "--force", type=int, default=0, help="Overwrite output if exists (0/1)"
    )


def check_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> argparse.Namespace:
    """Validate HOM_containerC arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    parser : argparse.ArgumentParser
        The argument parser.

    Returns
    -------
    argparse.Namespace
        The validated arguments.
    """
    if Path(args.output_star).exists() and not args.force:
        raise HeliconFileExistsError(
            f"Refusing to overwrite existing file: {args.output_star} (use --force 1)"
        )
    return args


def main(args: argparse.Namespace) -> None:
    """Run the helical segment consistency analysis on a STAR file.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    """
    df_in = _read_star(args.input_star)
    df_out, stats = HelicalSegmentConsistency(
        df_in,
        convert_path_fn=None,
        verbose=args.verbose,
        input_star_path=args.input_star,
        output_star_path=args.output_star,
        param=args.param,
    )
    _write_star(df_out, args.output_star, like=args.input_star)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Standalone helical segment consistency on a RELION STAR (no filesystem lookups)."
    )
    add_args(p)
    args = check_args(p.parse_args(), p)
    main(args)

# python images2star.py Refine3D/job123/run_data.star out.variance.star --HelicalSegmentConsistency 1 --verbose 2
# helicon images2star run_data_with_lineage.star outTestHOM.star   --HelicalSegmentConsistency helicon.commands.HOM_container:HelicalSegmentConsistency    --verbose 2 --ignoreBadParticlePath 1
# helicon images2star run_data_Swikriti_Job408.star outTestHOM.star   --HelicalSegmentConsistency helicon.commands.HOM_container:HelicalSegmentConsistency    --verbose 2 --ignoreBadParticlePath 1 --force=1

# python HOM_container.py run_data_with_lineage.star outTestHOM.star
