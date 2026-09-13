#!/usr/bin/env python3
"""Standalone test script comparing gauss_reconstruct vs lsq_reconstruct.

Scans a twist range on the default EMPIAR-10940 2D class image and produces
an overlay plot of the score curves for both solvers.

Usage:
    python tests/test_gauss_solver_search.py [options]
"""

import argparse
import concurrent.futures
import logging
import os
import sys
import time
import tempfile
import urllib.request
from pathlib import Path

# Override PyTorch thread limit before anything else imports it.
# The gauss solver sets torch.set_num_threads(1) at import time for
# multi-worker safety, but in this single-worker test we want full
# parallelism.  We set it here, then the module-level call in
# solver_gaussian.py is a no-op (PyTorch thread setting is sticky).
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
import torch

# First attempt — undone by solver_gaussian.py's module-level
# torch.set_num_threads(1).  We restore it again after import below.
torch.set_num_threads(os.cpu_count())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import helicon
from helicon.webApps.lib import denovo3d_pipeline as pipeline

# Restore full parallelism.  solver_gaussian.py calls
# torch.set_num_threads(1) at module level for multi-worker safety,
# but in this single-worker test we want all cores.
torch.set_num_threads(os.cpu_count())

try:
    from skimage.filters import threshold_otsu
except ImportError:
    threshold_otsu = None

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_URL = (
    "https://ftp.ebi.ac.uk/empiar/world_availability/10940/data/"
    "EMPIAR/Class2D/job010/run_it020_classes.mrcs"
)
DEFAULT_RISE = 4.76  # angstroms, typical for amyloid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def download_mrcs(url: str, cache_dir: str | None = None) -> str:
    """Download an MRCs file, caching it locally. Returns local path."""
    if cache_dir is None:
        cache_dir = tempfile.mkdtemp(prefix="helicon_test_")
    local_name = url.rstrip("/").split("/")[-1] or "classes.mrcs"
    local_path = os.path.join(cache_dir, local_name)
    if os.path.exists(local_path):
        logger.info("Using cached MRCs file: %s", local_path)
        return local_path
    logger.info("Downloading %s ...", url)
    try:
        urllib.request.urlretrieve(url, local_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download {url}.\n"
            "Check internet connectivity or provide the file manually.\n"
            f"Error: {exc}"
        ) from exc
    logger.info("Downloaded to %s", local_path)
    return local_path


def preprocess_like_web_app(
    data: np.ndarray, apix: float, threshold: float | None = None
) -> np.ndarray:
    """Reproduce the web app's UI preprocessing on a raw 2D class-average image.

    The web app UI applies these steps before passing the image to the solver:
    1. Otsu threshold (or user-supplied value) — zeros background noise.
    2. No negate/transpose/flip for square images like 128×128 EMPIAR-10940.
    3. Auto-rotate to horizontal via region-props orientation.
    4. Vertical shift to centre the helix bounding-box.
    5. Vertical crop to ~2× helix diameter (rounded to multiple of 4).

    Without this preprocessing, the solver receives a raw class average with
    background noise and misaligned helix, producing degraded scores (~0.61
    instead of the expected ~0.77 for the correct twist/rise).
    """
    ny, nx = data.shape

    # Step 1 — Otsu threshold  (same as web app auto_adjust_threshold)
    if threshold is None:
        if threshold_otsu is not None:
            threshold = float(threshold_otsu(data))
        else:
            threshold = float(np.median(data[data > data.min()]))  # fallback
    data = np.clip(data, threshold, None) - threshold

    # Step 2 — estimate rotation, shift, diameter
    # Use 20% of max as threshold, matching the web app's call in
    # update_selected_image_rotation_shift_diameter() (app.py line 2133).
    rotation, shift_y, diameter = helicon.estimate_helix_rotation_center_diameter(
        data, threshold=np.max(data) * 0.2
    )

    # Step 3 — rotate to horizontal
    if rotation != 0:
        data = helicon.transform_image(image=data, rotation=rotation)

    # Step 4 — vertical shift to centre
    if shift_y != 0:
        data = helicon.transform_image(
            image=data, rotation=0, post_translation=(shift_y, 0)
        )

    # Step 5 — vertical crop to 2× diameter (rounded down to multiple of 4)
    crop_size = int(diameter * 2) // 4 * 4
    if crop_size < ny:
        data = helicon.crop_center(data, shape=(crop_size, nx))

    return data


def load_first_image(mrcs_path: str) -> tuple[np.ndarray, float]:
    """Load the first non-blank 2D class from an MRCs stack.

    Skips all-zero placeholder images (common in cryoSPARC exports).

    Returns (image, pixel_size_angstrom).
    """
    import mrcfile

    with mrcfile.open(mrcs_path) as mrc:
        data = np.array(mrc.data, dtype=np.float32)
        if data.ndim == 3:
            image = None
            for i in range(data.shape[0]):
                if data[i].std() > 1e-6:
                    image = data[i]
                    break
            if image is None:
                raise ValueError("No non-blank images found in stack")
        elif data.ndim == 2:
            image = data
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")
        apix = float(mrc.voxel_size.x)
    return image, apix


def build_task(
    data: np.ndarray,
    apix: float,
    twist: float,
    rise: float,
    algorithm: dict,
    image_index: int = 1,
) -> tuple:
    """Build a task tuple matching the web app's format.

    Returns a tuple that can be unpacked as *task into
    pipeline.process_one_task().
    """
    ny, nx = data.shape
    tube_length = nx * apix
    tube_diameter = ny * apix
    reconstruct_length = (
        3 * rise
    )  # matches web app UI default (reconstruct_length_rise=3)

    return (
        0,  # ti
        1,  # ntasks
        data,  # data
        "empiar-10940",  # imageFile
        image_index,  # imageIndex
        twist,  # twist_degree
        rise,  # rise
        (rise, rise),  # rise_range
        1,  # csym
        0.0,  # tilt
        (0.0, 0.0),  # tilt_range
        0.0,  # psi
        0.0,  # dy
        apix,  # apix2d_orig
        "",  # denoise
        0,  # low_pass
        0,  # transpose
        0,  # horizontalize
        -1,  # target_apix3d (auto-compute, matching web app)
        apix,  # target_apix2d
        -1,  # thresh_fraction
        -1,  # positive_constraint
        tube_length,  # tube_length
        tube_diameter,  # tube_diameter
        0,  # tube_diameter_inner
        reconstruct_length,  # reconstruct_length
        1,  # sym_oversample
        "nn",  # interpolation
        0,  # fsc_test
        False,  # return_3d
        "cosine",  # score_metric
        algorithm,  # algorithm
        0,  # verbose
    )


def find_peak(twists: list[float], scores: list[float | None]) -> float | None:
    """Find the twist value with the highest score (local maximum)."""
    valid = [(t, s) for t, s in zip(twists, scores) if s is not None]
    if not valid:
        return None
    twists_arr, scores_arr = zip(*valid)
    max_idx = int(np.argmax(scores_arr))
    return twists_arr[max_idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare gauss vs lsq solver twist search curves"
    )
    parser.add_argument("--lsq-only", action="store_true", help="Run only LSQ solver")
    parser.add_argument(
        "--gauss-only", action="store_true", help="Run only gauss solver"
    )
    parser.add_argument(
        "--gauss-fixed",
        action="store_true",
        help="Use fixed-grid gauss solver (amp-only)",
    )
    parser.add_argument(
        "--model",
        choices=["fb", "gauss_legacy", "gauss_fixed", "lsq"],
        default=None,
        help="Solver model to run (default: run gauss + lsq)",
    )
    parser.add_argument("--fb-n-max", type=int, default=4, help="Max Bessel order |n|")
    parser.add_argument(
        "--fb-l-max", type=int, default=4, help="Max layer-line index |l|"
    )
    parser.add_argument(
        "--fb-n-radial", type=int, default=12, help="Radial sample points per mode"
    )
    parser.add_argument("--fb-iter", type=int, default=200, help="FB solver iterations")
    parser.add_argument(
        "--fb-lr", type=float, default=0.01, help="FB solver learning rate"
    )
    parser.add_argument("--gauss-n", type=int, default=100, help="Number of Gaussians")
    parser.add_argument(
        "--gauss-iter", type=int, default=300, help="Number of iterations"
    )
    parser.add_argument("--gauss-lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument(
        "--grid-n-z",
        type=int,
        default=1,
        help="Number of z-planes for gauss_fixed grid (1=2D, 3+ = 3D)",
    )
    parser.add_argument(
        "--grid-n-radial", type=int, default=5, help="Number of radial layers"
    )
    parser.add_argument(
        "--grid-n-angular",
        type=int,
        default=12,
        help="Number of angular points per layer",
    )
    parser.add_argument(
        "--grid-sigma",
        type=float,
        default=10.0,
        help="Gaussian sigma in angstroms for gauss_fixed",
    )
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Enable position fine-tuning after amplitude-only stage",
    )
    parser.add_argument(
        "--fine-tune-iter", type=int, default=100, help="Fine-tuning iterations"
    )
    parser.add_argument("--n-restarts", type=int, default=2, help="Number of restarts")
    parser.add_argument(
        "--twist-start", type=float, default=0.5, help="Start twist (°)"
    )
    parser.add_argument("--twist-stop", type=float, default=3.0, help="Stop twist (°)")
    parser.add_argument("--twist-step", type=float, default=0.1, help="Twist step (°)")
    parser.add_argument("--rise", type=float, default=DEFAULT_RISE, help="Rise (Å)")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Parallel workers (0=use helicon.available_cpu(), default: 0)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="tests/output", help="Output directory"
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()

    # Validate twist step
    step = max(args.twist_step, 0.025)

    # Resolve output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download / cache image
    try:
        mrcs_path = download_mrcs(DEFAULT_URL)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # Load image
    try:
        image, apix = load_first_image(mrcs_path)
    except Exception as e:
        print(f"ERROR loading MRCs: {e}", file=sys.stderr)
        return 1

    ny, nx = image.shape
    print(f"Raw image: {nx}x{ny} pixels, pixel size={apix:.4f} Å")

    # Preprocess to match web app UI — Otsu threshold, auto-rotate, centre-crop
    image = preprocess_like_web_app(image, apix)
    ny, nx = image.shape
    print(f"Preprocessed image: {nx}x{ny} pixels")

    print(f"Rise: {args.rise:.2f} Å")
    print(f"Twist range: {args.twist_start}–{args.twist_stop}° step={step}°")

    # Build twist grid
    twists = list(np.arange(args.twist_start, args.twist_stop + step * 0.5, step))
    n_twists = len(twists)
    print(f"\nScanning {n_twists} twist values...\n")

    lsq_scores: list[float | None] = [None] * n_twists
    gauss_scores: list[float | None] = [None] * n_twists
    fb_scores: list[float | None] = [None] * n_twists

    if args.model == "fb":
        run_lsq = True  # LSQ runs as reference
        run_fb = True
        run_gauss = False
    elif args.model == "lsq":
        run_lsq = True
        run_fb = False
        run_gauss = False
    elif args.model is not None:
        run_lsq = not args.gauss_only
        run_fb = False
        run_gauss = True
        if args.model == "gauss_fixed":
            args.gauss_fixed = True
    else:
        run_lsq = not args.gauss_only
        run_fb = False
        run_gauss = not args.lsq_only

    # Determine parallelism
    n_workers = args.num_workers if args.num_workers > 0 else helicon.available_cpu()
    print(f"Using {n_workers} parallel workers\n")

    # Build all task tuples upfront for parallel submission
    all_tasks: list[tuple[int, str, tuple]] = (
        []
    )  # (twist_index, solver_tag, task_tuple)
    for i, twist in enumerate(twists):
        if run_lsq:
            task = build_task(image, apix, twist, args.rise, dict(model="lsq"))
            all_tasks.append((i, "lsq", task))
        if run_fb:
            algo = dict(
                model="fb",
                fb_n_max=args.fb_n_max,
                fb_l_max=args.fb_l_max,
                fb_n_radial=args.fb_n_radial,
                fb_iter=args.fb_iter,
                fb_lr=args.fb_lr,
                device="cpu",
            )
            task = build_task(image, apix, twist, args.rise, algo)
            all_tasks.append((i, "fb", task))
        if run_gauss:
            if args.gauss_fixed:
                algo = dict(
                    model="gauss_fixed",
                    gauss_iter=args.gauss_iter,
                    gauss_lr=args.gauss_lr,
                    grid_n_z=args.grid_n_z,
                    grid_n_radial=args.grid_n_radial,
                    grid_n_angular=args.grid_n_angular,
                    grid_sigma=args.grid_sigma,
                    fine_tune=args.fine_tune,
                    fine_tune_iter=args.fine_tune_iter,
                    device="cpu",
                )
            else:
                algo = dict(
                    model="gauss_legacy",
                    gauss_n=args.gauss_n,
                    gauss_iter=args.gauss_iter,
                    gauss_lr=args.gauss_lr,
                    n_restarts=args.n_restarts,
                    device="cpu",
                )
            task = build_task(image, apix, twist, args.rise, algo)
            all_tasks.append((i, "gauss_legacy", task))

    n_total = len(all_tasks)
    completed = 0
    t0 = time.time()

    n_threads = 1 if n_workers > 1 else os.cpu_count()

    def _init_worker():
        try:
            import torch

            torch.set_num_threads(n_threads)
        except ImportError:
            pass
        try:
            import numba

            numba.config.NUMBA_NUM_THREADS = n_threads
        except (ImportError, AttributeError):
            pass

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers, initializer=_init_worker
    ) as executor:
        future_map = {
            executor.submit(pipeline.process_one_task, *task_tuple): (idx, tag)
            for idx, tag, task_tuple in all_tasks
        }

        for future in concurrent.futures.as_completed(future_map):
            idx, tag = future_map[future]
            try:
                result = future.result()
                if result is not None:
                    score = float(result[0])
                    if tag == "lsq":
                        lsq_scores[idx] = score
                    elif tag == "fb":
                        fb_scores[idx] = score
                    else:
                        gauss_scores[idx] = score
                else:
                    score = None
            except Exception as e:
                score = None

            completed += 1
            twist_val = twists[idx]
            score_str = f"{score:.6f}" if score is not None else "None"
            print(
                f"  [{completed}/{n_total}] twist={twist_val:.3f}°  {tag}={score_str}"
            )

    elapsed = time.time() - t0
    print(
        f"\nFinished {n_total} tasks in {elapsed:.1f}s ({elapsed / n_total:.1f}s per task)"
    )

    # Print table
    has_fb = any(s is not None for s in fb_scores)
    if has_fb:
        print(
            f"\n{'Twist(°)':>10} {'LSQ Score':>12} {'FB Score':>12} {'Gauss Score':>12}"
        )
        print("-" * 48)
    else:
        print(f"\n{'Twist(°)':>10} {'LSQ Score':>12} {'Gauss Score':>12}")
        print("-" * 36)
    for i, twist in enumerate(twists):
        lsq_str = f"{lsq_scores[i]:.6f}" if lsq_scores[i] is not None else "N/A"
        gauss_str = f"{gauss_scores[i]:.6f}" if gauss_scores[i] is not None else "N/A"
        fb_str = f"{fb_scores[i]:.6f}" if fb_scores[i] is not None else "N/A"
        if has_fb:
            print(f"{twist:>10.3f} {lsq_str:>12} {fb_str:>12} {gauss_str:>12}")
        else:
            print(f"{twist:>10.3f} {lsq_str:>12} {gauss_str:>12}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    if run_lsq:
        valid_lsq = [(t, s) for t, s in zip(twists, lsq_scores) if s is not None]
        if valid_lsq:
            lt, ls = zip(*valid_lsq)
            ax.plot(lt, ls, "b-o", label="LSQ", markersize=4)

    if run_fb:
        valid_fb = [(t, s) for t, s in zip(twists, fb_scores) if s is not None]
        if valid_fb:
            ft, fs = zip(*valid_fb)
            ax.plot(ft, fs, "r-^", label="FB", markersize=4)

    if run_gauss:
        valid_gauss = [(t, s) for t, s in zip(twists, gauss_scores) if s is not None]
        if valid_gauss:
            gt, gs = zip(*valid_gauss)
            ax.plot(gt, gs, color="orange", marker="s", label="Gauss", markersize=4)

    has_fb = any(s is not None for s in fb_scores)
    title_suffix = " + FB" if has_fb else ""
    ax.set_xlabel("Twist (°)")
    ax.set_ylabel("Score (cosine similarity)")
    ax.set_title(f"Gauss vs LSQ{title_suffix}: twist search")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = out_dir / "gauss_vs_lsq_search_curve.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {plot_path}")

    # Peak analysis
    lsq_peak = find_peak(twists, lsq_scores) if run_lsq else None
    gauss_peak = find_peak(twists, gauss_scores) if run_gauss else None
    fb_peak = find_peak(twists, fb_scores) if run_fb else None

    print()
    if lsq_peak is not None:
        print(f"LSQ peak: {lsq_peak:.3f}°")
    if gauss_peak is not None:
        print(f"Gauss peak: {gauss_peak:.3f}°")
    if fb_peak is not None:
        print(f"FB peak: {fb_peak:.3f}°")

    if lsq_peak is not None and gauss_peak is not None:
        diff = abs(gauss_peak - lsq_peak)
        if diff <= 0.1:
            print(
                f"\nPASS: Gauss peak ({gauss_peak:.3f}°) within 0.1° of LSQ peak ({lsq_peak:.3f}°)"
            )
        else:
            print(
                f"\nFAIL: Gauss peak ({gauss_peak:.3f}°) is {diff:.3f}° away from LSQ peak ({lsq_peak:.3f}°)"
            )

    if lsq_peak is not None and fb_peak is not None:
        diff = abs(fb_peak - lsq_peak)
        threshold = 0.2
        if diff <= threshold:
            print(
                f"\nPASS: FB peak ({fb_peak:.3f}°) within {threshold}° of LSQ peak ({lsq_peak:.3f}°)"
            )
        else:
            print(
                f"\nFAIL: FB peak ({fb_peak:.3f}°) is {diff:.3f}° away from LSQ peak ({lsq_peak:.3f}°)"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
