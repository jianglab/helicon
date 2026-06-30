"""Curvelet transform utilities for image denoising and analysis."""

from __future__ import annotations

import io
import logging
from contextlib import redirect_stdout

import numpy as np

from helicon.lib.system import has_curvelet_fdct, has_curvelet_udct_gpu

try:
    import torch

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        import inspect
        import types as _types

        def _patch_curvelets_for_mps():
            import curvelets.torch._forward_transform as _ft
            import curvelets.torch._udct as _u

            _patched = 0
            for _mod in [_ft, _u]:
                for _name in dir(_mod):
                    _obj = getattr(_mod, _name)
                    if not isinstance(_obj, _types.FunctionType):
                        continue
                    _src = inspect.getsource(_obj)
                    if "dtype=torch.float64" in _src:
                        _new_src = _src.replace(
                            "dtype=torch.float64",
                            "dtype=image_frequency.real.dtype",
                        )
                        if _src != _new_src:
                            _code = compile(_new_src, inspect.getfile(_obj), "exec")
                            _ns = {}
                            exec(_code, _obj.__globals__, _ns)
                            setattr(_mod, _name, _ns[_name])
                            _patched += 1
            return _patched

        _patch_curvelets_for_mps()
except Exception:
    pass

logger = logging.getLogger(__name__)

AUTO_GPU_MIN_SIZE = 512

__all__ = [
    "curvelet_denoise_fdct",
    "curvelet_denoise_batch_fdct",
    "curvelet_denoise_udct",
    "curvelet_denoise_batch_udct",
    "curvelet_denoise_3d_udct",
    "curvelet_denoise_3d_udct_tiled",
    "curvelet_denoise_mct",
    "curvelet_denoise_batch_mct",
    "curvelet_denoise_udct_tiled",
    "curvelet_denoise_fdct_tiled",
    "curvelet_denoise_mct_tiled",
    "curvelet_denoise_3d_mct",
    "curvelet_denoise_3d_mct_tiled",
    "has_curvelet_udct_gpu",
]

_GRID_CACHE: dict[tuple[int, int, int], object] = {}

# curvepy prints "Curvepy accelerated with Cython" at import time;
# suppress it so it doesn't clutter user output.
if has_curvelet_fdct():
    with redirect_stdout(io.StringIO()):
        from curvepy.curvepy import CurveletFrequencyGrid
        from curvepy.denoise import soft_threshold


def _get_grid(shape: tuple[int, int], num_scales: int):
    key = (shape[0], shape[1], num_scales)
    if key not in _GRID_CACHE:
        _GRID_CACHE[key] = CurveletFrequencyGrid(shape[0], shape[1], scales=num_scales)
    return _GRID_CACHE[key]


_auto_num_scales_logged: set = set()
_auto_overlap_logged: set = set()


def _auto_num_scales(dim: int) -> int:
    """Pick a reasonable number of curvelet scales for a given spatial dimension."""
    result = max(2, min(6, int(np.floor(np.log2(dim)) - 2)))
    if dim not in _auto_num_scales_logged:
        _auto_num_scales_logged.add(dim)
        logger.info("\tauto-decided numScales=%d (from dimension %d px)", result, dim)
    return result


def _mad_std(values: np.ndarray) -> float:
    median = np.median(np.abs(values.ravel()))
    return float(median / 0.6745)


def _elbow_threshold(values: np.ndarray) -> float:
    s = np.sort(np.abs(values.ravel()))[::-1]
    if len(s) < 3:
        return 0.0
    n = len(s)
    x = np.arange(n, dtype=np.float64)
    y = s
    dx = x[-1] - x[0]
    dy = y[-1] - y[0]
    denom = np.hypot(dx, dy) + 1e-10
    d = np.abs(dy * x - dx * y + x[-1] * y[0] - y[-1] * x[0]) / denom
    return float(y[np.argmax(d)])


def _compute_thresholds_mad(coeffs: list, sigma_scale: float) -> list[list[float]]:
    thresholds = []
    for i, scale in enumerate(coeffs):
        scale_thresholds = []
        for wedge in scale:
            if i == 0:
                scale_thresholds.append(0.0)
            else:
                noise_std = _mad_std(wedge)
                scale_thresholds.append(sigma_scale * noise_std)
        thresholds.append(scale_thresholds)
    return thresholds


def _compute_thresholds_elbow(
    coeffs: list,
) -> list[list[float]]:
    thresholds = []
    for i, scale in enumerate(coeffs):
        scale_thresholds = []
        for wedge in scale:
            if i == 0:
                scale_thresholds.append(0.0)
            else:
                t = _elbow_threshold(wedge)
                t = max(t, 1.0 * _mad_std(wedge))
                scale_thresholds.append(t)
        thresholds.append(scale_thresholds)
    return thresholds


def _compute_thresholds_elbow_pooled(
    all_coeffs: list[list],
) -> list[list[float]]:
    n_scales = len(all_coeffs[0])
    thresholds = []
    for scale_idx in range(n_scales):
        n_wedges = len(all_coeffs[0][scale_idx])
        scale_thresholds = []
        for wedge_idx in range(n_wedges):
            if scale_idx == 0:
                scale_thresholds.append(0.0)
            else:
                pooled = np.concatenate(
                    [
                        img_coeffs[scale_idx][wedge_idx].ravel()
                        for img_coeffs in all_coeffs
                    ]
                )
                scale_thresholds.append(_elbow_threshold(pooled))
        thresholds.append(scale_thresholds)
    return thresholds


def _denoise(
    image: np.ndarray,
    grid,
    sigma_scale: float,
    thresholds: list[list[float]] | None = None,
    coeffs: list | None = None,
) -> tuple[np.ndarray, float]:
    if coeffs is None:
        coeffs = grid.forward_transform(image)
    if thresholds is None:
        thresholds = _compute_thresholds_mad(coeffs, sigma_scale)

    total = 0
    kept = 0
    new_coeffs = []
    for i, scale in enumerate(coeffs):
        new_scale = []
        for w, wedge in enumerate(scale):
            T = thresholds[i][w]
            filtered = soft_threshold(wedge, T)
            new_scale.append(filtered)
            total += wedge.size
            kept += np.count_nonzero(filtered)
        new_coeffs.append(new_scale)

    result = grid.inverse_transform(new_coeffs)
    retained_pct = 100.0 * kept / total if total > 0 else 0.0
    return result, retained_pct


def curvelet_denoise_fdct(
    image: np.ndarray,
    sigma: float | None = None,
    num_scales: int = 4,
) -> np.ndarray:
    """Denoise an image using the curvelet transform with soft thresholding.

    The image is normalized to ``[0, 1]`` before processing so that
    ``sigma`` and auto-estimated noise levels are relative to a
    consistent dynamic range. The output is rescaled to the original
    input range.

    Parameters
    ----------
    image : numpy.ndarray
        2D input image. May be non-square.
    sigma : float, optional
        When ``None`` (default) or ``<= 0``, uses automatic elbow threshold
        detection per wedge. When positive, used as a scale factor on
        per-wedge data-derived noise (MAD).
    num_scales : int, optional
        Number of curvelet scales. Defaults to 4.

    Returns
    -------
    numpy.ndarray
        Denoised image with the same shape as the input.

    Raises
    ------
    ImportError
        If ``curvepy-fdct`` is not installed.
    """
    image = np.asarray(image, dtype=np.float64)
    vmin, vmax = image.min(), image.max()
    if vmax > vmin:
        image = (image - vmin) / (vmax - vmin)

    orig_shape = image.shape
    if num_scales is None or num_scales <= 0:
        num_scales = _auto_num_scales(min(orig_shape))

    grid = _get_grid(orig_shape, num_scales)
    coeffs = grid.forward_transform(image)

    if sigma is None or sigma <= 0:
        thresholds = _compute_thresholds_elbow(coeffs)
    else:
        thresholds = _compute_thresholds_mad(coeffs, sigma)

    result, _ = _denoise(
        image, grid, sigma_scale=1, thresholds=thresholds, coeffs=coeffs
    )

    if vmax > vmin:
        result = result * (vmax - vmin) + vmin

    return result


def _compute_thresholds_mad_pooled(
    all_coeffs: list[list], sigma_scale: float
) -> list[list[float]]:
    n_scales = len(all_coeffs[0])
    thresholds = []
    for scale_idx in range(n_scales):
        n_wedges = len(all_coeffs[0][scale_idx])
        scale_thresholds = []
        for wedge_idx in range(n_wedges):
            if scale_idx == 0:
                scale_thresholds.append(0.0)
            else:
                pooled = np.concatenate(
                    [
                        img_coeffs[scale_idx][wedge_idx].ravel()
                        for img_coeffs in all_coeffs
                    ]
                )
                noise_std = _mad_std(pooled)
                scale_thresholds.append(sigma_scale * noise_std)
        thresholds.append(scale_thresholds)
    return thresholds


def curvelet_denoise_batch_fdct(
    images: list[np.ndarray],
    sigma: float | None = None,
    num_scales: int = 4,
    n_jobs: int = -1,
) -> list[np.ndarray]:
    """Denoise a batch of images in parallel using the curvelet transform.

    Parameters
    ----------
    images : list of numpy.ndarray
        2D input images. All should have the same shape for best performance
        (grids are cached per shape).
    sigma : float, optional
        When ``None`` (default) or ``<= 0``, uses automatic elbow threshold
        detection per wedge. When positive, used as a scale factor on
        per-wedge data-derived noise (MAD).
    num_scales : int, optional
        Number of curvelet scales. Defaults to 4.
    n_jobs : int, optional
        Number of parallel jobs. ``-1`` uses all CPUs. Defaults to ``-1``.

    Returns
    -------
    list of numpy.ndarray
        Denoised images in the same order as the input.

    Raises
    ------
    ImportError
        If ``curvepy-fdct`` is not installed.
    """
    from joblib import Parallel, delayed

    if n_jobs == -1 or n_jobs is None:
        from helicon.lib.system import available_cpu

        n_jobs = available_cpu()

    elbow_mode = sigma is None or sigma <= 0
    sigma_scale = sigma if (sigma is not None and sigma > 0) else 1.5
    if num_scales is None or num_scales <= 0:
        num_scales = _auto_num_scales(min(images[0].shape))

    def _forward(img):
        vmin, vmax = img.min(), img.max()
        normalized = img
        if vmax > vmin:
            normalized = (img - vmin) / (vmax - vmin)
        grid = _get_grid(img.shape, num_scales)
        coeffs = grid.forward_transform(normalized)
        return coeffs, vmin, vmax, grid

    fwd_results = Parallel(n_jobs=n_jobs)(delayed(_forward)(img) for img in images)

    all_coeffs = [r[0] for r in fwd_results]

    if elbow_mode:
        thresholds = _compute_thresholds_elbow_pooled(all_coeffs)
    else:
        thresholds = _compute_thresholds_mad_pooled(all_coeffs, sigma_scale)

    def _apply(img_info):
        coeffs, vmin, vmax, grid = img_info
        result, retained_pct = _denoise(
            None, grid, sigma_scale=1, thresholds=thresholds, coeffs=coeffs
        )
        if vmax > vmin and result is not None:
            result = result * (vmax - vmin) + vmin
        return result, retained_pct

    results = Parallel(n_jobs=n_jobs)(delayed(_apply)(info) for info in fwd_results)

    denoised = [r for r, _ in results]
    retained_pcts = [r for _, r in results]

    if retained_pcts:
        mean_pct = np.mean(retained_pcts)
        logger.info(
            "\tretained %.1f%% of curvelet coefficients (min=%.1f%%, max=%.1f%%)",
            mean_pct,
            min(retained_pcts),
            max(retained_pcts),
        )

    return denoised


# ---------------------------------------------------------------------------
# UDCT backend (curvelets package)
# ---------------------------------------------------------------------------

_UDCT_GRID_CACHE: dict[tuple, object] = {}


def _udct_compatible_shape(shape: tuple[int, ...], num_scales: int) -> tuple[int, ...]:
    factor = 2 ** (num_scales - 1)
    return tuple(int(np.ceil(d / factor) * factor) for d in shape)


def _get_device() -> torch.device:
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _gpu_dtype(device: torch.device) -> torch.dtype:
    import torch

    return torch.float32 if device.type == "mps" else torch.float64


def _move_grid_to_device(grid, device):
    """Move all tensors in a UDCT grid to *device*, converting dtypes for MPS.

    MPS does not support float64 or int64, so those are downcast to float32 /
    int32. For CUDA the original dtypes are preserved.
    """
    import torch

    def _to(item):
        if isinstance(item, torch.Tensor):
            if device.type == "mps":
                if item.is_floating_point():
                    return item.to(device=device, dtype=torch.float32)
                return item.to(device=device, dtype=torch.int32)
            return item.to(device=device)
        if isinstance(item, tuple):
            return tuple(_to(x) for x in item)
        if isinstance(item, list):
            return [_to(x) for x in item]
        return item

    grid._windows = _to(grid._windows)
    grid._decimation_ratios = _to(grid._decimation_ratios)
    return grid


def _get_udct_grid(
    shape: tuple[int, ...],
    num_scales: int,
    wedges_per_dir: int = 3,
    use_gpu: bool = False,
):
    key = (*shape, num_scales, wedges_per_dir, use_gpu)
    if key not in _UDCT_GRID_CACHE:
        if use_gpu:
            from curvelets.torch import UDCT
        else:
            from curvelets.numpy import UDCT
        _UDCT_GRID_CACHE[key] = UDCT(
            shape, num_scales=num_scales, wedges_per_direction=wedges_per_dir
        )
    return _UDCT_GRID_CACHE[key]


def _coeffs_to_numpy(tree: list, keep_as_numpy: bool = False) -> list:
    """Recursively convert a nested list of tensors to numpy arrays."""
    lst = []
    for item in tree:
        if isinstance(item, list):
            lst.append(_coeffs_to_numpy(item))
        else:
            lst.append(item.cpu().numpy() if not keep_as_numpy else item)
    return lst


def _coeffs_from_numpy(tree: list, device, dtype: torch.dtype | None = None) -> list:
    """Recursively convert a nested list of numpy arrays to torch tensors on *device*."""
    import torch

    lst = []
    for item in tree:
        if isinstance(item, list):
            lst.append(_coeffs_from_numpy(item, device, dtype=dtype))
        else:
            if dtype is not None:
                lst.append(torch.from_numpy(item).to(device=device, dtype=dtype))
            else:
                lst.append(torch.from_numpy(item).to(device=device))
    return lst


def _udct_pool_wedge(wedge: list) -> np.ndarray:
    """Concatenate all sub-bands within a UDCT wedge into a flat array."""
    return np.concatenate([sb.ravel() for sb in wedge])


def _udct_apply_threshold_to_wedge(
    wedge: list[np.ndarray], T: float
) -> list[np.ndarray]:
    """Soft-threshold all sub-bands in a wedge with a single threshold value."""
    if T <= 0:
        return [sb.copy() for sb in wedge]
    return [np.sign(sb) * np.maximum(np.abs(sb) - T, 0.0) for sb in wedge]


def _udct_compute_thresholds_mad(coeffs: list, sigma_scale: float) -> list[list[float]]:
    thresholds = []
    for i, scale in enumerate(coeffs):
        scale_thresholds = []
        for wedge in scale:
            if i == 0:
                scale_thresholds.append(0.0)
            else:
                pooled = _udct_pool_wedge(wedge)
                noise_std = _mad_std(pooled)
                scale_thresholds.append(sigma_scale * noise_std)
        thresholds.append(scale_thresholds)
    return thresholds


def _udct_compute_thresholds_elbow(
    coeffs: list,
) -> list[list[float]]:
    thresholds = []
    for i, scale in enumerate(coeffs):
        scale_thresholds = []
        for wedge in scale:
            if i == 0:
                scale_thresholds.append(0.0)
            else:
                pooled = _udct_pool_wedge(wedge)
                t = _elbow_threshold(pooled)
                t = max(t, 1.0 * _mad_std(pooled))
                scale_thresholds.append(t)
        thresholds.append(scale_thresholds)
    return thresholds


def _udct_compute_thresholds_mad_pooled(
    all_coeffs: list[list], sigma_scale: float
) -> list[list[float]]:
    n_scales = len(all_coeffs[0])
    finest_pooled = np.concatenate([_udct_pool_wedge(img[-1][0]) for img in all_coeffs])
    thresholds = []
    for scale_idx in range(n_scales):
        n_wedges = len(all_coeffs[0][scale_idx])
        scale_thresholds = []
        for wedge_idx in range(n_wedges):
            if scale_idx == 0:
                scale_thresholds.append(0.0)
            else:
                pooled = np.concatenate(
                    [_udct_pool_wedge(img[scale_idx][wedge_idx]) for img in all_coeffs]
                )
                noise_std = _mad_std(pooled)
                scale_thresholds.append(sigma_scale * noise_std)
        thresholds.append(scale_thresholds)
    return thresholds


def _udct_compute_thresholds_elbow_pooled(
    all_coeffs: list[list],
) -> list[list[float]]:
    n_scales = len(all_coeffs[0])
    thresholds = []
    for scale_idx in range(n_scales):
        n_wedges = len(all_coeffs[0][scale_idx])
        scale_thresholds = []
        for wedge_idx in range(n_wedges):
            if scale_idx == 0:
                scale_thresholds.append(0.0)
            else:
                pooled = np.concatenate(
                    [_udct_pool_wedge(img[scale_idx][wedge_idx]) for img in all_coeffs]
                )
                t = _elbow_threshold(pooled)
                scale_thresholds.append(t)
        thresholds.append(scale_thresholds)
    return thresholds


def _udct_threshold_apply_thresholds(
    coeffs: list, thresholds: list[list[float]]
) -> list:
    new_coeffs = []
    for i, scale in enumerate(coeffs):
        new_scale = []
        for w, wedge in enumerate(scale):
            T = thresholds[i][w]
            new_scale.append(_udct_apply_threshold_to_wedge(wedge, T))
        new_coeffs.append(new_scale)
    return new_coeffs


def curvelet_denoise_udct(
    image: np.ndarray,
    sigma: float | None = None,
    num_scales: int = 4,
    wedges_per_dir: int = 3,
    use_gpu: bool = False,
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    vmin, vmax = image.min(), image.max()
    if vmax > vmin:
        image = (image - vmin) / (vmax - vmin)

    orig_shape = image.shape
    if num_scales is None or num_scales <= 0:
        num_scales = _auto_num_scales(min(orig_shape))
    pad_shape = _udct_compatible_shape(orig_shape, num_scales)
    if pad_shape != orig_shape:
        pads = tuple((0, ps - ds) for ds, ps in zip(orig_shape, pad_shape))
        image = np.pad(image, pads, mode="edge")

    if use_gpu and max(orig_shape) < AUTO_GPU_MIN_SIZE:
        logger.info(
            "\timage too small (%s) for GPU benefit, falling back to CPU",
            orig_shape,
        )
        use_gpu = False

    if use_gpu:
        import torch

        device = _get_device()
        gpu_dtype = _gpu_dtype(device)
        grid = _get_udct_grid(pad_shape, num_scales, wedges_per_dir, use_gpu=True)
        _move_grid_to_device(grid, device)
        tensor = torch.from_numpy(image).to(device=device, dtype=gpu_dtype)
        coeffs = grid.forward(tensor)
        coeffs = _coeffs_to_numpy(coeffs)
    else:
        grid = _get_udct_grid(pad_shape, num_scales, wedges_per_dir, use_gpu=False)
        coeffs = grid.forward(image)

    thresholds: list[list[float]]
    if sigma is None or sigma <= 0:
        thresholds = _udct_compute_thresholds_elbow(coeffs)
    else:
        thresholds = _udct_compute_thresholds_mad(coeffs, sigma)

    new_coeffs = _udct_threshold_apply_thresholds(coeffs, thresholds)

    if use_gpu:
        import torch

        device = _get_device()
        new_coeffs = _coeffs_from_numpy(new_coeffs, device)
        result = grid.backward(new_coeffs)
        result = result.cpu().numpy()
    else:
        result = grid.backward(new_coeffs)

    if pad_shape != orig_shape:
        result = result[: orig_shape[0], : orig_shape[1]]

    if vmax > vmin:
        result = result * (vmax - vmin) + vmin

    return result


def _udct_coeff_stats(coeffs: list) -> tuple[int, int]:
    total = 0
    kept = 0
    for scale in coeffs:
        for wedge in scale:
            for sb in wedge:
                total += sb.size
                kept += np.count_nonzero(sb)
    return total, kept


def curvelet_denoise_batch_udct(
    images: list[np.ndarray],
    sigma: float | None = None,
    num_scales: int = 4,
    wedges_per_dir: int = 3,
    n_jobs: int = -1,
    use_gpu: bool = False,
) -> list[np.ndarray]:
    import helicon

    if n_jobs == -1 or n_jobs is None:
        n_jobs = helicon.available_cpu()

    elbow_mode = sigma is None or sigma <= 0
    sigma_scale = sigma if (sigma is not None and sigma > 0) else 1.5
    orig_shapes = [img.shape for img in images]
    if num_scales is None or num_scales <= 0:
        num_scales = _auto_num_scales(min(orig_shapes[0]))
    pad_shape = _udct_compatible_shape(orig_shapes[0], num_scales)

    if use_gpu and max(images[0].shape) < AUTO_GPU_MIN_SIZE:
        logger.info(
            "\tparticles too small (%s) for GPU benefit, falling back to CPU",
            images[0].shape,
        )
        use_gpu = False

    if use_gpu:
        import torch
        from joblib import Parallel, delayed

        device = _get_device()
        gpu_dtype = _gpu_dtype(device)
        grid = _get_udct_grid(pad_shape, num_scales, wedges_per_dir, use_gpu=True)
        _move_grid_to_device(grid, device)

        def _forward(img):
            vmin, vmax = img.min(), img.max()
            normalized = img
            if vmax > vmin:
                normalized = (img - vmin) / (vmax - vmin)
            if img.shape != pad_shape:
                pads = tuple((0, ps - ds) for ds, ps in zip(img.shape, pad_shape))
                normalized = np.pad(normalized, pads, mode="edge")
            tensor = torch.from_numpy(normalized).to(device=device, dtype=gpu_dtype)
            coeffs = grid.forward(tensor)
            coeffs_np = _coeffs_to_numpy(coeffs)
            return coeffs_np, vmin, vmax

        fwd_results = Parallel(n_jobs=n_jobs)(delayed(_forward)(img) for img in images)
        all_coeffs = [r[0] for r in fwd_results]

        if elbow_mode:
            thresholds = _udct_compute_thresholds_elbow_pooled(all_coeffs)
        else:
            thresholds = _udct_compute_thresholds_mad_pooled(all_coeffs, sigma_scale)

        def _apply(info):
            coeffs_np, vmin, vmax = info
            new_coeffs_np = _udct_threshold_apply_thresholds(coeffs_np, thresholds)
            total, kept = _udct_coeff_stats(new_coeffs_np)
            retained_pct = 100.0 * kept / total if total > 0 else 0.0
            new_coeffs = _coeffs_from_numpy(new_coeffs_np, device)
            result = grid.backward(new_coeffs)
            result = result.cpu().numpy()
            if result.shape != orig_shapes[0]:
                result = result[: orig_shapes[0][0], : orig_shapes[0][1]]
            if vmax > vmin:
                result = result * (vmax - vmin) + vmin
            return result, retained_pct

        results = Parallel(n_jobs=n_jobs)(delayed(_apply)(info) for info in fwd_results)
    else:
        from joblib import Parallel, delayed

        def _forward(img):
            vmin, vmax = img.min(), img.max()
            normalized = img
            if vmax > vmin:
                normalized = (img - vmin) / (vmax - vmin)
            if img.shape != pad_shape:
                pads = tuple((0, ps - ds) for ds, ps in zip(img.shape, pad_shape))
                normalized = np.pad(normalized, pads, mode="edge")
            grid = _get_udct_grid(pad_shape, num_scales, wedges_per_dir, use_gpu=False)
            coeffs = grid.forward(normalized)
            return coeffs, vmin, vmax, grid

        fwd_results = Parallel(n_jobs=n_jobs)(delayed(_forward)(img) for img in images)
        all_coeffs = [r[0] for r in fwd_results]

        if elbow_mode:
            thresholds = _udct_compute_thresholds_elbow_pooled(all_coeffs)
        else:
            thresholds = _udct_compute_thresholds_mad_pooled(all_coeffs, sigma_scale)

        def _apply(img_info):
            coeffs, vmin, vmax, grid = img_info
            new_coeffs = _udct_threshold_apply_thresholds(coeffs, thresholds)
            total, kept = _udct_coeff_stats(new_coeffs)
            retained_pct = 100.0 * kept / total if total > 0 else 0.0
            result = grid.backward(new_coeffs)
            if result.shape != orig_shapes[0]:
                result = result[: orig_shapes[0][0], : orig_shapes[0][1]]
            if vmax > vmin and result is not None:
                result = result * (vmax - vmin) + vmin
            return result, retained_pct

        results = Parallel(n_jobs=n_jobs)(delayed(_apply)(info) for info in fwd_results)

    denoised = [r for r, _ in results]
    retained_pcts = [r for _, r in results]

    if retained_pcts:
        mean_pct = np.mean(retained_pcts)
        logger.info(
            "\tretained %.1f%% of curvelet coefficients (min=%.1f%%, max=%.1f%%)",
            mean_pct,
            min(retained_pcts),
            max(retained_pcts),
        )

    return denoised


# ---------------------------------------------------------------------------
# MCT backend (Monogenic Curvelet Transform)
# ---------------------------------------------------------------------------

# The Monogenic Curvelet Transform (MCT) processes each curvelet sub-band
# through the Riesz transform to extract local amplitude, then applies
# soft thresholding to the amplitude only — preserving phase and orientation
# for cleaner edge reconstruction.


def _riesz_transform_x(image: np.ndarray) -> np.ndarray:
    h, w = image.shape
    u = np.fft.fftfreq(w)[None, :]
    v = np.fft.fftfreq(h)[:, None]
    radius = np.sqrt(u**2 + v**2)
    radius[0, 0] = 1.0
    kernel = -1j * u / radius
    kernel[0, 0] = 0.0
    return np.fft.ifft2(np.fft.fft2(image) * kernel).real


def _riesz_transform_y(image: np.ndarray) -> np.ndarray:
    h, w = image.shape
    u = np.fft.fftfreq(w)[None, :]
    v = np.fft.fftfreq(h)[:, None]
    radius = np.sqrt(u**2 + v**2)
    radius[0, 0] = 1.0
    kernel = -1j * v / radius
    kernel[0, 0] = 0.0
    return np.fft.ifft2(np.fft.fft2(image) * kernel).real


def _mct_denoise_subband(q0: np.ndarray, T: float) -> np.ndarray:
    if T <= 0:
        return q0.copy()
    q1 = _riesz_transform_x(q0)
    q2 = _riesz_transform_y(q0)
    amplitude = np.sqrt(q0**2 + q1**2 + q2**2 + 1e-30)
    thresholded = np.maximum(amplitude - T, 0.0)
    atten = np.divide(
        thresholded, amplitude, out=np.ones_like(amplitude), where=amplitude > 1e-30
    )
    return q0 * atten


def _mct_apply_threshold_to_wedge(
    wedge: list[np.ndarray], T: float
) -> list[np.ndarray]:
    return [_mct_denoise_subband(sb, T) for sb in wedge]


def _mct_threshold_apply_thresholds(
    coeffs: list, thresholds: list[list[float]]
) -> list:
    new_coeffs = []
    for i, scale in enumerate(coeffs):
        new_scale = []
        for w, wedge in enumerate(scale):
            T = thresholds[i][w]
            new_scale.append(_mct_apply_threshold_to_wedge(wedge, T))
        new_coeffs.append(new_scale)
    return new_coeffs


def curvelet_denoise_mct(
    image: np.ndarray,
    sigma: float | None = None,
    num_scales: int = 4,
    wedges_per_dir: int = 3,
) -> np.ndarray:
    """Denoise an image using the Monogenic Curvelet Transform (MCT).

    The MCT approach applies the Riesz transform to each curvelet sub-band
    to extract monogenic amplitude, phase, and orientation. Thresholding is
    applied to the amplitude only, preserving the phase (edge locations)
    for cleaner reconstruction.

    Parameters
    ----------
    image : numpy.ndarray
        2D input image.
    sigma : float, optional
        When ``None`` (default) or ``<= 0``, uses automatic elbow threshold
        detection per wedge. When positive, used as a scale factor on
        per-wedge data-derived noise (MAD).
    num_scales : int, optional
        Number of curvelet scales. Defaults to 4.
    wedges_per_dir : int, optional
        Number of wedges per direction (for UDCT). Defaults to 3.

    Returns
    -------
    numpy.ndarray
        Denoised image with the same shape as the input.

    Raises
    ------
    ImportError
        If the ``curvelets`` package is not installed.
    """
    image = np.asarray(image, dtype=np.float64)
    vmin, vmax = image.min(), image.max()
    if vmax > vmin:
        image = (image - vmin) / (vmax - vmin)

    orig_shape = image.shape
    if num_scales is None or num_scales <= 0:
        num_scales = _auto_num_scales(min(orig_shape))
    pad_shape = _udct_compatible_shape(orig_shape, num_scales)
    if pad_shape != orig_shape:
        pads = tuple((0, ps - ds) for ds, ps in zip(orig_shape, pad_shape))
        image = np.pad(image, pads, mode="edge")

    grid = _get_udct_grid(pad_shape, num_scales, wedges_per_dir, use_gpu=False)
    coeffs = grid.forward(image)

    if sigma is None or sigma <= 0:
        thresholds = _udct_compute_thresholds_elbow(coeffs)
    else:
        thresholds = _udct_compute_thresholds_mad(coeffs, sigma)

    new_coeffs = _mct_threshold_apply_thresholds(coeffs, thresholds)
    result = grid.backward(new_coeffs)

    if pad_shape != orig_shape:
        result = result[: orig_shape[0], : orig_shape[1]]

    if vmax > vmin:
        result = result * (vmax - vmin) + vmin

    return result


def curvelet_denoise_batch_mct(
    images: list[np.ndarray],
    sigma: float | None = None,
    num_scales: int = 4,
    wedges_per_dir: int = 3,
    n_jobs: int = -1,
) -> list[np.ndarray]:
    """Denoise a batch of images in parallel using the MCT approach.

    Parameters
    ----------
    images : list of numpy.ndarray
        2D input images. All should have the same shape.
    sigma : float, optional
        When ``None`` (default) or ``<= 0``, uses automatic elbow threshold
        detection per wedge. When positive, used as a scale factor on
        per-wedge data-derived noise (MAD).
    num_scales : int, optional
        Number of curvelet scales. Defaults to 4.
    wedges_per_dir : int, optional
        Number of wedges per direction. Defaults to 3.
    n_jobs : int, optional
        Number of parallel jobs. ``-1`` uses all CPUs. Defaults to ``-1``.

    Returns
    -------
    list of numpy.ndarray
        Denoised images in the same order as the input.
    """
    import helicon
    from joblib import Parallel, delayed

    if n_jobs == -1 or n_jobs is None:
        n_jobs = helicon.available_cpu()

    elbow_mode = sigma is None or sigma <= 0
    sigma_scale = sigma if (sigma is not None and sigma > 0) else 1.5
    orig_shapes = [img.shape for img in images]
    if num_scales is None or num_scales <= 0:
        num_scales = _auto_num_scales(min(orig_shapes[0]))
    pad_shape = _udct_compatible_shape(orig_shapes[0], num_scales)

    def _forward(img):
        vmin, vmax = img.min(), img.max()
        normalized = img
        if vmax > vmin:
            normalized = (img - vmin) / (vmax - vmin)
        if img.shape != pad_shape:
            pads = tuple((0, ps - ds) for ds, ps in zip(img.shape, pad_shape))
            normalized = np.pad(normalized, pads, mode="edge")
        grid = _get_udct_grid(pad_shape, num_scales, wedges_per_dir, use_gpu=False)
        coeffs = grid.forward(normalized)
        return coeffs, vmin, vmax, grid

    fwd_results = Parallel(n_jobs=n_jobs)(delayed(_forward)(img) for img in images)
    all_coeffs = [r[0] for r in fwd_results]

    if elbow_mode:
        thresholds = _udct_compute_thresholds_elbow_pooled(all_coeffs)
    else:
        thresholds = _udct_compute_thresholds_mad_pooled(all_coeffs, sigma_scale)

    def _apply(img_info):
        coeffs, vmin, vmax, grid = img_info
        new_coeffs = _mct_threshold_apply_thresholds(coeffs, thresholds)
        total, kept = _udct_coeff_stats(new_coeffs)
        retained_pct = 100.0 * kept / total if total > 0 else 0.0
        result = grid.backward(new_coeffs)
        if result.shape != orig_shapes[0]:
            result = result[: orig_shapes[0][0], : orig_shapes[0][1]]
        if vmax > vmin and result is not None:
            result = result * (vmax - vmin) + vmin
        return result, retained_pct

    results = Parallel(n_jobs=n_jobs)(delayed(_apply)(info) for info in fwd_results)

    denoised = [r for r, _ in results]
    retained_pcts = [r for _, r in results]

    if retained_pcts:
        mean_pct = np.mean(retained_pcts)
        logger.info(
            "\tretained %.1f%% of curvelet coefficients (min=%.1f%%, max=%.1f%%)",
            mean_pct,
            min(retained_pcts),
            max(retained_pcts),
        )

    return denoised


# ---------------------------------------------------------------------------
# 2D tiled denoising (parallel within a single large image)
# ---------------------------------------------------------------------------


def _tile_indices_2d(
    shape: tuple[int, int], tile_size: int, overlap: int
) -> list[tuple[slice, slice]]:
    """Generate overlapping 2D tile slices."""
    ny, nx = shape
    stride = max(1, tile_size - overlap)
    y_starts = list(range(0, ny - tile_size + 1, stride)) if tile_size < ny else [0]
    if y_starts[-1] + tile_size < ny:
        y_starts.append(ny - tile_size)
    x_starts = list(range(0, nx - tile_size + 1, stride)) if tile_size < nx else [0]
    if x_starts[-1] + tile_size < nx:
        x_starts.append(nx - tile_size)
    tiles = []
    for y in y_starts:
        y_end = min(y + tile_size, ny)
        for x in x_starts:
            x_end = min(x + tile_size, nx)
            tiles.append((slice(y, y_end), slice(x, x_end)))
    return tiles


def _cosine_taper_2d(shape: tuple[int, int], overlap: int) -> np.ndarray:
    """Generate a 2D cosine taper for feathering tile seams."""
    ny, nx = shape
    wy = np.ones(ny)
    wx = np.ones(nx)
    if overlap > 0 and ny > 1 and nx > 1:
        oy = min(overlap, ny)
        ox = min(overlap, nx)
        ramp_y = np.sin(np.linspace(0, np.pi / 2, oy)) ** 2
        wy[:oy] = ramp_y
        wy[-oy:] = ramp_y[::-1]
        ramp_x = np.sin(np.linspace(0, np.pi / 2, ox)) ** 2
        wx[:ox] = ramp_x
        wx[-ox:] = ramp_x[::-1]
    return wy[:, None] * wx[None, :]


def _denoise_2d_tiled(
    image: np.ndarray,
    tile_denoiser,
    tile_size: int,
    overlap: int,
    n_jobs: int,
) -> tuple[np.ndarray, list[float]]:
    """Split *image* into overlapping tiles, denoise each, blend with cosine taper.

    *tile_denoiser* may return either an ndarray (denoised tile) or a tuple
    ``(ndarray, retained_pct: float)``. Returned second element is the list of
    retained percentages.
    """
    from joblib import Parallel, delayed

    tiles = _tile_indices_2d(image.shape, tile_size, overlap)

    def _process(slc_y, slc_x):
        tile = image[slc_y, slc_x]
        ret = tile_denoiser(tile)
        if isinstance(ret, tuple):
            denoised, retained_pct = ret
        else:
            denoised = ret
            retained_pct = None
        weight = _cosine_taper_2d(tile.shape, overlap)
        return denoised, weight, slc_y, slc_x, retained_pct

    if n_jobs == 1 or len(tiles) == 1:
        results = [_process(slc_y, slc_x) for slc_y, slc_x in tiles]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_process)(slc_y, slc_x) for slc_y, slc_x in tiles
        )

    output = np.zeros_like(image)
    weights = np.zeros_like(image)
    retained_pcts: list[float] = []
    for item in results:
        denoised, weight, slc_y, slc_x, retained_pct = item
        output[slc_y, slc_x] += denoised * weight
        weights[slc_y, slc_x] += weight
        if retained_pct is not None:
            retained_pcts.append(retained_pct)

    mask = weights > 1e-10
    output[mask] /= weights[mask]
    return output, retained_pcts


def curvelet_denoise_udct_tiled(
    image: np.ndarray,
    sigma: float | None = None,
    num_scales: int = 3,
    wedges_per_dir: int = 3,
    tile_size: int = 256,
    overlap: int = 64,
    n_jobs: int = -1,
    use_gpu: bool = False,
) -> np.ndarray:
    """Denoise a large image via overlapping tiles using UDCT.

    Parameters
    ----------
    image : numpy.ndarray
        2D input image.
    sigma : float, optional
        MAD noise scaling or ``None`` for elbow detection.
    num_scales : int, optional
        Number of curvelet scales per tile. When ``None`` or ``<= 0``,
        auto-selected from tile size. Defaults to 3.
    wedges_per_dir : int, optional
        Number of wedges per direction. Defaults to 3.
    tile_size : int, optional
        Edge length of each square tile in pixels. Defaults to 256.
    overlap : int, optional
        Overlap between adjacent tiles in pixels. Larger values reduce visible
        seams. Defaults to 64.
    n_jobs : int, optional
        Number of parallel workers. ``-1`` uses all CPUs. Defaults to ``-1``.
    use_gpu : bool, optional
        Whether to use GPU acceleration for each tile. Defaults to False.

    Returns
    -------
    numpy.ndarray
        Denoised image of the same shape as the input.
    """
    import helicon

    if n_jobs == -1 or n_jobs is None:
        n_jobs = helicon.available_cpu()

    image = np.asarray(image, dtype=np.float64)
    vmin, vmax = image.min(), image.max()
    if vmax > vmin:
        image = (image - vmin) / (vmax - vmin)

    if num_scales is None or num_scales <= 0:
        num_scales = _auto_num_scales(tile_size)
    overlap = max(overlap, int(2 * 2 ** (num_scales - 1)))
    if (num_scales, overlap) not in _auto_overlap_logged:
        _auto_overlap_logged.add((num_scales, overlap))
        logger.info(
            "\toverlap auto-set to %d px (from numScales=%d)", overlap, num_scales
        )

    grid_cache: dict = {}

    def _tile_denoiser(tile):
        pad_shape = _udct_compatible_shape(tile.shape, num_scales)
        if pad_shape != tile.shape:
            pads = tuple((0, ps - ds) for ds, ps in zip(tile.shape, pad_shape))
            tile_padded = np.pad(tile, pads, mode="edge")
        else:
            tile_padded = tile

        key = (*pad_shape, num_scales, wedges_per_dir, use_gpu)
        if key not in grid_cache:
            grid_cache[key] = _get_udct_grid(
                pad_shape, num_scales, wedges_per_dir, use_gpu
            )
            if use_gpu:
                _move_grid_to_device(grid_cache[key], _get_device())

        grid = grid_cache[key]

        if use_gpu:
            import torch

            device = _get_device()
            gpu_dtype = _gpu_dtype(device)
            tensor = torch.from_numpy(tile_padded.copy()).to(
                device=device, dtype=gpu_dtype
            )
            coeffs = grid.forward(tensor)
            coeffs = _coeffs_to_numpy(coeffs)
        else:
            coeffs = grid.forward(tile_padded)

        if sigma is None or sigma <= 0:
            thresholds = _udct_compute_thresholds_elbow(coeffs)
        else:
            thresholds = _udct_compute_thresholds_mad(coeffs, sigma)

        new_coeffs = _udct_threshold_apply_thresholds(coeffs, thresholds)

        total, kept = _udct_coeff_stats(new_coeffs)
        retained_pct = 100.0 * kept / total if total > 0 else 0.0

        if use_gpu:
            import torch

            device = _get_device()
            new_coeffs = _coeffs_from_numpy(new_coeffs, device)
            result = grid.backward(new_coeffs).cpu().numpy()
        else:
            result = grid.backward(new_coeffs)

        if result.shape != tile.shape:
            result = result[: tile.shape[0], : tile.shape[1]]
        return result, retained_pct

    result, retained_pcts = _denoise_2d_tiled(
        image, _tile_denoiser, tile_size, overlap, n_jobs
    )
    if retained_pcts:
        logger.info(
            "\tretained %.1f%% of curvelet coefficients per tile (min=%.1f%%, max=%.1f%%)",
            np.mean(retained_pcts),
            min(retained_pcts),
            max(retained_pcts),
        )

    if vmax > vmin:
        result = result * (vmax - vmin) + vmin
    return result


def curvelet_denoise_fdct_tiled(
    image: np.ndarray,
    sigma: float | None = None,
    num_scales: int = 3,
    tile_size: int = 256,
    overlap: int = 64,
    n_jobs: int = -1,
) -> np.ndarray:
    """Denoise a large image via overlapping tiles using FDCT.

    Parameters
    ----------
    image : numpy.ndarray
        2D input image.
    sigma : float, optional
        MAD noise scaling or ``None`` for elbow detection.
    num_scales : int, optional
        Number of curvelet scales per tile. Defaults to 3.
    tile_size : int, optional
        Edge length of each square tile. Defaults to 256.
    overlap : int, optional
        Overlap between adjacent tiles in pixels. Larger values reduce visible
        seams. Defaults to 64.
    n_jobs : int, optional
        Number of parallel workers. ``-1`` uses all CPUs. Defaults to ``-1``.

    Returns
    -------
    numpy.ndarray
        Denoised image of the same shape as the input.
    """
    import helicon

    if n_jobs == -1 or n_jobs is None:
        n_jobs = helicon.available_cpu()

    image = np.asarray(image, dtype=np.float64)
    vmin, vmax = image.min(), image.max()
    if vmax > vmin:
        image = (image - vmin) / (vmax - vmin)

    if num_scales is None or num_scales <= 0:
        num_scales = _auto_num_scales(tile_size)
    overlap = max(overlap, int(2 * 2 ** (num_scales - 1)))
    if (num_scales, overlap) not in _auto_overlap_logged:
        _auto_overlap_logged.add((num_scales, overlap))
        logger.info(
            "\toverlap auto-set to %d px (from numScales=%d)", overlap, num_scales
        )

    def _tile_denoiser(tile):
        grid = _get_grid(tile.shape, num_scales)
        coeffs = grid.forward_transform(tile)

        if sigma is None or sigma <= 0:
            thresholds = _compute_thresholds_elbow(coeffs)
        else:
            thresholds = _compute_thresholds_mad(coeffs, sigma)

        result, _ = _denoise(
            None, grid, sigma_scale=1, thresholds=thresholds, coeffs=coeffs
        )
        return result

    result, _retained_pcts = _denoise_2d_tiled(
        image, _tile_denoiser, tile_size, overlap, n_jobs
    )

    if vmax > vmin:
        result = result * (vmax - vmin) + vmin
    return result


def curvelet_denoise_mct_tiled(
    image: np.ndarray,
    sigma: float | None = None,
    num_scales: int = 3,
    wedges_per_dir: int = 3,
    tile_size: int = 256,
    overlap: int = 64,
    n_jobs: int = -1,
) -> np.ndarray:
    """Denoise a large image via overlapping tiles using MCT.

    Parameters
    ----------
    image : numpy.ndarray
        2D input image.
    sigma : float, optional
        MAD noise scaling or ``None`` for elbow detection.
    num_scales : int, optional
        Number of curvelet scales per tile. Defaults to 3.
    wedges_per_dir : int, optional
        Number of wedges per direction. Defaults to 3.
    tile_size : int, optional
        Edge length of each square tile in pixels. Defaults to 256.
    overlap : int, optional
        Overlap between adjacent tiles in pixels. Larger values reduce visible
        seams. Defaults to 64.
    n_jobs : int, optional
        Number of parallel workers. ``-1`` uses all CPUs. Defaults to ``-1``.

    Returns
    -------
    numpy.ndarray
        Denoised image of the same shape as the input.
    """
    import helicon

    if n_jobs == -1 or n_jobs is None:
        n_jobs = helicon.available_cpu()

    image = np.asarray(image, dtype=np.float64)
    vmin, vmax = image.min(), image.max()
    if vmax > vmin:
        image = (image - vmin) / (vmax - vmin)

    if num_scales is None or num_scales <= 0:
        num_scales = _auto_num_scales(tile_size)
    overlap = max(overlap, int(2 * 2 ** (num_scales - 1)))
    if (num_scales, overlap) not in _auto_overlap_logged:
        _auto_overlap_logged.add((num_scales, overlap))
        logger.info(
            "\toverlap auto-set to %d px (from numScales=%d)", overlap, num_scales
        )

    grid_cache: dict = {}

    def _tile_denoiser(tile):
        pad_shape = _udct_compatible_shape(tile.shape, num_scales)
        if pad_shape != tile.shape:
            pads = tuple((0, ps - ds) for ds, ps in zip(tile.shape, pad_shape))
            tile_padded = np.pad(tile, pads, mode="edge")
        else:
            tile_padded = tile

        key = (*pad_shape, num_scales, wedges_per_dir, False)
        if key not in grid_cache:
            grid_cache[key] = _get_udct_grid(
                pad_shape, num_scales, wedges_per_dir, use_gpu=False
            )

        grid = grid_cache[key]
        coeffs = grid.forward(tile_padded)

        if sigma is None or sigma <= 0:
            thresholds = _udct_compute_thresholds_elbow(coeffs)
        else:
            thresholds = _udct_compute_thresholds_mad(coeffs, sigma)

        new_coeffs = _mct_threshold_apply_thresholds(coeffs, thresholds)

        total, kept = _udct_coeff_stats(new_coeffs)
        retained_pct = 100.0 * kept / total if total > 0 else 0.0

        result = grid.backward(new_coeffs)

        if result.shape != tile.shape:
            result = result[: tile.shape[0], : tile.shape[1]]
        return result, retained_pct

    result, retained_pcts = _denoise_2d_tiled(
        image, _tile_denoiser, tile_size, overlap, n_jobs
    )
    if retained_pcts:
        logger.info(
            "\tretained %.1f%% of curvelet coefficients per tile (min=%.1f%%, max=%.1f%%)",
            np.mean(retained_pcts),
            min(retained_pcts),
            max(retained_pcts),
        )

    if vmax > vmin:
        result = result * (vmax - vmin) + vmin
    return result


# ---------------------------------------------------------------------------
# UDCT 3D backend
# ---------------------------------------------------------------------------


def _udct_threshold_subband_3d(
    subband: np.ndarray,
    sigma: float | None = None,
    scale_idx: int = 0,
) -> np.ndarray:
    T = 0.0
    if sigma is None or sigma <= 0:
        if scale_idx > 0:
            T = _elbow_threshold(subband)
            T = max(T, 1.0 * _mad_std(subband))
    else:
        if scale_idx > 0:
            noise_std = _mad_std(subband)
            T = sigma * noise_std
    return np.sign(subband) * np.maximum(np.abs(subband) - T, 0.0)


def curvelet_denoise_3d_udct(
    volume: np.ndarray,
    sigma: float | None = None,
    num_scales: int = 4,
    wedges_per_dir: int = 3,
    use_gpu: bool = False,
) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float64)
    vmin, vmax = volume.min(), volume.max()
    if vmax > vmin:
        volume = (volume - vmin) / (vmax - vmin)

    orig_shape = volume.shape
    if num_scales is None or num_scales <= 0:
        num_scales = _auto_num_scales(min(orig_shape))
    pad_shape = _udct_compatible_shape(orig_shape, num_scales)
    if pad_shape != orig_shape:
        pads = tuple((0, ps - ds) for ds, ps in zip(orig_shape, pad_shape))
        volume = np.pad(volume, pads, mode="edge")

    if use_gpu:
        import torch

        device = _get_device()
        gpu_dtype = _gpu_dtype(device)
        grid = _get_udct_grid(pad_shape, num_scales, wedges_per_dir, use_gpu=True)
        _move_grid_to_device(grid, device)
        tensor = torch.from_numpy(volume).to(device=device, dtype=gpu_dtype)
        coeffs = grid.forward(tensor)
        coeffs = _coeffs_to_numpy(coeffs)
    else:
        grid = _get_udct_grid(pad_shape, num_scales, wedges_per_dir, use_gpu=False)
        coeffs = grid.forward(volume)

    new_coeffs = []
    for scale_idx, scale in enumerate(coeffs):
        new_scale = []
        for direction in scale:
            new_direction = []
            for subband in direction:
                new_direction.append(
                    _udct_threshold_subband_3d(
                        subband,
                        sigma=sigma,
                        scale_idx=scale_idx,
                    )
                )
            new_scale.append(new_direction)
        new_coeffs.append(new_scale)

    total, kept = _udct_coeff_stats(new_coeffs)
    retained_pct = 100.0 * kept / total if total > 0 else 0.0
    logger.info("\tretained %.1f%% of curvelet coefficients", retained_pct)

    if use_gpu:
        import torch

        device = _get_device()
        new_coeffs = _coeffs_from_numpy(new_coeffs, device)
        result = grid.backward(new_coeffs)
        result = result.cpu().numpy()
    else:
        result = grid.backward(new_coeffs)

    if pad_shape != orig_shape:
        result = result[: orig_shape[0], : orig_shape[1], : orig_shape[2]]

    if vmax > vmin:
        result = result * (vmax - vmin) + vmin

    return result


def _curvelet_denoise_3d_udct_with_grid(
    volume: np.ndarray,
    grid,
    sigma: float | None = None,
    device=None,
    gpu_dtype=None,
) -> np.ndarray:
    """Perform 3D UDCT denoising using a pre-computed grid.

    Parameters
    ----------
    volume : np.ndarray
        3D volume to denoise (normalized to [0, 1]).
    grid : UDCT
        Pre-computed UDCT grid.
    sigma : float | None, optional
        Standard deviation parameter for MAD threshold.
    device : torch.device, optional
        GPU device if using GPU.
    gpu_dtype : torch.dtype, optional
        GPU dtype if using GPU.

    Returns
    -------
    tuple[np.ndarray, float]
        Denoised volume (still normalized) and percentage of retained coefficients.
    """
    if device is not None:
        import torch

        tensor = torch.from_numpy(np.ascontiguousarray(volume)).to(
            device=device, dtype=gpu_dtype
        )
        coeffs = grid.forward(tensor)
        coeffs = _coeffs_to_numpy(coeffs)
    else:
        coeffs = grid.forward(volume)

    new_coeffs = []
    for scale_idx, scale in enumerate(coeffs):
        new_scale = []
        for direction in scale:
            new_direction = []
            for subband in direction:
                new_direction.append(
                    _udct_threshold_subband_3d(
                        subband,
                        sigma=sigma,
                        scale_idx=scale_idx,
                    )
                )
            new_scale.append(new_direction)
        new_coeffs.append(new_scale)

    total, kept = _udct_coeff_stats(new_coeffs)
    retained_pct = 100.0 * kept / total if total > 0 else 0.0

    if device is not None:
        import torch

        new_coeffs = _coeffs_from_numpy(new_coeffs, device)
        result = grid.backward(new_coeffs)
        result = result.cpu().numpy()
    else:
        result = grid.backward(new_coeffs)

    return result, retained_pct


def _get_tile_indices(
    shape: tuple[int, ...], tile_size: tuple[int, ...], overlap: int
) -> list[tuple]:
    """Generate tile indices for a 3D volume with overlap.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the 3D volume (nx, ny, nz).
    tile_size : tuple[int, ...]
        Size of each tile (tx, ty, tz) including overlap.
    overlap : int
        Overlap size in voxels between adjacent tiles.

    Returns
    -------
    list[tuple]
        List of tuples (z_start, y_start, x_start, z_slice, y_slice, x_slice).
    """
    nx, ny, nz = shape
    tx, ty, tz = tile_size

    stride_z = tz if overlap >= tz else max(1, tz - overlap)
    stride_y = ty if overlap >= ty else max(1, ty - overlap)
    stride_x = tx if overlap >= tx else max(1, tx - overlap)

    tiles = []
    z_start = 0
    while z_start < nz:
        z_end = min(z_start + tz, nz)
        y_start = 0
        while y_start < ny:
            y_end = min(y_start + ty, ny)
            x_start = 0
            while x_start < nx:
                x_end = min(x_start + tx, nx)
                tiles.append(
                    (
                        z_start,
                        y_start,
                        x_start,
                        slice(z_start, z_end),
                        slice(y_start, y_end),
                        slice(x_start, x_end),
                    )
                )
                x_start += stride_x
            y_start += stride_y
        z_start += stride_z

    return tiles


def _generate_cosine_taper(shape: tuple[int, ...], overlap: int) -> np.ndarray:
    """Generate a 3D cosine taper array for feathering tile edges.

    Returns weight = 1 everywhere. Tiling with overlap > 0 works correctly
    because at interior seams, both tiles contribute weight 1, and the
    average of the two denoised tiles is the correct blended result.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the volume (nx, ny, nz).
    overlap : int
        Number of voxels from edge to apply taper (ignored).

    Returns
    -------
    np.ndarray
        3D array with weight 1.0 everywhere.
    """
    return np.ones(shape, dtype=np.float64)


def curvelet_denoise_3d_udct_tiled(
    vol: np.ndarray,
    sigma: float | None = None,
    num_scales: int = 3,
    wedges_per_dir: int = 3,
    tile_size: tuple[int, ...] | None = None,
    overlap: int = 32,
    use_gpu: bool = False,
    n_jobs: int | None = None,
    outdir: str | None = None,
) -> np.ndarray:
    """Denoise a 3D volume using UDCT curvelet transform with tiling.

    Processes large volumes in overlapping tiles to manage memory usage. Each tile
    is denoised independently and combined with cosine-tapered feathering.

    Parameters
    ----------
    vol : np.ndarray
        3D volume to denoise.
    sigma : float | None, optional
        Standard deviation parameter for MAD threshold. If None, uses elbow detection.
    num_scales : int, optional
        Number of curvelet scales. Default is 3 (lower than the default 4 for
        `curvelet_denoise_3d_udct` to reduce memory usage).
    wedges_per_dir : int, optional
        Number of wedges per direction. Must be ≥ 3. Default is 3.
    tile_size : tuple[int, ...] | None, optional
        Size of each tile as (nx, ny, nz). If None, auto-detects based on available
        memory to keep peak usage under 80% of available RAM. Default is None.
    overlap : int, optional
        Number of overlapping voxels between adjacent tiles for feathering.
        Default is 32.
    use_gpu : bool, optional
        Whether to use GPU acceleration. Default is False.
    n_jobs : int | None, optional
        Number of parallel workers. If None, uses all available CPUs.
        Default is None.
    outdir : str | None, optional
        Directory for memory-mapped output. If provided, writes output as a memmap
        file that can be used directly without loading into memory.

    Returns
    -------
    np.ndarray
        Denoised volume. If `outdir` is provided, returns a memory-mapped array.

    Raises
    ------
    ValueError
        If input is not 3D or parameters are invalid.
    RuntimeError
        If GPU requested but curvelets package unavailable.

    Notes
    -----
    - Peak memory per tile is roughly `tile_size³ × 300×` the voxel size (due to
      coefficient storage and intermediate arrays)
    - For a 4000³ volume with 256³ tiles: expect ~2-3 GB peak per tile with 3 scales
    - A cosine taper is applied to each tile to smooth seams between tiles
    """
    import os
    from typing import Any

    vol = np.asarray(vol, dtype=np.float64)
    if vol.ndim != 3:
        raise ValueError(f"Input must be 3D, got {vol.ndim}D")
    if num_scales is None or num_scales <= 0:
        num_scales = _auto_num_scales(min(vol.shape))
    if wedges_per_dir < 3:
        raise ValueError(f"wedges_per_dir must be ≥ 3, got {wedges_per_dir}")
    if overlap < 0:
        raise ValueError(f"overlap must be ≥ 0, got {overlap}")

    if use_gpu and not has_curvelet_udct_gpu():
        raise RuntimeError("GPU acceleration requires optional 'curvelets' package")

    vmin, vmax = vol.min(), vol.max()
    if vmax > vmin:
        vol_norm = (vol - vmin) / (vmax - vmin)
    else:
        vol_norm = vol.copy()

    nx, ny, nz = vol_norm.shape

    if tile_size is None:
        try:
            import psutil

            available_memory = psutil.virtual_memory().available
        except ImportError:
            available_memory = 8 * 1024 * 1024 * 1024

        safe_memory = available_memory * 0.8
        bytes_per_voxel = 8
        estimated_factor = 300

        max_total_voxels = int(
            np.floor(safe_memory / bytes_per_voxel / estimated_factor)
        )
        target_edge = int(np.floor(max_total_voxels ** (1 / 3)))
        tile_size = tuple(min(target_edge, d) for d in (nx, ny, nz))

    tile_with_overlap = tuple(
        min(t + overlap, d) for t, d in zip(tile_size, (nx, ny, nz))
    )

    tiles = _get_tile_indices((nx, ny, nz), tile_with_overlap, overlap)

    grid_cache: dict[tuple, Any] = {}
    results = []
    retained_pcts = []

    if n_jobs is None or n_jobs == 1:
        for tile_idx in tiles:
            z_start, y_start, x_start, z_slc, y_slc, x_slc = tile_idx
            tile_data = vol_norm[z_slc, y_slc, x_slc]

            pad_shape = _udct_compatible_shape(tile_data.shape, num_scales)
            if pad_shape != tile_data.shape:
                pads = tuple((0, ps - ds) for ds, ps in zip(tile_data.shape, pad_shape))
                tile_padded = np.pad(tile_data, pads, mode="edge")
            else:
                tile_padded = tile_data

            key = (*pad_shape, num_scales, wedges_per_dir, use_gpu)
            if key not in grid_cache:
                grid_cache[key] = _get_udct_grid(
                    pad_shape, num_scales, wedges_per_dir, use_gpu
                )
                if use_gpu:
                    _move_grid_to_device(grid_cache[key], _get_device())

            grid = grid_cache[key]
            device = _get_device() if use_gpu else None
            gpu_dtype = _gpu_dtype(device) if use_gpu else None

            denoised, retained_pct = _curvelet_denoise_3d_udct_with_grid(
                tile_padded, grid, sigma, device, gpu_dtype
            )
            retained_pcts.append(retained_pct)

            dz = (pad_shape[0] - tile_data.shape[0]) // 2
            dz_end = pad_shape[0] - dz
            dy = (pad_shape[1] - tile_data.shape[1]) // 2
            dy_end = pad_shape[1] - dy
            dx = (pad_shape[2] - tile_data.shape[2]) // 2
            dx_end = pad_shape[2] - dx
            denoised_cropped = denoised[dz:dz_end, dy:dy_end, dx:dx_end]

            weight = _generate_cosine_taper(tile_data.shape, overlap)
            results.append((denoised_cropped, weight, z_slc, y_slc, x_slc))
    else:
        from joblib import Parallel, delayed

        def process_tile(tile_idx):
            z_start, y_start, x_start, z_slc, y_slc, x_slc = tile_idx
            tile_data = vol_norm[z_slc, y_slc, x_slc]

            pad_shape = _udct_compatible_shape(tile_data.shape, num_scales)
            if pad_shape != tile_data.shape:
                pads = tuple((0, ps - ds) for ds, ps in zip(tile_data.shape, pad_shape))
                tile_padded = np.pad(tile_data, pads, mode="edge")
            else:
                tile_padded = tile_data

            key = (*pad_shape, num_scales, wedges_per_dir, use_gpu)
            if key not in grid_cache:
                grid_cache[key] = _get_udct_grid(
                    pad_shape, num_scales, wedges_per_dir, use_gpu
                )
                if use_gpu:
                    _move_grid_to_device(grid_cache[key], _get_device())

            grid = grid_cache[key]
            device = _get_device() if use_gpu else None
            gpu_dtype = _gpu_dtype(device) if use_gpu else None

            denoised, retained_pct = _curvelet_denoise_3d_udct_with_grid(
                tile_padded, grid, sigma, device, gpu_dtype
            )

            dz = (pad_shape[0] - tile_data.shape[0]) // 2
            dz_end = pad_shape[0] - dz
            dy = (pad_shape[1] - tile_data.shape[1]) // 2
            dy_end = pad_shape[1] - dy
            dx = (pad_shape[2] - tile_data.shape[2]) // 2
            dx_end = pad_shape[2] - dx
            denoised_cropped = denoised[dz:dz_end, dy:dy_end, dx:dx_end]

            weight = _generate_cosine_taper(tile_data.shape, overlap)
            return denoised_cropped, weight, z_slc, y_slc, x_slc, retained_pct

        tile_results = Parallel(n_jobs=n_jobs)(
            delayed(process_tile)(tile_idx) for tile_idx in tiles
        )
        results = [(r[0], r[1], r[2], r[3], r[4]) for r in tile_results]
        retained_pcts = [r[5] for r in tile_results]

    output = np.zeros_like(vol_norm)
    weights = np.zeros_like(vol_norm)

    for denoised_tile, weight_tile, z_slc, y_slc, x_slc in results:
        z_valid = min(z_slc.stop, nz) - z_slc.start
        y_valid = min(y_slc.stop, ny) - y_slc.start
        x_valid = min(x_slc.stop, nx) - x_slc.start

        output[
            z_slc.start : z_slc.start + z_valid,
            y_slc.start : y_slc.start + y_valid,
            x_slc.start : x_slc.start + x_valid,
        ] += denoised_tile[:z_valid, :y_valid, :x_valid]
        weights[
            z_slc.start : z_slc.start + z_valid,
            y_slc.start : y_slc.start + y_valid,
            x_slc.start : x_slc.start + x_valid,
        ] += weight_tile[:z_valid, :y_valid, :x_valid]

    if retained_pcts:
        logger.info(
            "\tretained %.1f%% of curvelet coefficients (min=%.1f%%, max=%.1f%%)",
            np.mean(retained_pcts),
            min(retained_pcts),
            max(retained_pcts),
        )

    mask = weights > 1e-10
    output[mask] /= weights[mask]

    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        mmap_path = os.path.join(outdir, "curvelet_denoised_tiled.npy")
        output_mmap = np.memmap(mmap_path, dtype=np.float64, mode="w+", shape=vol.shape)
        if vmax > vmin:
            output_mmap[:] = output * (vmax - vmin) + vmin
        else:
            output_mmap[:] = output
        output_mmap.flush()
        return output_mmap

    if vmax > vmin:
        output = output * (vmax - vmin) + vmin

    return output


# ---------------------------------------------------------------------------
# 3D MCT backend (Monogenic Curvelet Transform for volumes)
# ---------------------------------------------------------------------------

# The 3D MCT extends the 2D monogenic approach by using the 3D Riesz
# transform (q1, q2, q3 for x, y, z) and thresholding on the 4D
# monogenic amplitude sqrt(q0^2 + q1^2 + q2^2 + q3^2).


def _riesz_transform_3d(voxels: np.ndarray, axis: int) -> np.ndarray:
    """Apply the Riesz transform to a 3D array along the specified axis.

    Parameters
    ----------
    voxels : np.ndarray
        3D input array.
    axis : int
        Axis along which to differentiate (0=z, 1=y, 2=x).

    Returns
    -------
    np.ndarray
        Real-valued Riesz-transformed array.
    """
    nz, ny, nx = voxels.shape
    w = np.fft.fftfreq(nz)[:, None, None]
    v = np.fft.fftfreq(ny)[None, :, None]
    u = np.fft.fftfreq(nx)[None, None, :]
    radius = np.sqrt(u**2 + v**2 + w**2)
    radius[0, 0, 0] = 1.0
    if axis == 0:
        kernel = -1j * w / radius
    elif axis == 1:
        kernel = -1j * v / radius
    else:
        kernel = -1j * u / radius
    kernel[0, 0, 0] = 0.0
    return np.fft.ifftn(np.fft.fftn(voxels) * kernel).real


def _mct_threshold_subband_3d(
    subband: np.ndarray,
    sigma: float | None = None,
    scale_idx: int = 0,
) -> np.ndarray:
    """Compute a threshold and apply MCT-style amplitude thresholding.

    The threshold is computed from the subband coefficients (same strategy
    as UDCT). The suppression factor is derived from the 3D monogenic
    amplitude rather than from the raw coefficients.

    Parameters
    ----------
    subband : np.ndarray
        3D coefficient subband.
    sigma : float, optional
        MAD scale factor, or ``None`` for elbow detection.
    scale_idx : int, optional
        Scale index (0 = coarse, no thresholding).

    Returns
    -------
    np.ndarray
        Thresholded subband.
    """
    T = 0.0
    if sigma is None or sigma <= 0:
        if scale_idx > 0:
            T = _elbow_threshold(subband)
            T = max(T, 1.0 * _mad_std(subband))
    else:
        if scale_idx > 0:
            noise_std = _mad_std(subband)
            T = sigma * noise_std
    if T <= 0:
        return subband.copy()
    q1 = _riesz_transform_3d(subband, axis=0)
    q2 = _riesz_transform_3d(subband, axis=1)
    q3 = _riesz_transform_3d(subband, axis=2)
    amplitude = np.sqrt(subband**2 + q1**2 + q2**2 + q3**2 + 1e-30)
    thresholded = np.maximum(amplitude - T, 0.0)
    atten = np.divide(
        thresholded, amplitude, out=np.ones_like(amplitude), where=amplitude > 1e-30
    )
    return subband * atten


def curvelet_denoise_3d_mct(
    volume: np.ndarray,
    sigma: float | None = None,
    num_scales: int = 4,
    wedges_per_dir: int = 3,
) -> np.ndarray:
    """Denoise a 3D volume using the MCT (Monogenic Curvelet Transform).

    Parameters
    ----------
    volume : np.ndarray
        3D volume to denoise.
    sigma : float, optional
        MAD scale factor, or ``None`` for elbow detection.
    num_scales : int, optional
        Number of curvelet scales. Defaults to 4.
    wedges_per_dir : int, optional
        Wedges per direction. Defaults to 3.

    Returns
    -------
    np.ndarray
        Denoised volume.
    """
    volume = np.asarray(volume, dtype=np.float64)
    vmin, vmax = volume.min(), volume.max()
    if vmax > vmin:
        volume = (volume - vmin) / (vmax - vmin)

    orig_shape = volume.shape
    if num_scales is None or num_scales <= 0:
        num_scales = _auto_num_scales(min(orig_shape))
    pad_shape = _udct_compatible_shape(orig_shape, num_scales)
    if pad_shape != orig_shape:
        pads = tuple((0, ps - ds) for ds, ps in zip(orig_shape, pad_shape))
        volume = np.pad(volume, pads, mode="edge")

    grid = _get_udct_grid(pad_shape, num_scales, wedges_per_dir, use_gpu=False)
    coeffs = grid.forward(volume)

    new_coeffs = []
    for scale_idx, scale in enumerate(coeffs):
        new_scale = []
        for direction in scale:
            new_direction = []
            for subband in direction:
                new_direction.append(
                    _mct_threshold_subband_3d(subband, sigma=sigma, scale_idx=scale_idx)
                )
            new_scale.append(new_direction)
        new_coeffs.append(new_scale)

    total, kept = _udct_coeff_stats(new_coeffs)
    retained_pct = 100.0 * kept / total if total > 0 else 0.0
    logger.info("\tretained %.1f%% of curvelet coefficients", retained_pct)

    result = grid.backward(new_coeffs)

    if pad_shape != orig_shape:
        result = result[: orig_shape[0], : orig_shape[1], : orig_shape[2]]

    if vmax > vmin:
        result = result * (vmax - vmin) + vmin

    return result


def _curvelet_denoise_3d_mct_with_grid(
    volume: np.ndarray,
    grid,
    sigma: float | None = None,
    device=None,
    gpu_dtype=None,
) -> np.ndarray:
    """Perform 3D MCT denoising using a pre-computed grid.

    Parameters
    ----------
    volume : np.ndarray
        3D volume to denoise (normalized to ``[0, 1]``).
    grid : UDCT
        Pre-computed UDCT grid.
    sigma : float, optional
        MAD scale factor, or ``None`` for elbow detection.
    device : torch.device, optional
        GPU device (unused — MCT always runs on CPU).
    gpu_dtype : torch.dtype, optional
        GPU dtype (unused).

    Returns
    -------
    np.ndarray
        Denoised volume (still normalized).
    """
    coeffs = grid.forward(volume)

    new_coeffs = []
    for scale_idx, scale in enumerate(coeffs):
        new_scale = []
        for direction in scale:
            new_direction = []
            for subband in direction:
                new_direction.append(
                    _mct_threshold_subband_3d(subband, sigma=sigma, scale_idx=scale_idx)
                )
            new_scale.append(new_direction)
        new_coeffs.append(new_scale)

    result = grid.backward(new_coeffs)

    return result


def curvelet_denoise_3d_mct_tiled(
    vol: np.ndarray,
    sigma: float | None = None,
    num_scales: int = 3,
    wedges_per_dir: int = 3,
    tile_size: tuple[int, ...] | None = None,
    overlap: int = 32,
    n_jobs: int | None = None,
    outdir: str | None = None,
) -> np.ndarray:
    """Denoise a 3D volume using MCT with tiling for memory efficiency.

    Parameters
    ----------
    vol : np.ndarray
        3D volume to denoise.
    sigma : float, optional
        MAD scale factor, or ``None`` for elbow detection.
    num_scales : int, optional
        Number of curvelet scales. Defaults to 3.
    wedges_per_dir : int, optional
        Wedges per direction. Defaults to 3.
    tile_size : tuple[int, ...], optional
        Size of each tile as ``(nx, ny, nz)``. If ``None``, auto-detects.
    overlap : int, optional
        Overlap in voxels. Defaults to 32.
    n_jobs : int, optional
        Number of parallel workers. ``None`` uses all CPUs.
    outdir : str, optional
        Directory for memory-mapped output.

    Returns
    -------
    np.ndarray
        Denoised volume.
    """
    import os

    vol = np.asarray(vol, dtype=np.float64)
    if vol.ndim != 3:
        raise ValueError(f"Input must be 3D, got {vol.ndim}D")
    if num_scales < 1:
        raise ValueError(f"num_scales must be >= 1, got {num_scales}")
    if wedges_per_dir < 3:
        raise ValueError(f"wedges_per_dir must be >= 3, got {wedges_per_dir}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")

    vmin, vmax = vol.min(), vol.max()
    if vmax > vmin:
        vol_norm = (vol - vmin) / (vmax - vmin)
    else:
        vol_norm = vol.copy()

    nx, ny, nz = vol_norm.shape

    if tile_size is None:
        try:
            import psutil

            available_memory = psutil.virtual_memory().available
        except ImportError:
            available_memory = 8 * 1024 * 1024 * 1024

        safe_memory = available_memory * 0.8
        bytes_per_voxel = 8
        estimated_factor = 300

        max_total_voxels = int(
            np.floor(safe_memory / bytes_per_voxel / estimated_factor)
        )
        target_edge = int(np.floor(max_total_voxels ** (1 / 3)))
        tile_size = tuple(min(target_edge, d) for d in (nx, ny, nz))

    tile_with_overlap = tuple(
        min(t + overlap, d) for t, d in zip(tile_size, (nx, ny, nz))
    )

    tiles = _get_tile_indices((nx, ny, nz), tile_with_overlap, overlap)

    grid_cache: dict[tuple, object] = {}
    results = []
    retained_pcts = []

    if n_jobs is None or n_jobs == 1:
        for tile_idx in tiles:
            _z_start, _y_start, _x_start, z_slc, y_slc, x_slc = tile_idx
            tile_data = vol_norm[z_slc, y_slc, x_slc]

            pad_shape = _udct_compatible_shape(tile_data.shape, num_scales)
            if pad_shape != tile_data.shape:
                pads = tuple((0, ps - ds) for ds, ps in zip(tile_data.shape, pad_shape))
                tile_padded = np.pad(tile_data, pads, mode="edge")
            else:
                tile_padded = tile_data

            key = (*pad_shape, num_scales, wedges_per_dir, False)
            if key not in grid_cache:
                grid_cache[key] = _get_udct_grid(
                    pad_shape, num_scales, wedges_per_dir, use_gpu=False
                )

            grid = grid_cache[key]
            denoised, retained_pct = _curvelet_denoise_3d_mct_with_grid(
                tile_padded, grid, sigma
            )
            retained_pcts.append(retained_pct)

            dz = (pad_shape[0] - tile_data.shape[0]) // 2
            dz_end = pad_shape[0] - dz
            dy = (pad_shape[1] - tile_data.shape[1]) // 2
            dy_end = pad_shape[1] - dy
            dx = (pad_shape[2] - tile_data.shape[2]) // 2
            dx_end = pad_shape[2] - dx
            denoised_cropped = denoised[dz:dz_end, dy:dy_end, dx:dx_end]

            weight = _generate_cosine_taper(tile_data.shape, overlap)
            results.append((denoised_cropped, weight, z_slc, y_slc, x_slc))
    else:
        from joblib import Parallel, delayed

        def process_tile(tile_idx):
            _z_start, _y_start, _x_start, z_slc, y_slc, x_slc = tile_idx
            tile_data = vol_norm[z_slc, y_slc, x_slc]

            pad_shape = _udct_compatible_shape(tile_data.shape, num_scales)
            if pad_shape != tile_data.shape:
                pads = tuple((0, ps - ds) for ds, ps in zip(tile_data.shape, pad_shape))
                tile_padded = np.pad(tile_data, pads, mode="edge")
            else:
                tile_padded = tile_data

            key = (*pad_shape, num_scales, wedges_per_dir, False)
            if key not in grid_cache:
                grid_cache[key] = _get_udct_grid(
                    pad_shape, num_scales, wedges_per_dir, use_gpu=False
                )

            grid = grid_cache[key]
            denoised, retained_pct = _curvelet_denoise_3d_mct_with_grid(
                tile_padded, grid, sigma
            )

            dz = (pad_shape[0] - tile_data.shape[0]) // 2
            dz_end = pad_shape[0] - dz
            dy = (pad_shape[1] - tile_data.shape[1]) // 2
            dy_end = pad_shape[1] - dy
            dx = (pad_shape[2] - tile_data.shape[2]) // 2
            dx_end = pad_shape[2] - dx
            denoised_cropped = denoised[dz:dz_end, dy:dy_end, dx:dx_end]

            weight = _generate_cosine_taper(tile_data.shape, overlap)
            return denoised_cropped, weight, z_slc, y_slc, x_slc, retained_pct

        tile_results = Parallel(n_jobs=n_jobs)(
            delayed(process_tile)(tile_idx) for tile_idx in tiles
        )
        results = [(r[0], r[1], r[2], r[3], r[4]) for r in tile_results]
        retained_pcts = [r[5] for r in tile_results]

    if retained_pcts:
        logger.info(
            "\tretained %.1f%% of curvelet coefficients (min=%.1f%%, max=%.1f%%)",
            np.mean(retained_pcts),
            min(retained_pcts),
            max(retained_pcts),
        )

    output = np.zeros_like(vol_norm)
    weights = np.zeros_like(vol_norm)

    for denoised_tile, weight_tile, z_slc, y_slc, x_slc in results:
        z_valid = min(z_slc.stop, nz) - z_slc.start
        y_valid = min(y_slc.stop, ny) - y_slc.start
        x_valid = min(x_slc.stop, nx) - x_slc.start

        output[
            z_slc.start : z_slc.start + z_valid,
            y_slc.start : y_slc.start + y_valid,
            x_slc.start : x_slc.start + x_valid,
        ] += denoised_tile[:z_valid, :y_valid, :x_valid]
        weights[
            z_slc.start : z_slc.start + z_valid,
            y_slc.start : y_slc.start + y_valid,
            x_slc.start : x_slc.start + x_valid,
        ] += weight_tile[:z_valid, :y_valid, :x_valid]

    mask = weights > 1e-10
    output[mask] /= weights[mask]

    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        mmap_path = os.path.join(outdir, "curvelet_denoised_mct_tiled.npy")
        output_mmap = np.memmap(mmap_path, dtype=np.float64, mode="w+", shape=vol.shape)
        if vmax > vmin:
            output_mmap[:] = output * (vmax - vmin) + vmin
        else:
            output_mmap[:] = output
        output_mmap.flush()
        return output_mmap

    if vmax > vmin:
        output = output * (vmax - vmin) + vmin

    return output
