"""Represent a helical map as a small set of gaussians, fitted once and cached.

The helicalProjection search spends most of its time turning each candidate map
into a side projection: reading the volume, filtering it, and expanding it by
its helical symmetry into a new box. All of that produces one 2D picture that
is then aligned against the query.

A gaussian mixture reaches the same picture without ever building the volume.
One asymmetric unit is fitted to a few hundred isotropic gaussians, the screw
operation is applied to those parameters rather than to voxels, and the side
view is rendered by accumulating each gaussian over its own footprint. The fit
is the expensive part and is done once per map, so it is cached.

The fit is a plain non-negative least squares over a fixed grid of candidate
centres, not an adaptive 3DGS-style optimisation with cloning and splitting.
That choice is measured, at both coarse and fine sampling: against a matched
component budget the grid fit scored 0.947 to 0.859 at 1.0 A/pixel on
EMD-46496 and 0.845 to 0.801 on EMD-12268, while running 40 times faster. The
adaptive fit won only where the budget was far too small for the map
(EMD-71835, 0.632 to 0.613), which is not the regime the search runs in.
"""

from dataclasses import dataclass

import numpy as np

import helicon

from .solver_gauss_analytic import nn_elasticnet


@dataclass
class MapGaussians:
    """A fitted asymmetric unit, in Angstroms relative to the map centre.

    Attributes
    ----------
    amplitudes : np.ndarray
        Per-gaussian amplitude, shape ``(G,)``.
    centers : np.ndarray
        Centres as ``(x, y, z)`` in Angstroms, shape ``(G, 3)``, with z along
        the helical axis as elsewhere in helicon.
    sigma : float
        The single isotropic width, in Angstroms.
    apix : float
        Pixel size of the map the fit was made from.
    """

    amplitudes: np.ndarray
    centers: np.ndarray
    sigma: float
    apix: float

    def __len__(self):
        return len(self.amplitudes)

    def to_set(self, device: str = "cpu"):
        """Build the torch :class:`IsotropicGaussianSet` this describes."""
        import torch

        return helicon.IsotropicGaussianSet(
            torch.tensor(self.amplitudes, dtype=torch.float32),
            torch.tensor(self.centers, dtype=torch.float32),
            torch.full((len(self.amplitudes),), float(self.sigma)),
            device=device,
        )


def _occupied_volume(
    slab, apix, sigma_voxels, threshold_fraction, threshold_value=None
):
    """Volume of the density worth fitting, in cubic Angstroms."""
    from scipy.ndimage import gaussian_filter

    smoothed = gaussian_filter(slab, sigma=sigma_voxels, mode="constant")
    if threshold_value is not None:
        return float((smoothed >= threshold_value).sum()) * apix**3
    peak = smoothed.max()
    if peak <= 0:
        return 0.0
    return float((smoothed > peak * threshold_fraction).sum()) * apix**3


def _grid_candidates(
    slab, apix, ny, nx, step, sigma_voxels, threshold_fraction, threshold_value=None
):
    """Grid points that carry density, with the smoothed value at each.

    The smoothing width is the width of the gaussians being fitted, not the
    grid step: the right hand side of the normal equations is the overlap of
    each basis gaussian with the map, which is exactly the map smoothed by that
    gaussian and read at the centre.
    """
    from scipy.ndimage import gaussian_filter

    gz, gy, gx = np.meshgrid(
        np.arange(0, slab.shape[0], step),
        np.arange(0, ny, step),
        np.arange(0, nx, step),
        indexing="ij",
    )
    centers = np.stack([gz.ravel(), gy.ravel(), gx.ravel()], -1).astype(np.int64)
    smoothed = gaussian_filter(slab, sigma=sigma_voxels, mode="constant")
    sampled = smoothed[centers[:, 0], centers[:, 1], centers[:, 2]].astype(np.float64)
    values = sampled * (2 * np.pi * sigma_voxels**2) ** 1.5
    if threshold_value is not None:
        # the depositors' level, in the map's own units, so it is compared
        # against the smoothed density rather than the overlap-scaled value
        keep = sampled >= threshold_value
    elif values.max() > 0:
        keep = values > values.max() * threshold_fraction
    else:
        keep = []
    return centers[keep], values[keep]


def fit_gaussians(
    data,
    apix,
    rise,
    n_components=600,
    slab_rises=1.0,
    threshold_fraction=0.05,
    threshold_value=None,
    candidate_factor=3,
    sigma_scale=0.45,
    l1_scale=3e-3,
    l2_scale=1e-4,
    max_candidates=4000,
):
    """Fit one asymmetric unit of a helical map with isotropic gaussians.

    Only a slab of the map is fitted -- the rest is reproduced by the screw
    operation -- so the cost is set by the slab, not by the box.

    Parameters
    ----------
    data : np.ndarray
        The map, shaped ``(nz, ny, nx)`` with the helical axis along z.
    apix : float
        Pixel size in Angstroms.
    rise : float
        Helical rise in Angstroms, which sets the slab thickness.
    n_components : int, optional
        Target number of gaussians. This sets the grid spacing and the width,
        so the count that comes back is close to it rather than exactly it.
        Defaults to 600.
    slab_rises : float, optional
        Slab thickness in units of the rise. Defaults to one rise, which tiles
        under the screw operation without double counting.
    threshold_fraction : float, optional
        Grid points below this fraction of the peak smoothed density are not
        offered to the solver. Defaults to 0.05. Ignored when
        ``threshold_value`` is given.
    threshold_value : float, optional
        An absolute floor, in the units of the data passed in, below which
        voxels are not fitted.

        EMDB's recommended contour level is the obvious candidate -- it is
        chosen per entry by the depositors, where a fraction of the map's
        maximum is a guess, and it excludes negative density outright
        (EMD-1427 recommends 28.0 for voxels running -39 to +46, so its
        negative lumen would never be fitted). It is available as
        ``recommended_contour``. Measured, it makes the search worse: over 60
        maps and 10 queries, fits masked at the contour put the right map
        first half the time against 70% for the relative threshold, losing
        clear hits rather than gaining them. Not used by default for that
        reason.

        Note the units. The level EMDB publishes belongs to the raw map; by
        the time :func:`fit_map` has low-passed, symmetry-averaged and
        resampled it, the values are much smaller and that level excludes
        everything -- four of twelve maps fitted zero components when it was
        applied there.
    candidate_factor : int, optional
        How many candidates per kept component to offer the solver. Defaults
        to 3.
    sigma_scale : float, optional
        Gaussian width as a fraction of the mean spacing between components.
        Defaults to 0.45.
    l1_scale, l2_scale : float, optional
        Elastic-net penalties, as fractions of the largest value and of the
        mean diagonal of the overlap matrix respectively.
    max_candidates : int, optional
        Hard ceiling on candidate centres, which bounds the dense overlap
        matrix the solver builds. Defaults to 4000.

    Returns
    -------
    MapGaussians
        The fitted set, with centres in Angstroms relative to the map centre.

    Notes
    -----
    The grid spacing is searched rather than derived from the box volume. A
    spacing computed from the box undercounts badly, because most of the box is
    background that the threshold removes: asking for 300 components that way
    produced 35, and the resulting fit scored 0.927 where a genuine 300
    scored 0.947.
    """
    nz, ny, nx = data.shape
    z_center = nz // 2
    half = max(0.5 * slab_rises * rise, apix)
    z0 = max(0, int(z_center - half / apix))
    z1 = min(nz, int(z_center + half / apix) + 1)
    slab = np.ascontiguousarray(data[z0:z1])

    # Width first, from how much density there is to share between the
    # components -- NOT from the grid step. Tying the two together is what
    # wrecked EMD-12268: needing enough candidates drove the step to one voxel,
    # which drove sigma to 0.7 A, and the fit reproduced the volume route at
    # ncc 0.55 instead of 0.96. The occupied volume is measured with a coarse
    # smoothing, since it only sets a scale.
    occupied = _occupied_volume(
        slab, apix, max(1.0, 2.0 / apix), threshold_fraction, threshold_value
    )
    if occupied <= 0:
        raise ValueError("the map has no density above the threshold")
    spacing = (occupied / max(1, n_components)) ** (1.0 / 3.0)
    sigma_angstrom = max(sigma_scale * spacing, apix)
    sigma_voxels = sigma_angstrom / apix

    # Then the candidate grid. Its STEP carries the budget, because selecting
    # candidates by strength is a spatial bias, not a neutral economy: on
    # EMD-12268 the brightest 1800 grid points all lay in one protofilament, so
    # the second one was dropped entirely and the expansion swept a single
    # strand where the map has two -- ncc 0.62 against the volume route while
    # the fit reproduced its own slab at 0.95. Coarsening the grid instead
    # keeps every part of the density represented.
    wanted = min(max(1, n_components) * max(1, candidate_factor), max_candidates)
    step = max(1, int(round(spacing / apix / candidate_factor ** (1.0 / 3.0))))
    centers, values = _grid_candidates(
        slab, apix, ny, nx, step, sigma_voxels, threshold_fraction, threshold_value
    )
    while len(values) > wanted:
        step += 1
        centers, values = _grid_candidates(
            slab, apix, ny, nx, step, sigma_voxels, threshold_fraction, threshold_value
        )
    while step > 1:
        finer = _grid_candidates(
            slab,
            apix,
            ny,
            nx,
            step - 1,
            sigma_voxels,
            threshold_fraction,
            threshold_value,
        )
        if len(finer[1]) > wanted:
            break
        step -= 1
        centers, values = finer

    if len(values) == 0:
        raise ValueError("the map has no density above the threshold")

    # a gaussian's overlap with another of the same width, which is the normal
    # equation for a non-negative fit of this basis to the smoothed density
    d2 = ((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    gram = (np.pi * sigma_voxels**2) ** 1.5 * np.exp(-d2 / (4 * sigma_voxels**2))
    amplitudes = nn_elasticnet(
        gram,
        values,
        l1_scale * np.abs(values).max(),
        l2_scale * np.trace(gram) / max(len(gram), 1),
    )

    # keep everything the solver gave weight to, rather than a fixed count:
    # the count is already governed by the grid step above, and truncating here
    # would reintroduce the same bias towards the brightest region
    keep = amplitudes > amplitudes.max() * 1e-3 if amplitudes.max() > 0 else []
    centers, amplitudes = centers[keep], amplitudes[keep]
    if len(amplitudes) == 0:
        raise ValueError("the fit produced no components")

    centers_angstrom = np.stack(
        [
            (centers[:, 2] - nx // 2) * apix,
            (centers[:, 1] - ny // 2) * apix,
            (centers[:, 0] + z0 - z_center) * apix,
        ],
        -1,
    ).astype(np.float32)

    return MapGaussians(
        amplitudes=amplitudes.astype(np.float32),
        centers=centers_angstrom,
        sigma=float(sigma_angstrom),
        apix=float(apix),
    )


def _splat_and_blur(centers, amplitudes, nz, ny, apix, sigma):
    """Render equal-width isotropic gaussians by scattering, then blurring once.

    Every gaussian in a fitted set has the same width, and a sum of identical
    gaussians is the convolution of that gaussian with a sum of deltas. So the
    amplitudes can be scattered into the image -- bilinearly, to keep sub-pixel
    placement -- and the whole image blurred a single time, instead of each
    gaussian being evaluated over its own footprint.

    That matters because the screw expansion is prolific: a fit of a few
    hundred components becomes tens or hundreds of thousands of copies over a
    full pitch, and at 13 by 13 pixels each that was 0.95 s per map, four times
    what building the volume costs. The work here is linear in the number of
    copies plus one convolution of the output image.
    """
    from scipy.ndimage import gaussian_filter

    sigma_pixels = sigma / apix
    weights = amplitudes * np.sqrt(2 * np.pi) * sigma / apix

    cz = centers[:, 2] / apix + nz // 2
    cy = centers[:, 1] / apix + ny // 2

    z0 = np.floor(cz).astype(np.int64)
    y0 = np.floor(cy).astype(np.int64)
    fz = cz - z0
    fy = cy - y0

    image = np.zeros(ny * nz, dtype=np.float64)
    for dy, wy in ((0, 1.0 - fy), (1, fy)):
        for dz, wz in ((0, 1.0 - fz), (1, fz)):
            yy = y0 + dy
            zz = z0 + dz
            inside = (yy >= 0) & (yy < ny) & (zz >= 0) & (zz < nz)
            if not inside.any():
                continue
            flat = (yy[inside] * nz + zz[inside]).astype(np.int64)
            image += np.bincount(
                flat, weights=(weights * wy * wz)[inside], minlength=ny * nz
            )

    image = image.reshape(ny, nz)
    # gaussian_filter normalises its kernel to unit sum; the projection wants
    # the unnormalised gaussian, whose peak is 1
    blurred = gaussian_filter(image, sigma=sigma_pixels, mode="constant")
    return (blurred * (2 * np.pi * sigma_pixels**2)).astype(np.float32)


def side_projection(
    fit,
    twist,
    rise,
    csym,
    length,
    ny,
    apix,
    cutoff_sigma=4.0,
):
    """Render the helical side view of a fitted map.

    The gaussian counterpart of symmetrising a volume and summing it along x.

    Parameters
    ----------
    fit : MapGaussians
        The fitted asymmetric unit.
    twist, rise : float
        Helical parameters, in degrees and Angstroms.
    csym : int
        Cyclic symmetry about the helical axis.
    length : int
        Width of the output image in pixels, along the helical axis.
    ny : int
        Height of the output image in pixels, across the filament.
    apix : float
        Pixel size of the output image.
    cutoff_sigma : float, optional
        How far past the window the expansion reaches, in standard deviations,
        so the rendered edges carry real density. Defaults to 4.

    Returns
    -------
    np.ndarray
        The projection, shaped ``(ny, length)``.
    """
    gaussians = fit.to_set()
    z_half = length * apix / 2
    pad = cutoff_sigma * fit.sigma
    symmetrized = gaussians.apply_helical_symmetry(
        twist=float(twist),
        rise=float(rise),
        csym=int(csym),
        zmin=-z_half - pad,
        zmax=z_half + pad,
        min_dist_sigma=0.0,
    )
    return _splat_and_blur(
        symmetrized.centers.cpu().numpy(),
        symmetrized.amplitudes.cpu().numpy(),
        int(length),
        int(ny),
        float(apix),
        float(fit.sigma),
    )


def projection_mixture(
    fit,
    twist,
    rise,
    csym,
    length,
    ny,
    apix,
    sigma=None,
    cutoff_sigma=4.0,
    bin_scale=0.5,
):
    """The side projection as a 2D gaussian mixture, never rendered.

    Projecting a 3D isotropic gaussian along x gives a 2D isotropic gaussian of
    the same width, with amplitude multiplied by ``sqrt(2 pi) sigma``, so the
    mixture the projection *is* falls straight out of the 3D fit and its screw
    expansion -- the same expansion ``side_projection`` does before splatting.
    That is what lets a search compare a class average to a map without either
    of them becoming pixels.

    The expansion over a pitch is prolific, tens of thousands of copies, so the
    projected centres are binned onto a grid finer than the width before being
    returned. Gaussians of one width summed at nearly the same place are one
    gaussian of that width, so this changes the function it represents by the
    little the binning moves a centre, and it keeps the component count in the
    hundreds where the overlap integrals are cheap.

    Parameters
    ----------
    fit : MapGaussians
        The fitted asymmetric unit.
    twist, rise : float
        Helical parameters, in degrees and Angstroms.
    csym : int
        Cyclic symmetry about the helical axis.
    length, ny : int
        The projection window, in pixels: along the filament and across it.
    apix : float
        Pixel size of that window.
    sigma : float, optional
        Re-express the mixture at this width instead of the fit's own. Maps are
        fitted to a component budget, so their widths differ from each other
        and from a query's; an overlap between two different widths is a
        smaller number for no reason to do with how well they match, and the
        widths must agree before scores from different maps can be compared.
        Amplitudes are rescaled to preserve each component's mass.
    cutoff_sigma : float, optional
        How far past the window the expansion reaches. Defaults to 4.
    bin_scale : float, optional
        Bin spacing as a fraction of the width. Defaults to 0.5.

    Returns
    -------
    ImageGaussians
        Centres as ``(y, x)`` in Angstroms from the window's centre.
    """
    gaussians = fit.to_set()
    z_half = length * apix / 2
    y_half = ny * apix / 2
    pad = cutoff_sigma * fit.sigma
    symmetrized = gaussians.apply_helical_symmetry(
        twist=float(twist),
        rise=float(rise),
        csym=int(csym),
        zmin=-z_half - pad,
        zmax=z_half + pad,
        min_dist_sigma=0.0,
    )
    centers3d = symmetrized.centers.cpu().numpy()
    amplitudes = symmetrized.amplitudes.cpu().numpy().astype(np.float64)

    # the line integral along x, which is all projecting does to an isotropic
    # gaussian, and the (y, x) of the image is the (y, z) of the volume
    amplitudes = amplitudes * np.sqrt(2 * np.pi) * float(fit.sigma)
    centers = np.stack([centers3d[:, 1], centers3d[:, 2]], -1).astype(np.float64)

    inside = (np.abs(centers[:, 0]) <= y_half + pad) & (
        np.abs(centers[:, 1]) <= z_half + pad
    )
    centers, amplitudes = centers[inside], amplitudes[inside]
    if len(amplitudes) == 0:
        raise ValueError("the expansion left nothing inside the window")

    out_sigma = float(fit.sigma) if sigma is None else float(sigma)

    # Binned against the width the mixture will be *used* at, not the width it
    # was fitted at. A map fitted at 2 A per pixel has gaussians about 1.5 A
    # wide while a class average's fit is 6 A, and binning at a fraction of the
    # former merged almost nothing: 5500 components survived where 800 describe
    # the same picture at the width they are compared at.
    #
    # Binned on a flat integer key rather than with np.unique over rows, which
    # sorts a hundred thousand pairs lexicographically and dominated this
    # function -- 2 s a map, more than rendering the projection costs.
    step = max(float(bin_scale) * out_sigma, float(apix))
    keys = np.rint(centers / step).astype(np.int64)
    keys -= keys.min(axis=0)
    flat = keys[:, 0] * (keys[:, 1].max() + 1) + keys[:, 1]
    _, inverse = np.unique(flat, return_inverse=True)
    binned_amps = np.bincount(inverse, weights=amplitudes)
    # the bin's centre of mass, not the grid point, so binning costs less than
    # the grid spacing suggests
    binned_centers = np.stack(
        [
            np.bincount(inverse, weights=amplitudes * centers[:, k]) / binned_amps
            for k in range(2)
        ],
        -1,
    )

    if out_sigma != float(fit.sigma):
        binned_amps = binned_amps * (float(fit.sigma) / out_sigma) ** 2

    return ImageGaussians(
        amplitudes=binned_amps.astype(np.float32),
        centers=binned_centers.astype(np.float32),
        sigma=out_sigma,
        apix=float(apix),
    )


def fit_map(
    data,
    apix,
    twist,
    rise,
    csym,
    fit_apix=2.0,
    n_rises=6,
    n_components=600,
    sigma_scale=0.45,
    contour_level=None,
):
    """Fit a helical map, symmetrising it first.

    The symmetrisation is what makes the fit usable. Fitting a raw slab and
    replicating it reproduces the volume route's projection at ncc 0.65 on
    EMD-12268, because a raw slab carries one map's worth of noise and of
    asymmetry and the screw operation then repeats it faithfully. Fitting the
    symmetry-averaged map instead gives 0.967 on the same map, and costs one
    short symmetrisation -- six rises, not a pitch -- which happens once and is
    cached.

    Parameters
    ----------
    data : np.ndarray
        The map, shaped ``(nz, ny, nx)``, helical axis along z.
    apix : float
        Pixel size of ``data`` in Angstroms.
    twist, rise : float
        Helical parameters, in degrees and Angstroms.
    csym : int
        Cyclic symmetry about the helical axis.
    fit_apix : float, optional
        Sampling the fit is made at. The gaussians are in Angstroms and can be
        rendered at any pixel size afterwards, but they cannot carry detail
        finer than this. Defaults to 2.0.
    n_rises : int, optional
        Length of the symmetrised box, in rises. Defaults to 6.
    contour_level : float, optional
        Density below this level is not fitted. EMDB's recommended contour is
        what to pass, via :func:`recommended_contour`; it is chosen per entry
        by the depositors, where a fraction of the map's maximum is a guess,
        and it excludes negative density outright.

        Measured against 42 real EMPIAR-10940 class averages searched over 61
        maps: it recovers one of the two searches the unthresholded fit lost
        (32 of 42 against 31, where the volume route gets 33) and improves the
        mean rank of the true map from 7.6 to 5.1, which is better than the
        volume route's 7.2. An earlier comparison said the opposite, but its
        queries were cut from volume projections and so rewarded fits that
        resemble them.

        Ignored when it would leave nothing to fit.
    n_components, sigma_scale
        Passed to :func:`fit_gaussians`.

    Returns
    -------
    MapGaussians
    """
    nz, ny, nx = data.shape

    if contour_level is not None:
        # Applied to the raw map, where the level's units belong. By the time
        # the map has been low-passed, symmetry-averaged and resampled the
        # values are much smaller and this level excludes everything -- four
        # of twelve maps fitted zero components when it was applied there.
        masked = np.where(data >= contour_level, data, 0).astype(np.float32)
        if float((masked != 0).mean()) > 0:
            data = masked

    filtered = helicon.low_high_pass_filter(
        data, low_pass_fraction=min(1.0, apix / fit_apix)
    )
    width = max(16, int(nx * apix / fit_apix) // 2 * 2)
    n_sym = max(4, int(n_rises * rise / fit_apix) // 2 * 2)
    symmetrized = helicon.apply_helical_symmetry(
        data=filtered,
        apix=apix,
        twist_degree=float(twist),
        rise_angstrom=float(rise),
        csym=int(csym),
        fraction=5 * rise / (nz * apix),
        new_size=(n_sym, width, width),
        new_apix=fit_apix,
        cpu=1,
    )
    return fit_gaussians(
        symmetrized,
        fit_apix,
        rise,
        n_components=n_components,
        slab_rises=1.0,
        sigma_scale=sigma_scale,
    )


@helicon.cache(
    expires_after=None, cache_dir=helicon.cache_dir / "helical_lab", verbose=0
)
def gaussians_for_map(
    emd_id,
    twist,
    rise,
    csym,
    fit_apix=2.0,
    n_components=600,
    sigma_scale=0.45,
    contour_level=None,
):
    """Fit a map from EMDB and cache the result by its identity, not its data.

    The cache never expires: a released EMDB map does not change, and the fit
    of one is a pure function of the parameters that make up the key.

    Parameters
    ----------
    emd_id : str
        EMDB identifier, e.g. ``"emd-46496"``.
    twist, rise : float
        Helical parameters, in degrees and Angstroms.
    csym : int
        Cyclic symmetry about the helical axis.
    fit_apix, n_components, sigma_scale, contour_level
        Passed to :func:`fit_map`.

    Returns
    -------
    MapGaussians
    """
    from .helical_projection_compute import MapInfo

    map_info = MapInfo(emd_id=emd_id, label=emd_id, twist=twist, rise=rise, csym=csym)
    data, data_apix = map_info.get_data()
    return fit_map(
        data,
        data_apix,
        twist,
        rise,
        csym,
        fit_apix=fit_apix,
        n_components=n_components,
        sigma_scale=sigma_scale,
        contour_level=contour_level,
    )


def gaussians_for_map_info(map_info, fit_apix=2.0, n_components=600, sigma_scale=0.45):
    """The fit for a :class:`MapInfo`, cached when the map has an identity.

    A map given as an EMDB entry, a URL or a file has a stable name, so its fit
    is cached under that name and is paid for once ever. A map handed over as
    an array in memory has no such name, so it is fitted on the spot.

    Parameters
    ----------
    map_info : MapInfo
        The map to fit.
    fit_apix, n_components, sigma_scale, contour_level
        Passed to :func:`fit_map`.

    Returns
    -------
    MapGaussians
    """
    level = recommended_contour(map_info.emd_id)

    identity = None
    for candidate in (map_info.emd_id, map_info.url, map_info.filename):
        if isinstance(candidate, str) and len(candidate):
            identity = candidate
            break

    if identity is not None and map_info.emd_id:
        return gaussians_for_map(
            map_info.emd_id,
            float(map_info.twist),
            float(map_info.rise),
            int(map_info.csym),
            fit_apix=fit_apix,
            n_components=n_components,
            sigma_scale=sigma_scale,
            contour_level=level,
        )

    data, apix = map_info.get_data()
    return fit_map(
        data,
        apix,
        float(map_info.twist),
        float(map_info.rise),
        int(map_info.csym),
        fit_apix=fit_apix,
        n_components=n_components,
        sigma_scale=sigma_scale,
        contour_level=level,
    )


@helicon.cache(
    expires_after=None, cache_dir=helicon.cache_dir / "helical_lab", verbose=0
)
def recommended_contour(emd_id):
    """The contour level EMDB records for an entry, or None.

    Cached separately from the fit so a map whose metadata is unreachable
    simply falls back to the relative threshold rather than failing.

    Parameters
    ----------
    emd_id : str or None

    Returns
    -------
    float or None
    """
    if not isinstance(emd_id, str) or not emd_id:
        return None
    try:
        return helicon.dataset.EMDB().contour_level(emd_id)
    except Exception:
        return None


@dataclass
class ImageGaussians:
    """A 2D image as a sum of isotropic gaussians, in Angstroms.

    Attributes
    ----------
    amplitudes : np.ndarray
        Per-gaussian amplitude, shape ``(G,)``.
    centers : np.ndarray
        Centres as ``(y, x)`` in Angstroms from the image centre, shape
        ``(G, 2)``: y across the filament, x along it, as elsewhere.
    sigma : float
        The single isotropic width, in Angstroms.
    apix : float
        Pixel size of the image the fit was made from.
    background : float
        The background level that was subtracted before fitting.
    """

    amplitudes: np.ndarray
    centers: np.ndarray
    sigma: float
    apix: float
    background: float = 0.0

    def __len__(self):
        return len(self.amplitudes)


def background_from_stripes(image, fraction: float = 0.15):
    """Background level and spread, from the top and bottom of the image.

    A class average that has been auto-transformed has its filament running
    horizontally through the middle, so the rows at the very top and bottom
    are solvent and nothing else. That makes them the image's own answer to
    the question EMDB's contour level answers for a map: what counts as
    nothing.

    Parameters
    ----------
    image : np.ndarray
        The 2D image, filament horizontal.
    fraction : float, optional
        How much of the height, at each edge, to treat as background.
        Defaults to 0.15.

    Returns
    -------
    tuple of float
        ``(mean, sigma)`` of the background.
    """
    array = np.asarray(image, dtype=np.float64)
    rows = max(1, int(round(array.shape[0] * fraction)))
    stripes = np.concatenate([array[:rows].ravel(), array[-rows:].ravel()])
    if stripes.size == 0:
        return 0.0, 0.0
    return float(stripes.mean()), float(stripes.std())


def fit_image(
    image,
    apix,
    n_components=200,
    sigma_scale=0.45,
    background_sigmas=3.0,
    candidate_factor=3,
    max_candidates=2500,
    sigma_angstrom=None,
):
    """Fit a 2D image with isotropic gaussians, above its own background.

    The counterpart of :func:`fit_gaussians` for a class average. The floor is
    measured from the image rather than supplied: solvent mean plus a few
    times its spread, taken from the top and bottom stripes.

    Parameters
    ----------
    image : np.ndarray
        The 2D image, filament horizontal.
    apix : float
        Pixel size in Angstroms.
    n_components : int, optional
        Target number of gaussians. Defaults to 200.
    sigma_scale : float, optional
        Width as a fraction of the mean spacing. Defaults to 0.45.
    background_sigmas : float, optional
        How far above the background's spread a pixel must sit to be fitted.
        Defaults to 3. Ignored where the background has no spread at all --
        a rendered projection is exactly zero outside its density, and the
        rule would then degenerate to "everything above zero".
    sigma_angstrom : float, optional
        Fit at this width instead of deriving one from ``n_components``.

        Two mixtures can only be compared by their overlap if they were
        fitted at the same granularity. Deriving the width from a component
        budget does not achieve that: a projection covers several times the
        area of a class average, so the same budget gives it far wider
        gaussians. Pass the query's width when fitting a projection it will
        be compared against.
    candidate_factor, max_candidates
        As in :func:`fit_gaussians`.

    Returns
    -------
    ImageGaussians
    """
    from scipy.ndimage import gaussian_filter

    array = np.asarray(image, dtype=np.float32)
    ny, nx = array.shape
    mean, spread = background_from_stripes(array)
    if spread > 0:
        level = mean + background_sigmas * spread
    else:
        # a rendered image has no noise to measure; fall back to a small
        # fraction of its peak so the fit still has a floor
        level = mean + 0.02 * float(array.max() - mean)

    above = array >= level
    if not above.any():
        raise ValueError("no pixels above the background")
    occupied = float(above.sum()) * apix**2
    if sigma_angstrom is None:
        spacing = (occupied / max(1, n_components)) ** 0.5
        sigma_angstrom = max(sigma_scale * spacing, apix)
    else:
        sigma_angstrom = max(float(sigma_angstrom), apix)
        spacing = sigma_angstrom / sigma_scale
    sigma_pixels = sigma_angstrom / apix

    # with an explicit width the count follows the area, which is the point
    implied = max(1, int(round(occupied / max(spacing**2, 1e-9))))
    target = implied if sigma_angstrom is not None else n_components
    wanted = min(max(1, target) * max(1, candidate_factor), max_candidates)
    smoothed = gaussian_filter(array - mean, sigma=sigma_pixels, mode="constant")

    step = max(1, int(round(spacing / apix / candidate_factor**0.5)))
    while True:
        gy, gx = np.meshgrid(
            np.arange(0, ny, step), np.arange(0, nx, step), indexing="ij"
        )
        centers = np.stack([gy.ravel(), gx.ravel()], -1)
        sampled = smoothed[centers[:, 0], centers[:, 1]].astype(np.float64)
        keep = sampled >= (level - mean)
        centers, sampled = centers[keep], sampled[keep]
        if len(sampled) <= wanted or step >= max(ny, nx):
            break
        step += 1
    if len(sampled) == 0:
        raise ValueError("no candidates above the background")

    values = sampled * (2 * np.pi * sigma_pixels**2)
    d2 = ((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    gram = (np.pi * sigma_pixels**2) * np.exp(-d2 / (4 * sigma_pixels**2))
    amplitudes = nn_elasticnet(
        gram,
        values,
        3e-3 * np.abs(values).max(),
        1e-4 * np.trace(gram) / max(len(gram), 1),
    )
    kept = amplitudes > amplitudes.max() * 1e-3 if amplitudes.max() > 0 else []
    centers, amplitudes = centers[kept], amplitudes[kept]
    if len(amplitudes) == 0:
        raise ValueError("the fit produced no components")

    centers_angstrom = np.stack(
        [(centers[:, 0] - ny // 2) * apix, (centers[:, 1] - nx // 2) * apix], -1
    ).astype(np.float32)
    return ImageGaussians(
        amplitudes=amplitudes.astype(np.float32),
        centers=centers_angstrom,
        sigma=float(sigma_angstrom),
        apix=float(apix),
        background=float(mean),
    )


def render_image(fit, ny, nx, apix=None):
    """Draw a fitted image back out, at any sampling.

    Parameters
    ----------
    fit : ImageGaussians
    ny, nx : int
        Output size in pixels.
    apix : float, optional
        Output pixel size. Defaults to the fit's own.

    Returns
    -------
    np.ndarray

    Notes
    -----
    The renderer is the one used for projections, so the amplitudes carry a
    constant factor from integrating out a third axis that a 2D fit does not
    have. It scales the whole image alike, and everything downstream compares
    by correlation, which is invariant to that.
    """
    apix = float(apix if apix is not None else fit.apix)
    centers = np.stack(
        [np.zeros(len(fit)), fit.centers[:, 0], fit.centers[:, 1]], -1
    ).astype(np.float32)
    return _splat_and_blur(centers, fit.amplitudes, nx, ny, apix, fit.sigma)


def fit_queries(images, apix, sigma_angstrom=None, **kwargs):
    """Fit several query images to gaussians of one common width.

    A search compares one query against many maps, and the maps' mixtures are
    put at the query's width, so the width has to be decided once for the whole
    set rather than per image. Left to itself each fit picks its own width from
    its own occupied area, which would make two class averages of the same
    filament score differently for no reason but their contrast.

    Parameters
    ----------
    images : sequence of np.ndarray
        The query images, filament horizontal.
    apix : float
        Pixel size, in Angstroms.
    sigma_angstrom : float, optional
        The common width. By default the median of the widths the first few
        images choose for themselves.
    **kwargs
        Passed to :func:`fit_image`.

    Returns
    -------
    list of ImageGaussians
        One fit per image, all of the same width. An image that cannot be
        fitted -- one that is blank, say -- is dropped from the sample used to
        choose the width, but raises if it is the query itself.
    """
    images = list(images)
    if not len(images):
        return []
    if sigma_angstrom is None:
        widths = []
        for image in images[:5]:
            try:
                widths.append(fit_image(image, apix, **kwargs).sigma)
            except Exception:
                continue
        if not widths:
            raise ValueError("none of the query images could be fitted")
        sigma_angstrom = float(np.median(widths))
    return [
        fit_image(image, apix, sigma_angstrom=sigma_angstrom, **kwargs)
        for image in images
    ]
