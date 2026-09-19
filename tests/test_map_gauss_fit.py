import numpy as np
import pytest

torch = pytest.importorskip("torch")

import helicon
from helicon.webApps.lib import map_gauss_fit as mgf


def _tube(nz=48, ny=48, nx=48, apix=2.0, radius=10.0, sigma=3.0, seed=0):
    """A helical map: blobs on a screw path, which is what the fit assumes."""
    rng = np.random.default_rng(seed)
    z, y, x = np.mgrid[0:nz, 0:ny, 0:nx].astype(np.float32)
    z = (z - nz // 2) * apix
    y = (y - ny // 2) * apix
    x = (x - nx // 2) * apix
    vol = np.zeros((nz, ny, nx), dtype=np.float32)
    rise, twist = 8.0, 30.0
    for k in range(-nz, nz):
        cz = k * rise
        if abs(cz) > nz * apix / 2 + 3 * sigma:
            continue
        ang = np.deg2rad(twist * k)
        cx, cy = radius * np.cos(ang), radius * np.sin(ang)
        vol += np.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / (2 * sigma**2))
    return vol, apix, twist, rise


class TestSplatAndBlur:
    """The fast renderer must agree with evaluating each gaussian directly.

    Every fitted set shares one width, so a sum of them is one gaussian
    convolved with a sum of deltas. That identity is what makes scattering and
    blurring once equivalent to footprint evaluation -- and 15 to 25 times
    faster on an expanded set, which is the whole reason the gaussian route is
    quicker than building the volume.
    """

    def _set(self, n=200, sigma=3.0, seed=1):
        rng = np.random.default_rng(seed)
        centers = np.stack(
            [
                rng.uniform(-20, 20, n),
                rng.uniform(-30, 30, n),
                rng.uniform(-60, 60, n),
            ],
            -1,
        ).astype(np.float32)
        amps = rng.uniform(0.2, 1.0, n).astype(np.float32)
        return centers, amps, sigma

    def test_it_matches_footprint_evaluation(self):
        centers, amps, sigma = self._set()
        nz, ny, apix = 96, 48, 2.0
        fast = mgf._splat_and_blur(centers, amps, nz, ny, apix, sigma)
        gset = helicon.IsotropicGaussianSet(
            torch.tensor(amps),
            torch.tensor(centers),
            torch.full((len(amps),), float(sigma)),
        )
        exact = gset.projection_x(nz=nz, ny=ny, apix=apix, cutoff_sigma=6.0).numpy()
        assert fast.shape == exact.shape
        a = fast - fast.mean()
        b = exact - exact.mean()
        ncc = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
        assert ncc > 0.999, ncc

    def test_out_of_frame_gaussians_are_dropped_not_wrapped(self):
        # a scatter that wraps would fold density round the edges and quietly
        # corrupt every projection whose filament runs past the window
        centers = np.array([[0.0, 0.0, 1e4]], dtype=np.float32)
        amps = np.array([1.0], dtype=np.float32)
        img = mgf._splat_and_blur(centers, amps, 32, 32, 2.0, 3.0)
        assert float(np.abs(img).max()) == 0.0


class TestFitMap:
    """The fit has to survive the screw expansion, not just reproduce a slab."""

    def test_the_projection_follows_the_volume_route(self):
        vol, apix, twist, rise = _tube()
        fit = mgf.fit_map(vol, apix, twist, rise, 1, fit_apix=apix, n_components=200)
        assert len(fit) > 0
        length = 64
        got = mgf.side_projection(fit, twist, rise, 1, length, 48, apix)
        sym = helicon.apply_helical_symmetry(
            data=vol,
            apix=apix,
            twist_degree=twist,
            rise_angstrom=rise,
            csym=1,
            fraction=5 * rise / (vol.shape[0] * apix),
            new_size=(length, 48, 48),
            new_apix=apix,
            cpu=1,
        )
        want = sym.sum(axis=2).T
        a = got - got.mean()
        b = want - want.mean()
        ncc = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
        assert ncc > 0.9, ncc

    def test_a_weaker_second_filament_is_not_discarded(self):
        # candidates used to be kept by strength, which threw away an entire
        # weaker protofilament on EMD-12268 and left the expansion sweeping one
        # strand where the map has two. After symmetrisation a one-rise slab
        # holds one azimuth per radius, so the check is that both radii are
        # represented, not that both sides of the axis are.
        vol, apix, twist, rise = _tube()
        faint = np.roll(vol, shift=12, axis=2) * 0.25
        fit = mgf.fit_map(
            vol + faint, apix, twist, rise, 1, fit_apix=apix, n_components=300
        )
        radius = np.hypot(fit.centers[:, 0], fit.centers[:, 1])
        assert radius.min() < 16.0, float(radius.min())
        assert radius.max() > 18.0, float(radius.max())
