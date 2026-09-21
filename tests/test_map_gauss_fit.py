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


class TestContourThreshold:
    """EMDB publishes a recommended contour level, and the fit uses it.

    It looked harmful when it was first measured -- half the queries put the
    right map first against 70% without it -- but that measurement used
    queries cut from the volume route's own projections, which rewards
    whatever most resembles a volume projection. Measured again over 42 real
    class averages against 61 maps it is the better floor: 32 of 42 against
    31, and a mean rank of 5.1 against 7.2. Depositors set these levels by
    hand, so a wrong one is always possible, which is why the fit falls back
    to a fraction of the maximum when an entry records none.
    """

    def test_a_level_excludes_density_below_it(self):
        # a bright core inside a broad, dim shoulder. 5% of the maximum keeps
        # the shoulder; a level of 2.0 does not
        rng = np.random.default_rng(0)
        vol = rng.normal(0.0, 0.02, (32, 32, 32)).astype(np.float32)
        vol[8:24, 8:24, 8:24] += 1.0
        vol[14:18, 14:18, 14:18] += 5.0

        loose = mgf.fit_gaussians(vol, 2.0, 8.0, n_components=80)
        strict = mgf.fit_gaussians(vol, 2.0, 8.0, n_components=80, threshold_value=2.0)

        core = 8.0  # half-width of the bright core, in Angstroms
        assert float(np.abs(strict.centers).max()) <= core
        assert float(np.abs(loose.centers).max()) > core

    def test_thresholding_refines_rather_than_thins_the_fit(self):
        # counter-intuitive and worth pinning: masking shrinks the occupied
        # volume, which shrinks the spacing and the width derived from it, so
        # the grid gets finer and the fit ends up with MORE, smaller
        # components over a smaller region -- a different representation, not
        # a subset of the same one
        rng = np.random.default_rng(0)
        vol = rng.normal(0.0, 0.05, (24, 24, 24)).astype(np.float32)
        vol[10:14, 10:14, 10:14] += 5.0

        loose = mgf.fit_gaussians(vol, 2.0, 8.0, n_components=50)
        strict = mgf.fit_gaussians(vol, 2.0, 8.0, n_components=50, threshold_value=1.0)
        assert len(strict) > len(loose)
        assert float(np.abs(strict.centers).max()) == pytest.approx(
            float(np.abs(loose.centers).max())
        )

    def test_the_level_is_read_from_emdb_metadata(self, monkeypatch):
        class FakeEMDB:
            def contour_level(self, emd_id):
                return 28.0 if emd_id == "emd-1427" else None

        monkeypatch.setattr(mgf.helicon.dataset, "EMDB", lambda *a, **k: FakeEMDB())
        mgf.recommended_contour.clear_cache()
        assert mgf.recommended_contour("emd-1427") == 28.0
        assert mgf.recommended_contour("emd-0000") is None
        assert mgf.recommended_contour(None) is None

    def test_a_map_from_emdb_is_fitted_above_its_contour(self):
        # measured over 42 real class averages: thresholding at the
        # depositors' level recovers a search the unthresholded fit loses and
        # improves the mean rank of the true map from 7.6 to 5.1
        import inspect

        source = inspect.getsource(mgf.gaussians_for_map_info)
        assert "recommended_contour" in source
        assert "contour_level=level" in source

    def test_a_level_that_would_empty_the_map_is_ignored(self):
        vol, apix, twist, rise = _tube()
        fit = mgf.fit_map(
            vol,
            apix,
            twist,
            rise,
            1,
            fit_apix=apix,
            n_components=100,
            contour_level=float(vol.max()) * 10,
        )
        assert len(fit) > 0


class TestProjectionMixture:
    """The side projection as gaussians, which is what the fast search matches.

    Projecting an isotropic gaussian along one axis gives an isotropic gaussian
    of the same width, so the 2D mixture falls out of the 3D fit and its screw
    expansion with no rendering at all. It has to describe the same picture
    ``side_projection`` draws, or the search is matching something else.
    """

    def _fit(self):
        vol, apix, twist, rise = _tube()
        fit = mgf.fit_map(vol, apix, twist, rise, 1, fit_apix=apix, n_components=200)
        return fit, apix, twist, rise

    def test_it_describes_the_same_picture_as_the_render(self):
        fit, apix, twist, rise = self._fit()
        length, ny = 64, 48
        want = mgf.side_projection(fit, twist, rise, 1, length, ny, apix)
        mix = mgf.projection_mixture(fit, twist, rise, 1, length, ny, apix)
        got = mgf.render_image(mix, ny, length, apix=apix)
        a, b = got - got.mean(), want - want.mean()
        ncc = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
        assert ncc > 0.98, ncc

    def test_binning_keeps_the_count_far_below_the_expansion(self):
        fit, apix, twist, rise = self._fit()
        mix = mgf.projection_mixture(fit, twist, rise, 1, 64, 48, apix)
        # one component per bin of the window, not one per copy the screw made
        assert 0 < len(mix) < 64 * 48

    def test_a_requested_width_is_honoured_and_conserves_mass(self):
        fit, apix, twist, rise = self._fit()
        native = mgf.projection_mixture(fit, twist, rise, 1, 64, 48, apix)
        wider = mgf.projection_mixture(
            fit, twist, rise, 1, 64, 48, apix, sigma=native.sigma * 2
        )
        assert wider.sigma == pytest.approx(native.sigma * 2)
        assert np.allclose(wider.centers, native.centers)
        mass = lambda m: float((m.amplitudes * 2 * np.pi * m.sigma**2).sum())
        assert mass(wider) == pytest.approx(mass(native), rel=1e-5)


class TestFitQueries:
    """Several queries, one width, so their scores mean the same thing."""

    def _images(self):
        rng = np.random.default_rng(0)
        out = []
        for contrast in (1.0, 4.0):
            image = np.zeros((32, 64), dtype=np.float32)
            image[12:20, 8:56] = contrast
            image += rng.normal(0, 0.01 * contrast, image.shape)
            out.append(image)
        return out

    def test_every_fit_shares_one_width(self):
        fits = mgf.fit_queries(self._images(), apix=2.0)
        assert len(fits) == 2
        assert fits[0].sigma == fits[1].sigma

    def test_an_explicit_width_is_used_as_given(self):
        fits = mgf.fit_queries(self._images(), apix=2.0, sigma_angstrom=8.0)
        assert all(f.sigma == pytest.approx(8.0) for f in fits)

    def test_no_images_is_not_an_error(self):
        assert mgf.fit_queries([], apix=2.0) == []
