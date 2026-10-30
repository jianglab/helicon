import numpy as np
import pytest

from helicon.lib import denoise


def filaments(shape=(192, 208), noise_sd=1.0, seed=0):
    """Straight filaments with a repeat along them, plus white noise.

    Returns
    -------
    clean, noisy : numpy.ndarray
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[: shape[0], : shape[1]].astype(float)
    clean = np.zeros(shape)
    for _ in range(4):
        angle = rng.uniform(0, np.pi)
        c, s = np.cos(angle), np.sin(angle)
        y0, x0 = rng.uniform(0.2, 0.8) * shape[0], rng.uniform(0.2, 0.8) * shape[1]
        across = (x - x0) * s - (y - y0) * c
        along = (x - x0) * c + (y - y0) * s
        clean += np.exp(-0.5 * (across / 3.0) ** 2) * (1 + 0.5 * np.cos(along / 2.5))
    return clean, clean + rng.normal(scale=noise_sd, size=shape)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


class TestWindows:
    def test_curvelet_windows_and_low_pass_add_up_to_one(self):
        low2, windows = denoise.curvelet_windows((70, 83), n_bands=3)
        total = low2 + sum(w**2 for _, w in windows)
        assert np.abs(total - 1).max() < 1e-12

    def test_wavelet_windows_and_approximation_add_up_to_one(self):
        low2, windows = denoise.wavelet_windows((70, 83), levels=3)
        total = low2 + sum(np.abs(w) ** 2 for _, w in windows)
        assert np.abs(total - 1).max() < 1e-10


class TestReconstruction:
    @pytest.mark.parametrize(
        "method", [denoise.curvelet_denoise, denoise.wavelet_denoise]
    )
    def test_nothing_thresholded_gives_the_image_back(self, method):
        image = np.random.default_rng(1).normal(size=(70, 83)).cumsum(0)
        back = method(image, strength=0.0, workers=1)
        assert back.shape == image.shape
        assert np.abs(back - image).max() < 1e-9 * np.abs(image).max()


class TestNoise:
    def test_white_noise_level_is_measured(self):
        noise = np.random.default_rng(2).normal(scale=3.0, size=(256, 256))
        assert denoise.white_noise_sd(noise) == pytest.approx(3.0, rel=0.03)

    def test_a_smooth_signal_does_not_raise_the_noise_level(self):
        clean, noisy = filaments(noise_sd=1.0)
        assert denoise.white_noise_sd(noisy) == pytest.approx(1.0, rel=0.05)


class TestDenoise:
    def test_filaments_come_closer_to_the_truth(self):
        clean, noisy = filaments(noise_sd=1.0)
        out, info = denoise.denoise_micrograph(noisy, workers=1)
        assert out.dtype == np.float32 and out.shape == noisy.shape
        assert rmse(out, clean) < 0.35 * rmse(noisy, clean)
        assert info["strength"] == denoise.DEFAULT_STRENGTH
        assert info["noise_from"] == "the outer frequency ring"
        assert info["noise_sd"] == pytest.approx(1.0, rel=0.05)

    def test_a_given_noise_level_is_used_and_reported(self):
        _, noisy = filaments(noise_sd=1.0)
        weak, info = denoise.denoise_micrograph(noisy, noise_sd=0.1, workers=1)
        strong, _ = denoise.denoise_micrograph(noisy, noise_sd=2.0, workers=1)
        assert info["noise_from"] == "given" and info["noise_sd"] == 0.1
        # a lower noise level removes less
        assert rmse(weak, noisy) < rmse(strong, noisy)

    def test_pure_noise_is_mostly_removed(self):
        noise = np.random.default_rng(3).normal(size=(128, 128))
        out, _ = denoise.denoise_micrograph(noise, workers=1)
        assert out.std() < 0.2


def rods_map(n=48, noise_sd=1.0, seed=0):
    """Two rods twisting round the z axis in coloured noise, flat outside the sphere
    (as RELION writes maps).

    Returns
    -------
    clean, noisy : numpy.ndarray
    """
    rng = np.random.default_rng(seed)
    z, y, x = np.mgrid[:n, :n, :n] - (n - 1) / 2
    clean = np.zeros((n, n, n))
    for phase in (0, np.pi):
        angle = 2 * np.pi * z / n + phase
        d2 = (x - 6 * np.cos(angle)) ** 2 + (y - 6 * np.sin(angle)) ** 2
        clean += np.exp(-0.5 * d2 / 2.5**2)
    f = np.sqrt(sum(np.fft.fftfreq(n)[s] ** 2 for s in np.ix_(*[range(n)] * 3)))
    noise = np.fft.ifftn(np.fft.fftn(rng.normal(size=clean.shape)) / (1 + 4 * f)).real
    noisy = clean + noise * noise_sd / noise.std()
    outside = x**2 + y**2 + z**2 > (n / 2) ** 2
    clean[outside] = noisy[outside] = 0.0
    return clean, noisy


class TestMaps:
    @pytest.mark.parametrize("tiling", ["curvelet", "wavelet"])
    def test_3d_windows_add_up_to_one(self, tiling):
        if tiling == "curvelet":
            low2, windows = denoise.curvelet_windows((20, 18, 16), n_bands=2)
        else:
            low2, windows = denoise.wavelet_windows((20, 18, 16), levels=2)
        total = low2 + sum(np.abs(w) ** 2 for _, w in windows)
        assert np.abs(total - 1).max() < 1e-10

    @pytest.mark.parametrize("dtype, tol", [(np.float64, 1e-9), (np.float32, 1e-5)])
    @pytest.mark.parametrize(
        "method", [denoise.curvelet_denoise, denoise.wavelet_denoise]
    )
    def test_nothing_thresholded_gives_the_map_back(self, method, dtype, tol):
        # even sizes: the Nyquist planes must not break the symmetry
        volume = np.random.default_rng(1).normal(size=(20, 18, 16)).cumsum(0)
        back = method(volume.astype(dtype), strength=0.0, noise_sd=1.0, workers=1)
        assert back.dtype == dtype
        assert np.abs(back - volume).max() < tol * np.abs(volume).max()

    def test_a_nearly_flat_outside_is_left_out_of_the_statistics(self):
        _, noisy = rods_map()
        flat = noisy == 0
        nearly = noisy + 1e-9 * np.random.default_rng(2).normal(size=noisy.shape) * flat
        inside = denoise._not_flat(noisy)
        assert inside.mean() < 0.7  # the sphere is 52% of the box
        assert (denoise._not_flat(nearly) == inside).all()

    def test_rods_come_closer_to_the_truth(self):
        clean, noisy = rods_map()
        out, info = denoise.denoise_map(noisy, workers=1)
        assert out.dtype == np.float32 and out.shape == noisy.shape
        assert rmse(out, clean) < 0.5 * rmse(noisy, clean)
        assert "not flat" in info["noise_from"]

    def test_single_precision_gives_the_double_result(self):
        _, noisy = rods_map(n=24)
        single, info = denoise.denoise_map(noisy, workers=1)
        double, _ = denoise.denoise_map(noisy, workers=1, precision="double")
        assert info["precision"] == "single"
        assert np.abs(single - double).max() < 1e-4 * np.abs(double).max()

    def test_tiles_give_nearly_the_whole_map_result(self):
        _, noisy = rods_map()
        whole, info = denoise.denoise_map(noisy, workers=1, tile_size=0)
        tiled, tiled_info = denoise.denoise_map(noisy, workers=1, tile_size=20)
        assert info["tile_size"] is None
        assert tiled_info["tile_size"] == 16  # 48 voxels in three even tiles
        assert tiled_info["tiles_why"] == "given"
        assert rmse(tiled, whole) < 0.1 * rmse(whole, noisy)

    def test_tiles_are_used_when_the_map_does_not_fit(self, monkeypatch):
        import helicon

        x = np.empty((300, 300, 300), np.float32)  # untouched: no memory used
        monkeypatch.setattr(helicon, "available_memory", lambda: 100 * 2**30)
        assert denoise._plan_tiles(x)[0] is None
        monkeypatch.setattr(helicon, "available_memory", lambda: 2**30)
        core, why = denoise._plan_tiles(x)
        # about 250-voxel tiles fit; 300 voxels are cut into two even ones
        assert core == 150 and "needs" in why
        # with no memory at all, the smallest tiles: three of 100 (at least 112)
        monkeypatch.setattr(helicon, "available_memory", lambda: 2**20)
        assert denoise._plan_tiles(x)[0] == 100

    def test_running_out_of_memory_halves_the_tiles(self, monkeypatch):
        def no_memory(*args, **kwargs):
            raise MemoryError

        monkeypatch.setattr(denoise, "curvelet_denoise", no_memory)
        _, noisy = rods_map(n=24)
        out, info = denoise.denoise_map(noisy, workers=1)
        assert info["tile_size"] == 12 and "halved" in info["tiles_why"]
        assert np.isfinite(out).all() and out.shape == noisy.shape

    def test_the_tuned_strength_is_as_good_as_the_best_tried(self):
        clean, half1 = rods_map(seed=0)
        _, half2 = rods_map(seed=1)  # the same rods, other noise
        strength, scores = denoise.tune_strength(half1, half2, workers=1)
        assert len(scores) >= 3
        errors = {
            s: rmse(denoise.denoise_map(half1, s, workers=1)[0], clean)
            for s in (*scores, strength)
        }
        assert errors[strength] <= 1.02 * min(errors.values())

    def test_a_2d_image_is_refused(self):
        with pytest.raises(ValueError, match="3D"):
            denoise.denoise_map(np.zeros((8, 8)))

    def test_the_half_maps_difference_has_the_noise_of_a_half_or_the_average(self):
        clean, half1 = rods_map(seed=0)
        _, half2 = rods_map(seed=1)
        noise = half1 - clean
        assert np.std(denoise.half_map_noise(half1, half2)) == pytest.approx(
            np.std(noise), rel=0.02
        )
        average = denoise.half_map_noise(half1, half2, "average")
        assert np.std(average) == pytest.approx(np.std(noise) / np.sqrt(2), rel=0.02)

    def test_noise_from_the_half_maps_does_at_least_as_well(self):
        clean, half1 = rods_map(seed=0)
        _, half2 = rods_map(seed=1)
        noise = denoise.half_map_noise(half1, half2)
        guessed, _ = denoise.denoise_map(half1, workers=1)
        measured, info = denoise.denoise_map(half1, workers=1, noise=noise)
        assert info["noise_from"] == "the half maps' difference"
        assert rmse(measured, clean) <= rmse(guessed, clean)
        # the tiles pool the same noise
        tiled, _ = denoise.denoise_map(half1, workers=1, noise=noise, tile_size=20)
        assert rmse(tiled, measured) < 0.1 * rmse(measured, half1)
        with pytest.raises(ValueError, match="noise map"):
            denoise.denoise_map(half1, noise=noise[:-1])


class TestFindHalfMaps:
    @pytest.mark.parametrize(
        "given, half1, half2",
        [
            # RELION Refine3D
            ("run_class001.mrc", "run_half1_class001_unfil", "run_half2_class001_unfil"),
            ("run_half2_class001_unfil.mrc", "run_half2_class001_unfil", "run_half1_class001_unfil"),
            # cryoSPARC
            ("J12_005_volume_map.mrc", "J12_005_volume_map_half_A", "J12_005_volume_map_half_B"),
            ("J12_005_volume_map_sharp.mrc", "J12_005_volume_map_half_A", "J12_005_volume_map_half_B"),
            ("J12_005_volume_map_half_A.mrc", "J12_005_volume_map_half_A", "J12_005_volume_map_half_B"),
            # EMDB
            ("emd_1234.map", "emd_1234_half_map_1", "emd_1234_half_map_2"),
            ("emd_1234_half_map_1.map", "emd_1234_half_map_1", "emd_1234_half_map_2"),
        ],
    )  # fmt: skip
    def test_half_maps_are_found_by_their_names(self, tmp_path, given, half1, half2):
        ext = given[given.rindex(".") :]
        for name in (half1, half2):
            (tmp_path / f"{name}{ext}").touch()
        found = denoise.find_half_maps(tmp_path / given)
        assert found == (tmp_path / f"{half1}{ext}", tmp_path / f"{half2}{ext}")

    def test_a_missing_half_or_another_name_gives_none(self, tmp_path):
        (tmp_path / "run_half1_class001_unfil.mrc").touch()
        assert denoise.find_half_maps(tmp_path / "run_class001.mrc") is None
        assert denoise.find_half_maps(tmp_path / "run_half1_class001_unfil.mrc") is None
        assert denoise.find_half_maps(tmp_path / "postprocess.mrc") is None
